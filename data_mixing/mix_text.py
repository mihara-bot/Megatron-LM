"""Build a mixed text dataset (parquet) from per-source document streams: estimate | plan | build | verify.

Runs on the data cluster (orion); the output directory is then copied to the training cluster and
tokenized there with the unchanged ladder/data scripts (see README.md and tokenize_polaris.sh).

    python data_mixing/mix_text.py estimate --spec data_mixing/configs/mixes/climbmix_m1_200b.json
    python data_mixing/mix_text.py plan   --spec SPEC --src /mnt/oss-zhuangxinlin/ClimbMix --out OUT \
        [--tokenizer llama3=/path/Llama3.1-8B --tokenizer qwen3=/path/Qwen3-0.6B-Base] [--workers 64]
    python data_mixing/mix_text.py build  --out OUT [--workers 64]     # resumable: re-run the same command
    python data_mixing/mix_text.py verify --out OUT [--workers 64]

estimate  quotas and passes from the frozen reference numbers only (no data access).
plan      reads the source footers, indexes the files it needs (exact text bytes), cuts every stream
          (reference head validation, tail validation, training passes), checks the head validation
          documents and the reconstructed val_all against the reference hashes, and writes OUT/plan.json.
build     executes plan.json: whole source files are copied, the files at the cuts are sliced, and
          the validation files are written. Every file is written to a .tmp name and renamed when
          complete, so an interrupted build is resumed by running it again.
verify    checks every output against the plan (sizes, row counts, validation hashes, no stray files),
          then writes files.sha256 (sha256sum format), manifest.json and, last, _DONE.

Output (OUT):
  plan.json                               the plan (operations, cuts, hashes, estimates)
  train/cluster_XX/e<p>_<source file>      pass p of cluster XX (e0 first; e1, e2 only for repeated clusters)
  val/all/val.parquet                     the reference (v1) val_all: same documents, same order
  val/<group>/val.parquet                 the tail validation documents of the group's clusters, shuffled
  files.sha256  manifest.json  _DONE      written by verify; copy OUT only once _DONE exists
  .cache/                                 footer and index caches (not part of the dataset; do not copy)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mixlib  # noqa: E402
from mixlib import SourceFile, human  # noqa: E402

VAL_DIRS = ("train", "val")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _load_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


# ----------------------------------------------------------------------------- estimate


def cmd_estimate(args: argparse.Namespace) -> None:
    spec = mixlib.load_spec(args.spec)
    mixlib.print_estimate(spec, mixlib.estimate(spec))


# ----------------------------------------------------------------------------- plan


def discover_sources(src: str, regex: str) -> Dict[int, List[str]]:
    """{source id: parquet files sorted by name} from SRC/<dir matching regex>/*.parquet."""
    out: Dict[int, List[str]] = {}
    for child in sorted(Path(src).iterdir()):
        m = re.match(regex, child.name)
        if m and child.is_dir():
            files = sorted(child.glob("*.parquet"))
            if files:
                out[int(m.group(1))] = [str(p) for p in files]
    if not out:
        raise SystemExit(f"no source directories matching {regex!r} with parquet files under {src}")
    return out


def load_footers(sources: Dict[int, List[str]], text_column: str, workers: int,
                 cache_path: Path) -> Dict[int, List[SourceFile]]:
    import parquet_io as pio

    cache = _load_json(cache_path, {})
    todo = []
    for paths in sources.values():
        for p in paths:
            st = os.stat(p)
            e = cache.get(p)
            if not e or e["size"] != st.st_size or e["mtime_ns"] != st.st_mtime_ns:
                todo.append(p)
    if todo:
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=max(1, min(64, workers))) as ex:
            for p, res in zip(todo, ex.map(lambda q: pio.footer(q, text_column), todo)):
                cache[p] = res
        print(f"[plan] read {len(todo)} parquet footers in {time.monotonic() - t0:.0f}s", flush=True)
        mixlib.write_json_atomic(cache_path, cache)
    return {sid: [SourceFile(name=os.path.basename(p), path=p, size=cache[p]["size"], rows=cache[p]["rows"],
                             rg_rows=cache[p]["rg_rows"], footer_bytes=cache[p]["footer_bytes"]) for p in paths]
            for sid, paths in sorted(sources.items())}


def check_source_against_reference(files: Dict[int, List[SourceFile]], ref: Dict[str, Any], allow: bool) -> None:
    rc = ref["clusters"]
    if sorted(files) != sorted(int(c) for c in rc):
        raise SystemExit(f"source ids {sorted(files)} differ from the reference clusters {sorted(int(c) for c in rc)}")
    bad = []
    for sid, fl in files.items():
        rows, fb = sum(f.rows for f in fl), sum(f.footer_bytes for f in fl)
        r = rc[str(sid)]
        if rows != r["source_docs"] or fb != r["source_bytes"]:
            bad.append(f"  source {sid}: rows {rows} vs {r['source_docs']}, footer bytes {fb} vs {r['source_bytes']}")
    if bad:
        msg = "the source differs from the reference download:\n" + "\n".join(bad)
        if not allow:
            raise SystemExit(msg + "\n(--allow_source_mismatch to continue; the validation hashes are still checked)")
        print("[plan] warning: " + msg, flush=True)


def first_texts(files: List[SourceFile], reader: Any, n: int) -> List[str]:
    """The first n non-empty documents of a stream (for bytes-per-token calibration)."""
    out: List[str] = []
    for f in files:
        lo = 0
        while lo < f.rows and len(out) < n:
            hi = min(f.rows, lo + 8192)
            out.extend(mixlib.nonempty(reader.texts(f, lo, hi)))
            lo = hi
        if len(out) >= n:
            break
    return out[:n]


def cmd_plan(args: argparse.Namespace) -> None:
    import parquet_io as pio

    t_start = time.monotonic()
    spec = mixlib.load_spec(args.spec)
    ref = spec["_reference"]
    col = spec.get("text_column", "text")
    out = Path(args.out)
    cache_dir = out / ".cache"
    n_tok = float(spec["train_tokens"])
    tail_tok = float(spec.get("val_tail_tokens_per_cluster", 0))
    weights = mixlib.weights_from_spec(spec)
    groups_of = mixlib.group_of(spec)

    files = load_footers(discover_sources(args.src, spec["source_dir_regex"]), col, args.workers,
                         cache_dir / "footers.json")
    check_source_against_reference(files, ref, args.allow_source_mismatch)
    rc = ref["clusters"]
    bpt = {sid: float(rc[str(sid)]["bytes_per_token"]) for sid in files}
    quota = {sid: weights[sid] * n_tok * bpt[sid] for sid in files}
    tail_bytes = {sid: tail_tok * bpt[sid] for sid in files}

    index_path = cache_dir / "index.json"
    reader = pio.ParquetReader(col, _load_json(index_path, {}))
    need = [files[sid][i] for sid in files for i in mixlib.likely_files(files[sid], quota[sid], tail_bytes[sid])]
    print(f"[plan] {sum(len(v) for v in files.values())} files in {len(files)} sources; indexing "
          f"{sum(1 for f in need if reader._cached(f) is None)} of the {len(need)} likely needed "
          f"({human(sum(f.size for f in need))}B on disk)", flush=True)
    last_save = [time.monotonic()]

    def save_index(force: bool = False) -> None:
        if force or time.monotonic() - last_save[0] > 60:
            mixlib.write_json_atomic(index_path, reader.cache)
            last_save[0] = time.monotonic()

    reader.prefetch(need, args.workers, on_done=save_index)
    save_index(force=True)

    sources: Dict[str, Any] = {}
    ops: List[Dict[str, Any]] = []
    head_texts: Dict[int, List[str]] = {}
    tail_texts: Dict[int, List[str]] = {}
    must = spec.get("contain_reference_train", True)
    for sid, fl in files.items():
        t0 = time.monotonic()
        r = rc[str(sid)]
        res = mixlib.plan_source(sid, fl, reader, head_docs=int(r["val_docs"]), tail_bytes=tail_bytes[sid],
                                 quota_bytes=quota[sid], max_epochs=float(spec.get("max_epochs", 1)),
                                 must_contain_docs=int(r.get("train_docs_50B", 0)) if must else 0)
        head_texts[sid] = mixlib.cut_texts(fl, reader, res["head"])
        head_sha = mixlib.texts_sha256(head_texts[sid])
        if res["head"].bytes != r["val_bytes"] or head_sha != r["val_sha256"]:
            raise SystemExit(f"source {sid}: the first {r['val_docs']} documents ({res['head'].bytes} bytes, "
                             f"sha256 {head_sha[:16]}) are not the reference validation set "
                             f"({r['val_bytes']} bytes, sha256 {r['val_sha256'][:16]})")
        tail_texts[sid] = mixlib.cut_texts(fl, reader, res["tail"])
        src_ops = res["ops"]
        if args.rechunk:
            for o in src_ops:
                o["kind"] = "slice"
        ops.extend(src_ops)
        footer_total = sum(f.footer_bytes for f in fl)
        region_est = footer_total - res["head"].bytes - res["tail"].bytes
        sources[str(sid)] = {
            "dir": mixlib.source_dirname(sid), "name": r.get("name", ""), "group": groups_of[sid],
            "files": len(fl), "rows": sum(f.rows for f in fl), "footer_bytes": footer_total,
            "weight": weights[sid], "bytes_per_token": bpt[sid], "quota_bytes": quota[sid],
            "head_val": {**mixlib.cut_json(fl, res["head"]), "sha256": head_sha},
            "tail_val": {**mixlib.cut_json(fl, res["tail"]), "sha256": mixlib.texts_sha256(tail_texts[sid])},
            "passes": res["passes"], "region_docs": res["region_docs"], "region_bytes": res["region_bytes"],
            "epochs": (res["train_bytes"] / res["region_bytes"]) if res["region_bytes"]
            else res["train_bytes"] / max(region_est, 1),
            "epochs_exact": res["region_bytes"] is not None,
            "train_docs": res["train_docs"], "train_bytes": res["train_bytes"],
            "train_tokens_est": res["train_bytes"] / bpt[sid],
            "repeated_bytes": max(0, res["train_bytes"] - (res["region_bytes"] or res["train_bytes"])),
            "pass0_docs": res["pass0_docs"], "reference_train_docs_50B": r.get("train_docs_50B"),
            "copy_ops": sum(o["kind"] == "copy" for o in src_ops), "slice_ops": sum(o["kind"] == "slice" for o in src_ops),
        }
        save_index()
        print(f"[plan] source {sid:2d}: {res['passes']} pass(es), {human(res['train_bytes'] / bpt[sid])} tokens, "
              f"{len(src_ops)} files ({sources[str(sid)]['slice_ops']} sliced), head {res['head'].docs} / tail "
              f"{res['tail'].docs} docs ({time.monotonic() - t0:.0f}s)", flush=True)
    save_index(force=True)
    total_tok = sum(s["train_tokens_est"] for s in sources.values())
    for s in sources.values():
        s["share_est"] = s["train_tokens_est"] / total_tok

    # val/all must be the reference val_all: same documents in the same order
    order_ids = sorted(files)
    all_texts = [t for sid in order_ids for t in head_texts[sid]]
    perm = pio.val_all_order(len(all_texts), int(ref["build"]["val_all_seed"]))
    all_sha = mixlib.texts_sha256([all_texts[i] for i in perm])
    if len(all_texts) != ref["val_all"]["docs"] or all_sha != ref["val_all"]["sha256"]:
        raise SystemExit(f"rebuilt val_all ({len(all_texts)} docs, sha256 {all_sha[:16]}) differs from the reference "
                         f"({ref['val_all']['docs']} docs, sha256 {ref['val_all']['sha256'][:16]})")
    val_groups = []
    seed = spec.get("group_val_seed", 0)
    for g in spec["_groups"]:
        items = [(t, c) for c in g["clusters"] for t in tail_texts[c]]
        items = mixlib.shuffled(items, f"{seed}:{g['slug']}")
        texts = [t for t, _ in items]
        nbytes = {c: sum(len(t.encode("utf-8")) for t in tail_texts[c]) for c in g["clusters"]}
        val_groups.append({
            "slug": g["slug"], "name": g["name"], "clusters": g["clusters"], "path": f"val/{g['slug']}/val.parquet",
            "shuffle_seed": f"{seed}:{g['slug']}", "docs": len(texts), "bytes": sum(nbytes.values()),
            "tokens_est": sum(nbytes[c] / bpt[c] for c in g["clusters"]), "sha256": mixlib.texts_sha256(texts),
        })
    val = {
        "all": {"path": "val/all/val.parquet", "docs": len(all_texts), "sha256": all_sha,
                "bytes": sum(len(t.encode("utf-8")) for t in all_texts),
                "tokens_est": sum(sources[str(s)]["head_val"]["bytes"] / bpt[s] for s in order_ids),
                "seed": int(ref["build"]["val_all_seed"]), "order": "clusters ascending, numpy default_rng(seed).permutation"},
        "groups": val_groups,
    }

    estimates: Dict[str, Any] = {}
    for spec_tok in args.tokenizer or []:
        name, _, path = spec_tok.partition("=")
        if not name or not path:
            raise SystemExit(f"--tokenizer expects NAME=PATH, got {spec_tok!r}")
        tok = pio.load_tokenizer(path)
        rates = {sid: pio.bytes_per_token(first_texts(fl, reader, args.calib_docs), tok) for sid, fl in files.items()}
        est = {
            "tokenizer": path, "bytes_per_token": {str(s): v for s, v in rates.items()},
            "train_tokens": sum(sources[str(s)]["train_bytes"] / rates[s] for s in rates),
            "val_all_tokens": sum(sources[str(s)]["head_val"]["bytes"] / rates[s] for s in rates),
            "val_group_tokens": {g["slug"]: sum(sum(len(t.encode("utf-8")) for t in tail_texts[c]) / rates[c]
                                                for c in g["clusters"]) for g in spec["_groups"]},
            "max_rel_diff_vs_reference_bpt": max(abs(rates[s] / bpt[s] - 1) for s in rates),
        }
        estimates[name] = est
        print(f"[plan] {name}: ~{human(est['train_tokens'])} training tokens, val/all ~{human(est['val_all_tokens'])}, "
              f"bytes/token differs from the reference by up to {100 * est['max_rel_diff_vs_reference_bpt']:.2f}%",
              flush=True)

    totals = {
        "train_docs": sum(s["train_docs"] for s in sources.values()),
        "train_bytes": sum(s["train_bytes"] for s in sources.values()),
        "train_tokens_est": total_tok,
        "repeated_tokens_est": sum(s["repeated_bytes"] / s["bytes_per_token"] for s in sources.values()),
        "copy_ops": sum(o["kind"] == "copy" for o in ops), "slice_ops": sum(o["kind"] == "slice" for o in ops),
        "copy_bytes_on_disk": sum(o["src_size"] for o in ops if o["kind"] == "copy"),
    }
    content = {
        "config": spec["_hashes"], "rechunk": args.rechunk,
        "files": {sid: [(f.name, f.size) for f in fl] for sid, fl in files.items()},
        "ops": [(o["dst"], o["src_name"], o["kind"], o["lo"], o["hi"]) for o in ops],
        "val": [val["all"]["sha256"]] + [g["sha256"] for g in val_groups],
    }
    fp = mixlib.fingerprint(content)
    plan = {
        "version": mixlib.PLAN_VERSION, "fingerprint": fp, "created": _now(),
        "spec": {k: v for k, v in spec.items() if not k.startswith("_")},
        "config_paths": spec["_paths"], "config_hashes": spec["_hashes"],
        "source_root": str(Path(args.src).resolve()), "text_column": col, "rechunk": args.rechunk,
        "row_group_rows": int(spec.get("row_group_rows", 8192)), "train_tokens": n_tok,
        "requirements": spec.get("requirements", {}), "sources": sources, "ops": ops, "val": val,
        "totals": totals, "estimates": estimates,
    }
    plan_path = out / "plan.json"
    old = _load_json(plan_path, None)
    if old and old["fingerprint"] != fp and any((out / d).exists() for d in VAL_DIRS) and not args.force:
        raise SystemExit(f"{out} holds a build of another plan ({old['fingerprint']}, new {fp}); use a new --out, or "
                         "--force and then `build --clean` to remove the files the new plan does not have")
    mixlib.write_json_atomic(plan_path, plan)
    print_plan(plan)
    print(f"[plan] wrote {plan_path} (fingerprint {fp}, {time.monotonic() - t_start:.0f}s)", flush=True)


def print_plan(plan: Dict[str, Any]) -> None:
    print(f"{'src':>4} {'group':15} {'weight%':>8} {'share%':>8} {'passes':>7} {'tokens':>8} {'files':>6} {'sliced':>6}"
          f" {'v1 docs in e0':>14}")
    for sid, s in plan["sources"].items():
        v1 = s["reference_train_docs_50B"]
        print(f"{int(sid):4d} {s['group']:15} {100 * s['weight']:8.3f} {100 * s['share_est']:8.3f} "
              f"{s['epochs']:6.3f}{'' if s['epochs_exact'] else '~'} {human(s['train_tokens_est']):>8} "
              f"{s['copy_ops'] + s['slice_ops']:6d} {s['slice_ops']:6d} "
              f"{'yes' if v1 is None or s['pass0_docs'] >= v1 else 'NO':>14}")
    t = plan["totals"]
    print(f"[plan] training: {human(t['train_tokens_est'])} tokens (reference tokenizer estimate), {t['train_docs']:,} "
          f"docs, {human(t['repeated_tokens_est'])} repeated tokens; {t['copy_ops']} copies "
          f"({human(t['copy_bytes_on_disk'])}B on disk) + {t['slice_ops']} slices")
    v = plan["val"]
    print(f"[plan] val/all: {v['all']['docs']:,} docs, ~{human(v['all']['tokens_est'])} tokens (= reference val_all)")
    for g in v["groups"]:
        print(f"[plan] val/{g['slug']}: {g['docs']:,} docs, ~{human(g['tokens_est'])} tokens, clusters {g['clusters']}")
    for name, need in plan.get("requirements", {}).items():
        print(f"[plan] requirement {name}: {human(need)} tokens -> "
              f"{'ok' if t['train_tokens_est'] >= need else 'NOT MET'} (reference tokenizer estimate)")


# ----------------------------------------------------------------------------- build


def invalidate(out: Path) -> None:
    """Remove the verify outputs (_DONE first): the directory is not ready to copy."""
    for name in ("_DONE", "files.sha256", "manifest.json"):
        if (out / name).exists():
            os.remove(out / name)


def load_plan(out: Path) -> Dict[str, Any]:
    path = out / "plan.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run `plan` first")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_files(plan: Dict[str, Any]) -> List[str]:
    return [o["dst"] for o in plan["ops"]] + [plan["val"]["all"]["path"]] + [g["path"] for g in plan["val"]["groups"]]


def existing_files(out: Path) -> List[str]:
    found = []
    for d in VAL_DIRS:
        for root, _, names in os.walk(out / d):
            for n in names:
                found.append(os.path.relpath(os.path.join(root, n), out))
    return sorted(found)


def val_documents(plan: Dict[str, Any], which: str) -> Dict[int, List[str]]:
    """Head ('head_val') or tail ('tail_val') validation documents of every source, read from the source."""
    import parquet_io as pio

    out = {}
    for sid, s in plan["sources"].items():
        texts: List[str] = []
        for r in s[which]["ranges"]:
            texts.extend(mixlib.nonempty(pio.read_texts(r["src"], plan["text_column"], r["lo"], r["hi"])))
        if len(texts) != s[which]["docs"] or mixlib.texts_sha256(texts) != s[which]["sha256"]:
            raise SystemExit(f"source {sid}: the {which} documents read now differ from the plan (source changed?)")
        out[int(sid)] = texts
    return out


def write_val_files(plan: Dict[str, Any], out: Path) -> None:
    import parquet_io as pio

    rg = plan["row_group_rows"]
    v_all = plan["val"]["all"]
    path = out / v_all["path"]
    if not (path.is_file() and mixlib.texts_sha256(pio.read_val_texts(str(path))) == v_all["sha256"]):
        head = val_documents(plan, "head_val")
        texts = [t for sid in sorted(head) for t in head[sid]]
        ids = [sid for sid in sorted(head) for _ in head[sid]]
        perm = pio.val_all_order(len(texts), v_all["seed"])
        texts, ids = [texts[i] for i in perm], [ids[i] for i in perm]
        assert mixlib.texts_sha256(texts) == v_all["sha256"]
        pio.write_val(str(path), texts, ids, rg)
        print(f"[build] wrote {v_all['path']} ({len(texts):,} docs)", flush=True)
    todo = [g for g in plan["val"]["groups"]
            if not ((out / g["path"]).is_file()
                    and mixlib.texts_sha256(pio.read_val_texts(str(out / g["path"]))) == g["sha256"])]
    if not todo:
        return
    tail = val_documents(plan, "tail_val")
    for g in todo:
        items = mixlib.shuffled([(t, c) for c in g["clusters"] for t in tail[c]], g["shuffle_seed"])
        texts, ids = [t for t, _ in items], [c for _, c in items]
        assert mixlib.texts_sha256(texts) == g["sha256"]
        pio.write_val(str(out / g["path"]), texts, ids, rg)
        print(f"[build] wrote {g['path']} ({len(texts):,} docs)", flush=True)


def cmd_build(args: argparse.Namespace) -> None:
    import parquet_io as pio

    out = Path(args.out)
    plan = load_plan(out)
    invalidate(out)  # whatever happens next, the old verify result no longer holds
    expected = set(expected_files(plan))
    found = existing_files(out)
    tmp = [p for p in found if p.endswith(".tmp")]
    for p in tmp:
        os.remove(out / p)
    strays = [p for p in found if not p.endswith(".tmp") and p not in expected]
    if strays:
        if not args.clean:
            raise SystemExit(f"{len(strays)} files under {out} are not in the plan (e.g. {strays[:3]}); "
                             "pass --clean to delete them")
        for p in strays:
            os.remove(out / p)
        print(f"[build] deleted {len(strays)} files that are not in the plan", flush=True)

    t0 = time.monotonic()
    todo = [o for o in plan["ops"] if not pio.output_ok(o, str(out / o["dst"]))]
    print(f"[build] {len(plan['ops'])} training files in the plan, {len(plan['ops']) - len(todo)} already done, "
          f"{len(todo)} to write ({sum(o['kind'] == 'slice' for o in todo)} slices), {args.workers} workers", flush=True)
    failed = []
    if todo:
        progress = pio.Progress(sum(o["src_size"] for o in todo), "build", unit="B")
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = {ex.submit(pio.execute_op, o, str(out), plan["row_group_rows"]): o for o in todo}
            for fut in as_completed(futures):
                o = futures[fut]
                try:
                    fut.result()
                except Exception as e:  # keep going; a re-run retries the failed files
                    failed.append((o["dst"], repr(e)))
                    print(f"[build] FAILED {o['dst']}: {e!r}", flush=True)
                progress.update(o["src_size"])
        progress.close()
    write_val_files(plan, out)
    if failed:
        raise SystemExit(f"[build] {len(failed)} files failed (e.g. {failed[0]}); run build again to retry them")
    print(f"[build] done in {time.monotonic() - t0:.0f}s; next: verify", flush=True)


# ----------------------------------------------------------------------------- verify


def cmd_verify(args: argparse.Namespace) -> None:
    import parquet_io as pio

    out = Path(args.out)
    plan = load_plan(out)
    problems = []
    with ThreadPoolExecutor(max_workers=max(1, min(64, args.workers))) as ex:
        oks = list(ex.map(lambda o: pio.output_ok(o, str(out / o["dst"])), plan["ops"]))
    problems += [f"missing or incomplete: {o['dst']}" for o, ok in zip(plan["ops"], oks) if not ok]
    expected = expected_files(plan)
    exp_set = set(expected)
    problems += [f"not in the plan: {p}" for p in existing_files(out) if p not in exp_set]
    for v in [plan["val"]["all"]] + plan["val"]["groups"]:
        path = out / v["path"]
        if not path.is_file():
            problems.append(f"missing: {v['path']}")
            continue
        texts = pio.read_val_texts(str(path))
        if len(texts) != v["docs"] or mixlib.texts_sha256(texts) != v["sha256"]:
            problems.append(f"wrong documents: {v['path']}")
    if problems:
        invalidate(out)
        for p in problems[:20]:
            print(f"[verify] {p}")
        raise SystemExit(f"[verify] {len(problems)} problems (first: {problems[0]}); run build (with --clean for "
                         "stray files) and verify again")

    names = sorted(expected) + ["plan.json"]
    t0 = time.monotonic()
    sizes = {n: os.path.getsize(out / n) for n in names}
    progress = pio.Progress(sum(sizes.values()), "sha256", unit="B")
    hashes = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(pio.sha256_file, str(out / n)): n for n in names}
        for fut in as_completed(futures):
            n = futures[fut]
            hashes[n] = fut.result()
            progress.update(sizes[n])
    progress.close()
    tmp = out / "files.sha256.tmp"
    tmp.write_text("".join(f"{hashes[n]}  {n}\n" for n in names), encoding="utf-8")
    os.replace(tmp, out / "files.sha256")
    total = sum(sizes.values())
    manifest = {
        "fingerprint": plan["fingerprint"], "created": _now(), "files": len(names), "bytes": total,
        "train_tokens": plan["train_tokens"], "requirements": plan["requirements"], "totals": plan["totals"],
        "sources": {sid: {k: s[k] for k in ("dir", "name", "group", "weight", "share_est", "passes", "epochs",
                                            "train_docs", "train_bytes", "train_tokens_est")}
                    for sid, s in plan["sources"].items()},
        "val": {"all": {k: v for k, v in plan["val"]["all"].items()},
                "groups": [{k: g[k] for k in ("slug", "name", "clusters", "path", "docs", "bytes", "tokens_est", "sha256")}
                           for g in plan["val"]["groups"]]},
        "estimates": {k: {kk: vv for kk, vv in v.items() if kk != "bytes_per_token"} for k, v in plan["estimates"].items()},
    }
    mixlib.write_json_atomic(out / "manifest.json", manifest)
    mixlib.write_json_atomic(out / "_DONE", {"fingerprint": plan["fingerprint"], "files": len(names), "bytes": total,
                                             "created": _now()})
    print(f"[verify] ok: {len(names)} files, {human(total)}B, sha256 in {time.monotonic() - t0:.0f}s; wrote "
          f"files.sha256, manifest.json, _DONE. Copy {out} (without .cache/) to the training cluster.", flush=True)


# ----------------------------------------------------------------------------- CLI


def main(argv: List[str] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    workers = min(64, os.cpu_count() or 1)

    s = sub.add_parser("estimate", help="quotas and passes from the reference numbers (no data access)")
    s.add_argument("--spec", required=True)
    s.set_defaults(func=cmd_estimate)

    s = sub.add_parser("plan", help="cut the sources and write OUT/plan.json")
    s.add_argument("--spec", required=True)
    s.add_argument("--src", required=True, help="source root (ClimbMix: cluster_id=<k>/*.parquet)")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=workers, help="processes for indexing (default: min(64, cores))")
    s.add_argument("--tokenizer", action="append", metavar="NAME=PATH",
                   help="also estimate token counts with this tokenizer (repeatable)")
    s.add_argument("--calib_docs", type=int, default=2000, help="documents per source for --tokenizer estimates")
    s.add_argument("--rechunk", action="store_true",
                   help="rewrite every training file with row_group_rows rows per row group instead of copying")
    s.add_argument("--allow_source_mismatch", action="store_true",
                   help="continue when footer rows/bytes differ from the reference (validation hashes still checked)")
    s.add_argument("--force", action="store_true", help="replace a plan whose build exists in --out")
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("build", help="write the files of OUT/plan.json (resumable)")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=workers, help="parallel copies/slices (default: min(64, cores))")
    s.add_argument("--clean", action="store_true", help="delete files under train/ and val/ that are not in the plan")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("verify", help="check the build, write files.sha256, manifest.json, _DONE")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=workers, help="parallel sha256 (default: min(64, cores))")
    s.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
