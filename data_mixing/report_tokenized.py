"""Check a tokenized mix against its plan and report exact token counts (after tokenize + merge).

    python data_mixing/report_tokenized.py --text-root TEXT --tok-root TOK --family llama3 \
        [--compare-val-all /fast/.../megatron_data/llama3/climbmix_v1_m1/val_all/val]

TEXT is the text dataset (holds plan.json), TOK the --output-root of ladder/data/tokenize_dataset.py
(holds its manifest_*.json, train/, val/ and, after merge_indexed.py, train_merged.*). Checks:
  - every planned file was tokenized, with exactly the planned documents and UTF-8 bytes
    (the plan counts non-empty texts, as tokenize_dataset.py does);
  - train_merged holds all training documents and tokens;
  - with --compare-val-all: val/all is byte-identical to the reference tokenized val_all;
and reports per-cluster tokens, shares against the mix weights, validation sizes and the token
requirements of the spec. Writes TOK/report.json ("ok": true/false); exit status 1 on any problem.
Stdlib only.
"""

from __future__ import annotations

import argparse
import filecmp
import glob
import json
import os
import sys
from collections import defaultdict


def load_tokenize_entries(tok_root: str):
    manifests = sorted(glob.glob(os.path.join(tok_root, "manifest_*.json")))
    if not manifests:
        raise SystemExit(f"no manifest_*.json in {tok_root}: run ladder/data/tokenize_dataset.py first")
    entries, tok_info = {}, None
    for path in manifests:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        if tok_info is not None and (m["tokenizer"]["vocab_size"], m["tokenizer"]["eos"]) != (
                tok_info["vocab_size"], tok_info["eos"]):
            raise SystemExit(f"{path}: made with another tokenizer than the other manifests")
        tok_info = m["tokenizer"]
        for e in m["files"]:
            entries[e["prefix"]] = e  # one entry per piece; re-runs list the same pieces again
    return list(entries.values()), tok_info, manifests


def report(text_root: str, tok_root: str, family: str, compare_val_all: str = "", require_merged: bool = True):
    with open(os.path.join(text_root, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    entries, tok_info, manifests = load_tokenize_entries(tok_root)
    problems = []
    root_real = os.path.realpath(text_root)

    # tokenized pieces, summed per input file (a large parquet file is tokenized in several pieces)
    per_file = defaultdict(lambda: {"docs": 0, "bytes": 0, "tokens": 0, "pieces": 0})
    for e in entries:
        rel = os.path.relpath(os.path.realpath(e["input"]), root_real)
        if rel.startswith(".."):
            problems.append(f"tokenized a file outside the text root: {e['input']}")
            continue
        d = per_file[rel]
        d["docs"] += e["docs"]
        d["bytes"] += e["bytes"]
        d["tokens"] += e["tokens"]
        d["pieces"] += 1

    sources = {}
    planned = {}
    for o in plan["ops"]:
        planned[o["dst"]] = o
    by_source = defaultdict(lambda: {"docs": 0, "bytes": 0, "tokens": 0, "files": 0})
    for dst, o in planned.items():
        got = per_file.get(dst)
        if got is None:
            problems.append(f"not tokenized: {dst}")
            continue
        if got["docs"] != o["docs"] or got["bytes"] != o["bytes"]:
            problems.append(f"{dst}: tokenized {got['docs']} docs / {got['bytes']} bytes, planned {o['docs']} / {o['bytes']}")
        s = by_source[str(o["source"])]
        for k in ("docs", "bytes", "tokens"):
            s[k] += got[k]
        s["files"] += 1
    train_tokens = sum(s["tokens"] for s in by_source.values())
    for sid, ps in plan["sources"].items():
        s = by_source[sid]
        sources[sid] = {
            "dir": ps["dir"], "group": ps["group"], "docs": s["docs"], "bytes": s["bytes"], "tokens": s["tokens"],
            "share": s["tokens"] / train_tokens if train_tokens else 0.0, "weight": ps["weight"],
            "bytes_per_token": s["bytes"] / s["tokens"] if s["tokens"] else None,
            "reference_bytes_per_token": ps["bytes_per_token"], "passes": ps["passes"],
        }

    val = {}
    for v in [dict(plan["val"]["all"], slug="all")] + plan["val"]["groups"]:
        got = per_file.get(v["path"])
        if got is None:
            problems.append(f"not tokenized: {v['path']}")
            continue
        if got["docs"] != v["docs"] or got["bytes"] != v["bytes"]:
            problems.append(f"{v['path']}: tokenized {got['docs']} docs / {got['bytes']} bytes, planned "
                            f"{v['docs']} / {v['bytes']}")
        if got["pieces"] != 1:
            problems.append(f"{v['path']} was tokenized in {got['pieces']} pieces; a validation set must be one prefix")
        val[v["slug"]] = {"prefix": os.path.join(tok_root, os.path.splitext(v["path"])[0]), "docs": got["docs"],
                          "tokens": got["tokens"]}
    extra = sorted(set(per_file) - set(planned) - {v["path"] for v in [plan["val"]["all"]] + plan["val"]["groups"]})
    problems += [f"tokenized but not in the plan: {x}" for x in extra[:10]]

    merged_path = os.path.join(tok_root, "train_merged.json")
    merged = None
    if os.path.isfile(merged_path):
        with open(merged_path) as f:
            merged = json.load(f)
        if merged["docs"] != sum(s["docs"] for s in by_source.values()) or merged["tokens"] != train_tokens:
            problems.append(f"train_merged has {merged['docs']} docs / {merged['tokens']} tokens, the pieces "
                            f"{sum(s['docs'] for s in by_source.values())} / {train_tokens}")
    elif require_merged:
        problems.append("train_merged.json missing: run ladder/data/merge_indexed.py first")

    val_all_identical = None
    if compare_val_all:
        mine = os.path.join(tok_root, "val", "all", "val")
        val_all_identical = all(
            os.path.isfile(compare_val_all + ext) and filecmp.cmp(compare_val_all + ext, mine + ext, shallow=False)
            for ext in (".bin", ".idx"))
        if not val_all_identical:
            problems.append(f"val/all differs from the reference {compare_val_all}.{{bin,idx}}")

    requirements = {name: {"tokens": need, "ok": train_tokens >= need} for name, need in plan["requirements"].items()}
    for name, r in requirements.items():
        if not r["ok"]:
            problems.append(f"requirement {name}: {need_str(r['tokens'])} tokens, the mix has {need_str(train_tokens)}")
    names = ["all"] + [g["slug"] for g in plan["val"]["groups"]]
    rep = {
        "ok": not problems, "family": family, "tokenizer": tok_info, "manifests": manifests,
        "plan_fingerprint": plan["fingerprint"],
        "train": {"docs": sum(s["docs"] for s in by_source.values()), "bytes": sum(s["bytes"] for s in by_source.values()),
                  "tokens": train_tokens, "merged": merged},
        "sources": sources, "val": val, "val_all_identical_to_reference": val_all_identical,
        "requirements": requirements, "problems": problems,
        "env": {"DATA_ROOT": tok_root, "TRAIN_DATA": os.path.join(tok_root, "train_merged"),
                "VALID_DATA": " ".join(val[n]["prefix"] for n in names if n in val), "VALID_NAMES": " ".join(names)},
    }
    return rep


def need_str(x: float) -> str:
    return f"{x / 1e9:.2f}B"


def print_report(rep) -> None:
    print(f"[report] {rep['family']}: tokenizer {rep['tokenizer']['name_or_path']} (vocab {rep['tokenizer']['vocab_size']})")
    print(f"{'src':>4} {'group':15} {'tokens':>10} {'share%':>8} {'weight%':>8} {'B/token':>8} {'ref B/tok':>9} {'passes':>6}")
    for sid, s in rep["sources"].items():
        bpt = f"{s['bytes_per_token']:.3f}" if s["bytes_per_token"] else "-"
        print(f"{int(sid):4d} {s['group']:15} {need_str(s['tokens']):>10} {100 * s['share']:8.3f} {100 * s['weight']:8.3f} "
              f"{bpt:>8} {s['reference_bytes_per_token']:9.3f} {s['passes']:6d}")
    t = rep["train"]
    print(f"[report] training: {t['docs']:,} docs, {need_str(t['tokens'])} tokens")
    for name, r in rep["requirements"].items():
        print(f"[report] requirement {name}: {need_str(r['tokens'])} -> {'ok' if r['ok'] else 'NOT MET'}")
    for name, v in rep["val"].items():
        print(f"[report] val/{name}: {v['docs']:,} docs, {v['tokens'] / 1e6:.2f}M tokens")
    if rep["val_all_identical_to_reference"] is not None:
        print(f"[report] val/all identical to the reference tokenized val_all: {rep['val_all_identical_to_reference']}")
    for p in rep["problems"][:30]:
        print(f"[report] PROBLEM {p}")
    env = rep["env"]
    print("[report] training data for ladder/train.sh (launch/slurm.sbatch, launch/cloud.sh):")
    print(f"  export DATA_ROOT={env['DATA_ROOT']}")
    print(f"  export VALID_DATA=\"{env['VALID_DATA']}\"")
    print(f"  export VALID_NAMES=\"{env['VALID_NAMES']}\"")
    print(f"[report] {'ok' if rep['ok'] else 'FAILED'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text-root", required=True)
    p.add_argument("--tok-root", required=True)
    p.add_argument("--family", default="")
    p.add_argument("--compare-val-all", default="", help="reference tokenized val_all prefix (same tokenizer)")
    p.add_argument("--no-merged", action="store_true", help="do not require train_merged (before merging)")
    p.add_argument("--out", default="", help="report path (default TOK/report.json)")
    args = p.parse_args(argv)
    rep = report(args.text_root, args.tok_root, args.family, args.compare_val_all, not args.no_merged)
    out = args.out or os.path.join(args.tok_root, "report.json")
    with open(out + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    os.replace(out + ".tmp", out)
    print_report(rep)
    print(f"[report] wrote {out}")
    if not rep["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
