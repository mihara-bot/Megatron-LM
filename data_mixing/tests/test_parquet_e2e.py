"""End-to-end test of mix_text.py (plan -> build -> verify) on synthetic ClimbMix-like parquet.

Needs pyarrow and numpy (skipped without them). Run it on the data cluster before the real plan:

    python data_mixing/tests/test_parquet_e2e.py
    python -m pytest data_mixing/tests/test_parquet_e2e.py -p no:cacheprovider -o addopts=""

The reference (v1 validation sets, weights, v1 training documents) is computed here independently, with
the v1 builder's rules, and every output is checked document by document against a brute-force reading.

With DM_TOKENIZER=<HF tokenizer dir> (and the Megatron environment), the built dataset also goes
through the Polaris side: ladder/data/tokenize_dataset.py (small chunks, so files are split into
several pieces), merge_indexed.py, report_tokenized.py and verify_files.py.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
TOKENIZER = os.environ.get("DM_TOKENIZER", "")

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    try:
        import pytest

        pytest.skip("pyarrow/numpy not installed", allow_module_level=True)
    except ImportError:
        pass

import mixlib  # noqa: E402

BPT = 4.0             # bytes per token of the synthetic "reference tokenizer"
VAL_TOKENS = 40       # v1 head validation per cluster
V1_TIER_TOKENS = 2000  # the synthetic "v1 50B tier"
TAIL_TOKENS = 50      # tail validation per cluster
SEED = 20261119
LAYOUT = {1: (4, 300), 2: (3, 250), 3: (2, 60)}  # cluster: (files, rows per file); cluster 3 gets repeated


def make_source(root: Path, rng: random.Random) -> None:
    for cid, (n_files, rows) in LAYOUT.items():
        d = root / f"cluster_id={cid}"
        d.mkdir(parents=True)
        for i in range(n_files):
            texts = []
            for _ in range(rows):
                u = rng.random()
                if u < 0.03:
                    texts.append(None)
                elif u < 0.06:
                    texts.append("")
                else:
                    texts.append(f"c{cid} " + "".join(rng.choice("abcdefgh é漢") for _ in range(rng.randint(5, 200))))
            pq.write_table(pa.table({"text": pa.array(texts, pa.string())}), d / f"part_{i:05d}.parquet",
                           row_group_size=37)


def stream(root: Path, cid: int):
    out = []
    for f in sorted((root / f"cluster_id={cid}").glob("*.parquet")):
        out.extend(t for t in pq.read_table(f, columns=["text"]).column(0).to_pylist() if t)
    return out


def nbytes(t: str) -> int:
    return len(t.encode("utf-8"))


def take_until(docs, start, target):
    """v1 take_until: documents from `start` until their bytes reach target (crossing one included)."""
    acc, i = 0, start
    while acc < target:
        acc += nbytes(docs[i])
        i += 1
    return i


def make_reference(root: Path, d: Path) -> dict:
    clusters = {}
    sizes = {}
    for cid in LAYOUT:
        rows = fb = 0
        for f in sorted((root / f"cluster_id={cid}").glob("*.parquet")):
            meta = pq.ParquetFile(f).metadata
            for g in range(meta.num_row_groups):
                grp = meta.row_group(g)
                rows += grp.num_rows
                for c in range(grp.num_columns):
                    if grp.column(c).path_in_schema == "text":
                        fb += max(0, grp.column(c).total_uncompressed_size - 4 * grp.num_rows)
        sizes[cid] = (rows, fb)
    raw = {cid: (fb / BPT) ** 0.5 for cid, (_, fb) in sizes.items()}
    weights = {cid: v / sum(raw.values()) for cid, v in raw.items()}
    val_all = []
    for cid in LAYOUT:
        docs = stream(root, cid)
        k = take_until(docs, 0, VAL_TOKENS * BPT)
        end_v1 = take_until(docs, k, weights[cid] * V1_TIER_TOKENS * BPT)
        val_all.extend(docs[:k])
        clusters[str(cid)] = {
            "name": f"cluster {cid}", "source_docs": sizes[cid][0], "source_bytes": sizes[cid][1],
            "bytes_per_token": BPT, "weight": weights[cid], "val_docs": k,
            "val_bytes": sum(nbytes(t) for t in docs[:k]), "val_sha256": mixlib.texts_sha256(docs[:k]),
            "train_docs_50B": end_v1 - k, "train_bytes_50B": sum(nbytes(t) for t in docs[k:end_v1]),
        }
    perm = np.random.default_rng(SEED).permutation(len(val_all))
    ref = {"build": {"val_all_seed": SEED}, "clusters": clusters,
           "val_all": {"docs": len(val_all), "sha256": mixlib.texts_sha256([val_all[i] for i in perm])}}
    (d / "ref.json").write_text(json.dumps(ref))
    (d / "groups.json").write_text(json.dumps({"groups": [{"slug": "big", "name": "Big", "clusters": [1, 2]},
                                                          {"slug": "small", "name": "Small", "clusters": [3]}]}))
    return ref


def make_spec(root: Path, d: Path, ref: dict) -> Path:
    # train_tokens such that cluster 3 needs 2.5 passes over its training documents
    docs3 = stream(root, 3)
    c3 = ref["clusters"]["3"]
    tail = 0
    for i in range(len(docs3) - 1, -1, -1):
        tail += nbytes(docs3[i])
        if tail >= TAIL_TOKENS * BPT:
            break
    avail3 = sum(nbytes(t) for t in docs3) - c3["val_bytes"] - tail
    n_tok = 2.5 * avail3 / (c3["weight"] * BPT)
    spec = {"name": "e2e", "reference": "ref.json", "groups": "groups.json", "source_dir_regex": r"^cluster_id=(\d+)$",
            "text_column": "text", "weights": "reference", "train_tokens": n_tok, "max_epochs": 4,
            "val_tail_tokens_per_cluster": TAIL_TOKENS, "group_val_seed": 7, "row_group_rows": 16,
            "requirements": {"e2e": 1}}
    path = d / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def expected(root: Path, ref: dict, spec: dict):
    """Brute force: per cluster, (training documents in pass order, tail documents)."""
    out = {}
    for cid in LAYOUT:
        docs = stream(root, cid)
        r = ref["clusters"][str(cid)]
        k = r["val_docs"]
        t_start, acc = len(docs), 0
        while acc < TAIL_TOKENS * BPT:
            t_start -= 1
            acc += nbytes(docs[t_start])
        region = docs[k:t_start]
        quota = r["weight"] * spec["train_tokens"] * BPT
        train, consumed = [], 0
        while consumed < quota:
            for t in region:
                train.append(t)
                consumed += nbytes(t)
                if consumed >= quota:
                    break
        out[cid] = (train, docs[t_start:])
    return out


def read_cluster(out: Path, cid: int):
    files = sorted((out / "train" / mixlib.source_dirname(cid)).glob("e*_*.parquet"),
                   key=lambda p: (int(p.name[1:p.name.index("_")]), p.name))
    texts = []
    for f in files:
        texts.extend(t for t in pq.read_table(f, columns=["text"]).column(0).to_pylist() if t)
    return texts


def run(*argv):
    import mix_text

    mix_text.main([str(a) for a in argv])


def expect_exit(*argv) -> str:
    try:
        run(*argv)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"expected {argv[0]} to fail")


def check_build(src: Path, out: Path, ref: dict, spec: dict) -> dict:
    plan = json.loads((out / "plan.json").read_text())
    exp = expected(src, ref, spec)
    for cid in LAYOUT:
        assert read_cluster(out, cid) == exp[cid][0], f"cluster {cid}: training documents"
    assert plan["sources"]["3"]["passes"] == 3 and plan["sources"]["3"]["epochs_exact"]
    assert plan["sources"]["1"]["passes"] == 1
    # val/all = the reference val_all, same order
    val_all = []
    for cid in LAYOUT:
        val_all.extend(stream(src, cid)[: ref["clusters"][str(cid)]["val_docs"]])
    perm = np.random.default_rng(SEED).permutation(len(val_all))
    got = pq.read_table(out / "val/all/val.parquet")
    assert got.column("text").to_pylist() == [val_all[i] for i in perm]
    # groups: the tail documents of their clusters
    for slug, members in (("big", [1, 2]), ("small", [3])):
        t = pq.read_table(out / f"val/{slug}/val.parquet")
        texts, ids = t.column("text").to_pylist(), t.column("cluster_id").to_pylist()
        assert sorted(texts) == sorted(x for c in members for x in exp[c][1])
        for x, c in zip(texts, ids):
            assert x.startswith(f"c{c} ")
    # copies are byte-identical to their source, slices have small row groups
    for o in plan["ops"]:
        dst = out / o["dst"]
        if o["kind"] == "copy":
            assert dst.read_bytes() == Path(o["src"]).read_bytes()
        else:
            meta = pq.ParquetFile(dst).metadata
            assert meta.num_rows == o["hi"] - o["lo"]
            assert max(meta.row_group(g).num_rows for g in range(meta.num_row_groups)) <= 16
    return plan


def check_tokenize(out: Path, tok: Path) -> None:
    """The Polaris side on the built dataset: tokenize (several pieces per file), merge, report."""
    import report_tokenized

    subprocess.run([sys.executable, str(REPO / "ladder/data/tokenize_dataset.py"), "--input-root", str(out),
                    "--input-glob", "**/*.parquet", "--output-root", str(tok), "--tokenizer", TOKENIZER,
                    "--workers", "2", "--docs-per-chunk", "50"], check=True)
    subprocess.run([sys.executable, str(REPO / "ladder/data/merge_indexed.py"), "--input-root", str(tok / "train"),
                    "--output-prefix", str(tok / "train_merged")], check=True)
    plan = json.loads((out / "plan.json").read_text())
    # the same text tokenized twice is byte-identical (stands in for the v1 val_all comparison)
    ref_val = tok.parent / "ref_val" / "val"
    ref_val.parent.mkdir(parents=True)
    for ext in (".bin", ".idx"):
        (Path(str(ref_val) + ext)).write_bytes((tok / "val" / "all" / ("val" + ext)).read_bytes())
    rep = report_tokenized.report(str(out), str(tok), "test", str(ref_val))
    assert rep["ok"], rep["problems"]
    assert rep["train"]["docs"] == plan["totals"]["train_docs"]
    assert rep["train"]["bytes"] == plan["totals"]["train_bytes"]
    assert rep["val_all_identical_to_reference"] is True
    assert set(rep["val"]) == {"all", "big", "small"}
    pieces = [e for m in rep["manifests"] for e in json.loads(Path(m).read_text())["files"]]
    assert len(pieces) > len(plan["ops"]) + 3, "expected files split into several pieces"
    subprocess.run([sys.executable, str(REPO / "data_mixing/report_tokenized.py"), "--text-root", str(out),
                    "--tok-root", str(tok), "--family", "test"], check=True)
    print(f"ok  tokenized with {TOKENIZER}: {rep['train']['tokens']} training tokens in {len(pieces)} pieces")


def test_end_to_end():
    if pa is None:
        return
    import verify_files

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, cfg, out = tmp / "src", tmp / "cfg", tmp / "out"
        cfg.mkdir()
        make_source(src, random.Random(0))
        ref = make_reference(src, cfg)
        spec_path = make_spec(src, cfg, ref)
        spec = json.loads(spec_path.read_text())

        run("estimate", "--spec", spec_path)
        run("plan", "--spec", spec_path, "--src", src, "--out", out, "--workers", 2)
        run("build", "--out", out, "--workers", 2)
        run("verify", "--out", out, "--workers", 2)
        plan = check_build(src, out, ref, spec)
        assert (out / "_DONE").is_file() and verify_files.verify(str(out), 2) == []
        sums = (out / "files.sha256").read_text()

        # the plan is reproducible (index cache reused)
        run("plan", "--spec", spec_path, "--src", src, "--out", out, "--workers", 2)
        assert json.loads((out / "plan.json").read_text())["fingerprint"] == plan["fingerprint"]

        # resume: a lost copy, a lost slice, a half-written file and a stray file
        copy = next(o["dst"] for o in plan["ops"] if o["kind"] == "copy")
        sliced = next(o["dst"] for o in plan["ops"] if o["kind"] == "slice")
        os.remove(out / copy)
        os.remove(out / sliced)
        (out / "train/cluster_01/e0_part_00001.parquet.tmp").write_bytes(b"partial")
        (out / "train/cluster_01/stray.parquet").write_bytes(b"stray")
        assert "problems" in expect_exit("verify", "--out", out, "--workers", 2)
        assert not (out / "_DONE").exists()
        assert "--clean" in expect_exit("build", "--out", out, "--workers", 2)
        assert not (out / "_DONE").exists()
        run("build", "--out", out, "--workers", 2, "--clean")
        run("verify", "--out", out, "--workers", 2)
        assert (out / "files.sha256").read_text() == sums, "rebuilt files differ"
        check_build(src, out, ref, spec)

        # a plan with all training files rewritten (--rechunk) holds the same documents
        out2 = tmp / "out_rechunk"
        run("plan", "--spec", spec_path, "--src", src, "--out", out2, "--workers", 2, "--rechunk")
        run("build", "--out", out2, "--workers", 2)
        for cid in LAYOUT:
            assert read_cluster(out2, cid) == read_cluster(out, cid)

        # a wrong reference validation set is refused
        bad = json.loads((cfg / "ref.json").read_text())
        bad["clusters"]["2"]["val_sha256"] = "0" * 64
        (cfg / "ref.json").write_text(json.dumps(bad))
        assert "not the reference validation set" in expect_exit(
            "plan", "--spec", spec_path, "--src", src, "--out", tmp / "out_bad", "--workers", 2)

        if TOKENIZER:
            check_tokenize(out, tmp / "tok")
        else:
            print("skip  tokenize check (set DM_TOKENIZER to a tokenizer directory to run it)")


if __name__ == "__main__":
    if pa is None:
        print("skip  test_end_to_end (pyarrow/numpy not installed)")
    else:
        test_end_to_end()
        print("ok  test_end_to_end")
