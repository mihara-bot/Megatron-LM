"""CPU tests for the Polaris-side tools: verify_files.py and report_tokenized.py (stdlib only).

    python -m pytest data_mixing/tests/test_polaris_tools.py -p no:cacheprovider -o addopts=""
    python3 data_mixing/tests/test_polaris_tools.py          # without pytest
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import report_tokenized  # noqa: E402
import verify_files  # noqa: E402


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def make_text_dataset(root: Path) -> None:
    files = {"train/cluster_01/e0_a.parquet": b"aaa", "train/cluster_02/e0_b.parquet": b"bbbb",
             "val/all/val.parquet": b"v", "val/g1/val.parquet": b"g"}
    for rel, data in files.items():
        _write(root / rel, data)
    _write(root / "plan.json", b"{}")
    names = sorted(files) + ["plan.json"]
    (root / "files.sha256").write_text("".join(f"{hashlib.sha256((root / n).read_bytes()).hexdigest()}  {n}\n"
                                               for n in names))
    (root / "_DONE").write_text(json.dumps({"files": len(names)}))


def test_verify_files():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        make_text_dataset(root)
        assert verify_files.verify(str(root), 2) == []
        _write(root / "train/cluster_01/e0_a.parquet", b"aaX")
        _write(root / "train/cluster_01/stray.parquet", b"s")
        os.remove(root / "val/g1/val.parquet")
        os.remove(root / "_DONE")
        problems = "\n".join(verify_files.verify(str(root), 2))
        for needle in ("sha256 mismatch: train/cluster_01/e0_a.parquet", "not listed: train/cluster_01/stray.parquet",
                       "missing: val/g1/val.parquet", "_DONE missing"):
            assert needle in problems, needle


def make_tokenized(text: Path, tok: Path, *, docs_off: int = 0, tokens_b: int = 300):
    plan = {
        "fingerprint": "f" * 16,
        "ops": [
            {"source": 1, "dst": "train/cluster_01/e0_a.parquet", "docs": 10, "bytes": 1000},
            {"source": 1, "dst": "train/cluster_01/e1_a.parquet", "docs": 10, "bytes": 1000},
            {"source": 2, "dst": "train/cluster_02/e0_b.parquet", "docs": 5, "bytes": 1200},
        ],
        "sources": {
            "1": {"dir": "cluster_01", "group": "g1", "weight": 0.6, "bytes_per_token": 4.0, "passes": 2},
            "2": {"dir": "cluster_02", "group": "g1", "weight": 0.4, "bytes_per_token": 4.0, "passes": 1},
        },
        "val": {"all": {"path": "val/all/val.parquet", "docs": 3, "bytes": 90},
                "groups": [{"slug": "g1", "path": "val/g1/val.parquet", "docs": 4, "bytes": 120}]},
        "requirements": {"r": 500},
    }
    text.mkdir(parents=True, exist_ok=True)
    (text / "plan.json").write_text(json.dumps(plan))
    entries = [
        # e0_a in two pieces (row-group chunks), the rest in one
        {"input": str(text / "train/cluster_01/e0_a.parquet"), "prefix": str(tok / "train/cluster_01/e0_a-rg00000-00001"),
         "docs": 6, "tokens": 150, "bytes": 600},
        {"input": str(text / "train/cluster_01/e0_a.parquet"), "prefix": str(tok / "train/cluster_01/e0_a-rg00002-00003"),
         "docs": 4, "tokens": 100, "bytes": 400},
        {"input": str(text / "train/cluster_01/e1_a.parquet"), "prefix": str(tok / "train/cluster_01/e1_a"),
         "docs": 10 + docs_off, "tokens": 250, "bytes": 1000},
        {"input": str(text / "train/cluster_02/e0_b.parquet"), "prefix": str(tok / "train/cluster_02/e0_b"),
         "docs": 5, "tokens": tokens_b, "bytes": 1200},
        {"input": str(text / "val/all/val.parquet"), "prefix": str(tok / "val/all/val"), "docs": 3, "tokens": 25, "bytes": 90},
        {"input": str(text / "val/g1/val.parquet"), "prefix": str(tok / "val/g1/val"), "docs": 4, "tokens": 30, "bytes": 120},
    ]
    tok.mkdir(parents=True, exist_ok=True)
    manifest = {"tokenizer": {"name_or_path": "tok", "vocab_size": 128256, "eos": ["</s>", 2]}, "files": entries}
    (tok / "manifest_0123abcd.json").write_text(json.dumps(manifest))
    train_tokens = 150 + 100 + 250 + tokens_b
    (tok / "train_merged.json").write_text(json.dumps({"docs": 25 + docs_off, "tokens": train_tokens}))
    for prefix in (tok / "val/all/val",):
        _write(Path(str(prefix) + ".bin"), b"BIN")
        _write(Path(str(prefix) + ".idx"), b"IDX")


def test_report_ok():
    with tempfile.TemporaryDirectory() as d:
        text, tok, ref = Path(d) / "text", Path(d) / "tok", Path(d) / "ref" / "val"
        make_tokenized(text, tok)
        _write(Path(str(ref) + ".bin"), b"BIN")
        _write(Path(str(ref) + ".idx"), b"IDX")
        rep = report_tokenized.report(str(text), str(tok), "llama3", str(ref))
        assert rep["ok"], rep["problems"]
        assert rep["train"]["tokens"] == 800 and rep["sources"]["1"]["tokens"] == 500
        assert abs(rep["sources"]["1"]["share"] - 0.625) < 1e-12
        assert rep["val"]["g1"]["tokens"] == 30 and rep["val_all_identical_to_reference"] is True
        assert rep["env"]["VALID_NAMES"] == "all g1"
        assert rep["env"]["VALID_DATA"] == f"{tok}/val/all/val {tok}/val/g1/val"
        assert rep["requirements"]["r"]["ok"]


def test_report_problems():
    with tempfile.TemporaryDirectory() as d:
        text, tok, ref = Path(d) / "text", Path(d) / "tok", Path(d) / "ref" / "val"
        make_tokenized(text, tok, docs_off=1, tokens_b=-560)
        _write(Path(str(ref) + ".bin"), b"BIN")
        _write(Path(str(ref) + ".idx"), b"XXX")
        rep = report_tokenized.report(str(text), str(tok), "llama3", str(ref))
        problems = "\n".join(rep["problems"])
        assert not rep["ok"]
        assert "train/cluster_01/e1_a.parquet: tokenized 11 docs" in problems
        assert "val/all differs" in problems
        assert "requirement r" in problems
        os.remove(tok / "train_merged.json")
        rep = report_tokenized.report(str(text), str(tok), "llama3")
        assert any("train_merged.json missing" in p for p in rep["problems"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
