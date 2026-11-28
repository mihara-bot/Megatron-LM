"""Parquet access for the mixing tool: footers, exact text lengths, slices, copies and validation files.

pyarrow everywhere; numpy only to reproduce the v1 val_all order. mixlib.py holds the logic and
never imports this module.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BATCH_ROWS = 65536


# ----------------------------------------------------------------------------- reading


def footer(path: str, text_column: str) -> Dict[str, Any]:
    """Rows, row-group sizes and the text-byte estimate of one file, from its footer only. The
    estimate is the v1 builder's: uncompressed size of the text column minus a 4-byte length prefix
    per row, per row group."""
    meta = pq.ParquetFile(path).metadata
    rg_rows, text_bytes = [], 0
    for rg in range(meta.num_row_groups):
        group = meta.row_group(rg)
        rg_rows.append(group.num_rows)
        for c in range(group.num_columns):
            col = group.column(c)
            if col.path_in_schema == text_column:
                text_bytes += max(0, col.total_uncompressed_size - 4 * group.num_rows)
    st = os.stat(path)
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "rows": int(meta.num_rows), "rg_rows": rg_rows,
            "footer_bytes": text_bytes}


def _as_string(arr):
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return arr
    return arr.cast(pa.string())


def _length_batches(path: str, text_column: str):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=[text_column]):
        yield pc.binary_length(_as_string(batch.column(0))).fill_null(0)


def index_file(path: str, text_column: str) -> Tuple[int, int]:
    """(non-empty documents, exact utf-8 text bytes) of one file (decodes the text column)."""
    docs = nbytes = 0
    for lens in _length_batches(path, text_column):
        docs += pc.sum(pc.cast(pc.greater(lens, 0), pa.int64())).as_py() or 0
        nbytes += pc.sum(lens).as_py() or 0
    return int(docs), int(nbytes)


def file_lengths(path: str, text_column: str) -> List[int]:
    """Text byte length of every row (0 for null or empty)."""
    out: List[int] = []
    for lens in _length_batches(path, text_column):
        out.extend(lens.to_pylist())
    return out


def _row_groups(rg_rows: Sequence[int], lo: int, hi: int) -> List[Tuple[int, int]]:
    """(row group, first row of the group) for the groups that intersect rows [lo, hi)."""
    out, start = [], 0
    for g, n in enumerate(rg_rows):
        if start < hi and start + n > lo:
            out.append((g, start))
        start += n
    return out


def iter_row_slices(path: str, lo: int, hi: int, columns: Optional[List[str]] = None):
    """Tables holding rows [lo, hi) of a file, one per intersecting row group, in order."""
    pf = pq.ParquetFile(path)
    rg_rows = [pf.metadata.row_group(g).num_rows for g in range(pf.metadata.num_row_groups)]
    for g, start in _row_groups(rg_rows, lo, hi):
        t = pf.read_row_group(g, columns=columns)
        a, b = max(lo, start) - start, min(hi, start + t.num_rows) - start
        yield t.slice(a, b - a)


def read_texts(path: str, text_column: str, lo: int, hi: int) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for t in iter_row_slices(path, lo, hi, [text_column]):
        out.extend(_as_string(t.column(0)).to_pylist())
    return out


class ParquetReader:
    """The reader interface of mixlib over parquet files, with an index cache shared across runs
    (entries keyed by path, valid while the file size is unchanged)."""

    def __init__(self, text_column: str, index_cache: Optional[Dict[str, Any]] = None, keep_lengths: int = 8):
        self.col = text_column
        self.cache: Dict[str, Any] = index_cache if index_cache is not None else {}
        self._lengths: "OrderedDict[str, List[int]]" = OrderedDict()
        self.keep = keep_lengths

    def _cached(self, f) -> Optional[Tuple[int, int]]:
        e = self.cache.get(f.path)
        if e and e["size"] == f.size:
            return e["docs"], e["bytes"]
        return None

    def index(self, f) -> Tuple[int, int]:
        hit = self._cached(f)
        if hit is not None:
            return hit
        if f.path in self._lengths:
            lens = self._lengths[f.path]
            res = (sum(1 for n in lens if n > 0), sum(lens))
        else:
            res = index_file(f.path, self.col)
        self.cache[f.path] = {"size": f.size, "docs": res[0], "bytes": res[1]}
        return res

    def lengths(self, f) -> List[int]:
        if f.path in self._lengths:
            self._lengths.move_to_end(f.path)
            return self._lengths[f.path]
        lens = file_lengths(f.path, self.col)
        if len(lens) != f.rows:
            raise RuntimeError(f"{f.path}: read {len(lens)} rows, the footer says {f.rows}")
        self._lengths[f.path] = lens
        if len(self._lengths) > self.keep:
            self._lengths.popitem(last=False)
        if self._cached(f) is None:
            self.cache[f.path] = {"size": f.size, "docs": sum(1 for n in lens if n > 0), "bytes": sum(lens)}
        return lens

    def texts(self, f, lo: int, hi: int) -> List[Optional[str]]:
        return read_texts(f.path, self.col, lo, hi)

    def prefetch(self, files: Sequence[Any], workers: int, on_done=None) -> int:
        """Index the files not yet in the cache, in parallel processes. Returns how many were indexed."""
        todo = [f for f in files if self._cached(f) is None]
        if not todo:
            return 0
        progress = Progress(sum(f.size for f in todo), "index", unit="B")
        with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
            for f, res in zip(todo, ex.map(index_file, [f.path for f in todo], repeat(self.col), chunksize=1)):
                self.cache[f.path] = {"size": f.size, "docs": res[0], "bytes": res[1]}
                progress.update(f.size)
                if on_done is not None:
                    on_done()
        progress.close()
        return len(todo)


# ----------------------------------------------------------------------------- writing


def _tmp(path: str) -> str:
    return path + ".tmp"


def copy_file(src: str, dst: str) -> int:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, _tmp(dst))
    os.replace(_tmp(dst), dst)
    return os.path.getsize(dst)


def write_slice(src: str, dst: str, lo: int, hi: int, row_group_rows: int) -> int:
    """Rows [lo, hi) of src (all columns, same schema) to dst, one source row group in memory at a time."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    schema = pq.ParquetFile(src).schema_arrow
    rows = 0
    w = pq.ParquetWriter(_tmp(dst), schema, compression="zstd")
    try:
        for t in iter_row_slices(src, lo, hi):
            if t.num_rows:
                w.write_table(t, row_group_size=row_group_rows)
                rows += t.num_rows
    finally:
        w.close()
    if rows != hi - lo:
        raise RuntimeError(f"{src}: wrote {rows} rows for [{lo}, {hi})")
    os.replace(_tmp(dst), dst)
    return rows


def output_ok(op: Dict[str, Any], dst: str) -> bool:
    """A finished output: copies have the source size, slices the planned row count."""
    if not os.path.exists(dst):
        return False
    try:
        if op["kind"] == "copy":
            return os.path.getsize(dst) == op["src_size"]
        return pq.ParquetFile(dst).metadata.num_rows == op["hi"] - op["lo"]
    except Exception:  # unreadable: an interrupted or foreign file
        return False


def execute_op(op: Dict[str, Any], out_dir: str, row_group_rows: int) -> Tuple[str, int]:
    dst = os.path.join(out_dir, op["dst"])
    if op["kind"] == "copy":
        copy_file(op["src"], dst)
    else:
        write_slice(op["src"], dst, op["lo"], op["hi"], row_group_rows)
    return op["dst"], op["src_size"]


def write_val(path: str, texts: List[str], cluster_ids: List[int], row_group_rows: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.table({"text": pa.array(texts, pa.string()), "cluster_id": pa.array(cluster_ids, pa.int16())})
    pq.write_table(table, _tmp(path), compression="zstd", row_group_size=row_group_rows)
    os.replace(_tmp(path), path)


def read_val_texts(path: str) -> List[str]:
    return pq.read_table(path, columns=["text"]).column(0).to_pylist()


def val_all_order(n: int, seed: int) -> List[int]:
    """The v1 val_all order: numpy default_rng(seed).permutation over the documents of clusters
    1..N concatenated in cluster order."""
    import numpy as np

    return np.random.default_rng(seed).permutation(n).tolist()


def sha256_file(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tok.model_max_length = sys.maxsize  # documents are only counted, never fed to a model whole
    return tok


def bytes_per_token(texts: List[str], tok, add_eos: bool = True) -> float:
    """Text bytes per token with one EOS per document and no BOS (the v1 calibration)."""
    eos = 1 if add_eos and tok.eos_token_id is not None else 0
    ids = tok(texts, add_special_tokens=False, verbose=False)["input_ids"]
    return sum(len(t.encode("utf-8")) for t in texts) / max(1, sum(len(x) + eos for x in ids))


# ----------------------------------------------------------------------------- progress


class Progress:
    """A line every `every` seconds (and at the end) with done/total, rate and ETA."""

    def __init__(self, total: float, desc: str, unit: str = "B", every: float = 30.0):
        self.total, self.desc, self.unit, self.every = total, desc, unit, every
        self.done = 0.0
        self.t0 = self.last = time.monotonic()

    def _fmt(self, x: float) -> str:
        for u, s in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
            if abs(x) >= s:
                return f"{x / s:.2f}{u}{self.unit}"
        return f"{x:.0f}{self.unit}"

    def update(self, n: float) -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            self._print(now)

    def _print(self, now: float) -> None:
        rate = self.done / max(now - self.t0, 1e-9)
        eta = (self.total - self.done) / rate if rate > 0 else float("inf")
        print(f"[{self.desc}] {self._fmt(self.done)} / {self._fmt(self.total)} "
              f"({100 * self.done / max(self.total, 1):.1f}%), {self._fmt(rate)}/s, ETA {eta / 60:.1f} min",
              flush=True)

    def close(self) -> None:
        self._print(time.monotonic())
