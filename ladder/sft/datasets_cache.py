"""Build `datasets`' Arrow cache of the SFT parquet files once, before the training ranks start (train_sft.sh).

    python ladder/sft/datasets_cache.py --root <cache root> <parquet>...    # prints the cache directory to use

Megatron's VarlenLowLevelDataset loads a parquet with datasets.load_dataset("parquet", data_files=path,
split="all"): on first use datasets converts it to Arrow files under HF_DATASETS_CACHE, then memory-maps them.
Every rank of every job does that at once, and the file lock datasets serializes the conversion with does not hold
for jobs on the cluster's shared filesystem: SFT jobs started together rewrote each other's Arrow files under the
readers ("OSError: Expected to be able to read N bytes for message body, got M").

Here one process builds the cache in a private directory, moves it to a second private name and checks that
datasets reuses it there (so a moved cache is a cache hit), then renames it to <root>/<key>. rename is atomic, so
<root>/<key> is complete whenever it exists, and the ranks (HF_DATASETS_CACHE=<root>/<key>) only read it. When
another job renamed its copy first, this one is discarded. key: the datasets version and every file's real path,
size and mtime, so a re-prepared parquet gets a new cache (the old directory can then be deleted). Leftover
<root>/.tmp-* directories (a job killed while building) can be deleted.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys

COMPLETE = "COMPLETE.json"


def log(msg):
    print(f"datasets_cache: {msg}", file=sys.stderr, flush=True)


def cache_key(paths):
    import datasets

    h = hashlib.sha256(f"datasets {datasets.__version__}".encode())
    for p in paths:
        st = os.stat(p)
        h.update(f"\n{os.path.realpath(p)} {st.st_size} {st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def snapshot(d):
    """The cache's data files with size and mtime; not the lock files datasets adds on every load, not COMPLETE."""
    out = {}
    for dirpath, _, files in os.walk(d):
        for f in files:
            if f.endswith(".lock") or f == COMPLETE:
                continue
            st = os.stat(os.path.join(dirpath, f))
            out[os.path.relpath(os.path.join(dirpath, f), d)] = (st.st_size, st.st_mtime_ns)
    return out


def rows(path, cache_dir):
    """What every rank does (VarlenLowLevelDataset), with the cache root given explicitly."""
    from datasets import load_dataset

    return len(load_dataset("parquet", data_files=path, split="all", cache_dir=cache_dir))


def build(paths, root, key):
    import pyarrow.parquet as pq

    final = os.path.join(root, key)
    tag = f".tmp-{key}-{socket.gethostname()}-{os.getpid()}"
    built, moved = os.path.join(root, tag + "-build"), os.path.join(root, tag + "-moved")
    try:
        for p in paths:
            rows(p, built)
        os.rename(built, moved)
        before = snapshot(moved)
        counts = {}
        for p in paths:
            counts[p] = rows(p, moved)
            expected = pq.ParquetFile(p).metadata.num_rows
            if counts[p] != expected:
                raise RuntimeError(f"{p}: datasets reads {counts[p]} rows, the parquet has {expected}")
        if snapshot(moved) != before:
            raise RuntimeError(f"datasets did not reuse its cache after it was moved ({moved}); the ranks would "
                               "convert the parquet again: this datasets version needs another way to pre-build")
        with open(os.path.join(moved, COMPLETE), "w") as f:
            json.dump({"files": [os.path.realpath(p) for p in paths], "rows": counts, "host": socket.gethostname()},
                      f, indent=1)
        try:
            os.rename(moved, final)
            log(f"built {final}")
        except OSError:
            if not os.path.isfile(os.path.join(final, COMPLETE)):
                raise
            log(f"another job built {final} first; using it")
    finally:
        shutil.rmtree(built, ignore_errors=True)
        shutil.rmtree(moved, ignore_errors=True)
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="cache root (the directory HF_DATASETS_CACHE would be)")
    p.add_argument("parquet", nargs="+")
    args = p.parse_args(argv)

    for path in args.parquet:
        if not os.path.isfile(path):
            raise SystemExit(f"datasets_cache: no {path}")
    os.makedirs(args.root, exist_ok=True)
    key = cache_key(args.parquet)
    final = os.path.join(args.root, key)
    if os.path.isfile(os.path.join(final, COMPLETE)):
        log(f"reusing {final}")
    elif os.path.exists(final):
        raise SystemExit(f"datasets_cache: {final} exists without {COMPLETE}, so this script did not build it; "
                         "delete it and rerun")
    else:
        log(f"building the cache of {', '.join(args.parquet)} (key {key})")
        build(args.parquet, args.root, key)
    print(final)


if __name__ == "__main__":
    main()
