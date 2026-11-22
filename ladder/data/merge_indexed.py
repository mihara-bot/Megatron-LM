"""Merge tokenized pieces (output of tokenize_dataset.py) into a single Megatron prefix.

Unlike tools/merge_datasets.py, which merges one flat directory, this takes a directory tree and
an optional glob, merges every .bin/.idx pair found (sorted by path), and writes <output>.json with
the summed stats (documents, tokens, bytes).

One prefix per dataset keeps training startup cheap (one GPTDataset and one index cache instead
of thousands) and lets Megatron shuffle documents across the whole dataset.

    python ladder/data/merge_indexed.py --input-root /data/megatron/llama3/climbmix_v1_m1/train \
        --output-prefix /data/megatron/llama3/climbmix_v1_m1/train_merged
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", required=True)
    p.add_argument("--input-glob", default="**/*.idx", help="index files to merge, relative to --input-root")
    p.add_argument("--output-prefix", required=True)
    args = p.parse_args()

    out_abs = os.path.abspath(args.output_prefix)
    prefixes = sorted(os.path.splitext(f)[0]
                      for f in glob.glob(os.path.join(args.input_root, args.input_glob), recursive=True)
                      if os.path.abspath(os.path.splitext(f)[0]) != out_abs)
    if not prefixes:
        raise SystemExit(f"no .idx files match {args.input_glob} under {args.input_root}")
    missing = [p for p in prefixes if not os.path.exists(p + ".json")]
    if missing:
        raise SystemExit(f"{len(missing)} pieces have no .json (incomplete?), e.g. {missing[0]}; "
                         "re-run tokenize_dataset.py before merging")

    docs = tokens = nbytes = 0
    dtype = IndexedDataset(prefixes[0]).index.dtype
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    builder = IndexedDatasetBuilder(out_abs + ".bin.tmp", dtype=dtype)
    for i, prefix in enumerate(prefixes, start=1):
        if IndexedDataset(prefix).index.dtype != dtype:
            raise SystemExit(f"{prefix} has a different dtype; were the pieces made with one tokenizer?")
        builder.add_index(prefix)
        stats_path = prefix + ".json"
        if not os.path.exists(stats_path):
            raise SystemExit(f"{stats_path} missing: {prefix} looks incomplete; re-run tokenize_dataset.py")
        with open(stats_path) as f:
            s = json.load(f)
        docs, tokens, nbytes = docs + s["docs"], tokens + s["tokens"], nbytes + s["bytes"]
        if i % 100 == 0 or i == len(prefixes):
            print(f"[{i}/{len(prefixes)}] merged, {tokens / 1e9:.2f}B tokens so far", flush=True)
    builder.finalize(out_abs + ".idx")
    os.replace(out_abs + ".bin.tmp", out_abs + ".bin")

    stats = {"prefix": out_abs, "merged_from": len(prefixes), "docs": docs, "tokens": tokens, "bytes": nbytes}
    with open(out_abs + ".json", "w") as f:
        json.dump(stats, f)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
