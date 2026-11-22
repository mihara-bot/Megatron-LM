"""Inspect a Megatron indexed dataset prefix (<prefix>.bin + <prefix>.idx).

Prints what the .idx header holds (dtype, sequence and document counts), document length
statistics, checks that the .bin size matches the index, and decodes the first and last tokens of
a few documents so the BOS/EOS layout is visible.

    python ladder/data/inspect_indexed.py /data/megatron/llama3/climbmix_v1_m1/train_merged \
        --tokenizer /models/Llama-3.1-8B --docs 0 1 -1
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from megatron.core.datasets.indexed_dataset import IndexedDataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix", help="path without the .bin/.idx suffix")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer path, to decode tokens")
    p.add_argument("--docs", type=int, nargs="*", default=[0, -1])
    p.add_argument("--show", type=int, default=24, help="tokens shown at each end of a document")
    args = p.parse_args()

    ds = IndexedDataset(args.prefix, mmap=True)
    idx = ds.index
    lengths = idx.sequence_lengths.astype(np.int64)
    doc_bounds = idx.document_indices
    itemsize = np.dtype(idx.dtype).itemsize
    bin_bytes = os.path.getsize(args.prefix + ".bin")

    print(f"dtype            {np.dtype(idx.dtype).name} ({itemsize} bytes/token)")
    print(f"sequences        {len(lengths)}")
    print(f"documents        {len(doc_bounds) - 1}")
    print(f"tokens           {lengths.sum():,}")
    print(f".bin size        {bin_bytes:,} bytes (index implies {lengths.sum() * itemsize:,})")
    assert bin_bytes == lengths.sum() * itemsize, ".bin size does not match the index"
    q = np.percentile(lengths, [0, 50, 90, 99, 100]).astype(int)
    print(f"doc length       min {q[0]}  p50 {q[1]}  p90 {q[2]}  p99 {q[3]}  max {q[4]}  mean {lengths.mean():.0f}")

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"tokenizer        bos={tok.bos_token!r}({tok.bos_token_id})  eos={tok.eos_token!r}({tok.eos_token_id})")

    n_docs = len(doc_bounds) - 1
    for d in args.docs:
        d = d % n_docs
        ids = np.concatenate([ds[i] for i in range(doc_bounds[d], doc_bounds[d + 1])])
        head, tail = ids[: args.show].tolist(), ids[-args.show :].tolist()
        print(f"\n--- document {d}: {len(ids)} tokens")
        print(f"first ids {head}")
        print(f"last  ids {tail}")
        if tok is not None:
            print(f"first text {tok.decode(head)!r}")
            print(f"last  text {tok.decode(tail)!r}")


if __name__ == "__main__":
    main()
