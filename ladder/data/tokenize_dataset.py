"""Tokenize a text dataset into Megatron indexed datasets (.bin/.idx), one pair per input file.

Works on any dataset made of parquet or JSON-lines files (.parquet, .jsonl, .json, .jsonl.gz,
.json.gz); only the text field is read, other columns are ignored. Input files are processed in
parallel, and the output mirrors the input tree:

    <input-root>/<rel>/<name>.parquet  ->  <output-root>/<rel>/<name>.{bin,idx,json}

The .json next to each pair records documents, tokens and UTF-8 bytes. Re-running resumes:
files whose .json exists are skipped. Merge the pieces into one prefix with merge_indexed.py.

Each document becomes  tokenizer(text, add_special_tokens=False).input_ids + [eos]:  no BOS, one
EOS, for every tokenizer family. Train and evaluate with --tokenizer-hf-no-include-special-tokens
so text tokenized at run time (e.g. lm-eval prompts) gets no BOS either. --add-special-tokens
instead prepends the tokenizer's BOS (Megatron's default, = tools/preprocess_data.py --append-eod).
Use the *base* model's tokenizer: the appended EOS is tokenizer.eos_token, which differs between
base and instruct tokenizers.

    python ladder/data/tokenize_dataset.py --input-root /data/climbmix_v1/m1/train \
        --output-root /data/megatron/llama3/climbmix_v1_m1/train \
        --tokenizer /models/Llama-3.1-8B --workers 64
"""

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder  # noqa: E402

SUFFIXES = (".parquet", ".jsonl", ".json", ".jsonl.gz", ".json.gz")

_TOKENIZER = None
_ARGS = None


def _init_worker(args):
    global _TOKENIZER, _ARGS
    from transformers import AutoTokenizer

    _ARGS = args
    _TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)


def _strip_suffix(path):
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if path.endswith(s):
            return path[: -len(s)]
    raise ValueError(f"unsupported file type: {path}")


def _iter_text_batches(path, key, batch_size):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(columns=[key], batch_size=batch_size):
            yield batch.column(0).to_pylist()
        return
    opener = gzip.open if path.endswith(".gz") else open
    batch = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                batch.append(json.loads(line).get(key))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def _process_file(job):
    in_path, out_prefix = job
    stats_path = out_prefix + ".json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    tok, args = _TOKENIZER, _ARGS
    eos = tok.eos_token_id
    builder = IndexedDatasetBuilder(out_prefix + ".bin.tmp", dtype=DType.optimal_dtype(len(tok)))
    n_docs = n_tokens = n_bytes = 0
    t0 = time.time()
    for texts in _iter_text_batches(in_path, args.text_key, args.batch_size):
        texts = [t for t in texts if t]
        if not texts:
            continue
        for ids in tok(texts, add_special_tokens=args.add_special_tokens).input_ids:
            ids.append(eos)
            builder.add_document(np.asarray(ids), [len(ids)])
            n_tokens += len(ids)
        n_docs += len(texts)
        n_bytes += sum(len(t.encode("utf-8")) for t in texts)
    builder.finalize(out_prefix + ".idx")
    os.replace(out_prefix + ".bin.tmp", out_prefix + ".bin")
    stats = {"input": in_path, "prefix": out_prefix, "docs": n_docs, "tokens": n_tokens,
             "bytes": n_bytes, "seconds": round(time.time() - t0, 1)}
    with open(stats_path + ".tmp", "w") as f:
        json.dump(stats, f)
    os.replace(stats_path + ".tmp", stats_path)
    return stats


def _tokenizer_info(tokenizer, add_special_tokens):
    with_special = tokenizer("a", add_special_tokens=add_special_tokens).input_ids
    plain = tokenizer("a", add_special_tokens=False).input_ids
    return {
        "name_or_path": tokenizer.name_or_path,
        "vocab_size": len(tokenizer),
        "bos": [tokenizer.bos_token, tokenizer.bos_token_id],
        "eos": [tokenizer.eos_token, tokenizer.eos_token_id],
        "prefix_ids_per_doc": with_special[: len(with_special) - len(plain)],
        "add_special_tokens": add_special_tokens,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", required=True, help="directory holding the input files")
    p.add_argument("--input-glob", default="**/*", help="files under --input-root to read (recursive)")
    p.add_argument("--output-root", required=True)
    p.add_argument("--tokenizer", required=True, help="HF tokenizer path (the base model's)")
    p.add_argument("--text-key", default="text", help="parquet column / JSON field holding the text")
    p.add_argument("--add-special-tokens", action="store_true", help="prepend BOS (default: none; EOS is always appended)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1024, help="documents per tokenizer call")
    args = p.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.input_root, args.input_glob), recursive=True)
                   if os.path.isfile(f) and f.endswith(SUFFIXES))
    if not files:
        raise SystemExit(f"no {'/'.join(SUFFIXES)} files match {args.input_glob} under {args.input_root}")
    jobs = [(f, os.path.join(args.output_root, _strip_suffix(os.path.relpath(f, args.input_root))))
            for f in files]

    _init_worker(args)
    tok_info = _tokenizer_info(_TOKENIZER, args.add_special_tokens)
    print(json.dumps(tok_info, ensure_ascii=False), flush=True)
    os.makedirs(args.output_root, exist_ok=True)

    t0, results = time.time(), []
    with Pool(args.workers, initializer=_init_worker, initargs=(args,)) as pool:
        for i, s in enumerate(pool.imap_unordered(_process_file, jobs), start=1):
            results.append(s)
            total = sum(r["tokens"] for r in results)
            print(f"[{i}/{len(jobs)}] {s['prefix']}: {s['tokens'] / 1e6:.1f}M tokens "
                  f"(total {total / 1e9:.2f}B, {time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda r: r["prefix"])
    manifest = {"tokenizer": tok_info, "input_root": args.input_root, "input_glob": args.input_glob,
                "files": results, "docs": sum(r["docs"] for r in results),
                "tokens": sum(r["tokens"] for r in results), "bytes": sum(r["bytes"] for r in results)}
    name = "manifest_" + hashlib.md5(args.input_glob.encode()).hexdigest()[:8] + ".json"
    with open(os.path.join(args.output_root, name), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"done: {len(results)} files, {manifest['docs']:,} docs, {manifest['tokens'] / 1e9:.3f}B tokens, "
          f"{manifest['bytes'] / max(manifest['tokens'], 1):.3f} bytes/token -> {name}")


if __name__ == "__main__":
    main()
