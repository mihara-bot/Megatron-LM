"""Tokenize a text dataset into Megatron indexed datasets (.bin/.idx).

Works on any dataset made of parquet or JSON-lines files (.parquet, .jsonl, .json, .jsonl.gz,
.json.gz); only the text field is read, other columns are ignored. The output mirrors the input
tree, one piece per input file, or several when a large parquet file is split into chunks:

    <input-root>/<rel>/<name>.parquet  ->  <output-root>/<rel>/<name>.{bin,idx,json}
                                       or  <output-root>/<rel>/<name>-rg<first>-<last>.{bin,idx,json}

Parquet files are split along row groups into chunks of about --docs-per-chunk documents, so all
workers stay busy to the end however the file sizes are spread; JSON-lines files are one chunk
each. The .json next to each piece records documents, tokens and UTF-8 bytes. Merge the pieces
into one prefix with merge_indexed.py.

Resuming: re-run the same command. A piece is complete once its .idx exists (it is renamed into
place last); complete pieces are skipped and leftovers of interrupted ones are overwritten. Keep
--docs-per-chunk unchanged across re-runs of one output root: the script refuses to mix pieces
of a file cut in two different ways, which would duplicate data in the merge.

Each document becomes  tokenizer(text, add_special_tokens=False).input_ids + [eos]:  no BOS, one
EOS, for every tokenizer family. Train and evaluate with --tokenizer-hf-no-include-special-tokens
so text tokenized at run time (e.g. lm-eval prompts) gets no BOS either. --add-special-tokens
instead prepends the tokenizer's BOS (Megatron's default, = tools/preprocess_data.py --append-eod).
Use the *base* model's tokenizer: the appended EOS is tokenizer.eos_token, which differs between
base and instruct tokenizers.

Speed: tokenization calls the Rust `tokenizers` backend directly (encode_batch_fast, which skips
character offsets), one process per CPU core; the first batches are checked to give exactly the
ids of tokenizer(texts). Each batch is written with a single write.

    python ladder/data/tokenize_dataset.py --input-root /data/climbmix_v1/m1 \
        --output-root /data/megatron/llama3/climbmix_v1_m1 \
        --tokenizer /models/Llama-3.1-8B --workers 64
"""

import argparse
import glob
import gzip
import hashlib
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # one process per core; no nested threads

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from megatron.core.datasets.indexed_dataset import DType, IndexedDatasetBuilder  # noqa: E402

SUFFIXES = (".parquet", ".jsonl", ".json", ".jsonl.gz", ".json.gz")
PIECE_FILES = (".bin", ".bin.tmp", ".idx", ".idx.tmp", ".json", ".json.tmp")

_TOK = None      # transformers tokenizer
_ENCODE = None   # fast batch encoder: list[str] -> list[list[int]]
_ARGS = None


def _make_encoder(tok, add_special_tokens):
    backend = getattr(tok, "backend_tokenizer", None) or getattr(tok, "_tokenizer", None)
    fast = getattr(backend, "encode_batch_fast", None) or getattr(backend, "encode_batch", None)
    if fast is None:
        return lambda texts: tok(texts, add_special_tokens=add_special_tokens).input_ids
    return lambda texts: [e.ids for e in fast(texts, add_special_tokens=add_special_tokens)]


def _init_worker(args):
    global _TOK, _ENCODE, _ARGS
    from transformers import AutoTokenizer

    _ARGS = args
    _TOK = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # Documents are only tokenized, never fed to a model whole, so the model's context length does
    # not apply; lifting it silences "Token indices sequence length is longer than ..." warnings.
    _TOK.model_max_length = sys.maxsize
    _ENCODE = _make_encoder(_TOK, args.add_special_tokens)


def _strip_suffix(path):
    for s in sorted(SUFFIXES, key=len, reverse=True):
        if path.endswith(s):
            return path[: -len(s)]
    raise ValueError(f"unsupported file type: {path}")


def _iter_text_batches(path, key, batch_size, row_groups):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=batch_size, row_groups=row_groups, columns=[key]
        ):
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


def _is_complete(prefix):
    return os.path.exists(prefix + ".idx") and os.path.exists(prefix + ".json")


def _read_stats(prefix):
    with open(prefix + ".json") as f:
        return json.load(f)


def _process_chunk(job):
    in_path, row_groups, prefix, n_rows = job
    if _is_complete(prefix):
        s = _read_stats(prefix)
        s["resumed"], s["rows_planned"] = True, n_rows
        return s

    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    eos = _TOK.eos_token_id
    builder = IndexedDatasetBuilder(prefix + ".bin.tmp", dtype=DType.optimal_dtype(len(_TOK)))
    dtype = builder.dtype
    n_docs = n_tokens = n_bytes = 0
    t0 = time.time()
    for texts in _iter_text_batches(in_path, _ARGS.text_key, _ARGS.batch_size, row_groups):
        texts = [t for t in texts if t]
        if not texts:
            continue
        docs = _ENCODE(texts)
        for ids in docs:
            ids.append(eos)
        lengths = [len(ids) for ids in docs]
        total = sum(lengths)
        flat = np.fromiter(itertools.chain.from_iterable(docs), dtype=dtype, count=total)
        # Batched equivalent of builder.add_document(ids, [len(ids)]) for every document.
        builder.data_file.write(flat.tobytes(order="C"))
        start = len(builder.sequence_lengths)
        builder.sequence_lengths.extend(lengths)
        builder.document_indices.extend(range(start + 1, start + 1 + len(lengths)))
        n_docs += len(texts)
        n_tokens += total
        n_bytes += sum(len(t.encode("utf-8")) for t in texts)
    builder.finalize(prefix + ".idx.tmp")
    os.replace(prefix + ".bin.tmp", prefix + ".bin")
    stats = {"input": in_path, "row_groups": row_groups, "prefix": prefix, "docs": n_docs,
             "tokens": n_tokens, "bytes": n_bytes, "seconds": round(time.time() - t0, 1)}
    with open(prefix + ".json.tmp", "w") as f:
        json.dump(stats, f)
    os.replace(prefix + ".json.tmp", prefix + ".json")
    os.replace(prefix + ".idx.tmp", prefix + ".idx")  # last: .idx marks the piece complete
    stats["rows_planned"] = n_rows
    return stats


def _plan_file(path, base_prefix, docs_per_chunk):
    """Split one input file into chunk jobs: (path, row_groups or None, prefix, rows or None)."""
    if not path.endswith(".parquet"):
        return [(path, None, base_prefix, None)]
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(path).metadata
    rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
    if sum(rows) <= docs_per_chunk or len(rows) == 1:
        return [(path, None, base_prefix, sum(rows))]
    jobs, cur, cur_rows = [], [], 0
    for i, n in enumerate(rows):
        cur.append(i)
        cur_rows += n
        if cur_rows >= docs_per_chunk or i == len(rows) - 1:
            jobs.append((path, cur, f"{base_prefix}-rg{cur[0]:05d}-{cur[-1]:05d}", cur_rows))
            cur, cur_rows = [], 0
    return jobs


def _check_layout(base_prefix, jobs):
    """Refuse to mix pieces of one file produced with different chunkings (would duplicate data)."""
    planned = {j[2] for j in jobs}
    existing = {p[: -len(".idx")] for p in glob.glob(glob.escape(base_prefix) + "*.idx")
                if p[: -len(".idx")] == base_prefix or p[len(base_prefix):].startswith("-rg")}
    stray = existing - planned
    if stray:
        raise SystemExit(
            f"{base_prefix}: existing pieces {sorted(stray)[:3]} do not match the planned chunks; "
            "re-run with the --docs-per-chunk of the earlier run, or delete that file's pieces"
        )


def _self_check(texts_iter, n_batches=3):
    """The fast encoder must give exactly the ids of tokenizer(texts, add_special_tokens=...)."""
    for texts in itertools.islice(texts_iter, n_batches):
        texts = [t for t in texts if t]
        ref = _TOK(texts, add_special_tokens=_ARGS.add_special_tokens).input_ids
        if _ENCODE(texts) != ref:
            raise SystemExit("fast encoder output differs from tokenizer(texts); not safe to use")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", required=True, help="directory holding the input files")
    p.add_argument("--input-glob", default="**/*", help="files under --input-root to read (recursive)")
    p.add_argument("--output-root", required=True)
    p.add_argument("--tokenizer", required=True, help="HF tokenizer path (the base model's)")
    p.add_argument("--text-key", default="text", help="parquet column / JSON field holding the text")
    p.add_argument("--add-special-tokens", action="store_true", help="prepend BOS (default: none; EOS is always appended)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="processes (default: all CPU cores)")
    p.add_argument("--batch-size", type=int, default=1024, help="documents per tokenizer call")
    p.add_argument("--docs-per-chunk", type=int, default=200_000,
                   help="parquet files are split along row groups into chunks of about this many documents")
    args = p.parse_args()

    from tqdm import tqdm

    files = sorted(f for f in glob.glob(os.path.join(args.input_root, args.input_glob), recursive=True)
                   if os.path.isfile(f) and f.endswith(SUFFIXES))
    if not files:
        raise SystemExit(f"no {'/'.join(SUFFIXES)} files match {args.input_glob} under {args.input_root}")

    _init_worker(args)
    eos = _TOK.eos_token_id
    plain = _TOK("a", add_special_tokens=False).input_ids
    with_special = _TOK("a", add_special_tokens=args.add_special_tokens).input_ids
    tok_info = {"name_or_path": _TOK.name_or_path, "vocab_size": len(_TOK),
                "bos": [_TOK.bos_token, _TOK.bos_token_id], "eos": [_TOK.eos_token, eos],
                "prefix_ids_per_doc": with_special[: len(with_special) - len(plain)],
                "add_special_tokens": args.add_special_tokens}
    print(json.dumps(tok_info, ensure_ascii=False), flush=True)
    _self_check(_iter_text_batches(files[0], args.text_key, args.batch_size, None))

    jobs = []
    for f in tqdm(files, desc="planning", unit="file", disable=len(files) < 200):
        base = os.path.join(args.output_root, _strip_suffix(os.path.relpath(f, args.input_root)))
        file_jobs = _plan_file(f, base, args.docs_per_chunk)
        if _is_complete(base) and file_jobs[0][2] != base:
            # Done whole by an earlier run (e.g. one without chunking): keep it, do not re-split.
            file_jobs = [(f, None, base, sum(j[3] or 0 for j in file_jobs) or None)]
        _check_layout(base, file_jobs)
        jobs += file_jobs
    jobs.sort(key=lambda j: -(j[3] or 0))  # largest first: shorter tail at the end
    known_rows = all(j[3] is not None for j in jobs)
    todo = sum(not _is_complete(j[2]) for j in jobs)
    print(f"{len(files)} files -> {len(jobs)} chunks ({len(jobs) - todo} already done), "
          f"{args.workers} workers", flush=True)

    os.makedirs(args.output_root, exist_ok=True)
    t0, tokens_new, total_tokens, results = time.time(), 0, 0, []
    bar = tqdm(total=sum(j[3] for j in jobs) if known_rows else len(jobs),
               unit="doc" if known_rows else "chunk", unit_scale=True, smoothing=0.05,
               mininterval=1 if sys.stderr.isatty() else 30, desc="tokenizing")
    with Pool(args.workers, initializer=_init_worker, initargs=(args,)) as pool:
        for s in pool.imap_unordered(_process_chunk, jobs):
            results.append(s)
            total_tokens += s["tokens"]
            if not s.get("resumed"):
                tokens_new += s["tokens"]
            bar.update(s["rows_planned"] if known_rows else 1)
            bar.set_postfix_str(f"{total_tokens / 1e9:.2f}B tok, "
                                f"{tokens_new / max(time.time() - t0, 1e-9) / 1e6:.1f}M tok/s")
    bar.close()

    results.sort(key=lambda r: r["prefix"])
    manifest = {"tokenizer": tok_info, "input_root": args.input_root, "input_glob": args.input_glob,
                "docs_per_chunk": args.docs_per_chunk, "pieces": len(results),
                "docs": sum(r["docs"] for r in results), "tokens": sum(r["tokens"] for r in results),
                "bytes": sum(r["bytes"] for r in results),
                "files": [{k: v for k, v in r.items() if k not in ("resumed", "rows_planned")} for r in results]}
    name = "manifest_" + hashlib.md5(args.input_glob.encode()).hexdigest()[:8] + ".json"
    with open(os.path.join(args.output_root, name), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"done: {len(files)} files, {len(results)} pieces, {manifest['docs']:,} docs, "
          f"{manifest['tokens'] / 1e9:.3f}B tokens, "
          f"{manifest['bytes'] / max(manifest['tokens'], 1):.3f} bytes/token, "
          f"{time.time() - t0:.0f}s -> {name}")


if __name__ == "__main__":
    main()
