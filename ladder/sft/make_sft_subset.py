"""Draw disjoint random train / validation subsets of a JSONL/Parquet dataset for SFT (OpenMathInstruct-2).

    python ladder/sft/make_sft_subset.py --input $DATA/OpenMathInstruct-2/data --file-glob 'train-*' \\
        --output-dir $DATA/sft/openmathinstruct2_500k --num-train 500000 --num-validation 2000 --seed 42 \\
        --columns problem generated_solution expected_answer problem_source --disjoint-column problem

Self-contained (Megatron environment; pyarrow for Parquet input). The draw:
  * the matched files (names sorted at every level, hidden entries skipped; --file-glob matches the basename or
    the path relative to --input) are numbered as one sequence of rows; a JSONL row is a non-empty line;
  * one random.Random(seed).sample(range(total), needed) without replacement; the first --num-validation draws
    are the validation set, the training set takes the next draws in draw order;
  * with --disjoint-column, a training candidate whose value occurs in the validation set is skipped
    (OpenMathInstruct-2 has many solutions per problem, so validation problems stay unseen); --train-oversample
    extra candidates (a fraction of --num-train) make up for the skipped ones.
This is the same draw as pretrain_LLM_pipeline's SFT subset, so the same source files and seed select the same rows
in both (e.g. to compare SFT across the two code bases); nothing here imports it.

Output, under --output-dir:
  train/train.jsonl, validation/validation.jsonl   rows in source order, sample_id (the global row index) first,
                                                    then --columns
  manifest.json                                     source files and row counts, seed, skipped candidates, counts
The train split is the --input of prepare_sft_data.py.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import random
import sys
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_sft_data import list_data_files  # noqa: E402

Span = Tuple[str, int, int]  # (file, global start row, rows)


def select_files(path: str, pattern: str) -> List[str]:
    """Data files under path whose basename or path relative to path matches pattern."""
    all_files = list_data_files(path)
    base = path if os.path.isdir(path) else os.path.dirname(path)
    files = [f for f in all_files
             if fnmatch.fnmatch(os.path.basename(f), pattern) or fnmatch.fnmatch(os.path.relpath(f, base), pattern)]
    if not files:
        listing = "\n  ".join(os.path.relpath(f, base) for f in all_files[:40])
        raise ValueError(f"no data files matching {pattern!r} under {path}; there are:\n  {listing}")
    return files


def count_rows(path: str) -> int:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def file_spans(files: Sequence[str]) -> List[Span]:
    spans, start = [], 0
    for f in files:
        n = count_rows(f)
        if n > 0:
            spans.append((f, start, n))
            start += n
    return spans


def draw_indices(total: int, num_train: int, num_validation: int, seed: int, extra: int) -> List[int]:
    """Global row indices in draw order: validation first, then training candidates."""
    if num_train <= 0 or num_validation < 0:
        raise ValueError("num_train must be > 0 and num_validation >= 0")
    if num_validation + num_train > total:
        raise ValueError(f"requested {num_validation + num_train} rows but the source has {total}")
    return random.Random(seed).sample(range(total), min(total, num_validation + num_train + extra))


def split_draws(draws: Sequence[int], num_train: int, num_validation: int,
                key: Optional[Callable[[int], Hashable]] = None) -> Tuple[List[int], List[int], int]:
    """(train, validation, skipped): indices sorted (source order); skipped = candidates dropped for overlap."""
    validation = sorted(draws[:num_validation])
    held_out = {key(i) for i in validation} if key else set()
    train: List[int] = []
    skipped = 0
    for i in draws[num_validation:]:
        if len(train) == num_train:
            break
        if key and key(i) in held_out:
            skipped += 1
            continue
        train.append(i)
    if len(train) < num_train:
        raise ValueError(f"only {len(train)} training rows after skipping {skipped} that overlap the validation "
                         "set; raise --train-oversample")
    return sorted(train), validation, skipped


def read_file_rows(path: str, local: List[int], columns: List[str]) -> Dict[int, Dict]:
    """Rows of one file at sorted file-local indices, keyed by local index."""
    if path.endswith(".parquet"):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns).take(pa.array(local, type=pa.int64()))
        return dict(zip(local, table.to_pylist()))
    wanted, out, row = set(local), {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if row in wanted:
                out[row] = json.loads(line)
                if len(out) == len(wanted):
                    break
            row += 1
    return out


def read_rows(spans: Sequence[Span], indices: Sequence[int], columns: List[str]) -> Dict[int, Dict]:
    """Rows at global indices, keyed by global index, each {sample_id, *columns}."""
    out: Dict[int, Dict] = {}
    wanted, cursor = sorted(set(indices)), 0
    for path, start, count in spans:
        local = []
        while cursor < len(wanted) and wanted[cursor] < start + count:
            local.append(wanted[cursor] - start)
            cursor += 1
        if not local:
            continue
        rows = read_file_rows(path, local, columns)
        for i in local:
            missing = [c for c in columns if c not in rows[i]]
            if missing:
                raise KeyError(f"{path} row {i} lacks column(s) {missing}")
            out[start + i] = {"sample_id": start + i, **{c: rows[i][c] for c in columns}}
        print(f"read {len(local)} row(s) from {os.path.basename(path)}", flush=True)
    return out


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="data file or directory (searched recursively)")
    parser.add_argument("--file-glob", default="*", help="keep files whose basename or relative path matches")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-train", type=int, required=True)
    parser.add_argument("--num-validation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--columns", nargs="+", required=True, help="columns to keep")
    parser.add_argument("--disjoint-column", default=None,
                        help="skip training candidates whose value in this column occurs in the validation set")
    parser.add_argument("--train-oversample", type=float, default=0.2,
                        help="extra training candidates, as a fraction of --num-train, with --disjoint-column")
    args = parser.parse_args(argv)

    columns = list(args.columns)
    if args.disjoint_column and args.disjoint_column not in columns:
        columns.append(args.disjoint_column)
    spans = file_spans(select_files(args.input, args.file_glob))
    total = sum(n for _, _, n in spans)
    extra = int(round(args.num_train * args.train_oversample)) if args.disjoint_column else 0
    draws = draw_indices(total, args.num_train, args.num_validation, args.seed, extra)
    print(f"{len(spans)} file(s), {total} rows; drawing {len(draws)}", flush=True)
    rows = read_rows(spans, draws, columns)
    key = (lambda i: rows[i][args.disjoint_column]) if args.disjoint_column else None
    train, validation, skipped = split_draws(draws, args.num_train, args.num_validation, key)

    counts = {}
    for split, idx in (("train", train), ("validation", validation)):
        if idx:
            write_jsonl(os.path.join(args.output_dir, split, f"{split}.jsonl"), [rows[i] for i in idx])
            counts[split] = len(idx)
    manifest = {
        "source": os.path.abspath(args.input),
        "file_glob": args.file_glob,
        "files": [{"path": p, "global_start": s, "rows": n} for p, s, n in spans],
        "source_rows": total,
        "seed": args.seed,
        "columns": columns,
        "disjoint_column": args.disjoint_column,
        "train_candidates_skipped_for_overlap": skipped,
        "counts": counts,
    }
    if args.disjoint_column:
        manifest["validation_unique_values"] = len({rows[i][args.disjoint_column] for i in validation})
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"output_dir": args.output_dir, "source_rows": total, "skipped": skipped, **counts}))
    return manifest


if __name__ == "__main__":
    main()
