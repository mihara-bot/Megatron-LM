# Copied from pretrain_LLM_pipeline src/cli/prepare_countdown_eval.py (commit 1aa819b6); run as a script:
#   python ladder/sft/eval/prepare_countdown_eval.py --input <dir> --output <jsonl>
"""Cut the Countdown eval set from Jiayi-Pan/Countdown-Tasks-3to4 the way TinyZero does.

TinyZero (examples/data_preprocess/countdown.py) trains on rows [0, 327680) of the single train
split and tests on the next 1024 rows, [327680, 328704). This writes those test rows, in order, as
jsonl with ``id`` (row index in the source), ``nums`` and ``target``:

    python ladder/sft/eval/prepare_countdown_eval.py \\
        --input /mnt/users/zhuangxinlin/data/Countdown-Tasks-3to4 \\
        --output /mnt/users/zhuangxinlin/data/countdown/test_tinyzero_1024.jsonl

``--input`` is the downloaded dataset directory (its ``*.parquet`` files are read in sorted path
order, as one table) or a single parquet file.
"""

import argparse
import glob
import json
import os
from typing import List

TINYZERO_TRAIN_SIZE = 327680
TINYZERO_TEST_SIZE = 1024


def parquet_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .parquet files under {path}.")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=TINYZERO_TRAIN_SIZE)
    parser.add_argument("--count", type=int, default=TINYZERO_TEST_SIZE)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    files = parquet_files(args.input)
    table = pa.concat_tables([pq.read_table(f, columns=["target", "nums"]) for f in files])
    end = args.start + args.count
    if table.num_rows < end:
        raise ValueError(f"{args.input} has {table.num_rows} rows; need at least {end}.")
    rows = table.slice(args.start, args.count).to_pylist()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    tmp = f"{args.output}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for offset, row in enumerate(rows):
            record = {"id": args.start + offset, "nums": [int(n) for n in row["nums"]], "target": int(row["target"])}
            f.write(json.dumps(record) + "\n")
    os.replace(tmp, args.output)
    sizes = {}
    for row in rows:
        sizes[len(row["nums"])] = sizes.get(len(row["nums"]), 0) + 1
    print(f"read {table.num_rows} rows from {len(files)} file(s); wrote rows [{args.start}, {end}) "
          f"to {args.output} (numbers per problem: {dict(sorted(sizes.items()))})")


if __name__ == "__main__":
    main()
