#!/bin/bash
# Build the margin toy's tokenizer and data (ladder/toy/TOY.md). CPU only, a few minutes. Run from the Megatron-LM
# root with the environment active (transformers, tokenizers, pyarrow, numpy, Megatron):
#   TOY_ROOT=/fast/users/zhuangxinlin/toy_margin bash ladder/toy/prepare.sh
#
# Output:
#   $TOY_ROOT/tokenizer/   the 100-token HF tokenizer: TOKENIZER of every toy run (ladder/toy/make_tokenizer.py)
#   $TOY_ROOT/data/        graph.json, pretrain/{train,val}.jsonl, sft/<mode>/{train.parquet,stats.json}
#                          (ladder/toy/make_data.py; the SFT rows are checked with Megatron's SFTTokenizer)
#   $TOY_ROOT/megatron/    train / val indexed datasets (TRAIN_DATA / VALID_DATA) + the manifest check_tokenizer.py reads
# Optional: MAKE_DATA_ARGS (extra make_data.py arguments, e.g. "--seed 1"), WORKERS (tokenizer processes, 16).
# Rerunning regenerates the same data (fixed seeds); tokenize_dataset.py skips pieces it already wrote.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
: "${TOY_ROOT:?set TOY_ROOT, e.g. /fast/users/zhuangxinlin/toy_margin}"
PY=${PYTHON:-python}

$PY ladder/toy/make_tokenizer.py --out "$TOY_ROOT/tokenizer"
# shellcheck disable=SC2086
$PY ladder/toy/make_data.py --out "$TOY_ROOT/data" --check-tokenizer "$TOY_ROOT/tokenizer" ${MAKE_DATA_ARGS:-}
$PY ladder/data/tokenize_dataset.py --input-root "$TOY_ROOT/data/pretrain" --input-glob '*.jsonl' \
    --output-root "$TOY_ROOT/megatron" --tokenizer "$TOY_ROOT/tokenizer" --workers "${WORKERS:-16}"
$PY ladder/data/inspect_indexed.py "$TOY_ROOT/megatron/train" --tokenizer "$TOY_ROOT/tokenizer" --docs 0 1 -1
$PY ladder/data/check_tokenizer.py --tokenizer "$TOY_ROOT/tokenizer" --expect-vocab 100 \
    "$TOY_ROOT/megatron/train" "$TOY_ROOT/megatron/val"
echo "toy data ready under $TOY_ROOT (TRAIN_DATA=$TOY_ROOT/megatron/train VALID_DATA=$TOY_ROOT/megatron/val)"
