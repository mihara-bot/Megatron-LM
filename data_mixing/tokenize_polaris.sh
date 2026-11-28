#!/bin/bash
# Text dataset (copied from orion) -> Megatron indexed data on Polaris, with the unchanged ladder/data scripts.
# Run from the Megatron-LM repository root in the Megatron environment; one run per tokenizer family.
#
#   FAMILY=llama3 bash data_mixing/tokenize_polaris.sh test      # CPU tests incl. a real tokenize of synthetic data
#   FAMILY=llama3 bash data_mixing/tokenize_polaris.sh all       # verify + tokenize + merge + inspect + report
#   FAMILY=llama3 bash data_mixing/tokenize_polaris.sh cleanup   # after a passing report: delete the train pieces
#   steps one at a time: verify | tokenize | merge | inspect | report | cleanup
#
# On SLURM submit the two sbatch files next to this script (logs in /fast/users/<user>/runs/):
#   FAMILY=llama3 sbatch data_mixing/test_polaris.sbatch                          # tests, 1 GPU, minutes
#   FAMILY=llama3 sbatch --job-name=tok-llama3 data_mixing/tokenize_polaris.sbatch  # STEP=all, 128 CPUs
# tokenize is resumable: submitting the same job again skips the finished pieces.
#
# Environment (defaults in brackets):
#   FAMILY  [llama3]   llama3 | qwen3 | any other name with TOK set (e.g. deepseek)
#   TOK     [per family: /fast/users/zhuangxinlin/tokenizers/{llama-3,qwen3}]  the *base* model's tokenizer
#   TEXT    [/fast/users/zhuangxinlin/data/climbmix_mix/m1_200b]               the copied text dataset
#   OUT     [/fast/users/zhuangxinlin/megatron_data/$FAMILY/climbmix_m1_200b]  tokenized output
#   V1_VAL  [/fast/users/zhuangxinlin/megatron_data/$FAMILY/climbmix_v1_m1/val_all/val]  reference val_all
#           prefix; report checks val/all is byte-identical to it (skipped when it does not exist)
#   WORKERS [$SLURM_CPUS_PER_TASK or all cores]
#
# Disk: the train pieces and train_merged each take ~4 bytes/token (~0.8 TB at 200B tokens) until cleanup.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
FAMILY=${FAMILY:-llama3}
case $FAMILY in
    llama3) DEFAULT_TOK=/fast/users/zhuangxinlin/tokenizers/llama-3 ;;
    qwen3)  DEFAULT_TOK=/fast/users/zhuangxinlin/tokenizers/qwen3 ;;
    *)      DEFAULT_TOK= ;;
esac
TOK=${TOK:-$DEFAULT_TOK}
: "${TOK:?set TOK to the tokenizer of FAMILY=$FAMILY}"
TEXT=${TEXT:-/fast/users/zhuangxinlin/data/climbmix_mix/m1_200b}
OUT=${OUT:-/fast/users/zhuangxinlin/megatron_data/$FAMILY/climbmix_m1_200b}
V1_VAL=${V1_VAL:-/fast/users/zhuangxinlin/megatron_data/$FAMILY/climbmix_v1_m1/val_all/val}
WORKERS=${WORKERS:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
PYTHON_BIN=${PYTHON_BIN:-python}

run() {
    echo "+ $*"
    "$@"
}

step_test()     {
    run "$PYTHON_BIN" data_mixing/tests/test_mixlib.py
    run "$PYTHON_BIN" data_mixing/tests/test_polaris_tools.py
    run "$PYTHON_BIN" -c 'import numpy, pyarrow; print("numpy", numpy.__version__, "pyarrow", pyarrow.__version__)'
    run env DM_TOKENIZER="$TOK" "$PYTHON_BIN" data_mixing/tests/test_parquet_e2e.py
    echo "all data_mixing tests passed"
}
step_verify()   { run "$PYTHON_BIN" data_mixing/verify_files.py "$TEXT" --workers "$WORKERS"; }
step_tokenize() {
    run "$PYTHON_BIN" ladder/data/tokenize_dataset.py --input-root "$TEXT" --input-glob '**/*.parquet' \
        --output-root "$OUT" --tokenizer "$TOK" --workers "$WORKERS"
}
step_merge()    { run "$PYTHON_BIN" ladder/data/merge_indexed.py --input-root "$OUT/train" --output-prefix "$OUT/train_merged"; }
step_inspect()  {
    run "$PYTHON_BIN" ladder/data/inspect_indexed.py "$OUT/train_merged" --tokenizer "$TOK"
    run "$PYTHON_BIN" ladder/data/inspect_indexed.py "$OUT/val/all/val" --tokenizer "$TOK"
}
step_report()   {
    local cmp=()
    if [ -f "$V1_VAL.bin" ]; then cmp=(--compare-val-all "$V1_VAL"); else echo "no $V1_VAL.bin: val/all not compared"; fi
    run "$PYTHON_BIN" data_mixing/report_tokenized.py --text-root "$TEXT" --tok-root "$OUT" --family "$FAMILY" ${cmp[@]+"${cmp[@]}"}
}
step_cleanup()  {
    "$PYTHON_BIN" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["ok"] else 1)' "$OUT/report.json" \
        || { echo "cleanup needs a passing $OUT/report.json" >&2; exit 1; }
    [ -f "$OUT/train_merged.bin" ] && [ -f "$OUT/train_merged.idx" ] || { echo "no train_merged in $OUT" >&2; exit 1; }
    run rm -rf "$OUT/train"
    echo "kept: $OUT/train_merged.*, $OUT/val/, manifest_*.json, report.json"
}

case "${1:-}" in
    test|verify|tokenize|merge|inspect|report|cleanup) "step_$1" ;;
    all) step_verify; step_tokenize; step_merge; step_inspect; step_report ;;
    *) echo "usage: FAMILY=<llama3|qwen3|...> bash data_mixing/tokenize_polaris.sh <test|all|verify|tokenize|merge|inspect|report|cleanup>" >&2
       exit 1 ;;
esac
