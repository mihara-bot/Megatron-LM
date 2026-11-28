#!/bin/bash
# Mixed text dataset on the data cluster (orion): plain bash on one large CPU node, no scheduler.
#
#   bash data_mixing/run_orion.sh test       # unit + end-to-end tests on synthetic parquet (seconds)
#   bash data_mixing/run_orion.sh estimate   # quotas / passes from the frozen reference, no data access
#   bash data_mixing/run_orion.sh plan       # exact plan -> $OUT/plan.json (reads footers, indexes files, checks val)
#   bash data_mixing/run_orion.sh build      # copy / slice files, write val/ (resumable: re-run the same command)
#   bash data_mixing/run_orion.sh verify     # check everything, write files.sha256, manifest.json, _DONE
#   bash data_mixing/run_orion.sh all        # plan + build + verify
#
# Then copy $OUT without .cache/ to Polaris (e.g. /fast/users/zhuangxinlin/data/climbmix_mix/m1_200b) and run
# data_mixing/tokenize_polaris.sh there. See data_mixing/README.md.
#
# Environment (defaults in brackets):
#   PYTHON_BIN [python]  needs pyarrow and numpy; transformers only for the TOKENIZERS estimates
#   SPEC  [data_mixing/configs/mixes/climbmix_m1_200b.json]
#   SRC   [/mnt/oss-zhuangxinlin/ClimbMix]            gvlassis/ClimbMix download, cluster_id=<k>/*.parquet
#   OUT   [/mnt/oss-zhuangxinlin/climbmix_mix/m1_200b]
#   WORKERS [64]  processes for indexing, copying/slicing and hashing; raise it until the storage is the limit
#   TOKENIZERS ["llama3=/mnt/users/zhuangxinlin/ckpt/Llama3.1-8B qwen3=/mnt/users/zhuangxinlin/ckpt/Qwen3-0.6B-Base"]
#                 token-count estimates in plan (NAME=PATH, space-separated; empty to skip)
#   PLAN_ARGS / BUILD_ARGS  extra arguments, e.g. PLAN_ARGS=--rechunk, BUILD_ARGS=--clean
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
SPEC=${SPEC:-$ROOT/data_mixing/configs/mixes/climbmix_m1_200b.json}
SRC=${SRC:-/mnt/oss-zhuangxinlin/ClimbMix}
OUT=${OUT:-/mnt/oss-zhuangxinlin/climbmix_mix/m1_200b}
WORKERS=${WORKERS:-64}
TOKENIZERS=${TOKENIZERS-"llama3=/mnt/users/zhuangxinlin/ckpt/Llama3.1-8B qwen3=/mnt/users/zhuangxinlin/ckpt/Qwen3-0.6B-Base"}
MIX="$PYTHON_BIN $ROOT/data_mixing/mix_text.py"

run() {
    echo "+ $*"
    "$@"
}

plan() {
    local tok_args=()
    for t in $TOKENIZERS; do tok_args+=(--tokenizer "$t"); done
    # shellcheck disable=SC2086
    run $MIX plan --spec "$SPEC" --src "$SRC" --out "$OUT" --workers "$WORKERS" ${tok_args[@]+"${tok_args[@]}"} ${PLAN_ARGS:-}
}

build() {
    # shellcheck disable=SC2086
    run $MIX build --out "$OUT" --workers "$WORKERS" ${BUILD_ARGS:-}
}

verify() {
    run $MIX verify --out "$OUT" --workers "$WORKERS"
}

case "${1:-}" in
    test)
        run "$PYTHON_BIN" "$ROOT/data_mixing/tests/test_mixlib.py"
        run "$PYTHON_BIN" "$ROOT/data_mixing/tests/test_polaris_tools.py"
        run "$PYTHON_BIN" "$ROOT/data_mixing/tests/test_parquet_e2e.py" ;;
    estimate) run $MIX estimate --spec "$SPEC" ;;
    plan) plan ;;
    build) build ;;
    verify) verify ;;
    all) plan; build; verify ;;
    *) echo "usage: bash data_mixing/run_orion.sh <test|estimate|plan|build|verify|all>" >&2; exit 1 ;;
esac
