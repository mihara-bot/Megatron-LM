#!/bin/bash
# Offline GSM8K / MATH-500 / AMC-2023 / Countdown pass@k of an SFT run, or of the pretraining checkpoint it starts
# from. Each Megatron checkpoint is exported to Hugging Face (ladder/export/export_hf.sh, which folds a
# feature-norm head and checks Megatron vs HF logits), then evaluated with vLLM (ladder/sft/eval/eval_vllm.py,
# settings in ladder/sft/eval/math_passk.yaml: 64 samples + greedy per problem, all problems). One GPU:
#   sbatch -J sft-eval -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
#     --mem=128G -t 24:00:00 -o logs/%x-%j.out \
#     --export=ALL,CONFIG=llama3_r1,OPT=muon,LR=2.8e-3,HEAD=hfn,MARGIN_D=0.02,PRETRAIN_TAG=tpp100 \
#     --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh'
#
# Which run: the variables of ladder/sft/train_sft.sh (CONFIG, OPT, LR, HEAD / MARGIN_D / ..., PRETRAIN_TAG,
# SFT_TAG, OUT_ROOT; see ladder/sft/common.sh).
# Optional:
#   TARGET       sft (default) | pretrained: the pretraining checkpoint SFT starts from (the pre-SFT reference)
#   ITERS        latest (default) | all | space-separated iterations
#   DATA_ROOT    eval data root (default /fast/users/zhuangxinlin/data)
#   VENV         Megatron environment for the export (activated by export_hf.sh)
#   VLLM_PYTHON  python of an environment with vLLM (default: python)
#   TASKS, NUM_SAMPLES, MAX_PROBLEMS   override math_passk.yaml; they add a suffix to the output directory
#   FORCE=1      redo the export and the evaluation; DRY_RUN=1 print the commands only
# Output: <SFT dir>/eval/iter_<N>[suffix]/ (TARGET=pretrained: <SFT dir>/eval_pretrained/iter_<N>[suffix]/) with
#   summary.json and <task>.jsonl, and one line per checkpoint in the parent's summary.jsonl.
#   HF exports: <SFT dir>/hf/iter_<N>, or <pretraining run>/hf/<PRETRAIN_TAG>_iter<N>.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
: "${CONFIG:?set CONFIG, e.g. CONFIG=llama3_r1}"
OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}
DATA_ROOT=${DATA_ROOT:-/fast/users/zhuangxinlin/data}
VLLM_PYTHON=${VLLM_PYTHON:-python}
TARGET=${TARGET:-sft}
ITERS=${ITERS:-latest}
source ladder/sft/common.sh

# The checkpoint directory, the head the export must pass (the checkpoint's), and where results go.
case "$TARGET" in
sft)
    CKPT_DIR=$SFT_DIR/ckpt
    EXPORT_HEAD=(HEAD="$( [ "$HEAD" = sfn ] && echo nfn || echo "$HEAD" )" MARGIN_D=0 MARGIN_REVERSE=0)
    EVAL_BASE=$SFT_DIR/eval
    hf_dir() { echo "$SFT_DIR/hf/iter_$1"; }
    ;;
pretrained)
    CKPT_DIR=$PRETRAIN_CKPT
    EXPORT_HEAD=(HEAD="$HEAD" MARGIN_D="$MARGIN_D" MARGIN_REVERSE="$MARGIN_REVERSE")
    EVAL_BASE=$SFT_DIR/eval_pretrained
    hf_dir() { echo "$(dirname "$(dirname "$CKPT_DIR")")/hf/${PRETRAIN_TAG}_iter$1"; }
    ;;
*)
    echo "TARGET must be sft or pretrained, got $TARGET" >&2; exit 1 ;;
esac
EXPORT_HEAD+=(HEAD_SCALE="$HEAD_SCALE" SFN_WEIGHT="$SFN_WEIGHT")

EVAL_ARGS=()
SUFFIX=""
if [ -n "${TASKS:-}" ]; then
    read -r -a T <<< "$TASKS"
    EVAL_ARGS+=(--tasks "${T[@]}"); SUFFIX+="_${TASKS// /_}"
fi
if [ -n "${NUM_SAMPLES:-}" ]; then EVAL_ARGS+=(--num-samples "$NUM_SAMPLES"); SUFFIX+="_n${NUM_SAMPLES}"; fi
if [ -n "${MAX_PROBLEMS:-}" ]; then EVAL_ARGS+=(--max-problems "$MAX_PROBLEMS"); SUFFIX+="_p${MAX_PROBLEMS}"; fi

case "$ITERS" in
latest)
    ITER_LIST=$(tracked_iter "$CKPT_DIR")
    if [ -z "$ITER_LIST" ]; then echo "no checkpoint in $CKPT_DIR" >&2; [ "${DRY_RUN:-0}" = 1 ] || exit 1; fi ;;
all)
    ITER_LIST=$(ls -d "$CKPT_DIR"/iter_* 2>/dev/null | sed 's#.*/iter_0*##' | sort -n | tr '\n' ' ')
    if [ -z "$ITER_LIST" ]; then echo "no iter_* checkpoints in $CKPT_DIR" >&2; [ "${DRY_RUN:-0}" = 1 ] || exit 1; fi ;;
*)
    ITER_LIST=$ITERS ;;
esac
echo "== eval $TARGET of $SFT_NAME: iterations [${ITER_LIST:-?}] of $CKPT_DIR -> $EVAL_BASE"

run() {
    echo "+ $*"
    if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi
}

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
for it in ${ITER_LIST:-}; do
    hf=$(hf_dir "$it")
    out=$EVAL_BASE/iter_${it}${SUFFIX}
    if [ -f "$out/summary.json" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "iteration $it: $out/summary.json exists, skipped (FORCE=1 redoes it)"; continue
    fi
    if [ ! -f "$hf/export_meta.json" ] || [ "${FORCE:-0}" = 1 ]; then
        run env "${EXPORT_HEAD[@]}" CONFIG="$CONFIG" CKPT="$CKPT_DIR" ITER="$it" OUT="$hf" TOKENIZER="$TOKENIZER" \
            bash ladder/export/export_hf.sh
    fi
    run "$VLLM_PYTHON" ladder/sft/eval/eval_vllm.py --model "$hf" --data-root "$DATA_ROOT" --output-dir "$out" \
        ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
    if [ "${DRY_RUN:-0}" != 1 ]; then
        python3 - "$out/summary.json" "$EVAL_BASE/summary.jsonl" "$it" "$TARGET" <<'EOF'
import json, sys
summary, dest, iteration, target = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
s = json.load(open(summary))
row = {"iteration": iteration, "target": target, "output": summary.rsplit("/", 1)[0], **s["metrics"],
       **{k: s[k] for k in ("num_samples", "greedy", "max_problems", "tasks", "model")}}
with open(dest, "a") as f:
    f.write(json.dumps(row) + "\n")
EOF
    fi
done
