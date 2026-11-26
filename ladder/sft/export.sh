#!/bin/bash
# Export the Megatron checkpoints of an SFT run, or the pretraining checkpoint it starts from, to Hugging Face
# directories (ladder/export/export_hf.sh: plain Llama / Qwen3, a feature-norm head folded into lm_head,
# Megatron vs HF logits checked). Their paths are the MODEL input of ladder/sft/eval.sh. One GPU, Megatron
# environment:
#   CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 PRETRAIN_TAG=tpp100 ITERS=all \
#     sbatch -J sft-export -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
#     --mem=128G -t 04:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh'
#
# Which run: the variables of ladder/sft/train_sft.sh (CONFIG, OPT, LR, HEAD / MARGIN_D / ..., PRETRAIN_TAG,
# SFT_TAG, OUT_ROOT; see ladder/sft/common.sh).
# Optional:
#   TARGET   sft (default) | pretrained: the pretraining checkpoint SFT starts from (the pre-SFT reference)
#   ITERS    latest (default) | all | space-separated iterations
#   VENV     Megatron environment (activated by export_hf.sh); FORCE=1 re-export; DRY_RUN=1 print only
# Output: <SFT dir>/hf/iter_<N>, or for TARGET=pretrained <pretraining run>/hf/<PRETRAIN_TAG>_iter<N>.
# A directory that already has export_meta.json is kept unless FORCE=1.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
: "${CONFIG:?set CONFIG, e.g. CONFIG=llama3_r1}"
OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}
TARGET=${TARGET:-sft}
ITERS=${ITERS:-latest}
source ladder/sft/common.sh

# The checkpoint directory, the head the export passes (the checkpoint's own), and the HF directory per iteration.
case "$TARGET" in
sft)
    CKPT_DIR=$SFT_DIR/ckpt
    EXPORT_HEAD=(HEAD="$( [ "$HEAD" = sfn ] && echo nfn || echo "$HEAD" )" MARGIN_D=0 MARGIN_REVERSE=0)
    hf_dir() { echo "$SFT_DIR/hf/iter_$1"; }
    ;;
pretrained)
    CKPT_DIR=$PRETRAIN_CKPT
    EXPORT_HEAD=(HEAD="$HEAD" MARGIN_D="$MARGIN_D" MARGIN_REVERSE="$MARGIN_REVERSE")
    hf_dir() { echo "$(dirname "$(dirname "$CKPT_DIR")")/hf/${PRETRAIN_TAG}_iter$1"; }
    ;;
*)
    echo "TARGET must be sft or pretrained, got $TARGET" >&2; exit 1 ;;
esac
EXPORT_HEAD+=(HEAD_SCALE="$HEAD_SCALE" SFN_WEIGHT="$SFN_WEIGHT")

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
echo "== export $TARGET of $SFT_NAME: iterations [${ITER_LIST:-?}] of $CKPT_DIR"

DONE=()
for it in ${ITER_LIST:-}; do
    hf=$(hf_dir "$it")
    DONE+=("$hf")
    if [ -f "$hf/export_meta.json" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "iteration $it: $hf already exported (FORCE=1 redoes it)"; continue
    fi
    cmd=(env "${EXPORT_HEAD[@]}" CONFIG="$CONFIG" CKPT="$CKPT_DIR" ITER="$it" OUT="$hf" TOKENIZER="$TOKENIZER"
         bash ladder/export/export_hf.sh)
    echo "+ ${cmd[*]}"
    if [ "${DRY_RUN:-0}" != 1 ]; then "${cmd[@]}"; fi
done
echo "== HF directories (MODEL for ladder/sft/eval.sh):"
printf '   %s\n' ${DONE[@]+"${DONE[@]}"}
