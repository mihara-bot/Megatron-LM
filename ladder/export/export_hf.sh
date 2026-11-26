#!/bin/bash
# Export a ladder checkpoint to a Hugging Face directory (plain LlamaForCausalLM / Qwen3ForCausalLM
# that transformers and vLLM load without remote code). One GPU; see ladder/export/export_hf.py.
#
#   CONFIG=llama3_r1 CKPT=$OUT_ROOT/runs/llama3_r1_muon_lr2.8e-3/ckpt/tpp50 bash ladder/export/export_hf.sh
#   CONFIG=llama3_r1 HEAD=hfn MARGIN_D=0.02 \
#       CKPT=$OUT_ROOT/runs/llama3_r1_muon_lr2.8e-3_hfn_d0.02/ckpt/tpp50 bash ladder/export/export_hf.sh
#
# Required:
#   CONFIG     ladder/configs/<CONFIG>.sh of the run (model shape)
#   CKPT       Megatron checkpoint directory (the one holding latest_checkpointed_iteration.txt)
#   TOKENIZER  HF tokenizer directory (the launchers' default for the family is not applied here)
# Optional:
#   HEAD, MARGIN_D, MARGIN_REVERSE, HEAD_SCALE, SFN_WEIGHT   as for training (ladder/configs/head.sh).
#              The head must match the checkpoint: load_checkpoint refuses another one. MARGIN_D only
#              goes into export_meta.json (the margin is training-only).
#   ITER       iteration to export (default: the latest in CKPT)
#   OUT        output directory (default: <run dir>/hf/<ckpt tag>_iter<ITER or latest>)
#   VALIDATE_SEQS / VALIDATE_LEN   Megatron vs HF logits check (default 4 x 512; VALIDATE_SEQS=0 skips)
#   VENV, MASTER_PORT, DRY_RUN=1 (print the command only)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
if [ "${DRY_RUN:-0}" != 1 ]; then source ladder/env/cuda_env.sh; fi

: "${CONFIG:?set CONFIG, e.g. CONFIG=llama3_r1}" "${CKPT:?set CKPT to a checkpoint directory}"
: "${TOKENIZER:?set TOKENIZER to the HF tokenizer directory}"
source "ladder/configs/${CONFIG}.sh"
source ladder/configs/head.sh

CKPT=${CKPT%/}
ITER_TAG=latest
CKPT_ARGS=(--load "$CKPT")
if [ -n "${ITER:-}" ]; then
    CKPT_ARGS+=(--ckpt-step "$ITER")
    ITER_TAG=$ITER
elif [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
    ITER_TAG=$(cat "$CKPT/latest_checkpointed_iteration.txt")
fi
# <run dir>/ckpt/<tag> -> <run dir>/hf/<tag>_iter<N>
OUT=${OUT:-$(dirname "$(dirname "$CKPT")")/hf/$(basename "$CKPT")_iter${ITER_TAG}}

ARGS=(
    "${MODEL_ARGS[@]}"
    ${HEAD_ARGS[@]+"${HEAD_ARGS[@]}"}
    "${CKPT_ARGS[@]}"
    --ckpt-format torch_dist
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER"
    --tokenizer-hf-no-include-special-tokens
    --bf16
    --micro-batch-size 1
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --hf-export-dir "$OUT"
    --hf-validate-seqs "${VALIDATE_SEQS:-4}"
    --hf-validate-len "${VALIDATE_LEN:-512}"
)

echo "== export $CKPT (iteration $ITER_TAG, LM head ${HEAD_ARGS[*]:-plain}) -> $OUT"
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' ladder/export/export_hf.py "${ARGS[@]}"; echo
    exit 0
fi
exec python -m torch.distributed.run --nnodes 1 --nproc-per-node 1 \
    --master-addr localhost --master-port "${MASTER_PORT:-29511}" \
    ladder/export/export_hf.py "${ARGS[@]}"
