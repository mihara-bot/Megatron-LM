#!/bin/bash
# End-to-end SFT smoke on one node: small data -> ~125 training steps from a real ladder checkpoint -> HF export
# -> quick vLLM eval -> ladder/sft/check_sft_smoke.py. Exercises what the unit tests cannot: VarlenDataset +
# online packing + --calculate-per-token-loss + --pretrained-checkpoint + the SFT head + Muon, the epoch
# checkpoints, the export of SFT checkpoints and vLLM on them.
#   CONFIG=llama3_r1 OPT=muon LR=2.8e-3 PRETRAIN_TAG=main VLLM_PYTHON=<vllm env>/bin/python \
#     sbatch -J sft-smoke -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=128 \
#     --mem=512G -t 03:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/sft/smoke_sft.sh'
# The last line of the log is SMOKE OK, or SMOKE FAILED with the problems.
#
# The pretraining run: the variables of ladder/sft/train_sft.sh (it must exist; any head).
# Optional:
#   SUBSET       the pipeline's SFT subset train split (default $DATA_ROOT/sft/openmathinstruct2_500k/train)
#   SMOKE_ROWS   rows used (default 4000), 2 epochs, global batch SFT_GBS (default 64): ~125 steps
#   VLLM_PYTHON  vLLM environment; without it the eval step is skipped
#   DATA_ROOT, OUT_ROOT, TOKENIZER, GPUS_PER_NODE
# The smoke SFT run is <run>_<PRETRAIN_TAG>_sft_smoke and is deleted and redone on every call.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
: "${CONFIG:?set CONFIG}" "${LR:?set LR}"
export OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}
DATA_ROOT=${DATA_ROOT:-/fast/users/zhuangxinlin/data}
export SFT_TAG=smoke
source ladder/sft/common.sh
SUBSET=${SUBSET:-$DATA_ROOT/sft/openmathinstruct2_500k/train}
SMOKE_ROWS=${SMOKE_ROWS:-4000}
export SFT_GBS=${SFT_GBS:-64}
FAMILY=${CONFIG%%_*}
SMOKE_DATA=$OUT_ROOT/sft_smoke_data/${FAMILY}_${SMOKE_ROWS}
export NNODES=1 NODE_RANK=0 MASTER_ADDR=localhost MASTER_PORT=${MASTER_PORT:-29533}

case "$SFT_DIR" in *_sft_smoke) rm -rf "$SFT_DIR" ;; *) echo "refusing to delete $SFT_DIR" >&2; exit 1 ;; esac
echo "== smoke $SFT_NAME from $PRETRAIN_CKPT"

echo "== [1/5] data: $SMOKE_ROWS rows x 2 epochs -> $SMOKE_DATA"
if [ ! -f "$SMOKE_DATA/stats.json" ]; then
    python ladder/sft/prepare_sft_data.py --input "$SUBSET" --tokenizer "$TOKENIZER" --output-dir "$SMOKE_DATA" \
        --epochs 2 --max-rows "$SMOKE_ROWS" --megatron-check-rows 200
fi

echo "== [2/5] train"
SFT_DATA_DIR=$SMOKE_DATA bash ladder/sft/train_sft.sh

echo "== [3/5] export every checkpoint"
ITERS=all bash ladder/sft/export.sh

if [ -n "${VLLM_PYTHON:-}" ]; then
    echo "== [4/5] eval (gsm8k + countdown, 20 problems, 2 samples)"
    MODEL="$SFT_DIR/hf/iter_*" TASKS="gsm8k countdown" NUM_SAMPLES=2 MAX_PROBLEMS=20 DATA_ROOT=$DATA_ROOT \
        bash ladder/sft/eval.sh
else
    echo "== [4/5] eval skipped (set VLLM_PYTHON)"
fi

echo "== [5/5] check"
python ladder/sft/check_sft_smoke.py --sft-dir "$SFT_DIR" --stats "$SMOKE_DATA/stats.json" \
    --head "$( [ "$HEAD" = sfn ] && echo nfn || echo "$HEAD" )" ${VLLM_PYTHON:+--expect-eval}
