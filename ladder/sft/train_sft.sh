#!/bin/bash
# SFT of a ladder checkpoint on OpenMathInstruct-2, the Megatron counterpart of pretrain_LLM_pipeline's
# configs/exp/sft/*_sft_omi2_500k_ep10.yaml (docs/sft.md there). Same script on every node; the launchers start
# it with ENTRY=ladder/sft/train_sft.sh:
#   CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 PRETRAIN_TAG=tpp100 \
#     SFT_DATA_DIR=/fast/users/zhuangxinlin/data/sft_megatron/omi2_500k_ep10_llama3 \
#     ENTRY=ladder/sft/train_sft.sh sbatch --nodes=1 -J sft-r1-hfn ladder/launch/slurm.sbatch
#
# Start point: the latest checkpoint of the pretraining run in $OUT_ROOT/runs/<run>/ckpt/<PRETRAIN_TAG>
# (ladder/sft/common.sh names it from CONFIG / OPT / LR / HEAD vars), model weights only.
# LM head (protocol A): the pretraining head, no margin, no sfn penalty (hfn -> hfn, nfn/sfn -> nfn).
#
# Same as the pipeline SFT:
#   data      ladder/sft/prepare_sft_data.py output: prompt + response + EOS, prompt masked, the pipeline's
#             per-epoch order; Megatron reads train.parquet front to back (--use-varlen-dataset, identity template)
#   optimizer Muon (ladder/configs/opt/muon.sh, as the pipeline SFT; SFT_OPT=adamw switches), peak LR =
#             pretraining LR / 3, 3% linear warmup, cosine to 0
#   batch     the pipeline steps over 32 packed 4096-token rows (~315 examples); here the global batch counts
#             examples (online packing, dp_balanced) and defaults to stats.json's match for this DP size
#   one checkpoint per epoch, all kept; pass@k is computed offline on them (export.sh, eval.sh)
# Different: the sampling order inside a step and the packing (online, balanced over DP ranks).
# Added: a validation loss on the held-out split when $SFT_DATA_DIR/valid.parquet exists (prepare_sft_data.py
# --output-name valid); the pipeline logged none. It reads no training data and changes nothing in training.
#   Every evaluation covers the whole set once, in one eval iteration whose global batch is the set (packed like
#   training), so the curve is on fixed examples; one evaluation per epoch by default (at the epoch checkpoints),
#   plus one at the end of training (the final checkpoint).
#
# Required: CONFIG, LR, OUT_ROOT, SFT_DATA_DIR (TOKENIZER defaults to the family's, see common.sh)
# Optional: OPT (the pretraining run's optimizer, default muon: it names the run), SFT_OPT (default muon),
#   HEAD / MARGIN_D / MARGIN_REVERSE / HEAD_SCALE / SFN_WEIGHT (the pretraining run's), PRETRAIN_TAG,
#   PRETRAIN_RUN, SFT_TAG, SFT_LR, SFT_GBS (examples per step), SEED (default 1234), WANDB_PROJECT, VENV, EXTRA_ARGS,
#   SFT_TOKENS_PER_RANK (tokens per packed microbatch per GPU, default SEQ_LEN; larger = fewer, bigger microbatches,
#   same examples per step and same loss), SFT_EVAL (auto: validation when valid.parquet exists | 1: require it |
#   0: off), SFT_EVAL_INTERVAL (default: iterations per epoch, the checkpoint interval),
#   NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT / GPUS_PER_NODE (set by the launchers), DRY_RUN=1,
#   ALLOW_UNFINISHED_PRETRAIN=1 (start although the pretraining checkpoint is short of its run's target)
# Output: $OUT_ROOT/sft/<run>_<PRETRAIN_TAG>_sft_<SFT_TAG>/{ckpt,tb,cmd_*.sh,sft_data_stats.json}.
# Rerunning resumes from that ckpt directory; the pretraining checkpoint is only the starting fallback.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
if [ "${DRY_RUN:-0}" != 1 ]; then source ladder/env/cuda_env.sh; fi

: "${CONFIG:?set CONFIG, e.g. CONFIG=llama3_r1}" "${SFT_DATA_DIR:?set SFT_DATA_DIR (ladder/sft/prepare_sft_data.py output)}"
source "ladder/configs/${CONFIG}.sh"
source ladder/sft/common.sh                   # OPT names the pretraining run; SFT_OPT is the SFT optimizer
source "ladder/configs/opt/${SFT_OPT}.sh"

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
SEED=${SEED:-1234}
DP=$((NNODES * GPUS_PER_NODE))   # TP = PP = CP = 1

STATS=$SFT_DATA_DIR/stats.json
DATA=$SFT_DATA_DIR/train.parquet
if [ ! -f "$STATS" ]; then echo "no $STATS: run ladder/sft/prepare_sft_data.py first" >&2; exit 1; fi
# examples per epoch, epochs, rows in train.parquet, the pipeline's examples per step, sequence length
read -r EXAMPLES EPOCHS ROWS EX_PER_STEP DATA_SEQ < <(python3 -c '
import json, sys
s = json.load(open(sys.argv[1]))
print(s["examples"], s["epochs"], s["rows_out"], s["pipeline"]["examples_per_step"], s["seq_len"])' "$STATS")
if [ "$DATA_SEQ" != "$SEQ_LEN" ]; then
    echo "$STATS was prepared for seq_len $DATA_SEQ, the config trains $SEQ_LEN" >&2; exit 1
fi
# The config's GBS is the pretraining batch (sequences); the SFT batch counts examples.
GBS=${SFT_GBS:-$(awk -v e="$EX_PER_STEP" -v dp="$DP" 'BEGIN { k = int(e / dp + 0.5); if (k < 1) k = 1; print k * dp }')}
if [ $((GBS % DP)) -ne 0 ]; then echo "SFT_GBS=$GBS must be a multiple of the data-parallel size $DP" >&2; exit 1; fi
TRAIN_ITERS=$((ROWS / GBS))            # every step reads GBS rows of train.parquet, never past its end
EPOCH_ITERS=$((EXAMPLES / GBS))        # ~ one epoch: the per-epoch checkpoint interval
SFT_LR=${SFT_LR:-$(awk -v x="$LR" 'BEGIN { printf "%.6g", x / 3 }')}
# Packed microbatch size per GPU (--max-seqlen-per-dp-cp-rank). The loss is per token over the whole step and
# attention stays within each example (cu_seqlens), so this only trades memory for fewer microbatches.
SFT_TOKENS_PER_RANK=${SFT_TOKENS_PER_RANK:-$SEQ_LEN}
if ! [[ "$SFT_TOKENS_PER_RANK" =~ ^[0-9]+$ ]] || [ "$SFT_TOKENS_PER_RANK" -lt "$SEQ_LEN" ]; then
    echo "SFT_TOKENS_PER_RANK=$SFT_TOKENS_PER_RANK must be an integer >= the sequence length $SEQ_LEN" >&2; exit 1
fi

# Validation loss on $SFT_DATA_DIR/valid.parquet (see the header).
SFT_EVAL=${SFT_EVAL:-auto}
VALID=$SFT_DATA_DIR/valid.parquet
VALID_STATS=$SFT_DATA_DIR/valid_stats.json
EVAL_ARGS=(--eval-iters 0 --eval-interval "$TRAIN_ITERS")
case "$SFT_EVAL" in
0)
    EVAL_NOTE="no validation (SFT_EVAL=0)" ;;
1 | auto)
    if [ -f "$VALID" ] && [ -f "$VALID_STATS" ]; then
        VAL_ROWS=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["rows_out"])' "$VALID_STATS")
        EVAL_GBS=$((VAL_ROWS - VAL_ROWS % DP))        # the eval global batch must divide by the DP size
        if [ "$EVAL_GBS" -lt "$DP" ]; then echo "$VALID_STATS: $VAL_ROWS rows, fewer than the $DP GPUs" >&2; exit 1; fi
        EVAL_INTERVAL=${SFT_EVAL_INTERVAL:-$EPOCH_ITERS}   # every epoch, at the same iterations as the checkpoints
        if [ "$EVAL_INTERVAL" -lt 1 ]; then EVAL_INTERVAL=1; fi
        EVAL_ARGS=(--valid-data-path "$VALID" --eval-iters 1 --eval-global-batch-size "$EVAL_GBS"
                   --eval-interval "$EVAL_INTERVAL")
        EVAL_NOTE="validation loss on $EVAL_GBS examples ($VALID) every $EVAL_INTERVAL iterations"
        if [ $((VAL_ROWS % DP)) -ne 0 ]; then
            EVAL_NOTE+="; $VAL_ROWS rows is not a multiple of $DP GPUs, so evaluations rotate by $((VAL_ROWS % DP))"
        fi
    elif [ "$SFT_EVAL" = 1 ]; then
        echo "SFT_EVAL=1 but no $VALID + $VALID_STATS (prepare_sft_data.py --output-name valid)" >&2; exit 1
    else
        EVAL_NOTE="no validation ($VALID missing: prepare_sft_data.py --output-name valid)"
    fi ;;
*)
    echo "SFT_EVAL must be auto, 1 or 0, got $SFT_EVAL" >&2; exit 1 ;;
esac

PRETRAIN_ITER=$(tracked_iter "$PRETRAIN_CKPT")
if [ -z "$PRETRAIN_ITER" ] && [ "${DRY_RUN:-0}" != 1 ]; then
    echo "no checkpoint in $PRETRAIN_CKPT (latest_checkpointed_iteration.txt missing)" >&2; exit 1
fi
check_pretrain_finished "$PRETRAIN_ITER"   # common.sh: not a save on the way to the run's end
SFT_ITER=$(tracked_iter "$SFT_DIR/ckpt")
if [ -n "$SFT_ITER" ] && [ "$SFT_ITER" -ge "$TRAIN_ITERS" ]; then
    echo "$SFT_NAME already reached iteration $TRAIN_ITERS; nothing to do"; exit 0
fi

ARGS=(
    "${MODEL_ARGS[@]}"
    ${SFT_HEAD_ARGS[@]+"${SFT_HEAD_ARGS[@]}"}
    "${OPT_ARGS[@]}"
    # schedule: 3% warmup, cosine to 0 over the whole run
    --train-iters "$TRAIN_ITERS"
    --lr "$SFT_LR"
    --min-lr 0
    --lr-decay-style cosine
    --lr-warmup-fraction 0.03
    # data: one example per row, packed online into <= SFT_TOKENS_PER_RANK tokens per rank and microbatch
    --use-varlen-dataset
    --train-data-path "$DATA"
    --tokenizer-type SFTTokenizer
    --tokenizer-model "$TOKENIZER"
    --sft-tokenizer-prompt-format identity
    --max-seqlen-per-dp-cp-rank "$SFT_TOKENS_PER_RANK"
    --calculate-per-token-loss
    --no-create-attention-mask-in-dataloader
    --num-workers 4
    --seed "$SEED"
    "${EVAL_ARGS[@]}"
    # start from the pretraining checkpoint (weights only) unless the SFT directory has one to resume
    --pretrained-checkpoint "$PRETRAIN_CKPT"
    --load "$SFT_DIR/ckpt"
    --save "$SFT_DIR/ckpt"
    --save-interval "$EPOCH_ITERS"
    --ckpt-format torch_dist
    --log-interval 10
    --log-throughput
    --tensorboard-dir "$SFT_DIR/tb"
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
    --micro-batch-size 1
    --global-batch-size "$GBS"
    --bf16
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --cross-entropy-loss-fusion
    --manual-gc
    --manual-gc-interval 100
    --distributed-timeout-minutes 30
)
if [ -n "${WANDB_PROJECT:-}" ]; then
    ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-exp-name "$SFT_NAME" --wandb-save-dir "$SFT_DIR/wandb")
fi
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})

if [ "$NODE_RANK" = 0 ]; then
    echo "== SFT $SFT_NAME"
    echo "   from $PRETRAIN_CKPT (iteration ${PRETRAIN_ITER:-?}), LM head: ${SFT_HEAD_ARGS[*]:-plain}"
    echo "   data $DATA: $EXAMPLES examples x $EPOCHS epochs; global batch $GBS examples" \
         "(pipeline ${EX_PER_STEP} per step), $TRAIN_ITERS iterations, checkpoint every $EPOCH_ITERS"
    echo "   optimizer $SFT_OPT, LR $SFT_LR (pretraining $LR / 3), 3% warmup, cosine to 0; $NNODES x $GPUS_PER_NODE GPUs" \
         "${SFT_ITER:+(resuming at $SFT_ITER)}"
    echo "   packing <= $SFT_TOKENS_PER_RANK tokens per GPU and microbatch; $EVAL_NOTE"
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' pretrain_gpt.py "${ARGS[@]}"; echo
    exit 0
fi

mkdir -p "$SFT_DIR/ckpt" "$SFT_DIR/tb"
if [ "$NODE_RANK" = 0 ]; then
    printf '%q ' pretrain_gpt.py "${ARGS[@]}" > "$SFT_DIR/cmd_$(date +%Y%m%d-%H%M%S).sh"
    cp "$STATS" "$SFT_DIR/sft_data_stats.json"
    if [ -n "${EVAL_GBS:-}" ]; then cp "$VALID_STATS" "$SFT_DIR/sft_valid_stats.json"; fi
fi
# VarlenDataset loads the parquet through `datasets`, which converts it to an Arrow cache once and memory-maps
# it; keep that cache on shared storage and never reach for the Hub (the cluster cannot).
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$OUT_ROOT/hf_datasets_cache}
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

exec python -m torch.distributed.run \
    --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$GPUS_PER_NODE" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    pretrain_gpt.py "${ARGS[@]}"
