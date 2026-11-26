#!/bin/bash
# Ladder training entry point: the same script runs on every node, on both clusters.
# The launchers in ladder/launch/ only set the distributed variables below and call it.
#
# Schedule. Each rung (config) has one "stable" WSD run to TPP_MAX and decay branches at BRANCH_TPPS:
#   end iteration E(T)  = T * N_EFF tokens, rounded to a multiple of Q = lcm(5/4*SAVE_INTERVAL, EVAL_INTERVAL)
#   decay start S(T)    = 0.8 * E(T)   (a multiple of SAVE_INTERVAL, so the stable run has that checkpoint;
#                                       E(T) a multiple of EVAL_INTERVAL, so a branch validates at its end)
#   stable run          : warmup, constant LR, 1-sqrt decay over the last 20% of E(TPP_MAX)
#   branch at T         : loads the stable checkpoint at S(T), decays to 0 over [S(T), E(T)], stops at E(T)
# Every run of a rung passes --train-iters E(TPP_MAX), so Megatron builds the same dataset index
# (same epochs, same shuffle) and a branch reads exactly the batches the stable run would have read.
# Branches stop through --exit-interval instead.
#
# Required environment:
#   CONFIG        ladder/configs/<CONFIG>.sh, e.g. llama3_r1 (model shape, batch, schedule, LR)
#   TRAIN_DATA    Megatron prefix (without .bin/.idx)
#   VALID_DATA    one or more prefixes, space-separated (VALID_NAMES: optional names, same order)
#   TOKENIZER     HF tokenizer directory
#   OUT_ROOT      shared directory for runs and the dataset index cache
# Optional:
#   OPT           ladder/configs/opt/<OPT>.sh: adamw (default) | muon
#   MODE          stable (default) | branch
#   TPP           branch target (MODE=branch), one of the config's BRANCH_TPPS
#   STOP_TPP      stable run only: stop at S(STOP_TPP) instead of finishing (LR sweeps stop at 100;
#                 rerunning later without STOP_TPP continues the same run to TPP_MAX)
#   HEAD, MARGIN_D, MARGIN_REVERSE, HEAD_SCALE, SFN_WEIGHT
#                 LM head + token margin, see ladder/configs/head.sh (default: plain head)
#   LR, MBS, SEED, RUN (run directory name, default <CONFIG>_<OPT>_lr<LR><HEAD_TAG>), WANDB_PROJECT, VENV
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE   (set by the launchers)
#   DRY_RUN=1     print the plan and the command, launch nothing
#   SKIP_TOKENIZER_CHECK=1   skip ladder/data/check_tokenizer.py (tokenizer vs config vs data)
#   EXTRA_ARGS    appended to the Megatron arguments
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
# CUDA_HOME, CPATH and TRITON_*_PATH for Triton / torch.compile (the image's own ENV is not in our jobs).
source ladder/env/cuda_env.sh

: "${CONFIG:?set CONFIG, e.g. CONFIG=llama3_r1}"
source "ladder/configs/${CONFIG}.sh"
OPT=${OPT:-adamw}
source "ladder/configs/opt/${OPT}.sh"
source ladder/configs/head.sh
: "${TRAIN_DATA:?}" "${VALID_DATA:?}" "${TOKENIZER:?}" "${OUT_ROOT:?}"

MODE=${MODE:-stable}
SEED=${SEED:-1234}
RUN=${RUN:-${CONFIG}_${OPT}_lr${LR}${HEAD_TAG}}
RUN_DIR=$OUT_ROOT/runs/$RUN

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
if [ "$NODE_RANK" -ge "$NNODES" ]; then
    echo "NODE_RANK=$NODE_RANK but NNODES=$NNODES: check how the launcher maps the cluster's variables" >&2
    exit 1
fi

# ---- schedule -----------------------------------------------------------------------------------
TOK_PER_ITER=$((GBS * SEQ_LEN))
if [ $((SAVE_INTERVAL % 4)) -ne 0 ]; then echo "SAVE_INTERVAL must be a multiple of 4" >&2; exit 1; fi
gcd() { local a=$1 b=$2 t; while [ "$b" -ne 0 ]; do t=$((a % b)); a=$b; b=$t; done; echo "$a"; }
Q_A=$((SAVE_INTERVAL * 5 / 4))
Q=$((Q_A / $(gcd "$Q_A" "$EVAL_INTERVAL") * EVAL_INTERVAL))   # lcm
end_iter() {   # E(T): T * N_EFF tokens, rounded to a multiple of Q iterations
    echo $(( (($1 * N_EFF) + TOK_PER_ITER * Q / 2) / (TOK_PER_ITER * Q) * Q ))
}
decay_start() { echo $(( $(end_iter "$1") * 4 / 5 )); }
latest_iter() {   # iteration in a checkpoint directory's tracker file, 0 if none
    local f=$1/latest_checkpointed_iteration.txt
    if [ -f "$f" ]; then cat "$f"; else echo 0; fi
}

E_MAX=$(end_iter "$TPP_MAX")
S_MAX=$(decay_start "$TPP_MAX")
# The stable run keeps only the checkpoints the branches start from: every multiple of the gcd of the
# branch decay starts (all multiples of SAVE_INTERVAL). Other regular saves are resume points only and
# are deleted once the next one lands; the final save at E(TPP_MAX) is kept.
RETAIN=0
for t in $BRANCH_TPPS; do RETAIN=$(gcd "$(decay_start "$t")" "$RETAIN"); done
if [ "$RETAIN" -eq 0 ]; then RETAIN=$SAVE_INTERVAL; fi
MAIN_CKPT=$RUN_DIR/ckpt/main

print_plan() {
    # ladder/analysis/lr_fit.py keys runs by this header: the head tag keeps margin runs apart from the
    # plain-head run with the same LR (empty for the plain head, so existing logs parse as before).
    echo "== $CONFIG + $OPT${HEAD_TAG}: N_eff $N_EFF, $TOK_PER_ITER tokens/iter, warmup $WARMUP_ITERS, LR $LR"
    echo "   LM head: ${HEAD_ARGS[*]:-plain}"
    echo "   save every $SAVE_INTERVAL (stable run keeps multiples of $RETAIN), validate every $EVAL_INTERVAL, branch ends on multiples of $Q"
    printf "   %-8s %10s %10s %12s %8s\n" target decay_from end tokens TPP
    for t in $BRANCH_TPPS $TPP_MAX; do
        local e s
        e=$(end_iter "$t"); s=$(decay_start "$t")
        awk -v t="$t" -v s="$s" -v e="$e" -v k="$TOK_PER_ITER" -v n="$N_EFF" \
            'BEGIN { printf "   TPP%-5s %10d %10d %11.2fB %8.1f\n", t, s, e, e*k/1e9, e*k/n }'
    done
}

SCHED_ARGS=(
    --train-iters "$E_MAX"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-warmup-iters "$WARMUP_ITERS"
    --lr-decay-style WSD
    --lr-wsd-decay-style minus_sqrt
)
case "$MODE" in
stable)
    TAG=main
    CKPT=$MAIN_CKPT
    TARGET=$E_MAX
    SCHED_ARGS+=(--lr-decay-iters "$E_MAX" --lr-wsd-decay-iters $((E_MAX - S_MAX)))
    if [ -n "${STOP_TPP:-}" ] && [ "$STOP_TPP" != "$TPP_MAX" ]; then
        TARGET=$(decay_start "$STOP_TPP")
        SCHED_ARGS+=(--exit-interval "$TARGET")
    fi
    LOAD_ARGS=(--load "$CKPT")
    SAVE_ARGS=(--save "$CKPT" --save-interval "$SAVE_INTERVAL" --save-retain-interval "$RETAIN")
    ;;
branch)
    : "${TPP:?MODE=branch needs TPP, one of: $BRANCH_TPPS}"
    case " $BRANCH_TPPS " in *" $TPP "*) ;; *) echo "TPP=$TPP is not in BRANCH_TPPS ($BRANCH_TPPS)" >&2; exit 1 ;; esac
    TAG=tpp$TPP
    CKPT=$RUN_DIR/ckpt/$TAG
    TARGET=$(end_iter "$TPP")
    S=$(decay_start "$TPP")
    SCHED_ARGS+=(--lr-decay-iters "$TARGET" --lr-wsd-decay-iters $((TARGET - S))
                 --override-opt-param-scheduler --exit-interval "$TARGET")
    if [ "$(latest_iter "$CKPT")" -gt 0 ]; then
        LOAD_ARGS=(--load "$CKPT")                         # resume this branch
    else
        start=$MAIN_CKPT/iter_$(printf %07d "$S")
        if [ ! -d "$start" ] && [ "${DRY_RUN:-0}" != 1 ]; then
            echo "branch TPP$TPP starts from $start, which does not exist yet (stable run at $(latest_iter "$MAIN_CKPT"))" >&2
            exit 1
        fi
        LOAD_ARGS=(--load "$MAIN_CKPT" --ckpt-step "$S")   # fork from the stable run
    fi
    # Megatron needs retain % save == 0, and E(T) is generally not a multiple of SAVE_INTERVAL. Save
    # every (E-S)/k instead, the smallest k that keeps it within SAVE_INTERVAL: S = 4(E-S), so the saves
    # land on S + j(E-S)/k and the last one on E; retain E keeps only that one.
    span=$((TARGET - S)); k=$(( (span + SAVE_INTERVAL - 1) / SAVE_INTERVAL ))
    while [ $((span % k)) -ne 0 ]; do k=$((k + 1)); done
    SAVE_ARGS=(--save "$CKPT" --save-interval $((span / k)) --save-retain-interval "$TARGET")
    ;;
*)
    echo "MODE must be stable or branch, got $MODE" >&2; exit 1 ;;
esac

# Already at the target: Megatron would load it and train past it, so stop here.
if [ "$(latest_iter "$CKPT")" -ge "$TARGET" ]; then
    echo "$RUN/$TAG already reached iteration $TARGET ($CKPT); nothing to do"
    exit 0
fi

# ---- data, logging ------------------------------------------------------------------------------
read -r -a VALID_LIST <<< "$VALID_DATA"
DATA_ARGS=(
    --train-data-path "$TRAIN_DATA"
    --valid-data-path "${VALID_LIST[@]}"
    --data-cache-path "$OUT_ROOT/data_cache"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER"
    --tokenizer-hf-no-include-special-tokens
    --num-workers 4
    --no-create-attention-mask-in-dataloader
    --seed "$SEED"
)
EVAL_ARGS=(--eval-interval "$EVAL_INTERVAL" --full-validation)
if [ "${#VALID_LIST[@]}" -gt 1 ]; then
    EVAL_ARGS+=(--multiple-validation-sets)
    if [ -n "${VALID_NAMES:-}" ]; then
        read -r -a NAMES <<< "$VALID_NAMES"
        EVAL_ARGS+=(--validation-set-names "${NAMES[@]}")
    fi
fi

LOG_ARGS=(
    --log-interval 10
    --log-throughput
    --tensorboard-dir "$RUN_DIR/tb/$TAG"
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
)
if [ -n "${WANDB_PROJECT:-}" ]; then
    LOG_ARGS+=(--wandb-project "$WANDB_PROJECT" --wandb-exp-name "$RUN-$TAG" --wandb-save-dir "$RUN_DIR/wandb")
fi

SYSTEM_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --bf16
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --cross-entropy-loss-fusion
    --manual-gc
    --manual-gc-interval 100
    --ckpt-format torch_dist
    --distributed-timeout-minutes 30
)

read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
ARGS=("${MODEL_ARGS[@]}" ${HEAD_ARGS[@]+"${HEAD_ARGS[@]}"} "${OPT_ARGS[@]}" "${SCHED_ARGS[@]}" "${DATA_ARGS[@]}" "${EVAL_ARGS[@]}"
      "${LOG_ARGS[@]}" "${SYSTEM_ARGS[@]}" "${LOAD_ARGS[@]}" "${SAVE_ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"})

if [ "$NODE_RANK" = 0 ]; then
    print_plan
    echo "== run $RUN_DIR, mode $MODE ($TAG), target iteration $TARGET, $NNODES x $GPUS_PER_NODE GPUs"
fi
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' pretrain_gpt.py "${ARGS[@]}"; echo
    exit 0
fi

# Tokenizer must match the config (TOKENIZER_VOCAB) and the data (manifest written by tokenize_dataset.py).
# A Llama-3 tokenizer on Qwen3 data or the reverse trains without any error, so stop here instead.
# Node 0 only; on SLURM, srun --kill-on-bad-exit takes the other nodes down with it.
if [ "$NODE_RANK" = 0 ] && [ "${SKIP_TOKENIZER_CHECK:-0}" != 1 ]; then
    : "${TOKENIZER_VOCAB:?ladder/configs/${CONFIG}.sh must set TOKENIZER_VOCAB}"
    python ladder/data/check_tokenizer.py --tokenizer "$TOKENIZER" --expect-vocab "$TOKENIZER_VOCAB" \
        "$TRAIN_DATA" "${VALID_LIST[@]}"
fi

mkdir -p "$RUN_DIR/tb" "$RUN_DIR/ckpt" "$OUT_ROOT/data_cache"
if [ "$NODE_RANK" = 0 ]; then
    printf '%q ' pretrain_gpt.py "${ARGS[@]}" > "$RUN_DIR/cmd_${TAG}_$(date +%Y%m%d-%H%M%S).sh"
fi

exec python -m torch.distributed.run \
    --nnodes "$NNODES" --node-rank "$NODE_RANK" --nproc-per-node "$GPUS_PER_NODE" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    pretrain_gpt.py "${ARGS[@]}"
