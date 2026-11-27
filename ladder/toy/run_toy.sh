#!/bin/bash
# The margin toy on SLURM (ladder/toy/TOY.md). Run from the Megatron-LM root with the environment active, after
#   TOY_ROOT=... bash ladder/toy/prepare.sh
# Writes the chains of a phase to spec files ($TOY_ROOT/specs/<phase>_<k>.txt, SLOTS chains each) and submits one
# ladder/toy/pack.sbatch job (1 node, 8 GPUs, RUNS_PER_GPU chains per GPU) per spec file.
#
#   PHASE=lr bash ladder/toy/run_toy.sh             plain head at every LR of the config's LR_GRID, seed 1:
#                                                   pretrain + probe of the final checkpoint
#   python3 ladder/toy/summarize.py lr --root $TOY_ROOT          -> the LR with the lowest walk KL
#   PHASE=grid LR=<that LR> bash ladder/toy/run_toy.sh   the head / margin variants below x SEEDS, each with SFT
#   python3 ladder/toy/summarize.py table --root $TOY_ROOT
#
# Environment: TOY_ROOT (default /fast/users/zhuangxinlin/toy_margin), PHASE (lr | grid), LR (grid: required),
#   SEEDS (grid, default "1 2 3"), CONFIG (default toy_l2), RUNS_PER_GPU (default 2), TIME (sbatch, 12:00:00),
#   QOS (e.g. polaris_idle), LOCAL=1 (run the packs one after another on this node with pack.sh instead of
#   sbatch), DRY_RUN=1 (print the specs and commands, write and submit nothing).
# Every chain skips what is already done, so resubmitting a phase only runs what is missing; but never resubmit
# while its jobs are still queued or running (two chains would write the same run directory).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
export TOY_ROOT=${TOY_ROOT:-/fast/users/zhuangxinlin/toy_margin}
PHASE=${PHASE:?set PHASE=lr or PHASE=grid}
CONFIG=${CONFIG:-toy_l2}
RUNS_PER_GPU=${RUNS_PER_GPU:-2}
SLOTS=$((8 * RUNS_PER_GPU))
LR_GRID=$(bash -c "source ladder/configs/$CONFIG.sh; echo \$LR_GRID")

# ---- the grid: LM head x margin (d is a cosine margin; logit margin m = d s = 60 d for hfn, d ||h|| for nfn/sfn) --
VARIANTS=(
    "HEAD=plain"
    "HEAD=hfn MARGIN_D=0"                                   # feature-norm-only control of hfn
    "HEAD=hfn MARGIN_D=0.005"                               # m = 0.3
    "HEAD=hfn MARGIN_D=0.02"                                # m = 1.2, the real runs' d
    "HEAD=hfn MARGIN_D=0.05"                                # m = 3
    "HEAD=hfn MARGIN_D=0.005 MARGIN_REVERSE=1"
    "HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=1"
    "HEAD=hfn MARGIN_D=0.05 MARGIN_REVERSE=1"
    "HEAD=sfn MARGIN_D=0"                                   # control of sfn
    "HEAD=sfn MARGIN_D=0.02"
    "HEAD=sfn MARGIN_D=0.05"
    "HEAD=sfn MARGIN_D=0.02 MARGIN_REVERSE=1"
    "HEAD=sfn MARGIN_D=0.05 MARGIN_REVERSE=1"
    "HEAD=nfn MARGIN_D=0.02"                                # ||h|| free: watch it under reverse
    "HEAD=nfn MARGIN_D=0.02 MARGIN_REVERSE=1"
)

CHAINS=()
case "$PHASE" in
lr)
    for lr in $LR_GRID; do
        CHAINS+=("CONFIG=$CONFIG HEAD=plain LR=$lr SEED=1 STAGES=pretrain,export,probe PRE_ITERS=latest")
    done
    ;;
grid)
    : "${LR:?PHASE=grid needs LR (ladder/toy/summarize.py lr picks it)}"
    for seed in ${SEEDS:-1 2 3}; do
        for v in "${VARIANTS[@]}"; do
            v=${v%%#*}
            CHAINS+=("CONFIG=$CONFIG $(echo $v) LR=$LR SEED=$seed")
        done
    done
    ;;
*)
    echo "PHASE must be lr or grid, got $PHASE" >&2; exit 1 ;;
esac

SPEC_DIR=$TOY_ROOT/specs
if [ "${DRY_RUN:-0}" = 1 ]; then SPEC_DIR=$(mktemp -d); fi
mkdir -p "$SPEC_DIR" logs
STAMP=$(date +%Y%m%d-%H%M%S)
N_PACKS=$(( (${#CHAINS[@]} + SLOTS - 1) / SLOTS ))
echo "== toy $PHASE: ${#CHAINS[@]} chains in $N_PACKS pack(s) of up to $SLOTS (8 GPUs x $RUNS_PER_GPU), TOY_ROOT $TOY_ROOT"
for ((p = 0; p < N_PACKS; p++)); do
    spec=$SPEC_DIR/${PHASE}_${STAMP}_$p.txt
    printf '%s\n' "${CHAINS[@]:$((p * SLOTS)):$SLOTS}" > "$spec"
    echo "-- $spec"
    sed 's/^/   /' "$spec"
    if [ "${LOCAL:-0}" = 1 ]; then
        cmd=(env RUNS_PER_GPU="$RUNS_PER_GPU" bash ladder/toy/pack.sh "$spec")
    else
        cmd=(sbatch --parsable --job-name="toy-$PHASE-$p" --time="${TIME:-12:00:00}"
             ${QOS:+--qos=$QOS} ladder/toy/pack.sbatch "$spec")
    fi
    echo "+ ${cmd[*]}"
    if [ "${DRY_RUN:-0}" != 1 ]; then RUNS_PER_GPU=$RUNS_PER_GPU "${cmd[@]}"; fi
done
if [ "${DRY_RUN:-0}" = 1 ]; then rm -rf "$SPEC_DIR"; fi
