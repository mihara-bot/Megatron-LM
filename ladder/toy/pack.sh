#!/bin/bash
# Run margin-toy chains (ladder/toy/chain.sh) in parallel on this node's GPUs (ladder/toy/TOY.md).
#   TOY_ROOT=... bash ladder/toy/pack.sh SPEC
#
# SPEC: one chain per line as KEY=VALUE assignments for chain.sh (no spaces inside a value; lists comma-separated),
# blank lines and '#' comments ignored, e.g.
#   HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=1 LR=4e-3 SEED=2
#   HEAD=plain LR=8e-3 SEED=1 STAGES=pretrain,export,probe PRE_ITERS=latest
# SLOTS = GPUS x RUNS_PER_GPU; line i runs in slot i mod SLOTS, a slot runs its lines one after another. Slot k
# uses GPU k mod GPUS and MASTER_PORT = BASE_PORT + k. One log per line in LOG_DIR. Exits 1 if any chain failed.
#
# Environment: TOY_ROOT (required), GPUS (default: GPUs nvidia-smi lists), RUNS_PER_GPU (default 2), BASE_PORT,
#   LOG_DIR (default $TOY_ROOT/logs/pack_<date>_<job>), VENV, DRY_RUN=1 (print the slot plan only)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
SPEC=${1:?usage: pack.sh SPEC}
: "${TOY_ROOT:?set TOY_ROOT}"
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
if [ "${GPUS:-0}" -lt 1 ]; then echo "no GPUs found (set GPUS)" >&2; exit 1; fi
RUNS_PER_GPU=${RUNS_PER_GPU:-2}
SLOTS=$((GPUS * RUNS_PER_GPU))
JOB=${SLURM_JOB_ID:-$$}
BASE_PORT=${BASE_PORT:-$((20000 + JOB % 20000))}
LOG_DIR=${LOG_DIR:-$TOY_ROOT/logs/pack_$(date +%Y%m%d-%H%M%S)_$JOB}

LINES=()
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%%#*}
    read -r -a toks <<< "$line"
    if [ ${#toks[@]} -eq 0 ]; then continue; fi
    for tok in "${toks[@]}"; do
        if ! [[ "$tok" =~ ^[A-Z_][A-Z0-9_]*=[^[:space:]]*$ ]]; then
            echo "$SPEC: '$tok' is not a KEY=VALUE assignment (line: $line)" >&2; exit 1
        fi
    done
    LINES+=("${toks[*]}")
done < "$SPEC"
if [ ${#LINES[@]} -eq 0 ]; then echo "$SPEC has no chains" >&2; exit 1; fi
DUP=$(printf '%s\n' "${LINES[@]}" | sort | uniq -d | head -n 1)
if [ -n "$DUP" ]; then echo "$SPEC lists the same chain twice (they would share a run directory): $DUP" >&2; exit 1; fi

name_of() { echo "$(printf %03d "$1")_$(echo "${LINES[$1]}" | tr ' =' '_-' | tr -cd 'A-Za-z0-9_.,+-')"; }

echo "== pack $SPEC: ${#LINES[@]} chains, $GPUS GPUs x $RUNS_PER_GPU = $SLOTS slots, ports $BASE_PORT+, logs $LOG_DIR"
for ((i = 0; i < ${#LINES[@]}; i++)); do
    k=$((i % SLOTS))
    echo "   slot $k (GPU $((k % GPUS)), port $((BASE_PORT + k))): ${LINES[$i]}"
done
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
mkdir -p "$LOG_DIR"
cp "$SPEC" "$LOG_DIR/spec.txt"

slot_run() {   # slot_run <slot>: the slot's lines in order; returns 1 if any failed
    local k=$1 i name fail=0
    for ((i = k; i < ${#LINES[@]}; i += SLOTS)); do
        name=$(name_of "$i")
        # shellcheck disable=SC2086
        if env CUDA_VISIBLE_DEVICES=$((k % GPUS)) MASTER_PORT=$((BASE_PORT + k)) ${LINES[$i]} \
            bash ladder/toy/chain.sh > "$LOG_DIR/$name.log" 2>&1; then
            echo "done   $name"
        else
            echo "FAILED $name (log $LOG_DIR/$name.log)"
            fail=1
        fi
    done
    return $fail
}

PIDS=()
for ((k = 0; k < SLOTS && k < ${#LINES[@]}; k++)); do
    slot_run "$k" &
    PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
echo "== pack $SPEC finished, $( [ "$FAIL" = 0 ] && echo 'all chains ok' || echo 'some chains FAILED' )"
exit $FAIL
