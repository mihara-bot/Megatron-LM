#!/bin/bash
# LR sweep of llama3_r1 + Muon on SLURM.
#
# All runs of a rung share one dataset index in $OUT_ROOT/data_cache, and Megatron does not lock it
# across jobs: several jobs that start before it exists all build and np.save the same files at once.
# So submit one LR first, wait until its log shows "after dataloaders are built" (index on disk),
# then submit the rest. The wait can take hours (queue + index build): run this in tmux, or
#   nohup bash ladder/run_llama_r1_lr_sweep.sh > logs/sweep_r1.out 2>&1 &
# Works from any directory. Before rerunning, scancel the jobs of an earlier attempt (squeue -u $USER):
# two jobs of the same LR would write the same checkpoint directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

export CONFIG=llama3_r1 OPT=muon
export EXCLUDE=${EXCLUDE:-htv02}   # keep the jobs off these nodes (sbatch --exclude)
# Measured on 8 H200: ~0.91 s/iter, so the stable run (35000 iters) takes ~9 h and the TPP50 branch
# (3500 iters) ~1 h, plus validation. A 72 h limit only starts on a node free for 72 h (no backfill,
# no upcoming maintenance reservation), so ask for what the jobs need.
export STABLE_TIME=${STABLE_TIME:-14:00:00} BRANCH_TIME=${BRANCH_TIME:-03:00:00}
FIRST_LR=1e-3
REST_LRS="5e-4 7e-4 1.4e-3 2e-3"

DRY_RUN=1 bash ladder/sweep.sh                    # the jobs of the full grid, nothing submitted

out=$(LRS=$FIRST_LR bash ladder/sweep.sh)
echo "$out"
jid=$(awk '/stable job/ {print $NF; exit}' <<< "$out")
log=logs/${CONFIG}-${OPT}-lr${FIRST_LR}-main-${jid}.out   # slurm.sbatch: --output=logs/%x-%j.out

echo "waiting for '$log' to show 'after dataloaders are built' ..."
until grep -q "after dataloaders are built" "$log" 2>/dev/null; do
    if [ -z "$(squeue -h -j "$jid" -o %T 2>/dev/null)" ]; then
        grep -q "after dataloaders are built" "$log" 2>/dev/null && break
        echo "job $jid left the queue before the dataloaders were built; see $log (sacct -j $jid)" >&2
        exit 1
    fi
    sleep 60
done
echo "dataset index is built; submitting the other LRs"

LRS=$REST_LRS bash ladder/sweep.sh
