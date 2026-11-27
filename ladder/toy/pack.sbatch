#!/bin/bash
# SLURM job (Polaris) for ladder/toy/pack.sh: one node, its 8 GPUs shared by the chains of one spec file.
# ladder/toy/run_toy.sh writes the spec files and submits these; by hand, from the repo root with the environment
# active (sbatch exports it, TOY_ROOT included):
#   mkdir -p logs
#   TOY_ROOT=/fast/users/zhuangxinlin/toy_margin sbatch --job-name=toy-grid-0 ladder/toy/pack.sbatch <spec file>
# Add --qos=polaris_idle for idle nodes: a requeued pack reruns every chain, and each chain skips what is done.
#SBATCH --job-name=toy
#SBATCH --account=polaris
#SBATCH --partition=polaris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SPEC=${1:?usage: sbatch ladder/toy/pack.sbatch <spec file>}
export GPUS=${GPUS:-8}
srun --ntasks=1 --kill-on-bad-exit=1 ht-run bash ladder/toy/pack.sh "$SPEC"
