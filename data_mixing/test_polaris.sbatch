#!/bin/bash
# data_mixing tests on Polaris: unit tests plus the end-to-end test, which also tokenizes synthetic data
# with ladder/data (1 GPU slot; the tests themselves only use the CPU).
#
# Submit from the Megatron-LM repo root with the Megatron environment activated (sbatch exports it):
#   mkdir -p /fast/users/$USER/runs
#   FAMILY=llama3 sbatch data_mixing/test_polaris.sbatch
# Log: /fast/users/<user>/runs/dm-test-<jobid>.out; while the job runs, this prints the path:
#   scontrol show job <jobid> | grep StdOut
# Success = the log ends with "all data_mixing tests passed".
#SBATCH --job-name=dm-test
#SBATCH --account=polaris
#SBATCH --partition=polaris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/fast/users/%u/runs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1   # progress shows up in the log as it happens
echo "[dm-test] $(date) host $(hostname) dir $PWD python $(command -v python) FAMILY=${FAMILY:-llama3}"
srun --ntasks-per-node=1 --kill-on-bad-exit=1 ht-run bash -e data_mixing/tokenize_polaris.sh test
