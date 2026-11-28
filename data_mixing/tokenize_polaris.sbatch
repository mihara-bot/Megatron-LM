#!/bin/bash
# Tokenize the copied text dataset on Polaris (CPU only, 128 CPUs): data_mixing/tokenize_polaris.sh <STEP>.
#
# Submit from the Megatron-LM repo root with the Megatron environment activated (sbatch exports it);
# one job per tokenizer family, they can run at the same time:
#   mkdir -p /fast/users/$USER/runs
#   FAMILY=llama3 sbatch --job-name=tok-llama3 data_mixing/tokenize_polaris.sbatch
#   FAMILY=qwen3  sbatch --job-name=tok-qwen3  data_mixing/tokenize_polaris.sbatch
#   FAMILY=deepseek TOK=/path/to/base/tokenizer sbatch --job-name=tok-deepseek data_mixing/tokenize_polaris.sbatch
# STEP (default all = verify + tokenize + merge + inspect + report) selects another step, e.g. STEP=report.
# tokenize is resumable: submitting the same job again skips the finished pieces.
# Log: /fast/users/<user>/runs/<job name>-<jobid>.out (scontrol show job <jobid> | grep StdOut).
#SBATCH --job-name=dm-tokenize
#SBATCH --account=polaris
#SBATCH --partition=polaris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=/fast/users/%u/runs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
echo "[dm-tokenize] $(date) host $(hostname) dir $PWD python $(command -v python) FAMILY=${FAMILY:-llama3} STEP=${STEP:-all}"
srun --ntasks-per-node=1 --kill-on-bad-exit=1 ht-run bash -e data_mixing/tokenize_polaris.sh "${STEP:-all}"
