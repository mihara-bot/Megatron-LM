#!/bin/bash
# SLURM entry point: one task per node, each runs ladder/train.sh with its node rank.
#
# Submit from the repo root (on shared storage) with the uv environment activated; sbatch exports
# the current environment, so CONFIG / OPT / LR / MODE / TPP / STOP_TPP ... are passed as variables:
#   mkdir -p logs
#   CONFIG=llama3_r1 OPT=muon LR=1e-3 sbatch --partition=polaris --nodes=2 ladder/launch/slurm.sbatch
#   CONFIG=llama3_r1 LR=1e-3 MODE=branch TPP=20 sbatch --partition=polaris --nodes=2 ladder/launch/slurm.sbatch
# Defaults below are this cluster's paths; override any of them the same way.
#SBATCH --job-name=ladder
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export TOKENIZER=${TOKENIZER:-/fast/users/zhuangxinlin/tokenizers/llama-3}
export DATA_ROOT=${DATA_ROOT:-/fast/users/zhuangxinlin/megatron_data/llama3/climbmix_v1_m1}
export TRAIN_DATA=${TRAIN_DATA:-$DATA_ROOT/train_merged}
export VALID_DATA=${VALID_DATA:-$DATA_ROOT/val_all/val}
export OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

srun --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash -c 'NODE_RANK=$SLURM_NODEID exec bash ladder/train.sh'
