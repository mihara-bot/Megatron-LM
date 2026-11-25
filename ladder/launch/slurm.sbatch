#!/bin/bash
# SLURM entry point (Polaris): one task per node, each runs ladder/train.sh with its node rank.
#
# Submit from the repo root (on shared storage) with the uv environment activated; sbatch exports
# the current environment, so CONFIG / OPT / LR / MODE / TPP / STOP_TPP ... are passed as variables:
#   mkdir -p logs
#   CONFIG=llama3_r1 OPT=muon LR=1e-3 sbatch --nodes=2 ladder/launch/slurm.sbatch
#   CONFIG=llama3_r1 LR=1e-3 MODE=branch TPP=20 sbatch --nodes=1 ladder/launch/slurm.sbatch
# Defaults below are this cluster's paths; override any of them the same way.
#
# QoS (docs/polaris user guide, section 2):
#   standard (default, no --qos): your own standard QoS; at most 32 GPUs (4 nodes) running at once
#                                 across all your standard jobs; not preempted.
#   idle (--qos=polaris_idle)   : no per-user GPU cap, starts only on free nodes, can be stopped and
#                                 requeued at any time. Safe here: a requeued job reruns this script and
#                                 train.sh resumes from the latest checkpoint (losing < SAVE_INTERVAL steps).
# Each job may run 72 h in total; requeues do not reset that.
#SBATCH --job-name=ladder
#SBATCH --account=polaris
#SBATCH --partition=polaris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
# polaris caps a job at 128 of each node's 192 CPUs (MaxCPUsPerNode=128), so no --exclusive:
# it would claim all 192 CPUs and the submission is rejected ("node configuration is not available").
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Each family has its own tokenizer and tokenized data; the family is the config's prefix (qwen3_r1 -> qwen3).
case "${CONFIG:-}" in
    qwen3_*) FAMILY=qwen3;  DEFAULT_TOKENIZER=/fast/users/zhuangxinlin/tokenizers/qwen3 ;;
    *)       FAMILY=llama3; DEFAULT_TOKENIZER=/fast/users/zhuangxinlin/tokenizers/llama-3 ;;
esac
export TOKENIZER=${TOKENIZER:-$DEFAULT_TOKENIZER}
export DATA_ROOT=${DATA_ROOT:-/fast/users/zhuangxinlin/megatron_data/$FAMILY/climbmix_v1_m1}
export TRAIN_DATA=${TRAIN_DATA:-$DATA_ROOT/train_merged}
export VALID_DATA=${VALID_DATA:-$DATA_ROOT/val_all/val}
export OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# ht-run (Polaris launcher) binds each task to the CPUs and GPUs this job was given; keep it after srun.
srun --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 ht-run \
    bash -c 'NODE_RANK=$SLURM_NODEID exec bash ladder/train.sh'
