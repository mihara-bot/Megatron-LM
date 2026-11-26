#!/bin/bash
# torchrun-cluster entry point: the platform starts this same command on every node.
#
#   CONFIG=llama3_r1 LR=1e-3 VENV=/path/to/venv bash ladder/launch/cloud.sh
#
# It maps the platform's per-node variables to the ones ladder/train.sh reads. Assumed convention
# (PyTorchJob style): MASTER_ADDR, MASTER_PORT, WORLD_SIZE = number of nodes, RANK = node index.
# If the platform uses other names, or WORLD_SIZE counts GPUs, set NNODES / NODE_RANK explicitly.
# Defaults below are this cluster's paths; override any of them with environment variables.
set -euo pipefail

# Each family has its own tokenizer and tokenized data; the family is the config's prefix (qwen3_r1 -> qwen3).
case "${CONFIG:-}" in
    qwen3_*) FAMILY=qwen3;  DEFAULT_TOKENIZER=/mnt/users/zhuangxinlin/ckpt/Qwen3-0.6B-Base ;;
    *)       FAMILY=llama3; DEFAULT_TOKENIZER=/mnt/users/zhuangxinlin/ckpt/Llama3.1-8B ;;
esac
export TOKENIZER=${TOKENIZER:-$DEFAULT_TOKENIZER}
export DATA_ROOT=${DATA_ROOT:-/mnt/oss-zhuangxinlin/megatron_data/$FAMILY/climbmix_v1_m1}
export TRAIN_DATA=${TRAIN_DATA:-$DATA_ROOT/train_merged}
export VALID_DATA=${VALID_DATA:-$DATA_ROOT/val_all/val}
export OUT_ROOT=${OUT_ROOT:-/mnt/oss-zhuangxinlin/ladder}

export NNODES=${NNODES:-${WORLD_SIZE:-1}}
export NODE_RANK=${NODE_RANK:-${RANK:-0}}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
unset WORLD_SIZE RANK LOCAL_RANK   # torchrun sets these per GPU worker
echo "node $NODE_RANK of $NNODES, master $MASTER_ADDR:$MASTER_PORT"

# ENTRY: the per-node script (default ladder/train.sh; ladder/sft/train_sft.sh for SFT).
exec bash "${ENTRY:-$(dirname "${BASH_SOURCE[0]}")/../train.sh}"
