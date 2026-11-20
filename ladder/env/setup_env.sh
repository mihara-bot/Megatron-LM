#!/bin/bash
# Install Megatron into an already-activated uv environment, on top of a pre-built NGC PyTorch image.
#
# The image already provides CUDA, NCCL, torch, TransformerEngine, apex and triton, built by NVIDIA.
# We keep them: the uv environment must see the image's packages (create it with
# `uv venv --system-site-packages`), and Megatron and the few missing packages are installed with
# --no-deps so nothing replaces the image's torch / TE. (`uv sync` would instead install the
# torch / TE versions pinned in uv.lock.)
#
# Usage: activate the uv environment first, then run this script once (from any directory):
#   source <VENV>/bin/activate
#   bash ladder/env/setup_env.sh
# Keep the repo and the environment on a path every node can read: the editable install points at
# this repo. In every job, activate the same environment before launching.

set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "No active environment: run 'source <VENV>/bin/activate' first." >&2
    exit 1
fi
cd "$(dirname "$0")/../.."   # repo root, so `-e .` and the check script resolve

EO_REF=${EO_REF:-v0.3.0}   # emerging_optimizers version pinned in pyproject.toml

# Megatron itself, editable; builds the C++ dataset helper (megatron/core/datasets/helpers_cpp).
uv pip install --no-deps --no-build-isolation -e .

# Muon and the other emerging optimizers (needs torch and absl-py, both already in the image).
uv pip install --no-deps "emerging_optimizers @ git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@${EO_REF}"

# Missing pieces of the `training` extra (the rest of it is in the image already).
uv pip install sentencepiece tiktoken omegaconf flask-restful

python ladder/env/check_env.py
