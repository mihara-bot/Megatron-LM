#!/bin/bash
# Set up the Megatron environment on top of a pre-built NGC PyTorch image.
#
# The image already provides CUDA, NCCL, torch, TransformerEngine, apex and triton, built by NVIDIA.
# We keep them: the venv sees the image's packages (--system-site-packages), and Megatron and the
# few missing packages are installed with --no-deps so nothing replaces the image's torch / TE.
# (`uv sync` would instead install the torch / TE versions pinned in uv.lock.)
#
# Run once, from the repo root, on a path every node can read (the editable install points here):
#   bash ladder/env/setup_env.sh            # creates .venv in the repo root
#   VENV=/shared/envs/megatron bash ladder/env/setup_env.sh
# Then in every job:  source <VENV>/bin/activate


# Megatron itself, editable; builds the C++ dataset helper (megatron/core/datasets/helpers_cpp).
uv pip install --no-deps --no-build-isolation -e .

# Muon and the other emerging optimizers (needs torch and absl-py, both already in the image).
uv pip install --no-deps "emerging_optimizers @ git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@${EO_REF}"

# Missing pieces of the `training` extra (the rest of it is in the image already).
uv pip install sentencepiece tiktoken omegaconf flask-restful

python ladder/env/check_env.py
