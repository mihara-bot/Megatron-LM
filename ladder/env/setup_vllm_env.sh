#!/bin/bash
# A separate Python environment for the offline vLLM eval (ladder/sft/eval.sh with VLLM_PYTHON=<VLLM_VENV>/bin/python).
#
# Never install vLLM into the Megatron environment: vLLM pins its own torch, and installing it there replaces
# the NGC image's torch (which TransformerEngine and apex are built against) inside that venv. The environment
# built here is isolated instead: no --system-site-packages, its own torch from the vLLM wheel. It needs no
# Megatron: ladder/sft/eval/eval_vllm.py uses vllm, transformers, pyyaml, pyarrow and sympy (+ antlr4 for its
# LaTeX parser). ladder/sft/eval/run_isolated.sh keeps the Megatron shell's PYTHONPATH / LD_LIBRARY_PATH away
# from it at run time (eval.sh calls it).
#
# Run once from the Megatron-LM root, where ladder/env/setup_env.sh ran (it needs uv and the package index; a login
# node is fine, no GPU needed), preferably with the Megatron venv active so the same base interpreter is used:
#   VLLM_VENV=/fast/users/zhuangxinlin/envs/vllm bash ladder/env/setup_vllm_env.sh
# then the GPU check, one GPU, launched like the eval jobs (it builds a tiny random Llama with the ladder tokenizer
# and generates with vLLM; the log ends in "vLLM env OK"):
#   sbatch -J vllm-check -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
#     --mem=128G -t 00:30:00 -o logs/%x-%j.out --wrap "srun --ntasks=1 ht-run bash ladder/sft/eval/run_isolated.sh \
#     $VLLM_VENV/bin/python ladder/env/check_vllm_env.py --gpu --tokenizer /fast/users/zhuangxinlin/tokenizers/llama-3"
#
# Variables:
#   VLLM_VENV        target directory on storage every node reads (required; must not exist)
#   VLLM_VERSION     default 0.11.0: torch 2.8 with CUDA 12.8 wheels, so the NVIDIA driver must be >= 570
#                    (nvidia-smi); Llama 3 and Qwen3 are both supported. Older driver: 0.8.5 (torch 2.6, CUDA 12.4,
#                    driver >= 550; it already has Qwen3).
#   TRANSFORMERS     transformers requirement, default "transformers<5" (vLLM 0.11 predates transformers 5)
#   BASE_PYTHON      interpreter the venv is built on; default the active venv's base interpreter (with the
#                    Megatron venv active: the image's python, present in every job)
#   UV_CACHE_DIR     default /fast/users/zhuangxinlin/.cache/uv (torch + vLLM are ~10 GB of wheels)
# The resolved versions go to <VLLM_VENV>/requirements.lock; eval summaries also record vllm_version.
set -euo pipefail

: "${VLLM_VENV:?set VLLM_VENV, e.g. /fast/users/zhuangxinlin/envs/vllm}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [ -e "$VLLM_VENV" ]; then echo "$VLLM_VENV exists; remove it first or pick another VLLM_VENV" >&2; exit 1; fi
if ! command -v uv >/dev/null; then echo "uv not found (ladder/env/setup_env.sh needs it too)" >&2; exit 1; fi

VLLM_VERSION=${VLLM_VERSION:-0.11.0}
TRANSFORMERS=${TRANSFORMERS:-transformers<5}
BASE_PYTHON=${BASE_PYTHON:-$(python3 -c '
import os, sys
p = os.path.join(sys.base_prefix, "bin", "python%d.%d" % sys.version_info[:2])
print(p if os.path.exists(p) else sys.executable)')}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fast/users/zhuangxinlin/.cache/uv}

if command -v nvidia-smi >/dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
    echo "NVIDIA driver here: ${DRIVER:-?} (vLLM 0.9+ wheels use CUDA 12.8: driver >= 570)"
    if [[ "${DRIVER%%.*}" =~ ^[0-9]+$ ]] && [ "${DRIVER%%.*}" -lt 570 ]; then
        echo "warning: driver $DRIVER < 570 cannot run CUDA 12.8 wheels; consider VLLM_VERSION=0.8.5" >&2
    fi
fi
echo "== vLLM $VLLM_VERSION, $TRANSFORMERS -> $VLLM_VENV (base interpreter $BASE_PYTHON)"

# Nothing of the Megatron environment or the image may steer the install: NGC images pin their own torch
# through PIP_CONSTRAINT, and an active VIRTUAL_ENV is where `uv pip` would install otherwise.
unset PIP_CONSTRAINT UV_CONSTRAINT PYTHONPATH VIRTUAL_ENV

uv venv --python "$BASE_PYTHON" "$VLLM_VENV"          # no --system-site-packages: nothing from the image
PY=$VLLM_VENV/bin/python
uv pip install --python "$PY" "vllm==$VLLM_VERSION" "$TRANSFORMERS" pyyaml pyarrow sympy \
    "antlr4-python3-runtime==4.11.*"                  # sympy's LaTeX parser is generated with ANTLR 4.11
uv pip freeze --python "$PY" > "$VLLM_VENV/requirements.lock"

bash ladder/sft/eval/run_isolated.sh "$PY" ladder/env/check_vllm_env.py
echo "VLLM_PYTHON=$PY   (GPU check: see the top of ladder/env/setup_vllm_env.sh)"
