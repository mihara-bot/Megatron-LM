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
# transformers and tokenizers are pinned to the Megatron environment's exact versions (read from the active venv),
# so SFT data preparation, the HF export and the eval all tokenize and read configs with the same libraries; the
# Megatron environment itself is not touched. ladder/export/export_hf.py records its versions in export_meta.json
# and eval_vllm.py refuses a mismatch.
#
# Run once from the Megatron-LM root, where ladder/env/setup_env.sh ran (it needs uv and the package index; a login
# node is fine, no GPU needed), with the Megatron venv active (the pins and the base interpreter come from it):
#   source <MEGATRON_VENV>/bin/activate
#   UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
#     VLLM_VENV=/fast/users/zhuangxinlin/envs/vllm bash ladder/env/setup_vllm_env.sh
# then the GPU check, one GPU, launched like the eval jobs (it builds a tiny random Llama with the ladder tokenizer
# and generates with vLLM; the log ends in "vLLM env OK"):
#   sbatch -J vllm-check -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
#     --mem=128G -t 00:30:00 -o logs/%x-%j.out --wrap "srun --ntasks=1 ht-run bash ladder/sft/eval/run_isolated.sh \
#     $VLLM_VENV/bin/python ladder/env/check_vllm_env.py --gpu --tokenizer /fast/users/zhuangxinlin/tokenizers/llama-3"
# lm-eval for the pretraining eval (ladder/eval/lm_eval.sh) goes into this venv too, from the lm-evaluation-harness
# clone next to Megatron-LM, editable, constrained to the lock so transformers / tokenizers / torch stay where they are:
#   cd ../lm-evaluation-harness && $VLLM_VENV/bin/python -m pip install -e . -c $VLLM_VENV/requirements.lock
#   bash ladder/sft/eval/run_isolated.sh $VLLM_VENV/bin/python ladder/env/check_vllm_env.py --lm-eval
#
# Variables:
#   VLLM_VENV        target directory on storage every node reads (required; must not exist)
#   VLLM_VERSION     default 0.28.0: torch 2.13 with CUDA 13.0 wheels, so the NVIDIA driver must be >= 580
#                    (nvidia-smi). It must accept the pinned transformers (its PyPI requires_dist; uv refuses
#                    otherwise). For transformers 5.9.0 (the NGC 26.11 Megatron environment): vLLM 0.19.1 - 0.28.0;
#                    0.29+ require transformers >= 5.10.4; 0.11 and older require transformers < 5.
#   TRANSFORMERS     transformers requirement, default "transformers==<the Megatron venv's version>"
#   TOKENIZERS       tokenizers requirement, default "tokenizers==<the Megatron venv's version>"
#   BASE_PYTHON      interpreter the venv is built on; default the active venv's base interpreter (with the
#                    Megatron venv active: the image's python, present in every job)
#   UV_CACHE_DIR     default /fast/users/zhuangxinlin/.cache/uv (torch + vLLM are ~10 GB of wheels)
# The resolved versions go to <VLLM_VENV>/requirements.lock; eval summaries also record vllm_version.
set -euo pipefail

: "${VLLM_VENV:?set VLLM_VENV, e.g. /fast/users/zhuangxinlin/envs/vllm}"
case "$VLLM_VENV" in /*) ;; *) VLLM_VENV=$PWD/$VLLM_VENV ;; esac   # uv runs from another directory below
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
if [ -e "$VLLM_VENV" ]; then echo "$VLLM_VENV exists; remove it first or pick another VLLM_VENV" >&2; exit 1; fi
if ! command -v uv >/dev/null; then echo "uv not found (ladder/env/setup_env.sh needs it too)" >&2; exit 1; fi

VLLM_VERSION=${VLLM_VERSION:-0.28.0}
# transformers / tokenizers pins: the Megatron environment's versions, read from the active venv before it is unset.
for pkg in transformers tokenizers; do
    var=$(echo "$pkg" | tr '[:lower:]' '[:upper:]')
    if [ -n "${!var:-}" ]; then continue; fi
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        echo "activate the Megatron venv first: $pkg is pinned to its version (or set $var)" >&2; exit 1
    fi
    v=$(python3 -c 'import sys; from importlib.metadata import version; print(version(sys.argv[1]))' "$pkg" \
        2>/dev/null || true)
    if [ -z "$v" ]; then echo "$pkg is not installed in the active environment $VIRTUAL_ENV; set $var" >&2; exit 1; fi
    printf -v "$var" '%s' "$pkg==$v"
done
unset pkg var v
BASE_PYTHON=${BASE_PYTHON:-$(python3 -c '
import os, sys
p = os.path.join(sys.base_prefix, "bin", "python%d.%d" % sys.version_info[:2])
print(p if os.path.exists(p) else sys.executable)')}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/fast/users/zhuangxinlin/.cache/uv}

if command -v nvidia-smi >/dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
    echo "NVIDIA driver here: ${DRIVER:-?} (vLLM 0.20+ wheels use CUDA 13.0: driver >= 580)"
    if [[ "${DRIVER%%.*}" =~ ^[0-9]+$ ]] && [ "${DRIVER%%.*}" -lt 580 ]; then
        echo "warning: driver $DRIVER < 580 cannot run the CUDA 13.0 wheels of vLLM 0.20+" >&2
    fi
fi
echo "== vLLM $VLLM_VERSION, $TRANSFORMERS, $TOKENIZERS -> $VLLM_VENV (base interpreter $BASE_PYTHON)"

# Nothing of the Megatron environment or the image may steer the install: NGC images pin their own torch
# through PIP_CONSTRAINT, and an active VIRTUAL_ENV is where `uv pip` would install otherwise.
unset PIP_CONSTRAINT UV_CONSTRAINT PYTHONPATH VIRTUAL_ENV
# uv also reads [tool.uv] from the pyproject.toml of the directory it runs in, and Megatron's overrides torch,
# torchvision and triton away (`sys_platform == 'never'`: in the Megatron env they come from the image). Run uv
# from an empty directory instead; user-level config (~/.config/uv/uv.toml, e.g. a mirror) and UV_* still apply.
NEUTRAL=$(mktemp -d)
trap 'rm -rf "$NEUTRAL"' EXIT
uvn() { (cd "$NEUTRAL" && uv "$@"); }

# No --system-site-packages: nothing from the image. --seed puts pip into the venv, so `<VLLM_VENV>/bin/python -m pip`
# installs lm-eval there (ladder/EVAL.md); without it, a bare `pip` could resolve to the Megatron venv's.
uvn venv --seed --python "$BASE_PYTHON" "$VLLM_VENV"
PY=$VLLM_VENV/bin/python
if ! uvn pip install --python "$PY" "vllm==$VLLM_VERSION" "$TRANSFORMERS" "$TOKENIZERS" pyyaml pyarrow sympy \
        "antlr4-python3-runtime==4.11.*"; then        # sympy's LaTeX parser is generated with ANTLR 4.11
    echo "install failed. A conflict on transformers / tokenizers means vLLM $VLLM_VERSION does not accept" \
         "$TRANSFORMERS $TOKENIZERS: pick another VLLM_VERSION (see the top of this script). Remove $VLLM_VENV" \
         "before retrying." >&2
    exit 1
fi
uvn pip freeze --python "$PY" > "$VLLM_VENV/requirements.lock"
for pkg in torch triton vllm; do
    if ! grep -qi "^${pkg}==" "$VLLM_VENV/requirements.lock"; then
        echo "$pkg is missing from $VLLM_VENV (a uv config overriding it away?): see requirements.lock" >&2; exit 1
    fi
done
for req in "$TRANSFORMERS" "$TOKENIZERS"; do
    if [[ "$req" == *==* ]] && ! grep -qiFx -- "$req" "$VLLM_VENV/requirements.lock"; then
        echo "$VLLM_VENV does not have $req: see requirements.lock" >&2; exit 1
    fi
done

bash ladder/sft/eval/run_isolated.sh "$PY" ladder/env/check_vllm_env.py
echo "VLLM_PYTHON=$PY   (GPU check: see the top of ladder/env/setup_vllm_env.sh)"
