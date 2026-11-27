#!/bin/bash
# Run a command of the vLLM eval environment (ladder/env/setup_vllm_env.sh) without what the Megatron
# environment leaves in the shell. Jobs are submitted with the Megatron venv active and sbatch exports that
# shell, so PYTHONPATH and the package directories in LD_LIBRARY_PATH (an NGC image puts its torch/lib there)
# would reach the vLLM environment, whose own torch must not load the image's libraries:
#   bash ladder/sft/eval/run_isolated.sh <vllm env>/bin/python <script> [args]
# Removed: PYTHONPATH, PYTHONHOME, VIRTUAL_ENV, the TRITON_* tool paths of ladder/env/cuda_env.sh (they point the
# image's triton at the CUDA toolkit; the vLLM environment's triton has its own), and every LD_LIBRARY_PATH entry
# inside a dist-packages / site-packages directory. Driver and CUDA compat entries stay.
set -euo pipefail

LD=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
     | { grep -v -e '/dist-packages' -e '/site-packages' -e '^$' || true; } | paste -sd: -)
UNSET=(-u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV
       -u TRITON_PTXAS_PATH -u TRITON_CUOBJDUMP_PATH -u TRITON_NVDISASM_PATH -u TRITON_LIBDEVICE_PATH)
if [ -n "$LD" ]; then
    exec env "${UNSET[@]}" LD_LIBRARY_PATH="$LD" "$@"
fi
exec env "${UNSET[@]}" -u LD_LIBRARY_PATH "$@"
