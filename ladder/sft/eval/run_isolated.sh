#!/bin/bash
# Run a command of the vLLM eval environment (ladder/env/setup_vllm_env.sh) without what the Megatron
# environment leaves in the shell. Jobs are submitted with the Megatron venv active and sbatch exports that
# shell, so PYTHONPATH and the package directories in LD_LIBRARY_PATH (an NGC image puts its torch/lib there)
# would reach the vLLM environment, whose own torch must not load the image's libraries:
#   bash ladder/sft/eval/run_isolated.sh <vllm env>/bin/python <script> [args]
# Removed: PYTHONPATH, PYTHONHOME, VIRTUAL_ENV, the TRITON_* tool paths of ladder/env/cuda_env.sh (they point the
# image's triton at the CUDA toolkit; the vLLM environment's triton has its own), and every LD_LIBRARY_PATH entry
# inside a dist-packages / site-packages directory. Driver and CUDA compat entries stay.
#
# GPUs: the cluster launcher (ht-run) sets CUDA_VISIBLE_DEVICES to GPU UUIDs, which torch accepts but vLLM 0.11
# parses as integers for its NVML queries (platforms/interface.py device_id_to_physical_device_id: ValueError).
# UUIDs are translated to the NVML indices nvidia-smi reports, with CUDA_DEVICE_ORDER=PCI_BUS_ID so CUDA numbers
# the devices the same way; the original list stays in CUDA_VISIBLE_DEVICES_UUIDS, which
# ladder/env/check_vllm_env.py compares with the device torch opens.
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *GPU-* ]]; then
    MAP=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)
    IDX=()
    IFS=, read -r -a UUIDS <<< "$CUDA_VISIBLE_DEVICES"
    for u in "${UUIDS[@]}"; do
        u=${u// /}
        i=$(awk -F', *' -v u="$u" '$2 == u { print $1 }' <<< "$MAP")
        if [ -z "$i" ]; then
            echo "run_isolated.sh: $u (CUDA_VISIBLE_DEVICES) is not listed by nvidia-smi:" >&2; echo "$MAP" >&2; exit 1
        fi
        IDX+=("$i")
    done
    export CUDA_VISIBLE_DEVICES_UUIDS=$CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${IDX[*]}")
    export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
    unset MAP IDX UUIDS u i
fi

LD=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
     | { grep -v -e '/dist-packages' -e '/site-packages' -e '^$' || true; } | paste -sd: -)
UNSET=(-u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV
       -u TRITON_PTXAS_PATH -u TRITON_CUOBJDUMP_PATH -u TRITON_NVDISASM_PATH -u TRITON_LIBDEVICE_PATH)
if [ -n "$LD" ]; then
    exec env "${UNSET[@]}" LD_LIBRARY_PATH="$LD" "$@"
fi
exec env "${UNSET[@]}" -u LD_LIBRARY_PATH "$@"
