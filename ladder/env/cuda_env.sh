#!/bin/bash
# CUDA toolkit locations for Triton / torch.compile, sourced by ladder/train.sh (and read by check_env.py).
#
# The NGC image's triton ships without its own CUDA headers and binaries
# (triton/backends/nvidia/{include,bin}); the image normally points it at /usr/local/cuda through
# Docker ENV (CUDA_HOME, TRITON_PTXAS_PATH, ...). Those variables are missing in our jobs, so
# Megatron's jit warmup (torch.compile -> Triton) fails with "cuda.h: No such file" or
# "Cannot find ptxas". Set them here, keeping any value already in the environment.
# Safe under `set -euo pipefail`.

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

case ":${PATH:-}:" in *":$CUDA_HOME/bin:"*) ;; *) export PATH="$CUDA_HOME/bin${PATH:+:$PATH}" ;; esac

# Triton's cuda_utils.c helper includes cuda.h; its gcc call only searches triton's own include dir.
if [ -f "$CUDA_HOME/include/cuda.h" ]; then
    case ":${CPATH:-}:" in *":$CUDA_HOME/include:"*) ;; *) export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}" ;; esac
fi

# Binaries Triton looks for only in triton/backends/nvidia/bin unless these are set.
for _tool in ptxas cuobjdump nvdisasm; do
    _var=TRITON_$(echo "$_tool" | tr '[:lower:]' '[:upper:]')_PATH
    if [ -z "${!_var:-}" ] && [ -x "$CUDA_HOME/bin/$_tool" ]; then export "$_var=$CUDA_HOME/bin/$_tool"; fi
done
unset _tool _var

# libdevice: use the toolkit's only if triton's bundled copy is missing.
if [ -z "${TRITON_LIBDEVICE_PATH:-}" ] && [ -f "$CUDA_HOME/nvvm/libdevice/libdevice.10.bc" ]; then
    _triton_dir=$(python -c 'import os, triton; print(os.path.dirname(triton.__file__))' 2>/dev/null || true)
    if [ -n "$_triton_dir" ] && [ ! -f "$_triton_dir/backends/nvidia/lib/libdevice.10.bc" ]; then
        export TRITON_LIBDEVICE_PATH=$CUDA_HOME/nvvm/libdevice/libdevice.10.bc
    fi
    unset _triton_dir
fi
