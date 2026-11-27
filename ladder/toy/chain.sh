#!/bin/bash
# One margin-toy variant end to end on one GPU (ladder/toy/TOY.md). Stages, in this order:
#   pretrain  ladder/train.sh: the stable run of CONFIG to TPP_MAX
#   export    pretraining checkpoints (PRE_ITERS) -> HF directories <run>/hf/main_iter<N> (ladder/export/export_hf.sh)
#   probe     ladder/toy/probe.py on them: walk positions against the exact q, geometry, decoding sweeps
#   sft       per SFT mode: ladder/sft/train_sft.sh (protocol A head) -> ladder/sft/export.sh (SFT_ITERS)
#             -> probe.py --paths (path task: teacher-forced q and sampled pass@k / coverage)
# Each step skips what is done (a finished run, an existing export_meta.json / probe.json), so a rerun only does
# what is missing. Normally started by ladder/toy/pack.sh, which sets CUDA_VISIBLE_DEVICES and MASTER_PORT.
#
# Environment:
#   TOY_ROOT      ladder/toy/prepare.sh output; runs go to $TOY_ROOT/out (the OUT_ROOT of the ladder scripts)
#   HEAD MARGIN_D MARGIN_REVERSE HEAD_SCALE SFN_WEIGHT   LM head and margin, ladder/configs/head.sh (default plain)
#   LR            peak LR (default: the config's), SEED (default 1: init and data order; same seed = same batches)
#   CONFIG        default toy_l2;  OPT  default muon
#   STAGES        comma- or space-separated subset of: pretrain export probe sft (default all four)
#   SFT_MODES     default cond,canon (ladder/toy/make_data.py also writes uniform)
#   PRE_ITERS     pretraining checkpoints to export + probe: all (default) | latest | "N N ..." (comma ok)
#   SFT_ITERS     SFT checkpoints to export + probe: all (default) | latest | list
#   PROBE_ARGS    extra probe.py arguments;  FORCE_PROBE=1 re-probes;  VENV;  PYTHON (default python)
#   DRY_RUN=1     print the commands (nothing exists yet, so no export / probe steps are listed)
# Run directory: $TOY_ROOT/out/runs/<CONFIG>_<OPT>_lr<LR><HEAD_TAG>_seed<SEED>; SFT: $TOY_ROOT/out/sft/<run>_main_sft_toy_<mode>.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
: "${TOY_ROOT:?set TOY_ROOT (ladder/toy/prepare.sh output)}"

export CONFIG=${CONFIG:-toy_l2} OPT=${OPT:-muon} SEED=${SEED:-1}
export HEAD=${HEAD:-plain} MARGIN_D=${MARGIN_D:-0} MARGIN_REVERSE=${MARGIN_REVERSE:-0}
export HEAD_SCALE=${HEAD_SCALE:-60} SFN_WEIGHT=${SFN_WEIGHT:-1e-3}
source "ladder/configs/${CONFIG}.sh"        # SEQ_LEN, the LR default
source ladder/configs/head.sh               # HEAD_TAG; rejects a bad head
export LR
export TOKENIZER=$TOY_ROOT/tokenizer
export TRAIN_DATA=$TOY_ROOT/megatron/train VALID_DATA=$TOY_ROOT/megatron/val
export OUT_ROOT=$TOY_ROOT/out
export NNODES=1 NODE_RANK=0 GPUS_PER_NODE=1 MASTER_ADDR=localhost MASTER_PORT=${MASTER_PORT:-29500}
export RUN=${CONFIG}_${OPT}_lr${LR}${HEAD_TAG}_seed${SEED}
export VALIDATE_LEN=$SEQ_LEN                # export_hf.sh logits check within the trained context
unset MODE TPP STOP_TPP
RUN_DIR=$OUT_ROOT/runs/$RUN
DATA_DIR=$TOY_ROOT/data
PY=${PYTHON:-python}
STAGES=" $(echo "${STAGES:-pretrain export probe sft}" | tr ',' ' ') "
SFT_MODES=$(echo "${SFT_MODES:-cond canon}" | tr ',' ' ')
PRE_ITERS=$(echo "${PRE_ITERS:-all}" | tr ',' ' ')
SFT_ITERS=$(echo "${SFT_ITERS:-all}" | tr ',' ' ')
for s in $STAGES; do
    case "$s" in pretrain|export|probe|sft) ;; *) echo "unknown stage $s (pretrain export probe sft)" >&2; exit 1 ;; esac
done

run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; fi; }
has() { [[ "$STAGES" == *" $1 "* ]]; }
iters_of() {   # iters_of <checkpoint dir> <all | latest | list>
    case "$2" in
    all)    ls -d "$1"/iter_* 2>/dev/null | sed 's#.*/iter_0*##' | sort -n | tr '\n' ' ' || true ;;
    latest) if [ -f "$1/latest_checkpointed_iteration.txt" ]; then cat "$1/latest_checkpointed_iteration.txt"; fi ;;
    *)      echo "$2" ;;
    esac
}
probe_dirs() {   # probe_dirs <probe.py args ...> -- <HF dirs ...>: probes those without probe.json
    local args=() dirs=() d
    while [ "$1" != -- ]; do args+=("$1"); shift; done
    shift
    for d in "$@"; do
        if [ -f "$d/probe.json" ] && [ "${FORCE_PROBE:-0}" != 1 ]; then continue; fi
        if [ -f "$d/export_meta.json" ]; then dirs+=("$d"); fi
    done
    if [ ${#dirs[@]} -gt 0 ]; then
        # shellcheck disable=SC2086
        run $PY ladder/toy/probe.py --data "$DATA_DIR" "${args[@]}" ${PROBE_ARGS:-} --models "${dirs[@]}"
    fi
}

echo "== toy chain $RUN: stages [$STAGES], SFT modes [$SFT_MODES], GPU ${CUDA_VISIBLE_DEVICES:-?}, port $MASTER_PORT"

if has pretrain; then
    run env MODE=stable bash ladder/train.sh
fi

PRE_CKPT=$RUN_DIR/ckpt/main
PRE_HF=()
for it in $(iters_of "$PRE_CKPT" "$PRE_ITERS"); do PRE_HF+=("$RUN_DIR/hf/main_iter$it"); done
if has export; then
    for out in ${PRE_HF[@]+"${PRE_HF[@]}"}; do
        if [ -f "$out/export_meta.json" ]; then continue; fi
        run env CKPT="$PRE_CKPT" ITER="${out##*_iter}" OUT="$out" bash ladder/export/export_hf.sh
    done
fi
if has probe; then
    probe_dirs -- ${PRE_HF[@]+"${PRE_HF[@]}"}
fi

if has sft; then
    for mode in $SFT_MODES; do
        export SFT_TAG=toy_$mode SFT_DATA_DIR=$DATA_DIR/sft/$mode PRETRAIN_RUN=$RUN PRETRAIN_TAG=main
        SFT_DIR=$OUT_ROOT/sft/${RUN}_main_sft_$SFT_TAG
        run bash ladder/sft/train_sft.sh
        SFT_HF=()
        for it in $(iters_of "$SFT_DIR/ckpt" "$SFT_ITERS"); do SFT_HF+=("$SFT_DIR/hf/iter_$it"); done
        if [ ${#SFT_HF[@]} -gt 0 ]; then
            run env TARGET=sft ITERS="${SFT_HF[*]##*iter_}" bash ladder/sft/export.sh
        fi
        probe_dirs --paths --sft-mode "$mode" -- ${SFT_HF[@]+"${SFT_HF[@]}"}
    done
fi
echo "== toy chain $RUN done"
