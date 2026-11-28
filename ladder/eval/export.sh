#!/bin/bash
# Export the checkpoints of a pretraining run to Hugging Face directories, the MODEL input of ladder/eval/lm_eval.sh
# (ladder/export/export_hf.sh: plain Llama / Qwen3, a feature-norm head folded into lm_head, Megatron vs HF logits
# checked). One GPU, Megatron environment:
#   CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 \
#     sbatch -J pre-export -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
#     --mem=128G -t 02:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/eval/export.sh'
#
# Which run: $OUT_ROOT/runs/<CONFIG>_<OPT>_lr<LR><HEAD_TAG>, named as ladder/train.sh names it:
#   CONFIG, OPT (default muon, as ladder/sft), LR, HEAD / MARGIN_D / MARGIN_REVERSE / HEAD_SCALE / SFN_WEIGHT
#   (ladder/configs/head.sh; the run's own head, which load_checkpoint checks), RUN (overrides the name)
# Which checkpoints:
#   TAGS    checkpoint directories under <run>/ckpt (default: every one present, e.g. main tpp20 tpp50)
#   ITERS   final (default) | all | space-separated iterations
#           final: per tag, its latest checkpoint if the LR has decayed to 0 there: the --exit-interval of a decay
#           branch (tpp<T>), the --train-iters of the stable run (main), read from the last cmd_<tag>_*.sh that
#           ladder/train.sh wrote. A stable run stopped at STOP_TPP is mid-schedule, and a branch still running
#           has saves on its way: such a tag is skipped with a note (ALLOW_UNFINISHED=1 exports its latest anyway).
#           all: every iter_* of the tag (the stable run keeps the branch starts: an LR trajectory, not decayed)
# Optional:
#   TOKENIZER  default: the --tokenizer-model of the tag's last cmd_<tag>_*.sh, else the family's directory
#   OUT_ROOT (/fast/users/zhuangxinlin/ckpts), VENV, VALIDATE_SEQS (export_hf.sh), FORCE=1 (re-export),
#   DRY_RUN=1 (print the export commands only)
# Output: <run>/hf/<tag>_iter<N>. A directory whose export_meta.json records a passed logits check is kept unless
# FORCE=1; a failed export (its export_meta.json has validation.passed false) is redone. A failed export does not
# stop the others; the script then exits 1.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
: "${CONFIG:?set CONFIG, e.g. CONFIG=llama3_r1}" "${LR:?set LR, the peak LR of the run (it names the run)}"
source "ladder/configs/${CONFIG}.sh"
source ladder/configs/head.sh
OPT=${OPT:-muon}
OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}
RUN=${RUN:-${CONFIG}_${OPT}_lr${LR}${HEAD_TAG}}
RUN_DIR=$OUT_ROOT/runs/$RUN
ITERS=${ITERS:-final}

last_cmd() {   # last_cmd <tag>: the newest cmd_<tag>_<stamp>.sh ladder/train.sh wrote for that tag, empty if none
    ls "$RUN_DIR/cmd_${1}_"*.sh 2>/dev/null | tail -n 1 || true
}
cmd_flag() {   # cmd_flag <cmd file> <flag>: the flag's value in that command, empty if absent
    python3 -c '
import shlex, sys
cmd = shlex.split(open(sys.argv[1]).read())
print(cmd[cmd.index(sys.argv[2]) + 1] if sys.argv[2] in cmd else "")' "$1" "$2"
}
tracked_iter() {   # latest iteration in a Megatron checkpoint directory, empty if none
    local f=$1/latest_checkpointed_iteration.txt
    if [ -f "$f" ]; then cat "$f"; fi
}
case "$CONFIG" in
    qwen3_*) FAMILY_TOKENIZER=/fast/users/zhuangxinlin/tokenizers/qwen3 ;;
    *)       FAMILY_TOKENIZER=/fast/users/zhuangxinlin/tokenizers/llama-3 ;;
esac

if [ -z "${TAGS:-}" ]; then
    TAGS=$(for d in "$RUN_DIR"/ckpt/*/; do
               if [ -f "$d/latest_checkpointed_iteration.txt" ]; then basename "$d"; fi
           done | sort -V | tr '\n' ' ')
    TAGS=${TAGS% }
fi
if [ -z "$TAGS" ]; then
    echo "no checkpoint directories in $RUN_DIR/ckpt (wrong CONFIG / OPT / LR / HEAD? see RUN)" >&2
    [ "${DRY_RUN:-0}" = 1 ] || exit 1
fi
echo "== export $RUN: tags [$TAGS], ITERS=$ITERS, LM head ${HEAD_ARGS[*]:-plain}"

export_ok() {   # export_ok <hf dir>: exported, and its Megatron vs HF logits check passed (or was skipped)
    [ -f "$1/export_meta.json" ] && python3 -c '
import json, sys
v = json.load(open(sys.argv[1])).get("validation")
sys.exit(0 if v is None or v.get("passed") is True else 1)' "$1/export_meta.json"
}

DONE=()
FAILED=()
for tag in $TAGS; do
    ckpt=$RUN_DIR/ckpt/$tag
    cmd_file=$(last_cmd "$tag")
    latest=$(tracked_iter "$ckpt")
    case "$ITERS" in
    final)
        target=""
        if [ -n "$cmd_file" ]; then
            if [ "$tag" = main ]; then target=$(cmd_flag "$cmd_file" --train-iters)
            else target=$(cmd_flag "$cmd_file" --exit-interval); fi
        fi
        if [ -z "$latest" ]; then
            echo "$tag: no checkpoint in $ckpt, skipped"; continue
        elif [ "${ALLOW_UNFINISHED:-0}" = 1 ]; then
            ITER_LIST=$latest
        elif [ -z "$target" ]; then
            echo "$tag: no cmd_${tag}_*.sh in $RUN_DIR to tell where its LR decays; skipped (ITERS=$latest or" \
                 "ALLOW_UNFINISHED=1 exports it)"; continue
        elif [ "$latest" != "$target" ]; then
            echo "$tag: latest checkpoint $latest, but the LR decays to 0 at $target (running, or a stable run" \
                 "stopped at STOP_TPP); skipped (ALLOW_UNFINISHED=1 exports it)"; continue
        else
            ITER_LIST=$latest
        fi ;;
    all)
        ITER_LIST=$(ls -d "$ckpt"/iter_* 2>/dev/null | sed 's#.*/iter_0*##' | sort -n | tr '\n' ' ' || true)
        if [ -z "$ITER_LIST" ]; then echo "$tag: no iter_* checkpoints in $ckpt, skipped"; continue; fi ;;
    *)
        ITER_LIST=$ITERS ;;
    esac
    tok=${TOKENIZER:-}
    if [ -z "$tok" ] && [ -n "$cmd_file" ]; then tok=$(cmd_flag "$cmd_file" --tokenizer-model); fi
    tok=${tok:-$FAMILY_TOKENIZER}
    for it in $ITER_LIST; do
        hf=$RUN_DIR/hf/${tag}_iter$it
        DONE+=("$hf")
        if [ "${FORCE:-0}" != 1 ] && export_ok "$hf"; then
            echo "$tag iteration $it: $hf already exported (FORCE=1 redoes it)"; continue
        fi
        cmd=(env HEAD="$HEAD" MARGIN_D="$MARGIN_D" MARGIN_REVERSE="$MARGIN_REVERSE" HEAD_SCALE="$HEAD_SCALE"
             SFN_WEIGHT="$SFN_WEIGHT" CONFIG="$CONFIG" CKPT="$ckpt" ITER="$it" OUT="$hf" TOKENIZER="$tok"
             bash ladder/export/export_hf.sh)
        echo "+ ${cmd[*]}"
        if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi
        if ! "${cmd[@]}"; then
            echo "$tag iteration $it: export failed" >&2
            FAILED+=("$hf")
        fi
    done
done
echo "== HF directories (MODEL for ladder/eval/lm_eval.sh):"
printf '   %s\n' ${DONE[@]+"${DONE[@]}"}
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "export failed for: ${FAILED[*]}" >&2
    exit 1
fi
