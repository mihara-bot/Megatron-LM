# Shared by ladder/sft/train_sft.sh and ladder/sft/eval.sh: which pretraining run SFT starts from, the SFT
# LM head, and the SFT run directory. Sourced from the repo root after the config (MODEL_ARGS, SEQ_LEN).
#
# Pretraining run, named as ladder/train.sh names it:
#   CONFIG, OPT (its optimizer, default muon), LR (its peak LR), HEAD / MARGIN_D / MARGIN_REVERSE / HEAD_SCALE /
#   SFN_WEIGHT
#   PRETRAIN_TAG  its checkpoint directory: main (default: the stable run) | tpp20 | tpp50 | ...
#   PRETRAIN_RUN  run directory name, overriding <CONFIG>_<OPT>_lr<LR><HEAD_TAG>
#   SFT_TAG       SFT run-name suffix (default omi2_500k_ep10)
#   SFT_OPT       SFT optimizer, ladder/configs/opt/<SFT_OPT>.sh: muon (default, whatever OPT is) | adamw; a
#                 non-default one is added to the SFT run name
# Sets: PRETRAIN_RUN PRETRAIN_CKPT PRETRAIN_HEAD_ARGS SFT_NAME SFT_DIR SFT_HEAD_ARGS SFT_OPT TOKENIZER (family default);
# functions tracked_iter, check_pretrain_finished
#
# SFT head (protocol A): the pretraining head with no margin and no sfn penalty. hfn stays hfn (same s, final
# norm still frozen, so the SFT checkpoint folds on export); nfn and sfn use nfn (same geometry); plain stays
# plain. load_checkpoint refuses a head whose geometry differs from the checkpoint's.

OPT=${OPT:-muon}
SFT_OPT=${SFT_OPT:-muon}
if [ ! -f "ladder/configs/opt/${SFT_OPT}.sh" ]; then echo "no ladder/configs/opt/${SFT_OPT}.sh (SFT_OPT)" >&2; exit 1; fi
: "${OUT_ROOT:?set OUT_ROOT}" "${LR:?set LR, the peak LR of the pretraining run (it names the run)}"
PRETRAIN_TAG=${PRETRAIN_TAG:-main}
SFT_TAG=${SFT_TAG:-omi2_500k_ep10}

source ladder/configs/head.sh                 # pretraining head: flags + run-name tag
PRETRAIN_HEAD_ARGS=(${HEAD_ARGS[@]+"${HEAD_ARGS[@]}"})
PRETRAIN_RUN=${PRETRAIN_RUN:-${CONFIG}_${OPT}_lr${LR}${HEAD_TAG}}
PRETRAIN_CKPT=$OUT_ROOT/runs/$PRETRAIN_RUN/ckpt/$PRETRAIN_TAG
SFT_NAME=${PRETRAIN_RUN}_${PRETRAIN_TAG}_sft_${SFT_TAG}
if [ "$SFT_OPT" != muon ]; then SFT_NAME+="_${SFT_OPT}"; fi
SFT_DIR=$OUT_ROOT/sft/$SFT_NAME

PRETRAIN_HEAD_TAG=$HEAD_TAG
_PRE_HEAD=$HEAD; _PRE_D=$MARGIN_D; _PRE_REV=$MARGIN_REVERSE
case "$HEAD" in sfn) HEAD=nfn ;; esac
MARGIN_D=0; MARGIN_REVERSE=0
source ladder/configs/head.sh                 # SFT head: same geometry, no margin
SFT_HEAD_ARGS=(${HEAD_ARGS[@]+"${HEAD_ARGS[@]}"})
HEAD=$_PRE_HEAD; MARGIN_D=$_PRE_D; MARGIN_REVERSE=$_PRE_REV
unset _PRE_HEAD _PRE_D _PRE_REV

case "$CONFIG" in
    qwen3_*) _TOKENIZER_DEFAULT=/fast/users/zhuangxinlin/tokenizers/qwen3 ;;
    *)       _TOKENIZER_DEFAULT=/fast/users/zhuangxinlin/tokenizers/llama-3 ;;
esac
TOKENIZER=${TOKENIZER:-$_TOKENIZER_DEFAULT}

tracked_iter() {   # latest iteration in a Megatron checkpoint directory, empty if none
    local f=$1/latest_checkpointed_iteration.txt
    if [ -f "$f" ]; then cat "$f"; fi
}

# The pretraining checkpoint directory's target iteration, from the last command ladder/train.sh wrote for this tag
# (--exit-interval, else --train-iters); empty without one. A decay branch also saves on its way to the end, so a
# checkpoint alone does not mean the run is done.
pretrain_target() {
    local cmd
    cmd=$(ls "$OUT_ROOT/runs/$PRETRAIN_RUN/cmd_${PRETRAIN_TAG}_"*.sh 2>/dev/null | tail -n 1 || true)
    if [ -z "$cmd" ]; then return 0; fi
    python3 -c '
import shlex, sys
cmd = shlex.split(open(sys.argv[1]).read())
flag = "--exit-interval" if "--exit-interval" in cmd else "--train-iters"
print(cmd[cmd.index(flag) + 1])' "$cmd"
}

check_pretrain_finished() {   # check_pretrain_finished <iteration>: exit 1 unless it is the run's target
    local target
    target=$(pretrain_target)
    if [ -n "$1" ] && [ -n "$target" ] && [ "$1" != "$target" ] && [ "${ALLOW_UNFINISHED_PRETRAIN:-0}" != 1 ]; then
        echo "$PRETRAIN_CKPT is at iteration $1, but its run trains it to $target: still running?" \
             "(ALLOW_UNFINISHED_PRETRAIN=1 goes ahead anyway)" >&2
        exit 1
    fi
}
