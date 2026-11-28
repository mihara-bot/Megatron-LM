#!/bin/bash
# HF export + lm-eval (vLLM) of pretraining runs (ladder/EVAL.md). Per run, two 1-GPU jobs, afterok between them:
#   export  Megatron env: ladder/eval/export.sh -> <run>/hf/<tag>_iter<N> for each checkpoint directory whose run
#           has finished its decay (main at TPP_MAX, tpp20, tpp50, ...; unfinished ones are skipped)
#   eval    vLLM env: ladder/eval/lm_eval.sh on every export of the run -> <run>/lm_eval/<tag>_iter<N>/results.json,
#           one line per model in <run>/lm_eval/summary.jsonl
# Run from the Megatron-LM root with the Megatron uv environment active: `bash ladder/eval/run_eval.sh` submits every
# chain at the bottom of the rungs in RUNGS (default both), e.g. `RUNGS=r1 bash ladder/eval/run_eval.sh`; CHAINS picks
# chains by name, e.g. `CHAINS="r1-plain r1-hfn-d0.02" bash ladder/eval/run_eval.sh`. IDLE: which steps run on idle
# resources (--qos=polaris_idle, preemptible; both steps survive a requeue: export keeps the exports that passed their
# check, eval skips the models with a results.json): 0 (default) | 1 or all | export | eval.
# DRY_RUN=1 prints the sbatch commands (and the variables the jobs read) instead of submitting.
# Once per cluster, before the first chain: the datasets (ladder/eval/download_datasets.sh) and lm-eval in the vLLM
# environment (ladder/EVAL.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

# ---- common ---------------------------------------------------------------------------------------------
export OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}
export VLLM_PYTHON=${VLLM_PYTHON:-/fast/users/zhuangxinlin/envs/vllm/bin/python}   # ladder/env/setup_vllm_env.sh
export EVAL_DATA_DIR=${EVAL_DATA_DIR:-/fast/users/zhuangxinlin/data/lm_eval_hf}   # ladder/eval/download_datasets.sh
export OPT=${OPT:-muon}
IDLE=${IDLE:-0}
SBATCH_1GPU=(sbatch --parsable -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1
             --cpus-per-task=16 --mem=128G -o logs/%x-%j.out --open-mode=append)
JOB_VARS=(CONFIG OPT LR HEAD MARGIN_D MARGIN_REVERSE HEAD_SCALE SFN_WEIGHT RUN TAGS ITERS MODEL TASK_SET TASKS)

submit() {   # submit <sbatch command>: its job id; with DRY_RUN=1 print it instead, and use the job name as the id
    if [ "${DRY_RUN:-0}" != 1 ]; then "$@"; return; fi
    local v vars="" name="" prev=""
    for v in "${JOB_VARS[@]}"; do
        if [ -n "${!v:-}" ]; then vars+="$v=${!v} "; fi
    done
    for v in "$@"; do
        if [ "$prev" = -J ]; then name=$v; fi
        prev=$v
    done
    echo "+ $vars$*" >&2
    echo "$name"
}

# eval_chain <name> CONFIG=.. LR=.. [OPT=.. HEAD=.. MARGIN_D=.. MARGIN_REVERSE=.. HEAD_SCALE=.. SFN_WEIGHT=.. RUN=..]
#            [TAGS=..] [ITERS=..] [TASK_SET=.. | TASKS=..] [AFTER=<job id>] [STEPS=export,eval] [IDLE=..]
# The run as ladder/train.sh names it (<CONFIG>_<OPT>_lr<LR><head tag>, RUN overrides). TAGS / ITERS go to export.sh
# (default: every checkpoint directory, its decayed checkpoint); the eval then reads <run>/hf/<tag>_iter* of those
# tags (default <run>/hf/*_iter*: every export of the run, earlier ones included; evaluated ones are skipped).
# AFTER: a job the chain waits for, e.g. the run's branch job. STEPS=eval redoes the eval of existing exports (no
# dependency), e.g. with TASK_SET=core.
eval_chain() (
    local name=$1 after="" steps=export,eval idle=$IDLE kv s run models t
    shift
    if [ -n "${CHAINS:-}" ] && [[ " $CHAINS " != *" $name "* ]]; then return 0; fi
    export HEAD=plain MARGIN_D=0 MARGIN_REVERSE=0
    unset RUN TAGS ITERS MODEL
    for kv in "$@"; do
        case $kv in
            AFTER=*) after=${kv#AFTER=} ;;
            STEPS=*) steps=${kv#STEPS=} ;;
            IDLE=*) idle=${kv#IDLE=} ;;
            *) export "${kv?}" ;;
        esac
    done
    case $idle in 0 | 1 | all) ;; *)
        for s in ${idle//,/ }; do
            case $s in export | eval) ;; *) echo "IDLE=$idle: use 0, 1, all or export,eval" >&2; exit 1 ;; esac
        done ;;
    esac
    qos() {   # qos <step>: the QoS flag of that step's job (nothing for the standard QoS)
        case ",$idle," in *,1,* | *,all,* | *,"$1",*) echo "--qos=polaris_idle" ;; esac
    }
    : "${CONFIG:?eval_chain $name: pass CONFIG=..}" "${LR:?eval_chain $name: pass LR=.., the peak LR of the run}"
    source ladder/configs/head.sh               # -> HEAD_TAG, as ladder/train.sh names the run
    run=${RUN:-${CONFIG}_${OPT}_lr${LR}${HEAD_TAG}}
    if [ -n "${TAGS:-}" ]; then
        models=""
        for t in $TAGS; do models+="$OUT_ROOT/runs/$run/hf/${t}_iter* "; done
        models=${models% }
    else
        models="$OUT_ROOT/runs/$run/hf/*_iter*"
    fi
    local export_job="" eval_job=""
    case ",$steps," in *,export,*)
        export_job=$(submit "${SBATCH_1GPU[@]}" $(qos export) -J "pre-export-$name" -t 02:00:00 \
            ${after:+--dependency=afterok:$after} --wrap 'srun --ntasks=1 ht-run bash ladder/eval/export.sh') ;;
    esac
    case ",$steps," in *,eval,*)
        export MODEL=$models
        eval_job=$(submit "${SBATCH_1GPU[@]}" $(qos eval) -J "lm-eval-$name" -t 06:00:00 \
            ${export_job:+--dependency=afterok:$export_job} --wrap 'srun --ntasks=1 ht-run bash ladder/eval/lm_eval.sh') ;;
    esac
    echo "== $name: $OUT_ROOT/runs/$run$( [ "$idle" = 0 ] || echo " (idle QoS: $idle)" )"
    echo "   export ${export_job:--}  eval ${eval_job:--}"
)

# ---- tests (once, before the real jobs; not submitted by this file) -------------------------------------------
# CPU: python -m pytest ladder/tests/test_eval_scripts.py -p no:cacheprovider -o addopts=
# Smoke of one export (1 GPU, vLLM env): MODEL=<run>/hf/tpp50_iter<N> TASKS=arc_easy,hellaswag LIMIT=200 \
#   VLLM_PYTHON=$VLLM_PYTHON sbatch ... --wrap 'srun --ntasks=1 ht-run bash ladder/eval/lm_eval.sh'

RUNGS=${RUNGS:-r2 r1}

# ---- the token-margin runs (ladder/margin/run_margin_pretrain.sh) and their plain-head baselines -------------
if [[ " $RUNGS " == *" r2 "* ]]; then
eval_chain r2-plain     CONFIG=llama3_r2 LR=2e-3                           # llama3_r2_muon_lr2e-3
eval_chain r2-hfn-d0.02 CONFIG=llama3_r2 LR=2e-3 HEAD=hfn MARGIN_D=0.02    # ..._hfn_d0.02
eval_chain r2-sfn-d0.02 CONFIG=llama3_r2 LR=2e-3 HEAD=sfn MARGIN_D=0.02    # ..._sfn_d0.02
fi
if [[ " $RUNGS " == *" r1 "* ]]; then
eval_chain r1-plain     CONFIG=llama3_r1 LR=2.8e-3                         # llama3_r1_muon_lr2.8e-3
eval_chain r1-hfn-d0.02 CONFIG=llama3_r1 LR=2.8e-3 HEAD=hfn MARGIN_D=0.02  # ..._hfn_d0.02
eval_chain r1-sfn-d0.02 CONFIG=llama3_r1 LR=2.8e-3 HEAD=sfn MARGIN_D=0.02  # ..._sfn_d0.02
fi
# queued behind pretraining:    eval_chain r1-hfn-d0.02 CONFIG=llama3_r1 LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 AFTER=<job>
# a larger task set afterwards: eval_chain r1-plain CONFIG=llama3_r1 LR=2.8e-3 STEPS=eval TASK_SET=core
# an LR sweep (every grid LR):  for lr in 1.4e-3 2e-3 2.8e-3 4e-3 5.6e-3; do
#                                   eval_chain q1-lr$lr CONFIG=qwen3_r1 LR=$lr; done
