#!/bin/bash
# SFT on OpenMathInstruct-2, HF export and offline eval (ladder/SFT.md) of the llama3_r2 and llama3_r1 TPP50 runs:
# the token-margin runs of ladder/margin/run_margin_pretrain.sh and the plain-head baseline with the same LR
# (ladder/run_llama_r2_lr_sweep.sh, ladder/run_llama_r1_lr_sweep.sh). Data first: ladder/sft/download_data.sh.
# Run from the Megatron-LM root with the Megatron uv environment active. `bash ladder/sft/run_sft.sh` submits every
# chain at the bottom of the rungs in RUNGS (default both), e.g. `RUNGS=r1 bash ladder/sft/run_sft.sh`; CHAINS picks
# chains by name, e.g. `CHAINS="r1-plain r1-hfn-d0.02" bash ladder/sft/run_sft.sh`. SFT_GPUS: GPUs per SFT training
# job (default 2; one node, so at most 8). IDLE: which steps run on idle resources (--qos=polaris_idle, see below),
# e.g. `IDLE=1 RUNGS=r1 bash ladder/sft/run_sft.sh` or `IDLE=eval ...`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

# ---- common ---------------------------------------------------------------------------------------------
DATA=/fast/users/zhuangxinlin/data
export OUT_ROOT=/fast/users/zhuangxinlin/ckpts
export SFT_DATA_DIR=$DATA/sft_megatron/omi2_500k_ep10_llama3   # prepare_sft_data.py output, Llama3 tokenizer
export DATA_ROOT=$DATA                                         # eval sets
export VLLM_PYTHON=/fast/users/zhuangxinlin/envs/vllm/bin/python   # built by ladder/env/setup_vllm_env.sh
# The pretraining runs, named as ladder/train.sh names them; the CONFIG / LR / HEAD vars of each chain pick the run.
# tpp50 = the TPP50 decay branch (iteration 27500 on R2, 17500 on R1); train_sft.sh refuses a checkpoint short of
# its run's end.
export OPT=muon PRETRAIN_TAG=tpp50
# Training job size. The global batch is fixed in examples rather than derived from the data-parallel size (train_sft.sh
# would give 316 on 2 GPUs, 312 on 8): every chain then has the same batch and schedule whatever SFT_GPUS is, and a
# run resumed on another GPU count keeps them. 312 divides by 1, 2, 4 and 8 GPUs.
SFT_GPUS=${SFT_GPUS:-2}
export SFT_GBS=${SFT_GBS:-312}
# Tokens per packed microbatch per GPU: ~4 microbatches per GPU and step on 2 GPUs (a step is ~122k tokens) instead
# of ~15 at 4096; R1 pretraining ran 8 x 4096 = 32768 per microbatch on these GPUs. Same examples and loss per step.
export SFT_TOKENS_PER_RANK=${SFT_TOKENS_PER_RANK:-16384}
# Validation loss during training: on when $SFT_DATA_DIR/valid.parquet exists (ladder/sft/download_data.sh, last
# step); pass@k stays offline (export + eval below). SFT_EVAL=0 turns it off, SFT_EVAL=1 requires it.
# Idle resources (ladder/launch/slurm.sbatch, QoS): --qos=polaris_idle has no per-user GPU cap but starts only on free
# nodes and can be preempted at any time; a preempted job is requeued and starts over, which every step survives:
# train resumes from its latest epoch checkpoint (losing < 1602 iterations), export keeps the exports that passed
# their check, eval skips the models with a summary.json (redoing the one it was on).
#   IDLE=0 (default) standard QoS | 1 or all: every step | a comma list of train, export, eval (e.g. IDLE=eval)
IDLE=${IDLE:-0}
SBATCH_1GPU=(sbatch --parsable -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1
             --cpus-per-task=16 --mem=128G -o logs/%x-%j.out --open-mode=append)

# sft_chain <name> CONFIG=.. LR=.. [HEAD=.. MARGIN_D=.. MARGIN_REVERSE=..] [AFTER=<job id>] [STEPS=train,export,eval]
#           [IDLE=..]   (per-chain override of the IDLE above)
# Submits, with afterok between them:
#   train   SFT_GPUS GPUs of one node, Megatron env: $SFT_DIR/ckpt/iter_<N>, one checkpoint per epoch (every 1602
#           iterations); LR = pretraining LR / 3, Muon, global batch 312 examples, 16025 iterations, ~2B tokens.
#           Time scales as 1 / SFT_GPUS: R2 ~1 h on 8 GPUs (~0.8M tokens/s), so ~4 h on 2; R1 less
#   export  1 GPU, Megatron env: every SFT checkpoint -> $SFT_DIR/hf/iter_<N> (head folded, Megatron vs HF logits
#           checked); the pre-SFT reference, the tpp50 checkpoint -> <pretraining run>/hf/tpp50_iter<N>
#   eval    1 GPU, vLLM env: GSM8K / MATH-500 / AMC23 / Countdown pass@k of every export ->
#           <X>/eval/<name>/summary.json for a model at <X>/hf/<name>, one line per model in <X>/eval/summary.jsonl
# AFTER: a job the chain waits for, e.g. the run's tpp50 branch job printed by run_margin_pretrain.sh; without
# it, submit once that branch has finished. STEPS: a subset, e.g. STEPS=eval to redo the eval of existing exports
# (it then has no dependency). Quick check of an export: TASKS=gsm8k NUM_SAMPLES=4 MAX_PROBLEMS=50 in eval's env.
sft_chain() (
    local name=$1 after="" steps=train,export,eval idle=$IDLE kv s
    shift
    if [ -n "${CHAINS:-}" ] && [[ " $CHAINS " != *" $name "* ]]; then return 0; fi
    export HEAD=plain MARGIN_D=0 MARGIN_REVERSE=0
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
            case $s in train | export | eval) ;; *) echo "IDLE=$idle: use 0, 1, all or train,export,eval" >&2; exit 1 ;; esac
        done ;;
    esac
    qos() {   # qos <step>: the QoS flag of that step's jobs (nothing for the standard QoS)
        case ",$idle," in *,1,* | *,all,* | *,"$1",*) echo "--qos=polaris_idle" ;; esac
    }
    unset PRETRAIN_RUN
    : "${CONFIG:?sft_chain $name: pass CONFIG=..}" "${LR:?sft_chain $name: pass LR=.., the pretraining LR}"
    source ladder/sft/common.sh               # -> PRETRAIN_RUN, SFT_DIR: the names train_sft.sh / export.sh use
    local train="" export_sft="" export_pre="" eval_sft="" eval_pre=""
    case ",$steps," in *,train,*)
        train=$(ENTRY=ladder/sft/train_sft.sh GPUS_PER_NODE=$SFT_GPUS sbatch --parsable --nodes=1 $(qos train) \
            --gpus-per-node="$SFT_GPUS" --cpus-per-task=$((16 * SFT_GPUS)) --mem=$((64 * SFT_GPUS))G \
            --time=12:00:00 -J "sft-$name" ${after:+--dependency=afterok:$after} ladder/launch/slurm.sbatch) ;;
    esac
    case ",$steps," in *,export,*)
        export_sft=$(ITERS=all "${SBATCH_1GPU[@]}" $(qos export) -J "sft-export-$name" -t 04:00:00 \
            ${train:+--dependency=afterok:$train} --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh')
        export_pre=$(TARGET=pretrained "${SBATCH_1GPU[@]}" $(qos export) -J "pre-export-$name" -t 01:00:00 \
            ${after:+--dependency=afterok:$after} --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh') ;;
    esac
    case ",$steps," in *,eval,*)
        eval_sft=$(MODEL="$SFT_DIR/hf/iter_*" "${SBATCH_1GPU[@]}" $(qos eval) -J "sft-eval-$name" -t 24:00:00 \
            ${export_sft:+--dependency=afterok:$export_sft} --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh')
        eval_pre=$(MODEL="$OUT_ROOT/runs/$PRETRAIN_RUN/hf/${PRETRAIN_TAG}_iter*" "${SBATCH_1GPU[@]}" $(qos eval) \
            -J "pre-eval-$name" -t 04:00:00 ${export_pre:+--dependency=afterok:$export_pre} \
            --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh') ;;
    esac
    echo "== $name: $PRETRAIN_RUN/$PRETRAIN_TAG -> $SFT_DIR$( [ "$idle" = 0 ] || echo " (idle QoS: $idle)" )"
    echo "   train ${train:--}  export ${export_sft:--} (pre ${export_pre:--})  eval ${eval_sft:--} (pre ${eval_pre:--})"
)

# ---- tests and smoke (once, before the real jobs; not submitted by this file) ---------------------------------
# CPU, Megatron env: subset draw, Megatron's data path on prepared data, launchers, eval logic
#   python -m pytest ladder/tests -p no:cacheprovider -o addopts=
# 1 node: ~125 SFT steps from a pretraining checkpoint -> export -> quick vLLM eval -> checks (log ends in SMOKE OK);
# with a margin run's tpp50 checkpoint it also covers the feature-norm head in SFT and its fold on export
#   CONFIG=llama3_r1 LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 VLLM_PYTHON=$VLLM_PYTHON sbatch -J sft-smoke -A polaris -p polaris -N 1 --ntasks-per-node=1 \
#     --gpus-per-node=8 --cpus-per-task=128 --mem=512G -t 03:00:00 -o logs/%x-%j.out \
#     --wrap 'srun --ntasks=1 ht-run bash ladder/sft/smoke_sft.sh'

RUNGS=${RUNGS:-r2 r1}

# ---- chains: llama3_r2_muon_lr2e-3 tpp50 ---------------------------------------------------------------------
if [[ " $RUNGS " == *" r2 "* ]]; then
sft_chain r2-plain     CONFIG=llama3_r2 LR=2e-3                         # baseline: llama3_r2_muon_lr2e-3
sft_chain r2-hfn-d0.02 CONFIG=llama3_r2 LR=2e-3 HEAD=hfn MARGIN_D=0.02  # ..._hfn_d0.02, SFT head hfn, d = 0
sft_chain r2-sfn-d0.02 CONFIG=llama3_r2 LR=2e-3 HEAD=sfn MARGIN_D=0.02  # ..._sfn_d0.02, SFT head nfn, d = 0
fi
# queued behind pretraining instead:  sft_chain r2-hfn-d0.02 CONFIG=llama3_r2 LR=2e-3 HEAD=hfn MARGIN_D=0.02 AFTER=<job>
# optional variants of run_margin_pretrain.sh, e.g.:  sft_chain r2-hfn-d0 CONFIG=llama3_r2 LR=2e-3 HEAD=hfn MARGIN_D=0

# ---- chains: llama3_r1_muon_lr2.8e-3 tpp50 -------------------------------------------------------------------
if [[ " $RUNGS " == *" r1 "* ]]; then
sft_chain r1-plain     CONFIG=llama3_r1 LR=2.8e-3                         # baseline: llama3_r1_muon_lr2.8e-3
sft_chain r1-hfn-d0.02 CONFIG=llama3_r1 LR=2.8e-3 HEAD=hfn MARGIN_D=0.02  # ..._hfn_d0.02, SFT head hfn, d = 0
sft_chain r1-sfn-d0.02 CONFIG=llama3_r1 LR=2.8e-3 HEAD=sfn MARGIN_D=0.02  # ..._sfn_d0.02, SFT head nfn, d = 0
fi

# ---- plot ------------------------------------------------------------------------------------------------
