#!/bin/bash
# SFT on OpenMathInstruct-2, HF export and offline eval (ladder/SFT.md) of the llama3_r2 TPP50 runs:
# the token-margin runs of ladder/margin/run_margin_pretrain.sh and the plain-head baseline with the same LR
# (ladder/run_llama_r2_lr_sweep.sh). Data first: ladder/sft/download_data.sh.
# Run from the Megatron-LM root with the Megatron uv environment active. `bash ladder/sft/run_sft.sh` submits every
# chain at the bottom; to submit one, paste the "common" block, the sft_chain function and that chain's line.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

# ---- common ---------------------------------------------------------------------------------------------
DATA=/fast/users/zhuangxinlin/data
export OUT_ROOT=/fast/users/zhuangxinlin/ckpts
export SFT_DATA_DIR=$DATA/sft_megatron/omi2_500k_ep10_llama3   # prepare_sft_data.py output, Llama3 tokenizer
export DATA_ROOT=$DATA                                         # eval sets
export VLLM_PYTHON=/path/to/vllm-env/bin/python                # TODO: python of an environment with vLLM
# The pretraining runs, named as ladder/train.sh names them; the HEAD vars of each chain pick the run.
# tpp50 = the TPP50 decay branch (iteration 27500); train_sft.sh refuses a checkpoint short of its run's end.
export CONFIG=llama3_r2 OPT=muon LR=2e-3 PRETRAIN_TAG=tpp50
SBATCH_1GPU=(sbatch --parsable -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1
             --cpus-per-task=16 --mem=128G -o logs/%x-%j.out)

# sft_chain <name> [HEAD=.. MARGIN_D=.. MARGIN_REVERSE=..] [AFTER=<job id>] [STEPS=train,export,eval]
# Submits, with afterok between them:
#   train   1 node (8 GPUs), Megatron env: $SFT_DIR/ckpt/iter_<N>, one checkpoint per epoch; LR 2e-3 / 3, Muon,
#           global batch 312 examples, 16025 iterations (R2 trains ~0.8M tokens/s on 8 GPUs: ~1 h)
#   export  1 GPU, Megatron env: every SFT checkpoint -> $SFT_DIR/hf/iter_<N> (head folded, Megatron vs HF logits
#           checked); the pre-SFT reference, the tpp50 checkpoint -> <pretraining run>/hf/tpp50_iter27500
#   eval    1 GPU, vLLM env: GSM8K / MATH-500 / AMC23 / Countdown pass@k of every export ->
#           <X>/eval/<name>/summary.json for a model at <X>/hf/<name>, one line per model in <X>/eval/summary.jsonl
# AFTER: a job the chain waits for, e.g. the run's tpp50 branch job printed by run_margin_pretrain.sh; without
# it, submit once that branch has finished. STEPS: a subset, e.g. STEPS=eval to redo the eval of existing exports
# (it then has no dependency). Quick check of an export: TASKS=gsm8k NUM_SAMPLES=4 MAX_PROBLEMS=50 in eval's env.
sft_chain() (
    local name=$1 after="" steps=train,export,eval kv
    shift
    export HEAD=plain MARGIN_D=0 MARGIN_REVERSE=0
    for kv in "$@"; do
        case $kv in
            AFTER=*) after=${kv#AFTER=} ;;
            STEPS=*) steps=${kv#STEPS=} ;;
            *) export "${kv?}" ;;
        esac
    done
    unset PRETRAIN_RUN
    source ladder/sft/common.sh               # -> PRETRAIN_RUN, SFT_DIR: the names train_sft.sh / export.sh use
    local train="" export_sft="" export_pre="" eval_sft="" eval_pre=""
    case ",$steps," in *,train,*)
        train=$(ENTRY=ladder/sft/train_sft.sh sbatch --parsable --nodes=1 --time=12:00:00 -J "sft-$name" \
            ${after:+--dependency=afterok:$after} ladder/launch/slurm.sbatch) ;;
    esac
    case ",$steps," in *,export,*)
        export_sft=$(ITERS=all "${SBATCH_1GPU[@]}" -J "sft-export-$name" -t 04:00:00 \
            ${train:+--dependency=afterok:$train} --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh')
        export_pre=$(TARGET=pretrained "${SBATCH_1GPU[@]}" -J "pre-export-$name" -t 01:00:00 \
            ${after:+--dependency=afterok:$after} --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh') ;;
    esac
    case ",$steps," in *,eval,*)
        eval_sft=$(MODEL="$SFT_DIR/hf/iter_*" "${SBATCH_1GPU[@]}" -J "sft-eval-$name" -t 24:00:00 \
            ${export_sft:+--dependency=afterok:$export_sft} --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh')
        eval_pre=$(MODEL="$OUT_ROOT/runs/$PRETRAIN_RUN/hf/${PRETRAIN_TAG}_iter*" "${SBATCH_1GPU[@]}" \
            -J "pre-eval-$name" -t 04:00:00 ${export_pre:+--dependency=afterok:$export_pre} \
            --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh') ;;
    esac
    echo "== $name: $PRETRAIN_RUN/$PRETRAIN_TAG -> $SFT_DIR"
    echo "   train ${train:--}  export ${export_sft:--} (pre ${export_pre:--})  eval ${eval_sft:--} (pre ${eval_pre:--})"
)

# ---- tests and smoke (once, before the real jobs; not submitted by this file) ---------------------------------
# CPU, Megatron env: subset draw, Megatron's data path on prepared data, launchers, eval logic
#   python -m pytest ladder/tests -p no:cacheprovider -o addopts=
# 1 node: ~125 SFT steps from a pretraining checkpoint -> export -> quick vLLM eval -> checks (log ends in SMOKE OK);
# with a margin run's tpp50 checkpoint it also covers the feature-norm head in SFT and its fold on export
#   HEAD=hfn MARGIN_D=0.02 VLLM_PYTHON=$VLLM_PYTHON sbatch -J sft-smoke -A polaris -p polaris -N 1 --ntasks-per-node=1 \
#     --gpus-per-node=8 --cpus-per-task=128 --mem=512G -t 03:00:00 -o logs/%x-%j.out \
#     --wrap 'srun --ntasks=1 ht-run bash ladder/sft/smoke_sft.sh'

# ---- chains: llama3_r2_muon_lr2e-3 tpp50 ---------------------------------------------------------------------
sft_chain r2-plain                                 # baseline: llama3_r2_muon_lr2e-3
sft_chain r2-hfn-d0.02 HEAD=hfn MARGIN_D=0.02      # llama3_r2_muon_lr2e-3_hfn_d0.02, SFT head hfn, d = 0
sft_chain r2-sfn-d0.02 HEAD=sfn MARGIN_D=0.02      # llama3_r2_muon_lr2e-3_sfn_d0.02, SFT head nfn, d = 0
# queued behind pretraining instead:  sft_chain r2-hfn-d0.02 HEAD=hfn MARGIN_D=0.02 AFTER=<tpp50 branch job id>
# optional variants of run_margin_pretrain.sh, e.g.:  sft_chain r2-hfn-d0 HEAD=hfn MARGIN_D=0

# ---- plot ------------------------------------------------------------------------------------------------
