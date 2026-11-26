#!/bin/bash
# SFT of one ladder run on OpenMathInstruct-2, then HF export and offline eval (ladder/SFT.md).
# Data first: ladder/sft/download_data.sh. Run from the Megatron-LM root with the Megatron uv environment active.
# `bash ladder/sft/run_sft.sh` submits every step, chained with afterok; to run one step, paste the
# "run" block and that step's block (without the previous job id there is no dependency).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

# ---- run ------------------------------------------------------------------------------------------------
DATA=/fast/users/zhuangxinlin/data
export OUT_ROOT=/fast/users/zhuangxinlin/ckpts
# the pretraining run SFT starts from, as ladder/train.sh names it (+ its checkpoint directory)
export CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 PRETRAIN_TAG=tpp100
VLLM_PYTHON=/path/to/vllm-env/bin/python    # TODO: python of an environment with vLLM
source ladder/sft/common.sh                  # -> PRETRAIN_RUN, SFT_DIR (the names train_sft.sh / export.sh use)
SBATCH_1GPU=(sbatch --parsable -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1
             --cpus-per-task=16 --mem=128G -o logs/%x-%j.out)

# ---- tests and smoke (run once before the real jobs; not submitted by this file) --------------------------
# CPU, Megatron env: Megatron's data path on prepared data vs the pipeline's encoding, launchers, eval logic
#   python -m pytest ladder/tests/test_sft_megatron_data.py ladder/tests/test_sft_scripts.py -p no:cacheprovider -o addopts=
# 1 node: ~125 SFT steps from this run's checkpoint -> export -> quick vLLM eval -> checks (log ends in SMOKE OK)
#   VLLM_PYTHON=$VLLM_PYTHON sbatch -J sft-smoke -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=8 \
#     --cpus-per-task=128 --mem=512G -t 03:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/sft/smoke_sft.sh'

# ---- train: 1 node (8 GPUs), Megatron env ----------------------------------------------------------------
# -> $SFT_DIR/ckpt/iter_<N>, one checkpoint per epoch
TRAIN_JOB=$(SFT_DATA_DIR=$DATA/sft_megatron/omi2_500k_ep10_llama3 ENTRY=ladder/sft/train_sft.sh \
  sbatch --parsable --nodes=1 -J sft-r1-hfn ladder/launch/slurm.sbatch)
echo "train   job $TRAIN_JOB: $SFT_DIR"

# ---- export: 1 GPU, Megatron env -------------------------------------------------------------------------
# Megatron checkpoint -> HF directory (head folded, Megatron vs HF logits checked)
#   SFT, every epoch:       $SFT_DIR/hf/iter_<N>
#   pre-SFT reference:      $OUT_ROOT/runs/$PRETRAIN_RUN/hf/${PRETRAIN_TAG}_iter<N>
EXPORT_JOB=$(ITERS=all "${SBATCH_1GPU[@]}" -J sft-export -t 04:00:00 ${TRAIN_JOB:+--dependency=afterok:$TRAIN_JOB} \
  --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh')
EXPORT_PRE_JOB=$(TARGET=pretrained "${SBATCH_1GPU[@]}" -J sft-export-pre -t 01:00:00 \
  --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh')
echo "export  jobs $EXPORT_JOB (SFT, after $TRAIN_JOB), $EXPORT_PRE_JOB (pretrained)"

# ---- eval: 1 GPU, vLLM env -------------------------------------------------------------------------------
# MODEL = HF directories (globs expand inside the job); results in <X>/eval/<name>/ for a model at <X>/hf/<name>,
# one line per model in <X>/eval/summary.jsonl. Quick check first: TASKS=gsm8k NUM_SAMPLES=4 MAX_PROBLEMS=50.
EVAL_JOB=$(MODEL="$SFT_DIR/hf/iter_*" VLLM_PYTHON=$VLLM_PYTHON DATA_ROOT=$DATA \
  "${SBATCH_1GPU[@]}" -J sft-eval -t 24:00:00 ${EXPORT_JOB:+--dependency=afterok:$EXPORT_JOB} \
  --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh')
EVAL_PRE_JOB=$(MODEL="$OUT_ROOT/runs/$PRETRAIN_RUN/hf/${PRETRAIN_TAG}_iter*" VLLM_PYTHON=$VLLM_PYTHON DATA_ROOT=$DATA \
  "${SBATCH_1GPU[@]}" -J sft-eval-pre -t 04:00:00 ${EXPORT_PRE_JOB:+--dependency=afterok:$EXPORT_PRE_JOB} \
  --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh')
echo "eval    jobs $EVAL_JOB (SFT), $EVAL_PRE_JOB (pretrained)"

# ---- plot ------------------------------------------------------------------------------------------------
