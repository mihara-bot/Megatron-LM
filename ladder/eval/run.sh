#!/bin/bash
# HF export + lm-eval of one pretraining run: llama3_r1_muon_lr2.8e-3 (plain head). From the Megatron-LM root with
# the Megatron venv active:
#   bash ladder/eval/run.sh            # smoke: arc_easy + hellaswag, 200 examples each
#   FULL=1 bash ladder/eval/run.sh     # the small task set (12 tasks), every example
# The export job writes <run>/hf/<tag>_iter<N> for each checkpoint directory whose LR has decayed to 0 (main at
# TPP 100, tpp20, tpp50; unfinished ones are skipped); the eval job starts after it (afterok) on all of them.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

RUN_DIR=/fast/users/zhuangxinlin/ckpts/runs/llama3_r1_muon_lr2.8e-3
SBATCH_1GPU=(sbatch --parsable -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16
             --mem=128G -o logs/%x-%j.out)

# Jobs inherit this shell's environment: nothing left over from another run may pick a different one.
unset RUN TAGS ITERS HEAD_SCALE SFN_WEIGHT EVAL_DIR TOKENS NUM_FEWSHOT TASK_SET MODEL_ARGS

# export (1 GPU, Megatron env): plain head -> the run directory above
EXPORT=$(CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=plain MARGIN_D=0 MARGIN_REVERSE=0 \
    "${SBATCH_1GPU[@]}" -J pre-export-r1-plain -t 02:00:00 \
    --wrap 'srun --ntasks=1 ht-run bash ladder/eval/export.sh')
echo "export job $EXPORT"

# eval (1 GPU, vLLM env), after the export
export MODEL="$RUN_DIR/hf/*_iter*" VLLM_PYTHON=/fast/users/zhuangxinlin/envs/vllm/bin/python
if [ "${FULL:-0}" = 1 ]; then
    unset TASKS LIMIT
    TIME=04:00:00
else
    export TASKS=arc_easy,hellaswag LIMIT=200
    TIME=01:00:00
fi
EVAL=$("${SBATCH_1GPU[@]}" -J lm-eval-r1-plain -t "$TIME" --dependency=afterok:"$EXPORT" \
    --wrap 'srun --ntasks=1 ht-run bash ladder/eval/lm_eval.sh')
echo "eval job $EVAL (after $EXPORT): results in $RUN_DIR/lm_eval/"
