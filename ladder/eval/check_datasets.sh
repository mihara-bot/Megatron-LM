#!/bin/bash
# Check that the lm-eval datasets of a task set are complete under EVAL_DATA_DIR (ladder/eval/check_datasets.py).
# The installed lm-eval decides which repo each task reads, so run this after download_datasets.sh and after every
# lm-eval update. No GPU needed (a login node is fine); the vLLM environment, from the Megatron-LM root:
#   VLLM_PYTHON=/fast/users/zhuangxinlin/envs/vllm/bin/python bash ladder/eval/check_datasets.sh
#   TASK_SET=core VLLM_PYTHON=... bash ladder/eval/check_datasets.sh
# It builds every task as lm-eval does, offline from EVAL_DATA_DIR, and prints per task:
#   OK    loaded
#   MISS  its repo is not downloaded
#   FAIL  another error, with a hint
#   PART  hfd.sh did not finish the repo
# The last lines give the download_datasets.sh command for what is missing or unfinished. Exit status 1 unless
# everything is OK.
# Variables: TASK_SET / TASKS (ladder/eval/task_sets.sh), EVAL_DATA_DIR (/fast/users/zhuangxinlin/data/lm_eval_hf),
#   VLLM_PYTHON (default python), DRY_RUN=1 (print the command)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
source ladder/eval/task_sets.sh            # -> TASKS, TASK_SET_NAME
VLLM_PYTHON=${VLLM_PYTHON:-python}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-/fast/users/zhuangxinlin/data/lm_eval_hf}

cmd=("$VLLM_PYTHON" "$REPO/ladder/eval/check_datasets.py" --tasks "$TASKS" --data-dir "$EVAL_DATA_DIR")
echo "== check $TASK_SET_NAME tasks in $EVAL_DATA_DIR"
echo "+ ${cmd[*]}"
if [ "${DRY_RUN:-0}" = 1 ]; then exit 0; fi
if [ ! -d "$EVAL_DATA_DIR" ]; then
    echo "$EVAL_DATA_DIR does not exist: run ladder/eval/download_datasets.sh first" >&2; exit 1
fi
# From EVAL_DATA_DIR with the Hub offline, as lm_eval.sh runs lm-eval; without the Megatron shell's Python paths.
cd "$EVAL_DATA_DIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1 PYTHONSAFEPATH=1
exec bash "$REPO/ladder/sft/eval/run_isolated.sh" "${cmd[@]}"
