#!/bin/bash
# Offline GSM8K / MATH-500 / AMC-2023 / Countdown pass@k of Hugging Face model directories with vLLM
# (ladder/sft/eval/eval_vllm.py; settings in ladder/sft/eval/math_passk.yaml: the pipeline's tasks, prompts and
# graders, 64 samples + greedy per problem, all problems). Any HF causal LM works: the exports of
# ladder/sft/export.sh, or a reference model. One GPU, an environment with vLLM:
#   MODEL="$OUT_ROOT/sft/<sft run>/hf/iter_*" VLLM_PYTHON=<vllm env>/bin/python \
#     sbatch -J sft-eval -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
#     --mem=128G -t 24:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh'
#
# Required:
#   MODEL        one or more HF directories, space-separated; globs are expanded when the job runs, so a
#                pattern such as ".../hf/iter_*" picks up every export present by then (in iteration order)
# Optional:
#   EVAL_DIR     output directory, for a single MODEL. Default: <X>/eval/<name> for a model at <X>/hf/<name>
#                (the layout export.sh writes), else <parent>/eval/<name>; plus a suffix when TASKS /
#                NUM_SAMPLES / MAX_PROBLEMS are set, so a quick subset never overwrites the full results
#   DATA_ROOT    eval data root (default /fast/users/zhuangxinlin/data)
#   VLLM_PYTHON  python of the vLLM environment (default: python)
#   TASKS, NUM_SAMPLES, MAX_PROBLEMS   override math_passk.yaml
#   FORCE=1      re-evaluate a model whose summary.json exists; DRY_RUN=1 print the commands only
# Output: <eval dir>/{summary.json,<task>.jsonl}, and one line per model in <eval dir>/../summary.jsonl.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
: "${MODEL:?set MODEL to one or more Hugging Face model directories (ladder/sft/export.sh writes them)}"
DATA_ROOT=${DATA_ROOT:-/fast/users/zhuangxinlin/data}
VLLM_PYTHON=${VLLM_PYTHON:-python}

EVAL_ARGS=()
SUFFIX=""
if [ -n "${TASKS:-}" ]; then
    read -r -a T <<< "$TASKS"
    EVAL_ARGS+=(--tasks "${T[@]}"); SUFFIX+="_${TASKS// /_}"
fi
if [ -n "${NUM_SAMPLES:-}" ]; then EVAL_ARGS+=(--num-samples "$NUM_SAMPLES"); SUFFIX+="_n${NUM_SAMPLES}"; fi
if [ -n "${MAX_PROBLEMS:-}" ]; then EVAL_ARGS+=(--max-problems "$MAX_PROBLEMS"); SUFFIX+="_p${MAX_PROBLEMS}"; fi

# Expand globs now (inside the job), drop the trailing slash, order iterations numerically.
MODELS=()
for m in $MODEL; do
    if [ ! -d "$m" ]; then
        echo "MODEL entry $m is not a directory (not exported yet? see ladder/sft/export.sh)" >&2
        [ "${DRY_RUN:-0}" = 1 ] || exit 1
    fi
    MODELS+=("${m%/}")
done
SORTED=()
while IFS= read -r m; do
    if [ -n "$m" ]; then SORTED+=("$m"); fi
done < <(printf '%s\n' ${MODELS[@]+"${MODELS[@]}"} | sort -V)
MODELS=(${SORTED[@]+"${SORTED[@]}"})
if [ -n "${EVAL_DIR:-}" ] && [ "${#MODELS[@]}" -ne 1 ]; then
    echo "EVAL_DIR names one output directory; MODEL has ${#MODELS[@]} entries" >&2; exit 1
fi

eval_dir() {   # default output directory of a model directory
    local m=$1 parent name
    parent=$(dirname "$m"); name=$(basename "$m")
    if [ "$(basename "$parent")" = hf ]; then
        echo "$(dirname "$parent")/eval/${name}${SUFFIX}"
    else
        echo "$parent/eval/${name}${SUFFIX}"
    fi
}

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
for m in ${MODELS[@]+"${MODELS[@]}"}; do
    out=${EVAL_DIR:-$(eval_dir "$m")}
    if [ -f "$out/summary.json" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "$m: $out/summary.json exists, skipped (FORCE=1 redoes it)"; continue
    fi
    cmd=("$VLLM_PYTHON" ladder/sft/eval/eval_vllm.py --model "$m" --data-root "$DATA_ROOT" --output-dir "$out"
         ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"})
    echo "+ ${cmd[*]}"
    if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi
    "${cmd[@]}"
    python3 - "$out/summary.json" "$(dirname "$out")/summary.jsonl" <<'EOF'
import json, os, sys
summary, dest = sys.argv[1], sys.argv[2]
s = json.load(open(summary))
model = s["model"]
meta = os.path.join(model, "export_meta.json")
iteration = json.load(open(meta)).get("iteration") if os.path.exists(meta) else None
row = {"model": model, "name": os.path.basename(model), "iteration": iteration, "output": os.path.dirname(summary),
       **s["metrics"], **{k: s[k] for k in ("num_samples", "greedy", "max_problems", "tasks")}}
with open(dest, "a") as f:
    f.write(json.dumps(row) + "\n")
EOF
done
