#!/bin/bash
# lm-evaluation-harness, vLLM backend, on Hugging Face model directories: the exports of ladder/eval/export.sh (or
# ladder/sft/export.sh), or any HF causal LM as a reference. One GPU, the vLLM environment
# (ladder/env/setup_vllm_env.sh) with lm-eval installed in it (ladder/EVAL.md):
#   MODEL="$OUT_ROOT/runs/<run>/hf/*_iter*" VLLM_PYTHON=<vllm env>/bin/python \
#     sbatch -J lm-eval -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
#     --mem=128G -t 04:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/eval/lm_eval.sh'
#
# Required:
#   MODEL        one or more HF directories, space-separated; globs are expanded when the job runs, so a pattern
#                such as ".../hf/*_iter*" picks up every export present by then (in version order)
# Optional:
#   TASK_SET     task preset, each one adding to the previous (ladder/eval/download_datasets.sh fetches their data):
#                  small (default)  hellaswag arc_easy arc_challenge piqa winogrande openbookqa sciq boolq
#                                   lambada_openai copa social_iqa commonsense_qa   (log-likelihood, 0-shot)
#                  core      + mmlu race wikitext xstorycloze_en truthfulqa_mc2 mmlu_continuation logiqa ceval-valid
#                              cmmlu   (log-likelihood; each task at its default shots, 0 for most)
#                  extended  + agieval_en agieval_cn leaderboard_mmlu_pro leaderboard_bbh leaderboard_gpqa
#                              leaderboard_musr triviaqa nq_open gsm8k   (some generate)
#                  all       + gsm8k_cot minerva_math bbh_cot_fewshot mmlu_pro drop humaneval mbpp (long
#                              generations; humaneval / mbpp run the generated code, minerva_math needs lm-eval[math])
#   TASKS        comma-separated lm-eval tasks, overrides TASK_SET
#   NUM_FEWSHOT  (the tasks' defaults), LIMIT (per-task cap, for smoke tests), LOG_SAMPLES=1 (per-sample records)
#   TOKENS       auto (default) | eos | default: token convention (ladder/eval/lm_eval_util.py). eos: no BOS, and
#                the EOS id of config.json as lm-eval's prefix token, as in ladder training data. auto: eos for a
#                ladder export (it has export_meta.json), lm-eval's default for other models
#   EVAL_DATA_DIR  datasets of ladder/eval/download_datasets.sh (default /fast/users/zhuangxinlin/data/lm_eval_hf);
#                when it exists, lm-eval runs from it with the Hub offline, else it reads the Hub (HF_ENDPOINT)
#   EVAL_ROOT    where results go (default <Megatron-LM>/logs/pretrain_evals): <EVAL_ROOT>/<run>/<name> for a model at
#                <run dir>/hf/<name> (the layout of the export scripts), else <EVAL_ROOT>/<parent dir name>/<name>; plus a
#                suffix when TASK_SET / TASKS / NUM_FEWSHOT / LIMIT / TOKENS differ from the defaults, so a subset never
#                overwrites the full results
#   EVAL_DIR     output directory, for a single MODEL (overrides the layout above)
#   VLLM_PYTHON  python of the vLLM environment (default: python); it runs through ladder/sft/eval/run_isolated.sh,
#                without the submitting shell's PYTHONPATH / package libraries
#   BATCH_SIZE (auto), GPU_MEM (gpu_memory_utilization, 0.85), SEED (1234, vLLM's seed),
#   MODEL_ARGS   appended to lm-eval's --model_args (comma-separated key=value, e.g. max_model_len=2048)
#   ALLOW_HF_MISMATCH=1   warn instead of failing when the vLLM environment's transformers / tokenizers differ from
#                the versions in the model's export_meta.json
#   FORCE=1      re-evaluate a model whose results.json exists; DRY_RUN=1 print the commands only
# Output: <eval dir>/results_<date>.json (lm-eval), results.json (copy of the latest), lm_eval_<stamp>.log,
# samples_*.jsonl (LOG_SAMPLES=1); one line per model in <eval dir>/../summary.jsonl (per run) and in
# <EVAL_ROOT>/summary.jsonl (every run): "score/<task>", "avg_score", every "<task>/<metric>", run / tag / iteration /
# LM head (ladder/eval/lm_eval_util.py collect).
# A model that fails does not stop the others; the script then exits 1.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
: "${MODEL:?set MODEL to one or more Hugging Face model directories (ladder/eval/export.sh writes them)}"
VLLM_PYTHON=${VLLM_PYTHON:-python}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-/fast/users/zhuangxinlin/data/lm_eval_hf}
EVAL_ROOT=${EVAL_ROOT:-$REPO/logs/pretrain_evals}
case "$EVAL_ROOT" in /*) ;; *) EVAL_ROOT=$REPO/$EVAL_ROOT ;; esac
TASK_SET=${TASK_SET:-small}
TOKENS=${TOKENS:-auto}
BATCH_SIZE=${BATCH_SIZE:-auto}
GPU_MEM=${GPU_MEM:-0.85}
SEED=${SEED:-1234}

SMALL_TASKS=hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,sciq,boolq,lambada_openai,copa,social_iqa,commonsense_qa
CORE_TASKS=$SMALL_TASKS,mmlu,race,wikitext,xstorycloze_en,truthfulqa_mc2,mmlu_continuation,logiqa,ceval-valid,cmmlu
EXTENDED_TASKS=$CORE_TASKS,agieval_en,agieval_cn,leaderboard_mmlu_pro,leaderboard_bbh,leaderboard_gpqa,leaderboard_musr,triviaqa,nq_open,gsm8k
ALL_TASKS=$EXTENDED_TASKS,gsm8k_cot,minerva_math,bbh_cot_fewshot,mmlu_pro,drop,humaneval,mbpp
case "$TASK_SET" in
    small)    SET_TASKS=$SMALL_TASKS ;;
    core)     SET_TASKS=$CORE_TASKS ;;
    extended) SET_TASKS=$EXTENDED_TASKS ;;
    all)      SET_TASKS=$ALL_TASKS ;;
    *) echo "TASK_SET must be small, core, extended or all, got $TASK_SET" >&2; exit 1 ;;
esac
case "$TOKENS" in auto | eos | default) ;; *) echo "TOKENS must be auto, eos or default, got $TOKENS" >&2; exit 1 ;; esac

SUFFIX=""
if [ -n "${TASKS:-}" ]; then
    TASKS=${TASKS// /}
    SUFFIX+="_${TASKS//,/_}"
    TASK_SET_NAME=custom
else
    TASKS=$SET_TASKS
    TASK_SET_NAME=$TASK_SET
    if [ "$TASK_SET" != small ]; then SUFFIX+="_$TASK_SET"; fi
fi
if [ -n "${NUM_FEWSHOT:-}" ]; then SUFFIX+="_${NUM_FEWSHOT}shot"; fi
if [ -n "${LIMIT:-}" ]; then SUFFIX+="_limit${LIMIT}"; fi
if [ "$TOKENS" != auto ]; then SUFFIX+="_tok${TOKENS}"; fi

LM_EVAL_FLAGS=(--batch_size "$BATCH_SIZE")
if [ -n "${NUM_FEWSHOT:-}" ]; then LM_EVAL_FLAGS+=(--num_fewshot "$NUM_FEWSHOT"); fi
if [ -n "${LIMIT:-}" ]; then LM_EVAL_FLAGS+=(--limit "$LIMIT"); fi
if [ "${LOG_SAMPLES:-0}" = 1 ]; then LM_EVAL_FLAGS+=(--log_samples); fi
if [[ ",$TASKS," =~ ,(humaneval|mbpp)[^,]*, ]]; then
    export HF_ALLOW_CODE_EVAL=1
    LM_EVAL_FLAGS+=(--confirm_run_unsafe_code)
fi
PREFLIGHT_FLAGS=()
if [ "${ALLOW_HF_MISMATCH:-0}" = 1 ]; then PREFLIGHT_FLAGS+=(--allow-hf-version-mismatch); fi

# Expand globs now (inside the job), make the paths absolute (lm-eval runs from EVAL_DATA_DIR), order by version.
# A pattern that matches nothing (a tag whose export was skipped) is only noted; a missing plain path is an error.
MODELS=()
for m in $MODEL; do
    if [ ! -d "$m" ]; then
        case "$m" in
        *[*?[]*)
            echo "MODEL pattern $m matches no directory (nothing exported there yet), skipped" >&2; continue ;;
        esac
        echo "MODEL entry $m is not a directory (not exported yet? see ladder/eval/export.sh)" >&2
        [ "${DRY_RUN:-0}" = 1 ] || exit 1
    else
        m=$(cd "$m" && pwd)
    fi
    MODELS+=("${m%/}")
done
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "MODEL ($MODEL) matches no directory (see ladder/eval/export.sh)" >&2; exit 1
fi
SORTED=()
while IFS= read -r m; do
    if [ -n "$m" ]; then SORTED+=("$m"); fi
done < <(printf '%s\n' ${MODELS[@]+"${MODELS[@]}"} | sort -V)
MODELS=(${SORTED[@]+"${SORTED[@]}"})
if [ -n "${EVAL_DIR:-}" ] && [ "${#MODELS[@]}" -ne 1 ]; then
    echo "EVAL_DIR names one output directory; MODEL has ${#MODELS[@]} entries" >&2; exit 1
fi
if [ -n "${EVAL_DIR:-}" ]; then
    case "$EVAL_DIR" in /*) ;; *) EVAL_DIR=$REPO/$EVAL_DIR ;; esac
fi

eval_dir() {   # default output directory of a model directory: <EVAL_ROOT>/<run>/<name><suffix>
    local m=$1 parent name group
    parent=$(dirname "$m"); name=$(basename "$m")
    if [ "$(basename "$parent")" = hf ]; then group=$(basename "$(dirname "$parent")"); else group=$(basename "$parent"); fi
    echo "$EVAL_ROOT/$group/${name}${SUFFIX}"
}
export_failed() {   # export_failed <model dir>: its export_meta.json records a failed Megatron vs HF logits check
    [ -f "$1/export_meta.json" ] && python3 -c '
import json, sys
v = json.load(open(sys.argv[1])).get("validation")
sys.exit(0 if v is not None and v.get("passed") is False else 1)' "$1/export_meta.json"
}

if [ -d "$EVAL_DATA_DIR" ]; then
    RUN_FROM=$EVAL_DATA_DIR
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
    echo "== datasets: $EVAL_DATA_DIR (offline)"
else
    RUN_FROM=$REPO
    echo "== datasets: ${HF_ENDPOINT:-https://huggingface.co} (no $EVAL_DATA_DIR; ladder/eval/download_datasets.sh)"
fi
echo "== tasks ($TASK_SET_NAME): $TASKS"
echo "== results: ${EVAL_DIR:-$EVAL_ROOT/<run>/<name>$SUFFIX}"

FAILED=()
STAMP=$(date +%Y%m%d-%H%M%S)
for m in ${MODELS[@]+"${MODELS[@]}"}; do
    out=${EVAL_DIR:-$(eval_dir "$m")}
    if [ -f "$out/results.json" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "$m: $out/results.json exists, skipped (FORCE=1 redoes it)"; continue
    fi
    if export_failed "$m"; then
        echo "$m: its export failed the Megatron vs HF logits check (export_meta.json); skipped" >&2
        FAILED+=("$m"); continue
    fi
    if ! conv=$(python3 ladder/eval/lm_eval_util.py model-args "$m" --tokens "$TOKENS"); then
        FAILED+=("$m"); continue
    fi
    read -r CONVENTION EXTRA <<< "$conv"
    margs="pretrained=$m,dtype=bfloat16,gpu_memory_utilization=$GPU_MEM,seed=$SEED${EXTRA:+,$EXTRA}"
    margs+="${MODEL_ARGS:+,$MODEL_ARGS}"
    cmd=("$VLLM_PYTHON" -m lm_eval run --model vllm --model_args "$margs" --tasks "$TASKS"
         --output_path "$out/results.json" "${LM_EVAL_FLAGS[@]}")
    echo "+ (tokens: $CONVENTION) ${cmd[*]}"
    if [ "${DRY_RUN:-0}" = 1 ]; then continue; fi
    mkdir -p "$out"
    rm -f "$out/results.json"
    if ! bash ladder/sft/eval/run_isolated.sh "$VLLM_PYTHON" ladder/eval/lm_eval_util.py preflight "$m" \
            ${PREFLIGHT_FLAGS[@]+"${PREFLIGHT_FLAGS[@]}"}; then
        FAILED+=("$m"); continue
    fi
    # Without the Megatron shell's PYTHONPATH / torch libraries; from EVAL_DATA_DIR, where load_dataset("<org>/<name>")
    # finds the local copy (a path relative to the working directory). PYTHONSAFEPATH keeps that directory off
    # sys.path, so a dataset directory such as openai/ cannot shadow a package.
    if ! (cd "$RUN_FROM" && PYTHONSAFEPATH=1 bash "$REPO/ladder/sft/eval/run_isolated.sh" "${cmd[@]}") 2>&1 \
            | tee "$out/lm_eval_${STAMP}.log"; then
        echo "$m: lm-eval failed (log: $out/lm_eval_${STAMP}.log)" >&2
        FAILED+=("$m"); continue
    fi
    if ! python3 ladder/eval/lm_eval_util.py collect --model "$m" --out "$out" --tasks "$TASKS" \
            --summary "$(dirname "$out")/summary.jsonl" --summary "$EVAL_ROOT/summary.jsonl" \
            --task-set "$TASK_SET_NAME" --num-fewshot "${NUM_FEWSHOT:-}" --limit "${LIMIT:-}" --tokens "$CONVENTION"; then
        FAILED+=("$m")
    fi
done
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "lm-eval failed for: ${FAILED[*]}" >&2
    exit 1
fi
