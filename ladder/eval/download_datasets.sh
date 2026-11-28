#!/bin/bash
# Downloads, with hfd.sh, the Hugging Face datasets of the lm-eval task sets of ladder/eval/lm_eval.sh (TASK_SET
# small, core, extended, all), plus the code_eval metric of humaneval / mbpp, so the evaluation runs on nodes that
# cannot reach the Hub. Moved from compass-merge (launchers/download_lm_eval_datasets.sh); a login node is enough:
#
#   bash ladder/eval/download_datasets.sh                  # every set
#   TASK_SET=small bash ladder/eval/download_datasets.sh   # only what the default set reads
#   DRY_RUN=1 bash ladder/eval/download_datasets.sh        # print the hfd.sh calls
#
# Each dataset repo is stored as a plain repo directory EVAL_DATA_DIR/<org>/<name>.
# datasets.load_dataset("<org>/<name>") takes a directory at that path relative to the working directory before
# the Hub (evaluate.load("code_eval") likewise takes code_eval/code_eval.py), so ladder/eval/lm_eval.sh runs lm-eval
# offline from EVAL_DATA_DIR once it exists (same default path in both scripts). Loading scripts (*.py) are skipped:
# datasets >= 4 rejects them, and offline they could not fetch their data anyway. Repos that need more than a plain
# download:
#   allenai/social_i_qa, EleutherAI/logiqa  script-only; their refs/convert/parquet branch is downloaded and a
#                                           README with its configs is written
#   mandarjoshi/trivia_qa                   35 GB; only the rc.nocontext config triviaqa reads (64 MB) is
#                                           downloaded, with a README for it alone
#   haonan-li/cmmlu                         script + zip of CSVs (the repo redirects to lmlmcat/cmmlu); the zip is
#                                           extracted and a README with one config per subject is written
#   Idavidrein/gpqa                         gated: pass --hf_username U --hf_token T
# Re-running resumes unfinished downloads (hfd.sh keeps its state in <repo>/.hfd) and rewrites the generated READMEs.
#
# Repos by the task set that first needs them (TASK_SET downloads its own and those of the sets before it):
#   small     Rowan/hellaswag allenai/ai2_arc baber/piqa allenai/winogrande allenai/openbookqa allenai/sciq
#             aps/super_glue (boolq, copa) EleutherAI/lambada_openai allenai/social_i_qa tau/commonsense_qa
#   core      cais/mmlu EleutherAI/race EleutherAI/wikitext_document_level juletxara/xstory_cloze
#             truthfulqa/truthful_qa EleutherAI/logiqa ceval/ceval-exam haonan-li/cmmlu
#             (mmlu_continuation reads cais/mmlu)
#   extended  hails/agieval-* (21 repos) TIGER-Lab/MMLU-Pro SaylorTwift/bbh Idavidrein/gpqa TAUR-Lab/MuSR
#             mandarjoshi/trivia_qa google-research-datasets/nq_open openai/gsm8k
#   all       EleutherAI/hendrycks_math EleutherAI/drop openai/openai_humaneval google-research-datasets/mbpp,
#             and the code_eval metric (EVAL_METRICS)
#
# Overrides (defaults in parentheses):
#   TASK_SET (all), EVAL_DATA_DIR (/fast/users/zhuangxinlin/data/lm_eval_hf), HFD (hfd.sh next to this repo)
#   HF_ENDPOINT (https://hf-mirror.com), HFD_TOOL (aria2c if installed, else wget)
#   EVAL_DATASETS (the TASK_SET repos above), EVAL_METRICS (code_eval for TASK_SET=all; "" skips it)
#   PYTHON (python3, stdlib only; extracts cmmlu and writes the READMEs)
#   Trailing CLI arguments go to every hfd.sh call, such as -x 8 or --hf_username U --hf_token T.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT_DIR="$(cd "$REPO_DIR/.." && pwd)"

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/fast/users/zhuangxinlin/data/lm_eval_hf}
HFD=${HFD:-$PARENT_DIR/hfd.sh}
PYTHON=${PYTHON:-python3}
TASK_SET=${TASK_SET:-all}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
if [ -z "${HFD_TOOL:-}" ]; then
    if command -v aria2c > /dev/null; then HFD_TOOL=aria2c; else HFD_TOOL=wget; fi
fi

AGIEVAL=""
for name in aqua-rat gaokao-biology gaokao-chemistry gaokao-chinese gaokao-english \
        gaokao-geography gaokao-history gaokao-mathcloze gaokao-mathqa gaokao-physics jec-qa-ca \
        jec-qa-kd logiqa-en logiqa-zh lsat-ar lsat-lr lsat-rc math sat-en \
        sat-en-without-passage sat-math; do
    AGIEVAL+=" hails/agieval-$name"
done
SMALL_DATASETS="Rowan/hellaswag allenai/ai2_arc baber/piqa allenai/winogrande allenai/openbookqa allenai/sciq aps/super_glue EleutherAI/lambada_openai allenai/social_i_qa tau/commonsense_qa"
CORE_DATASETS="cais/mmlu EleutherAI/race EleutherAI/wikitext_document_level juletxara/xstory_cloze truthfulqa/truthful_qa EleutherAI/logiqa ceval/ceval-exam haonan-li/cmmlu"
EXTENDED_DATASETS="$AGIEVAL TIGER-Lab/MMLU-Pro SaylorTwift/bbh Idavidrein/gpqa TAUR-Lab/MuSR mandarjoshi/trivia_qa google-research-datasets/nq_open openai/gsm8k"
ALL_DATASETS="EleutherAI/hendrycks_math EleutherAI/drop openai/openai_humaneval google-research-datasets/mbpp"
case "$TASK_SET" in
    small)    SET_DATASETS="$SMALL_DATASETS"; SET_METRICS="" ;;
    core)     SET_DATASETS="$SMALL_DATASETS $CORE_DATASETS"; SET_METRICS="" ;;
    extended) SET_DATASETS="$SMALL_DATASETS $CORE_DATASETS $EXTENDED_DATASETS"; SET_METRICS="" ;;
    all)      SET_DATASETS="$SMALL_DATASETS $CORE_DATASETS $EXTENDED_DATASETS $ALL_DATASETS"; SET_METRICS=code_eval ;;
    *) echo "TASK_SET must be small, core, extended or all, got $TASK_SET" >&2; exit 1 ;;
esac
EVAL_DATASETS=${EVAL_DATASETS:-$SET_DATASETS}
EVAL_METRICS=${EVAL_METRICS-$SET_METRICS}

if [ ! -f "$HFD" ]; then
    echo "hfd.sh not found at $HFD (set HFD)" >&2
    exit 1
fi
echo "Downloading TASK_SET=$TASK_SET to ${EVAL_DATA_DIR} from ${HF_ENDPOINT} with ${HFD_TOOL}"

# Writes <repo>/README.md with the configs of a downloaded layout, for repos whose own README
# does not describe the files on disk:
#   dirs   <config>/<split>/*.parquet (refs/convert/parquet) or <config>/<split>-*.parquet
#   cmmlu  extracts cmmlu_v1_0_1.zip into test/<subject>.csv and dev/<subject>.csv
write_readme() {
    "$PYTHON" - "$1" "$2" <<'EOF'
import os, sys, zipfile
root, layout = sys.argv[1], sys.argv[2]
configs = []
if layout == "dirs":
    for config in sorted(os.listdir(root)):
        path = os.path.join(root, config)
        if config.startswith(".") or not os.path.isdir(path):
            continue
        splits = []
        for entry in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, entry)):
                splits.append((entry, f"{config}/{entry}/*.parquet"))
            elif entry.endswith(".parquet"):
                split = entry.split("-", 1)[0]
                if split not in [s for s, _ in splits]:
                    splits.append((split, f"{config}/{split}-*.parquet"))
        configs.append((config, splits))
elif layout == "cmmlu":
    with zipfile.ZipFile(os.path.join(root, "cmmlu_v1_0_1.zip")) as archive:
        archive.extractall(root)
    for name in sorted(os.listdir(os.path.join(root, "test"))):
        if not name.endswith(".csv"):  # the zip also carries a stray test/.zip
            continue
        subject = name[: -len(".csv")]
        configs.append((subject, [("test", f"test/{name}"), ("dev", f"dev/{name}")]))
if not configs:
    sys.exit(f"no configs found under {root}")
lines = ["---", "configs:"]
for config, splits in configs:
    lines += [f"- config_name: {config}", "  data_files:"]
    for split, pattern in splits:
        lines += [f"  - split: {split}", f"    path: {pattern}"]
lines.append("---")
with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(f"{root}/README.md: {len(configs)} configs")
EOF
}

FAILED=()
for repo in $EVAL_DATASETS; do
    case "$repo" in
        allenai/social_i_qa|EleutherAI/logiqa)
            HFD_ARGS=(--revision "refs%2Fconvert%2Fparquet"); LAYOUT=dirs ;;
        mandarjoshi/trivia_qa)
            HFD_ARGS=(--include "rc.nocontext/*"); LAYOUT=dirs ;;
        haonan-li/cmmlu)
            HFD_ARGS=(--include "*.zip"); LAYOUT=cmmlu ;;
        *)
            HFD_ARGS=(--exclude "*.py"); LAYOUT="" ;;
    esac
    CMD=(bash "$HFD" "$repo" --dataset --local-dir "$EVAL_DATA_DIR/$repo" --tool "$HFD_TOOL"
         "${HFD_ARGS[@]}" "$@")
    echo ">>> [$repo] -> $EVAL_DATA_DIR/$repo${LAYOUT:+ (README from the $LAYOUT layout)}"
    if [ "${DRY_RUN:-0}" = 1 ]; then
        printf '%q ' "${CMD[@]}"
        echo
        continue
    fi
    mkdir -p "$EVAL_DATA_DIR/$repo"
    if ! "${CMD[@]}"; then
        FAILED+=("$repo")
        continue
    fi
    if [ -n "$LAYOUT" ] && ! write_readme "$EVAL_DATA_DIR/$repo" "$LAYOUT"; then
        FAILED+=("$repo")
    fi
done

# evaluate.load(<name>) fetches the metric script from the Hub space evaluate-metric/<name>.
for metric in $EVAL_METRICS; do
    case "$metric" in
        code_eval) FILES="code_eval.py execute.py" ;;
        *) echo "[$metric] unknown metric" >&2; FAILED+=("$metric"); continue ;;
    esac
    echo ">>> [metric $metric] -> $EVAL_DATA_DIR/$metric"
    for file in $FILES; do
        url="$HF_ENDPOINT/spaces/evaluate-metric/$metric/resolve/main/$file"
        if [ "${DRY_RUN:-0}" = 1 ]; then
            echo "curl -fsSL -o $EVAL_DATA_DIR/$metric/$file $url"
            continue
        fi
        mkdir -p "$EVAL_DATA_DIR/$metric"
        curl -fsSL --retry 3 -o "$EVAL_DATA_DIR/$metric/$file" "$url" || { FAILED+=("$metric"); break; }
    done
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Download failed for: ${FAILED[*]} (re-run to resume)" >&2
    exit 1
fi
echo "lm-eval datasets ready: ${EVAL_DATA_DIR}"
