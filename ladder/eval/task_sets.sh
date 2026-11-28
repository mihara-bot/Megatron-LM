# lm-eval task presets, shared by ladder/eval/lm_eval.sh and ladder/eval/check_datasets.sh. Sourced from the repo root;
# reads TASK_SET (default small) and TASKS (comma-separated, overrides the preset). Sets TASKS (no spaces) and
# TASK_SET_NAME (the preset, or "custom" when TASKS was given). Each preset adds to the previous one:
#   small     12 log-likelihood tasks, 0-shot, all with signal at 0.4-8B / TPP 20-100
#   core      + mmlu race wikitext xstorycloze_en truthfulqa_mc2 mmlu_continuation logiqa ceval-valid cmmlu
#   extended  + agieval_en agieval_cn leaderboard_mmlu_pro leaderboard_bbh leaderboard_gpqa leaderboard_musr triviaqa
#               nq_open gsm8k   (some generate)
#   all       + gsm8k_cot minerva_math bbh_cot_fewshot mmlu_pro drop humaneval mbpp   (long generations; humaneval /
#               mbpp run the generated code, minerva_math needs lm-eval[math])
SMALL_TASKS=hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,sciq,boolq,lambada_openai,copa,social_iqa,commonsense_qa
CORE_TASKS=$SMALL_TASKS,mmlu,race,wikitext,xstorycloze_en,truthfulqa_mc2,mmlu_continuation,logiqa,ceval-valid,cmmlu
EXTENDED_TASKS=$CORE_TASKS,agieval_en,agieval_cn,leaderboard_mmlu_pro,leaderboard_bbh,leaderboard_gpqa,leaderboard_musr,triviaqa,nq_open,gsm8k
ALL_TASKS=$EXTENDED_TASKS,gsm8k_cot,minerva_math,bbh_cot_fewshot,mmlu_pro,drop,humaneval,mbpp

TASK_SET=${TASK_SET:-small}
case "$TASK_SET" in
    small)    SET_TASKS=$SMALL_TASKS ;;
    core)     SET_TASKS=$CORE_TASKS ;;
    extended) SET_TASKS=$EXTENDED_TASKS ;;
    all)      SET_TASKS=$ALL_TASKS ;;
    *) echo "TASK_SET must be small, core, extended or all, got $TASK_SET" >&2; exit 1 ;;
esac
if [ -n "${TASKS:-}" ]; then
    TASKS=${TASKS// /}
    TASK_SET_NAME=custom
else
    TASKS=$SET_TASKS
    TASK_SET_NAME=$TASK_SET
fi
