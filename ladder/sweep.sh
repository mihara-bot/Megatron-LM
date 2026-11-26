#!/bin/bash
# Submit an LR sweep of one rung on SLURM: for every LR, one stable job and one job per decay branch.
# Branch jobs wait for their stable job (afterok), since they start from its checkpoints.
#
# Run from the repo root (on shared storage) with the environment train.sh needs; sbatch exports it:
#   CONFIG=llama3_r1 bash ladder/sweep.sh                       # LR_GRID of the config, all TPPs
#   CONFIG=llama3_r2 LRS="6e-4 9e-4" OPT=muon bash ladder/sweep.sh
#   CONFIG=llama3_r1 UPTO=20 bash ladder/sweep.sh               # cheap screen: each LR only to TPP 20
#   CONFIG=llama3_r1 LRS=1e-3 bash ladder/sweep.sh              # later: finish the best LR to TPP_MAX
#
# Environment:
#   CONFIG   ladder/configs/<CONFIG>.sh (reads NODES, LR_GRID, TPP_MAX, BRANCH_TPPS)
#   LRS      LRs to run (default: the config's LR_GRID)
#   OPT      adamw (default) | muon
#   HEAD, MARGIN_D, MARGIN_REVERSE, HEAD_SCALE, SFN_WEIGHT
#            LM head + token margin (ladder/configs/head.sh), passed to every job and added to its name
#   UPTO     last TPP to train (default TPP_MAX). Below TPP_MAX the stable job stops at the decay start of
#            UPTO, and only the branches up to UPTO are submitted. Rerunning with a larger UPTO continues
#            the same stable run (same run directory, same data order) and adds the missing branches.
#   NODES    override the config's node count
#   QOS      e.g. polaris_idle (default: the account's standard QoS, max 32 GPUs at once)
#   EXCLUDE  nodes to keep the jobs off, sbatch --exclude syntax (e.g. htv02 or htv[02,05])
#   STABLE_TIME / BRANCH_TIME   sbatch --time for the two job kinds (default 72:00:00 / 24:00:00)
#   DRY_RUN=1  print the sbatch commands only
# Every job is ladder/launch/slurm.sbatch -> ladder/train.sh, which exits at once when a run has already
# reached its target, so resubmitting a sweep only runs what is missing.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
: "${CONFIG:?set CONFIG, e.g. CONFIG=llama3_r1}"
CFG=ladder/configs/${CONFIG}.sh
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 1; }
# Read the schedule variables without leaking the config's defaults (LR, MBS) into the jobs.
read -r C_NODES C_TPP_MAX C_BRANCHES C_GRID < <(
    unset LR MBS; source "$CFG"
    printf '%s %s %s %s\n' "$NODES" "$TPP_MAX" "${BRANCH_TPPS// /,}" "${LR_GRID// /,}")
NODES=${NODES:-$C_NODES}
LRS=${LRS:-${C_GRID//,/ }}
OPT=${OPT:-adamw}
source ladder/configs/head.sh   # validates the head settings and sets HEAD_TAG
UPTO=${UPTO:-$C_TPP_MAX}
STABLE_TIME=${STABLE_TIME:-72:00:00}
BRANCH_TIME=${BRANCH_TIME:-24:00:00}

BRANCHES=()
for t in ${C_BRANCHES//,/ }; do
    if [ "$t" -le "$UPTO" ]; then BRANCHES+=("$t"); fi
done
if [ "$UPTO" != "$C_TPP_MAX" ]; then
    case " ${BRANCHES[*]-} " in *" $UPTO "*) ;; *)
        echo "UPTO=$UPTO must be TPP_MAX ($C_TPP_MAX) or one of BRANCH_TPPS (${C_BRANCHES//,/ })" >&2; exit 1 ;;
    esac
fi

SBATCH=(sbatch --parsable --nodes="$NODES")
if [ -n "${QOS:-}" ]; then SBATCH+=(--qos="$QOS"); fi
if [ -n "${EXCLUDE:-}" ]; then SBATCH+=(--exclude="$EXCLUDE"); fi
if [ "${DRY_RUN:-0}" != 1 ]; then mkdir -p logs; fi   # slurm.sbatch writes logs/%x-%j.out

submit() {   # submit <time> <job name> <dependency or ""> VAR=value ...
    local time=$1 name=$2 dep=$3; shift 3
    local cmd=("${SBATCH[@]}" --time="$time" --job-name="$name" --export="ALL,$(IFS=,; echo "$*")")
    if [ -n "$dep" ]; then cmd+=(--dependency="afterok:$dep"); fi
    cmd+=(ladder/launch/slurm.sbatch)
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "${cmd[*]}" >&2; echo "dry:$name"   # stands in for the job id
    else
        "${cmd[@]}"
    fi
}

echo "== $CONFIG + $OPT${HEAD_TAG}: LRs [$LRS] up to TPP $UPTO (TPP_MAX $C_TPP_MAX), branches [${BRANCHES[*]-}], $NODES node(s) per job"
for lr in $LRS; do
    common=(CONFIG="$CONFIG" OPT="$OPT" LR="$lr" HEAD="$HEAD" MARGIN_D="$MARGIN_D"
            MARGIN_REVERSE="$MARGIN_REVERSE" HEAD_SCALE="$HEAD_SCALE" SFN_WEIGHT="$SFN_WEIGHT")
    stop=()
    if [ "$UPTO" != "$C_TPP_MAX" ]; then stop=(STOP_TPP="$UPTO"); fi
    jid=$(submit "$STABLE_TIME" "${CONFIG}-${OPT}-lr${lr}${HEAD_TAG}-main" "" "${common[@]}" MODE=stable ${stop[@]+"${stop[@]}"})
    echo "   lr $lr: stable job $jid"
    for t in ${BRANCHES[@]+"${BRANCHES[@]}"}; do
        bid=$(submit "$BRANCH_TIME" "${CONFIG}-${OPT}-lr${lr}${HEAD_TAG}-tpp${t}" "$jid" "${common[@]}" MODE=branch TPP="$t")
        echo "            TPP$t branch job $bid (after $jid)"
    done
done
