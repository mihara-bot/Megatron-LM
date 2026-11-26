#!/bin/bash
# LR sweep of qwen3_r2 + Muon. Run from the Megatron-LM root with the uv environment active.
# Run the whole script or create logs/ and paste one LR block into the shell.
# Logs: logs/<job-name>-<jobid>.out. Runs: $OUT_ROOT/runs/qwen3_r2_muon_lr<LR>.
#
# Data: the launchers pick the Qwen3 tokenizer and $DATA_ROOT=.../megatron_data/qwen3/climbmix_v1_m1 from
# the config name. That data has to be tokenized with the Qwen3 base tokenizer first:
#   FAMILY=qwen3 bash ladder/data/prepare_dataset.sh
#
# Each stable job runs to iter 55000 (TPP 97, 57.7B tokens; the same iterations as Llama3-R2). TPP20 and
# TPP50 branches resume at 8800 and 22000 and stop at 11000 / 27500. Time estimate: Llama3-R2 ran at
# 645 ms/iter on 16 H200; Qwen3-R2 has 1.20x its FLOPs/token, so ~0.78 s/iter: a stable job takes ~12 h,
# a STOP_TPP=50 stable ~5 h, a TPP20 branch ~0.5 h, a TPP50 branch ~1.2 h.
# If MBS=8 runs out of memory, prefix the commands with MBS=4 (global batch unchanged).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use the shared sweep launcher for custom selections and previews, e.g.
#   UPTO=20 bash ladder/run_qwen3_r2_lr_sweep.sh
#   QOS=polaris_idle LRS=1.4e-3 bash ladder/run_qwen3_r2_lr_sweep.sh
#   DRY_RUN=1 bash ladder/run_qwen3_r2_lr_sweep.sh
if [[ -n "${LRS:-}${UPTO:-}${QOS:-}${EXCLUDE:-}${NODES:-}${STABLE_TIME:-}${BRANCH_TIME:-}" || ${DRY_RUN:-0} == 1 ]]; then
  export CONFIG=qwen3_r2 OPT=muon LRS="${LRS:-7e-4 1e-3 1.4e-3 2e-3 2.8e-3 4e-3}"
  exec bash ladder/sweep.sh
fi

mkdir -p logs

# Grid (2026-11-25), taken from the Llama3-R2 Muon sweep. N_eff differs by 1.5% (0.596B vs 0.587B).
# Llama3-R2: lr* = 2.1e-3 / 1.8e-3 / 1.6e-3 (TPP 20 / 50 / 100), robust to the point subset (<5%);
# R2/R1 lr* ratio ~0.62 at every TPP. The grid uses the Llama3-R2 points near that optimum,
# 1e-3 / 1.4e-3 / 2e-3 / 2.8e-3 (sqrt(2) apart), so both families are fitted on the same LRs:
#   TPP100: 1e-3 / 1.4e-3 below, 2e-3 above lr* ~1.6e-3
#   TPP20/50: 2.8e-3 above lr* ~1.8-2.1e-3
# Qwen3-R2 has 28 layers instead of 20, d 1024 instead of 1280, attention width 2048, and QK-norm, so its
# optimum can move by one grid step either way; the follow-ups below say which point to add.
# One R2 job is 2 nodes, so two of them fill the 32-GPU standard-QoS cap (shared with the R1 sweep):
#   standard QoS: 1.4e-3, 2e-3 (closest to the expected optimum)
#   idle QoS (preemptible, resumes from the latest checkpoint): 1e-3, 2.8e-3
#
# Dependencies, as in the Llama3 sweeps: within each LR block of 3 jobs, the TPP20 / TPP50 branches wait
# for their own stable job (afterok); the LR blocks are independent of each other.
# All LRs of this rung share one dataset index in $OUT_ROOT/data_cache, built by the first job that gets
# there. Megatron writes it without a lock, so two jobs reaching dataset building in the same few
# seconds can make one of them fail while loading the index (a numpy load error). Its branches then
# never start (DependencyNeverSatisfied): scancel them and resubmit that LR block.
#
# 2026-11-26: logs/ has no log for 2e-3 or 1.4e-3 yet. Check `squeue --me` before running this script and
# comment out any of the two blocks below that is already queued: train.sh has no run lock, and two jobs
# on one run directory overwrite each other's checkpoints.

# ---- lr 2e-3 (standard QoS) ----
# J=$(CONFIG=qwen3_r2 OPT=muon LR=2e-3 MODE=stable \
#   sbatch --parsable --nodes=2 --time=72:00:00 --job-name=qwen3_r2-muon-lr2e-3-main ladder/launch/slurm.sbatch)
# echo "lr 2e-3 stable: $J"
# CONFIG=qwen3_r2 OPT=muon LR=2e-3 MODE=branch TPP=20 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r2 OPT=muon LR=2e-3 MODE=branch TPP=50 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.4e-3 (standard QoS) ----
# J=$(CONFIG=qwen3_r2 OPT=muon LR=1.4e-3 MODE=stable \
#   sbatch --parsable --nodes=2 --time=72:00:00 \
#   --job-name=qwen3_r2-muon-lr1.4e-3-main ladder/launch/slurm.sbatch)
# echo "lr 1.4e-3 stable: $J"
# CONFIG=qwen3_r2 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r2 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1e-3 (idle QoS) ---- full run to TPP100: the low side matters most there.
# J=$(CONFIG=qwen3_r2 OPT=muon LR=1e-3 MODE=stable \
#   sbatch --parsable --qos=polaris_idle --nodes=2 --time=72:00:00 \
#   --job-name=qwen3_r2-muon-lr1e-3-main ladder/launch/slurm.sbatch)
# echo "lr 1e-3 stable (idle): $J"
# CONFIG=qwen3_r2 OPT=muon LR=1e-3 MODE=branch TPP=20 \
#   sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr1e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r2 OPT=muon LR=1e-3 MODE=branch TPP=50 \
#   sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr1e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2.8e-3 (idle QoS) ---- the stable run stops at the TPP50 decay start (iter 22000, STOP_TPP=50);
# TPP20 / TPP50 branches. If 2e-3 turns out best at TPP100, resubmit this stable job without STOP_TPP:
# it continues the same run to TPP100.
# Ran 2026-11-26 (jobs 4679-4681): stable to 22000 and TPP20 done; the TPP50 branch failed, rerun below.
# J=$(CONFIG=qwen3_r2 OPT=muon LR=2.8e-3 MODE=stable STOP_TPP=50 \
#   sbatch --parsable --qos=polaris_idle --nodes=2 --time=72:00:00 \
#   --job-name=qwen3_r2-muon-lr2.8e-3-main ladder/launch/slurm.sbatch)
# echo "lr 2.8e-3 stable (idle): $J"
# CONFIG=qwen3_r2 OPT=muon LR=2.8e-3 MODE=branch TPP=20 \
#   sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr2.8e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r2 OPT=muon LR=2.8e-3 MODE=branch TPP=50 \
#   sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r2-muon-lr2.8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- 2026-11-26 results (final val loss, TPP 20 / 50 / 100; TPP100 is the final re-eval) ----
#   1e-3    2.6536 / 2.5583 / 2.5037
#   2.8e-3  2.6418 / (TPP50 branch failed) / (stable stopped at 22000)
# TPP20: 2.8e-3 beats 1e-3 by 11.8e-3, twice Llama3-R2's gap on the same pair. With Llama3-R2's curvature
# that puts lr*_20 near 2.8e-3, so 4e-3 is needed to bracket it.
# TPP100: Qwen3-R1 prefers a lower LR than Llama3-R1 there (2e-3 best, against 2.8e-3). Scaled by the
# R2/R1 ratio of ~0.62, Qwen3-R2's lr*_100 may sit at ~1-1.5e-3: 7e-4 brackets it from below and 2.8e-3,
# continued to TPP100, from above.
# The 2.8e-3 TPP50 branch (job 4681) died at kernel warmup with CUDA OOM: the GPUs of htv00 / htv11
# already held ~86 + ~29 GiB from two processes outside this sweep. Its rerun below excludes both nodes.
# Round 2, all on idle QoS (1.4e-3 / 2e-3 above take the 32-GPU standard cap): 2.8e-3 to TPP100 + TPP50
# rerun, 7e-4 (full run), 4e-3 (STOP_TPP=50). Stable jobs: ~8 h / ~13-17 h / ~5.5 h.

# ---- lr 2.8e-3, continued to TPP100 (idle QoS) ----
# Without STOP_TPP the stable job resumes the same run from iteration 22000 and goes on to 55000.
# The TPP50 branch forks from the iteration-22000 checkpoint, which the stable run keeps (a multiple of
# 4400), so the rerun needs no dependency and runs alongside it. The TPP20 branch is done.
J=$(CONFIG=qwen3_r2 OPT=muon LR=2.8e-3 MODE=stable \
  sbatch --parsable --qos=polaris_idle --nodes=2 --time=72:00:00 \
  --job-name=qwen3_r2-muon-lr2.8e-3-main ladder/launch/slurm.sbatch)
echo "lr 2.8e-3 stable to TPP100 (idle): $J"
CONFIG=qwen3_r2 OPT=muon LR=2.8e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --exclude='htv[00,11]' \
  --job-name=qwen3_r2-muon-lr2.8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 7e-4 (idle QoS) ---- full run to TPP100: brackets lr*_100 from below. Its TPP20 / TPP50
# branches lie outside --window 2 of those optima. They are there to complete the grid and cost ~2 h on
# 2 nodes.
J=$(CONFIG=qwen3_r2 OPT=muon LR=7e-4 MODE=stable \
  sbatch --parsable --qos=polaris_idle --nodes=2 --time=72:00:00 \
  --job-name=qwen3_r2-muon-lr7e-4-main ladder/launch/slurm.sbatch)
echo "lr 7e-4 stable (idle): $J"
CONFIG=qwen3_r2 OPT=muon LR=7e-4 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r2-muon-lr7e-4-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r2 OPT=muon LR=7e-4 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r2-muon-lr7e-4-tpp50 ladder/launch/slurm.sbatch

# ---- lr 4e-3 (idle QoS) ---- the stable run stops at the TPP50 decay start (iter 22000, STOP_TPP=50);
# TPP20 / TPP50 branches. Brackets lr*_20, and lr*_50 if 2.8e-3 wins there, from above. It does not go to
# TPP100: that is 2.5-4x the expected lr*_100, outside --window 2. Resubmitting the stable job without
# STOP_TPP continues the same run if it is needed after all.
J=$(CONFIG=qwen3_r2 OPT=muon LR=4e-3 MODE=stable STOP_TPP=50 \
  sbatch --parsable --qos=polaris_idle --nodes=2 --time=72:00:00 \
  --job-name=qwen3_r2-muon-lr4e-3-main ladder/launch/slurm.sbatch)
echo "lr 4e-3 stable (idle): $J"
CONFIG=qwen3_r2 OPT=muon LR=4e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r2-muon-lr4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r2 OPT=muon LR=4e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r2-muon-lr4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- follow-ups (not submitted) ----
# 4e-3 best at TPP20  -> add 5.6e-3 to TPP20 only: QOS=polaris_idle LRS=5.6e-3 UPTO=20 bash ladder/run_qwen3_r2_lr_sweep.sh
# 7e-4 best at TPP100 -> add 5e-4 (full run).

# ---- fit the sweep ----
# Run this once jobs have finished. It reads every logs/*.out written so far; unfinished runs are listed,
# not fitted. Writes logs/lr_fit/lr_fit_qwen3_r2_muon.png and logs/lr_fit/lr_points_qwen3_r2_muon.csv.
# Keep --window 2.0, the same as the other rungs. Needs numpy, scipy and matplotlib.
python ladder/analysis/lr_fit.py --logs 'logs/*.out' --config qwen3_r2 --opt muon \
  --window 2.0 --noise 1e-3 --out-dir logs/lr_fit
