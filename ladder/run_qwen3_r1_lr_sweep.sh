#!/bin/bash
# LR sweep of qwen3_r1 + Muon. Run from the Megatron-LM root with the uv environment active.
# Run the whole script or create logs/ and paste one LR block into the shell.
# Logs: logs/<job-name>-<jobid>.out. Runs: $OUT_ROOT/runs/qwen3_r1_muon_lr<LR>.
#
# Data: the launchers pick the Qwen3 tokenizer and $DATA_ROOT=.../megatron_data/qwen3/climbmix_v1_m1 from
# the config name. That data has to be tokenized with the Qwen3 base tokenizer first:
#   FAMILY=qwen3 bash ladder/data/prepare_dataset.sh
#
# Each stable job runs to iter 37500 (TPP 99, 39.3B tokens). TPP20 and TPP50 branches resume at 6000 and
# 15000 and stop at 7500 / 18750. Time estimate: Llama3-R1 ran at 882 ms/iter on 8 H200; Qwen3-R1 has
# 1.40x its FLOPs/token, so ~1.25 s/iter if MFU holds (d=768 may run at a lower MFU): a stable job takes
# ~13 h, a STOP_TPP=50 stable ~5 h, a TPP20 branch ~0.5 h, a TPP50 branch ~1.3 h.
# If MBS=8 runs out of memory (28 layers, attention width 2048), prefix the commands with MBS=4: the
# global batch stays 256, only the grad-accum steps change.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use the shared sweep launcher for custom selections and previews, e.g.
#   UPTO=20 bash ladder/run_qwen3_r1_lr_sweep.sh
#   QOS=polaris_idle LRS=2e-3 bash ladder/run_qwen3_r1_lr_sweep.sh
#   DRY_RUN=1 bash ladder/run_qwen3_r1_lr_sweep.sh
if [[ -n "${LRS:-}${UPTO:-}${QOS:-}${EXCLUDE:-}${NODES:-}${STABLE_TIME:-}${BRANCH_TIME:-}" || ${DRY_RUN:-0} == 1 ]]; then
  export CONFIG=qwen3_r1 OPT=muon LRS="${LRS:-1e-3 1.4e-3 2e-3 2.8e-3 4e-3 5.6e-3 8e-3}"
  exec bash ladder/sweep.sh
fi

mkdir -p logs

# Grid (2026-11-25), taken from the Llama3-R1 Muon sweep. N_eff differs by 5% (0.397B vs 0.377B).
# Llama3-R1 is bracketed at lr* = 3.4e-3 / 2.9e-3 / 2.7e-3 (TPP 20 / 50 / 100), and 2.8e-3 has regret
# <= 0.002 at every TPP. The grid uses the same points as Llama3-R1 (sqrt(2) apart, centred on 2.8e-3)
# so both families are fitted on the same LRs and the same --window.
# Qwen3-R1 is not the same network: 28 layers instead of 18, d 768 instead of 1024, attention width
# 2048 = 2.7d, QK-norm, and a lm_head that is a smaller share of N_eff. QK-norm usually tolerates a higher
# LR and the extra depth usually wants a lower one, so the grid keeps a point on each side of Llama3's
# optimum at every TPP: 1.4e-3 / 2e-3 below and 4e-3 above at TPP100, and 5.6e-3 above at TPP20/50.
#   standard QoS (not preempted, cap 32 GPUs): 2e-3, 2.8e-3, 4e-3 (1 node each)
#   idle QoS (preemptible, resumes from the latest checkpoint): 1.4e-3, 5.6e-3
#
# Dependencies, as in the Llama3 sweeps: within each LR block of 3 jobs, the TPP20 / TPP50 branches wait
# for their own stable job (afterok); the LR blocks are independent of each other.
# All LRs of this rung share one dataset index in $OUT_ROOT/data_cache, built by the first job that gets
# there. Megatron writes it without a lock, so two jobs reaching dataset building in the same few
# seconds can make one of them fail while loading the index (a numpy load error). Its branches then
# never start (DependencyNeverSatisfied): scancel them and resubmit that LR block.

# ---- lr 2.8e-3 (standard QoS) ----
# J=$(CONFIG=qwen3_r1 OPT=muon LR=2.8e-3 MODE=stable \
#   sbatch --parsable --nodes=1 --time=72:00:00 --job-name=qwen3_r1-muon-lr2.8e-3-main ladder/launch/slurm.sbatch)
# echo "lr 2.8e-3 stable: $J"
# CONFIG=qwen3_r1 OPT=muon LR=2.8e-3 MODE=branch TPP=20 \
#   sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr2.8e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r1 OPT=muon LR=2.8e-3 MODE=branch TPP=50 \
#   sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr2.8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2e-3 (standard QoS) ----
# J=$(CONFIG=qwen3_r1 OPT=muon LR=2e-3 MODE=stable \
#   sbatch --parsable --nodes=1 --time=72:00:00 \
#   --job-name=qwen3_r1-muon-lr2e-3-main ladder/launch/slurm.sbatch)
# echo "lr 2e-3 stable: $J"
# CONFIG=qwen3_r1 OPT=muon LR=2e-3 MODE=branch TPP=20 \
#   sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r1 OPT=muon LR=2e-3 MODE=branch TPP=50 \
#   sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 4e-3 (standard QoS) ---- full run to TPP100: Llama3-R1's 4e-3 lost to 2.8e-3 there, which is
# what brackets lr*_100 from above.
# J=$(CONFIG=qwen3_r1 OPT=muon LR=4e-3 MODE=stable \
#   sbatch --parsable --nodes=1 --time=72:00:00 \
#   --job-name=qwen3_r1-muon-lr4e-3-main ladder/launch/slurm.sbatch)
# echo "lr 4e-3 stable: $J"
# CONFIG=qwen3_r1 OPT=muon LR=4e-3 MODE=branch TPP=20 \
#   sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr4e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r1 OPT=muon LR=4e-3 MODE=branch TPP=50 \
#   sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.4e-3 (idle QoS) ---- full run to TPP100: the low side, in case depth pulls lr* down.
# Submitted 2026-11-26 (stable job 4749, running): do not resubmit while it runs, two jobs on one run
# directory overwrite each other's checkpoints.
# J=$(CONFIG=qwen3_r1 OPT=muon LR=1.4e-3 MODE=stable \
#   sbatch --parsable --qos=polaris_idle --nodes=1 --time=72:00:00 \
#   --job-name=qwen3_r1-muon-lr1.4e-3-main ladder/launch/slurm.sbatch)
# echo "lr 1.4e-3 stable (idle): $J"
# CONFIG=qwen3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
#   sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=qwen3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
#   sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=qwen3_r1-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- 2026-11-26 results (final val loss, TPP 20 / 50 / 100; TPP100 is the final re-eval) ----
#   2e-3    2.7582 / 2.6585 / 2.6016
#   2.8e-3  2.7550 / 2.6582 / 2.6035
#   4e-3    2.7529 / 2.6577 / 2.6052
# TPP20: 4e-3 best, not bracketed. The steps 2e-3 -> 2.8e-3 -> 4e-3 (-3.2e-3, -2.1e-3) repeat those of
# Llama3-R1 on the same points (-3.4e-3, -1.9e-3), whose optimum came out at 3.4e-3 once 5.6e-3 was in, so
# the 3-point extrapolation (5.7e-3 here) is not to be trusted. TPP50: flat, 0.75e-3 across the three.
# TPP100: the lowest LR is best, not bracketed below. With Llama3-like curvature lr*_100 is ~1.7-2.5e-3,
# so 1.4e-3 (running) is expected to tie or lose; 1e-3 is the insurance below it.
# The horizon trend is steeper than Llama3-R1's. A single LR of 2.8e-3 still has regret <= 2.1e-3 at
# every TPP, the same as in Llama3-R1.
# Round 2, all on idle QoS (R2's 1.4e-3 / 2e-3 take the 32-GPU standard cap): 5.6e-3 (STOP_TPP=50, below),
# 1e-3 (full run), 8e-3 (STOP_TPP=50). About 16 h for the 1e-3 stable job, 6.5 h for each STOP_TPP=50 one.

# ---- lr 5.6e-3 (idle QoS) ---- the stable run stops at the TPP50 decay start (iter 15000, STOP_TPP=50);
# TPP20 / TPP50 branches. Brackets lr*_20 from above (Llama3-R1: 3.4e-3; 5.6e-3 lost at TPP20 and TPP50).
# 4e-3 is the worst LR at TPP100, so this run never needs to go past TPP50.
# The AdamW part (embedding, lm_head) runs at this LR too: watch grad norm and loss spikes.
J=$(CONFIG=qwen3_r1 OPT=muon LR=5.6e-3 MODE=stable STOP_TPP=50 \
  sbatch --parsable --qos=polaris_idle --nodes=1 --time=72:00:00 \
  --job-name=qwen3_r1-muon-lr5.6e-3-main ladder/launch/slurm.sbatch)
echo "lr 5.6e-3 stable (idle): $J"
CONFIG=qwen3_r1 OPT=muon LR=5.6e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr5.6e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r1 OPT=muon LR=5.6e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr5.6e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1e-3 (idle QoS) ---- full run to TPP100: brackets lr*_100 from below if 1.4e-3 beats 2e-3.
# Its TPP20 / TPP50 branches lie outside --window 2 of those optima. They are there to complete the grid
# shared with Llama3-R1, which has 1e-3 at every TPP, and cost ~2 h on 1 node.
J=$(CONFIG=qwen3_r1 OPT=muon LR=1e-3 MODE=stable \
  sbatch --parsable --qos=polaris_idle --nodes=1 --time=72:00:00 \
  --job-name=qwen3_r1-muon-lr1e-3-main ladder/launch/slurm.sbatch)
echo "lr 1e-3 stable (idle): $J"
CONFIG=qwen3_r1 OPT=muon LR=1e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr1e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r1 OPT=muon LR=1e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr1e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 8e-3 (idle QoS) ---- the stable run stops at the TPP50 decay start (iter 15000, STOP_TPP=50);
# TPP20 / TPP50 branches. Brackets TPP20 / TPP50 from above if 5.6e-3 still wins there. This is 1.4x the
# highest LR tried so far in either family. If the loss diverges or grad-norm spikes keep coming, scancel
# it and take 5.6e-3 as the upper edge.
J=$(CONFIG=qwen3_r1 OPT=muon LR=8e-3 MODE=stable STOP_TPP=50 \
  sbatch --parsable --qos=polaris_idle --nodes=1 --time=72:00:00 \
  --job-name=qwen3_r1-muon-lr8e-3-main ladder/launch/slurm.sbatch)
echo "lr 8e-3 stable (idle): $J"
CONFIG=qwen3_r1 OPT=muon LR=8e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr8e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r1 OPT=muon LR=8e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- follow-ups (not submitted) ----
# 1e-3 best at TPP100 -> add 7e-4 (full run): QOS=polaris_idle LRS=7e-4 bash ladder/run_qwen3_r1_lr_sweep.sh

# ---- fit the sweep ----
# Run this once jobs have finished. It reads every logs/*.out written so far; unfinished runs are listed,
# not fitted. Writes logs/lr_fit/lr_fit_qwen3_r1_muon.png and logs/lr_fit/lr_points_qwen3_r1_muon.csv.
# Keep --window 2.0, the same as the Llama3 rungs: the window shifts lr* by 10-15%.
# Needs numpy, scipy and matplotlib in the active environment.
python ladder/analysis/lr_fit.py --logs 'logs/*.out' --config qwen3_r1 --opt muon \
  --window 2.0 --noise 1e-3 --out-dir logs/lr_fit
