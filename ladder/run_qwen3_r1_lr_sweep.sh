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
  export CONFIG=qwen3_r1 OPT=muon LRS="${LRS:-1.4e-3 2e-3 2.8e-3 4e-3 5.6e-3}"
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
J=$(CONFIG=qwen3_r1 OPT=muon LR=2.8e-3 MODE=stable \
  sbatch --parsable --nodes=1 --time=72:00:00 --job-name=qwen3_r1-muon-lr2.8e-3-main ladder/launch/slurm.sbatch)
echo "lr 2.8e-3 stable: $J"
CONFIG=qwen3_r1 OPT=muon LR=2.8e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr2.8e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r1 OPT=muon LR=2.8e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr2.8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2e-3 (standard QoS) ----
J=$(CONFIG=qwen3_r1 OPT=muon LR=2e-3 MODE=stable \
  sbatch --parsable --nodes=1 --time=72:00:00 \
  --job-name=qwen3_r1-muon-lr2e-3-main ladder/launch/slurm.sbatch)
echo "lr 2e-3 stable: $J"
CONFIG=qwen3_r1 OPT=muon LR=2e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r1 OPT=muon LR=2e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 4e-3 (standard QoS) ---- full run to TPP100: Llama3-R1's 4e-3 lost to 2.8e-3 there, which is
# what brackets lr*_100 from above.
J=$(CONFIG=qwen3_r1 OPT=muon LR=4e-3 MODE=stable \
  sbatch --parsable --nodes=1 --time=72:00:00 \
  --job-name=qwen3_r1-muon-lr4e-3-main ladder/launch/slurm.sbatch)
echo "lr 4e-3 stable: $J"
CONFIG=qwen3_r1 OPT=muon LR=4e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r1 OPT=muon LR=4e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.4e-3 (idle QoS) ---- full run to TPP100: the low side, in case depth pulls lr* down.
J=$(CONFIG=qwen3_r1 OPT=muon LR=1.4e-3 MODE=stable \
  sbatch --parsable --qos=polaris_idle --nodes=1 --time=72:00:00 \
  --job-name=qwen3_r1-muon-lr1.4e-3-main ladder/launch/slurm.sbatch)
echo "lr 1.4e-3 stable (idle): $J"
CONFIG=qwen3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=qwen3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=1 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=qwen3_r1-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 5.6e-3 (idle QoS) ---- the stable run stops at the TPP50 decay start (iter 15000, STOP_TPP=50);
# TPP20 / TPP50 branches. Brackets lr*_20 from above (Llama3-R1: 3.4e-3; 5.6e-3 lost at TPP20 and TPP50).
# If 4e-3 turns out best at TPP100, resubmit this stable job without STOP_TPP: it continues the same run
# to TPP100. The AdamW part (embedding, lm_head) runs at this LR too: watch grad norm and loss spikes.
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

# ---- fit the sweep ----
# Run this once jobs have finished. It reads every logs/*.out written so far; unfinished runs are listed,
# not fitted. Writes logs/lr_fit/lr_fit_qwen3_r1_muon.png and logs/lr_fit/lr_points_qwen3_r1_muon.csv.
# Keep --window 2.0, the same as the Llama3 rungs: the window shifts lr* by 10-15%.
# Needs numpy, scipy and matplotlib in the active environment.
python ladder/analysis/lr_fit.py --logs 'logs/*.out' --config qwen3_r1 --opt muon \
  --window 2.0 --noise 1e-3 --out-dir logs/lr_fit
