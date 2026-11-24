#!/bin/bash
# Run from the Megatron-LM root with the uv environment active.
# Run the whole script or create logs/ and paste one LR block into the shell.
# Each stable job runs to iter 60000 (TPP 101, 94B tokens). TPP20 and TPP50 resume at 9600 and 24000.
# Estimated from R2 (645 ms/iter on 16 H200) scaled by FLOPs/token (x1.57): ~0.76 s/iter on 32 H200,
# so a stable job takes ~13 h, a TPP20 branch ~0.5 h, a TPP50 branch ~1.3 h, and a STOP_TPP=50 stable ~5 h.
#
# One R3 job is 4 nodes = 32 GPUs, which is the whole standard-QoS per-user cap. A second standard job
# (including any R2 job) waits until the first one finishes. So only the centre LR runs on standard
# QoS and the others run on idle QoS (preemptible; a requeued job resumes from its latest checkpoint).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use the shared sweep launcher for custom selections and previews, e.g.
#   UPTO=20 bash ladder/run_llama_r3_lr_sweep.sh
#   QOS=polaris_idle LRS=1.4e-3 bash ladder/run_llama_r3_lr_sweep.sh
#   DRY_RUN=1 bash ladder/run_llama_r3_lr_sweep.sh
if [[ -n "${LRS:-}${UPTO:-}${QOS:-}${EXCLUDE:-}${NODES:-}${STABLE_TIME:-}${BRANCH_TIME:-}" || ${DRY_RUN:-0} == 1 ]]; then
  export CONFIG=llama3_r3 OPT=muon LRS="${LRS:-1.4e-3 2e-3 2.8e-3 4e-3}"
  exec bash ladder/sweep.sh
fi

mkdir -p logs

# Grid (2026-11-24). R1 is bracketed: lr* = 3.4e-3 / 2.9e-3 / 2.6e-3 at TPP 20 / 50 / 100, joint fit
# eta*_50 = 2.9e-3, gamma = -0.17. R2 so far only shows that lr* >= 1.4e-3 (best = largest grid point).
# DeepSeek-LLM eta_opt ~ C^-0.125 at fixed TPP gives R3/R1 = (933/377)^(-0.25) = 0.80, i.e. ~2.3e-3 at
# TPP50. The grid uses the R1/R2 points 1.4e-3 / 2e-3 / 2.8e-3 / 4e-3 so that the rungs compare directly.
# If R2 finishes with lr* clearly below 2e-3, add 1e-3. If 2.8e-3 wins at TPP50, run 4e-3 to TPP100.

# ---- lr 2e-3 (standard QoS) ----
J=$(CONFIG=llama3_r3 OPT=muon LR=2e-3 MODE=stable \
  sbatch --parsable --nodes=4 --time=72:00:00 --job-name=llama3_r3-muon-lr2e-3-main ladder/launch/slurm.sbatch)
echo "lr 2e-3 stable: $J"
CONFIG=llama3_r3 OPT=muon LR=2e-3 MODE=branch TPP=20 \
  sbatch --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r3 OPT=muon LR=2e-3 MODE=branch TPP=50 \
  sbatch --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.4e-3 (idle QoS) ----
J=$(CONFIG=llama3_r3 OPT=muon LR=1.4e-3 MODE=stable \
  sbatch --parsable --qos=polaris_idle --nodes=4 --time=72:00:00 --job-name=llama3_r3-muon-lr1.4e-3-main ladder/launch/slurm.sbatch)
echo "lr 1.4e-3 stable (idle): $J"
CONFIG=llama3_r3 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r3 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2.8e-3 (idle QoS) ----
J=$(CONFIG=llama3_r3 OPT=muon LR=2.8e-3 MODE=stable \
  sbatch --parsable --qos=polaris_idle --nodes=4 --time=72:00:00 --job-name=llama3_r3-muon-lr2.8e-3-main ladder/launch/slurm.sbatch)
echo "lr 2.8e-3 stable (idle): $J"
CONFIG=llama3_r3 OPT=muon LR=2.8e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr2.8e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r3 OPT=muon LR=2.8e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr2.8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 4e-3 (idle QoS) ----
# The stable run stops at the TPP50 decay start (iter 24000, STOP_TPP=50); TPP20 / TPP50 branches.
# If 2.8e-3 turns out best at TPP50, resubmit this stable job without STOP_TPP: it continues the same run
# to TPP100.
J=$(CONFIG=llama3_r3 OPT=muon LR=4e-3 MODE=stable STOP_TPP=50 \
  sbatch --parsable --qos=polaris_idle --nodes=4 --time=72:00:00 --job-name=llama3_r3-muon-lr4e-3-main ladder/launch/slurm.sbatch)
echo "lr 4e-3 stable (idle): $J"
CONFIG=llama3_r3 OPT=muon LR=4e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r3 OPT=muon LR=4e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=4 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r3-muon-lr4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- fit the sweep ----
# Run this once jobs have finished. It reads every logs/*.out written so far; unfinished runs are listed,
# not fitted. Writes logs/lr_fit/lr_fit_llama3_r3_muon.png and logs/lr_fit/lr_points_llama3_r3_muon.csv.
# Keep --window 2.0, the same as R1/R2: the window shifts lr* by 10-15%.
# Needs numpy, scipy and matplotlib in the active environment.
# python ladder/analysis/lr_fit.py --logs 'logs/*.out' --config llama3_r3 --opt muon \
#   --window 2.0 --noise 1e-3 --out-dir logs/lr_fit
