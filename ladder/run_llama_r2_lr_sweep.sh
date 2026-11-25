#!/bin/bash
# Run from the Megatron-LM root with the uv environment active.
# Run the whole script or create logs/ and paste one LR block into the shell.
# Each stable job runs to iter 55000. TPP20 and TPP50 resume at 8800 and 22000.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use the shared sweep launcher for custom selections and previews, e.g.
#   UPTO=20 bash ladder/run_llama_r2_lr_sweep.sh
#   QOS=polaris_idle LRS=1.3e-3 bash ladder/run_llama_r2_lr_sweep.sh
#   DRY_RUN=1 bash ladder/run_llama_r2_lr_sweep.sh
if [[ -n "${LRS:-}${UPTO:-}${QOS:-}${EXCLUDE:-}${NODES:-}${STABLE_TIME:-}${BRANCH_TIME:-}" || ${DRY_RUN:-0} == 1 ]]; then
  export CONFIG=llama3_r2 OPT=muon LRS="${LRS:-4.5e-4 6e-4 1.4e-3 2e-3 2.8e-3}"
  exec bash ladder/sweep.sh
fi

mkdir -p logs

# ---- lr 4.5e-4 ----
# J=$(CONFIG=llama3_r2 OPT=muon LR=4.5e-4 MODE=stable \
#   sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr4.5e-4-main ladder/launch/slurm.sbatch)
# echo "lr 4.5e-4 stable: $J"
# CONFIG=llama3_r2 OPT=muon LR=4.5e-4 MODE=branch TPP=20 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr4.5e-4-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r2 OPT=muon LR=4.5e-4 MODE=branch TPP=50 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr4.5e-4-tpp50 ladder/launch/slurm.sbatch

# ---- lr 6e-4 ----
# J=$(CONFIG=llama3_r2 OPT=muon LR=6e-4 MODE=stable \
#   sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr6e-4-main ladder/launch/slurm.sbatch)
# echo "lr 6e-4 stable: $J"
# CONFIG=llama3_r2 OPT=muon LR=6e-4 MODE=branch TPP=20 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr6e-4-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r2 OPT=muon LR=6e-4 MODE=branch TPP=50 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr6e-4-tpp50 ladder/launch/slurm.sbatch

# The R1 sweep (2026-11-23) found 2e-3 best at TPP 20/50/100, unbracketed, so the rest of the R2 grid
# uses the R1 grid points 1.4e-3 / 2e-3 / 2.8e-3 (same values as R1 for a direct comparison).

# ---- lr 1.4e-3 ----
# J=$(CONFIG=llama3_r2 OPT=muon LR=1.4e-3 MODE=stable \
#   sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr1.4e-3-main ladder/launch/slurm.sbatch)
# echo "lr 1.4e-3 stable: $J"
# CONFIG=llama3_r2 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r2 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2e-3 ----
J=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 MODE=stable \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr2e-3-main ladder/launch/slurm.sbatch)
echo "lr 2e-3 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=2e-3 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=2e-3 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2.8e-3 ----
# J=$(CONFIG=llama3_r2 OPT=muon LR=2.8e-3 MODE=stable \
#   sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr2.8e-3-main ladder/launch/slurm.sbatch)
# echo "lr 2.8e-3 stable: $J"
# CONFIG=llama3_r2 OPT=muon LR=2.8e-3 MODE=branch TPP=20 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr2.8e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r2 OPT=muon LR=2.8e-3 MODE=branch TPP=50 \
#   sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr2.8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 4e-3 ---- (2026-11-23: R1 optimum at TPP20 is >= 4e-3; R2's is estimated >= 2.7e-3)
# The stable run stops at the TPP50 decay start (iter 22000, STOP_TPP=50); TPP20 / TPP50 branches.
# If 2.8e-3 turns out best at TPP100, resubmit this stable job without STOP_TPP: it continues the same run
# to TPP100.
J=$(CONFIG=llama3_r2 OPT=muon LR=4e-3 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr4e-3-main ladder/launch/slurm.sbatch)
echo "lr 4e-3 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=4e-3 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=4e-3 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.4e-3 / 2e-3 on idle QoS ---- (copies of the blocks above with --qos=polaris_idle)
# Idle QoS has no 32-GPU cap but can be preempted; a requeued job resumes from the latest checkpoint.
# These write to the same run directory as the standard blocks above (llama3_r2_muon_lr<LR>), so run
# only one of the two per LR: scancel the standard jobs of that LR first, or paste only these blocks.

# ---- lr 1.4e-3 (idle) ----
# J=$(CONFIG=llama3_r2 OPT=muon LR=1.4e-3 MODE=stable \
#   sbatch --parsable --qos=polaris_idle --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr1.4e-3-main ladder/launch/slurm.sbatch)
# echo "lr 1.4e-3 stable (idle): $J"
# CONFIG=llama3_r2 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
#   sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r2 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
#   sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
#   --job-name=llama3_r2-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2e-3 (idle) ----
J=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 MODE=stable \
  sbatch --parsable --qos=polaris_idle --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr2e-3-main ladder/launch/slurm.sbatch)
echo "lr 2e-3 stable (idle): $J"
CONFIG=llama3_r2 OPT=muon LR=2e-3 MODE=branch TPP=20 \
  sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=2e-3 MODE=branch TPP=50 \
  sbatch --qos=polaris_idle --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1e-3 ---- (2026-11-25: fills the 6e-4 -> 1.4e-3 gap on the low side of lr* ~1.6-2.1e-3, and
# makes the R2 grid near the optimum 1e-3 / 1.4e-3 / 2e-3 / 2.8e-3 / 4e-3, the same points as R1, so both
# rungs are fitted on the same LRs and window.) Full stable run to TPP100: the low side matters most there.
# The branches wait for the whole stable run (afterok). To start them earlier, submit them without
# --dependency once the stable run has saved iteration 8800 (TPP20) / 22000 (TPP50).
J=$(CONFIG=llama3_r2 OPT=muon LR=1e-3 MODE=stable \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr1e-3-main ladder/launch/slurm.sbatch)
echo "lr 1e-3 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=1e-3 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr1e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=1e-3 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr1e-3-tpp50 ladder/launch/slurm.sbatch

# ---- fit the sweep ----
# Reads every logs/*.out written so far (unfinished runs are listed, not fitted), so rerun it whenever
# more jobs finish. Writes logs/lr_fit/lr_fit_llama3_r2_muon.png and logs/lr_fit/lr_points_llama3_r2_muon.csv.
# --window 2, the same as R1. It needs 2e-3 (and 1e-3) to be finished: until then TPP50 / TPP100 have
# only 1.4e-3 and 2.8e-3 within x2 of the best LR and are not fitted. Use --window 3 for an interim look.
# Needs numpy, scipy and matplotlib in the active environment.
python ladder/analysis/lr_fit.py --logs 'logs/*.out' --config llama3_r2 --opt muon \
  --window 2.0 --noise 1e-3 --out-dir logs/lr_fit
