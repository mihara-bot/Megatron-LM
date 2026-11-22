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
  export CONFIG=llama3_r2 OPT=muon LRS="${LRS:-4.5e-4 6e-4 9e-4 1.3e-3 1.8e-3}"
  exec bash ladder/sweep.sh
fi

mkdir -p logs

# ---- lr 4.5e-4 ----
J=$(CONFIG=llama3_r2 OPT=muon LR=4.5e-4 MODE=stable \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr4.5e-4-main ladder/launch/slurm.sbatch)
echo "lr 4.5e-4 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=4.5e-4 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr4.5e-4-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=4.5e-4 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr4.5e-4-tpp50 ladder/launch/slurm.sbatch

# ---- lr 6e-4 ----
J=$(CONFIG=llama3_r2 OPT=muon LR=6e-4 MODE=stable \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr6e-4-main ladder/launch/slurm.sbatch)
echo "lr 6e-4 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=6e-4 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr6e-4-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=6e-4 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr6e-4-tpp50 ladder/launch/slurm.sbatch

# ---- lr 9e-4 ----
J=$(CONFIG=llama3_r2 OPT=muon LR=9e-4 MODE=stable \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr9e-4-main ladder/launch/slurm.sbatch)
echo "lr 9e-4 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=9e-4 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr9e-4-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=9e-4 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr9e-4-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.3e-3 ----
J=$(CONFIG=llama3_r2 OPT=muon LR=1.3e-3 MODE=stable \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr1.3e-3-main ladder/launch/slurm.sbatch)
echo "lr 1.3e-3 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=1.3e-3 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr1.3e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=1.3e-3 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr1.3e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.8e-3 ----
J=$(CONFIG=llama3_r2 OPT=muon LR=1.8e-3 MODE=stable \
  sbatch --parsable --nodes=2 --time=72:00:00 --job-name=llama3_r2-muon-lr1.8e-3-main ladder/launch/slurm.sbatch)
echo "lr 1.8e-3 stable: $J"
CONFIG=llama3_r2 OPT=muon LR=1.8e-3 MODE=branch TPP=20 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr1.8e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r2 OPT=muon LR=1.8e-3 MODE=branch TPP=50 \
  sbatch --nodes=2 --time=24:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr1.8e-3-tpp50 ladder/launch/slurm.sbatch
