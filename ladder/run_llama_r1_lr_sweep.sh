# LR sweep of llama3_r1 + Muon: plain sbatch commands, submitted by hand from the Megatron-LM root
# (uv environment activated). Logs: logs/<job-name>-<jobid>.out. Runs: $OUT_ROOT/runs/llama3_r1_muon_lr<LR>.
#
# Per LR: one stable job (35000 iters, ~9 h on 8 H200) and two decay branches (TPP20 from iter 5600,
# TPP50 from iter 14000; ~1 h each). A branch needs the stable checkpoint it starts from, so it waits
# for its stable job (--dependency=afterok). To submit a branch later by hand instead, drop the
# --dependency flag once the stable job has finished.
# The dataset index is already in $OUT_ROOT/data_cache (built by the debug run), so all stable jobs
# can be submitted at once.
mkdir -p logs

# ---- lr 5e-4 ----
J=$(CONFIG=llama3_r1 OPT=muon LR=5e-4 MODE=stable \
  sbatch --parsable --nodes=1 --exclude=htv02 --job-name=r1-muon-lr5e-4-main ladder/launch/slurm.sbatch); echo "lr 5e-4 stable: $J"
CONFIG=llama3_r1 OPT=muon LR=5e-4 MODE=branch TPP=20 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr5e-4-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r1 OPT=muon LR=5e-4 MODE=branch TPP=50 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr5e-4-tpp50 ladder/launch/slurm.sbatch

# ---- lr 7e-4 ----
J=$(CONFIG=llama3_r1 OPT=muon LR=7e-4 MODE=stable \
  sbatch --parsable --nodes=1 --exclude=htv02 --job-name=r1-muon-lr7e-4-main ladder/launch/slurm.sbatch); echo "lr 7e-4 stable: $J"
CONFIG=llama3_r1 OPT=muon LR=7e-4 MODE=branch TPP=20 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr7e-4-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r1 OPT=muon LR=7e-4 MODE=branch TPP=50 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr7e-4-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1e-3 ----
J=$(CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=stable \
  sbatch --parsable --nodes=1 --exclude=htv02 --job-name=r1-muon-lr1e-3-main ladder/launch/slurm.sbatch); echo "lr 1e-3 stable: $J"
CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr1e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr1e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 1.4e-3 ----
J=$(CONFIG=llama3_r1 OPT=muon LR=1.4e-3 MODE=stable \
  sbatch --parsable --nodes=1 --exclude=htv02 --job-name=r1-muon-lr1.4e-3-main ladder/launch/slurm.sbatch); echo "lr 1.4e-3 stable: $J"
CONFIG=llama3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 2e-3 ----
J=$(CONFIG=llama3_r1 OPT=muon LR=2e-3 MODE=stable \
  sbatch --parsable --nodes=1 --exclude=htv02 --job-name=r1-muon-lr2e-3-main ladder/launch/slurm.sbatch); echo "lr 2e-3 stable: $J"
CONFIG=llama3_r1 OPT=muon LR=2e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r1 OPT=muon LR=2e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --exclude=htv02 --dependency=afterok:$J --job-name=r1-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch
