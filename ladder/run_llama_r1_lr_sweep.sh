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
# J=$(CONFIG=llama3_r1 OPT=muon LR=5e-4 MODE=stable \
#   sbatch --parsable --nodes=1 --job-name=r1-muon-lr5e-4-main ladder/launch/slurm.sbatch); echo "lr 5e-4 stable: $J"
# CONFIG=llama3_r1 OPT=muon LR=5e-4 MODE=branch TPP=20 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr5e-4-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r1 OPT=muon LR=5e-4 MODE=branch TPP=50 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr5e-4-tpp50 ladder/launch/slurm.sbatch

# # ---- lr 7e-4 ----
# J=$(CONFIG=llama3_r1 OPT=muon LR=7e-4 MODE=stable \
#   sbatch --parsable --nodes=1 --job-name=r1-muon-lr7e-4-main ladder/launch/slurm.sbatch); echo "lr 7e-4 stable: $J"
# CONFIG=llama3_r1 OPT=muon LR=7e-4 MODE=branch TPP=20 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr7e-4-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r1 OPT=muon LR=7e-4 MODE=branch TPP=50 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr7e-4-tpp50 ladder/launch/slurm.sbatch

# # ---- lr 1e-3 ----
# J=$(CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=stable \
#   sbatch --parsable --nodes=1 --job-name=r1-muon-lr1e-3-main ladder/launch/slurm.sbatch); echo "lr 1e-3 stable: $J"
# CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=branch TPP=20 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr1e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=branch TPP=50 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr1e-3-tpp50 ladder/launch/slurm.sbatch

# # ---- lr 1.4e-3 ----
# J=$(CONFIG=llama3_r1 OPT=muon LR=1.4e-3 MODE=stable \
#   sbatch --parsable --nodes=1 --job-name=r1-muon-lr1.4e-3-main ladder/launch/slurm.sbatch); echo "lr 1.4e-3 stable: $J"
# CONFIG=llama3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=20 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr1.4e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r1 OPT=muon LR=1.4e-3 MODE=branch TPP=50 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr1.4e-3-tpp50 ladder/launch/slurm.sbatch

# # ---- lr 2e-3 ----
# J=$(CONFIG=llama3_r1 OPT=muon LR=2e-3 MODE=stable \
#   sbatch --parsable --nodes=1 --job-name=r1-muon-lr2e-3-main ladder/launch/slurm.sbatch); echo "lr 2e-3 stable: $J"
# CONFIG=llama3_r1 OPT=muon LR=2e-3 MODE=branch TPP=20 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr2e-3-tpp20 ladder/launch/slurm.sbatch
# CONFIG=llama3_r1 OPT=muon LR=2e-3 MODE=branch TPP=50 \
#   sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr2e-3-tpp50 ladder/launch/slurm.sbatch


# ---- lr 2.8e-3 ---- full stable run to TPP100 plus the TPP20 / TPP50 branches
J=$(CONFIG=llama3_r1 OPT=muon LR=2.8e-3 MODE=stable \
  sbatch --parsable --nodes=1 --job-name=r1-muon-lr2.8e-3-main ladder/launch/slurm.sbatch); echo "lr 2.8e-3 stable: $J"
CONFIG=llama3_r1 OPT=muon LR=2.8e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr2.8e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r1 OPT=muon LR=2.8e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr2.8e-3-tpp50 ladder/launch/slurm.sbatch

# ---- lr 4e-3 ---- stable run stops at the TPP50 decay start (iter 14000, STOP_TPP=50); TPP20 / TPP50 branches.
# If 2.8e-3 turns out best at TPP100, resubmit this stable job without STOP_TPP: it continues the same run
# to TPP100.
J=$(CONFIG=llama3_r1 OPT=muon LR=4e-3 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=1 --job-name=r1-muon-lr4e-3-main ladder/launch/slurm.sbatch); echo "lr 4e-3 stable: $J"
CONFIG=llama3_r1 OPT=muon LR=4e-3 MODE=branch TPP=20 \
  sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr4e-3-tpp20 ladder/launch/slurm.sbatch
CONFIG=llama3_r1 OPT=muon LR=4e-3 MODE=branch TPP=50 \
  sbatch --nodes=1 --dependency=afterok:$J --job-name=r1-muon-lr4e-3-tpp50 ladder/launch/slurm.sbatch

# ---- fit the sweep ----
# Reads every logs/*.out written so far (unfinished runs are listed, not fitted), so rerun it whenever
# more jobs finish. Writes logs/lr_fit/lr_fit_llama3_r1_muon.png and logs/lr_fit/lr_points.csv.
# Needs numpy, scipy and matplotlib in the active environment.
python ladder/analysis/lr_fit.py --logs 'logs/*.out' --config llama3_r1 --opt muon \
  --window 2.0 --noise 1e-3 --out-dir logs/lr_fit
