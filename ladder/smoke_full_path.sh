# Short end-to-end test of every code path a ladder run takes, on the real llama3_r1 + Muon setup
# (same dataset index, so no rebuild). Three chained jobs in one throwaway run directory; ~20 min total.
# Submit from the Megatron-LM root with the uv environment activated, all lines in one shell.
#
#   job 1  stable from scratch, exit at 120 : validation every 20, saves every 40 and deletes the old
#                                             ones (--save-retain-interval), manual GC at 100, exit save
#   job 2  stable, resume, exit at 160      : loads the Muon distributed-optimizer checkpoint at 120
#   job 3  branch TPP20                     : forks from the stable checkpoint (--ckpt-step), WSD
#                                             minus_sqrt decay 160 -> 240 to LR 0, saves into ckpt/tpp20
# train.sh starts a TPP20 branch from main/iter_0005600; the symlink below points it at iter 160, and
# --ckpt-step 160 in EXTRA_ARGS (later flags win) loads that iteration. Megatron never deletes symlinks.
# The scheduler asserts warmup < decay end, so job 3 shortens warmup to 100 (the real branch: 500 < 7000).
#
# Afterwards: rm -rf "$OUT_ROOT/runs/debug_full"
OUT_ROOT=${OUT_ROOT:-/fast/users/zhuangxinlin/ckpts}
D=$OUT_ROOT/runs/debug_full/ckpt/main
mkdir -p logs "$D" && ln -sfn iter_0000160 "$D/iter_0005600"

J1=$(CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=stable RUN=debug_full \
  EXTRA_ARGS="--eval-interval 20 --save-interval 40 --exit-interval 120" \
  sbatch --parsable --nodes=1 --exclude=htv02 --job-name=smoke-1-stable ladder/launch/slurm.sbatch); echo "job 1: $J1"
J2=$(CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=stable RUN=debug_full \
  EXTRA_ARGS="--eval-interval 20 --save-interval 40 --exit-interval 160" \
  sbatch --parsable --nodes=1 --exclude=htv02 --dependency=afterok:$J1 --job-name=smoke-2-resume ladder/launch/slurm.sbatch); echo "job 2: $J2"
J3=$(CONFIG=llama3_r1 OPT=muon LR=1e-3 MODE=branch TPP=20 RUN=debug_full \
  EXTRA_ARGS="--ckpt-step 160 --lr-warmup-iters 100 --lr-decay-iters 240 --lr-wsd-decay-iters 80 --exit-interval 240 --save-interval 40 --save-retain-interval 240 --eval-interval 20" \
  sbatch --parsable --nodes=1 --exclude=htv02 --dependency=afterok:$J2 --job-name=smoke-3-branch ladder/launch/slurm.sbatch); echo "job 3: $J3"
