#!/bin/bash
# Token-margin pretraining on the ladder (ladder/TOKEN_MARGIN.md): Muon, to TPP 50, on two rungs
#   llama3_r2  LR 2e-3,   2 nodes (16 GPUs)
#   llama3_r1  LR 2.8e-3, 1 node  (8 GPUs)
# Run from the Megatron-LM root with the uv environment active: run the script, or create logs/ and paste one
# block into the shell. RUNGS picks the rungs the script submits (default both), e.g.
#   RUNGS=r1 bash ladder/margin/run_margin_pretrain.sh
# SFT, export and eval of these runs: ladder/sft/run_sft.sh.
#
# Per variant, two jobs:
#   stable  WSD run that stops at the TPP50 decay start (STOP_TPP=50); resubmitting it later without STOP_TPP
#           continues the same run to TPP 100
#   branch  the TPP50 decay into $OUT_ROOT/runs/<run>/ckpt/tpp50, the checkpoint SFT starts from (PRETRAIN_TAG=tpp50)
#   rung   stable          branch                                       plain-head time (ladder_runtime_summary.csv)
#   R2     0 -> 22000      22000 -> 27500 (28.8B tokens, TPP 49.1)      ~4.2 h + ~1 h on 16 GPUs
#   R1     0 -> 14000      14000 -> 17500 (18.4B tokens, TPP 48.7)      ~3.5 h + ~1 h on 8 GPUs
# Run directory: <config>_muon_lr<LR><head tag>, e.g. llama3_r2_muon_lr2e-3_hfn_d0.02 (ladder/configs/head.sh).
#
# LR: per rung, the grid point nearest the Muon optimum at TPP 50 (ladder/analysis/lr_fit.py), which is also the
# LR of a plain-head baseline with a finished tpp50 branch:
#   R2  optimum ~1.8e-3 -> 2e-3,   baseline llama3_r2_muon_lr2e-3   (ladder/run_llama_r2_lr_sweep.sh)
#   R1  optimum ~2.9e-3 -> 2.8e-3, baseline llama3_r1_muon_lr2.8e-3 (ladder/run_llama_r1_lr_sweep.sh)
# The head does not change the data order, so a margin run reads the same batches as its baseline and their
# validation losses pair.
# d = 0.02 is the centre of the margin grid (logit margin m = d s = 1.2 for hfn, d ||h|| for nfn / sfn); s = 60 and
# t = 1e-3 are the head.sh defaults. Besides lm loss, TensorBoard has ce loss (the plain CE of the same logits),
# margin r, margin logit m, lm head input norm, logit max / rms, and sfn sq dev for sfn.
#
# Standard QoS runs at most 32 GPUs of yours at once: the four stable jobs here take 2 x 16 + 2 x 8 = 48, so one
# waits. Add --qos=polaris_idle to the sbatch lines for idle nodes (preemptible; a requeued job resumes from its
# latest checkpoint). Resubmitting a finished block only queues jobs that exit at once (train.sh exits when a run
# has reached its target), but never resubmit one whose jobs are still queued or running: two jobs would write
# the same run directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs
RUNGS=${RUNGS:-r2 r1}

# ======== llama3_r2, LR 2e-3, 2 nodes ===========================================================================
if [[ " $RUNGS " == *" r2 "* ]]; then

# ---- hfn, d = 0.02: s (cos - d), final norm frozen at 1 ----------------------------------------------------
J=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=hfn MARGIN_D=0.02 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=2 --time=24:00:00 --job-name=llama3_r2-muon-lr2e-3_hfn_d0.02-main ladder/launch/slurm.sbatch)
echo "r2 hfn d0.02 stable: $J"
B=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=hfn MARGIN_D=0.02 MODE=branch TPP=50 \
  sbatch --parsable --nodes=2 --time=12:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3_hfn_d0.02-tpp50 ladder/launch/slurm.sbatch)
echo "r2 hfn d0.02 tpp50 branch: $B (after $J)"

# ---- sfn, d = 0.02: ||h|| (cos - d) + t (||h|| - s)^2 ------------------------------------------------------
J=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=sfn MARGIN_D=0.02 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=2 --time=24:00:00 --job-name=llama3_r2-muon-lr2e-3_sfn_d0.02-main ladder/launch/slurm.sbatch)
echo "r2 sfn d0.02 stable: $J"
B=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=sfn MARGIN_D=0.02 MODE=branch TPP=50 \
  sbatch --parsable --nodes=2 --time=12:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3_sfn_d0.02-tpp50 ladder/launch/slurm.sbatch)
echo "r2 sfn d0.02 tpp50 branch: $B (after $J)"

fi

# ======== llama3_r1, LR 2.8e-3, 1 node ==========================================================================
if [[ " $RUNGS " == *" r1 "* ]]; then

# ---- hfn, d = 0.02 -----------------------------------------------------------------------------------------
J=$(CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=1 --time=12:00:00 --job-name=llama3_r1-muon-lr2.8e-3_hfn_d0.02-main ladder/launch/slurm.sbatch)
echo "r1 hfn d0.02 stable: $J"
B=$(CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 MODE=branch TPP=50 \
  sbatch --parsable --nodes=1 --time=6:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r1-muon-lr2.8e-3_hfn_d0.02-tpp50 ladder/launch/slurm.sbatch)
echo "r1 hfn d0.02 tpp50 branch: $B (after $J)"

# ---- sfn, d = 0.02 -----------------------------------------------------------------------------------------
J=$(CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=sfn MARGIN_D=0.02 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=1 --time=12:00:00 --job-name=llama3_r1-muon-lr2.8e-3_sfn_d0.02-main ladder/launch/slurm.sbatch)
echo "r1 sfn d0.02 stable: $J"
B=$(CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=sfn MARGIN_D=0.02 MODE=branch TPP=50 \
  sbatch --parsable --nodes=1 --time=6:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r1-muon-lr2.8e-3_sfn_d0.02-tpp50 ladder/launch/slurm.sbatch)
echo "r1 sfn d0.02 tpp50 branch: $B (after $J)"

fi

# ---- optional variants: copy a block above and change HEAD / MARGIN_D / MARGIN_REVERSE and the job names -------
#   HEAD=nfn MARGIN_D=0.02                    ||h|| (cos - d), no penalty
#   HEAD=hfn MARGIN_D=0       (_hfn_d0)       feature-norm-only controls: the head geometry without a margin,
#   HEAD=sfn MARGIN_D=0       (_sfn_d0)       which splits a margin run's gap to the baseline into head + margin
#   HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=1   (_hfn_rev_d0.02) s (cos + d)
#   HEAD=hfn MARGIN_D=0.05                    a larger margin (m = 3)
