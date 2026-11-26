#!/bin/bash
# Token-margin pretraining on the ladder (ladder/TOKEN_MARGIN.md): llama3_r2 + Muon, LR 2e-3, to TPP 50.
# Run from the Megatron-LM root with the uv environment active: run the whole script, or create logs/ and paste
# one block into the shell. SFT, export and eval of these runs: ladder/sft/run_sft.sh.
#
# Per variant, two jobs on 2 nodes (16 GPUs; the plain head takes ~4.2 h + ~1 h, ladder_runtime_summary.csv):
#   stable  WSD run that stops at the TPP50 decay start, iteration 22000 (STOP_TPP=50); resubmitting it later
#           without STOP_TPP continues the same run to TPP 100
#   branch  decay 22000 -> 27500 (28.8B tokens, TPP 49.1) into $OUT_ROOT/runs/<run>/ckpt/tpp50, the checkpoint
#           SFT starts from (PRETRAIN_TAG=tpp50)
# Run directory: llama3_r2_muon_lr2e-3<head tag>, e.g. llama3_r2_muon_lr2e-3_hfn_d0.02 (ladder/configs/head.sh).
#
# LR 2e-3: the R2 grid point nearest the Muon optimum at TPP 50 (~1.8e-3), the LR of the plain-head baseline
# llama3_r2_muon_lr2e-3 in ladder/run_llama_r2_lr_sweep.sh. The head does not change the data order, so a margin
# run reads the same batches as that baseline and their validation losses pair.
# d = 0.02 is the centre of the margin grid (logit margin m = d s = 1.2 for hfn, d ||h|| for nfn / sfn); s = 60 and
# t = 1e-3 are the head.sh defaults. Besides lm loss, TensorBoard has ce loss (the plain CE of the same logits),
# margin r, margin logit m, lm head input norm, logit max / rms, and sfn sq dev for sfn.
#
# Standard QoS runs at most 32 GPUs at once (two of these 16-GPU jobs, with nothing else of yours running).
# Add --qos=polaris_idle to the sbatch lines for idle nodes (preemptible; a requeued job resumes from its latest
# checkpoint). Resubmitting a block only runs what is missing: train.sh exits when a run has reached its target.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs

# ---- hfn, d = 0.02: s (cos - d), final norm frozen at 1 ----------------------------------------------------
J=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=hfn MARGIN_D=0.02 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=2 --time=24:00:00 --job-name=llama3_r2-muon-lr2e-3_hfn_d0.02-main ladder/launch/slurm.sbatch)
echo "hfn d0.02 stable: $J"
B=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=hfn MARGIN_D=0.02 MODE=branch TPP=50 \
  sbatch --parsable --nodes=2 --time=12:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3_hfn_d0.02-tpp50 ladder/launch/slurm.sbatch)
echo "hfn d0.02 tpp50 branch: $B (after $J)"

# ---- sfn, d = 0.02: ||h|| (cos - d) + t (||h|| - s)^2 ------------------------------------------------------
J=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=sfn MARGIN_D=0.02 MODE=stable STOP_TPP=50 \
  sbatch --parsable --nodes=2 --time=24:00:00 --job-name=llama3_r2-muon-lr2e-3_sfn_d0.02-main ladder/launch/slurm.sbatch)
echo "sfn d0.02 stable: $J"
B=$(CONFIG=llama3_r2 OPT=muon LR=2e-3 HEAD=sfn MARGIN_D=0.02 MODE=branch TPP=50 \
  sbatch --parsable --nodes=2 --time=12:00:00 --dependency="afterok:$J" \
  --job-name=llama3_r2-muon-lr2e-3_sfn_d0.02-tpp50 ladder/launch/slurm.sbatch)
echo "sfn d0.02 tpp50 branch: $B (after $J)"

# ---- optional variants: copy a block above and change HEAD / MARGIN_D / MARGIN_REVERSE and the job names -------
#   HEAD=nfn MARGIN_D=0.02                    ||h|| (cos - d), no penalty
#   HEAD=hfn MARGIN_D=0       (_hfn_d0)       feature-norm-only controls: the head geometry without a margin,
#   HEAD=sfn MARGIN_D=0       (_sfn_d0)       which splits a margin run's gap to the baseline into head + margin
#   HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=1   (_hfn_rev_d0.02) s (cos + d)
#   HEAD=hfn MARGIN_D=0.05                    a larger margin (m = 3)
