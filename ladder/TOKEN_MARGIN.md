# Token-margin loss (SphereFace-R LM head) on the ladder

Port of `pretrain_LLM_pipeline`'s `train.loss.margin` (docs/token_margin.md there). Same objective:
SphereFace-R (arXiv 2109.05565) with the additive cosine margin `psi = cos - d`, modes nfn / hfn / sfn,
`reverse` for `cos + d`.

| mode | logits | training loss per token |
|---|---|---|
| nfn | `‖h‖ cos(h, W_i)` | margin CE, `m = d ‖h‖` (gradient through `‖h‖`) |
| hfn | `s cos(h, W_i)`; final RMSNorm weight frozen at 1 | margin CE, `m = d s` |
| sfn | as nfn | margin CE + `t (‖h‖ - s)^2` |

`h` is the output-layer input (after the final norm), `W_i` the output rows, which are unit-normalized
in the forward pass in every mode. Margin CE is computed in closed form from the per-token cross
entropy `L`: `L + m + log((1 - p_y) + p_y e^{-m})`. Validation reports the plain CE of the head.

## Code

- `megatron/core/models/gpt/token_margin.py`: `FeatureNormColumnParallelLinear` (output layer that
  normalizes its rows inside its own forward, after the DDP param-gather pre-hook, with
  gradient-accumulation fusion off for this one layer), the hfn input scaling, the closed form, the
  metrics, `head_geometry`.
- `GPTModel`: picks that output layer, freezes the final norm under hfn, scales `h` before the head,
  and in training (`model.train()` with labels) returns the per-token margin objective.
- `TransformerConfig`: `--lm-head-feature-norm {nfn,hfn,sfn}`, `--lm-head-feature-norm-scale` (s, 60),
  `--soft-feature-norm-weight` (t, 1e-3), `--token-margin` (d, 0), `--token-margin-reverse`.
  Rejected: MTP, `defer_embedding_wgrad_compute`, full-iteration CUDA graphs, hfn without RMSNorm;
  GPTModel also rejects tied embeddings and the MXFP8 TE LM head.
- `pretrain_gpt.py` `loss_func`: adds the metrics to the loss report.
- `checkpointing.py`: `load_checkpoint` refuses a checkpoint whose head geometry differs from the
  run's (also for `--finetune`, which skips the other argument checks). nfn and sfn share a
  geometry. `--use-checkpoint-args` restores the head (not the margin) when none is given.

## Training

```bash
HEAD=hfn MARGIN_D=0.02 CONFIG=llama3_r1 OPT=muon LR=2.8e-3 sbatch ... ladder/launch/slurm.sbatch
HEAD=sfn MARGIN_D=0.02 CONFIG=llama3_r1 OPT=muon LRS=2.8e-3 bash ladder/sweep.sh
HEAD=hfn CONFIG=llama3_r1 ...                  # d = 0: feature-norm-only control
HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=1 ...    # cos + d
```

`ladder/configs/head.sh` turns `HEAD MARGIN_D MARGIN_REVERSE HEAD_SCALE SFN_WEIGHT` into flags and a
run-name suffix (`_hfn_d0.02`, `_sfn_rev_d0.02`, `_hfn_d0_s40`, ...; empty for the plain head). The
head does not change the data order, so a margin run and the baseline of the same rung, LR and seed
read the same batches. `analysis/lr_fit.py` and `analysis/scaling.py` see the suffix as part of the
optimizer name (`muon_hfn_d0.02`), so margin runs never merge into the baseline's LR fits.

Logged per iteration (TensorBoard / W&B / console), on top of `lm loss` (the optimized objective):

| key | meaning |
|---|---|
| `ce loss` | plain CE of the normalized head; compare with a baseline's `lm loss` |
| `margin r` | mean gradient factor `r = 1 / ((1 - p_y) + p_y e^{-m})` |
| `margin logit m` | mean logit margin (`d s` or `d ‖h‖`, negative under reverse) |
| `lm head input norm` | mean `‖h‖` |
| `sfn sq dev` | sfn only: mean `(‖h‖ - s)^2`, before `t` |
| `logit max`, `logit rms` | on 256 strided token rows per microbatch |

`lm loss validation` is the plain CE (no margin, no penalty).

## Export to Hugging Face / vLLM

```bash
CONFIG=llama3_r1 HEAD=hfn MARGIN_D=0.02 TOKENIZER=<hf tokenizer dir> \
    CKPT=$OUT_ROOT/runs/llama3_r1_muon_lr2.8e-3_hfn_d0.02/ckpt/tpp50 bash ladder/export/export_hf.sh
```

One GPU. Writes `<run>/hf/<tag>_iter<N>/`: safetensors, `config.json` (LlamaForCausalLM, or
Qwen3ForCausalLM when the config has `--qk-layernorm`), tokenizer, `export_meta.json`. The head is
folded into `lm_head.weight`: nfn / sfn `W / ‖W‖` (exact), hfn `(s / sqrt(D)) W / ‖W‖` (exact up to the
RMSNorm epsilon; refused if the final norm weight is not exactly 1). The result is a plain model for
transformers and vLLM. The exporter then reloads it with transformers and compares its logits with the
Megatron model's on random tokens (`validation` in `export_meta.json`; fails at mean KL > 1e-2 or top-1
agreement < 0.98). Dense Llama3 / Qwen3 only; MoE / MLA / MTP go through Megatron-Bridge.

Ladder data is EOS-only (no BOS); the exported tokenizer may add BOS by default. Use the same
convention for every run in evaluation.

## SFT in Megatron (protocol A)

Keep the run's head, no margin, no sfn penalty: hfn runs fine-tune with `HEAD=hfn MARGIN_D=0`, nfn and sfn
runs with `HEAD=nfn MARGIN_D=0`, the baseline with the plain head. hfn keeps the final norm frozen, so the SFT
checkpoint still folds on export. A mismatched head fails at load time instead of training on raw rows.
`ladder/sft/train_sft.sh` does this from the pretraining run's variables; see SFT.md.

## Tests

- `python -m pytest ladder/tests -p no:cacheprovider -o addopts=""`: head flags, run names, branch
  paths, log header, sweep, export dry run, tag agreement with `scaling.py` (CPU).
- `pytest tests/unit_tests/models/test_gpt_token_margin.py`: closed form vs modified logits (CPU);
  on one GPU, eval logits and training loss + all parameter gradients vs the explicit formula for every
  mode, d, reverse and both CE implementations; Megatron vs HF logits after `hf_mapping` for
  Llama / Qwen3 and every head.

Not covered by tests: TP > 1 with sequence parallelism (the `‖h‖` gather; the ladder runs TP = 1),
pipeline parallelism, `--cross-entropy-fusion-impl te`.
