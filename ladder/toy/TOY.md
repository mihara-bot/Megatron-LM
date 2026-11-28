# Margin toy: token margin on a 2-layer model with a known next-token distribution

The token-margin LM head (`ladder/TOKEN_MARGIN.md`) on a toy problem where the true next-token distribution `q` is
known exactly at every position. Three questions:

1. **Distribution**: does the margin change the LM head's output distribution, and by as much as theory says?
2. **Geometry**: does the SphereFace picture hold on the LM head (angular gap between the target and the rest)?
3. **Usefulness**: does a margin-trained model (reverse margin in particular) do something that decoding a plain
   model cannot (temperature, or the post-hoc margin map below), and does it survive SFT?

Everything runs through the ladder (`train.sh`, `train_sft.sh`, `export_hf.sh`) with the same head code as the real
runs; only the data, the config and the probe are toy-specific.

## Theory: what the margin does to the output distribution

Margin CE with logit margin `m` (`m = d s` for hfn, `d ||h||` for nfn / sfn, negative under reverse) is
`CE + m + log(1 - c p_y)` with `c = 1 - e^{-m}`. Its expected value under a target distribution `q` is minimized at

    p_k (1 - c p_k)  proportional to  q_k          (ladder/toy/margin_theory.py)

- `m = 0`: `p = q` (plain CE is a proper scoring rule).
- forward (`m > 0`): sharper than `q`; reverse (`m < 0`): flatter, with `p -> sqrt(q)` on the support as `m -> -inf`.
- one-hot `q` (one correct token, as in face recognition) stays one-hot: there the margin only changes the geometry.

| `q` = (0.5, 0.3, 0.2) | m = +3 | +1.2 | 0 | -1.2 | -3 |
|---|---|---|---|---|---|
| optimum `p` | .76 .15 .09 | .59 .25 .16 | .50 .30 .20 | .45 .32 .24 | .42 .32 .26 |

With `s = 60` (kept from the real runs), `d = 0.005 / 0.02 / 0.05` give `m = 0.3 / 1.2 / 3`. At the optimum, reverse
`m = -1.2` is close to temperature 1.4 but leaves the low-probability tail less raised than temperature does, which
is the kind of difference question 3 looks for.

## Data (`graph.py`, `make_data.py`)

- **Vocabulary (100)**: `<unk> <eos> <walk> <path> : <rsv0..2>`, then nodes `n0..n91` (token `8 + i`).
  `make_tokenizer.py` writes it as an HF WordLevel tokenizer (whitespace split, no BOS), so every ladder script that
  takes `TOKENIZER` works as is; `--make-vocab-size-divisible-by 4` keeps the vocabulary unpadded.
- **Graph**: a layered random DAG, 8 layers of 12/11 nodes, edges only to the next layer. 20% of the nodes have one
  successor (deterministic positions); the others 2-6, with edge probabilities `0.02 + (1 - 0.02 k) Dirichlet(alpha)`,
  `alpha` in {0.3, 1, 3} per node (skewed to near-uniform rows). Every edge has probability >= 0.02.
- **Pretraining** (walks only): `<walk> v0 v1 ... v7 <eos>`, `v0` uniform over layer 0, then a step along the edges.
  The next token depends on the previous token only, so every position has an exact `q`: uniform over layer 0 after
  `<walk>`, the edge row after a node, `<eos>` after a last-layer node, `<walk>` after `<eos>`. The entropy rate of
  the stream (0.860 nats for graph seed 0) is the floor of the plain-CE validation loss. 4M training walks (40M
  tokens; the loader repeats them), 50k validation walks.
- **Path task** (SFT): prompt `<path> s t : `, response `s v ... t <eos>`, any path along the edges; `t` at least
  2 layers after `s`. The 2435 reachable pairs (seed 0) are split 80/20 into training and held-out pairs, stratified by
  span; valid paths per pair grow from ~1.6 (span 2) to ~530 (span 7). Three SFT distributions over the valid paths:
  `cond` (a walk conditioned on reaching `t`), `uniform`, `canon` (one path per pair, the most probable). 16
  examples per training pair per epoch, 8 epochs (fresh paths each epoch for cond / uniform), 256 examples per step.

## Model and schedule (`ladder/configs/toy_l2.sh`)

2 layers, d 256, 4 heads x 64, SwiGLU 896, RMSNorm, RoPE, untied; N_eff 1.93M. One GPU, 128 x 256 tokens per
step, Muon + WSD to TPP 100 (6250 iterations, 205M tokens, decay from 5000), a checkpoint every 500 iterations, all
kept. SFT (`train_sft.sh`, protocol A: the run's head without margin): ~970 iterations at LR / 3, a checkpoint per
epoch.

## Running

From the Megatron-LM root, with the environment active:

```bash
export TOY_ROOT=/fast/users/zhuangxinlin/toy_margin
bash ladder/toy/prepare.sh                          # tokenizer + data + indexed datasets (CPU, minutes)
PHASE=lr bash ladder/toy/run_toy.sh                  # plain head over LR_GRID (1 node job)
python3 ladder/toy/summarize.py lr --root $TOY_ROOT  # -> the LR with the lowest walk KL
PHASE=grid LR=<that LR> bash ladder/toy/run_toy.sh   # 15 variants x 3 seeds, each with SFT (3 node jobs)
python3 ladder/toy/summarize.py table --root $TOY_ROOT
PHASE=decode LR=<that LR> bash ladder/toy/run_toy.sh # re-sample the SFT models: T up to 2, margin map (1 node job)
python3 ladder/toy/summarize.py table --root $TOY_ROOT --lr <that LR>   # -> report/sft_decode_summary.csv
```

- `run_toy.sh` writes spec files (one chain per line) and submits `pack.sbatch` jobs: one node, 8 GPUs, 2 chains per
  GPU. `LOCAL=1` runs the packs on the current node instead; `DRY_RUN=1` prints everything and submits nothing.
- `chain.sh` is one variant end to end on one GPU: pretrain -> export (every checkpoint) -> probe -> per SFT mode
  (cond, canon): SFT -> export (every epoch) -> probe with `--paths`. Every step skips what is done, so a chain or a
  whole phase can be resubmitted after a failure or preemption.
- Variants (`run_toy.sh`): plain; hfn d = 0 / 0.005 / 0.02 / 0.05 and reverse 0.005 / 0.02 / 0.05; sfn d = 0 / 0.02 /
  0.05 and reverse 0.02 / 0.05; nfn 0.02 and reverse 0.02. Same `SEED` = same init and same batches, so variants pair.
- Layout under `$TOY_ROOT`: `tokenizer/`, `data/`, `megatron/`, `out/runs/<run>/{ckpt,tb,hf/main_iter<N>/probe.json}`,
  `out/sft/<run>_main_sft_toy_<mode>/{ckpt,tb,hf/iter_<N>/probe.json}`, `logs/pack_*/<chain>.log`, `report/*.csv`.
  Run name: `toy_l2_muon_lr<LR><head tag>_seed<SEED>`.

## What the probe measures (`probe.py` -> `probe.json`)

All against the exact `q`, on the same probe data for every model.

| question | where in probe.json | read it as |
|---|---|---|
| 1 distribution | `walk.classes.multi`: `top1_p` vs `top1_theory` vs `top1_q`, `h_p` vs `h_theory` vs `h_q`, `kl`, `kl_theory` | the model follows the margin optimum if `kl_theory` << `kl`; `walk.multi_nodes` has the per-node `q`, `p`, `p*` for scatter plots; `walk.multi_by_hq` splits by entropy of `q` |
| 2 geometry | `walk.quantiles.det_gap` (one-label positions: `cos_y - max cos_other`), `multi_supp_gap`, `multi_supp_spread`, `walk.head` (sibling vs same-layer row cosine), `hnorm` | forward should widen the gap, reverse narrow it; nfn / sfn reverse: watch `hnorm` grow |
| 3 usefulness | `walk.temperature`, `walk.posthoc_margin` (decoding the model's own `p`), `junk` (mass off the support), `supp_kl`; SFT: `paths.sample.T*.{train,heldout}` pass@k, `distinct`, `cover_count`, `cover_mass`, `fail`; `paths.tf` | does a reverse-margin model reach a (junk, entropy) or (pass@k, coverage) point that the plain model cannot reach by temperature or the post-hoc map? |

`s_fit` is the `s` of `s cos(h, W)` that fits a model's distribution best (for comparison with `s = 60`).

**Decoding with the margin map instead of training with the margin.** A model trained to the margin CE's optimum puts
out `p*(q, m)`; a plain model puts out `p ~ q`, so mapping its `p` through `p'_k (1 - c p'_k) ~ p_k` gives the same
distribution at decode time (`walk.posthoc_margin` checks this on the pretrained models). `PHASE=decode` samples the
SFT models through that map (`probe.py --update --map-ms=...`, keys `M<m>` next to the temperatures `T<T>` in
`paths.sample`); `report/sft_decode_summary.csv` then compares, per SFT mode and variant, temperature against the map
(pass@k, distinct paths, coverage, and the failure rates such as `heldout_fail_bad_edge`), e.g. the plain-SFT model
decoded through the map against a model whose SFT itself used the margin.

## Tests

`python -m pytest ladder/tests/test_toy.py -p no:cacheprovider -o addopts=""`: graph and sampling against brute
force and Monte Carlo, the entropy rate, the SFT rows against the identity-template invariant, the margin optimum
against gradient descent, the launchers' dry runs, the summary. The probe tests need torch / transformers (cluster).
`prepare.sh` additionally checks the tokenizer, the SFT rows with Megatron's own `SFTTokenizer`, and the indexed data
with `check_tokenizer.py`.
