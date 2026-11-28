# SFT and offline math eval on the ladder

The Megatron counterpart of pretrain_LLM_pipeline's SFT (docs/sft.md there): OpenMathInstruct-2, 500k examples x
10 epochs, then GSM8K / MATH-500 / AMC-2023 / Countdown pass@k. Protocol A for margin runs: SFT keeps the
pretraining LM head with no margin (see TOKEN_MARGIN.md).

## 1. Data (once per cluster)

```bash
export HF_ENDPOINT=https://hf-mirror.com            # the cluster cannot reach huggingface.co
HFD=<path to hfd.sh>; DATA=/fast/users/zhuangxinlin/data
# SFT source: the full OpenMathInstruct-2 train split (data/train-000xx-of-00032.parquet)
bash $HFD nvidia/OpenMathInstruct-2 --dataset --include "data/train-*" --local-dir $DATA/OpenMathInstruct-2
# eval sets
bash $HFD openai/gsm8k --dataset --include "main/test-*" --local-dir $DATA/gsm8k
bash $HFD HuggingFaceH4/MATH-500 --dataset --local-dir $DATA/MATH-500
bash $HFD math-ai/amc23 --dataset --local-dir $DATA/amc23
bash $HFD Jiayi-Pan/Countdown-Tasks-3to4 --dataset --local-dir $DATA/Countdown-Tasks-3to4
python ladder/sft/eval/prepare_countdown_eval.py --input $DATA/Countdown-Tasks-3to4 \
    --output $DATA/countdown/test_tinyzero_1024.jsonl              # TinyZero test rows 327680..328703
```

The 500k train / 2k validation subset (Megatron environment, CPU): a seeded draw without replacement over all rows,
validation problems kept out of training. It is the same draw as pretrain_LLM_pipeline's SFT subset, so the same
source files and seed give the same rows there (frozen in ladder/tests/test_sft_scripts.py); nothing here needs that
repository.

```bash
python ladder/sft/make_sft_subset.py --input $DATA/OpenMathInstruct-2/data --file-glob 'train-*' \
    --output-dir $DATA/sft/openmathinstruct2_500k --num-train 500000 --num-validation 2000 --seed 42 \
    --columns problem generated_solution expected_answer problem_source --disjoint-column problem
```

Then, per tokenizer family (Megatron environment, CPU is enough):

```bash
python ladder/sft/prepare_sft_data.py --input $DATA/sft/openmathinstruct2_500k/train \
    --tokenizer /fast/users/zhuangxinlin/tokenizers/llama-3 \
    --output-dir $DATA/sft_megatron/omi2_500k_ep10_llama3 --epochs 10
```

It writes `train.parquet` (5M rows: every epoch in the pipeline's shuffled order, since Megatron reads the rows
front to back) and `stats.json`, and checks a few rows with Megatron's own SFT tokenizer. For each example,
prompt + response + EOS, the prompt unsupervised, both pieces tokenized without BOS: the pipeline's tokens and mask.
Dropped and counted: overlong (> 4096 tokens), empty responses, and rows whose BPE merges across the
prompt/response boundary, which Megatron's identity template cannot represent. The pipeline had no overlong
or empty rows, and boundary rows should be ~0. `stats.json` also replays the pipeline's packing (15851 steps of 32
rows, ~315 examples per step) and gives the matching Megatron global batch per data-parallel size.

The validation split, for the validation loss during training, goes into the same directory as `valid.parquet` +
`valid_stats.json`:

```bash
python ladder/sft/prepare_sft_data.py --input $DATA/sft/openmathinstruct2_500k/validation \
    --tokenizer /fast/users/zhuangxinlin/tokenizers/llama-3 \
    --output-dir $DATA/sft_megatron/omi2_500k_ep10_llama3 --epochs 1 --output-name valid
```

## 2. Training

```bash
CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 PRETRAIN_TAG=tpp100 \
SFT_DATA_DIR=$DATA/sft_megatron/omi2_500k_ep10_llama3 ENTRY=ladder/sft/train_sft.sh \
    sbatch --nodes=1 -J sft-r1-hfn ladder/launch/slurm.sbatch
```

`ladder/sft/train_sft.sh` (any variable can be checked with `DRY_RUN=1 bash ladder/sft/train_sft.sh`):
- start: the latest checkpoint of `$OUT_ROOT/runs/<run>/ckpt/<PRETRAIN_TAG>`, where the run is named from CONFIG / OPT
  / LR / HEAD vars as `ladder/train.sh` names it. Only the weights are loaded, through `--pretrained-checkpoint`.
  A rerun resumes from the SFT checkpoints. A decay branch also saves on its way, so the latest checkpoint must be
  the run's target (`--exit-interval`, else `--train-iters`, of the last `cmd_<tag>_*.sh` there); otherwise SFT
  (and `export.sh TARGET=pretrained`) stops, unless `ALLOW_UNFINISHED_PRETRAIN=1`.
- head: hfn -> hfn (s kept, final norm frozen), nfn / sfn -> nfn, plain -> plain; all with d = 0.
- optimizer: Muon (`ladder/configs/opt/muon.sh`) by default, whatever the pretraining run used (`OPT` only names that
  run, default muon); `SFT_OPT=adamw` switches and adds `_adamw` to the SFT run name. Peak LR = pretraining LR / 3
  (`SFT_LR`), 3% warmup, cosine to 0.
- batch: `--use-varlen-dataset` packs examples online (dp_balanced), and the global batch counts examples: by
  default the pipeline's examples per step rounded to the data-parallel size (8 GPUs: 312 examples, 16025
  iterations; 16 GPUs: 320, 15625; pipeline: 15851). `SFT_GBS` overrides it; `run_sft.sh` pins 312 so the batch and
  schedule do not depend on `SFT_GPUS`. `SFT_TOKENS_PER_RANK` (`--max-seqlen-per-dp-cp-rank`, default 4096;
  `run_sft.sh` and the smoke use 16384) caps the tokens of one packed microbatch per GPU: larger means fewer,
  bigger microbatches for the same examples and the same per-token loss (attention stays within each example).
- one checkpoint per epoch (all kept). Output: `$OUT_ROOT/sft/<run>_<tag>_sft_omi2_500k_ep10/`.
- validation loss (`lm loss validation` in TensorBoard), when `$SFT_DATA_DIR/valid.parquet` exists (`SFT_EVAL=auto`;
  `1` requires it, `0` turns it off): every evaluation is one eval iteration over the whole validation set
  (`--eval-global-batch-size` = its rows, rounded down to the data-parallel size), so every point is on the same
  examples; once per epoch (`SFT_EVAL_INTERVAL`, default the iterations per epoch = 1602, so at the epoch
  checkpoints) plus one at the end of training (the final checkpoint). The
  pipeline logged no validation loss; it reads no training data. pass@k stays offline (section 3).

What differs from the pipeline: the packing (online and balanced over the ranks, instead of next-fit
rows) and therefore which examples share a step. The examples, their order, tokens, mask, optimizer, LR schedule
and the number of steps all match.

## 3. Export and offline eval

Two steps, two environments; `ladder/sft/run_sft.sh` has both, chained after training (`sft_chain`: train ->
export -> eval per pretraining run, plus the pre-SFT reference; the llama3_r2 and llama3_r1 TPP50 runs of
`ladder/margin/run_margin_pretrain.sh` and their plain-head baseline).

```bash
# export (1 GPU, Megatron env): Megatron checkpoints -> HF directories
CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 PRETRAIN_TAG=tpp100 ITERS=all \
  sbatch -J sft-export -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=128G -t 04:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/sft/export.sh'
# eval (1 GPU, vLLM env): MODEL = HF directories
MODEL="$OUT_ROOT/sft/<sft run>/hf/iter_*" VLLM_PYTHON=<vllm env>/bin/python \
  sbatch -J sft-eval -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=128G -t 24:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh'
```

`ladder/sft/export.sh` takes the run's variables (as train_sft.sh) and writes `<sft dir>/hf/iter_<N>` with
`ladder/export/export_hf.sh`: a plain Llama / Qwen3 with the head folded, Megatron vs HF logits checked.
`ITERS=latest|all|<list>`; `TARGET=pretrained` exports the start checkpoint instead (the pre-SFT reference,
`<pretraining run>/hf/<PRETRAIN_TAG>_iter<N>`). Directories that already have `export_meta.json` are kept.

`ladder/sft/eval.sh` takes only `MODEL`: one or more HF directories, any HF causal LM (e.g. an official base
model as a reference). Globs expand inside the job and models run in iteration order. It runs
`ladder/sft/eval/eval_vllm.py` with `ladder/sft/eval/math_passk.yaml`, which has the pipeline's tasks, prompts (the
SFT template; Countdown's own few-shot prompt and stop) and offline settings: 64 samples (T 0.6, top_p 0.95) +
greedy, all problems, GSM8K 512 / others 1024 new tokens. Grading uses `math_grader.py`, copied unchanged from
the pipeline. Prompts are tokenized without BOS and passed to vLLM as token ids.
Results: `<X>/eval/<name>/{summary.json,<task>.jsonl}` for a model at `<X>/hf/<name>` (`EVAL_DIR` overrides for
a single model), plus one line per model in `<X>/eval/summary.jsonl` (keys as the pipeline's: `gsm8k/pass@1`,
`math500/greedy_acc`, ...). `TASKS` / `NUM_SAMPLES` / `MAX_PROBLEMS` give quick subsets in their own directories.
vLLM samples differently from HF generate, so per-sample outputs differ from the pipeline's, but the metric
distributions are the same.

vLLM is not in the NGC image, and must not go into the Megatron environment (it brings its own torch, which
would replace the image's torch that TransformerEngine is built against). `ladder/env/setup_vllm_env.sh` builds a
separate venv (vLLM 0.28.0 by default: torch 2.13, CUDA 13.0 wheels, NVIDIA driver >= 580; pyarrow, pyyaml,
sympy + antlr4-python3-runtime 4.11 for the MATH equivalence check) and records its versions in
`requirements.lock`; `ladder/env/check_vllm_env.py --gpu --tokenizer <llama-3 tokenizer>` checks it on a GPU (the
command is at the top of the setup script). eval.sh runs `VLLM_PYTHON` through `ladder/sft/eval/run_isolated.sh`,
which drops the submitting shell's PYTHONPATH and the package directories of LD_LIBRARY_PATH, so the Megatron
venv and the image's torch libraries cannot leak into the vLLM process.

Only torch is separate: transformers and tokenizers are pinned to the Megatron venv's exact versions (5.9.0 /
0.22.2 in the NGC 26.11 image), read when the setup script runs, so SFT data, export and eval tokenize with the
same libraries. The vLLM version must accept that transformers: 0.19.1 - 0.28.0 do for 5.9.0, 0.29+ need >= 5.10.4.
The export writes its versions to `export_meta.json` (`hf_versions`); `eval_vllm.py` stops on a mismatch
(`ALLOW_HF_MISMATCH=1` only warns) and records both in `summary.json`. Models without the record (reference
models, older exports) are evaluated with a note.

```bash
source <MEGATRON_VENV>/bin/activate   # once, no GPU needed; pins come from this venv
UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple \
  VLLM_VENV=/fast/users/zhuangxinlin/envs/vllm bash ladder/env/setup_vllm_env.sh
```

## Tests

- `python -m pytest ladder/tests/test_sft_scripts.py -p no:cacheprovider -o addopts=""` (CPU, no dependencies):
  subset draw, template / epoch order / packing against values frozen from the pipeline, boundary / overlong
  drops, schedule, eval problems / grading / summary, the train / export / eval launchers' plans.
- `python -m pytest ladder/tests/test_sft_megatron_data.py -p no:cacheprovider -o addopts=""` (CPU, Megatron env with
  the ladder tokenizers): prepare_sft_data.py end to end on synthetic rows, then Megatron's VarlenDataset + identity
  SFTTokenizer on its output, item by item against the pipeline's example (tokens, targets, loss mask), Llama3 and
  Qwen3 tokenizers.
- `bash ladder/sft/smoke_sft.sh` (one node, see run_sft.sh): 4000 rows x 2 epochs (~125 steps) from a real ladder
  checkpoint, export of every checkpoint, a quick vLLM eval, then `ladder/sft/check_sft_smoke.py`: the run
  finished, the loss is finite, starts low (pretrained weights loaded) and goes down, LR peaks and decays to 0, the
  epoch checkpoints exist, head metrics are as expected (d = 0), every export passed its logits check, eval
  results exist.
