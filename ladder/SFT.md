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

The 500k / 2k subset is the pipeline's own (`src.cli.prepare_sft_subset`; same seed and source files give the same
rows). Copy it from the cloud cluster (`/mnt/users/zhuangxinlin/data/sft/openmathinstruct2_500k`) or rebuild it
from a pretrain_LLM_pipeline checkout:

```bash
python -m src.cli.prepare_sft_subset --input $DATA/OpenMathInstruct-2/data --file_glob 'train-*' \
    --output_dir $DATA/sft/openmathinstruct2_500k --num_train 500000 --num_validation 2000 --seed 42 \
    --columns problem generated_solution expected_answer problem_source --disjoint_column problem
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

## 2. Training

```bash
CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 PRETRAIN_TAG=tpp100 \
SFT_DATA_DIR=$DATA/sft_megatron/omi2_500k_ep10_llama3 ENTRY=ladder/sft/train_sft.sh \
    sbatch --nodes=1 -J sft-r1-hfn ladder/launch/slurm.sbatch
```

`ladder/sft/train_sft.sh` (any variable can be checked with `DRY_RUN=1 bash ladder/sft/train_sft.sh`):
- start: the latest checkpoint of `$OUT_ROOT/runs/<run>/ckpt/<PRETRAIN_TAG>`, where the run is named from CONFIG / OPT
  / LR / HEAD vars as `ladder/train.sh` names it. Only the weights are loaded, through `--pretrained-checkpoint`.
  A rerun resumes from the SFT checkpoints.
- head: hfn -> hfn (s kept, final norm frozen), nfn / sfn -> nfn, plain -> plain; all with d = 0.
- optimizer: the pretraining `ladder/configs/opt/<OPT>.sh`, peak LR = LR / 3 (`SFT_LR`), 3% warmup, cosine to 0.
- batch: `--use-varlen-dataset` packs examples online (dp_balanced, `--max-seqlen-per-dp-cp-rank 4096`), and the
  global batch counts examples: by default the pipeline's examples per step rounded to the data-parallel size
  (8 GPUs: 312 examples, 16025 iterations; 16 GPUs: 320, 15625; pipeline: 15851). `SFT_GBS` overrides it.
- one checkpoint per epoch (all kept), no validation. Output: `$OUT_ROOT/sft/<run>_<tag>_sft_omi2_500k_ep10/`.

What differs from the pipeline: the packing (online and balanced over the ranks, instead of next-fit
rows) and therefore which examples share a step. The examples, their order, tokens, mask, optimizer, LR schedule
and the number of steps all match.

## 3. Offline eval

```bash
sbatch -J sft-eval -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=128G -t 24:00:00 -o logs/%x-%j.out \
  --export=ALL,CONFIG=llama3_r1,OPT=muon,LR=2.8e-3,HEAD=hfn,MARGIN_D=0.02,PRETRAIN_TAG=tpp100,VLLM_PYTHON=<vllm env>/bin/python \
  --wrap 'srun --ntasks=1 ht-run bash ladder/sft/eval.sh'
# ITERS=all: every epoch checkpoint; TARGET=pretrained: the start checkpoint (pre-SFT reference);
# TASKS / NUM_SAMPLES / MAX_PROBLEMS: quick subsets (written to their own directories)
```

`ladder/sft/eval.sh` exports each checkpoint with `ladder/export/export_hf.sh` (plain Llama / Qwen3, head folded,
Megatron vs HF logits checked), then runs `ladder/sft/eval/eval_vllm.py`. The settings are in
`ladder/sft/eval/math_passk.yaml`, which has the pipeline's tasks, prompts (the SFT template; Countdown's own
few-shot prompt and stop) and offline settings: 64 samples (T 0.6, top_p 0.95) + greedy, all problems,
GSM8K 512 / others 1024 new tokens. Grading uses `math_grader.py`, copied unchanged from the pipeline.
Prompts are tokenized without BOS and passed to vLLM as token ids.
Results: `<sft dir>/eval/iter_<N>/{summary.json,<task>.jsonl}` plus one line per checkpoint in
`<sft dir>/eval/summary.jsonl` (keys as the pipeline's: `gsm8k/pass@1`, `math500/greedy_acc`, ...).
vLLM samples differently from HF generate, so per-sample outputs differ from the pipeline's, but the metric
distributions are the same.

vLLM is not in the NGC image: point `VLLM_PYTHON` at an environment that has it (and transformers, pyarrow,
pyyaml; sympy + antlr4-python3-runtime for the MATH equivalence check). The export runs in the Megatron
environment (`VENV`).

## Tests

`python -m pytest ladder/tests/test_sft_scripts.py -p no:cacheprovider -o addopts=""` (CPU): data preparation
against the pipeline's functions (when ../pretrain_LLM_pipeline exists), boundary / overlong drops, schedule,
eval problems / grading / summary, and the train / eval launchers' plans.
