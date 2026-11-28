# Pretraining eval: HF export + lm-eval on vLLM

Pretraining checkpoints are exported to plain Hugging Face models (`ladder/export/export_hf.sh`, the exporter SFT
uses too) and evaluated with lm-evaluation-harness on its vLLM backend. These are two jobs in two environments: the
export runs in the Megatron environment, the eval in the vLLM one.

| step | script | env | GPUs | output |
|---|---|---|---|---|
| datasets (once) | `ladder/eval/download_datasets.sh` | any python3 | 0 | `$EVAL_DATA_DIR/<org>/<name>` |
| lm-eval install (once) | `pip install -e` (below) | vLLM | 0 | lm_eval in the vLLM venv |
| export | `ladder/eval/export.sh` | Megatron | 1 | `<run>/hf/<tag>_iter<N>` |
| eval | `ladder/eval/lm_eval.sh` | vLLM | 1 | `<run>/lm_eval/<tag>_iter<N>/results.json`, `<run>/lm_eval/summary.jsonl` |
| both, per run | `ladder/eval/run_eval.sh` | (submits) | | export job, then eval job with afterok |

## 1. Once per cluster

Datasets: the polaris nodes cannot reach the Hub. This script was moved from compass-merge. It fetches the task
repos through hf-mirror with `hfd.sh` (taken from next to the repo, or set `HFD`) into
`/fast/users/zhuangxinlin/data/lm_eval_hf`. lm-eval then runs from that directory with the Hub offline, because
`load_dataset("<org>/<name>")` looks for a directory at that relative path before it asks the Hub.

```bash
TASK_SET=small bash ladder/eval/download_datasets.sh   # what the default task set reads; no TASK_SET = every set
```

lm-eval goes into the vLLM venv built by `ladder/env/setup_vllm_env.sh`, never into the Megatron venv. Install it
from the clone next to Megatron-LM as an editable package. Use the venv's own interpreter, so that pip cannot be
another environment's. Constrain the install to the venv's lock, so that transformers / tokenizers keep the versions
of the Megatron environment and torch / vLLM do not move:

```bash
cd ../lm-evaluation-harness
/fast/users/zhuangxinlin/envs/vllm/bin/python -m pip install -e . -c /fast/users/zhuangxinlin/envs/vllm/requirements.lock
cd ../Megatron-LM
bash ladder/sft/eval/run_isolated.sh /fast/users/zhuangxinlin/envs/vllm/bin/python ladder/env/check_vllm_env.py --lm-eval
```

- A venv built before `--seed` was added to the setup script has no pip of its own (`No module named pip`). Use
  `uv pip install --python <venv>/bin/python -e . -c <venv>/requirements.lock` instead.
- `minerva_math` (TASK_SET=all) also needs the `[math]` extra.

## 2. Export (Megatron env, 1 GPU)

```bash
CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 \
  sbatch -J pre-export -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=128G -t 02:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/eval/export.sh'
```

- **Which run:** `export.sh` names the run as `ladder/train.sh` does: `<CONFIG>_<OPT>_lr<LR><head tag>`. `OPT`
  defaults to muon, and `RUN` overrides the name. The head variables must be the run's own; `load_checkpoint`
  refuses another head.
- **What gets exported:** a SphereFace-R head is folded into a plain `lm_head`, so a margin run exports to the same
  Llama / Qwen3 architecture as its baseline. The token margin itself only shapes the training loss. Every export
  checks Megatron's logits against the exported model's.
- **Which checkpoints:** `ITERS=final` (the default) exports, for each checkpoint directory (`TAGS`, default every
  one), the checkpoint where the LR has decayed to 0. For `tpp<T>` that is the `--exit-interval` of the branch; for
  `main` it is `--train-iters` (TPP_MAX). Both are read from the last `cmd_<tag>_*.sh` the run wrote. Some tags are
  skipped with a note: a stable run stopped at `STOP_TPP` (it is mid-schedule), and a branch that is still running.
- **LR trajectories:** `ITERS=all` exports every kept checkpoint. These are mid-schedule, not decayed.
- **Tokenizer:** the training run's own, read from its command.

## 3. Eval (vLLM env, 1 GPU)

```bash
MODEL="$OUT_ROOT/runs/llama3_r1_muon_lr2.8e-3_hfn_d0.02/hf/*_iter*" VLLM_PYTHON=/fast/users/zhuangxinlin/envs/vllm/bin/python \
  sbatch -J lm-eval -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=128G -t 06:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/eval/lm_eval.sh'
```

**Tasks.** `TASK_SET=small` is the default. It holds 12 log-likelihood tasks, 0-shot, all of which give signal at
0.4-8B / TPP 20-100:

- hellaswag, arc_easy, arc_challenge, piqa, winogrande, openbookqa
- sciq, boolq, lambada_openai, copa, social_iqa, commonsense_qa

`core`, `extended` and `all` are compass-merge's presets. Each one adds to the previous: mmlu, race, wikitext, ... in
`core`; generation tasks in `extended` and `all`. `TASKS=a,b` overrides the preset, `NUM_FEWSHOT` sets the shots for
every task, and `LIMIT` caps each task, for smoke tests.

**Token convention** (`TOKENS=auto`). Ladder data is tokenized EOS-only: `text + EOS`, no BOS. For a ladder export
(any directory with `export_meta.json`), lm-eval therefore runs with `add_bos_token=False,prefix_token_id=<EOS id>`:

- The Llama-3 tokenizer would otherwise prepend `<|begin_of_text|>`. These models never saw that token; its embedding
  row was only ever touched by weight decay.
- The vLLM backend's prefix token, used as the context of empty-context requests and as the first token of
  `loglikelihood_rolling` (wikitext), would otherwise be the BOS. With this setting it is the EOS, which precedes
  every document in training.

The EOS id comes from the export's `config.json`. The Qwen3 base tokenizer has no BOS and gets the same EOS prefix.
Other models (official checkpoints, as references) keep lm-eval's defaults. `TOKENS=eos` or `TOKENS=default`
forces one convention, and the results then go to their own directory.

**Outputs.** For a model at `<X>/hf/<name>`:

- `<X>/lm_eval/<name>/results.json`, which is lm-eval's `results_<date>.json`, plus the log.
- One line in `<X>/lm_eval/summary.jsonl` with:
  - run, tag and iteration, LM head and margin;
  - `score/<task>` (acc_norm, else acc);
  - `avg_score` over the requested tasks;
  - every `<task>/<metric>`.

A non-default TASK_SET / TASKS / NUM_FEWSHOT / LIMIT / TOKENS adds a suffix to the output directory, so a subset
never overwrites the full results. A model whose `results.json` exists is skipped (`FORCE=1` reruns it).

**Safeguards.**

- Before each model, the script checks that lm_eval and vllm import from the vLLM venv.
- It also checks that transformers / tokenizers equal the versions in `export_meta.json`. `ALLOW_HF_MISMATCH=1`
  only warns.
- Exports that failed their logits check are refused.

## 4. Chains

`ladder/eval/run_eval.sh` submits, per run, the export job and then the eval job (afterok). The default chains are
the token-margin runs and their baselines: R2 and R1 × plain / hfn / sfn. `RUNGS`, `CHAINS`, `IDLE` and `AFTER=<job>`
work as in `ladder/sft/run_sft.sh`, and `DRY_RUN=1` prints the sbatch commands. `STEPS=eval TASK_SET=core` re-runs
only the eval, with a larger set.

## Tests

`python -m pytest ladder/tests/test_eval_scripts.py -p no:cacheprovider -o addopts=""` runs on the CPU and needs no
dependencies. It covers:

- export.sh: run naming, tag / checkpoint selection, head passthrough, skip logic;
- lm_eval.sh: commands, token convention, output layout, refused exports;
- lm_eval_util.py collect;
- run_eval.sh: job graph.

The first GPU check is one export with `TASKS=arc_easy,hellaswag LIMIT=200`.
