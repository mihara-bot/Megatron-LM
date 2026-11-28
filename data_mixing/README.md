# data_mixing: mixed pretraining text, then Megatron data

Builds a pretraining mixture as **text** (parquet) on the data cluster, copies it to the training cluster,
and tokenizes it there with the unchanged `ladder/data` scripts, once per tokenizer family. The text step is
tokenizer-free, so llama3, qwen3 and later DeepSeek all train on exactly the same documents.

Independent of pretrain_LLM_pipeline: the v1 build (ClimbMix v1 m1, 50B) is used only through the values
frozen in `configs/climbmix_v1_m1_reference.json`.

## The first mix: `climbmix_m1_200b`

- **Purpose:** 200B Llama-3.1 tokens for Llama3 R3 and R4 at TPP100. R3 needs 60000 x 384 x 4096 = 94.4B
  tokens and R4 needs 85000 x 512 x 4096 = 178.3B. Both use the same merged prefix, and Megatron samples
  from all of it.
- **Weights:** the v1 m1 cluster weights (α=0.5), copied, not recomputed, so R3/R4 see the same distribution
  as R1/R2.
- **Order within a cluster:** each cluster continues its v1 stream after the v1 validation documents.
  The v1 50B training documents are therefore the start of pass `e0`.
- **Repeats:** clusters whose quota exceeds their documents are read again (`e1`, `e2`, ...). At 200B that
  is 13 clusters, up to 3.35 passes (c15), about 23B repeated tokens (11.5%).
- **Validation:**
  - `val/all` is the v1 `val_all`: the same 16,228 documents in the same order, about 10M tokens. Curves stay
    comparable with R1/R2.
  - `val/<group>`: the last 5M tokens of every cluster, which the v1 50B build never read. They are taken
    out of training and grouped by the 8 labels in `configs/climbmix_groups_v1.json` (5–20M tokens per
    group, 100M in total).

`python data_mixing/mix_text.py estimate --spec data_mixing/configs/mixes/climbmix_m1_200b.json` prints the
per-cluster quotas and passes from the reference numbers. It does not read any data.

## Pipeline

```
orion (one big CPU node, plain bash)            Polaris (SLURM CPU job per family)
  run_orion.sh test       synthetic e2e test
  run_orion.sh plan   ->  plan.json               copy OUT (without .cache/)
  run_orion.sh build  ->  train/ val/       ==>   tokenize_polaris.sh verify    files.sha256 check
  run_orion.sh verify ->  files.sha256, _DONE     tokenize_polaris.sh tokenize  ladder/data/tokenize_dataset.py
                                                  tokenize_polaris.sh merge     ladder/data/merge_indexed.py (train/)
                                                  tokenize_polaris.sh inspect   ladder/data/inspect_indexed.py
                                                  tokenize_polaris.sh report    exact counts vs plan -> report.json
                                                  tokenize_polaris.sh cleanup   delete train pieces
```

### 1. orion: text

```bash
cd Megatron-LM
bash data_mixing/run_orion.sh test        # first time: the end-to-end test on synthetic parquet
bash data_mixing/run_orion.sh plan        # minutes: footers, indexing, cuts, hash checks
bash data_mixing/run_orion.sh build       # copying; re-run the same command after an interruption
bash data_mixing/run_orion.sh verify      # files.sha256, manifest.json, _DONE
```

Defaults: `SRC=/mnt/oss-zhuangxinlin/ClimbMix`, `OUT=/mnt/oss-zhuangxinlin/climbmix_mix/m1_200b`,
`WORKERS=64`. Raise `WORKERS` until the storage is the bottleneck. `plan` also estimates the token counts
with the tokenizers in `TOKENIZERS` (llama3 and qwen3 by default).

What `plan` checks before writing anything:
- **Source download:** every cluster's footer rows and text bytes equal the reference, i.e. the same download.
- **v1 validation:** the first `val_docs` documents of every cluster have the reference byte count and
  sha256, and the rebuilt `val_all` order matches the reference sha256.
- **v1 training documents:** pass `e0` of every cluster contains its v1 50B training documents.
- **Repeats:** no cluster needs more than `max_epochs` passes.

### 2. Copy to Polaris

Copy `OUT` without `.cache/` to `/fast/users/zhuangxinlin/data/climbmix_mix/m1_200b`, and only after `_DONE`
exists. `verify` writes `_DONE` last, and `build` or a failed `verify` deletes it.

### 3. Polaris: tokenize, one job per family

```bash
cd Megatron-LM   # Megatron environment
sbatch --partition=polaris --account=polaris --job-name=tok-llama3 --nodes=1 --ntasks=1 \
    --cpus-per-task=128 --mem=512G --time=24:00:00 --chdir="$PWD" --export=ALL,FAMILY=llama3 \
    --output="tok-llama3-%j.out" \
    --wrap='srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" ht-run bash -e data_mixing/tokenize_polaris.sh all'
# same with FAMILY=qwen3; later FAMILY=deepseek TOK=/path/to/its/base/tokenizer
FAMILY=llama3 bash data_mixing/tokenize_polaris.sh cleanup   # after report.json says "ok": true
```

**What `report` checks:**
- **Per-file counts:** every planned file was tokenized with exactly the planned documents and UTF-8 bytes.
- **Merged prefix:** `train_merged` holds every training token.
- **val/all:** byte-identical to the v1 tokenized `val_all` of the same family.
- **Requirements:** the token totals meet the spec's requirements. These are Llama numbers; other families
  are checked against them as a rough bound.

**What `report` prints:** exact tokens and share per cluster, and the environment for training.

Disk: about 4 bytes per token for the pieces and the same again for `train_merged`, so ~0.8 TB each at 200B,
until cleanup.

### 4. Training

`ladder/train.sh` needs no change. Before launching, set the variables that `report` printed:

```bash
export DATA_ROOT=/fast/users/zhuangxinlin/megatron_data/llama3/climbmix_m1_200b
export VALID_DATA="$DATA_ROOT/val/all/val $DATA_ROOT/val/science_nature/val $DATA_ROOT/val/consumer/val \
  $DATA_ROOT/val/games_arts/val $DATA_ROOT/val/health/val $DATA_ROOT/val/education/val $DATA_ROOT/val/society/val \
  $DATA_ROOT/val/business/val $DATA_ROOT/val/computing/val"
export VALID_NAMES="all science_nature consumer games_arts health education society business computing"
```

- **Data order:** `TRAIN_DATA` defaults to `$DATA_ROOT/train_merged`. Never add `--split`: the merged file is
  ordered by cluster, and Megatron's per-epoch document shuffle is what mixes it.
- **Epoch check:** on the first run, check that the log says `total number of epochs: 1` for the training set.
- **Validation tags:** with several validation sets, the TensorBoard tags are `lm loss validation-<name>`.
  R1/R2 logged `lm loss validation`, so compare their curves with `-all`.
- **Eval cost:** about 110M validation tokens per eval, roughly +3.5% (R4) or +4.7% (R3) at `EVAL_INTERVAL=500`.

## Output layout (orion `OUT`, copied as is)

```
plan.json                                 operations, cuts, hashes, estimates (written by plan)
train/cluster_XX/e<p>_<source file>.parquet   pass p of cluster XX: source files copied whole, sliced at the cuts
val/all/val.parquet                       v1 val_all (text, cluster_id)
val/<group>/val.parquet                   tail validation documents of the group's clusters, shuffled
files.sha256  manifest.json  _DONE        written by verify
.cache/                                   footer + index caches: do not copy
```

`ladder/data/tokenize_dataset.py` mirrors this tree, so on Polaris each `val/<name>/val` is a ready-made
Megatron prefix and `train/` is merged into `train_merged`.

## Semantics

- **Stream:** a cluster's documents are the non-empty `text` values of its parquet files in name order.
  Null and empty texts are skipped, as `tokenize_dataset.py` skips them.
- **Cuts:** every stream is cut into
  `[head val: the reference val_docs] [train region] [tail val: the last documents, >= tail bytes]`.
- **Quota:** a cluster's quota is `weight x train_tokens x bytes_per_token`, in bytes, with the reference
  weights and bytes/token. Training reads the region from its start until the quota is reached, and the
  document that crosses it is included, as in the v1 builder. When the region is exhausted, reading
  restarts at the region start.
- **Files:** a pass becomes whole-file copies, plus slices at the head, the tail and the crossing file.
  `--rechunk` rewrites every file instead, with `row_group_rows` rows per row group, for sources whose row
  groups are too large for `tokenize_dataset.py` to split across workers.
- **Token counts:** before tokenizing they are estimates: bytes divided by the reference tokenizer's
  bytes/token. `report` gives the exact counts.

## New mixes

Write a spec in `configs/mixes/`:
- `weights` is `"reference"`, `{"alpha": a}` or `{"explicit": {"<cluster>": w}}`;
- also set `train_tokens`, `max_epochs` and `val_tail_tokens_per_cluster`.

Then run `run_orion.sh` with `SPEC=... OUT=...`. Mixes with the same reference and
`val_tail_tokens_per_cluster` have identical validation sets, whatever their weights, so their validation
losses are comparable.

## Tests

```bash
python data_mixing/tests/test_mixlib.py         # stdlib: plans vs a brute-force reading, frozen m1_200b numbers
python data_mixing/tests/test_polaris_tools.py  # stdlib: verify_files.py, report_tokenized.py
python data_mixing/tests/test_parquet_e2e.py    # pyarrow + numpy: plan/build/verify/resume on synthetic parquet
DM_TOKENIZER=/path/to/tokenizer python data_mixing/tests/test_parquet_e2e.py   # + ladder/data tokenize/merge + report
```

On Polaris, `FAMILY=llama3 bash data_mixing/tokenize_polaris.sh test` runs all of them, including the real
tokenize, in the Megatron environment. The no-GPU sbatch line is at the top of that script. All of them
also run under pytest (`-p no:cacheprovider -o addopts=""`).
