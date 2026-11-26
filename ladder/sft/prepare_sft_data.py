"""Turn the pipeline's OpenMathInstruct-2 SFT subset into the Megatron --use-varlen-dataset input.

Reproduces pretrain_LLM_pipeline's SFT data (src/data/sft.py, configs/data/sft/openmathinstruct2_*.yaml)
for ladder/sft/train_sft.sh:

    python ladder/sft/prepare_sft_data.py \\
        --input /fast/users/zhuangxinlin/data/sft/openmathinstruct2_500k/train \\
        --tokenizer /fast/users/zhuangxinlin/tokenizers/llama-3 \\
        --output-dir /fast/users/zhuangxinlin/data/sft_megatron/omi2_500k_ep10_llama3 --epochs 10

Output (--output-dir):
  train.parquet  epochs x examples rows (prompt, response, sample_id, epoch) in the pipeline's epoch order.
                 Megatron's VarlenDataset reads rows front to back without shuffling, so the order is set here.
  stats.json     counts, tokens, the pipeline's packed schedule replayed, and the Megatron global batch
                 (examples per step) / train iterations that match it; train_sft.sh reads it.

What matches the pipeline:
  * example = prompt + response + EOS, prompt = the plain-text template with {problem}, response =
    generated_solution; prompt tokens unsupervised, response + EOS supervised. In Megatron the identity chat
    template (--tokenizer-type SFTTokenizer --sft-tokenizer-prompt-format identity) concatenates the user turn
    (prompt) and the assistant turn (response), masks the user turn, and VarlenDataset appends EOS as a target.
  * prompt and response are tokenized separately, without special tokens (no BOS). Megatron tokenizes their
    concatenation and asserts it equals the per-turn tokens, so a row whose BPE merges across the boundary
    would stop training: such rows are dropped here and counted (expected ~0). Their order slots are kept,
    so every other row keeps the pipeline's position.
  * overlong examples (prompt + response + EOS > seq_len) and empty responses are dropped, as overlong: drop.
  * epoch order: random.Random(f"sft:{seed}:{epoch}").shuffle, the pipeline's epoch_order.
  * schedule: the pipeline packs each epoch next-fit into seq_len rows and steps over 32 rows. Megatron packs
    online (dp_balanced) and its global batch counts examples, so the matching global batch is the
    pipeline's examples per step, rounded to a multiple of the data-parallel size.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# configs/data/sft/openmathinstruct2_500k.yaml of pretrain_LLM_pipeline (YAML block scalar: one trailing
# newline, so the prompt ends with "Solution:\n").
PROMPT_TEMPLATE = (
    "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n"
    "\n"
    "Problem:\n"
    "{problem}\n"
    "\n"
    "Solution:\n"
)
RESPONSE_TEMPLATE = "{generated_solution}"
PIPELINE_GLOBAL_BATCH_ROWS = 32

_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")


def render_template(template: str, row: Mapping[str, Any]) -> str:
    """pretrain_LLM_pipeline src/data/sft.py render_template: {col}, {{ / }} literal, other braces kept."""

    def replace(match):
        token = match.group(0)
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        name = match.group(1)
        if name not in row:
            raise KeyError(f"template column {name!r} missing from row with keys {sorted(row)}")
        value = row[name]
        return "" if value is None else str(value)

    return _PLACEHOLDER.sub(replace, template)


def epoch_order(num_examples: int, seed: int, epoch: int) -> List[int]:
    """pretrain_LLM_pipeline src/data/sft.py epoch_order (str seed: SHA-512, platform independent)."""
    order = list(range(num_examples))
    random.Random(f"sft:{int(seed)}:{int(epoch)}").shuffle(order)
    return order


def pack_next_fit(lengths: Sequence[int], order: Iterable[int], seq_len: int) -> List[Tuple[int, ...]]:
    """pretrain_LLM_pipeline src/data/sft.py pack_next_fit: examples whole and in order, rows <= seq_len."""
    rows: List[Tuple[int, ...]] = []
    current: List[int] = []
    used = 0
    for index in order:
        n = int(lengths[index])
        if n > seq_len or n <= 0:
            raise ValueError(f"example {index} has length {n}, outside (0, seq_len={seq_len}]")
        if used + n > seq_len:
            rows.append(tuple(current))
            current, used = [], 0
        current.append(int(index))
        used += n
    if current:
        rows.append(tuple(current))
    return rows


def global_batch_for(examples_per_step: float, dp_size: int) -> int:
    """Examples per step rounded to a multiple of the data-parallel size (micro batch 1)."""
    return dp_size * max(1, round(examples_per_step / dp_size))


def list_data_files(path: str) -> List[str]:
    """.jsonl/.json/.parquet files under path, names sorted at every level, hidden entries skipped."""
    if os.path.isfile(path):
        return [path]
    out = []
    for root, dirs, names in os.walk(path):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out.extend(
            os.path.join(root, n)
            for n in sorted(names)
            if not n.startswith(".") and n.endswith((".jsonl", ".json", ".parquet"))
        )
    if not out:
        raise FileNotFoundError(f"no .jsonl/.json/.parquet files under {path}")
    return out


def read_rows(path: str, columns: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in list_data_files(path):
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq

            table = pq.read_table(f)
            keep = [c for c in table.column_names if c in columns]
            rows.extend(table.select(keep).to_pylist())
        else:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        rows.append({k: row[k] for k in columns if k in row})
    return rows


def _batched(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def encode(
    prompts: Sequence[str], responses: Sequence[str], tokenize, seq_len: int, batch: int = 4096
) -> Tuple[List[Optional[int]], List[str], List[int]]:
    """Per row: example length (prompt + response + EOS) or None, the reason it is dropped, prompt length.

    ``tokenize(list_of_texts) -> list of token-id lists`` (no special tokens). Reasons: "" kept,
    "empty_response", "overlong" (pipeline drops), "boundary" (kept by the pipeline; Megatron cannot
    represent it, see the module docstring).
    """
    lengths: List[Optional[int]] = []
    reasons: List[str] = []
    prompt_lens: List[int] = []
    for start in range(0, len(prompts), batch):
        p_batch = list(prompts[start : start + batch])
        r_batch = list(responses[start : start + batch])
        p_ids = tokenize(p_batch)
        r_ids = tokenize(r_batch)
        joint = tokenize([p + r for p, r in zip(p_batch, r_batch)])
        for p, r, j in zip(p_ids, r_ids, joint):
            if not p:
                raise ValueError("a prompt rendered to zero tokens; the template needs fixed text")
            prompt_lens.append(len(p))
            n = len(p) + len(r) + 1
            if not r:
                lengths.append(None)
                reasons.append("empty_response")
            elif n > seq_len:
                lengths.append(None)
                reasons.append("overlong")
            else:
                lengths.append(n)
                reasons.append("" if list(j) == list(p) + list(r) else "boundary")
    return lengths, reasons, prompt_lens


def plan(
    lengths: Sequence[Optional[int]],
    reasons: Sequence[str],
    prompt_lens: Sequence[int],
    *,
    epochs: int,
    seed: int,
    seq_len: int,
    dp_sizes: Sequence[int],
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Per-epoch row orders for Megatron (pipeline order minus boundary rows) and the stats."""
    # The pipeline's examples: every row it keeps (boundary rows included), in source order.
    pipe = [i for i, n in enumerate(lengths) if n is not None]
    pipe_lengths = [lengths[i] for i in pipe]
    keep = set(i for i in pipe if reasons[i] == "")
    orders: List[List[int]] = []
    pipe_rows = []
    for epoch in range(epochs):
        order = epoch_order(len(pipe), seed, epoch)
        pipe_rows.append(len(pack_next_fit(pipe_lengths, order, seq_len)))
        orders.append([pipe[k] for k in order if pipe[k] in keep])
    total_rows = sum(pipe_rows)
    pipe_steps = total_rows // PIPELINE_GLOBAL_BATCH_ROWS
    examples_per_step = len(pipe) * epochs / max(1, pipe_steps)
    kept = len(keep)
    tokens = sum(lengths[i] for i in keep)
    supervised = sum(lengths[i] - prompt_lens[i] for i in keep)   # response + EOS
    megatron = {}
    for dp in dp_sizes:
        gbs = global_batch_for(examples_per_step, dp)
        megatron[str(dp)] = {
            "global_batch_size": gbs,
            "train_iters": kept * epochs // gbs,
            "iters_per_epoch": kept // gbs,
            "tokens_per_step": tokens / kept * gbs,
        }
    stats = {
        "seq_len": seq_len,
        "epochs": epochs,
        "seed": seed,
        "rows_in": len(lengths),
        "dropped_empty_response": sum(r == "empty_response" for r in reasons),
        "dropped_overlong": sum(r == "overlong" for r in reasons),
        "dropped_boundary": sum(r == "boundary" for r in reasons),
        "examples": kept,
        "rows_out": kept * epochs,
        "tokens_per_epoch": tokens,
        "supervised_tokens_per_epoch": supervised,
        "mean_example_tokens": tokens / max(1, kept),
        "pipeline": {
            "examples": len(pipe),
            "rows_per_epoch": pipe_rows,
            "global_batch_rows": PIPELINE_GLOBAL_BATCH_ROWS,
            "total_steps": pipe_steps,
            "examples_per_step": examples_per_step,
            "tokens_per_step": sum(pipe_lengths) * epochs / max(1, pipe_steps),
        },
        "megatron_by_dp_size": megatron,
    }
    return orders, stats


def megatron_check(tokenizer_path: str, rows: Sequence[Tuple[str, str]], prompt_lens: Sequence[int],
                   eos_id: int) -> None:
    """Run Megatron's own SFT tokenizer (identity format) on a few kept rows: same tokens, same mask."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from megatron.core.tokenizers.text.libraries.sft_tokenizer import IGNORE_INDEX, SFTTokenizer

    tok = SFTTokenizer(tokenizer_path, "identity")
    for (prompt, response), n_prompt in zip(rows, prompt_lens):
        conversation = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        tokens, targets = tok.tokenize_conversation(conversation, return_target=True, add_generation_prompt=False)
        tokens, targets = list(tokens), list(targets)
        if tokens and tokens[-1] == eos_id:
            raise AssertionError("the identity template already ends with EOS; VarlenDataset would not add one")
        if any(t != IGNORE_INDEX for t in targets[:n_prompt]) or IGNORE_INDEX in targets[n_prompt:]:
            raise AssertionError("Megatron's mask differs: prompt tokens must be ignored, response tokens kept")
        if len(tokens) <= n_prompt:
            raise AssertionError("Megatron's tokens have no response")


def write_parquet(path: str, rows: Sequence[Mapping[str, Any]], prompts: Sequence[str], responses: Sequence[str],
                  orders: Sequence[Sequence[int]], chunk: int = 100_000) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("prompt", pa.string()), ("response", pa.string()),
                        ("sample_id", pa.int64()), ("epoch", pa.int32())])
    tmp = path + ".tmp"
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for epoch, order in enumerate(orders):
            for part in _batched(order, chunk):
                writer.write_table(pa.table({
                    "prompt": [prompts[i] for i in part],
                    "response": [responses[i] for i in part],
                    "sample_id": [int(rows[i].get("sample_id", i)) for i in part],
                    "epoch": [epoch] * len(part),
                }, schema=schema))
    os.replace(tmp, path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="pipeline SFT subset train split (file or directory)")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer directory (the pretraining one)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="pipeline shuffle_seed")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dp-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--megatron-check-rows", type=int, default=200,
                        help="check this many kept rows with Megatron's SFT tokenizer (0 skips)")
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    t0 = time.time()
    rows = read_rows(args.input, ["problem", "generated_solution", "sample_id"])
    print(f"read {len(rows)} rows from {args.input} ({time.time() - t0:.0f}s)", flush=True)
    prompts = [render_template(PROMPT_TEMPLATE, r) for r in rows]
    responses = [render_template(RESPONSE_TEMPLATE, r) for r in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.eos_token_id is None:
        raise SystemExit("the tokenizer has no eos_token_id")

    def tokenize(texts):
        return tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                         return_token_type_ids=False, verbose=False)["input_ids"]

    lengths, reasons, prompt_lens = encode(prompts, responses, tokenize, args.seq_len)
    print(f"tokenized ({time.time() - t0:.0f}s)", flush=True)
    orders, stats = plan(lengths, reasons, prompt_lens, epochs=args.epochs, seed=args.seed,
                         seq_len=args.seq_len, dp_sizes=args.dp_sizes)
    stats.update(input=os.path.abspath(args.input), tokenizer=os.path.abspath(args.tokenizer),
                 eos_token_id=int(tokenizer.eos_token_id), prompt_template=PROMPT_TEMPLATE)

    if args.megatron_check_rows > 0:
        kept = [i for i, r in enumerate(reasons) if r == ""][: args.megatron_check_rows]
        megatron_check(args.tokenizer, [(prompts[i], responses[i]) for i in kept],
                       [prompt_lens[i] for i in kept], int(tokenizer.eos_token_id))
        print(f"Megatron SFT tokenizer (identity) agrees on {len(kept)} rows", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    write_parquet(os.path.join(args.output_dir, "train.parquet"), rows, prompts, responses, orders)
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "prompt_template"}, indent=2))
    print(f"wrote {args.output_dir}/train.parquet ({stats['rows_out']} rows) and stats.json "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
