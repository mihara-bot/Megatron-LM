"""Generate the margin toy's data (ladder/toy/TOY.md): the graph, pretraining walks, SFT path examples.

    python ladder/toy/make_data.py --out $TOY_ROOT/data [--check-tokenizer $TOY_ROOT/tokenizer]

Writes under --out:
  graph.json              the graph, the (s, t) pair split, these settings, exact reference numbers
                          (entropy rate = the plain-CE floor of pretraining validation, path counts, ...)
  pretrain/train.jsonl    one {"text": "<walk> n.. n.."} per walk; ladder/data/tokenize_dataset.py appends <eos>
  pretrain/val.jsonl
  sft/<mode>/train.parquet  ladder/sft/train_sft.sh input (columns prompt, response, sample_id, epoch; Megatron's
  sft/<mode>/stats.json     VarlenDataset reads prompt/response as one user + one assistant turn, prompt masked)

SFT modes (ladder/toy/graph.py): cond, uniform, canon. Each epoch has --sft-reps examples per training pair,
shuffled; cond and uniform draw fresh paths every epoch, canon repeats the pair's single path. train_sft.sh saves
one checkpoint per epoch, so --sft-epochs sets how finely SFT can be followed over time.

Standard library, except pyarrow for the parquet files and Megatron + transformers for --check-tokenizer.
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graph as G  # noqa: E402

SFT_MODES = ("cond", "uniform", "canon")


def write_walks(path: str, graph: G.Graph, n: int, rng: random.Random, chunk: int = 100_000) -> int:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for start in range(0, n, chunk):
            f.write("".join(json.dumps({"text": G.walk_text(graph.sample_walk(rng))}) + "\n"
                            for _ in range(min(chunk, n - start))))
    os.replace(tmp, path)
    return n


def sft_rows(graph: G.Graph, pairs, mode: str, reps: int, epochs: int, seed: int):
    """[(prompt, response, sample_id, epoch)] in training order: epoch by epoch, each epoch shuffled."""
    out = []
    canon = {}
    for epoch in range(epochs):
        rng = random.Random(f"toy-sft:{seed}:{mode}:{epoch}")
        rows = []
        for i, (s, t) in enumerate(pairs):
            for r in range(reps):
                if mode == "canon":
                    if (s, t) not in canon:
                        canon[(s, t)] = graph.sample_path(s, t, "canon", rng)
                    path = canon[(s, t)]
                else:
                    path = graph.sample_path(s, t, mode, rng)
                rows.append((G.path_prompt_text(s, t), G.path_response_text(path), i * reps + r, epoch))
        rng.shuffle(rows)
        out += rows
    return out


def write_sft(out_dir: str, rows, stats: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(out_dir, exist_ok=True)
    schema = pa.schema([("prompt", pa.string()), ("response", pa.string()),
                        ("sample_id", pa.int64()), ("epoch", pa.int32())])
    tmp = os.path.join(out_dir, "train.parquet.tmp")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as w:
        for start in range(0, len(rows), 200_000):
            part = rows[start:start + 200_000]
            w.write_table(pa.table({"prompt": [r[0] for r in part], "response": [r[1] for r in part],
                                    "sample_id": [r[2] for r in part], "epoch": [r[3] for r in part]},
                                   schema=schema))
    os.replace(tmp, os.path.join(out_dir, "train.parquet"))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


def sft_stats(rows, mode: str, n_pairs: int, reps: int, epochs: int, seq_len: int, examples_per_step: int) -> dict:
    """The fields ladder/sft/train_sft.sh reads (examples, epochs, rows_out, seq_len, pipeline.examples_per_step)
    plus a description of the data."""
    lengths = [len(p.split()) + len(r.split()) + 1 for p, r, _, _ in rows]   # + the <eos> VarlenDataset appends
    return {"mode": mode, "train_pairs": n_pairs, "reps": reps, "examples": n_pairs * reps, "epochs": epochs,
            "rows_out": len(rows), "seq_len": seq_len, "max_example_tokens": max(lengths),
            "tokens": sum(lengths), "pipeline": {"examples_per_step": examples_per_step}}


def check_with_megatron(tokenizer_dir: str, rows, n: int = 64) -> None:
    """Megatron's SFT tokenizer (identity format) must give prompt ids + response ids, prompt masked."""
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
    from megatron.core.tokenizers.text.libraries.sft_tokenizer import IGNORE_INDEX, SFTTokenizer

    tok = SFTTokenizer(tokenizer_dir, "identity")
    for prompt, response, _, _ in rows[:n]:
        conv = [{"role": "system", "content": ""}, {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
        tokens, target = tok.tokenize_conversation(conv, return_target=True, add_generation_prompt=False)
        _, s, t, _ = prompt.split()
        want_prompt = G.path_prompt_ids(int(s[1:]), int(t[1:]))
        want_resp = [G.node_token(int(x[1:])) for x in response.split()]
        tokens, target = list(map(int, tokens)), list(map(int, target))
        if tokens != want_prompt + want_resp:
            raise SystemExit(f"SFTTokenizer gives {tokens} for {prompt!r} + {response!r}")
        if target != [IGNORE_INDEX] * len(want_prompt) + want_resp:
            raise SystemExit(f"SFTTokenizer target {target} for {prompt!r} + {response!r}")
    print(f"Megatron SFTTokenizer (identity) agrees on {min(n, len(rows))} rows", flush=True)


def reference(graph: G.Graph, train, held) -> dict:
    mu = graph.layer_marginals()
    multi = [u for u in graph.nodes if not graph.is_last(u) and graph.out_degree(u) > 1]
    det = [u for u in graph.nodes if not graph.is_last(u) and graph.out_degree(u) == 1]
    spans = {}
    for s, t in train + held:
        span = graph.layer_of[t] - graph.layer_of[s]
        spans.setdefault(span, {"pairs": 0, "paths": 0})
        spans[span]["pairs"] += 1
        spans[span]["paths"] += graph.path_count(t)[s]
    return {
        "entropy_rate": graph.entropy_rate(),
        "nodes": len(graph.nodes), "multi_nodes": len(multi), "det_nodes": len(det),
        "multi_row_entropy_mean": sum(G.entropy(p for _, p in graph.succ[u]) for u in multi) / len(multi),
        "visit_prob_min": min(mu.values()), "visit_prob_max": max(mu.values()),
        "pairs_train": len(train), "pairs_heldout": len(held),
        "by_span": {str(k): {"pairs": v["pairs"], "mean_paths": v["paths"] / v["pairs"]} for k, v in sorted(spans.items())},
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0, help="graph and pair split")
    p.add_argument("--data-seed", type=int, default=0, help="walk and path sampling")
    p.add_argument("--train-walks", type=int, default=4_000_000, help="40M tokens; the loader repeats epochs")
    p.add_argument("--val-walks", type=int, default=50_000)
    p.add_argument("--min-span", type=int, default=2, help="path task: t at least this many layers after s")
    p.add_argument("--heldout-frac", type=float, default=0.2)
    p.add_argument("--sft-modes", default=",".join(SFT_MODES))
    p.add_argument("--sft-reps", type=int, default=16, help="examples per training pair per epoch")
    p.add_argument("--sft-epochs", type=int, default=8)
    p.add_argument("--sft-examples-per-step", type=int, default=256, help="SFT global batch (examples)")
    p.add_argument("--seq-len", type=int, default=256, help="must equal SEQ_LEN of ladder/configs/toy_*.sh")
    p.add_argument("--check-tokenizer", help="tokenizer dir: check the SFT rows with Megatron's SFTTokenizer")
    p.add_argument("--skip-pretrain", action="store_true", help="only graph.json and SFT data")
    args = p.parse_args(argv)

    modes = [m for m in args.sft_modes.split(",") if m]
    bad = set(modes) - set(SFT_MODES)
    if bad:
        raise SystemExit(f"unknown SFT modes {sorted(bad)}; choose from {SFT_MODES}")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    graph = G.build_graph(args.seed)
    train, held = G.split_pairs(graph.pairs(args.min_span), graph, args.heldout_frac, args.seed)
    ref = reference(graph, train, held)
    with open(os.path.join(args.out, "graph.json"), "w") as f:
        json.dump({"graph": graph.to_json(), "pairs": {"train": train, "heldout": held},
                   "settings": vars(args), "reference": ref}, f, indent=1)
    print(json.dumps(ref, indent=1), flush=True)

    if not args.skip_pretrain:
        os.makedirs(os.path.join(args.out, "pretrain"), exist_ok=True)
        for split, n in (("train", args.train_walks), ("val", args.val_walks)):
            rng = random.Random(f"toy-walks:{args.data_seed}:{split}")
            write_walks(os.path.join(args.out, "pretrain", f"{split}.jsonl"), graph, n, rng)
            print(f"pretrain/{split}.jsonl: {n} walks, {n * (graph.num_layers + 2)} tokens "
                  f"({time.time() - t0:.0f}s)", flush=True)

    for mode in modes:
        rows = sft_rows(graph, train, mode, args.sft_reps, args.sft_epochs, args.data_seed)
        stats = sft_stats(rows, mode, len(train), args.sft_reps, args.sft_epochs, args.seq_len,
                          args.sft_examples_per_step)
        if stats["max_example_tokens"] > args.seq_len:
            raise SystemExit(f"SFT examples up to {stats['max_example_tokens']} tokens > seq_len {args.seq_len}")
        if args.check_tokenizer:
            check_with_megatron(args.check_tokenizer, rows)
        write_sft(os.path.join(args.out, "sft", mode), rows, stats)
        print(f"sft/{mode}: {stats['rows_out']} rows ({stats['examples']} per epoch x {stats['epochs']}), "
              f"{stats['tokens']} tokens ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
