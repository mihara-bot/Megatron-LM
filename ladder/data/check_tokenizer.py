"""Fail fast when the tokenizer does not belong to the config or to the data.

    python ladder/data/check_tokenizer.py --tokenizer DIR --expect-vocab N PREFIX [PREFIX ...]

Called by ladder/train.sh before every launch. Two checks:
  1. config <-> tokenizer: len(tokenizer) == the config's TOKENIZER_VOCAB (Llama-3.1 128256, Qwen3 151669).
  2. tokenizer <-> data: tokenize_dataset.py writes manifest_*.json (vocab size and EOS id of the tokenizer
     it used) into its --output-root. For each data prefix, the nearest manifest in the prefix's directory
     or up to MAX_UP parents above it must match the tokenizer.
A mismatch is otherwise silent: Llama-3 token ids are all valid Qwen3 ids, so training runs, just on
garbage. Prefixes without a manifest are only warned about (data made by other tools).
"""
import argparse
import glob
import json
import os
import sys

MAX_UP = 3


def find_manifests(prefix):
    d = os.path.dirname(os.path.abspath(prefix))
    for _ in range(MAX_UP + 1):
        found = sorted(glob.glob(os.path.join(d, "manifest_*.json")))
        if found:
            return found
        d = os.path.dirname(d)
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--expect-vocab", type=int, required=True, help="the config's TOKENIZER_VOCAB")
    ap.add_argument("prefixes", nargs="+", help="Megatron data prefixes (train and validation)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab, eos = len(tok), tok.eos_token_id
    errors = []
    if vocab != args.expect_vocab:
        errors.append(f"tokenizer {args.tokenizer} has {vocab} tokens, the config expects {args.expect_vocab}")

    for prefix in args.prefixes:
        manifests = find_manifests(prefix)
        if not manifests:
            print(f"check_tokenizer: warning: no manifest_*.json near {prefix}, data tokenizer not checked",
                  file=sys.stderr)
            continue
        for path in manifests:
            with open(path) as f:
                info = json.load(f)["tokenizer"]
            if info["vocab_size"] != vocab or info["eos"][1] != eos:
                errors.append(f"{prefix} was tokenized with {info['name_or_path']} (vocab {info['vocab_size']}, "
                              f"eos {info['eos']}), but TOKENIZER={args.tokenizer} has vocab {vocab}, "
                              f"eos [{tok.eos_token!r}, {eos}]  ({path})")

    if errors:
        for e in errors:
            print(f"check_tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"check_tokenizer: ok ({args.tokenizer}: vocab {vocab}, eos {eos}; {len(args.prefixes)} prefixes)")


if __name__ == "__main__":
    main()
