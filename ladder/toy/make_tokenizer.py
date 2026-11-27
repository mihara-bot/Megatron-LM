"""Write the margin toy's 100-token Hugging Face tokenizer (ladder/toy/TOY.md).

    python ladder/toy/make_tokenizer.py --out $TOY_ROOT/tokenizer

A WordLevel model over ladder/toy/graph.py's vocabulary, split on whitespace, no BOS and no post-processor (the
ladder data is EOS-only: tokenize_dataset.py appends <eos> to every document). <unk> doubles as the pad of the SFT
identity format (Megatron's SFTTokenizer looks it up by name). With a real HF tokenizer directory, every ladder
script that takes TOKENIZER (train.sh + check_tokenizer.py, train_sft.sh, export_hf.sh, the vLLM eval) works
unchanged. Needs `tokenizers` and `transformers`.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph as G  # noqa: E402


def build(out: str) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    vocab = {tok: i for i, tok in enumerate(G.vocab())}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>",
                                   model_max_length=4096, clean_up_tokenization_spaces=False)
    os.makedirs(out, exist_ok=True)
    fast.save_pretrained(out)

    # Reload the way the ladder does and check the layout.
    tok = AutoTokenizer.from_pretrained(out)
    problems = []
    if len(tok) != G.VOCAB_SIZE:
        problems.append(f"len(tokenizer) = {len(tok)}, expected {G.VOCAB_SIZE}")
    if tok.eos_token_id != G.EOS or tok.bos_token_id is not None:
        problems.append(f"eos {tok.eos_token_id} (expected {G.EOS}), bos {tok.bos_token_id} (expected None)")
    if tok.convert_tokens_to_ids("<unk>") != G.UNK:
        problems.append("<unk> is not id 0")
    walk = [3, 17, 40]
    if tok(G.walk_text(walk), add_special_tokens=False).input_ids != G.walk_ids(walk)[:-1]:
        problems.append(f"walk text does not tokenize to {G.walk_ids(walk)[:-1]}")
    if tok(G.walk_text(walk)).input_ids != G.walk_ids(walk)[:-1]:
        problems.append("tokenizer adds special tokens by default")
    prompt, resp = G.path_prompt_text(3, 40), G.path_response_text([3, 17, 40])
    joined = tok(prompt + resp, add_special_tokens=False).input_ids
    if joined != G.path_prompt_ids(3, 40) + [G.node_token(v) for v in (3, 17, 40)]:
        problems.append(f"prompt + response tokenizes to {joined}")
    if problems:
        raise SystemExit("tokenizer check failed:\n  " + "\n  ".join(problems))
    print(f"wrote {out}: {len(tok)} tokens, eos {tok.eos_token_id}, no bos; "
          f"'{G.walk_text(walk)}' -> {tok(G.walk_text(walk)).input_ids}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, help="tokenizer directory to write")
    build(p.parse_args().out)


if __name__ == "__main__":
    main()
