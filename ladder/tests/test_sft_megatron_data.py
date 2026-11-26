"""Megatron's SFT data path on prepare_sft_data.py output, against the pipeline's encoding (CPU, real tokenizer).

    python -m pytest ladder/tests/test_sft_megatron_data.py -p no:cacheprovider -o addopts=""

Runs in the Megatron environment (transformers, datasets, pyarrow) with the ladder tokenizers; a tokenizer
directory that does not exist is skipped (SFT_TEST_TOKENIZER_LLAMA3 / SFT_TEST_TOKENIZER_QWEN3 override them).
For synthetic OpenMathInstruct-2 rows it runs prepare_sft_data.py end to end, then reads train.parquet with
Megatron's own VarlenLowLevelDataset + VarlenDataset and the SFTTokenizer (identity), exactly what
train_sft.sh trains on, and checks every item against the pipeline's example: prompt + response + EOS tokenized
the pipeline's way, inputs = all but the last token, loss only on the response and the EOS.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ladder" / "sft"))

import prepare_sft_data as prep  # noqa: E402

TOKENIZERS = {
    "llama3": os.environ.get("SFT_TEST_TOKENIZER_LLAMA3", "/fast/users/zhuangxinlin/tokenizers/llama-3"),
    "qwen3": os.environ.get("SFT_TEST_TOKENIZER_QWEN3", "/fast/users/zhuangxinlin/tokenizers/qwen3"),
}
SEQ_LEN = 512

ROWS = [
    ("What is $2 + 3$?", "We add: $2 + 3 = 5$. The answer is $\\boxed{5}$."),
    ("Solve for $x$: $3x - 7 = 11$.", "Add 7 to both sides: $3x = 18$, so $x = \\boxed{6}$."),
    ("Compute $\\frac{1}{2} + \\frac{1}{3}$.", "$\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}$, i.e. $\\boxed{\\frac{5}{6}}$."),
    ("Let $f(x) = x^2$. Find $f(f(2))$.", "First $f(2) = 4$.\nThen $f(4) = 16$.\n\nSo the answer is $\\boxed{16}$."),
    ("How many primes are below 20?", "2, 3, 5, 7, 11, 13, 17, 19: that is $\\boxed{8}$ primes."),
    ("Évaluez $\\sqrt{49}$ (unicode — test).", "√49 = 7 → $\\boxed{7}$"),
    ("A train goes 60 km/h for 2.5 h. Distance?", "  Distance = 60 × 2.5 = 150 km, so $\\boxed{150}$."),
    ("Find the remainder of $17^{2}$ mod 5.", "\nLeading newline: $289 = 57 \\cdot 5 + 4$, remainder $\\boxed{4}$."),
    ("Empty answer row.", ""),
    ("Overlong row.", "word " * 2000),
    ("What is $10 \\div 4$?", "$10 / 4 = 2.5$, so $\\boxed{\\tfrac{5}{2}}$."),
    ("Sum of the angles of a triangle?", "It is $180^\\circ$: $\\boxed{180}$."),
]


@pytest.fixture(params=sorted(TOKENIZERS))
def tokenizer_path(request):
    path = TOKENIZERS[request.param]
    if not os.path.isdir(path):
        pytest.skip(f"no tokenizer at {path}")
    return path


@pytest.fixture
def prepared(tmp_path, tokenizer_path):
    pytest.importorskip("transformers")
    pytest.importorskip("pyarrow")
    src = tmp_path / "subset" / "train"
    src.mkdir(parents=True)
    with open(src / "train.jsonl", "w") as f:
        for i, (problem, solution) in enumerate(ROWS):
            f.write(json.dumps({"problem": problem, "generated_solution": solution, "sample_id": 1000 + i}) + "\n")
    out = tmp_path / "out"
    prep.main(["--input", str(src), "--tokenizer", tokenizer_path, "--output-dir", str(out), "--epochs", "2",
               "--seq-len", str(SEQ_LEN), "--megatron-check-rows", "100", "--dp-sizes", "8"])
    from transformers import AutoTokenizer

    return out, json.loads((out / "stats.json").read_text()), AutoTokenizer.from_pretrained(tokenizer_path), tokenizer_path


def test_prepared_file(prepared):
    import pyarrow.parquet as pq

    out, stats, _, _ = prepared
    assert stats["rows_in"] == len(ROWS)
    assert stats["dropped_empty_response"] == 1 and stats["dropped_overlong"] == 1
    n = stats["examples"]
    assert n == len(ROWS) - 2 - stats["dropped_boundary"] and stats["dropped_boundary"] <= 2
    assert stats["rows_out"] == 2 * n
    table = pq.read_table(out / "train.parquet").to_pylist()
    assert [r["epoch"] for r in table] == [0] * n + [1] * n
    ids = [[r["sample_id"] for r in table if r["epoch"] == e] for e in (0, 1)]
    assert sorted(ids[0]) == sorted(ids[1]) and len(set(ids[0])) == n
    assert ids[0] != ids[1]                              # reshuffled every epoch
    assert 1008 not in ids[0] and 1009 not in ids[0]     # empty and overlong rows are gone
    assert set(table[0]) == {"prompt", "response", "sample_id", "epoch"}
    assert table[0]["prompt"].startswith("Solve the following math problem.")


def test_megatron_items_match_pipeline_encoding(prepared):
    import numpy as np
    import pyarrow.parquet as pq

    pytest.importorskip("datasets")
    from megatron.core.tokenizers import MegatronTokenizer
    from megatron.training.datasets.varlen_dataset import VarlenDataset, VarlenLowLevelDataset

    out, stats, hf, path = prepared
    low = VarlenLowLevelDataset(str(out / "train.parquet"))
    assert low.schema_name == "alpaca" and len(low) == stats["rows_out"]
    tok = MegatronTokenizer.from_pretrained(tokenizer_path=path, metadata_path={"library": "sft"},
                                            prompt_format="identity", use_gigatoken=False)
    assert tok.eod == hf.eos_token_id
    ds = VarlenDataset.__new__(VarlenDataset)
    ds.config = SimpleNamespace(tokenizer=tok, sequence_length=SEQ_LEN, reset_position_ids=False,
                                create_attention_mask=False, reset_attention_mask=False,
                                varlen_sbhd_validation=False, data_parallel_size=1, context_parallel_size=1,
                                dynamic_context_parallel=False, sequence_parallel_size=1)
    ds.dataset = low
    ds.indices = np.arange(len(low))

    rows = pq.read_table(out / "train.parquet").to_pylist()
    supervised = 0
    for i, row in enumerate(rows):
        p = hf(row["prompt"], add_special_tokens=False)["input_ids"]
        r = hf(row["response"], add_special_tokens=False)["input_ids"]
        ids = p + r + [hf.eos_token_id]                  # the pipeline's example
        item = ds[i]
        assert item["tokens"].tolist() == ids[:-1], f"row {i}: input tokens differ"
        expected_mask = [0.0] * (len(p) - 1) + [1.0] * (len(r) + 1)
        assert item["loss_mask"].tolist() == expected_mask, f"row {i}: loss mask differs"
        labels = item["labels"].tolist()
        assert [l for l, m in zip(labels, expected_mask) if m] == ids[len(p):], f"row {i}: targets differ"
        assert int(item["original_seq_len"]) == len(ids) - 1
        supervised += len(r) + 1
    assert supervised == 2 * stats["supervised_tokens_per_epoch"]
