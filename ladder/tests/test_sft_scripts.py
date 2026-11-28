"""CPU tests for the ladder SFT tooling: subset draw, data preparation, the launchers' plans, the vLLM eval's grading.

    python -m pytest ladder/tests/test_sft_scripts.py -p no:cacheprovider -o addopts=""

No GPU, Megatron, vLLM or tokenizer needed. The design follows pretrain_LLM_pipeline's SFT; where the two must agree
(subset draw, prompt template, epoch order, packing), the expected values below were computed once with that
repository and are frozen here, so these tests do not need it.
"""
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ladder" / "sft"))
sys.path.insert(0, str(REPO / "ladder" / "sft" / "eval"))

import eval_vllm  # noqa: E402
import make_sft_subset as subset  # noqa: E402
import prepare_sft_data as prep  # noqa: E402


def digest(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


# ---- make_sft_subset.py ----------------------------------------------------------------------------
def write_subset_source(root):
    """140 rows matched by 'train-*' in three JSONL files (one in a subdirectory, blank lines in between), plus one
    file the glob leaves out; 50 distinct problems, so validation problems recur among the candidates."""
    rows = [{"problem": f"p{(i * 7) % 50}", "generated_solution": f"s{i}", "expected_answer": str(i % 9),
             "problem_source": "x"} for i in range(150)]
    parts = {"train-00000.jsonl": rows[:50], "sub/train-00001.jsonl": rows[50:110],
             "train-00002.jsonl": rows[110:140], "test-00000.jsonl": rows[140:]}
    for name, part in parts.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" + ("\n" if j % 7 == 0 else "") for j, r in enumerate(part)))


# The pipeline's draw (src/cli/prepare_sft_subset.py) on write_subset_source with the arguments of the test below.
SUBSET_TRAIN = [0, 3, 5, 7, 8, 11, 12, 13, 19, 20, 23, 25, 27, 29, 33, 43, 44, 45, 48, 53, 54, 55, 58, 64, 69, 71,
                77, 83, 89, 91, 103, 104, 109, 111, 112, 118, 124, 134, 135, 139]
SUBSET_VALIDATION = [6, 22, 26, 28, 35, 57, 62, 70]


def test_subset_draw(tmp_path):
    write_subset_source(tmp_path / "src")
    out = tmp_path / "out"
    manifest = subset.main(["--input", str(tmp_path / "src"), "--file-glob", "train-*", "--output-dir", str(out),
                            "--num-train", "40", "--num-validation", "8", "--seed", "42", "--train-oversample", "0.5",
                            "--columns", "problem", "generated_solution", "--disjoint-column", "problem"])
    train = [json.loads(l) for l in (out / "train" / "train.jsonl").read_text().splitlines()]
    validation = [json.loads(l) for l in (out / "validation" / "validation.jsonl").read_text().splitlines()]
    assert [r["sample_id"] for r in train] == SUBSET_TRAIN
    assert [r["sample_id"] for r in validation] == SUBSET_VALIDATION
    assert train[0] == {"sample_id": 0, "problem": "p0", "generated_solution": "s0"}
    assert not {r["problem"] for r in train} & {r["problem"] for r in validation}
    # files in sorted order, the subdirectory last; the test file is not matched; blank lines are not rows
    assert [(Path(f["path"]).relative_to(tmp_path / "src").as_posix(), f["global_start"], f["rows"])
            for f in manifest["files"]] == [("train-00000.jsonl", 0, 50), ("train-00002.jsonl", 50, 30),
                                            ("sub/train-00001.jsonl", 80, 60)]
    assert manifest["train_candidates_skipped_for_overlap"] == 3 and manifest["validation_unique_values"] == 7
    assert manifest["counts"] == {"train": 40, "validation": 8}


def test_subset_split_rules():
    draws = subset.draw_indices(100, 10, 3, seed=1, extra=5)
    assert len(draws) == 18 and len(set(draws)) == 18 and draws == subset.draw_indices(100, 10, 3, 1, 5)
    train, validation, skipped = subset.split_draws(draws, 10, 3)
    assert validation == sorted(draws[:3]) and train == sorted(draws[3:13]) and skipped == 0
    key = lambda i: i % 4                                 # noqa: E731
    train, validation, skipped = subset.split_draws(draws, 3, 3, key)
    assert not {key(i) for i in train} & {key(i) for i in validation}
    with pytest.raises(ValueError, match="raise --train-oversample"):
        subset.split_draws(draws, 10, 3, lambda i: 0)
    with pytest.raises(ValueError, match="requested 101"):
        subset.draw_indices(100, 100, 1, 1, 0)


# ---- prepare_sft_data.py ---------------------------------------------------------------------------
def test_matches_pipeline_functions():
    row = {"problem": "What is {x}? Use \\boxed{} and {{braces}}.", "generated_solution": "It is 3."}
    assert prep.render_template(prep.PROMPT_TEMPLATE, row) == (
        "Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"
        "Problem:\nWhat is {x}? Use \\boxed{} and {{braces}}.\n\nSolution:\n")
    assert prep.render_template("{problem}|{{lit}}|{missing_brace|{generated_solution}", row) == (
        "What is {x}? Use \\boxed{} and {{braces}}.|{lit}|{missing_brace|It is 3.")
    for epoch, head, h in [(0, [970, 272, 297, 321, 497, 66], "3e651b2021aaffea"),
                           (1, [245, 373, 10, 415, 358, 808], "3797e4f0c0405925"),
                           (2, [249, 544, 178, 209, 280, 83], "d498bffd28b9dd98")]:
        order = prep.epoch_order(1000, 42, epoch)
        assert order[:6] == head and digest(order) == h
    lengths = [((i * 37) % 900) + 1 for i in range(500)]
    packs = prep.pack_next_fit(lengths, prep.epoch_order(500, 42, 0), 4096)
    assert len(packs) == 59 and list(packs[0]) == [277, 189, 371, 239, 354, 480, 451, 197]
    assert digest([list(p) for p in packs]) == "45000e04ca233ed3"


def test_prompt_template_frozen():
    assert prep.PROMPT_TEMPLATE == ("Solve the following math problem. Make sure to put the answer (and only answer) "
                                    "inside \\boxed{}.\n\nProblem:\n{problem}\n\nSolution:\n")
    assert prep.RESPONSE_TEMPLATE == "{generated_solution}"


def char_tokenize(texts):
    return [[ord(c) for c in t] for t in texts]


def merging_tokenize(texts):
    """A tokenizer whose "\\n" + "!" merges into one token, so separate and joint tokenization differ."""
    out = []
    for t in texts:
        ids, i = [], 0
        while i < len(t):
            if t[i:i + 2] == "\n!":
                ids.append(-1); i += 2
            else:
                ids.append(ord(t[i])); i += 1
        out.append(ids)
    return out


def test_encode_drops_and_lengths():
    prompts = ["ab\n", "ab\n", "ab\n", "ab\n"]
    responses = ["xyz", "", "y" * 20, "!z"]
    lengths, reasons, plens = prep.encode(prompts, responses, merging_tokenize, seq_len=10)
    assert lengths == [7, None, None, 6]          # 3 + 3 + EOS; empty; overlong (3 + 20 + 1 > 10); kept length
    assert reasons == ["", "empty_response", "overlong", "boundary"]
    assert plens == [3, 3, 3, 3]


def test_plan_keeps_pipeline_order_and_counts():
    lengths = [5, None, 7, 6, 4, 8]
    reasons = ["", "overlong", "", "boundary", "", ""]
    plens = [2, 2, 3, 2, 1, 4]
    orders, stats = prep.plan(lengths, reasons, plens, epochs=2, seed=42, seq_len=16, dp_sizes=[2])
    pipe = [0, 2, 3, 4, 5]                          # the pipeline's examples (boundary row included)
    for epoch, order in enumerate(orders):
        expected = [pipe[k] for k in prep.epoch_order(len(pipe), 42, epoch) if pipe[k] != 3]
        assert order == expected
    assert stats["examples"] == 4 and stats["rows_out"] == 8
    assert stats["dropped_overlong"] == 1 and stats["dropped_boundary"] == 1
    assert stats["tokens_per_epoch"] == 5 + 7 + 4 + 8
    assert stats["supervised_tokens_per_epoch"] == (5 - 2) + (7 - 3) + (4 - 1) + (8 - 4)
    assert stats["pipeline"]["examples"] == 5
    gbs = stats["megatron_by_dp_size"]["2"]["global_batch_size"]
    assert gbs % 2 == 0 and gbs >= 2


@pytest.mark.parametrize("eps,dp,gbs", [(315.43, 8, 312), (315.43, 16, 320), (3.0, 8, 8), (319.9, 8, 320)])
def test_global_batch_for(eps, dp, gbs):
    assert prep.global_batch_for(eps, dp) == gbs


# ---- eval_vllm.py ----------------------------------------------------------------------------------
def test_problems_and_summary():
    task = eval_vllm.EvalTask(name="gsm8k", path="x", grader="gsm8k", problem_field="question",
                              gold_field="answer", expected_size=2, max_new_tokens=8)
    docs = [{"question": "1+1?", "answer": "so #### 2"}, {"question": "2+2?", "answer": "#### 4"}]
    problems = eval_vllm.problems_from_docs(task, docs, prep.PROMPT_TEMPLATE, 0)
    assert problems[0].prompt == prep.render_template(prep.PROMPT_TEMPLATE, {"problem": "1+1?"})
    assert problems[0].gold == "2"
    results = []
    for p, texts in zip(problems, [["\\boxed{2}", "3"], ["\\boxed{4}", "\\boxed{4}"]]):
        results.append(eval_vllm.grade_completion(task, p, -1, texts[0], 5, "stop"))
        for j, t in enumerate(texts):
            results.append(eval_vllm.grade_completion(task, p, j, t, 3, "length"))
    m = eval_vllm.summarize(results, {"gsm8k": problems}, num_samples=2, greedy=True)
    assert m["gsm8k/pass@1"] == pytest.approx(0.75)
    assert m["gsm8k/pass@2"] == pytest.approx(1.0)
    assert m["gsm8k/greedy_acc"] == pytest.approx(1.0)
    assert m["gsm8k/finish_rate"] == pytest.approx(2 / 6)


def test_countdown_problem_uses_task_prompt():
    task = eval_vllm.EvalTask(name="countdown", path="x", grader="countdown", problem_field="problem",
                              gold_field="answer", expected_size=0, max_new_tokens=8,
                              problem_template="Using the numbers {nums}, make {target}.",
                              gold_fields=("nums", "target"), prompt_template="Q:\n{problem}\nA \\boxed{1 + 2}:\n",
                              stop=("\nProblem:",))
    (p,) = eval_vllm.problems_from_docs(task, [{"nums": [1, 2], "target": 3}], prep.PROMPT_TEMPLATE, 0)
    assert p.prompt == "Q:\nUsing the numbers [1, 2], make 3.\nA \\boxed{1 + 2}:\n"
    assert eval_vllm.grade_completion(task, p, 0, "\\boxed{2 + 1}", 4, "stop")["correct"]


def test_eval_config_loads():
    pytest.importorskip("yaml")
    cfg = eval_vllm.load_config(str(REPO / "ladder/sft/eval/math_passk.yaml"), "/data")
    names = [t.name for t in cfg["tasks"]]
    assert names == ["gsm8k", "math500", "amc23", "countdown"]
    assert cfg["tasks"][0].path == "/data/gsm8k/main/test-00000-of-00001.parquet"
    assert cfg["prompt_template"] == prep.PROMPT_TEMPLATE
    assert cfg["num_samples"] == 64 and cfg["greedy"] and cfg["max_problems"] == 0
    assert cfg["tasks"][3].stop == ("\nProblem:",)


def test_compare_hf_versions(tmp_path):
    here = {"transformers": "5.9.0", "tokenizers": "0.22.2"}
    assert eval_vllm.compare_hf_versions(str(tmp_path), here) == (None, [])           # reference model: no meta
    meta = tmp_path / "export_meta.json"
    meta.write_text(json.dumps({"iteration": 3}))
    assert eval_vllm.compare_hf_versions(str(tmp_path), here) == (None, [])           # export predating the record
    meta.write_text(json.dumps({"hf_versions": here}))
    assert eval_vllm.compare_hf_versions(str(tmp_path), here) == (here, [])
    exported, diff = eval_vllm.compare_hf_versions(str(tmp_path), {"transformers": "4.57.1", "tokenizers": None})
    assert exported == here
    assert diff == ["transformers 5.9.0 (export) vs 4.57.1 (here)", "tokenizers 0.22.2 (export) vs None (here)"]
    assert set(eval_vllm.local_hf_versions()) == {"transformers", "tokenizers"}


# ---- launchers (DRY_RUN) ---------------------------------------------------------------------------
HEAD_VARS = ("HEAD", "MARGIN_D", "MARGIN_REVERSE", "HEAD_SCALE", "SFN_WEIGHT")


def run_script(script, env):
    full = {k: v for k, v in os.environ.items()
            if k not in HEAD_VARS + ("GBS", "SFT_GBS", "SFT_LR", "PRETRAIN_TAG", "PRETRAIN_RUN", "ITERS",
                                     "TARGET", "TASKS", "NUM_SAMPLES", "MAX_PROBLEMS", "NNODES", "TOKENIZER",
                                     "MODEL", "EVAL_DIR", "FORCE", "OUT_ROOT", "OPT", "SFT_OPT")}
    full.update({"DRY_RUN": "1", **{k: str(v) for k, v in env.items()}})
    return subprocess.run(["bash", script], cwd=REPO, env=full, capture_output=True, text=True)


@pytest.fixture
def sft_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "stats.json").write_text(json.dumps({"examples": 500000, "epochs": 10, "rows_out": 5000000,
                                              "seq_len": 4096, "pipeline": {"examples_per_step": 315.43}}))
    return d


def flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


@pytest.mark.parametrize("env,head,run", [
    ({}, [], "llama3_r1_muon_lr2.8e-3"),
    ({"HEAD": "hfn", "MARGIN_D": "0.02"}, ["--lm-head-feature-norm", "hfn", "--token-margin", "0",
                                          "--lm-head-feature-norm-scale", "60"], "llama3_r1_muon_lr2.8e-3_hfn_d0.02"),
    ({"HEAD": "sfn", "MARGIN_D": "0.02", "MARGIN_REVERSE": "1"}, ["--lm-head-feature-norm", "nfn", "--token-margin",
                                                                 "0"], "llama3_r1_muon_lr2.8e-3_sfn_rev_d0.02"),
])
def test_train_sft_plan(tmp_path, sft_data, env, head, run):
    res = run_script("ladder/sft/train_sft.sh", {"CONFIG": "llama3_r1", "OPT": "muon", "LR": "2.8e-3",
                                                 "OUT_ROOT": tmp_path, "SFT_DATA_DIR": sft_data,
                                                 "PRETRAIN_TAG": "tpp100", **env})
    assert res.returncode == 0, res.stderr
    cmd = shlex.split(res.stdout.strip().splitlines()[-1])
    got_head = [t for i, t in enumerate(cmd) if t.startswith(("--lm-head", "--token-margin", "--soft-feature"))
                or (i > 0 and cmd[i - 1].startswith(("--lm-head", "--token-margin", "--soft-feature")))]
    assert got_head == head
    assert flag(cmd, "--pretrained-checkpoint") == f"{tmp_path}/runs/{run}/ckpt/tpp100"
    assert flag(cmd, "--load") == flag(cmd, "--save") == f"{tmp_path}/sft/{run}_tpp100_sft_omi2_500k_ep10/ckpt"
    assert flag(cmd, "--global-batch-size") == "312"            # 8 GPUs: 315.43 -> 39 x 8
    assert flag(cmd, "--train-iters") == str(5000000 // 312)
    assert flag(cmd, "--save-interval") == str(500000 // 312)
    assert float(flag(cmd, "--lr")) == pytest.approx(2.8e-3 / 3, rel=1e-5)
    for f in ("--use-varlen-dataset", "--calculate-per-token-loss", "--no-create-attention-mask-in-dataloader"):
        assert f in cmd
    assert flag(cmd, "--sft-tokenizer-prompt-format") == "identity"
    assert flag(cmd, "--lr-decay-style") == "cosine" and flag(cmd, "--min-lr") == "0"
    assert flag(cmd, "--optimizer") == "muon"


def test_train_sft_requires_finished_pretraining(tmp_path, sft_data):
    run = tmp_path / "runs" / "llama3_r2_muon_lr2e-3_hfn_d0.02"
    ckpt = run / "ckpt" / "tpp50"
    ckpt.mkdir(parents=True)
    (run / "cmd_tpp50_20261126-010000.sh").write_text("pretrain_gpt.py --train-iters 55000 --exit-interval 27500\n")
    (run / "cmd_main_20261126-000000.sh").write_text("pretrain_gpt.py --train-iters 55000\n")
    env = {"CONFIG": "llama3_r2", "OPT": "muon", "LR": "2e-3", "HEAD": "hfn", "MARGIN_D": "0.02",
           "OUT_ROOT": tmp_path, "SFT_DATA_DIR": sft_data, "PRETRAIN_TAG": "tpp50"}
    (ckpt / "latest_checkpointed_iteration.txt").write_text("23375")       # a save on the decay
    res = run_script("ladder/sft/train_sft.sh", env)
    assert res.returncode != 0 and "trains it to 27500" in res.stderr
    assert run_script("ladder/sft/train_sft.sh", {**env, "ALLOW_UNFINISHED_PRETRAIN": "1"}).returncode == 0
    res = run_script("ladder/sft/export.sh", {**env, "TARGET": "pretrained"})
    assert res.returncode != 0 and "trains it to 27500" in res.stderr
    (ckpt / "latest_checkpointed_iteration.txt").write_text("27500")
    res = run_script("ladder/sft/train_sft.sh", env)
    assert res.returncode == 0, res.stderr
    res = run_script("ladder/sft/export.sh", {**env, "TARGET": "pretrained"})
    assert res.returncode == 0 and f"OUT={run}/hf/tpp50_iter27500 " in res.stdout, res.stderr
    main = run / "ckpt" / "main"                                            # no --exit-interval: --train-iters
    main.mkdir()
    (main / "latest_checkpointed_iteration.txt").write_text("55000")
    assert run_script("ladder/sft/train_sft.sh", {**env, "PRETRAIN_TAG": "main"}).returncode == 0


@pytest.mark.parametrize("env,opt,pre_run,suffix", [
    ({}, "muon", "llama3_r1_muon_lr2e-3", ""),                                  # defaults: Muon run, Muon SFT
    ({"OPT": "adamw"}, "muon", "llama3_r1_adamw_lr2e-3", ""),                   # AdamW-pretrained, still Muon SFT
    ({"SFT_OPT": "adamw"}, "adam", "llama3_r1_muon_lr2e-3", "_adamw"),          # explicit switch, own run name
])
def test_train_sft_optimizer(tmp_path, sft_data, env, opt, pre_run, suffix):
    res = run_script("ladder/sft/train_sft.sh", {"CONFIG": "llama3_r1", "LR": "2e-3", "OUT_ROOT": tmp_path,
                                                 "SFT_DATA_DIR": sft_data, **env})
    assert res.returncode == 0, res.stderr
    cmd = shlex.split(res.stdout.strip().splitlines()[-1])
    assert flag(cmd, "--optimizer") == opt
    assert flag(cmd, "--pretrained-checkpoint") == f"{tmp_path}/runs/{pre_run}/ckpt/main"
    assert flag(cmd, "--save") == f"{tmp_path}/sft/{pre_run}_main_sft_omi2_500k_ep10{suffix}/ckpt"


def test_train_sft_rejects_bad_batch_and_seq(tmp_path, sft_data):
    base = {"CONFIG": "llama3_r1", "LR": "2.8e-3", "OUT_ROOT": tmp_path, "SFT_DATA_DIR": sft_data}
    assert run_script("ladder/sft/train_sft.sh", {**base, "SFT_GBS": "300"}).returncode != 0
    (sft_data / "stats.json").write_text(json.dumps({"examples": 10, "epochs": 1, "rows_out": 10, "seq_len": 2048,
                                                     "pipeline": {"examples_per_step": 8}}))
    assert run_script("ladder/sft/train_sft.sh", base).returncode != 0


def make_sft_run(tmp_path):
    run = "llama3_r1_muon_lr2.8e-3_hfn_d0.02"
    sft = tmp_path / "sft" / f"{run}_main_sft_omi2_500k_ep10"
    (sft / "ckpt" / "iter_0001602").mkdir(parents=True)
    (sft / "ckpt" / "iter_0016025").mkdir()
    (sft / "ckpt" / "latest_checkpointed_iteration.txt").write_text("16025")
    pre = tmp_path / "runs" / run / "ckpt" / "main"
    pre.mkdir(parents=True)
    (pre / "latest_checkpointed_iteration.txt").write_text("35000")
    return run, sft


def test_export_plan(tmp_path):
    run, sft = make_sft_run(tmp_path)
    base = {"CONFIG": "llama3_r1", "OPT": "muon", "LR": "2.8e-3", "HEAD": "hfn", "MARGIN_D": "0.02",
            "OUT_ROOT": tmp_path}
    out = run_script("ladder/sft/export.sh", base)
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert len(lines) == 1                                    # latest only
    assert "HEAD=hfn MARGIN_D=0 MARGIN_REVERSE=0" in lines[0] and "ITER=16025" in lines[0]
    assert f"OUT={sft}/hf/iter_16025 " in lines[0]
    (sft / "hf" / "iter_1602").mkdir(parents=True)
    (sft / "hf" / "iter_1602" / "export_meta.json").write_text("{}")
    out = run_script("ladder/sft/export.sh", {**base, "ITERS": "all"})
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert len(lines) == 1 and "ITER=16025" in lines[0]        # iteration 1602 is already exported
    assert out.stdout.rstrip().endswith(f"{sft}/hf/iter_16025")
    out = run_script("ladder/sft/export.sh", {**base, "TARGET": "pretrained"})
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert "HEAD=hfn MARGIN_D=0.02" in lines[0] and f"OUT={tmp_path}/runs/{run}/hf/main_iter35000" in lines[0]


def test_eval_models(tmp_path):
    run, sft = make_sft_run(tmp_path)
    for it in (16025, 1602, 3204):
        (sft / "hf" / f"iter_{it}").mkdir(parents=True)
    other = tmp_path / "models" / "Qwen3-0.6B-Base"
    other.mkdir(parents=True)
    out = run_script("ladder/sft/eval.sh", {"MODEL": f"{sft}/hf/iter_*"})
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert [l.split("--output-dir ")[1] for l in lines] == [f"{sft}/eval/iter_{it}" for it in (1602, 3204, 16025)]
    out = run_script("ladder/sft/eval.sh", {"MODEL": f"{other} {sft}/hf/iter_1602/", "TASKS": "gsm8k math500",
                                            "NUM_SAMPLES": "4", "MAX_PROBLEMS": "50"})
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert lines[0].endswith(f"--output-dir {tmp_path}/models/eval/Qwen3-0.6B-Base_gsm8k_math500_n4_p50 "
                             "--tasks gsm8k math500 --num-samples 4 --max-problems 50")
    assert f"--model {sft}/hf/iter_1602 " in lines[1]
    out = run_script("ladder/sft/eval.sh", {"MODEL": f"{other}", "EVAL_DIR": tmp_path / "x"})
    assert out.stdout.splitlines()[-1].endswith(f"--output-dir {tmp_path}/x")
    out = run_script("ladder/sft/eval.sh", {"MODEL": f"{other}", "ALLOW_HF_MISMATCH": "1"})
    assert out.stdout.splitlines()[-1].endswith(" --allow-hf-version-mismatch")
    assert run_script("ladder/sft/eval.sh", {"MODEL": f"{sft}/hf/iter_*", "EVAL_DIR": "/x"}).returncode != 0
    full = {k: v for k, v in os.environ.items() if k not in ("MODEL", "DRY_RUN")}
    res = subprocess.run(["bash", "ladder/sft/eval.sh"], cwd=REPO, env={**full, "MODEL": "/nope"},
                         capture_output=True, text=True)
    assert res.returncode != 0 and "export.sh" in res.stderr
