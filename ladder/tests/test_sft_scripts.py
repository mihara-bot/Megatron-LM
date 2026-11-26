"""CPU tests for the ladder SFT tooling: data preparation, the launchers' plans, the vLLM eval's grading path.

    python -m pytest ladder/tests/test_sft_scripts.py -p no:cacheprovider -o addopts=""

No GPU, Megatron, vLLM or tokenizer needed. Checks against pretrain_LLM_pipeline run when that repository sits
next to this one (../pretrain_LLM_pipeline) and is importable; they are skipped otherwise.
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PIPELINE = REPO.parent / "pretrain_LLM_pipeline"
sys.path.insert(0, str(REPO / "ladder" / "sft"))
sys.path.insert(0, str(REPO / "ladder" / "sft" / "eval"))

import eval_vllm  # noqa: E402
import prepare_sft_data as prep  # noqa: E402


def pipeline_sft():
    if not (PIPELINE / "src" / "data" / "sft.py").exists():
        pytest.skip("pretrain_LLM_pipeline is not next to this repository")
    sys.path.insert(0, str(PIPELINE))
    from src.data import sft
    return sft


# ---- prepare_sft_data.py ---------------------------------------------------------------------------
def test_matches_pipeline_functions():
    sft = pipeline_sft()
    row = {"problem": "What is {x}? Use \\boxed{} and {{braces}}.", "generated_solution": "It is 3."}
    assert prep.render_template(prep.PROMPT_TEMPLATE, row) == sft.render_template(prep.PROMPT_TEMPLATE, row)
    for epoch in range(3):
        assert prep.epoch_order(1000, 42, epoch) == sft.epoch_order(1000, 42, epoch)
    lengths = [((i * 37) % 900) + 1 for i in range(500)]
    order = prep.epoch_order(500, 42, 0)
    assert prep.pack_next_fit(lengths, order, 4096) == sft.pack_next_fit(lengths, order, 4096)


def test_prompt_template_matches_pipeline_config():
    yaml = pytest.importorskip("yaml")
    cfg = PIPELINE / "configs" / "data" / "sft" / "openmathinstruct2_500k.yaml"
    if not cfg.exists():
        pytest.skip("pretrain_LLM_pipeline is not next to this repository")
    sft_cfg = yaml.safe_load(cfg.read_text())["sft"]
    assert sft_cfg["prompt_template"] == prep.PROMPT_TEMPLATE
    assert sft_cfg["response_template"] == prep.RESPONSE_TEMPLATE


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


# ---- launchers (DRY_RUN) ---------------------------------------------------------------------------
HEAD_VARS = ("HEAD", "MARGIN_D", "MARGIN_REVERSE", "HEAD_SCALE", "SFN_WEIGHT")


def run_script(script, env):
    full = {k: v for k, v in os.environ.items()
            if k not in HEAD_VARS + ("GBS", "SFT_GBS", "SFT_LR", "PRETRAIN_TAG", "PRETRAIN_RUN", "ITERS",
                                     "TARGET", "TASKS", "NUM_SAMPLES", "MAX_PROBLEMS", "NNODES", "TOKENIZER")}
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


def test_train_sft_rejects_bad_batch_and_seq(tmp_path, sft_data):
    base = {"CONFIG": "llama3_r1", "LR": "2.8e-3", "OUT_ROOT": tmp_path, "SFT_DATA_DIR": sft_data}
    assert run_script("ladder/sft/train_sft.sh", {**base, "SFT_GBS": "300"}).returncode != 0
    (sft_data / "stats.json").write_text(json.dumps({"examples": 10, "epochs": 1, "rows_out": 10, "seq_len": 2048,
                                                     "pipeline": {"examples_per_step": 8}}))
    assert run_script("ladder/sft/train_sft.sh", base).returncode != 0


def test_eval_plan(tmp_path):
    run = "llama3_r1_muon_lr2.8e-3_hfn_d0.02"
    sft = tmp_path / "sft" / f"{run}_main_sft_omi2_500k_ep10" / "ckpt"
    (sft / "iter_0001602").mkdir(parents=True)
    (sft / "iter_0016025").mkdir()
    (sft / "latest_checkpointed_iteration.txt").write_text("16025")
    pre = tmp_path / "runs" / run / "ckpt" / "main"
    pre.mkdir(parents=True)
    (pre / "latest_checkpointed_iteration.txt").write_text("35000")
    base = {"CONFIG": "llama3_r1", "OPT": "muon", "LR": "2.8e-3", "HEAD": "hfn", "MARGIN_D": "0.02",
            "OUT_ROOT": tmp_path}
    out = run_script("ladder/sft/eval.sh", base)
    assert out.returncode == 0, out.stderr
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert len(lines) == 2
    assert "HEAD=hfn MARGIN_D=0 MARGIN_REVERSE=0" in lines[0] and "ITER=16025" in lines[0]
    assert lines[1].endswith(f"--output-dir {tmp_path}/sft/{run}_main_sft_omi2_500k_ep10/eval/iter_16025")
    out = run_script("ladder/sft/eval.sh", {**base, "ITERS": "all", "TASKS": "gsm8k", "NUM_SAMPLES": "16"})
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert len(lines) == 4 and "eval/iter_1602_gsm8k_n16 --tasks gsm8k --num-samples 16" in lines[1]
    out = run_script("ladder/sft/eval.sh", {**base, "TARGET": "pretrained"})
    lines = [l for l in out.stdout.splitlines() if l.startswith("+ ")]
    assert "HEAD=hfn MARGIN_D=0.02" in lines[0] and f"OUT={tmp_path}/runs/{run}/hf/main_iter35000" in lines[0]
    assert lines[1].endswith("eval_pretrained/iter_35000")
