"""CPU-only tests for the pretraining eval scripts (ladder/EVAL.md): export.sh, lm_eval.sh, run_eval.sh (DRY_RUN plans)
and lm_eval_util.py (token convention, result collection).

    python -m pytest ladder/tests/test_eval_scripts.py -p no:cacheprovider -o addopts=""

No GPU, SLURM, Megatron, vLLM or lm-eval needed: the scripts run with DRY_RUN=1 and lm_eval_util.py is stdlib only.
"""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ladder" / "eval"))

import lm_eval_util as util  # noqa: E402

SCRUB = ("HEAD", "MARGIN_D", "MARGIN_REVERSE", "HEAD_SCALE", "SFN_WEIGHT", "CONFIG", "OPT", "LR", "RUN", "TAGS", "ITERS",
         "TOKENIZER", "OUT_ROOT", "FORCE", "ALLOW_UNFINISHED", "MODEL", "TASK_SET", "TASKS", "NUM_FEWSHOT", "LIMIT",
         "TOKENS", "EVAL_DIR", "EVAL_DATA_DIR", "VLLM_PYTHON", "LOG_SAMPLES", "MODEL_ARGS", "ALLOW_HF_MISMATCH",
         "CHAINS", "RUNGS", "IDLE", "STEPS", "MODE", "TPP", "STOP_TPP", "DRY_RUN")
SMALL = ("hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,sciq,boolq,lambada_openai,copa,social_iqa,"
         "commonsense_qa")
RUN = "llama3_r1_muon_lr2.8e-3_hfn_d0.02"


def run_script(script, env, dry=True):
    full = {k: v for k, v in os.environ.items() if k not in SCRUB}
    if dry:
        full["DRY_RUN"] = "1"
    full.update({k: str(v) for k, v in env.items()})
    return subprocess.run(["bash", script], cwd=REPO, env=full, capture_output=True, text=True)


def plus_lines(text):
    return [line[2:] for line in text.splitlines() if line.startswith("+ ")]


def make_tag(run_dir, tag, latest, cmd=None, iters=()):
    ckpt = run_dir / "ckpt" / tag
    ckpt.mkdir(parents=True)
    (ckpt / "latest_checkpointed_iteration.txt").write_text(str(latest))
    for it in iters:
        (ckpt / f"iter_{it:07d}").mkdir()
    if cmd is not None:
        (run_dir / f"cmd_{tag}_20261126-000000.sh").write_text(
            f"pretrain_gpt.py --tokenizer-model /tok/llama-3 {cmd}\n")
    return ckpt


def make_run(tmp_path):
    """A run with a stable run stopped at STOP_TPP=50, a finished TPP50 branch and a TPP20 branch still running."""
    run_dir = tmp_path / "runs" / RUN
    make_tag(run_dir, "main", 14000, "--train-iters 35000 --exit-interval 14000", iters=(2800, 5600, 11200, 14000))
    make_tag(run_dir, "tpp50", 17500, "--train-iters 35000 --exit-interval 17500", iters=(17500,))
    make_tag(run_dir, "tpp20", 6300, "--train-iters 35000 --exit-interval 7000", iters=(6300,))
    return run_dir


EXPORT_ENV = {"CONFIG": "llama3_r1", "LR": "2.8e-3", "HEAD": "hfn", "MARGIN_D": "0.02"}


# ---- export.sh -------------------------------------------------------------------------------------
def test_export_final_checkpoints(tmp_path):
    run_dir = make_run(tmp_path)
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path})
    assert res.returncode == 0, res.stderr
    lines = plus_lines(res.stdout)
    assert len(lines) == 1                                  # main is mid-schedule, tpp20 still running
    line = lines[0]
    assert "HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=0" in line and f"CKPT={run_dir}/ckpt/tpp50 " in line
    assert "ITER=17500 " in line and f"OUT={run_dir}/hf/tpp50_iter17500 " in line
    assert "TOKENIZER=/tok/llama-3 " in line                 # the training run's tokenizer
    assert line.endswith("bash ladder/export/export_hf.sh")
    assert "main: latest checkpoint 14000, but the LR decays to 0 at 35000" in res.stdout
    assert "tpp20: latest checkpoint 6300, but the LR decays to 0 at 7000" in res.stdout
    assert res.stdout.rstrip().endswith(f"{run_dir}/hf/tpp50_iter17500")

    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path, "ALLOW_UNFINISHED": "1"})
    assert [l.split("OUT=")[1].split()[0] for l in plus_lines(res.stdout)] == [
        f"{run_dir}/hf/{t}" for t in ("main_iter14000", "tpp20_iter6300", "tpp50_iter17500")]

    (run_dir / "ckpt" / "main" / "latest_checkpointed_iteration.txt").write_text("35000")   # stable run finished
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path, "TAGS": "main"})
    lines = plus_lines(res.stdout)
    assert len(lines) == 1 and f"OUT={run_dir}/hf/main_iter35000 " in lines[0]


def test_export_iters_and_skips(tmp_path):
    run_dir = make_run(tmp_path)
    ok = run_dir / "hf" / "main_iter5600"
    ok.mkdir(parents=True)
    (ok / "export_meta.json").write_text(json.dumps({"validation": {"passed": True}}))
    bad = run_dir / "hf" / "main_iter11200"
    bad.mkdir()
    (bad / "export_meta.json").write_text(json.dumps({"validation": {"passed": False}}))
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path, "TAGS": "main", "ITERS": "all"})
    assert res.returncode == 0, res.stderr
    assert [l.split("ITER=")[1].split()[0] for l in plus_lines(res.stdout)] == ["2800", "11200", "14000"]
    assert "iteration 5600: " in res.stdout and "already exported" in res.stdout
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path, "TAGS": "main", "ITERS": "all",
                                               "FORCE": "1"})
    assert len(plus_lines(res.stdout)) == 4
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path, "TAGS": "tpp20", "ITERS": "6300",
                                               "TOKENIZER": "/other"})
    lines = plus_lines(res.stdout)
    assert len(lines) == 1 and "ITER=6300 " in lines[0] and "TOKENIZER=/other " in lines[0]


def test_export_no_cmd_file_and_no_run(tmp_path):
    run_dir = tmp_path / "runs" / RUN
    make_tag(run_dir, "tpp50", 17500)                        # copied run: no cmd_tpp50_*.sh
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path})
    assert res.returncode == 0 and not plus_lines(res.stdout)
    assert "no cmd_tpp50_*.sh" in res.stdout
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path, "ALLOW_UNFINISHED": "1"})
    line = plus_lines(res.stdout)[0]
    assert "TOKENIZER=/fast/users/zhuangxinlin/tokenizers/llama-3 " in line   # family default
    res = run_script("ladder/eval/export.sh", {**EXPORT_ENV, "OUT_ROOT": tmp_path, "MARGIN_D": "0.05"}, dry=False)
    assert res.returncode == 1 and "no checkpoint directories" in res.stderr


def train_run_dir(out_root, env):
    full = {"TRAIN_DATA": "x", "VALID_DATA": "x", "TOKENIZER": "x", "OUT_ROOT": out_root, "CONFIG": "llama3_r1",
            "LR": "2.8e-3", "OPT": "muon", **env}
    res = run_script("ladder/train.sh", full)
    assert res.returncode == 0, res.stderr
    line = next(l for l in res.stdout.splitlines() if l.startswith("== run "))
    return line.split()[2].rstrip(",")


def test_export_names_runs_as_train(tmp_path):
    for env in ({}, {"HEAD": "hfn", "MARGIN_D": "0.02"}, {"HEAD": "sfn", "MARGIN_D": "0.02", "MARGIN_REVERSE": "1"},
                {"HEAD": "hfn", "MARGIN_D": "0", "HEAD_SCALE": "30"}, {"HEAD": "nfn", "MARGIN_D": "0.05"}):
        run_dir = Path(train_run_dir(tmp_path, env))
        make_tag(run_dir, "tpp50", 17500, "--exit-interval 17500")
        res = run_script("ladder/eval/export.sh", {"CONFIG": "llama3_r1", "LR": "2.8e-3", "OUT_ROOT": tmp_path, **env})
        assert res.returncode == 0, res.stderr
        assert f"== export {run_dir.name}:" in res.stdout
        assert f"OUT={run_dir}/hf/tpp50_iter17500 " in plus_lines(res.stdout)[0]


# ---- lm_eval.sh ------------------------------------------------------------------------------------
def make_export(path, eos=128001, passed=True, iteration=17500):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"bos_token_id": 128000, "eos_token_id": eos}))
    (path / "export_meta.json").write_text(json.dumps({
        "iteration": iteration, "source_checkpoint": "/ckpt", "validation": {"passed": passed},
        "hf_versions": {"transformers": "5.9.0", "tokenizers": "0.22.2"},
        "lm_head": {"lm_head_feature_norm": "hfn", "token_margin": 0.02, "token_margin_reverse": False}}))
    return path


def model_args(line):
    cmd = shlex.split(line.split(") ", 1)[1])
    return dict(kv.split("=", 1) for kv in cmd[cmd.index("--model_args") + 1].split(",")), cmd


def test_lm_eval_export_convention_and_layout(tmp_path):
    hf = tmp_path / "runs" / RUN / "hf"
    for name in ("tpp50_iter17500", "main_iter14000", "main_iter2800", "tpp20_iter7000"):
        make_export(hf / name)
    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": f"{hf}/*_iter*", "VLLM_PYTHON": "/venv/bin/python",
                                                "EVAL_DATA_DIR": tmp_path / "nope"})
    assert res.returncode == 0, res.stderr
    lines = plus_lines(res.stdout)
    outs = [model_args(l)[1] for l in lines]
    assert [c[c.index("--output_path") + 1] for c in outs] == [
        f"{tmp_path}/runs/{RUN}/lm_eval/{n}/results.json"
        for n in ("main_iter2800", "main_iter14000", "tpp20_iter7000", "tpp50_iter17500")]   # version order
    args, cmd = model_args(lines[0])
    assert lines[0].startswith("(tokens: eos) /venv/bin/python -m lm_eval run --model vllm ")
    assert args["pretrained"] == f"{hf}/main_iter2800" and args["add_bos_token"] == "False"
    assert args["prefix_token_id"] == "128001" and args["dtype"] == "bfloat16"
    assert cmd[cmd.index("--tasks") + 1] == SMALL and cmd[cmd.index("--batch_size") + 1] == "auto"
    assert "--num_fewshot" not in cmd and "--limit" not in cmd and "--confirm_run_unsafe_code" not in cmd
    assert "(no " in res.stdout                               # no local datasets: the Hub

    (tmp_path / "data").mkdir()
    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": f"{hf}/tpp50_iter17500/", "TASK_SET": "core",
                                                "NUM_FEWSHOT": "5", "EVAL_DATA_DIR": tmp_path / "data",
                                                "MODEL_ARGS": "max_model_len=2048"})
    args, cmd = model_args(plus_lines(res.stdout)[0])
    assert cmd[cmd.index("--output_path") + 1] == f"{tmp_path}/runs/{RUN}/lm_eval/tpp50_iter17500_core_5shot/results.json"
    assert cmd[cmd.index("--tasks") + 1].startswith(SMALL + ",mmlu,race,wikitext")
    assert cmd[cmd.index("--num_fewshot") + 1] == "5" and args["max_model_len"] == "2048"
    assert "(offline)" in res.stdout

    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": f"{hf}/tpp50_iter17500", "TASKS": "arc_easy, humaneval",
                                                "LIMIT": "200"})
    args, cmd = model_args(plus_lines(res.stdout)[0])
    assert cmd[cmd.index("--tasks") + 1] == "arc_easy,humaneval" and cmd[cmd.index("--limit") + 1] == "200"
    assert "--confirm_run_unsafe_code" in cmd
    assert cmd[cmd.index("--output_path") + 1].endswith("/tpp50_iter17500_arc_easy_humaneval_limit200/results.json")


def test_lm_eval_reference_model_and_overrides(tmp_path):
    ref = tmp_path / "models" / "Qwen3-0.6B-Base"
    ref.mkdir(parents=True)
    (ref / "config.json").write_text(json.dumps({"eos_token_id": 151643}))
    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": ref})
    assert res.returncode == 0, res.stderr
    line = plus_lines(res.stdout)[0]
    args, cmd = model_args(line)
    assert line.startswith("(tokens: default) ") and "add_bos_token" not in args and "prefix_token_id" not in args
    assert cmd[cmd.index("--output_path") + 1] == f"{tmp_path}/models/lm_eval/Qwen3-0.6B-Base/results.json"
    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": ref, "TOKENS": "eos", "EVAL_DIR": tmp_path / "x"})
    args, cmd = model_args(plus_lines(res.stdout)[0])
    assert args["prefix_token_id"] == "151643" and cmd[cmd.index("--output_path") + 1] == f"{tmp_path}/x/results.json"
    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": ref, "TOKENS": "eos"})
    cmd = model_args(plus_lines(res.stdout)[0])[1]
    assert cmd[cmd.index("--output_path") + 1] == f"{tmp_path}/models/lm_eval/Qwen3-0.6B-Base_tokeos/results.json"


def test_lm_eval_skips_and_refusals(tmp_path):
    hf = tmp_path / "runs" / RUN / "hf"
    make_export(hf / "tpp50_iter17500")
    make_export(hf / "tpp20_iter7000", passed=False)
    done = tmp_path / "runs" / RUN / "lm_eval" / "main_iter35000"
    make_export(hf / "main_iter35000")
    done.mkdir(parents=True)
    (done / "results.json").write_text("{}")
    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": f"{hf}/*"})
    assert res.returncode == 1                               # the failed export
    assert "failed the Megatron vs HF logits check" in res.stderr
    assert "results.json exists, skipped" in res.stdout
    assert [model_args(l)[0]["pretrained"] for l in plus_lines(res.stdout)] == [f"{hf}/tpp50_iter17500"]
    assert run_script("ladder/eval/lm_eval.sh", {"MODEL": f"{hf}/*", "EVAL_DIR": "/x"}).returncode != 0
    assert run_script("ladder/eval/lm_eval.sh", {"MODEL": f"{hf}/tpp50_iter17500", "TASK_SET": "huge"}).returncode != 0
    res = run_script("ladder/eval/lm_eval.sh", {"MODEL": "/nope"}, dry=False)
    assert res.returncode != 0 and "export.sh" in res.stderr


# ---- lm_eval_util.py -------------------------------------------------------------------------------
def test_token_args(tmp_path):
    exp = make_export(tmp_path / "exp")
    assert util.token_args(str(exp)) == ("eos", "add_bos_token=False,prefix_token_id=128001")
    assert util.token_args(str(exp), "default") == ("default", "")
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "config.json").write_text(json.dumps({"eos_token_id": [7]}))
    assert util.token_args(str(ref)) == ("default", "")
    assert util.token_args(str(ref), "eos") == ("eos", "add_bos_token=False,prefix_token_id=7")
    (ref / "config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))
    with pytest.raises(SystemExit):
        util.token_args(str(ref), "eos")
    with pytest.raises(SystemExit):
        util.token_args(str(ref), "bos")


def test_version_diff(tmp_path):
    exp = make_export(tmp_path / "exp")
    here = {"transformers": "5.9.0", "tokenizers": "0.22.2"}
    assert util.version_diff(str(exp), here) == (here, [])
    assert util.version_diff(str(exp), {**here, "tokenizers": "0.23.0"})[1] == ["tokenizers 0.22.2 (export) vs 0.23.0 (here)"]
    assert util.version_diff(str(tmp_path), here) == (None, [])


RESULTS = {
    "results": {
        "hellaswag": {"alias": "hellaswag", "acc,none": 0.40, "acc_stderr,none": 0.005, "acc_norm,none": 0.50,
                      "acc_norm_stderr,none": 0.005},
        "winogrande": {"alias": "winogrande", "acc,none": 0.55, "acc_stderr,none": 0.01},
        "lambada_openai": {"alias": "lambada_openai", "perplexity,none": 20.0, "perplexity_stderr,none": 1.0,
                           "acc,none": 0.35, "acc_stderr,none": 0.01},
        "wikitext": {"alias": "wikitext", "word_perplexity,none": 30.0, "byte_perplexity,none": 1.8,
                     "bits_per_byte,none": 0.85, "word_perplexity_stderr,none": "N/A"},
        "gsm8k": {"alias": "gsm8k", "exact_match,strict-match": 0.02, "exact_match,flexible-extract": 0.03},
    },
    "n-shot": {"hellaswag": 0},
}


def test_flatten_and_scores():
    m = util.flatten_metrics(RESULTS["results"])
    assert m["hellaswag/acc_norm"] == 0.5 and m["hellaswag/acc_stderr"] == 0.005
    assert "wikitext/word_perplexity_stderr" not in m        # "N/A"
    assert m["gsm8k/exact_match,strict-match"] == 0.02
    assert util.task_score(m, "hellaswag") == ("acc_norm", 0.5)
    assert util.task_score(m, "lambada_openai") == ("acc", 0.35)
    assert util.task_score(m, "gsm8k") == ("exact_match,strict-match", 0.02)
    assert util.task_score(m, "wikitext")[0] == "word_perplexity"
    assert util.task_score(m, "piqa") is None
    assert util.name_tag_iter(f"/o/runs/{RUN}/hf/tpp50_iter17500") == (RUN, "tpp50", 17500)
    assert util.name_tag_iter("/o/sft/x/hf/iter_1602/") == ("x", None, 1602)
    assert util.name_tag_iter("/m/Qwen3-0.6B-Base") == (None, None, None)


def test_collect(tmp_path):
    exp = make_export(tmp_path / "runs" / RUN / "hf" / "tpp50_iter17500")
    out = tmp_path / "runs" / RUN / "lm_eval" / "tpp50_iter17500"
    out.mkdir(parents=True)
    (out / "results_2026-11-28T09-00-00.000000.json").write_text(json.dumps({"results": {}}))
    (out / "results_2026-11-28T10-00-00.000000.json").write_text(json.dumps(RESULTS))
    tasks = "hellaswag,winogrande,lambada_openai,wikitext,gsm8k,piqa"
    util.main(["collect", "--model", str(exp), "--out", str(out), "--tasks", tasks, "--task-set", "custom",
               "--num-fewshot", "", "--limit", "200", "--tokens", "eos"])
    assert json.loads((out / "results.json").read_text()) == RESULTS
    rows = [json.loads(l) for l in (out.parent / "summary.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert (row["run"], row["tag"], row["iteration"], row["lm_head"], row["token_margin"]) == (RUN, "tpp50", 17500,
                                                                                               "hfn", 0.02)
    assert row["score/hellaswag"] == 0.5 and row["score/wikitext"] == 30.0
    assert row["avg_score"] == pytest.approx((0.5 + 0.55 + 0.35 + 0.02) / 4) and row["avg_score_tasks"] == 4
    assert row["missing_tasks"] == ["piqa"] and row["limit"] == 200 and row["num_fewshot"] is None
    assert row["tokens"] == "eos" and row["results_file"].startswith("results_2026-11-28T10")
    assert row["hellaswag/acc"] == 0.4
    with pytest.raises(SystemExit):
        util.main(["collect", "--model", str(exp), "--out", str(tmp_path), "--tasks", "hellaswag"])


# ---- run_eval.sh -----------------------------------------------------------------------------------
def test_run_eval_chain(tmp_path):
    res = run_script("ladder/eval/run_eval.sh", {"OUT_ROOT": tmp_path, "CHAINS": "r1-hfn-d0.02"})
    assert res.returncode == 0, res.stderr
    export, ev = plus_lines(res.stderr)
    assert "CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 " in export
    assert "-J pre-export-r1-hfn-d0.02 " in export and "--dependency" not in export and "qos" not in export
    assert export.endswith("ht-run bash ladder/eval/export.sh")
    assert f"MODEL={tmp_path}/runs/{RUN}/hf/*_iter* " in ev
    assert "--dependency=afterok:pre-export-r1-hfn-d0.02 " in ev and ev.endswith("ht-run bash ladder/eval/lm_eval.sh")
    assert "export pre-export-r1-hfn-d0.02  eval lm-eval-r1-hfn-d0.02" in res.stdout

    res = run_script("ladder/eval/run_eval.sh", {"OUT_ROOT": tmp_path, "CHAINS": "r2-plain", "IDLE": "eval"})
    export, ev = plus_lines(res.stderr)
    assert "--qos" not in export and "--qos=polaris_idle" in ev
    assert f"MODEL={tmp_path}/runs/llama3_r2_muon_lr2e-3/hf/*_iter* " in ev
    res = run_script("ladder/eval/run_eval.sh", {"OUT_ROOT": tmp_path, "RUNGS": "r1"})
    assert len(plus_lines(res.stderr)) == 6
    assert run_script("ladder/eval/run_eval.sh", {"OUT_ROOT": tmp_path, "IDLE": "train"}).returncode != 0
