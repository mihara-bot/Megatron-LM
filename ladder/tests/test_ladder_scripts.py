"""CPU-only tests for the ladder scripts: train.sh schedule, sweep.sh job graph, analysis/scaling.py.

    python -m pytest ladder/tests -p no:cacheprovider -o addopts=""

No GPU, SLURM or Megatron import needed: train.sh and sweep.sh run with DRY_RUN=1 (they only print the
commands), and scaling.py is exercised on fake runs whose TensorBoard files follow a known scaling law.
The scaling.py tests need numpy and scipy (collect also tensorboard) and are skipped without them.
"""
import importlib.util
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIGS = sorted(p.stem for p in (REPO / "ladder/configs").glob("llama3_r*.sh"))
FAKE_ENV = {"TRAIN_DATA": "x", "VALID_DATA": "x", "TOKENIZER": "x"}
VALID_TAG = "lm loss validation"   # what Megatron's torch SummaryWriter writes, spaces included


# ---- helpers ------------------------------------------------------------------------------------
def config_vars(config):
    """The config's variables, as train.sh sees them (LR/MBS defaults included)."""
    out = subprocess.run(
        ["bash", "-c", f"source ladder/configs/{config}.sh; "
                       "echo $N_EFF $GBS $SEQ_LEN $TPP_MAX $SAVE_INTERVAL $EVAL_INTERVAL $NODES; "
                       "echo $BRANCH_TPPS; echo $LR_GRID"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    keys = ["N_EFF", "GBS", "SEQ_LEN", "TPP_MAX", "SAVE_INTERVAL", "EVAL_INTERVAL", "NODES"]
    v = dict(zip(keys, map(int, out[0].split())))
    v["BRANCH_TPPS"] = [int(t) for t in out[1].split()]
    v["LR_GRID"] = out[2].split()
    return v


def train_cmd(config, out_root, **env):
    """Megatron arguments train.sh would launch, as a list."""
    full = {**os.environ, **FAKE_ENV, "OUT_ROOT": str(out_root), "CONFIG": config, "DRY_RUN": "1",
            **{k: str(v) for k, v in env.items()}}
    for k in ("MODE", "TPP", "STOP_TPP", "LR", "OPT"):
        if k not in env:
            full.pop(k, None)
    res = subprocess.run(["bash", "ladder/train.sh"], cwd=REPO, env=full, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    return shlex.split(res.stdout.strip().splitlines()[-1])


def arg(cmd, flag, cast=int):
    return cast(cmd[cmd.index(flag) + 1]) if flag in cmd else None


def run_sweep(**env):
    full = {**os.environ, "DRY_RUN": "1", **{k: str(v) for k, v in env.items()}}
    for k in ("LRS", "UPTO", "QOS", "NODES", "OPT"):
        if k not in env:
            full.pop(k, None)
    return subprocess.run(["bash", "ladder/sweep.sh"], cwd=REPO, env=full, capture_output=True, text=True)


def sbatch_jobs(stderr):
    """Parse the sbatch lines sweep.sh prints: [{name, nodes, qos, time, dep, export: {..}}]."""
    jobs = []
    for line in stderr.splitlines():
        if not line.startswith("sbatch "):
            continue
        opts = dict(re.findall(r"--([a-z-]+)=(\S+)", line))
        export = dict(kv.split("=", 1) for kv in opts["export"].split(",")[1:])
        dep = opts.get("dependency", "")
        jobs.append({"name": opts["job-name"], "nodes": int(opts["nodes"]), "qos": opts.get("qos"),
                     "time": opts["time"], "dep": dep.removeprefix("afterok:"), "export": export})
    return jobs


# ---- train.sh schedule --------------------------------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS)
def test_train_schedule_invariants(config, tmp_path):
    v = config_vars(config)
    tok = v["GBS"] * v["SEQ_LEN"]
    stable = train_cmd(config, tmp_path)
    e_max = arg(stable, "--train-iters")
    save, retain = arg(stable, "--save-interval"), arg(stable, "--save-retain-interval")
    assert save == v["SAVE_INTERVAL"]
    assert retain % save == 0, "Megatron asserts save_retain_interval % save_interval == 0"
    assert e_max % v["EVAL_INTERVAL"] == 0, "the stable run must validate at its last iteration"
    assert abs(e_max * tok / v["N_EFF"] / v["TPP_MAX"] - 1) < 0.05
    assert arg(stable, "--lr-wsd-decay-iters") == e_max - e_max * 4 // 5

    for t in v["BRANCH_TPPS"]:
        br = train_cmd(config, tmp_path, MODE="branch", TPP=t)
        start, end = arg(br, "--ckpt-step"), arg(br, "--exit-interval")
        bsave, bretain = arg(br, "--save-interval"), arg(br, "--save-retain-interval")
        # same dataset index as the stable run, so the branch reads the batches the stable run would
        assert arg(br, "--train-iters") == e_max
        assert start == end * 4 // 5 and end * 4 % 5 == 0
        assert start % retain == 0, "the stable run must keep the checkpoint the branch starts from"
        assert end % v["EVAL_INTERVAL"] == 0, "the branch must validate at its exit iteration"
        assert bretain == end and bretain % bsave == 0 and start % bsave == 0 and bsave <= save
        assert arg(br, "--lr-decay-iters") == end and arg(br, "--lr-wsd-decay-iters") == end - start
        assert "--override-opt-param-scheduler" in br
        assert abs(end * tok / v["N_EFF"] / t - 1) < 0.05


def test_train_stop_tpp_exits_at_decay_start(tmp_path):
    stable = train_cmd("llama3_r1", tmp_path)
    stop = train_cmd("llama3_r1", tmp_path, STOP_TPP=20)
    br = train_cmd("llama3_r1", tmp_path, MODE="branch", TPP=20)
    assert arg(stop, "--exit-interval") == arg(br, "--ckpt-step")
    assert arg(stop, "--train-iters") == arg(stable, "--train-iters")   # same data order and schedule
    assert arg(stop, "--lr-wsd-decay-iters") == arg(stable, "--lr-wsd-decay-iters")


def test_train_rejects_unknown_branch(tmp_path):
    full = {**os.environ, **FAKE_ENV, "OUT_ROOT": str(tmp_path), "CONFIG": "llama3_r1", "DRY_RUN": "1",
            "MODE": "branch", "TPP": "30"}
    res = subprocess.run(["bash", "ladder/train.sh"], cwd=REPO, env=full, capture_output=True, text=True)
    assert res.returncode != 0 and "not in BRANCH_TPPS" in res.stderr


# ---- sweep.sh -----------------------------------------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS)
def test_sweep_full_grid(config):
    v = config_vars(config)
    res = run_sweep(CONFIG=config)
    assert res.returncode == 0, res.stderr
    jobs = sbatch_jobs(res.stderr)
    assert len(jobs) == len(v["LR_GRID"]) * (1 + len(v["BRANCH_TPPS"]))
    stables = [j for j in jobs if j["export"]["MODE"] == "stable"]
    assert [j["export"]["LR"] for j in stables] == v["LR_GRID"]
    for j in jobs:
        assert j["nodes"] == v["NODES"] and j["qos"] is None
        assert j["export"]["CONFIG"] == config and j["export"]["OPT"] == "adamw"
    for s in stables:
        assert s["dep"] == "" and "STOP_TPP" not in s["export"] and s["time"] == "72:00:00"
        branches = [j for j in jobs if j["dep"] == f"dry:{s['name']}"]
        assert sorted(int(b["export"]["TPP"]) for b in branches) == v["BRANCH_TPPS"]
        for b in branches:
            assert b["export"]["MODE"] == "branch" and b["export"]["LR"] == s["export"]["LR"]


def test_sweep_upto_screen_and_options():
    res = run_sweep(CONFIG="llama3_r3", UPTO=20, LRS="8e-4 1e-3", QOS="polaris_idle", NODES=2, OPT="muon")
    assert res.returncode == 0, res.stderr
    jobs = sbatch_jobs(res.stderr)
    assert len(jobs) == 4                                   # 2 LRs x (stable + TPP20 branch)
    for j in jobs:
        assert j["nodes"] == 2 and j["qos"] == "polaris_idle" and j["export"]["OPT"] == "muon"
    stables = [j for j in jobs if j["export"]["MODE"] == "stable"]
    assert {j["export"]["LR"] for j in stables} == {"8e-4", "1e-3"}
    assert all(j["export"]["STOP_TPP"] == "20" for j in stables)
    assert all(j["export"]["TPP"] == "20" for j in jobs if j["export"]["MODE"] == "branch")


def test_sweep_does_not_leak_config_lr():
    # the config's LR default must not reach the jobs: each job gets exactly one LR, its own
    jobs = sbatch_jobs(run_sweep(CONFIG="llama3_r1", LRS="5e-4").stderr)
    assert {j["export"]["LR"] for j in jobs} == {"5e-4"}


@pytest.mark.parametrize("upto", ["30", "200"])
def test_sweep_rejects_bad_upto(upto):
    res = run_sweep(CONFIG="llama3_r1", UPTO=upto)
    assert res.returncode != 0 and "UPTO" in res.stderr and "sbatch" not in res.stderr


def test_sweep_dry_run_has_no_side_effects():
    before = (REPO / "logs").exists()
    run_sweep(CONFIG="llama3_r1")
    assert (REPO / "logs").exists() == before


# ---- analysis/scaling.py ------------------------------------------------------------------------
def load_scaling():
    spec = importlib.util.spec_from_file_location("ladder_scaling", REPO / "ladder/analysis/scaling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TRUE = dict(A=400.0, B=410.0, E=1.8, alpha=0.34, beta=0.28)
LR_STAR = {"llama3_r1": 1.1e-3, "llama3_r2": 9e-4, "llama3_r3": 7.5e-4, "llama3_r4": 7e-4}


def true_loss(n, d, lr=None, lr_star=None):
    loss = TRUE["E"] + TRUE["A"] / n ** TRUE["alpha"] + TRUE["B"] / d ** TRUE["beta"]
    if lr is not None:
        loss += 0.03 * math.log(lr / lr_star) ** 2
    return loss


def write_scalars(tb_dir, points):
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    w = EventFileWriter(str(tb_dir))
    for step, value in points:
        summ = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=VALID_TAG, simple_value=value)])
        w.add_event(event_pb2.Event(step=step, wall_time=time.time(), summary=summ))
    w.close()


def make_run(out_root, config, lr, tag, cmd, end, loss):
    """A finished (or not) run as train.sh + Megatron leave it: cmd_<tag>_*.sh and tb/<tag>/."""
    run_dir = out_root / "runs" / f"{config}_adamw_lr{lr}"
    (run_dir / "tb" / tag).mkdir(parents=True, exist_ok=True)
    (run_dir / f"cmd_{tag}_20261121-000000.sh").write_text(shlex.join(["pretrain_gpt.py", *cmd]))
    write_scalars(run_dir / "tb" / tag, [(it, loss + 0.3 * (1 - it / end)) for it in range(500, end + 1, 500)])


@pytest.fixture(scope="module")
def fake_out_root(tmp_path_factory):
    pytest.importorskip("tensorboard")
    out = tmp_path_factory.mktemp("out_root")
    grids = {"llama3_r1": ["5e-4", "1e-3", "2e-3"], "llama3_r2": ["6e-4", "9e-4", "1.3e-3"],
             "llama3_r3": ["8e-4"], "llama3_r4": ["7e-4"]}
    for config, lrs in grids.items():
        v = config_vars(config)
        for lr in lrs:
            runs = [("main", train_cmd(config, out, LR=lr))]
            runs += [(f"tpp{t}", train_cmd(config, out, LR=lr, MODE="branch", TPP=t)) for t in v["BRANCH_TPPS"]]
            for tag, cmd in runs:
                end = arg(cmd, "--exit-interval") or arg(cmd, "--train-iters")
                d = end * v["GBS"] * v["SEQ_LEN"]
                make_run(out, config, lr, tag, cmd, end, true_loss(v["N_EFF"], d, float(lr), LR_STAR[config]))
    # a screening run stopped at the TPP20 decay start (constant LR): must be skipped
    cmd = train_cmd("llama3_r1", out, LR="3e-3", STOP_TPP=20)
    make_run(out, "llama3_r1", "3e-3", "main", cmd, arg(cmd, "--exit-interval"), 2.0)
    # a branch that has not reached its end yet: must be skipped
    cmd = train_cmd("llama3_r2", out, LR="2e-3", MODE="branch", TPP=50)
    run_dir = out / "runs" / "llama3_r2_adamw_lr2e-3"
    (run_dir / "tb" / "tpp50").mkdir(parents=True)
    (run_dir / "cmd_tpp50_20261121-000000.sh").write_text(shlex.join(["pretrain_gpt.py", *cmd]))
    write_scalars(run_dir / "tb" / "tpp50", [(500, 4.0), (1000, 3.9)])
    return out


def test_model_numbers_match_configs(tmp_path):
    sc = load_scaling()
    for config in CONFIGS:
        cmd_file = tmp_path / f"cmd_{config}.sh"
        cmd_file.write_text(shlex.join(["pretrain_gpt.py", *train_cmd(config, tmp_path)]))
        n_eff, fpt = sc.model_numbers(sc.parse_cmd(cmd_file), 128256)
        assert n_eff == config_vars(config)["N_EFF"], config
        if config == "llama3_r1":   # hand count of Megatron's num_floating_point_operations (causal attention)
            fwd = 18 * (4 * 1024 * (1024 + 1024 * 2 / 8) + 2 * 4096 * 1024 + 6 * 1024 * 3584) + 2 * 1024 * 128256
            assert fpt == pytest.approx(3 * fwd)


def test_lr_optimum_recovers_vertex():
    pytest.importorskip("numpy")
    sc = load_scaling()
    pts = [(lr, 2.5 + 0.03 * math.log(lr / 8e-4) ** 2) for lr in (3e-4, 6e-4, 1.2e-3, 2.4e-3)]
    lr, loss = sc.lr_optimum(pts)
    assert lr == pytest.approx(8e-4, rel=1e-6) and loss == pytest.approx(2.5, abs=1e-9)
    assert sc.lr_optimum([(1e-3, 2.0), (2e-3, 2.5), (4e-3, 2.0)]) is None   # concave: no optimum


def test_fit_law_recovers_parameters():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    sc = load_scaling()
    N, D = np.meshgrid([3.8e8, 5.9e8, 9.3e8, 1.8e9], [20, 50, 100])
    D = (D * N).ravel()
    N = N.ravel()
    A, B, E, al, be = sc.fit_law(N, D, true_loss(N, D))
    assert E == pytest.approx(TRUE["E"], abs=0.02)
    assert al == pytest.approx(TRUE["alpha"], abs=0.02) and be == pytest.approx(TRUE["beta"], abs=0.02)
    assert np.abs(sc.law((A, B, E, al, be), N, D) - true_loss(N, D)).max() < 1e-3


def test_collect_lr_fit_end_to_end(fake_out_root, tmp_path):
    pytest.importorskip("scipy")
    script = REPO / "ladder/analysis/scaling.py"
    csv_path = tmp_path / "results.csv"
    res = subprocess.run([sys.executable, str(script), "collect", "--out-root", str(fake_out_root),
                          "--csv", str(csv_path)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    # 3 + 3 + 1 + 1 LRs, each main + TPP20 + TPP50; the stopped and the unfinished run are skipped
    assert len(rows) == 8 * 3
    assert "llama3_r1_adamw_lr3e-3/main: stopped" in res.stderr
    assert "llama3_r2_adamw_lr2e-3/tpp50: no validation at end" in res.stderr
    for r in rows:
        v = config_vars(r["config"])
        assert int(r["n_eff"]) == v["N_EFF"] and r["opt"] == "adam"
        assert float(r["loss"]) == pytest.approx(
            true_loss(v["N_EFF"], float(r["tokens"]), float(r["lr"]), LR_STAR[r["config"]]), abs=1e-5)

    res = subprocess.run([sys.executable, str(script), "lr", "--csv", str(csv_path)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "== llama3_r1 / adam" in res.stdout and "best grid lr 1.00e-03" in res.stdout

    fit = subprocess.run([sys.executable, str(script), "fit", "--csv", str(csv_path), "--family", "llama3",
                          "--lr-fit", "--bootstrap", "5"], capture_output=True, text=True)
    assert fit.returncode == 0, fit.stderr
    assert "12 points, 4 model sizes" in fit.stdout
    m = re.search(r"L\(N,D\) = ([\d.]+) \+ \S+ / N\^([\d.]+) \+ \S+ / D\^([\d.]+)", fit.stdout)
    E, al, be = map(float, m.groups())
    # R3/R4 have one LR each, at 8e-4 / 7e-4 vs optima 7.5e-4 / 7e-4: a small, known bias
    assert E == pytest.approx(TRUE["E"], abs=0.05)
    assert al == pytest.approx(TRUE["alpha"], abs=0.03) and be == pytest.approx(TRUE["beta"], abs=0.03)
    errors = [float(x) for x in re.findall(r"error ([+-][\d.]+)", fit.stdout)]
    assert len(errors) == 3 and max(map(abs, errors)) < 0.01   # hold-out: predict R4 from R1-R3


def test_fit_refuses_too_few_sizes(tmp_path):
    pytest.importorskip("scipy")
    csv_path = tmp_path / "r.csv"
    sc = load_scaling()
    with open(csv_path, "w") as fh:
        fh.write(",".join(sc.FIELDS) + "\n")
        for n in (3.8e8, 5.9e8):
            for tpp in (20, 50, 100):
                row = dict.fromkeys(sc.FIELDS, 0)
                row.update(run="r", tag=f"tpp{tpp}", config=f"llama3_{n}", opt="adam", lr=1e-3, n_eff=n,
                           tokens=n * tpp, tpp=tpp, flops_per_token=6 * n, loss=true_loss(n, n * tpp))
                fh.write(",".join(str(row[k]) for k in sc.FIELDS) + "\n")
    res = subprocess.run([sys.executable, str(REPO / "ladder/analysis/scaling.py"), "fit", "--csv", str(csv_path)],
                         capture_output=True, text=True)
    assert res.returncode != 0 and "at least 3 model sizes" in res.stderr
