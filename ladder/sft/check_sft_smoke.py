"""Check an SFT smoke run (ladder/sft/smoke_sft.sh): training, checkpoints, head metrics, export, eval.

    python ladder/sft/check_sft_smoke.py --sft-dir <dir> --stats <data>/stats.json --head hfn [--expect-eval]

Fails (exit 1) with one line per problem:
  * training reached --train-iters; every logged lm loss finite; the first one below 5 (a pretrained model, not a
    random one: ln V ~ 12) and the last few lower than the first few
  * LR peaked at --lr and decayed to ~0 (cosine to 0)
  * with --expect-valid: the validation loss (lm loss validation) logged at least twice, finite, below 5
  * a checkpoint per epoch plus the final one, the tracker at the final iteration
  * head metrics: present for nfn / hfn (ce loss = lm loss, margin r = 1, since d = 0 in SFT), absent for plain
  * every HF export passed its Megatron vs HF logits check
  * with --expect-eval: every export has eval results with finite pass@1 per task
Needs tensorboard (in the NGC image).
"""

import argparse
import glob
import json
import math
import os
import shlex
import sys


def scalars(tb_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in ea.Tags().get("scalars", [])}


def flag(cmd, name, cast=str):
    return cast(cmd[cmd.index(name) + 1]) if name in cmd else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sft-dir", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--head", choices=["plain", "nfn", "hfn"], required=True)
    parser.add_argument("--expect-eval", action="store_true")
    parser.add_argument("--expect-valid", action="store_true")
    args = parser.parse_args(argv)
    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)
        return ok

    cmds = sorted(glob.glob(os.path.join(args.sft_dir, "cmd_*.sh")))
    if not cmds:
        sys.exit(f"no cmd_*.sh in {args.sft_dir}")
    cmd = shlex.split(open(cmds[-1]).read())
    train_iters, save_interval = flag(cmd, "--train-iters", int), flag(cmd, "--save-interval", int)
    peak_lr = flag(cmd, "--lr", float)

    sc = scalars(os.path.join(args.sft_dir, "tb"))
    loss = sc.get("lm loss", [])
    if check(len(loss) >= 4, f"only {len(loss)} 'lm loss' points in tensorboard"):
        values = [v for _, v in loss]
        check(all(math.isfinite(v) for v in values), "non-finite lm loss")
        check(loss[-1][0] >= train_iters - 10, f"last logged iteration {loss[-1][0]} < train iters {train_iters}")
        check(values[0] < 5.0, f"first lm loss {values[0]:.3f} >= 5: pretrained weights not loaded?")
        k = max(1, len(values) // 5)
        first, last = sum(values[:k]) / k, sum(values[-k:]) / k
        check(last < first, f"lm loss did not go down: first {first:.4f}, last {last:.4f}")
        print(f"lm loss {values[0]:.4f} -> {values[-1]:.4f} over {loss[-1][0]} iterations")
    if args.expect_valid:
        val = sc.get("lm loss validation", [])
        if check(len(val) >= 2, f"only {len(val)} 'lm loss validation' points in tensorboard"):
            values = [v for _, v in val]
            check(all(math.isfinite(v) for v in values), "non-finite validation loss")
            check(values[0] < 5.0, f"first validation loss {values[0]:.3f} >= 5")
            print(f"validation loss {values[0]:.4f} -> {values[-1]:.4f} at iterations {[s for s, _ in val]}")
    lr = [v for _, v in sc.get("learning-rate", [])]
    if check(bool(lr), "no learning-rate in tensorboard"):
        check(abs(max(lr) - peak_lr) <= 0.05 * peak_lr, f"peak LR {max(lr):.3g} != --lr {peak_lr:.3g}")
        check(lr[-1] <= 0.05 * peak_lr, f"final LR {lr[-1]:.3g} not ~0 (cosine to 0)")

    head_tags = ("ce loss", "margin r", "lm head input norm", "logit max", "logit rms")
    if args.head == "plain":
        check(not any(t in sc for t in head_tags), "head metrics logged for a plain head")
    else:
        missing = [t for t in head_tags if t not in sc]
        if check(not missing, f"head metrics missing: {missing}"):
            ce = dict(sc["ce loss"])
            gap = max(abs(ce[s] - v) for s, v in loss if s in ce)
            check(gap < 1e-3, f"ce loss differs from lm loss by {gap:.2e} although SFT has d = 0 and no penalty")
            r = [v for _, v in sc["margin r"]]
            check(all(abs(x - 1) < 1e-6 for x in r), "margin r != 1 with d = 0")

    ckpt = os.path.join(args.sft_dir, "ckpt")
    iters = sorted(int(os.path.basename(p)[5:]) for p in glob.glob(os.path.join(ckpt, "iter_*")))
    tracker = os.path.join(ckpt, "latest_checkpointed_iteration.txt")
    expected = sorted(set(list(range(save_interval, train_iters + 1, save_interval)) + [train_iters]))
    check(iters == expected, f"checkpoints {iters}, expected {expected}")
    check(os.path.exists(tracker) and open(tracker).read().strip() == str(train_iters),
          f"tracker does not point at {train_iters}")

    for it in iters:
        meta_path = os.path.join(args.sft_dir, "hf", f"iter_{it}", "export_meta.json")
        if not check(os.path.exists(meta_path), f"iteration {it} not exported ({meta_path})"):
            continue
        v = json.load(open(meta_path)).get("validation") or {}
        check(v.get("passed") is True, f"iteration {it}: Megatron vs HF logits check {v}")
        if args.expect_eval:
            found = glob.glob(os.path.join(args.sft_dir, "eval", f"iter_{it}_*", "summary.json"))
            if check(bool(found), f"iteration {it}: no eval summary"):
                m = json.load(open(found[0]))["metrics"]
                for task in json.load(open(found[0]))["tasks"]:
                    x = m.get(f"{task}/pass@1")
                    check(x is not None and math.isfinite(x), f"iteration {it}: {task}/pass@1 = {x}")
                print(f"iteration {it}: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(m.items())
                                                       if k.endswith(("pass@1", "greedy_acc"))))

    stats = json.load(open(args.stats))
    print(f"data: {stats['examples']} examples x {stats['epochs']} epochs; train iters {train_iters}, "
          f"checkpoints {iters}")
    if problems:
        print("SMOKE FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
