#!/usr/bin/env python3
"""Collect ladder results, pick LRs, and fit a Chinchilla-form scaling law.

    python ladder/analysis/scaling.py collect --out-root $OUT_ROOT [--csv results.csv]
    python ladder/analysis/scaling.py lr      --csv results.csv [--family llama3]
    python ladder/analysis/scaling.py fit     --csv results.csv [--family llama3] [--opt adamw] [--plot fit.png]

collect  Walks $OUT_ROOT/runs/<run>/tb/<tag>/ (tag = main | tppXX). Model shape, LR, batch and the end
         iteration come from the newest $OUT_ROOT/runs/<run>/cmd_<tag>_*.sh that train.sh writes, so any
         run name works. Rows: the validation loss at the run's end iteration. A stable run stopped early
         (STOP_TPP, i.e. --exit-interval) is still at constant LR and is skipped.
lr       Per (config, optimizer, tag): validation loss for each LR, and the optimum of a quadratic fit
         in log LR (>= 3 LRs). Warns when the best LR is at the edge of the grid.
fit      L(N, D) = E + A / N^alpha + B / D^beta on the best LR per (config, tag), N = N_eff
         (non-embedding params + lm_head), D = training tokens. Huber loss on log L, L-BFGS from a grid
         of starts (Hoffmann et al. 2022), bootstrap intervals, and a hold-out check: refit without the
         largest N and predict its points. Compute-optimal allocation uses the real training FLOPs per
         token (Megatron's formula: causal attention, lm_head), not 6 N_eff.

Needs numpy and scipy; collect also needs tensorboard, --plot needs matplotlib.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

FIELDS = ["run", "tag", "config", "opt", "lr", "layers", "hidden", "ffn", "heads", "groups", "kv_channels",
          "seq", "gbs", "vocab", "n_eff", "flops_per_token", "iteration", "tokens", "tpp", "loss"]


# ---- model arithmetic ---------------------------------------------------------------------------
# Padded vocab (Megatron pads len(tokenizer) to a multiple of 128 at TP=1) per config family.
FAMILY_VOCAB = {"llama3": 128256, "qwen3": 151680}   # Llama-3.1: 128256; Qwen3: 151669 -> 151680


def model_numbers(a, vocab):
    """N_eff and training FLOPs per token of a dense Llama/Qwen3-style GPT (SwiGLU, GQA, untied, no bias)."""
    L, d, f, H, g, c, s = (a[k] for k in ("layers", "hidden", "ffn", "heads", "groups", "kv_channels", "seq"))
    p = c * H                                               # attention projection width
    layer = d * p + 2 * d * c * g + p * d + 3 * d * f + 2 * d
    if a.get("qk_norm"):
        layer += 2 * c                                      # q / k RMSNorm (Qwen3), no FLOPs counted
    n_eff = L * layer + d + d * vocab                       # + final norm + lm_head
    fwd = L * (4 * p * (d + d * g / H) + 2 * s * p + 6 * d * f) + 2 * d * vocab
    return n_eff, 3 * fwd


# ---- collect ------------------------------------------------------------------------------------
def parse_cmd(path):
    toks = open(path).read().split()
    def val(flag, cast=int, default=None):
        if flag in toks:
            return cast(toks[toks.index(flag) + 1])
        return default
    a = {
        "layers": val("--num-layers"), "hidden": val("--hidden-size"), "ffn": val("--ffn-hidden-size"),
        "heads": val("--num-attention-heads"), "seq": val("--seq-length"), "gbs": val("--global-batch-size"),
        "lr": val("--lr", float), "opt": val("--optimizer", str), "train_iters": val("--train-iters"),
        "exit_interval": val("--exit-interval"),
    }
    a["groups"] = val("--num-query-groups", int, a["heads"]) if "--group-query-attention" in toks else a["heads"]
    a["kv_channels"] = val("--kv-channels", int, a["hidden"] // a["heads"])
    a["qk_norm"] = "--qk-layernorm" in toks
    # LM head + token margin (ladder/configs/head.sh): append its run-name tag to the optimizer, so
    # margin runs group apart from the plain-head runs of the same config and LR.
    mode = val("--lm-head-feature-norm", str)
    if mode is not None:
        tag = f"_{mode}" + ("_rev" if "--token-margin-reverse" in toks else "")
        tag += f"_d{val('--token-margin', str, '0')}"
        scale = val("--lm-head-feature-norm-scale", str)
        if mode != "nfn" and scale not in (None, "60"):
            tag += f"_s{scale}"
        weight = val("--soft-feature-norm-weight", str)
        if mode == "sfn" and weight not in (None, "1e-3"):
            tag += f"_t{weight}"
        a["opt"] += tag
    if "--num-experts" in toks:
        raise ValueError("MoE runs are not supported by model_numbers()")
    return a


def last_validation(tb_dir, tag_name):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag_name not in ea.Tags().get("scalars", []):
        return {}
    return {e.step: e.value for e in ea.Scalars(tag_name)}


def cmd_collect(args):
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.out_root, "runs", "*"))):
        run = os.path.basename(run_dir)
        for tb_dir in sorted(glob.glob(os.path.join(run_dir, "tb", "*"))):
            tag = os.path.basename(tb_dir)
            cmds = sorted(glob.glob(os.path.join(run_dir, f"cmd_{tag}_*.sh")))
            if not cmds:
                print(f"skip {run}/{tag}: no cmd_{tag}_*.sh", file=sys.stderr)
                continue
            a = parse_cmd(cmds[-1])
            if tag == "main":
                if a["exit_interval"]:
                    print(f"skip {run}/main: stopped at {a['exit_interval']} (constant LR)", file=sys.stderr)
                    continue
                end = a["train_iters"]
            else:
                end = a["exit_interval"]
            vals = last_validation(tb_dir, args.valid_tag)
            if end not in vals:
                done = max(vals) if vals else 0
                print(f"skip {run}/{tag}: no validation at end iteration {end} (latest {done})", file=sys.stderr)
                continue
            m = re.match(r"(.+)_(adam|adamw|muon)_lr", run)
            config = m.group(1) if m else run
            vocab = args.vocab or FAMILY_VOCAB.get(config.split("_")[0], 128256)
            n_eff, fpt = model_numbers(a, vocab)
            tokens = end * a["gbs"] * a["seq"]
            rows.append({
                "run": run, "tag": tag, "config": config, "opt": a["opt"], "lr": a["lr"],
                **{k: a[k] for k in ("layers", "hidden", "ffn", "heads", "groups", "kv_channels", "seq", "gbs")},
                "vocab": vocab, "n_eff": n_eff, "flops_per_token": fpt, "iteration": end,
                "tokens": tokens, "tpp": tokens / n_eff, "loss": vals[end],
            })
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} finished runs -> {args.csv}")


def read_rows(args):
    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        for k in FIELDS[4:]:
            r[k] = float(r[k])
    if args.family:
        rows = [r for r in rows if r["config"].startswith(args.family)]
    if getattr(args, "opt", None):
        rows = [r for r in rows if r["opt"] == args.opt]
    return rows


def group_by(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def tag_order(tag):
    return int(tag[3:]) if tag.startswith("tpp") else 10**9


# ---- LR -----------------------------------------------------------------------------------------
def lr_optimum(points):
    """(lr*, loss*) of a quadratic in log LR through (lr, loss) points, or None if not convex."""
    import numpy as np
    x = np.log([p[0] for p in points])
    y = np.array([p[1] for p in points])
    c2, c1, c0 = np.polyfit(x, y, 2)
    if c2 <= 0:
        return None
    xs = -c1 / (2 * c2)
    return math.exp(xs), c0 + c1 * xs + c2 * xs * xs


def cmd_lr(args):
    rows = read_rows(args)
    groups = group_by(rows, ("config", "opt"))
    for (config, opt), rs in sorted(groups.items()):
        print(f"\n== {config} / {opt}")
        lrs = sorted({r["lr"] for r in rs})
        tags = sorted({r["tag"] for r in rs}, key=tag_order)
        print("   " + f"{'lr':>9}" + "".join(f"{t:>10}" for t in tags))
        table = {(r["lr"], r["tag"]): r["loss"] for r in rs}
        for lr in lrs:
            cells = "".join(f"{table[(lr, t)]:10.4f}" if (lr, t) in table else f"{'-':>10}" for t in tags)
            print(f"   {lr:9.2e}{cells}")
        for t in tags:
            pts = sorted((lr, table[(lr, t)]) for lr in lrs if (lr, t) in table)
            best = min(pts, key=lambda p: p[1])
            msg = f"   {t:>6}: best grid lr {best[0]:.2e} (loss {best[1]:.4f})"
            if len(pts) >= 3:
                opt_pt = lr_optimum(pts)
                if opt_pt:
                    msg += f", quadratic optimum lr {opt_pt[0]:.2e} (loss {opt_pt[1]:.4f})"
            if len(pts) > 1 and best[0] in (pts[0][0], pts[-1][0]):
                msg += "  <- at the grid edge: extend the grid"
            print(msg)


# ---- fit ----------------------------------------------------------------------------------------
def huber(r, delta):
    import numpy as np
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def fit_law(N, D, L, delta=1e-3, starts=None):
    """Fit log L = LSE(a - alpha log N, b - beta log D, e); returns (A, B, E, alpha, beta)."""
    import itertools
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    lN, lD, lL = np.log(N), np.log(D), np.log(L)

    def obj(t):
        a, b, e, al, be = t
        pred = logsumexp(np.stack([a - al * lN, b - be * lD, np.full_like(lN, e)]), axis=0)
        return huber(pred - lL, delta).sum()

    if starts is None:
        starts = itertools.product([0, 5, 10, 15, 20, 25], [0, 5, 10, 15, 20, 25], [-1, -0.5, 0, 0.5, 1],
                                   [0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7])
    best = None
    for t0 in starts:
        res = minimize(obj, np.array(t0, float), method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    a, b, e, al, be = best.x
    return math.exp(a), math.exp(b), math.exp(e), al, be


def law(p, N, D):
    A, B, E, al, be = p
    return E + A / N**al + B / D**be


def cmd_fit(args):
    import numpy as np
    rows = read_rows(args)
    best = {}
    for key, rs in group_by(rows, ("config", "opt", "tag")).items():
        r = min(rs, key=lambda r: r["loss"])
        if args.lr_fit and len({x["lr"] for x in rs}) >= 3:
            o = lr_optimum([(x["lr"], x["loss"]) for x in rs])
            if o and min(x["lr"] for x in rs) < o[0] < max(x["lr"] for x in rs):
                r = dict(r, loss=o[1], lr=o[0])
        best[key] = r
    opts = {k[1] for k in best}
    if len(opts) > 1:
        sys.exit(f"several optimizers in the data ({sorted(opts)}): pass --opt")
    pts = sorted(best.values(), key=lambda r: (r["n_eff"], r["tokens"]))
    N = np.array([r["n_eff"] for r in pts])
    D = np.array([r["tokens"] for r in pts])
    L = np.array([r["loss"] for r in pts])
    n_sizes = len(set(N))
    print(f"{len(pts)} points, {n_sizes} model sizes")
    for r in pts:
        print(f"   {r['config']:>12} {r['tag']:>6}  N {r['n_eff']/1e9:6.3f}B  D {r['tokens']/1e9:7.2f}B  "
              f"TPP {r['tpp']:6.1f}  lr {r['lr']:.2e}  loss {r['loss']:.4f}")
    if n_sizes < 3 or len(pts) < 6:
        sys.exit("need at least 3 model sizes and 6 points for a 5-parameter fit")

    p = fit_law(N, D, L, args.huber_delta)
    resid = L - law(p, N, D)
    print(f"\nL(N,D) = {p[2]:.4f} + {p[0]:.4g} / N^{p[3]:.4f} + {p[1]:.4g} / D^{p[4]:.4f}")
    print(f"   residual RMS {np.sqrt(np.mean(resid**2)):.4f}, max |res| {np.abs(resid).max():.4f}")

    rng = np.random.default_rng(0)
    boots = []
    warm = [tuple(np.log([p[0], p[1], p[2]])) + (p[3], p[4])]
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(pts), len(pts))
        if len(set(N[idx])) < 3:
            continue
        boots.append(fit_law(N[idx], D[idx], L[idx], args.huber_delta,
                             starts=warm + [(a, b, e, 0.3, 0.3) for a in (5, 15) for b in (5, 15) for e in (0, 0.5)]))
    if boots:
        B_ = np.array(boots)
        a_exp = B_[:, 4] / (B_[:, 3] + B_[:, 4])
        q = lambda v: f"[{np.percentile(v, 5):.3f}, {np.percentile(v, 95):.3f}]"
        print(f"   bootstrap 90% ({len(boots)} fits): E {q(B_[:, 2])}  alpha {q(B_[:, 3])}  beta {q(B_[:, 4])}  "
              f"a=beta/(alpha+beta) {q(a_exp)}")

    # hold-out: refit without the largest model and predict it
    big = N.max()
    keep = N < big
    if len(set(N[keep])) >= 3:
        ph = fit_law(N[keep], D[keep], L[keep], args.huber_delta)
        print(f"\nhold-out: fit without N={big/1e9:.3f}B, predict its points")
        for n, d, l in zip(N[~keep], D[~keep], L[~keep]):
            pr = law(ph, n, d)
            print(f"   D {d/1e9:7.2f}B  actual {l:.4f}  predicted {pr:.4f}  error {pr - l:+.4f}")

    # compute-optimal allocation with the real FLOPs/token: log F = k0 + k1 log N over the rungs
    sizes = sorted({(r["n_eff"], r["flops_per_token"]) for r in pts})
    k1, k0 = np.polyfit(np.log([s[0] for s in sizes]), np.log([s[1] for s in sizes]), 1)
    fpt = lambda n: math.exp(k0) * n**k1
    print(f"\nFLOPs/token ~ {math.exp(k0):.3g} N^{k1:.3f} (6 N would be 6 N^1); "
          f"closed form N_opt ~ C^{p[4]/(p[3]+p[4]):.3f} for C = 6ND")
    from scipy.optimize import minimize_scalar
    print(f"   {'C (FLOPs)':>10} {'N_opt':>9} {'D_opt':>9} {'TPP':>7} {'loss':>7}")
    c_lo = min(r["flops_per_token"] * r["tokens"] for r in pts)
    for c in np.logspace(math.log10(c_lo), math.log10(c_lo) + 4, 9):
        res = minimize_scalar(lambda ln: law(p, math.exp(ln), c / fpt(math.exp(ln))),
                              bounds=(math.log(1e7), math.log(1e13)), method="bounded")
        n = math.exp(res.x)
        d = c / fpt(n)
        print(f"   {c:10.2e} {n/1e9:8.3f}B {d/1e9:8.1f}B {d/n:7.1f} {res.fun:7.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for n in sorted(set(N)):
            sel = N == n
            line, = ax.plot(D[sel] / 1e9, L[sel], "o", label=f"N_eff {n/1e9:.2f}B")
            ds = np.logspace(math.log10(D[sel].min() * 0.7), math.log10(D[sel].max() * 1.4), 50)
            ax.plot(ds / 1e9, law(p, n, ds), "-", color=line.get_color(), alpha=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("training tokens (B)")
        ax.set_ylabel(f"validation loss ({args.valid_tag})" if hasattr(args, "valid_tag") else "validation loss")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nplot -> {args.plot}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out-root", required=True)
    c.add_argument("--csv", default="ladder_results.csv")
    c.add_argument("--valid-tag", default="lm loss validation",
                   help="TensorBoard scalar; with several validation sets 'lm loss validation-<name>'")
    c.add_argument("--vocab", type=int, default=0,
                   help="padded vocab for every run (default: by config family, see FAMILY_VOCAB)")
    for name in ("lr", "fit"):
        s = sub.add_parser(name)
        s.add_argument("--csv", default="ladder_results.csv")
        s.add_argument("--family", default="", help="config prefix, e.g. llama3")
        s.add_argument("--opt", default="", help="adam | muon")
    f = sub.choices["fit"]
    f.add_argument("--lr-fit", action="store_true",
                   help="use the quadratic-in-log-LR optimum instead of the best grid point (>= 3 LRs)")
    f.add_argument("--huber-delta", type=float, default=1e-3)
    f.add_argument("--bootstrap", type=int, default=200)
    f.add_argument("--plot", default="")
    args = ap.parse_args()
    {"collect": cmd_collect, "lr": cmd_lr, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
