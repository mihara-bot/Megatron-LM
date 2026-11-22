#!/usr/bin/env python3
"""Read ladder training logs (the SLURM .out files), fit the LR sweep, and plot it.

    python ladder/analysis/lr_fit.py [--logs 'logs/*.out'] [--config llama3_r1] [--opt muon]
                                     [--window 2.0] [--noise 1e-3] [--out-dir logs/lr_fit]

Input. Every log train.sh writes starts with a plan header:
    == llama3_r1 + muon: N_eff 376738816, 1048576 tokens/iter, warmup 500, LR 1e-3
       TPP20   5600   7000   7.34B   19.5          (target, decay start, end iteration, tokens, TPP)
    == run .../runs/llama3_r1_muon_lr1e-3, mode branch (tpp20), target iteration 7000, ...
Config, optimizer, LR, tag (main | tppXX) and the branch end iterations come from it, so the file name
does not matter. Logs of the same run (a job resumed from a checkpoint) are merged. The stable run
("main") ends at the largest TPP of the plan; a branch ends at its own TPP.

Fit, per (config, optimizer, TPP). The endpoint loss is the validation loss at the end iteration.
  * quadratic in ln(LR), on the LRs within a factor --window of the best grid LR (far points are not
    quadratic and drag the vertex); needs 3 points. Reported: lr*, L*, curvature k, and an 80% interval
    for lr* from refitting with Gaussian noise --noise added to every loss (the seed noise is unknown;
    LRs share seed and data order, so their differences are less noisy than that).
  * "bracketed" = the best grid LR has a tested LR on each side. If not, lr* is an extrapolation.
Joint fit across the TPPs of a config (>= 2 TPPs):
    L(lr, T) = L*_T + k0 (T/50)^kappa (ln lr - ln lr*_T)^2,   ln lr*_T = x0 + gamma ln(T/50)
gamma > 0 means the optimal LR grows with the training horizon.

Output: a table on stdout, <out-dir>/lr_fit_<config>_<opt>.png, and <out-dir>/lr_points.csv.
Needs numpy, scipy and matplotlib.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np

HEADER = re.compile(r"^== (\S+) \+ (\S+): N_eff (\d+), (\d+) tokens/iter, warmup \d+, LR (\S+)")
PLAN = re.compile(r"^\s+TPP(\d+)\s+(\d+)\s+(\d+)\s+[\d.]+B\s+([\d.]+)")
RUN = re.compile(r"^== run (\S+), mode (\w+) \((\w+)\), target iteration (\d+)")
VALID = re.compile(r"validation loss at iteration\s+(\d+).*?lm loss value: ([0-9.Ee+-]+)")
TRAIN = re.compile(r"iteration\s+(\d+)/\s*\d+ \|.*?learning rate: ([0-9.Ee+-]+).*?lm loss: ([0-9.Ee+-]+)")
T_REF = 50.0


# ---- parsing ------------------------------------------------------------------------------------
def parse_log(path):
    info = {"plan": {}, "valid": {}, "train": {}}
    with open(path, errors="ignore") as fh:
        for line in fh:
            if "header" not in info:
                m = HEADER.match(line)
                if m:
                    info["header"] = True
                    info["config"], info["opt"] = m.group(1), m.group(2)
                    info["n_eff"], info["tok_per_iter"] = int(m.group(3)), int(m.group(4))
                    info["lr"] = float(m.group(5))
                    continue
            m = PLAN.match(line)
            if m and "tag" not in info:
                info["plan"][int(m.group(1))] = {"start": int(m.group(2)), "end": int(m.group(3)),
                                                 "tpp": float(m.group(4))}
                continue
            m = RUN.match(line)
            if m and "tag" not in info:
                info["mode"], info["tag"], info["target"] = m.group(2), m.group(3), int(m.group(4))
                continue
            m = VALID.search(line)
            if m:
                info["valid"][int(m.group(1))] = float(m.group(2))   # final re-eval overwrites: same value
                continue
            m = TRAIN.search(line)
            if m:
                info["train"][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    if "header" not in info or "tag" not in info:
        return None
    return info


def collect(pattern):
    runs = {}
    files = sorted(glob.glob(pattern))
    for f in files:
        info = parse_log(f)
        if info is None:
            print(f"skip {f}: no train.sh header", file=sys.stderr)
            continue
        key = (info["config"], info["opt"], info["lr"], info["tag"])
        if key in runs:                                  # resumed job: merge
            runs[key]["valid"].update(info["valid"])
            runs[key]["train"].update(info["train"])
            runs[key]["files"].append(os.path.basename(f))
        else:
            info["files"] = [os.path.basename(f)]
            runs[key] = info
    return runs


def endpoints(runs):
    """Rows (config, opt, lr, tpp_target, tpp_actual, end, loss or None, latest_iter)."""
    rows = []
    for (config, opt, lr, tag), r in runs.items():
        plan = r["plan"]
        if tag == "main":
            t = max(plan)
        else:
            t = int(tag[3:])
        end = plan[t]["end"]
        latest = max(list(r["valid"]) + list(r["train"]) + [0])
        rows.append({"config": config, "opt": opt, "lr": lr, "tag": tag, "tpp": t, "tpp_actual": plan[t]["tpp"],
                     "end": end, "loss": r["valid"].get(end), "latest": latest})
    return rows


# ---- fits ---------------------------------------------------------------------------------------
def quad_vertex(x, y):
    """(x*, L*, k) of y = k (x - x*)^2 + L*, or (inf/-inf, None, k) when not convex."""
    c2, c1, c0 = np.polyfit(x, y, 2)
    if c2 <= 0:
        return (math.inf if c1 < 0 else -math.inf), None, c2
    xs = -c1 / (2 * c2)
    return xs, c0 + c1 * xs + c2 * xs * xs, c2


def fit_tpp(points, window, noise, rng, n_boot=4000):
    """points: sorted [(lr, loss)]. Returns a dict describing the fit on the windowed points."""
    lrs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    i_best = int(np.argmin(ys))
    best = lrs[i_best]
    sel = np.abs(np.log(lrs / best)) <= math.log(window) + 1e-9
    out = {"best_lr": best, "best_loss": ys[i_best], "bracketed": 0 < i_best < len(lrs) - 1,
           "used": lrs[sel], "n_used": int(sel.sum())}
    if sel.sum() < 3:
        return out
    x, y = np.log(lrs[sel]), ys[sel]
    xs, ls, k = quad_vertex(x, y)
    c = np.polyfit(x, y, 2)
    out.update(lr_star=math.exp(xs) if np.isfinite(xs) else xs, loss_star=ls, k=k, coef=c,
               resid=float(np.abs(y - np.polyval(c, x)).max()))
    if noise > 0:
        v = np.array([quad_vertex(x, y + rng.normal(0, noise, len(y)))[0] for _ in range(n_boot)])
        v = np.clip(v, -50, 50)
        out["lr_ci"] = (math.exp(np.percentile(v, 10)), math.exp(np.percentile(v, 90)))
    return out


def joint_fit(points_by_t, window):
    """Joint fit across TPPs on the windowed points; None if under-determined."""
    from scipy.optimize import least_squares
    ts = sorted(points_by_t)
    if len(ts) < 2:
        return None
    X, T, Y = [], [], []
    for t in ts:
        pts = points_by_t[t]
        best = min(pts, key=lambda p: p[1])[0]
        for lr, loss in pts:
            if abs(math.log(lr / best)) <= math.log(window) + 1e-9:
                X.append(math.log(lr)); T.append(t); Y.append(loss)
    X, T, Y = np.array(X), np.array(T, float), np.array(Y)
    n_par = len(ts) + 4
    if len(Y) <= n_par:
        return None
    idx = np.array([ts.index(t) for t in T])

    def model(p, x, tt, ii):
        ls = p[:len(ts)][ii]
        x0, g, lk, kap = p[len(ts):]
        return ls + np.exp(lk) * (tt / T_REF) ** kap * (x - (x0 + g * np.log(tt / T_REF))) ** 2

    best_x = X[np.argmin(Y)]
    p0 = np.r_[[Y[T == t].min() for t in ts], best_x + 0.2, 0.0, math.log(0.02), 0.0]
    r = least_squares(lambda p: model(p, X, T, idx) - Y, p0)
    dof = len(Y) - n_par
    s2 = float((r.fun ** 2).sum() / dof)
    cov = np.linalg.pinv(r.jac.T @ r.jac) * s2
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    x0, g, lk, kap = r.x[len(ts):]
    return {"ts": ts, "p": r.x, "se": se, "rms": float(np.sqrt((r.fun ** 2).mean())), "n": len(Y),
            "x0": x0, "gamma": g, "gamma_se": se[len(ts) + 1], "k0": math.exp(lk), "kappa": kap,
            "kappa_se": se[len(ts) + 3],
            "lr_star": {t: math.exp(x0 + g * math.log(t / T_REF)) for t in ts},
            "loss_star": {t: r.x[i] for i, t in enumerate(ts)},
            "predict": lambda lr, t: float(model(r.x, np.log(np.atleast_1d(lr)), np.full(np.size(lr), float(t)),
                                                 np.full(np.size(lr), ts.index(t)))[0]) if t in ts else None}


# ---- plotting -----------------------------------------------------------------------------------
def plot(config, opt, runs, pts_by_t, fits, joint, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ts = sorted(pts_by_t)
    all_lrs = sorted({k[2] for k in runs if k[0] == config and k[1] == opt})
    cmap = plt.get_cmap("viridis")
    color = {lr: cmap(i / max(1, len(all_lrs) - 1) * 0.9) for i, lr in enumerate(all_lrs)}
    tcolor = {t: plt.get_cmap("tab10")(i) for i, t in enumerate(ts)}

    ncol = max(len(ts), 1) + 1
    fig = plt.figure(figsize=(4.2 * ncol, 8.4))
    gs = GridSpec(2, ncol, figure=fig, height_ratios=[1, 1.05], hspace=0.32, wspace=0.3)

    lo = min(all_lrs) / 1.6
    hi = max(all_lrs) * 2.2
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 200))
    for j, t in enumerate(ts):
        ax = fig.add_subplot(gs[0, j])
        pts = pts_by_t[t]
        f = fits[t]
        lr_p = np.array([p[0] for p in pts]); y_p = np.array([p[1] for p in pts])
        used = np.isin(lr_p, f["used"])
        ax.scatter(lr_p[used], y_p[used], color=tcolor[t], zorder=3, label="endpoint (in fit window)")
        if (~used).any():
            ax.scatter(lr_p[~used], y_p[~used], facecolors="none", edgecolors=tcolor[t], zorder=3,
                       label="endpoint (outside window)")
        if "coef" in f:
            u_lo, u_hi = f["used"].min(), f["used"].max()
            yy = np.polyval(f["coef"], np.log(grid))
            inside = (grid >= u_lo) & (grid <= u_hi)
            ax.plot(grid[inside], yy[inside], "-", color=tcolor[t], alpha=0.8, label="quadratic in ln lr")
            ax.plot(grid[grid >= u_hi], yy[grid >= u_hi], ":", color=tcolor[t], alpha=0.8)
            ax.plot(grid[grid <= u_lo], yy[grid <= u_lo], ":", color=tcolor[t], alpha=0.8, label="extrapolation")
            if np.isfinite(f["lr_star"]) and f["loss_star"] is not None:
                ls = f["lr_star"]
                lab = f"lr* {ls:.2e}" + ("" if f["bracketed"] else " (extrapolated)")
                ax.axvline(ls, color="k", ls="--", lw=0.8)
                if "lr_ci" in f:
                    ax.axvspan(max(f["lr_ci"][0], lo), min(f["lr_ci"][1], hi), color="0.5", alpha=0.15,
                               label="80% lr* band (noise)")
                ax.scatter([ls], [f["loss_star"]], marker="*", s=140, color="k", zorder=4, label=lab)
        if joint and t in joint["ts"]:
            yj = [joint["predict"](g, t) for g in grid]
            ax.plot(grid, yj, "-", color="crimson", lw=0.9, alpha=0.6, label="joint fit")
        ax.set_xscale("log")
        ymin = y_p.min()
        ax.set_ylim(ymin - 0.004, max(y_p.max(), ymin + 0.02) + 0.004)
        ax.set_xlim(lo, hi)
        ax.set_title(f"TPP {t}: endpoint validation loss")
        ax.set_xlabel("peak LR")
        if j == 0:
            ax.set_ylabel("validation loss")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper right")

    # lr* vs TPP
    ax = fig.add_subplot(gs[0, ncol - 1])
    for t in ts:
        f = fits[t]
        if "lr_star" in f and np.isfinite(f["lr_star"]):
            ci = f.get("lr_ci", (f["lr_star"], f["lr_star"]))
            yerr = [[f["lr_star"] - max(min(ci[0], f["lr_star"]), lo)], [max(min(ci[1], hi), f["lr_star"]) - f["lr_star"]]]
            ax.errorbar([t], [f["lr_star"]], yerr=yerr, fmt="o" if f["bracketed"] else "^",
                        color=tcolor[t], capsize=3,
                        label=f"TPP{t} per-TPP fit" + ("" if f["bracketed"] else " (extrap.)"))
        ax.scatter([t], [f["best_lr"]], marker="x", color=tcolor[t])
    if joint:
        tt = np.linspace(min(ts) * 0.8, max(ts) * 1.25, 50)
        ax.plot(tt, np.exp(joint["x0"] + joint["gamma"] * np.log(tt / T_REF)), "-", color="crimson",
                label=f"joint: lr* ~ T^{joint['gamma']:+.2f}±{joint['gamma_se']:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xticks(ts)
    ax.set_xticklabels([str(t) for t in ts])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("TPP (target)"); ax.set_ylabel("optimal LR")
    ax.set_title("optimal LR vs horizon (x = best grid LR,\nbars = 80% band clipped to the plot)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    # validation curves
    ax = fig.add_subplot(gs[1, :ncol - 1])
    ax2 = fig.add_subplot(gs[1, ncol - 1])
    floor, ceil = math.inf, -math.inf
    for (c, o, lr, tag), r in sorted(runs.items(), key=lambda kv: (kv[0][2], kv[0][3])):
        if c != config or o != opt or not r["valid"]:
            continue
        it = np.array(sorted(r["valid"]))
        v = np.array([r["valid"][i] for i in it])
        tok = it * r["tok_per_iter"] / 1e9
        ls = "-" if tag == "main" else "--"
        ax.plot(tok, v, ls, color=color[lr], lw=1.2 if tag == "main" else 1.0,
                label=f"lr {lr:.1e}" if tag == "main" else None)
        ax.scatter(tok[-1:], v[-1:], color=color[lr], s=12)
        floor = min(floor, v.min())
        ceil = max(ceil, np.percentile(v, 30))
        if r["train"]:
            ti = np.array(sorted(r["train"]))
            ax2.plot(ti * r["tok_per_iter"] / 1e9, [r["train"][i][0] for i in ti], ls, color=color[lr], lw=1)
    ax.set_ylim(floor - 0.01, ceil + 0.05)
    ax.set_xlabel("tokens (B)"); ax.set_ylabel("validation loss")
    ax.set_title("validation curves (solid = stable run, dashed = decay branches)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax2.set_xlabel("tokens (B)"); ax2.set_ylabel("learning rate")
    ax2.set_title("LR schedules")
    ax2.grid(alpha=0.3)

    fig.suptitle(f"{config} + {opt}: LR sweep", fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---- main ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs/*.out", help="glob of SLURM logs")
    ap.add_argument("--config", default="", help="only this config, e.g. llama3_r1")
    ap.add_argument("--opt", default="", help="only this optimizer, e.g. muon")
    ap.add_argument("--window", type=float, default=2.0,
                    help="fit only LRs within this factor of the best grid LR (default 2)")
    ap.add_argument("--noise", type=float, default=1e-3,
                    help="assumed per-run loss noise for the lr* interval (0 to skip)")
    ap.add_argument("--out-dir", default="logs/lr_fit",
                    help="folder for the figures (lr_fit_<config>_<opt>.png) and lr_points.csv; created if missing")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    runs = collect(args.logs)
    if args.config:
        runs = {k: v for k, v in runs.items() if k[0] == args.config}
    if args.opt:
        runs = {k: v for k, v in runs.items() if k[1] == args.opt}
    if not runs:
        sys.exit("no runs found")
    rows = endpoints(runs)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "lr_points.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["config"], r["opt"], r["tpp"], r["lr"])))

    rng = np.random.default_rng(0)
    for config, opt in sorted({(r["config"], r["opt"]) for r in rows}):
        rs = [r for r in rows if r["config"] == config and r["opt"] == opt]
        lrs = sorted({r["lr"] for r in rs})
        ts = sorted({r["tpp"] for r in rs})
        table = {(r["lr"], r["tpp"]): r for r in rs}
        print(f"\n=== {config} + {opt}")
        print("   " + f"{'lr':>9}" + "".join(f"{'TPP' + str(t):>12}" for t in ts))
        for lr in lrs:
            cells = []
            for t in ts:
                r = table.get((lr, t))
                if r is None:
                    cells.append(f"{'-':>12}")
                elif r["loss"] is None:
                    cells.append(f"{'@' + str(r['latest']):>12}")
                else:
                    cells.append(f"{r['loss']:12.4f}")
            print(f"   {lr:9.2e}" + "".join(cells))
        print("   (@N = still running, latest iteration N)")

        pts_by_t, fits = {}, {}
        for t in ts:
            pts = sorted((r["lr"], r["loss"]) for r in rs if r["tpp"] == t and r["loss"] is not None)
            if len(pts) < 2:
                continue
            pts_by_t[t] = pts
            f = fit_tpp(pts, args.window, args.noise, rng)
            fits[t] = f
            msg = f"   TPP{t:<4} best grid lr {f['best_lr']:.2e} ({f['best_loss']:.4f})"
            if not f["bracketed"]:
                edge = "high" if f["best_lr"] == pts[-1][0] else "low"
                msg += f" at the {edge} edge"
            if "lr_star" in f:
                if np.isfinite(f["lr_star"]):
                    msg += (f"; quadratic on {f['n_used']} pts: lr* {f['lr_star']:.2e}, L* {f['loss_star']:.4f}, "
                            f"k {f['k']:.4f}")
                    if "lr_ci" in f:
                        msg += f", 80% [{f['lr_ci'][0]:.2e}, {f['lr_ci'][1]:.2e}]"
                    if not f["bracketed"]:
                        msg += "  <- extrapolated"
                else:
                    msg += "; quadratic not convex (no minimum yet)"
            else:
                msg += f"; only {f['n_used']} LRs within x{args.window:g}: no fit"
            print(msg)
            if not f["bracketed"]:
                nxt = f["best_lr"] * (1.4 if f["best_lr"] == pts[-1][0] else 1 / 1.4)
                print(f"           -> add lr {nxt:.2e} to bracket it")

        joint = joint_fit(pts_by_t, args.window)
        if joint:
            print(f"   joint fit ({joint['n']} pts, resid RMS {joint['rms'] * 1e3:.2f}e-3): "
                  f"lr* ~ T^({joint['gamma']:+.3f} ± {joint['gamma_se']:.3f}), "
                  f"curvature ~ T^({joint['kappa']:+.3f} ± {joint['kappa_se']:.3f})")
            print("   " + ", ".join(f"TPP{t}: lr* {joint['lr_star'][t]:.2e} L* {joint['loss_star'][t]:.4f}"
                                     for t in joint["ts"]))
        if not args.no_plot and pts_by_t:
            path = os.path.join(args.out_dir, f"lr_fit_{config}_{opt}.png")
            plot(config, opt, runs, pts_by_t, fits, joint, path)
            print(f"   plot -> {path}")


if __name__ == "__main__":
    main()
