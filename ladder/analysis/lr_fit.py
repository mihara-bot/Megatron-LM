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

Output: a table on stdout, and per (config, optimizer) <out-dir>/lr_fit_<config>_<opt>.png and
<out-dir>/lr_points_<config>_<opt>.csv (one file each, so fitting one rung never overwrites another).
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
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
JOINT = "#eb6834"                                   # joint fit (categorical slot 2)
BAND = "#e9e8e4"                                    # lr* noise band
LR_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
           "#104281", "#0d366b"]   # one blue hue, steps 250-700; light = small LR


def set_style():
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    serif = [n for n in ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif") if n in names]
    if "Times New Roman" not in names:
        print(f"   note: Times New Roman not installed, using {serif[0] if serif else 'DejaVu Serif'}",
              file=sys.stderr)
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": serif + ["DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def sci(x, digits=2):
    """x as mathtext a x 10^b."""
    if not np.isfinite(x):
        return r"\infty"
    e = int(math.floor(math.log10(abs(x))))
    return rf"{x / 10**e:.{digits}f}\times10^{{{e}}}"


def lr_axis(ax, lrs, lo, hi, which="x"):
    """Log LR axis with ticks at the tested LRs, labelled in units of 1e-3."""
    import matplotlib.ticker as mt
    ticks = [l for l in lrs if lo <= l <= hi]
    labels = [f"{l * 1e3:g}" for l in ticks]
    if which == "x":
        ax.set_xscale("log"); ax.set_xlim(lo, hi)
        ax.set_xticks(ticks); ax.set_xticklabels(labels)
        ax.xaxis.set_minor_formatter(mt.NullFormatter())
    else:
        ax.set_yscale("log"); ax.set_ylim(lo, hi)
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
        ax.yaxis.set_minor_formatter(mt.NullFormatter())


def plot(config, opt, runs, pts_by_t, fits, joint, path, window, noise):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    set_style()

    mine = {k: r for k, r in runs.items() if k[0] == config and k[1] == opt}
    any_run = next(iter(mine.values()))
    plan, tok = any_run["plan"], any_run["tok_per_iter"]
    ts = sorted(pts_by_t)
    all_lrs = sorted({k[2] for k in mine})
    idx = np.linspace(0, len(LR_RAMP) - 1, len(all_lrs)).round().astype(int) if len(all_lrs) > 1 else [2]
    color = {lr: LR_RAMP[i] for lr, i in zip(all_lrs, idx)}
    lo, hi = min(all_lrs) / 1.5, max(all_lrs) * 2.4
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), 300))

    ncol = len(ts) + 1
    fig = plt.figure(figsize=(4.1 * ncol, 10.6))
    gs = GridSpec(2, ncol, figure=fig, height_ratios=[1, 0.95], hspace=0.40, wspace=0.28,
                  left=0.055, right=0.985, top=0.88, bottom=0.215)

    # -- row 1: loss vs LR per TPP ---------------------------------------------------------------
    for j, t in enumerate(ts):
        ax = fig.add_subplot(gs[0, j])
        f = fits[t]
        lr_p = np.array([p[0] for p in pts_by_t[t]]); y_p = np.array([p[1] for p in pts_by_t[t]])
        used = np.isin(lr_p, f["used"])
        if "lr_ci" in f and np.isfinite(f.get("lr_star", np.nan)):
            ax.axvspan(max(f["lr_ci"][0], lo), min(f["lr_ci"][1], hi), color=BAND, lw=0, zorder=0)
        if joint and t in joint["ts"]:
            ax.plot(grid, [joint["predict"](g, t) for g in grid], "--", color=JOINT, lw=1.3, zorder=2)
        if "coef" in f:
            yy = np.polyval(f["coef"], np.log(grid))
            u_lo, u_hi = f["used"].min(), f["used"].max()
            ins = (grid >= u_lo) & (grid <= u_hi)
            ax.plot(grid[ins], yy[ins], "-", color=INK, lw=1.6, zorder=3)
            ax.plot(grid[grid >= u_hi], yy[grid >= u_hi], ":", color=INK, lw=1.4, zorder=3)
            ax.plot(grid[grid <= u_lo], yy[grid <= u_lo], ":", color=INK, lw=1.4, zorder=3)
        ax.scatter(lr_p[used], y_p[used], s=46, color=INK2, edgecolors="white", linewidths=1.2, zorder=4)
        if (~used).any():
            ax.scatter(lr_p[~used], y_p[~used], s=46, facecolors="white", edgecolors=INK2, linewidths=1.2,
                       zorder=4)
        txt = []
        if "lr_star" in f and np.isfinite(f["lr_star"]) and f["loss_star"] is not None:
            ax.scatter([f["lr_star"]], [f["loss_star"]], marker="*", s=190, color=INK, edgecolors="white",
                       linewidths=0.8, zorder=5)
            txt.append(rf"$\eta^* = {sci(f['lr_star'])}$" + ("" if f["bracketed"] else "  (extrapolated)"))
            if "lr_ci" in f:
                c_lo, c_hi = f["lr_ci"]
                c_hi_s = rf"> {sci(hi, 1)}" if c_hi > hi else sci(c_hi)
                txt.append(rf"80%: $[{sci(c_lo)},\ {c_hi_s}]$")
            txt.append(rf"$L^* = {f['loss_star']:.4f}$,  $k = {f['k']:.4f}$")
        elif "lr_star" in f:
            txt.append("no minimum yet (fit not convex)")
        else:
            txt.append(f"only {f['n_used']} LRs in window: no fit")
        ax.text(0.03, 0.04, "\n".join(txt), transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom",
                linespacing=1.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8))
        lr_axis(ax, all_lrs, lo, hi)
        ymin, ymax = y_p.min(), max(y_p.max(), y_p.min() + 0.02)
        pad = 0.12 * (ymax - ymin)
        ax.set_ylim(ymin - 2.2 * pad - 0.004, ymax + pad)
        n_tok = plan[t]["end"] * tok / 1e9
        ax.set_title(f"TPP {t}  ({n_tok:.1f}B tokens)", color=INK)
        ax.set_xlabel(r"peak LR ($\times10^{-3}$)")
        if j == 0:
            ax.set_ylabel("final validation loss")
        ax.grid(axis="x", visible=False)

    # -- row 1, last: lr* vs TPP -----------------------------------------------------------------
    ax = fig.add_subplot(gs[0, ncol - 1])
    for t in ts:
        f = fits[t]
        ax.scatter([t], [f["best_lr"]], marker="x", s=46, color=MUTED, linewidths=1.4, zorder=3)
        if "lr_star" in f and np.isfinite(f["lr_star"]):
            s = f["lr_star"]
            ci = f.get("lr_ci", (s, s))
            lo_e, hi_e = max(min(ci[0], s), lo), max(min(ci[1], hi), s)
            ax.errorbar([t], [s], yerr=[[s - lo_e], [hi_e - s]], fmt="none", ecolor=INK2, elinewidth=1.1,
                        capsize=3, zorder=3)
            ax.scatter([t], [s], marker="o" if f["bracketed"] else "^", s=50,
                       facecolors=INK if f["bracketed"] else "white", edgecolors=INK, linewidths=1.3, zorder=4)
            if ci[1] > hi:
                ax.annotate("", xy=(t, hi), xytext=(t, hi / 1.25),
                            arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
    if joint:
        tt = np.exp(np.linspace(math.log(min(ts) * 0.75), math.log(max(ts) * 1.35), 60))
        ax.plot(tt, np.exp(joint["x0"] + joint["gamma"] * np.log(tt / T_REF)), "--", color=JOINT, lw=1.4)
    ax.set_xscale("log")
    ax.set_xticks(ts); ax.set_xticklabels([str(t) for t in ts])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(min(ts) * 0.75, max(ts) * 1.35)
    lr_axis(ax, all_lrs, lo, hi, which="y")
    ax.set_xlabel("TPP (training horizon)")
    ax.set_ylabel(r"optimal LR ($\times10^{-3}$)")
    ax.set_title(r"$\eta^*$ vs. horizon", color=INK)
    ax.legend(handles=[
        Line2D([], [], marker="x", ls="none", color=MUTED, markersize=7, label="best grid LR"),
        Line2D([], [], marker="o", ls="none", color=INK, markersize=7, label=r"$\eta^*$ bracketed"),
        Line2D([], [], marker="^", ls="none", mfc="white", mec=INK, markersize=7, label=r"$\eta^*$ extrapolated"),
        Line2D([], [], ls="--", color=JOINT, lw=1.4, label="joint fit"),
    ], loc="best", fontsize=9, handletextpad=0.4, borderaxespad=0.2, framealpha=0.9, frameon=True,
        edgecolor=GRID)

    # shared legend for row 1
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=INK2, mec="white", markersize=8, label="run (in fit window)"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, markersize=8,
               label=f"run (outside ×{window:g} window)"),
        Line2D([], [], ls="-", color=INK, lw=1.6, label="per-TPP quadratic fit"),
        Line2D([], [], ls=":", color=INK, lw=1.4, label="extrapolation"),
        Line2D([], [], marker="*", ls="none", color=INK, markersize=12, label=r"vertex $(\eta^*, L^*)$"),
        Patch(color=BAND, label=rf"80% band of $\eta^*$ ($\sigma={noise:g}$)"),
        Line2D([], [], ls="--", color=JOINT, lw=1.3, label="joint fit"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=7, fontsize=10, handlelength=1.8,
        columnspacing=1.4)

    # -- row 2: validation curves + LR schedule --------------------------------------------------
    ax = fig.add_subplot(gs[1, :ncol - 1])
    ax2 = fig.add_subplot(gs[1, ncol - 1])
    floor, ceil = math.inf, -math.inf
    for (c, o, lr, tag), r in sorted(mine.items(), key=lambda kv: (kv[0][2], kv[0][3])):
        if r["valid"]:
            it = np.array(sorted(r["valid"]))
            v = np.array([r["valid"][i] for i in it])
            x = it * r["tok_per_iter"] / 1e9
            main = tag == "main"
            ax.plot(x, v, ls="-" if main else (0, (4, 2)), color=color[lr], lw=1.6 if main else 1.3,
                    label=f"{lr * 1e3:g}" if main else None, zorder=3)
            ax.scatter(x[-1:], v[-1:], s=22, color=color[lr], edgecolors="white", linewidths=0.8, zorder=4)
            floor = min(floor, v.min()); ceil = max(ceil, np.percentile(v, 30))
        if r["train"]:
            ti = np.array(sorted(r["train"]))
            ax2.plot(ti * r["tok_per_iter"] / 1e9, [r["train"][i][0] * 1e3 for i in ti],
                     ls="-" if tag == "main" else (0, (4, 2)), color=color[lr], lw=1.3)
    for t, p in plan.items():                         # decay starts of every horizon
        for a in (ax, ax2):
            a.axvline(p["start"] * tok / 1e9, color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.text(p["start"] * tok / 1e9, 1.0, f" TPP{t} decay", transform=ax.get_xaxis_transform(),
                fontsize=9, color=MUTED, va="bottom", ha="left")
    ax.set_ylim(floor - 0.015, ceil + 0.05)
    ax.set_xlim(0, max(p["end"] for p in plan.values()) * tok / 1e9 * 1.02)
    ax.set_xlabel("training tokens (B)")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation curves   (solid: stable run,  dashed: decay branches,  dot: endpoint)",
                 color=INK, pad=16)
    leg = ax.legend(title=r"peak LR ($\times10^{-3}$)", ncol=len(all_lrs), loc="upper right", fontsize=9.5,
                    title_fontsize=9.5, handlelength=1.8)
    leg._legend_box.align = "right"
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("training tokens (B)")
    ax2.set_ylabel(r"learning rate ($\times10^{-3}$)")
    ax2.set_title("LR schedules (WSD, 1-sqrt decay)", color=INK, pad=16)

    # -- row 3: the models -----------------------------------------------------------------------
    axf = fig.add_axes([0.055, 0.018, 0.93, 0.118])
    axf.axis("off")
    axf.add_patch(matplotlib.patches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02",
                                                    transform=axf.transAxes, fc="#f7f7f5", ec=GRID, lw=0.8))
    l1 = [
        r"$\bf{Fit\ 1:\ per\text{-}TPP}$   ordinary least squares on the LRs within "
        + rf"$\times{window:g}$ of the best grid LR",
        r"$L(\eta) = L^* + k\,\left(\ln\eta - \ln\eta^*\right)^2$",
        rf"80% band of $\eta^*$: refit 4000 times with $\epsilon \sim \mathcal{{N}}(0, \sigma^2)$ added to "
        + rf"each loss,  $\sigma = {noise:g}$ (assumed)",
    ]
    if joint:
        l2 = [
            r"$\bf{Fit\ 2:\ joint\ across\ TPPs}$   nonlinear least squares, "
            + rf"{joint['n']} points, {len(joint['p'])} params, residual RMS $= {joint['rms'] * 1e3:.2f}\times10^{{-3}}$",
            r"$L(\eta, T) = L^*_T + k_0\,(T/50)^{\kappa}\,\left[\ln\eta - \ln\eta^*_{50} - \gamma\ln(T/50)\right]^2$",
            rf"$\eta^*_{{50}} = {sci(math.exp(joint['x0']))}$,  "
            + rf"$\gamma = {joint['gamma']:+.2f} \pm {joint['gamma_se']:.2f}$,  "
            + rf"$k_0 = {joint['k0']:.4f}$,  $\kappa = {joint['kappa']:+.2f} \pm {joint['kappa_se']:.2f}$",
        ]
    else:
        l2 = [r"$\bf{Fit\ 2:\ joint\ across\ TPPs}$   needs at least 2 TPPs with enough points"]
    for col, lines in ((0.015, l1), (0.5, l2)):
        for i, s in enumerate(lines):
            axf.text(col, 0.80 - i * 0.30, s, transform=axf.transAxes, fontsize=13 if i == 1 else 10.5,
                     color=INK, va="center", ha="left")

    fig.suptitle(f"{config} + {opt}: learning-rate sweep", fontsize=16, color=INK, y=0.988)
    fig.savefig(path, dpi=150)
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
                    help="folder for lr_fit_<config>_<opt>.png and lr_points_<config>_<opt>.csv; created if missing")
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

    rng = np.random.default_rng(0)
    for config, opt in sorted({(r["config"], r["opt"]) for r in rows}):
        rs = [r for r in rows if r["config"] == config and r["opt"] == opt]
        with open(os.path.join(args.out_dir, f"lr_points_{config}_{opt}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(sorted(rs, key=lambda r: (r["tpp"], r["lr"])))
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
            plot(config, opt, runs, pts_by_t, fits, joint, path, args.window, args.noise)
            print(f"   plot -> {path}")


if __name__ == "__main__":
    main()
