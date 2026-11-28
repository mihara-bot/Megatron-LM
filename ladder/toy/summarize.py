"""Collect the margin toy's probe.json files into tables (ladder/toy/TOY.md). Standard library only.

    python3 ladder/toy/summarize.py lr    --root $TOY_ROOT     # plain-head LR sweep: pick the LR
    python3 ladder/toy/summarize.py table --root $TOY_ROOT     # variants x seeds, pretraining and SFT

Finds $TOY_ROOT/out/runs/<run>/hf/main_iter<N>/probe.json and $TOY_ROOT/out/sft/<run>_main_sft_toy_<mode>[_B...]/
hf/iter_<N>/probe.json (sft_protocol: A, or B_[rev_]d<D> for the margin kept in SFT). `table` uses each run's last probed iteration, averages over seeds (mean and sd), prints the
key columns and writes every row and the grouped means to $TOY_ROOT/report/{pretrain,sft}_{rows,summary}.csv.
Also in report/:
  pretrain_iters.csv, sft_iters.csv   every probed checkpoint (pretraining every 500 iterations, SFT every epoch)
  pretrain_decode{,_summary}.csv      the last checkpoint decoded at temperature T and through the post-hoc margin
                                      map (decode = T | map_m): multi-position KL, H(p), junk, ... per value
  sft_decode{,_summary}.csv           the last SFT checkpoint sampled at every decoding setting of its probe.json:
                                      decode = T<temperature> | M<logit margin of the post-hoc margin map>
                                      (ladder/toy/run_toy.sh PHASE=decode adds settings), with the failure rates
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys

RUN_RE = re.compile(r"^(?P<config>.+?)_(?P<opt>adamw|muon)_lr(?P<lr>[0-9.eE+-]+?)(?P<tag>(_(?:nfn|hfn|sfn).*)?)"
                    r"_seed(?P<seed>\d+)$")
SFT_RE = re.compile(r"^(?P<run>.+)_(?P<tag>main|tpp\d+)_sft_toy_(?P<mode>[a-z]+)"
                    r"(?P<protocol>_B(?:_rev)?_d[0-9.eE+-]+)?$")            # ladder/sft/common.sh protocol B
ITER_RE = re.compile(r"(?:main_)?iter_?(\d+)$")

PRE_COLS = [
    ("kl", "walk multi KL(q||p)"), ("kl_theory", "KL(p*||p)"), ("kl_q_theory", "KL(q||p*)"),
    ("h_p", "H(p) multi"), ("h_q", "H(q) multi"), ("h_theory", "H(p*)"),
    ("top1_p", "top-1 p"), ("top1_theory", "top-1 p*"), ("junk", "junk mass"), ("supp_kl", "KL on supp"),
    ("m", "logit margin m"), ("hnorm", "||h||"), ("det_top1", "det top-1"), ("det_gap", "det cos gap p50"),
    ("supp_gap", "multi supp gap p50"), ("cos_sibling", "cos sibling rows"), ("s_fit", "s fit"),
    ("kl_all", "KL all positions"),
]
SFT_COLS = [
    ("heldout_pass@1", "heldout pass@1"), ("heldout_pass@8", "pass@8"), ("heldout_pass@64", "pass@64"),
    ("heldout_distinct", "distinct valid"), ("heldout_cover_count", "coverage"), ("heldout_cover_mass", "cover mass"),
    ("heldout_path_entropy", "path entropy"), ("train_pass@1", "train pass@1"), ("train_cover_count", "train coverage"),
    ("tf_hop_kl", "tf hop KL"), ("tf_hop_h_p", "tf hop H(p)"), ("tf_hop_junk", "tf hop junk"),
]
DECODE_COLS = [(k, k) for k in ("kl", "h_p", "junk", "supp_kl", "top1_p", "det_top1", "det_ce")]
FAILS = ("bad_token", "empty", "bad_start", "bad_edge", "wrong_end", "no_eos")   # graph.py check_generation
SFT_DECODE_COLS = SFT_COLS + [(f"{split}_{k}", f"{split}_{k}") for split in ("heldout", "train")
                              for k in ("valid_rate", "pass@16", *(f"fail_{f}" for f in FAILS))]


def variant_of(tag: str) -> str:
    return tag.lstrip("_") or "plain"


def walk_row(res: dict) -> dict:
    w = res.get("walk")
    if not w:
        return {}
    cls, quant = w["classes"], w.get("quantiles", {})
    multi = cls["multi"]
    det = [cls[c] for c in ("det", "end") if cls[c]["n"]]
    n_all = sum(c["n"] for c in cls.values() if c["n"])
    # The excess CE over the floor is the mean KL(q || p) over the probe positions, i.e. CE minus the mean H(q) of
    # those same positions. Not CE minus the stream's entropy rate: a probe row drops its last context (an <eos>,
    # H = 0), which puts the positions' mean H(q) ~rate/249 = 0.0035 above the rate, more than the excess itself.
    kl_all = sum(c["kl"] * c["n"] for c in cls.values() if c["n"]) / n_all
    row = {k: multi.get(k) for k in ("kl", "kl_theory", "kl_q_theory", "h_p", "h_q", "h_theory", "top1_p",
                                      "top1_theory", "junk", "supp_kl", "m", "hnorm")}
    row["det_top1"] = sum(c["top1_p"] * c["n"] for c in det) / sum(c["n"] for c in det) if det else None
    row["det_gap"] = quant.get("det_gap", {}).get("p50")
    row["supp_gap"] = quant.get("multi_supp_gap", {}).get("p50")
    row["cos_sibling"] = w.get("head", {}).get("cos_sibling_mean")
    row["s_fit"] = w.get("s_fit", {}).get("s")
    row["kl_all"] = kl_all
    return row


def path_row(res: dict, temp: str = "T1") -> dict:
    pth = res.get("paths")
    if not pth:
        return {}
    row = {"mode": pth.get("mode")}
    for split, rec in pth.get("sample", {}).get(temp, {}).items():
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                row[f"{split}_{k}"] = v
        for f in FAILS:                                    # the probe lists only the failures that occurred
            row[f"{split}_fail_{f}"] = rec.get("fail", {}).get(f, 0.0)
    hop = pth.get("tf", {}).get("heldout", {}).get("classes", {}).get("hop_multi", {})
    row.update({"tf_hop_kl": hop.get("kl"), "tf_hop_h_p": hop.get("h_p"), "tf_hop_junk": hop.get("junk")})
    return row


def sft_protocol(tag) -> str:
    """A (SFT without margin) or B_[rev_]d<D> (the margin kept in the SFT loss) from the SFT run-name suffix."""
    return tag.lstrip("_") if tag else "A"


def decode_rows(res: dict) -> list:
    """The probe's decoding sweeps of a model's walk distribution: temperature, and the post-hoc margin map."""
    w = res.get("walk") or {}
    out = []
    for kind, key in (("T", "temperature"), ("map_m", "posthoc_margin")):
        for e in w.get(key, []):
            multi, det = e.get("multi") or {}, e.get("det") or {}
            out.append({"decode": kind, "value": e["value"],
                        **{k: multi.get(k) for k in ("kl", "h_p", "junk", "supp_kl", "top1_p")},
                        "det_top1": det.get("top1_p"), "det_ce": det.get("ce")})
    return out


def collect(root: str):
    out = os.path.join(root, "out")
    pre, sft = [], []
    for path in sorted(glob.glob(os.path.join(out, "runs", "*", "hf", "*", "probe.json"))):
        run = path.split(os.sep)[-4]
        m, it = RUN_RE.match(run), ITER_RE.search(path.split(os.sep)[-2])
        if not m or not it:
            continue
        res = json.load(open(path))
        pre.append({"run": run, "variant": variant_of(m["tag"]), "lr": m["lr"], "seed": int(m["seed"]),
                    "iteration": int(it.group(1)), **walk_row(res), "probe": path})
    for path in sorted(glob.glob(os.path.join(out, "sft", "*", "hf", "*", "probe.json"))):
        name = path.split(os.sep)[-4]
        s, it = SFT_RE.match(name), ITER_RE.search(path.split(os.sep)[-2])
        m = RUN_RE.match(s["run"]) if s else None
        if not m or not it:
            continue
        res = json.load(open(path))
        sft.append({"run": s["run"], "variant": variant_of(m["tag"]), "lr": m["lr"], "seed": int(m["seed"]),
                    "sft_mode": s["mode"], "sft_protocol": sft_protocol(s["protocol"]), "iteration": int(it.group(1)),
                    **{f"walk_{k}": v for k, v in walk_row(res).items()}, **path_row(res), "probe": path})
    return pre, sft


def last_iteration(rows, key=("run",)):
    best = {}
    for r in rows:
        k = tuple(r[x] for x in key)
        if k not in best or r["iteration"] > best[k]["iteration"]:
            best[k] = r
    return list(best.values())


def mean_sd(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return None, None
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
    return mu, sd


def group(rows, keys, cols):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for k, rs in groups.items():
        g = dict(zip(keys, k))
        g["seeds"] = len(rs)
        for c, _ in cols:
            g[c], g[c + "_sd"] = mean_sd([r.get(c) for r in rs])
        out.append(g)
    return out


def variant_order(v: str):
    """plain, then hfn / sfn / nfn; inside a head forward by d, then reverse by d."""
    if v == "plain":
        return (0, 0, 0.0)
    head = v.split("_")[0]
    d = re.search(r"_d([0-9.]+)", v)
    return (1 + ["hfn", "sfn", "nfn"].index(head) if head in ("hfn", "sfn", "nfn") else 9,
            1 if "_rev" in v else 0, float(d.group(1)) if d else 0.0)


def fmt(v, sd=None):
    if v is None:
        return "-"
    s = f"{v:.4g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
    return s + (f"±{sd:.2g}" if sd else "")


def print_table(rows, keys, cols, file=sys.stdout):
    head = keys + ["seeds"] + [label for _, label in cols]
    body = [[str(r[k]) for k in keys] + [str(r["seeds"])] + [fmt(r[c], r.get(c + "_sd")) for c, _ in cols]
            for r in rows]
    width = [max(len(x) for x in col) for col in zip(head, *body)]
    for line in [head] + body:
        print("  ".join(x.ljust(w) for x, w in zip(line, width)), file=file)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cmd_lr(args):
    pre, _ = collect(args.root)
    rows = sorted((r for r in last_iteration(pre) if r["variant"] == "plain" and r.get("kl") is not None),
                  key=lambda r: float(r["lr"]))
    if not rows:
        raise SystemExit(f"no plain-head probe.json under {args.root}/out/runs")
    print("lr        seed  iter   walk multi KL   KL all positions    det top-1")
    for r in rows:
        print(f"{r['lr']:<9} {r['seed']:<5} {r['iteration']:<6} {r['kl']:<15.5f} {fmt(r['kl_all']):<19} {fmt(r['det_top1'])}")
    best = min(rows, key=lambda r: r["kl"])
    print(f"lowest walk KL: LR={best['lr']}  ->  PHASE=grid LR={best['lr']} bash ladder/toy/run_toy.sh")


def decode_order(key: str):
    """T<temperature> by temperature, then M<margin> by margin."""
    try:
        return (0 if key[0] == "T" else 1, float(key[1:]))
    except (ValueError, IndexError):
        return (2, 0.0)


def by_run(r):
    return (r["lr"], variant_order(r["variant"]), r.get("sft_mode", ""), r.get("sft_protocol", ""), r["seed"],
            r["iteration"])


def cmd_table(args):
    pre_all, sft_all = collect(args.root)
    lr = args.lr
    pre_all = sorted((r for r in pre_all if lr is None or r["lr"] == lr), key=by_run)
    sft_all = sorted((r for r in sft_all if lr is None or r["lr"] == lr), key=by_run)
    pre = last_iteration(pre_all)
    sft = last_iteration(sft_all, ("run", "sft_mode", "sft_protocol"))
    report = os.path.join(args.root, "report")
    os.makedirs(report, exist_ok=True)
    write_csv(os.path.join(report, "pretrain_iters.csv"), pre_all)
    write_csv(os.path.join(report, "sft_iters.csv"), sft_all)
    keys = ("run", "variant", "lr", "seed", "iteration")
    dec = [{**{k: r[k] for k in keys}, **d} for r in pre for d in decode_rows(json.load(open(r["probe"])))]
    write_csv(os.path.join(report, "pretrain_decode.csv"), dec)
    write_csv(os.path.join(report, "pretrain_decode_summary.csv"),
              sorted(group(dec, ["variant", "lr", "decode", "value"], DECODE_COLS),
                     key=lambda r: (r["lr"], variant_order(r["variant"]), r["decode"], r["value"])))
    sdec = []
    for r in sft:
        res = json.load(open(r["probe"]))
        for d in (res.get("paths") or {}).get("sample", {}):
            sdec.append({**{k: r[k] for k in keys}, "sft_mode": r["sft_mode"], "sft_protocol": r["sft_protocol"],
                         "decode": d, **path_row(res, d)})
    write_csv(os.path.join(report, "sft_decode.csv"), sdec)
    write_csv(os.path.join(report, "sft_decode_summary.csv"),
              sorted(group(sdec, ["sft_mode", "sft_protocol", "variant", "lr", "decode"], SFT_DECODE_COLS),
                     key=lambda r: (r["sft_mode"], r["lr"], r["sft_protocol"], variant_order(r["variant"]),
                                    decode_order(r["decode"]))))
    g_pre = sorted(group(pre, ["variant", "lr"], PRE_COLS), key=lambda r: (r["lr"], variant_order(r["variant"])))
    print(f"== pretraining, last probed checkpoint ({len(pre)} runs)")
    print_table(g_pre, ["variant", "lr"], PRE_COLS)
    write_csv(os.path.join(report, "pretrain_rows.csv"), pre)
    write_csv(os.path.join(report, "pretrain_summary.csv"), g_pre)
    if sft:
        g_sft = sorted(group(sft, ["sft_mode", "sft_protocol", "variant", "lr"], SFT_COLS),
                       key=lambda r: (r["sft_mode"], r["lr"], r["sft_protocol"], variant_order(r["variant"])))
        print(f"\n== SFT, last checkpoint, sampling at T=1 ({len(sft)} runs)")
        print_table(g_sft, ["sft_mode", "sft_protocol", "variant", "lr"], SFT_COLS)
        write_csv(os.path.join(report, "sft_rows.csv"), sft)
        write_csv(os.path.join(report, "sft_summary.csv"), g_sft)
    print(f"\nCSV files in {report}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("lr", "table"):
        s = sub.add_parser(name)
        s.add_argument("--root", default=os.environ.get("TOY_ROOT"), help="TOY_ROOT")
        if name == "table":
            s.add_argument("--lr", help="only runs with this LR (as in the run name)")
    args = p.parse_args(argv)
    if not args.root:
        raise SystemExit("set --root or TOY_ROOT")
    {"lr": cmd_lr, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
