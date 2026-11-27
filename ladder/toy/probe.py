"""Probe exported margin-toy models against the exact distributions of the synthetic data (ladder/toy/TOY.md).

    python ladder/toy/probe.py --data $TOY_ROOT/data --models <HF dir> [<HF dir> ...]
    python ladder/toy/probe.py --data $TOY_ROOT/data --paths --sft-mode cond --models <SFT HF dirs>

Reads HF directories written by ladder/export/export_hf.sh (export_meta.json tells the LM head and margin) and
writes <model>/probe.json. Every model sees the same probe data (fixed seed), so runs pair position by position.

walk   Fresh walk documents, 25 per sequence as in training. Every position has an exact next-token q (graph.py).
       Position classes by the previous token: start (<walk>: uniform over layer 0), multi (a node with >= 2
       successors), det (a node with one successor), end (last-layer node: <eos>), doc (<eos>: <walk>).
       Per class, the means of
         ce, kl = KL(q || p), h_p, h_q, top1_p, top1_q, acc (argmax p is a most likely token of q),
         junk = p mass outside supp(q), supp_kl = KL(q || p renormalized on supp(q)),
         m (the run's logit margin at that position: s d for hfn, d ||h|| for nfn / sfn, negative under reverse),
         p* = the margin loss's population optimum for q and m (margin_theory.py): kl_theory = KL(p* || p),
           top1_theory, h_theory, kl_q_theory = KL(q || p*) (how far the theory moves away from q),
         geometry with cos_k = cos(h, W_k) at the LM head: hnorm = ||h||, gap = cos_y - max_{k != y} cos_k
           (y = argmax q; the SphereFace margin on one-label positions), supp_gap = min_{supp} cos - max_{not supp}
           cos, supp_spread = max_{supp} cos - min_{supp} cos, logit_max, logit_std.
       Also: multi positions by H(q) bucket; per node (mean p after that node vs q and p*); a temperature sweep and
       the post-hoc margin map p -> p*(p, m) of the model's own distribution (multi and det positions), i.e. what
       decoding alone does to this model; head geometry (row norms, cosine of sibling rows vs same-layer rows);
       s_fit, the s of logits s cos(h, W) with the lowest CE (as ladder lm_head_scale tools pick s).
paths  (--paths) The path task on training and held-out (s, t) pairs.
       tf: teacher forcing on paths drawn from the SFT mode's distribution: q = the mode's next hop, valid set =
         successors that still reach t (junk = mass outside it); classes copy (first node = s), hop_multi /
         hop_det (2+ / 1 valid next hops), stop (<eos> after t). For mode != cond also KL to the cond q.
       sample: --samples completions per pair at each of --temps: pass@k (unbiased), valid rate, distinct valid
         paths, coverage (distinct / all valid paths; cond probability mass of the distinct ones), entropy of the
         valid paths drawn, failure types (graph.py check_generation), by span.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graph as G  # noqa: E402

PROBE_VERSION = 1
CLASSES = ("start", "multi", "det", "end", "doc")
PATH_CLASSES = ("copy", "hop_multi", "hop_det", "stop")
H_EDGES = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 10.0)
TEMPS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)
MAP_MS = (-3.0, -1.2, -0.3, 0.3, 1.2, 3.0)
TINY = 1e-300


# ---- margin theory, batched (margin_theory.margin_optimum row by row) -------------------------------------
def margin_optimum(q: torch.Tensor, m: torch.Tensor, iters: int = 80) -> torch.Tensor:
    """p with p_k (1 - c p_k) proportional to q_k, c = 1 - e^{-m}: the population optimum of the margin CE.
    q [N, V] rows summing to 1, m [N] logit margins. Returns float64."""
    q = q.double()
    m = m.double().reshape(-1, 1)
    c = -torch.expm1(-m)
    p = q.clone()
    rev = (c < -1e-12).squeeze(1)
    fwd = (c > 1e-12).squeeze(1)
    if rev.any():
        qq, a = q[rev], -c[rev]

        def f(lam):
            return 2 * lam * qq / (1 + torch.sqrt(1 + 4 * a * lam * qq))

        lo, hi = torch.ones_like(a), (1 + a) / qq.max(1, keepdim=True).values
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            right = f(mid).sum(1, keepdim=True) < 1
            lo, hi = torch.where(right, mid, lo), torch.where(right, hi, mid)
        p[rev] = f(0.5 * (lo + hi))
    if fwd.any():
        qq, cc = q[fwd], c[fwd]
        qmax, top = qq.max(1, keepdim=True)
        lam_max = 1 / (4 * cc * qmax)

        def small(lam):
            return 2 * lam * qq / (1 + torch.sqrt(torch.clamp(1 - 4 * cc * lam * qq, min=0)))

        def mixed(lam):
            large = (1 + torch.sqrt(torch.clamp(1 - 4 * cc * lam * qmax, min=0))) / (2 * cc)
            return small(lam).scatter(1, top, large)

        all_small = small(lam_max).sum(1, keepdim=True) >= 1
        lo, hi = torch.zeros_like(cc), lam_max.clone()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            tot = torch.where(all_small, small(mid), mixed(mid)).sum(1, keepdim=True)
            right = torch.where(all_small, tot < 1, tot > 1)
            lo, hi = torch.where(right, mid, lo), torch.where(right, hi, mid)
        lam = 0.5 * (lo + hi)
        p[fwd] = torch.where(all_small, small(lam), mixed(lam))
    return p / p.sum(1, keepdim=True)


# ---- per-position metrics ------------------------------------------------------------------------------
def xlogy(x, y):
    return torch.where(x > 0, x * torch.log(y.clamp_min(TINY)), torch.zeros_like(x))


def dist_metrics(logp: torch.Tensor, q: torch.Tensor, supp: torch.Tensor = None) -> dict:
    """Per-position metrics of model log-probs logp against a reference q (float64 [N, V]); supp = the valid set
    (default supp(q)), which junk and supp_kl refer to."""
    logp, q = logp.double(), q.double()
    p = logp.exp()
    if supp is None:
        supp = q > 0
    h_q = -xlogy(q, q).sum(1)
    ce = -(torch.where(q > 0, q * logp, torch.zeros_like(q))).sum(1)
    mass = (p * supp).sum(1)
    log_ps = logp - torch.log(mass.clamp_min(TINY)).unsqueeze(1)
    top1_q = q.max(1).values
    return {
        "ce": ce, "kl": ce - h_q, "h_p": -(p * logp).sum(1), "h_q": h_q,
        "junk": 1 - mass, "supp_kl": (xlogy(q, q) - torch.where(q > 0, q * log_ps, torch.zeros_like(q))).sum(1),
        "top1_p": p.max(1).values, "top1_q": top1_q,
        "acc": (q.gather(1, p.argmax(1, keepdim=True)).squeeze(1) >= top1_q - 1e-12).double(),
    }


def geometry(h: torch.Tensor, W: torch.Tensor, logits: torch.Tensor, q: torch.Tensor, supp: torch.Tensor) -> dict:
    big = 4.0
    cos = F.normalize(h.double(), dim=-1) @ F.normalize(W.double(), dim=-1).T
    y = q.argmax(1, keepdim=True)
    cos_y = cos.gather(1, y).squeeze(1)
    s_min = torch.where(supp, cos, torch.full_like(cos, big)).min(1).values
    s_max = torch.where(supp, cos, torch.full_like(cos, -big)).max(1).values
    ns_max = torch.where(supp, torch.full_like(cos, -big), cos).max(1).values
    return {"hnorm": h.double().norm(dim=-1), "gap": cos_y - cos.scatter(1, y, -big).max(1).values,
            "supp_gap": s_min - ns_max, "supp_spread": s_max - s_min,
            "logit_max": logits.double().max(1).values, "logit_std": logits.double().std(1)}


def head_margin(head: dict, hnorm: torch.Tensor) -> torch.Tensor:
    """Logit margin of the run at each position (0 for the plain head or d = 0)."""
    fn, d = head.get("lm_head_feature_norm"), float(head.get("token_margin") or 0)
    if not fn or d == 0:
        m = torch.zeros_like(hnorm, dtype=torch.float64)
    elif fn == "hfn":
        m = torch.full_like(hnorm, float(head["lm_head_feature_norm_scale"]) * d, dtype=torch.float64)
    else:
        m = d * hnorm.double()
    return -m if head.get("token_margin_reverse") else m


def mean_of(metrics: dict, mask: torch.Tensor) -> dict:
    n = int(mask.sum())
    out = {"n": n}
    for k, v in metrics.items():
        out[k] = float(v[mask].mean()) if n else None
    return out


def quantiles(x: torch.Tensor, mask: torch.Tensor, qs=(0.1, 0.5, 0.9)) -> dict:
    v = x[mask].double()
    if v.numel() == 0:
        return {}
    return {f"p{int(100 * a)}": float(torch.quantile(v, a)) for a in qs}


# ---- model ----------------------------------------------------------------------------------------------
def load_model(path: str, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(device).eval()
    meta_path = os.path.join(path, "export_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    head = {k: v for k, v in meta.get("lm_head", {}).items() if k not in ("fold", "note")}
    return model, meta, head


@torch.no_grad()
def forward(model, ids: torch.Tensor, chunk: int = 64):
    """Final hidden state (the LM head's input) and logits, float32, in chunks of sequences."""
    hs, ls = [], []
    for i in range(0, ids.shape[0], chunk):
        h = model.model(input_ids=ids[i:i + chunk]).last_hidden_state.float()
        hs.append(h)
        ls.append(model.lm_head(h).float())
    return torch.cat(hs), torch.cat(ls)


# ---- walk section ---------------------------------------------------------------------------------------
def walk_tables(graph: G.Graph, device):
    """Q[prev token] = exact next-token distribution, CLS[prev token] = position class (-1: never a context)."""
    V = G.VOCAB_SIZE
    Q = torch.zeros(V, V, dtype=torch.float64)
    cls = torch.full((V,), -1, dtype=torch.long)
    for tok in range(V):
        dist = graph.walk_next(tok)
        if dist is None:
            continue
        for k, p in dist.items():
            Q[tok, k] = p
        u = G.token_node(tok)
        if tok == G.WALK:
            cls[tok] = CLASSES.index("start")
        elif tok == G.EOS:
            cls[tok] = CLASSES.index("doc")
        elif graph.is_last(u):
            cls[tok] = CLASSES.index("end")
        else:
            cls[tok] = CLASSES.index("multi" if graph.out_degree(u) > 1 else "det")
    return Q.to(device), cls.to(device)


def walk_ids(graph: G.Graph, n_docs: int, seq_len: int, seed: int) -> torch.Tensor:
    doc_len = graph.num_layers + 2
    per_seq = max(1, seq_len // doc_len)
    n_seq = max(1, n_docs // per_seq)
    rng = random.Random(f"toy-probe-walks:{seed}")
    rows = []
    for _ in range(n_seq):
        row = []
        for _ in range(per_seq):
            row += G.walk_ids(graph.sample_walk(rng))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def sweep(logp: torch.Tensor, q: torch.Tensor, groups: dict, transform, values) -> list:
    out = []
    for v in values:
        lp = transform(logp, v)
        met = dist_metrics(lp, q)
        out.append({"value": v, **{g: mean_of(met, mask) for g, mask in groups.items()}})
    return out


def s_fit(h: torch.Tensor, W: torch.Tensor, q: torch.Tensor) -> dict:
    """argmin_s mean CE(q, softmax(s cos(h, W))) by golden-section search on log s in [0, log 400]."""
    cos = F.normalize(h.double(), dim=-1) @ F.normalize(W.double(), dim=-1).T

    def ce(log_s):
        return float(-(torch.where(q > 0, q * torch.log_softmax(math.exp(log_s) * cos, 1), torch.zeros_like(q)))
                     .sum(1).mean())

    a, b = 0.0, math.log(400.0)
    g = (math.sqrt(5) - 1) / 2
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = ce(x1), ce(x2)
    for _ in range(60):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = ce(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = ce(x2)
    x = 0.5 * (a + b)
    return {"s": math.exp(x), "ce": ce(x)}


def head_geometry(graph: G.Graph, W: torch.Tensor) -> dict:
    rows = W.double()[G.NODE0:]
    norms = rows.norm(dim=1)
    cos = F.normalize(rows, dim=1) @ F.normalize(rows, dim=1).T

    def mean_pairs(nodes):
        vals = [float(cos[a, b]) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
        return vals

    sib, same = [], []
    for u in graph.nodes:
        ws = [w for w, _ in graph.succ[u]]
        if len(ws) > 1:
            sib += mean_pairs(ws)
    for layer in graph.layers[1:]:
        same += mean_pairs(layer)
    off = cos[~torch.eye(len(rows), dtype=torch.bool, device=cos.device)]
    return {"row_norm_mean": float(norms.mean()), "row_norm_cv": float(norms.std() / norms.mean()),
            "special_row_norm_mean": float(W.double()[:G.NODE0].norm(dim=1).mean()),
            "cos_all_mean": float(off.mean()), "cos_sibling_mean": sum(sib) / len(sib),
            "cos_same_layer_mean": sum(same) / len(same)}


def probe_walk(model, head, graph: G.Graph, args, device) -> dict:
    ids = walk_ids(graph, args.walk_docs, args.seq_len, args.seed).to(device)
    h, logits = forward(model, ids)
    prev, h, logits = ids[:, :-1].reshape(-1), h[:, :-1].reshape(-1, h.shape[-1]), logits[:, :-1].reshape(-1, logits.shape[-1])
    Q, CLS = walk_tables(graph, device)
    keep = CLS[prev] >= 0
    prev, h, logits = prev[keep], h[keep], logits[keep]
    q, cls = Q[prev], CLS[prev]
    supp = q > 0
    logp = torch.log_softmax(logits.double(), dim=1)
    W = model.lm_head.weight.detach()

    met = dist_metrics(logp, q)
    geo = geometry(h, W, logits, q, supp)
    m = head_margin(head, geo["hnorm"])
    p_star = margin_optimum(q, m)
    log_star = torch.log(p_star.clamp_min(TINY))
    met.update(geo)
    met["m"] = m
    met["kl_theory"] = (xlogy(p_star, p_star) - torch.where(p_star > 0, p_star * logp, torch.zeros_like(p_star))).sum(1)
    met["kl_q_theory"] = (xlogy(q, q) - torch.where(q > 0, q * log_star, torch.zeros_like(q))).sum(1)
    met["top1_theory"] = p_star.max(1).values
    met["h_theory"] = -xlogy(p_star, p_star).sum(1)

    masks = {c: cls == i for i, c in enumerate(CLASSES)}
    out = {"positions": int(cls.numel()), "classes": {c: mean_of(met, mk) for c, mk in masks.items()}}
    out["quantiles"] = {
        "det_gap": quantiles(met["gap"], masks["det"] | masks["end"]),
        "multi_supp_gap": quantiles(met["supp_gap"], masks["multi"]),
        "multi_supp_spread": quantiles(met["supp_spread"], masks["multi"]),
        "hnorm": quantiles(met["hnorm"], torch.ones_like(masks["multi"])),
    }
    buckets = []
    for lo, hi in zip(H_EDGES, H_EDGES[1:]):
        mk = masks["multi"] & (met["h_q"] >= lo) & (met["h_q"] < hi)
        buckets.append({"h_lo": lo, "h_hi": hi, **mean_of(met, mk)})
    out["multi_by_hq"] = buckets

    p = logp.exp()
    nodes = []
    for tok in torch.unique(prev[masks["multi"]]).tolist():
        mk = (prev == tok)
        p_bar, q_row, star_bar = p[mk].mean(0), Q[tok], p_star[mk].mean(0)
        nodes.append({"node": G.token_node(tok), "n": int(mk.sum()), "out_degree": graph.out_degree(G.token_node(tok)),
                      "h_q": float(met["h_q"][mk][0]), "q_top": float(q_row.max()),
                      "p_top": float(p_bar[q_row.argmax()]), "theory_top": float(star_bar[q_row.argmax()]),
                      "q": [float(x) for x in q_row[q_row > 0]], "p": [float(x) for x in p_bar[q_row > 0]],
                      "theory": [float(x) for x in star_bar[q_row > 0]], "junk": float(1 - p_bar[q_row > 0].sum()),
                      "h_p": float(met["h_p"][mk].mean()), "kl": float(met["kl"][mk].mean()),
                      "kl_theory": float(met["kl_theory"][mk].mean())})
    out["multi_nodes"] = nodes

    groups = {"multi": masks["multi"], "det": masks["det"] | masks["end"]}
    out["temperature"] = sweep(logits.double(), q, groups, lambda z, t: torch.log_softmax(z / t, 1), TEMPS)
    out["posthoc_margin"] = sweep(
        logp, q, groups,
        lambda lp, mm: torch.log(margin_optimum(lp.exp(), torch.full((lp.shape[0],), mm, device=lp.device)).clamp_min(TINY)),
        MAP_MS)
    out["head"] = head_geometry(graph, W)
    out["s_fit"] = s_fit(h, W, q)
    return out


# ---- path section ---------------------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pick_pairs(pairs, n: int, seed: int, split: str):
    pairs = [tuple(x) for x in pairs]
    if len(pairs) <= n:
        return pairs
    return sorted(random.Random(f"toy-probe-pairs:{seed}:{split}").sample(pairs, n))


def probe_paths_tf(model, graph: G.Graph, pairs, mode: str, args, device) -> dict:
    rng = random.Random(f"toy-probe-tf:{args.seed}:{mode}")
    V = G.VOCAB_SIZE
    seqs, rows = [], []          # rows: (sequence index, position, class, q_mode, q_cond, valid)
    for s, t in pairs:
        for _ in range(args.tf_paths):
            path = graph.sample_path(s, t, mode, rng)
            ids = G.path_prompt_ids(s, t) + [G.node_token(v) for v in path] + [G.EOS]
            i = len(seqs)
            seqs.append(ids)
            base = len(G.path_prompt_ids(s, t)) - 1   # logits at the prompt's last token predict the first node
            rows.append((i, base, "copy", {G.node_token(s): 1.0}, {G.node_token(s): 1.0}, [G.node_token(s)]))
            for j, u in enumerate(path):
                if u == t:
                    rows.append((i, base + j + 1, "stop", {G.EOS: 1.0}, {G.EOS: 1.0}, [G.EOS]))
                    break
                valid = graph.valid_next(u, t)
                qm = {G.node_token(w): p for w, p in graph.next_hop(u, t, mode).items()}
                qc = {G.node_token(w): p for w, p in graph.next_hop(u, t, "cond").items()}
                rows.append((i, base + j + 1, "hop_multi" if len(valid) > 1 else "hop_det", qm, qc,
                             [G.node_token(w) for w in valid]))
    L = max(len(x) for x in seqs)
    ids = torch.tensor([x + [G.EOS] * (L - len(x)) for x in seqs], dtype=torch.long, device=device)
    h, logits = forward(model, ids, chunk=1024)
    si = torch.tensor([r[0] for r in rows], device=device)
    pi = torch.tensor([r[1] for r in rows], device=device)
    h, logits = h[si, pi], logits[si, pi]
    q = torch.zeros(len(rows), V, dtype=torch.float64, device=device)
    qc = torch.zeros_like(q)
    valid = torch.zeros(len(rows), V, dtype=torch.bool, device=device)
    for r, (_, _, _, qm_r, qc_r, val) in enumerate(rows):
        for k, p in qm_r.items():
            q[r, k] = p
        for k, p in qc_r.items():
            qc[r, k] = p
        valid[r, val] = True
    cls = torch.tensor([PATH_CLASSES.index(r[2]) for r in rows], device=device)
    logp = torch.log_softmax(logits.double(), 1)
    met = dist_metrics(logp, q, valid)
    met.update(geometry(h, model.lm_head.weight.detach(), logits, q, valid))
    if mode != "cond":
        met.update({f"cond_{k}": v for k, v in dist_metrics(logp, qc, valid).items() if k in ("ce", "kl", "h_q")})
    return {"positions": len(rows), "classes": {c: mean_of(met, cls == i) for i, c in enumerate(PATH_CLASSES)}}


@torch.no_grad()
def sample_paths(model, graph: G.Graph, pairs, n: int, temp: float, seed: int, device, batch: int) -> list:
    """n completions per pair; returns per-pair lists of generated token-id lists (after the prompt)."""
    max_new = graph.num_layers + 1
    prompts = [G.path_prompt_ids(s, t) for s, t in pairs for _ in range(n)]
    gen = torch.Generator(device=device).manual_seed(seed)
    outs = []
    for i in range(0, len(prompts), batch):
        ids = torch.tensor(prompts[i:i + batch], dtype=torch.long, device=device)
        start = ids.shape[1]
        done = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)
        for _ in range(max_new):
            logits = model(input_ids=ids).logits[:, -1, :].float()
            nxt = torch.multinomial(torch.softmax(logits / temp, -1), 1, generator=gen)
            ids = torch.cat([ids, nxt], 1)
            done |= nxt.squeeze(1) == G.EOS
            if bool(done.all()):
                break
        outs += ids[:, start:].tolist()
    return [outs[j * n:(j + 1) * n] for j in range(len(pairs))]


def grade(graph: G.Graph, pairs, gens, n: int) -> dict:
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= n]
    fails, per_pair = {}, []
    by_span = {}
    for (s, t), samples in zip(pairs, gens):
        found, counts = {}, 0
        for g in samples:
            status, path = graph.check_generation(s, t, g)
            fails[status] = fails.get(status, 0) + 1
            if status == "ok":
                counts += 1
                found[path] = found.get(path, 0) + 1
        total = graph.path_count(t)[s]
        ent = -sum(c / counts * math.log(c / counts) for c in found.values()) if counts else 0.0
        rec = {"c": counts, "distinct": len(found), "total_paths": total, "cover_count": len(found) / total,
               "cover_mass": sum(graph.path_prob(p, t, "cond") for p in found), "path_entropy": ent,
               "pass": {k: pass_at_k(n, counts, k) for k in ks}}
        per_pair.append(rec)
        span = graph.layer_of[t] - graph.layer_of[s]
        by_span.setdefault(span, []).append(rec)

    def agg(recs):
        return {"pairs": len(recs), "valid_rate": sum(r["c"] for r in recs) / (n * len(recs)),
                **{f"pass@{k}": sum(r["pass"][k] for r in recs) / len(recs) for k in ks},
                **{k: sum(r[k] for r in recs) / len(recs)
                   for k in ("distinct", "total_paths", "cover_count", "cover_mass", "path_entropy")}}

    total = sum(fails.values())
    return {**agg(per_pair), "samples_per_pair": n, "fail": {k: v / total for k, v in sorted(fails.items())},
            "by_span": {str(sp): agg(recs) for sp, recs in sorted(by_span.items())}}


def probe_paths(model, graph: G.Graph, pair_split: dict, args, device) -> dict:
    out = {"mode": args.sft_mode, "tf": {}, "sample": {}}
    for split in ("train", "heldout"):
        pairs = pick_pairs(pair_split[split], args.pairs, args.seed, split)
        out["tf"][split] = probe_paths_tf(model, graph, pairs, args.sft_mode, args, device)
        for temp in args.temps:
            gens = sample_paths(model, graph, pairs, args.samples, temp, args.seed + 1000 * int(round(temp * 100)),
                                device, args.gen_batch)
            out["sample"].setdefault(f"T{temp:g}", {})[split] = grade(graph, pairs, gens, args.samples)
    return out


# ---- driver ---------------------------------------------------------------------------------------------
def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="ladder/toy/make_data.py output (graph.json)")
    p.add_argument("--models", nargs="+", required=True, help="HF directories (ladder/export/export_hf.sh)")
    p.add_argument("--out-name", default="probe.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0, help="probe data and sampling (same for every model)")
    p.add_argument("--seq-len", type=int, default=256, help="walk sequence length (the config's SEQ_LEN)")
    p.add_argument("--walk-docs", type=int, default=4000)
    p.add_argument("--no-walk", action="store_true")
    p.add_argument("--paths", action="store_true", help="also probe the path task (SFT models)")
    p.add_argument("--sft-mode", default="cond", choices=("cond", "uniform", "canon"),
                   help="the SFT distribution: teacher-forcing paths and q")
    p.add_argument("--pairs", type=int, default=256, help="pairs per split (train / heldout)")
    p.add_argument("--tf-paths", type=int, default=4, help="teacher-forced paths per pair")
    p.add_argument("--samples", type=int, default=64, help="sampled completions per pair")
    p.add_argument("--temps", default="0.7,1.0,1.3", help="sampling temperatures")
    p.add_argument("--gen-batch", type=int, default=16384)
    args = p.parse_args(argv)
    args.temps = [float(x) for x in str(args.temps).split(",") if x]

    with open(os.path.join(args.data, "graph.json")) as f:
        data = json.load(f)
    graph = G.Graph.from_json(data["graph"])
    torch.backends.cuda.matmul.allow_tf32 = False
    for path in args.models:
        t0 = time.time()
        model, meta, head = load_model(path, args.device)
        res = {"probe_version": PROBE_VERSION, "model": os.path.abspath(path), "iteration": meta.get("iteration"),
               "source_checkpoint": meta.get("source_checkpoint"), "head": head,
               "reference": data.get("reference", {}),
               "settings": {k: v for k, v in vars(args).items() if k not in ("models", "device")}}
        if not args.no_walk:
            res["walk"] = probe_walk(model, head, graph, args, args.device)
        if args.paths:
            res["paths"] = probe_paths(model, graph, data["pairs"], args, args.device)
        res["seconds"] = round(time.time() - t0, 1)
        tmp = os.path.join(path, args.out_name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(res, f, indent=1)
        os.replace(tmp, os.path.join(path, args.out_name))
        w = res.get("walk", {}).get("classes", {}).get("multi", {})
        msg = f"kl {w.get('kl', float('nan')):.4f} kl_theory {w.get('kl_theory', float('nan')):.4f} " \
              f"h_p/h_q {w.get('h_p', float('nan')):.3f}/{w.get('h_q', float('nan')):.3f}" if w else ""
        if args.paths:
            s = res["paths"]["sample"].get("T1", {}).get("heldout", {})
            msg += f" | heldout T1 pass@1 {s.get('pass@1', float('nan')):.3f} cover {s.get('cover_count', float('nan')):.3f}"
        print(f"probe {path}: {msg} ({res['seconds']}s)", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
