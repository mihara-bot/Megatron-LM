"""Population optimum of the token-margin loss (standard library; ladder/toy/probe.py has the batched torch version).

Margin CE of one target y with logit margin m (m = d s for hfn, d ||h|| for nfn / sfn, negative under reverse):
    L_y(z) = -log softmax(z - m e_y)_y = CE_y(z) + m + log(1 - c p_y),   c = 1 - e^{-m},  p = softmax(z).
Its gradient is r_y (p - e_y) with r_y = 1 / (1 - c p_y), so the expected loss under a target distribution q is
minimized where
    q_k  is proportional to  p_k (1 - c p_k)        (every k)
i.e. p_k (1 - c p_k) = lam q_k for one lam > 0. The loss is convex in z, so this p is the unique optimum:
  * m = 0: p = q (plain CE is a proper scoring rule).
  * m > 0 (forward, 0 < c < 1): sharper than q. At most one token can sit on the large root of p (1 - c p).
  * m < 0 (reverse, c < 0): flatter than q; as m -> -inf, p -> sqrt(q) / sum sqrt(q) on the support.
A one-hot q stays one-hot for any m: the margin then only changes how the logits get there (the geometry).
"""
from __future__ import annotations

import math
from typing import List, Sequence


def margin_loss(z: Sequence[float], y: int, m: float) -> float:
    """-log softmax(z - m e_y)_y, written out (no closed form)."""
    zz = list(z)
    zz[y] -= m
    mx = max(zz)
    return -(zz[y] - mx - math.log(sum(math.exp(v - mx) for v in zz)))


def margin_loss_closed(z: Sequence[float], y: int, m: float) -> float:
    """The closed form the ladder trains with: CE + m + log((1 - p_y) + p_y e^{-m})."""
    mx = max(z)
    lse = mx + math.log(sum(math.exp(v - mx) for v in z))
    p_y = math.exp(z[y] - lse)
    return (lse - z[y]) + m + math.log((1 - p_y) + p_y * math.exp(-m))


def _bisect(f, lo: float, hi: float, go_right, iters: int) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if go_right(f(mid)):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def margin_optimum(q: Sequence[float], m: float, iters: int = 200) -> List[float]:
    """The p minimizing E_{y ~ q} L_y for logit margin m (see the module docstring)."""
    q = [float(x) for x in q]
    tot = sum(q)
    if tot <= 0:
        raise ValueError("q must have positive mass")
    q = [x / tot for x in q]
    c = -math.expm1(-m)                       # 1 - e^{-m}, exact near 0
    if abs(c) < 1e-15:
        return q
    if c < 0:                                 # reverse: p_k = 2 lam q_k / (1 + sqrt(1 + 4 a lam q_k)), a = -c
        a = -c

        def p_of(lam):
            return [2 * lam * x / (1 + math.sqrt(1 + 4 * a * lam * x)) for x in q]

        # sum(p) <= lam, and p_top >= 1 once lam >= (1 + a) / q_max
        lam = _bisect(lambda l: sum(p_of(l)), 1.0, (1 + a) / max(q), lambda s: s < 1, iters)
        p = p_of(lam)
    else:                                     # forward: small roots, the top token possibly on the large one
        q_max = max(q)
        top = q.index(q_max)
        lam_max = 1 / (4 * c * q_max)

        def small(lam):
            return [2 * lam * x / (1 + math.sqrt(max(0.0, 1 - 4 * c * lam * x))) for x in q]

        if sum(small(lam_max)) >= 1:          # all small roots; their sum grows with lam
            lam = _bisect(lambda l: sum(small(l)), 0.0, lam_max, lambda s: s < 1, iters)
            p = small(lam)
        else:                                 # top on the large root: the sum falls from 1/c > 1 below 1

            def mixed(lam):
                p = small(lam)
                p[top] = (1 + math.sqrt(max(0.0, 1 - 4 * c * lam * q[top]))) / (2 * c)
                return p

            lam = _bisect(lambda l: sum(mixed(l)), 0.0, lam_max, lambda s: s > 1, iters)
            p = mixed(lam)
    s = sum(p)
    return [x / s for x in p]


def stationarity_residual(p: Sequence[float], q: Sequence[float], m: float) -> float:
    """max_k |p_k (1 - c p_k) / sum_j p_j (1 - c p_j) - q_k|: 0 at the optimum."""
    c = -math.expm1(-m)
    g = [x * (1 - c * x) for x in p]
    s = sum(g)
    return max(abs(gk / s - qk) for gk, qk in zip(g, q))


def expected_margin_loss(p: Sequence[float], q: Sequence[float], m: float) -> float:
    """E_{y ~ q} L_y at the logits log p."""
    z = [math.log(x) if x > 0 else -1e30 for x in p]
    return sum(qy * margin_loss_closed(z, y, m) for y, qy in enumerate(q) if qy > 0)
