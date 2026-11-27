"""The margin toy's synthetic language (ladder/toy/TOY.md): a layered random DAG over 92 node tokens.

Standard library only, so the tests run anywhere. Everything the probe needs is exact:

  walk documents   <walk> v0 v1 ... v7 <eos>, v0 uniform over layer 0, then v_{l+1} ~ P[v_l, .]. The next token
                   depends on the previous token only, so every position has a known distribution q: uniform over
                   layer 0 after <walk>, the row P[u] after a node u, <eos> after a last-layer node, <walk> after
                   <eos> (pretraining is walks only).
  path task (SFT)  prompt  <path> s t :   response  s v ... t <eos>: any path from s to t along the edges. Three
                   training distributions over the valid paths: cond (a walk from s conditioned on reaching t:
                   next hop w with P[u, w] R_t(w) / R_t(u)), uniform (every valid path equally likely) and canon (one
                   path per pair, the most probable one).

Token ids: 8 specials, then node i is token 8 + i. The vocabulary is exactly 100 tokens.
"""
from __future__ import annotations

import bisect
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---- vocabulary ------------------------------------------------------------------------------------
SPECIALS = ("<unk>", "<eos>", "<walk>", "<path>", ":", "<rsv0>", "<rsv1>", "<rsv2>")
UNK, EOS, WALK, PATH, SEP = 0, 1, 2, 3, 4
NODE0 = len(SPECIALS)
NUM_NODES = 92
VOCAB_SIZE = NODE0 + NUM_NODES  # 100

DEFAULT_LAYER_SIZES = (12, 12, 12, 12, 11, 11, 11, 11)
DEFAULT_ALPHAS = (0.3, 1.0, 3.0)   # Dirichlet concentration per node: skewed, flat-ish, near-uniform rows
DEFAULT_DET_FRAC = 0.2             # share of non-last nodes with a single successor (one-hot q)
DEFAULT_MIN_DEGREE = 2             # out-degree range of the other nodes
DEFAULT_MAX_DEGREE = 6
DEFAULT_WEIGHT_FLOOR = 0.02        # smallest edge probability: every edge is seen in training


def vocab() -> List[str]:
    return list(SPECIALS) + [f"n{i}" for i in range(NUM_NODES)]


def node_token(v: int) -> int:
    return NODE0 + v


def token_node(tok: int) -> Optional[int]:
    return tok - NODE0 if NODE0 <= tok < VOCAB_SIZE else None


# ---- text / ids of the two document kinds ---------------------------------------------------------
def walk_text(nodes: Sequence[int]) -> str:
    """A pretraining document; tokenize_dataset.py appends the <eos>."""
    return " ".join(["<walk>"] + [f"n{v}" for v in nodes])


def walk_ids(nodes: Sequence[int]) -> List[int]:
    return [WALK] + [node_token(v) for v in nodes] + [EOS]


def path_prompt_text(s: int, t: int) -> str:
    """User turn of an SFT example. The trailing space matters: the identity chat template concatenates the
    user and assistant turns, and the WordLevel tokenizer splits on whitespace only."""
    return f"<path> n{s} n{t} : "


def path_response_text(nodes: Sequence[int]) -> str:
    return " ".join(f"n{v}" for v in nodes)


def path_prompt_ids(s: int, t: int) -> List[int]:
    return [PATH, node_token(s), node_token(t), SEP]


def entropy(dist: Iterable[float]) -> float:
    return -sum(p * math.log(p) for p in dist if p > 0)


# ---- the graph -----------------------------------------------------------------------------------------
class Graph:
    """Layered DAG: edges only from layer l to layer l + 1; every non-last node has at least one successor."""

    def __init__(self, layers: Sequence[Sequence[int]], succ: Dict[int, Sequence[Tuple[int, float]]],
                 alpha: Optional[Dict[int, float]] = None, params: Optional[dict] = None):
        self.layers = [list(map(int, layer)) for layer in layers]
        self.succ = {int(u): sorted((int(w), float(p)) for w, p in s) for u, s in succ.items()}
        self.alpha = {int(u): float(a) for u, a in (alpha or {}).items()}
        self.params = dict(params or {})
        self.layer_of = {v: i for i, layer in enumerate(self.layers) for v in layer}
        self.num_layers = len(self.layers)
        self.nodes = sorted(self.layer_of)
        for v in self.nodes:
            self.succ.setdefault(v, [])
        self._succ_set = {u: {w for w, _ in s} for u, s in self.succ.items()}
        self._cum = {}
        for u, s in self.succ.items():
            acc, cum = 0.0, []
            for _, p in s:
                acc += p
                cum.append(acc)
            self._cum[u] = ([w for w, _ in s], cum)
        self._reach: Dict[int, Dict[int, float]] = {}
        self._count: Dict[int, Dict[int, int]] = {}
        self._canon: Dict[int, Dict[int, int]] = {}

    # -- structure
    def is_last(self, u: int) -> bool:
        return self.layer_of[u] == self.num_layers - 1

    def out_degree(self, u: int) -> int:
        return len(self.succ[u])

    def has_edge(self, u: int, w: int) -> bool:
        return w in self._succ_set.get(u, ())

    def prob(self, u: int, w: int) -> float:
        for x, p in self.succ[u]:
            if x == w:
                return p
        return 0.0

    # -- walks
    def step(self, u: int, rng: random.Random) -> int:
        ws, cum = self._cum[u]
        i = bisect.bisect_right(cum, rng.random() * cum[-1])
        return ws[min(i, len(ws) - 1)]

    def sample_walk(self, rng: random.Random) -> List[int]:
        u = self.layers[0][rng.randrange(len(self.layers[0]))]
        walk = [u]
        while not self.is_last(u):
            u = self.step(u, rng)
            walk.append(u)
        return walk

    def layer_marginals(self) -> Dict[int, float]:
        """P(a walk visits v): uniform over layer 0, pushed through P layer by layer."""
        mu = {v: 1.0 / len(self.layers[0]) for v in self.layers[0]}
        for layer in self.layers[:-1]:
            for u in layer:
                for w, p in self.succ[u]:
                    mu[w] = mu.get(w, 0.0) + mu[u] * p
        return {v: mu.get(v, 0.0) for v in self.nodes}

    def walk_next(self, prev: int) -> Optional[Dict[int, float]]:
        """Exact next-token distribution (token -> probability) in the pretraining stream, given the previous
        token; None for tokens that never precede a prediction there."""
        if prev == WALK:
            n = len(self.layers[0])
            return {node_token(v): 1.0 / n for v in self.layers[0]}
        if prev == EOS:
            return {WALK: 1.0}
        u = token_node(prev)
        if u is None:
            return None
        if self.is_last(u):
            return {EOS: 1.0}
        return {node_token(w): p for w, p in self.succ[u]}

    def entropy_rate(self) -> float:
        """Mean entropy (nats) per predicted token of the walk stream: the plain-CE floor of validation."""
        mu = self.layer_marginals()
        per_doc = math.log(len(self.layers[0]))   # after <walk>; after the last node and after <eos>: 0
        for layer in self.layers[:-1]:
            per_doc += sum(mu[u] * entropy(p for _, p in self.succ[u]) for u in layer)
        return per_doc / (self.num_layers + 2)    # <walk>, the nodes, <eos>: one prediction each

    # -- paths to a target t
    def reach_prob(self, t: int) -> Dict[int, float]:
        """R_t(x): probability that a walk from x passes through t (0 for x at or after t's layer, but t)."""
        if t not in self._reach:
            lt = self.layer_of[t]
            r = {v: 0.0 for v in self.nodes}
            r[t] = 1.0
            for layer in reversed(self.layers[:lt]):
                for u in layer:
                    r[u] = sum(p * r[w] for w, p in self.succ[u])
            self._reach[t] = r
        return self._reach[t]

    def path_count(self, t: int) -> Dict[int, int]:
        """N_t(x): number of paths from x to t."""
        if t not in self._count:
            lt = self.layer_of[t]
            n = {v: 0 for v in self.nodes}
            n[t] = 1
            for layer in reversed(self.layers[:lt]):
                for u in layer:
                    n[u] = sum(n[w] for w, _ in self.succ[u])
            self._count[t] = n
        return self._count[t]

    def canon_next(self, t: int) -> Dict[int, int]:
        """Next hop of the most probable path to t from every node that reaches t (ties: smallest node id)."""
        if t not in self._canon:
            lt = self.layer_of[t]
            best = {t: 0.0}
            nxt: Dict[int, int] = {}
            for layer in reversed(self.layers[:lt]):
                for u in layer:
                    cands = [(math.log(p) + best[w], -w) for w, p in self.succ[u] if w in best]
                    if cands:
                        score, neg_w = max(cands)
                        best[u], nxt[u] = score, -neg_w
            self._canon[t] = nxt
        return self._canon[t]

    def valid_next(self, u: int, t: int) -> List[int]:
        """Successors of u from which t is still reachable."""
        n = self.path_count(t)
        return [w for w, _ in self.succ[u] if n[w] > 0]

    def next_hop(self, u: int, t: int, mode: str) -> Dict[int, float]:
        """Next-hop distribution (node -> probability) of a path to t standing at u != t, under one of the SFT
        distributions: cond | uniform | canon."""
        if mode == "cond":
            r = self.reach_prob(t)
            if r[u] <= 0:
                raise ValueError(f"t={t} is not reachable from u={u}")
            return {w: p * r[w] / r[u] for w, p in self.succ[u] if r[w] > 0}
        if mode == "uniform":
            n = self.path_count(t)
            if n[u] <= 0:
                raise ValueError(f"t={t} is not reachable from u={u}")
            return {w: n[w] / n[u] for w, _ in self.succ[u] if n[w] > 0}
        if mode == "canon":
            nxt = self.canon_next(t)
            if u not in nxt:
                raise ValueError(f"t={t} is not reachable from u={u}")
            return {nxt[u]: 1.0}
        raise ValueError(f"unknown path mode {mode!r}")

    def sample_path(self, s: int, t: int, mode: str, rng: random.Random) -> List[int]:
        path = [s]
        u = s
        while u != t:
            dist = self.next_hop(u, t, mode)
            ws = list(dist)
            acc, r = 0.0, rng.random()
            w = ws[-1]
            for x in ws:
                acc += dist[x]
                if r < acc:
                    w = x
                    break
            path.append(w)
            u = w
        return path

    def path_prob(self, path: Sequence[int], t: int, mode: str) -> float:
        """Probability of a complete path under a mode (0 if it is not a valid path ending at t)."""
        if not path or path[-1] != t:
            return 0.0
        prob = 1.0
        for u, w in zip(path, path[1:]):
            if not self.has_edge(u, w) or self.path_count(t)[w] == 0:
                return 0.0
            prob *= self.next_hop(u, t, mode).get(w, 0.0)
        return prob

    def check_generation(self, s: int, t: int, gen: Sequence[int]) -> Tuple[str, Optional[Tuple[int, ...]]]:
        """Grade generated token ids (after the prompt) of the path task: ('ok', path) or (failure, None).
        Failures, first match wins: bad_token (a non-node before <eos>), empty, bad_start, bad_edge,
        wrong_end (valid edges but not ending at t), no_eos (the right path but no <eos> within the budget)."""
        gen = list(gen)
        finished = EOS in gen
        body = gen[: gen.index(EOS)] if finished else gen
        nodes = [token_node(x) for x in body]
        if any(v is None for v in nodes):
            return "bad_token", None
        if not nodes:
            return "empty", None
        if nodes[0] != s:
            return "bad_start", None
        if any(not self.has_edge(u, w) for u, w in zip(nodes, nodes[1:])):
            return "bad_edge", None
        if nodes[-1] != t:
            return "wrong_end", None
        if not finished:
            return "no_eos", None
        return "ok", tuple(nodes)

    def pairs(self, min_span: int = 2) -> List[Tuple[int, int]]:
        """(s, t) with t at least min_span layers after s and reachable from s."""
        out = []
        for t in self.nodes:
            n = self.path_count(t)
            for s in self.nodes:
                if self.layer_of[t] - self.layer_of[s] >= min_span and n[s] > 0:
                    out.append((s, t))
        return sorted(out)

    # -- (de)serialization
    def to_json(self) -> dict:
        return {"layers": self.layers,
                "succ": {str(u): [[w, p] for w, p in s] for u, s in self.succ.items() if s},
                "alpha": {str(u): a for u, a in self.alpha.items()},
                "params": self.params}

    @classmethod
    def from_json(cls, d: dict) -> "Graph":
        return cls(d["layers"], {int(u): [(w, p) for w, p in s] for u, s in d["succ"].items()},
                   {int(u): a for u, a in d.get("alpha", {}).items()}, d.get("params"))

    @classmethod
    def load(cls, path: str) -> "Graph":
        with open(path) as f:
            d = json.load(f)
        return cls.from_json(d["graph"] if "graph" in d else d)


def _dirichlet(rng: random.Random, alpha: float, k: int) -> List[float]:
    g = [rng.gammavariate(alpha, 1.0) for _ in range(k)]
    tot = sum(g)
    if tot <= 0:            # all draws underflowed (tiny alpha): fall back to one-hot
        g = [1.0] + [0.0] * (k - 1)
        tot = 1.0
    return [x / tot for x in g]


def build_graph(seed: int = 0, layer_sizes: Sequence[int] = DEFAULT_LAYER_SIZES,
                alphas: Sequence[float] = DEFAULT_ALPHAS, det_frac: float = DEFAULT_DET_FRAC,
                min_degree: int = DEFAULT_MIN_DEGREE, max_degree: int = DEFAULT_MAX_DEGREE,
                weight_floor: float = DEFAULT_WEIGHT_FLOOR) -> Graph:
    """Random layered DAG. Each non-last node draws: deterministic (one successor) with probability det_frac,
    else an out-degree in [min_degree, max_degree]; successors uniformly from the next layer; edge probabilities
    floor + (1 - k floor) * Dirichlet(alpha), alpha drawn per node from alphas. Every node after layer 0 gets an
    in-edge (added to a random non-deterministic node of the previous layer, whose row is then redrawn)."""
    if sum(layer_sizes) != NUM_NODES:
        raise ValueError(f"layer sizes must sum to {NUM_NODES}, got {sum(layer_sizes)}")
    if max_degree * weight_floor >= 1:
        raise ValueError("max_degree * weight_floor must be < 1")
    rng = random.Random(f"toy-graph:{seed}")
    layers, start = [], 0
    for size in layer_sizes:
        layers.append(list(range(start, start + size)))
        start += size

    det, alpha, succs = {}, {}, {}

    def draw_row(u: int) -> List[Tuple[int, float]]:
        ws = sorted(succs[u])
        if len(ws) == 1:
            return [(ws[0], 1.0)]
        d = _dirichlet(rng, alpha[u], len(ws))
        free = 1.0 - weight_floor * len(ws)
        return [(w, weight_floor + free * x) for w, x in zip(ws, d)]

    succ: Dict[int, List[Tuple[int, float]]] = {}
    for li, layer in enumerate(layers[:-1]):
        nxt = layers[li + 1]
        for u in layer:
            det[u] = rng.random() < det_frac
            alpha[u] = alphas[rng.randrange(len(alphas))]
            k = 1 if det[u] else rng.randint(min(min_degree, len(nxt)), min(max_degree, len(nxt)))
            succs[u] = set(rng.sample(nxt, k))
        for w in nxt:                                   # every node of the next layer gets an in-edge
            if not any(w in succs[u] for u in layer):
                cands = [u for u in layer if not det[u] and len(succs[u]) < max_degree] \
                    or [u for u in layer if not det[u]] or list(layer)
                u = cands[rng.randrange(len(cands))]
                succs[u].add(w)
                det[u] = False
        for u in layer:
            succ[u] = draw_row(u)
    params = {"seed": seed, "layer_sizes": list(layer_sizes), "alphas": list(alphas), "det_frac": det_frac,
              "min_degree": min_degree, "max_degree": max_degree, "weight_floor": weight_floor}
    return Graph(layers, succ, alpha, params)


def split_pairs(pairs: Sequence[Tuple[int, int]], graph: Graph, heldout_frac: float, seed: int
                ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Train / held-out split of (s, t) pairs, stratified by span (layer distance)."""
    by_span: Dict[int, List[Tuple[int, int]]] = {}
    for s, t in pairs:
        by_span.setdefault(graph.layer_of[t] - graph.layer_of[s], []).append((s, t))
    train, held = [], []
    for span in sorted(by_span):
        group = sorted(by_span[span])
        random.Random(f"toy-split:{seed}:{span}").shuffle(group)
        k = int(round(heldout_frac * len(group)))
        held += group[:k]
        train += group[k:]
    return sorted(train), sorted(held)
