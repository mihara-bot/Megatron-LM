"""Tests for the margin toy (ladder/toy): the synthetic language, the margin theory, the launchers, the summary.

    python -m pytest ladder/tests/test_toy.py -p no:cacheprovider -o addopts=""

Standard library for most tests. The probe tests need torch (and transformers for the end-to-end one) and are
skipped without them; run them on the cluster.
"""
import csv
import json
import math
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ladder" / "toy"))

import graph as G  # noqa: E402
import make_data  # noqa: E402
import margin_theory as T  # noqa: E402
import summarize  # noqa: E402


def small_graph():
    """3 layers x 3 nodes plus a 4th layer of 2 (11 nodes; build_graph is fixed to 92): known paths."""
    layers = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    succ = {0: [(3, 0.5), (4, 0.3), (5, 0.2)], 1: [(4, 1.0)], 2: [(4, 0.6), (5, 0.4)],
            3: [(6, 0.7), (7, 0.3)], 4: [(6, 0.1), (7, 0.2), (8, 0.7)], 5: [(8, 1.0)],
            6: [(9, 0.5), (10, 0.5)], 7: [(9, 1.0)], 8: [(9, 0.2), (10, 0.8)]}
    return G.Graph(layers, succ)


def all_paths(g, s, t):
    if s == t:
        return [[t]]
    return [[s] + rest for w, _ in g.succ[s] for rest in all_paths(g, w, t)]


# ---- vocabulary and graph -------------------------------------------------------------------------------
def test_vocab_layout():
    v = G.vocab()
    assert len(v) == G.VOCAB_SIZE == 100 and len(set(v)) == 100
    assert v[G.UNK] == "<unk>" and v[G.EOS] == "<eos>" and v[G.WALK] == "<walk>" and v[G.PATH] == "<path>"
    assert v[G.SEP] == ":" and v[G.node_token(0)] == "n0" and v[-1] == f"n{G.NUM_NODES - 1}"
    assert all(G.token_node(G.node_token(i)) == i for i in range(G.NUM_NODES))
    assert G.token_node(G.EOS) is None and G.token_node(100) is None
    assert G.walk_ids([3, 4]) == [G.WALK, G.node_token(3), G.node_token(4), G.EOS]
    assert G.walk_text([3, 4]).split() == [v[i] for i in G.walk_ids([3, 4])[:-1]]


@pytest.mark.parametrize("seed", [0, 3])
def test_graph_structure(seed):
    g = G.build_graph(seed)
    p = g.params
    assert [len(layer) for layer in g.layers] == list(G.DEFAULT_LAYER_SIZES) and len(g.nodes) == G.NUM_NODES
    indeg = {v: 0 for v in g.nodes}
    for u in g.nodes:
        if g.is_last(u):
            assert g.succ[u] == []
            continue
        assert g.succ[u], u
        assert abs(sum(q for _, q in g.succ[u]) - 1) < 1e-12
        for w, q in g.succ[u]:
            assert g.layer_of[w] == g.layer_of[u] + 1
            assert q >= p["weight_floor"] - 1e-12 or g.out_degree(u) == 1
            indeg[w] += 1
        assert g.out_degree(u) == 1 or p["min_degree"] <= g.out_degree(u) <= p["max_degree"] + 1
    assert all(indeg[v] >= 1 for layer in g.layers[1:] for v in layer)
    det = sum(g.out_degree(u) == 1 for u in g.nodes if not g.is_last(u))
    assert 5 <= det <= 30
    again = G.build_graph(seed)
    assert again.succ == g.succ and again.alpha == g.alpha
    back = G.Graph.from_json(json.loads(json.dumps(g.to_json())))
    assert back.succ == g.succ and back.layers == g.layers and back.params == g.params
    assert G.build_graph(seed + 1).succ != g.succ


def test_walk_next_tables():
    g = G.build_graph(0)
    never_context = {G.UNK, G.PATH, G.SEP} | set(range(G.SEP + 1, G.NODE0))   # <rsv*> too
    for tok in range(G.VOCAB_SIZE):
        d = g.walk_next(tok)
        if tok in never_context:
            assert d is None
            continue
        assert d is not None and abs(sum(d.values()) - 1) < 1e-12, tok
    assert g.walk_next(G.EOS) == {G.WALK: 1.0}
    assert set(g.walk_next(G.WALK)) == {G.node_token(v) for v in g.layers[0]}
    last = g.layers[-1][0]
    assert g.walk_next(G.node_token(last)) == {G.EOS: 1.0}


def test_walk_frequencies_match_transition_rows():
    g = G.build_graph(0)
    rng = random.Random(1)
    counts, visits = {}, {}
    n = 60_000
    for _ in range(n):
        w = g.sample_walk(rng)
        assert len(w) == g.num_layers
        assert w[0] in g.layers[0]
        for a, b in zip(w, w[1:]):
            assert g.has_edge(a, b)
            visits[a] = visits.get(a, 0) + 1
            counts[(a, b)] = counts.get((a, b), 0) + 1
    mu = g.layer_marginals()
    for u in sorted(visits, key=visits.get)[-10:]:            # the most visited nodes: tight estimates
        assert abs(visits[u] / n - mu[u]) < 5 * math.sqrt(mu[u] / n) + 1e-3
        for w, p in g.succ[u]:
            f = counts.get((u, w), 0) / visits[u]
            assert abs(f - p) < 5 * math.sqrt(p * (1 - p) / visits[u]) + 1e-3, (u, w, f, p)


def test_entropy_rate_is_the_stream_average():
    g = G.build_graph(0)
    rng = random.Random(2)
    tot, cnt = 0.0, 0
    for _ in range(20_000):
        ids = G.walk_ids(g.sample_walk(rng)) + [G.WALK]          # the next document starts with <walk>
        for prev, nxt in zip(ids, ids[1:]):
            tot -= math.log(g.walk_next(prev)[nxt])
            cnt += 1
    assert cnt == 20_000 * (g.num_layers + 2)
    assert abs(tot / cnt - g.entropy_rate()) < 0.01


def test_paths_against_brute_force():
    g = small_graph()
    for s in g.nodes:
        for t in g.nodes:
            paths = all_paths(g, s, t) if g.layer_of[t] > g.layer_of[s] else ([[s]] if s == t else [])
            assert g.path_count(t)[s] == len(paths)
            reach = sum(math.prod(g.prob(a, b) for a, b in zip(p, p[1:])) for p in paths)
            assert g.reach_prob(t)[s] == pytest.approx(reach)
            if g.layer_of[t] <= g.layer_of[s] or not paths:
                continue
            for mode in ("cond", "uniform", "canon"):
                probs = [g.path_prob(p, t, mode) for p in paths]
                assert sum(probs) == pytest.approx(1.0), (s, t, mode)
            walk = [math.prod(g.prob(a, b) for a, b in zip(p, p[1:])) for p in paths]
            for p, w in zip(paths, walk):          # cond = walk probability conditioned on reaching t
                assert g.path_prob(p, t, "cond") == pytest.approx(w / reach)
                assert g.path_prob(p, t, "uniform") == pytest.approx(1 / len(paths))
            best = max(range(len(paths)), key=lambda i: (walk[i], [-x for x in paths[i]]))
            assert g.path_prob(paths[best], t, "canon") == 1.0
            assert g.valid_next(s, t) == sorted({p[1] for p in paths})


def test_sampled_paths_follow_their_mode():
    g = small_graph()
    s, t = 0, 9
    paths = all_paths(g, s, t)
    for mode in ("cond", "uniform"):
        rng = random.Random(mode)
        n = 20_000
        freq = {}
        for _ in range(n):
            p = tuple(g.sample_path(s, t, mode, rng))
            freq[p] = freq.get(p, 0) + 1
        for p in paths:
            want = g.path_prob(p, t, mode)
            assert abs(freq.get(tuple(p), 0) / n - want) < 5 * math.sqrt(want * (1 - want) / n) + 1e-3
    assert {tuple(g.sample_path(s, t, "canon", random.Random(i))) for i in range(20)} == {(0, 3, 6, 9)}


def test_check_generation():
    g = small_graph()
    tok = G.node_token
    E = G.EOS
    assert g.check_generation(0, 9, [tok(0), tok(3), tok(6), tok(9), E]) == ("ok", (0, 3, 6, 9))
    assert g.check_generation(0, 9, [tok(0), tok(3), tok(6), tok(9), E, tok(1)])[0] == "ok"
    assert g.check_generation(0, 9, [tok(1), tok(4), tok(6), tok(9), E])[0] == "bad_start"
    assert g.check_generation(0, 9, [tok(0), tok(5), tok(6), tok(9), E])[0] == "bad_edge"
    assert g.check_generation(0, 9, [tok(0), tok(3), tok(6), tok(10), E])[0] == "wrong_end"
    assert g.check_generation(0, 9, [tok(0), tok(3), E])[0] == "wrong_end"
    assert g.check_generation(0, 9, [tok(0), tok(3), tok(6), tok(9)])[0] == "no_eos"
    assert g.check_generation(0, 9, [E])[0] == "empty"
    assert g.check_generation(0, 9, [tok(0), G.SEP, E])[0] == "bad_token"


def test_pair_split():
    g = G.build_graph(0)
    pairs = g.pairs(2)
    assert pairs and all(g.layer_of[t] - g.layer_of[s] >= 2 and g.path_count(t)[s] > 0 for s, t in pairs)
    tr, he = G.split_pairs(pairs, g, 0.2, 0)
    assert not set(tr) & set(he) and sorted(tr + he) == pairs
    assert abs(len(he) / len(pairs) - 0.2) < 0.01
    assert (tr, he) == G.split_pairs(pairs, g, 0.2, 0)
    spans = {g.layer_of[t] - g.layer_of[s] for s, t in he}
    assert spans == {g.layer_of[t] - g.layer_of[s] for s, t in pairs}


# ---- make_data ------------------------------------------------------------------------------------------
def test_sft_rows():
    g = G.build_graph(0)
    tr, _ = G.split_pairs(g.pairs(2), g, 0.2, 0)
    pairs = tr[:30]
    for mode in ("cond", "uniform", "canon"):
        rows = make_data.sft_rows(g, pairs, mode, reps=3, epochs=2, seed=0)
        assert len(rows) == len(pairs) * 3 * 2
        assert [r[3] for r in rows] == [0] * 90 + [1] * 90
        assert sorted(r[2] for r in rows[:90]) == list(range(90))
        for prompt, resp, sid, _ in rows:
            s, t = pairs[sid // 3]
            assert prompt == G.path_prompt_text(s, t) and prompt.endswith(" ")
            # identity template: the HF tokenizer sees prompt + response and splits on whitespace
            assert (prompt + resp).split() == prompt.split() + resp.split()
            nodes = [int(x[1:]) for x in resp.split()]
            assert g.check_generation(s, t, [G.node_token(v) for v in nodes] + [G.EOS])[0] == "ok"
        by_id = {}
        for _, resp, sid, ep in rows:
            by_id.setdefault(sid, set()).add(resp)
        if mode == "canon":
            assert all(len(v) == 1 for v in by_id.values())
        else:
            assert any(len(v) > 1 for v in by_id.values())
    stats = make_data.sft_stats(rows, "canon", len(pairs), 3, 2, 256, 256)
    assert stats["rows_out"] == stats["examples"] * stats["epochs"] == 180
    assert stats["pipeline"]["examples_per_step"] == 256 and stats["seq_len"] == 256
    assert stats["max_example_tokens"] <= 4 + g.num_layers + 1


def test_make_data_cli(tmp_path):
    make_data.main(["--out", str(tmp_path), "--train-walks", "500", "--val-walks", "20", "--sft-modes", ""])
    d = json.load(open(tmp_path / "graph.json"))
    g = G.Graph.from_json(d["graph"])
    assert g.succ == G.build_graph(0).succ
    ref = d["reference"]
    assert ref["entropy_rate"] == pytest.approx(g.entropy_rate())
    assert ref["pairs_train"] + ref["pairs_heldout"] == len(g.pairs(2))
    lines = (tmp_path / "pretrain" / "train.jsonl").read_text().splitlines()
    assert len(lines) == 500
    words = json.loads(lines[0])["text"].split()
    assert words[0] == "<walk>" and len(words) == g.num_layers + 1
    again = tmp_path / "again"
    make_data.main(["--out", str(again), "--train-walks", "500", "--val-walks", "20", "--sft-modes", ""])
    assert (again / "pretrain" / "train.jsonl").read_text() == (tmp_path / "pretrain" / "train.jsonl").read_text()


# ---- margin theory ------------------------------------------------------------------------------------------
def gd_optimum(q, m, steps=20000, lr=0.5):
    z = [math.log(x) for x in q]
    c = -math.expm1(-m)
    for _ in range(steps):
        mx = max(z)
        e = [math.exp(v - mx) for v in z]
        s = sum(e)
        p = [x / s for x in e]
        r = [qy / (1 - c * p[y]) for y, qy in enumerate(q)]
        R = sum(r)
        z = [zk - lr * (p[k] * R - r[k]) for k, zk in enumerate(z)]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    return [x / sum(e) for x in e]


def test_closed_form_matches_explicit_margin():
    rng = random.Random(0)
    for _ in range(50):
        z = [rng.gauss(0, 3) for _ in range(7)]
        y, m = rng.randrange(7), rng.choice([-3.0, -1.2, -0.1, 0.0, 0.3, 1.2, 5.0])
        assert T.margin_loss_closed(z, y, m) == pytest.approx(T.margin_loss(z, y, m), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("q", [[0.5, 0.3, 0.2], [0.9, 0.1], [0.6, 0.3, 0.09, 0.01], [0.4, 0.4, 0.2], [0.97, 0.01, 0.01, 0.01]])
@pytest.mark.parametrize("m", [-3.0, -1.2, -0.3, 0.3, 1.2, 3.0])
def test_margin_optimum(q, m):
    p = T.margin_optimum(q, m)
    assert abs(sum(p) - 1) < 1e-12
    assert T.stationarity_residual(p, q, m) < 1e-10
    gd = gd_optimum(q, m)
    assert max(abs(a - b) for a, b in zip(p, gd)) < 1e-4
    best = T.expected_margin_loss(p, q, m)
    rng = random.Random(1)
    for _ in range(20):                                # a minimum: perturbed logits never do better
        z = [math.log(x) + rng.gauss(0, 0.05) for x in p]
        s = sum(math.exp(v) for v in z)
        assert T.expected_margin_loss([math.exp(v) / s for v in z], q, m) >= best - 1e-12
    top = q.index(max(q))
    if m > 0:
        assert p[top] >= q[top] - 1e-12                # forward sharpens
    else:
        assert p[top] <= q[top] + 1e-12                # reverse flattens


def test_margin_optimum_limits():
    q = [0.6, 0.3, 0.09, 0.01]
    assert T.margin_optimum(q, 0.0) == pytest.approx(q)
    assert T.margin_optimum(q, 1e-9) == pytest.approx(q, abs=1e-8)
    root = [math.sqrt(x) for x in q]
    assert max(abs(a - b / sum(root)) for a, b in zip(T.margin_optimum(q, -60.0), root)) < 1e-6
    assert T.margin_optimum([1.0, 0.0, 0.0], 3.0) == pytest.approx([1.0, 0.0, 0.0])
    assert T.margin_optimum([1.0, 0.0, 0.0], -3.0) == pytest.approx([1.0, 0.0, 0.0])
    # numbers quoted in ladder/toy/TOY.md
    assert [round(x, 3) for x in T.margin_optimum([0.5, 0.3, 0.2], 1.2)] == [0.591, 0.253, 0.156]
    assert [round(x, 3) for x in T.margin_optimum([0.5, 0.3, 0.2], -1.2)] == [0.448, 0.316, 0.236]


# ---- probe (torch) ------------------------------------------------------------------------------------
def test_probe_margin_optimum_matches_theory():
    torch = pytest.importorskip("torch")
    import probe

    rng = random.Random(0)
    qs, ms = [], []
    for _ in range(40):
        k = rng.randint(1, 8)
        row = [0.0] * 12
        for i in rng.sample(range(12), k):
            row[i] = rng.random() ** 3 + 1e-3
        s = sum(row)
        qs.append([x / s for x in row])
        ms.append(rng.choice([-3.0, -1.2, 0.0, 0.3, 1.2, 3.0, 8.0]))
    # float64 inputs: torch.tensor defaults to float32, and rounding q and m to float32 alone moves p* by ~1e-8
    got = probe.margin_optimum(torch.tensor(qs, dtype=torch.float64), torch.tensor(ms, dtype=torch.float64))
    for q, m, row in zip(qs, ms, got.tolist()):
        assert max(abs(a - b) for a, b in zip(row, T.margin_optimum(q, m))) < 1e-9


def test_probe_metrics():
    torch = pytest.importorskip("torch")
    import probe

    q = torch.tensor([[0.5, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    p = torch.tensor([[0.4, 0.4, 0.2, 0.0], [0.5, 0.5, 0.0, 0.0]], dtype=torch.float64).clamp_min(1e-30)
    met = probe.dist_metrics(torch.log(p), q)
    assert met["junk"].tolist() == pytest.approx([0.2, 0.5])
    assert met["supp_kl"][0] == pytest.approx(0.0, abs=1e-12)          # right shape on the support
    assert met["kl"][0] == pytest.approx(math.log(1 / 0.8))
    assert met["acc"].tolist() == [1.0, 1.0]
    same = probe.dist_metrics(torch.log(q.clamp_min(1e-300)), q)
    assert same["kl"].abs().max() < 1e-12 and same["junk"].abs().max() < 1e-12
    assert probe.pass_at_k(10, 0, 1) == 0 and probe.pass_at_k(10, 10, 5) == 1
    assert probe.pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert probe.pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)


def test_probe_decode_probs():
    torch = pytest.importorskip("torch")
    import probe

    torch.manual_seed(0)
    logits = torch.randn(5, 12) * 3
    p = torch.softmax(logits.double(), -1)
    assert torch.allclose(probe.decode_probs(logits, "T", 1.0).double(), p, atol=1e-6)
    assert torch.allclose(probe.decode_probs(logits, "T", 2.0).double(), torch.softmax(logits.double() / 2, -1),
                          atol=1e-6)
    assert torch.allclose(probe.decode_probs(logits, "M", 0.0), p, atol=1e-12)
    for m in (-3.0, -1.2, 1.2):
        got = probe.decode_probs(logits, "M", m)
        assert torch.allclose(got.sum(-1), torch.ones(5, dtype=torch.float64))
        for row, prow in zip(got.tolist(), p.tolist()):
            assert max(abs(a - b) for a, b in zip(row, T.margin_optimum(prow, m))) < 1e-9
    # reverse: flatter at the top while a small probability is only rescaled (x1.83 here; temperature 1.4, which
    # gives the same top, raises it x28)
    q = torch.tensor([[x * (1 - 3e-4) for x in (0.5, 0.3, 0.2)] + [3e-4 / 95] * 95], dtype=torch.float64)
    mp = probe.decode_probs(torch.log(q), "M", -1.2)[0]
    assert float(mp[0]) == pytest.approx(0.44763, abs=1e-4) and float(mp[3] / q[0, 3]) == pytest.approx(1.8255, abs=1e-3)
    with pytest.raises(ValueError):
        probe.decode_probs(logits, "X", 1.0)
    assert probe.decode_key("T", 1.0) == "T1" and probe.decode_key("M", -1.2) == "M-1.2"
    assert probe.decode_seed(0, "T", 0.7) == 70000 and probe.decode_seed(0, "M", -1.2) != probe.decode_seed(0, "T", -1.2)


def test_probe_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import probe

    make_data.main(["--out", str(tmp_path / "data"), "--train-walks", "10", "--val-walks", "10", "--sft-modes", ""])
    cfg = transformers.LlamaConfig(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
                                   tie_word_embeddings=False)
    torch.manual_seed(0)
    model_dir = tmp_path / "hf"
    transformers.LlamaForCausalLM(cfg).save_pretrained(model_dir)
    head = {"lm_head_feature_norm": "hfn", "lm_head_feature_norm_scale": 60.0, "token_margin": 0.02,
            "token_margin_reverse": True}
    (model_dir / "export_meta.json").write_text(json.dumps({"iteration": 5, "lm_head": head}))
    probe.main(["--data", str(tmp_path / "data"), "--models", str(model_dir), "--device", "cpu", "--walk-docs", "100",
                "--paths", "--pairs", "8", "--samples", "8", "--temps", "1.0"])
    res = json.load(open(model_dir / "probe.json"))
    w = res["walk"]
    assert w["classes"]["multi"]["n"] > 0 and w["classes"]["multi"]["m"] == pytest.approx(-1.2)
    assert len(w["temperature"]) == len(probe.TEMPS) and len(w["posthoc_margin"]) == len(probe.MAP_MS)
    assert {"s", "ce"} <= set(w["s_fit"])
    s = res["paths"]["sample"]["T1"]["heldout"]
    assert s["pairs"] == 8 and 0 <= s["pass@1"] <= 1 and abs(sum(s["fail"].values()) - 1) < 1e-9
    assert set(res["paths"]["tf"]["train"]["classes"]) == set(probe.PATH_CLASSES)
    # --update: add a temperature and a margin-map setting, keep everything else
    base = ["--data", str(tmp_path / "data"), "--models", str(model_dir), "--device", "cpu", "--paths",
            "--samples", "8", "--update", "--temps", "1.0,2.0"]
    probe.main(base + ["--pairs", "8", "--map-ms=-1.2"])
    text = (model_dir / "probe.json").read_text()
    new = json.loads(text)

    def same(a, b):                                                          # NaN-safe
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    assert set(new["paths"]["sample"]) == {"T1", "T2", "M-1.2"} and new["updates"][0]["added"] == ["T2", "M-1.2"]
    assert same(new["paths"]["sample"]["T1"], res["paths"]["sample"]["T1"]) and same(new["walk"], res["walk"])
    assert same(new["paths"]["tf"], res["paths"]["tf"])
    mm = new["paths"]["sample"]["M-1.2"]["heldout"]
    assert mm["pairs"] == 8 and 0 <= mm["pass@1"] <= 1 and abs(sum(mm["fail"].values()) - 1) < 1e-9
    probe.main(base + ["--pairs", "8", "--map-ms=-1.2"])                     # nothing to add: file untouched
    assert (model_dir / "probe.json").read_text() == text
    with pytest.raises(SystemExit):                                          # other pairs: not comparable
        probe.main(base + ["--pairs", "4", "--map-ms=-3"])


# ---- launchers (dry runs) -------------------------------------------------------------------------------
def bash(script, env=None, args=()):
    full = {**os.environ, **(env or {})}
    for k in ("HEAD", "MARGIN_D", "MARGIN_REVERSE", "LR", "MODE", "TPP", "STOP_TPP", "STAGES", "SEED"):
        if k not in (env or {}):
            full.pop(k, None)
    return subprocess.run(["bash", script, *args], cwd=REPO, env=full, capture_output=True, text=True)


def test_toy_config_numbers_and_schedule(tmp_path):
    out = subprocess.run(["bash", "-c", "source ladder/configs/toy_l2.sh; echo $N_EFF $TOKENIZER_VOCAB; "
                          "printf '%s\\n' \"${MODEL_ARGS[@]}\""], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout.split("\n")
    n_eff, vocab = map(int, out[0].split())
    a = out[1:]
    val = lambda f: int(a[a.index(f) + 1])  # noqa: E731
    L, d, f, H, c = (val(x) for x in ("--num-layers", "--hidden-size", "--ffn-hidden-size", "--num-attention-heads",
                                      "--kv-channels"))
    layer = d * c * H + 2 * d * c * H + c * H * d + 3 * d * f + 2 * d
    assert n_eff == L * layer + d + d * vocab
    assert vocab % val("--make-vocab-size-divisible-by") == 0, "the toy vocabulary must not be padded"
    assert vocab == G.VOCAB_SIZE

    res = bash("ladder/train.sh", {"CONFIG": "toy_l2", "OPT": "muon", "DRY_RUN": "1", "TRAIN_DATA": "x",
                                   "VALID_DATA": "x", "TOKENIZER": "x", "OUT_ROOT": str(tmp_path), "GPUS_PER_NODE": "1"})
    assert res.returncode == 0, res.stderr
    cmd = shlex.split(res.stdout.strip().splitlines()[-1])
    arg = lambda f: cmd[cmd.index(f) + 1]  # noqa: E731
    assert arg("--train-iters") == "6250" and arg("--lr-wsd-decay-iters") == "1250"
    assert arg("--save-interval") == arg("--save-retain-interval") == "500"
    assert arg("--make-vocab-size-divisible-by") == "4" and arg("--global-batch-size") == "128"
    assert "1 x 1 GPUs" in res.stdout


def test_chain_dry_run(tmp_path):
    env = {"TOY_ROOT": str(tmp_path), "DRY_RUN": "1", "HEAD": "hfn", "MARGIN_D": "0.02", "MARGIN_REVERSE": "1",
           "LR": "4e-3", "SEED": "2"}
    res = bash("ladder/toy/chain.sh", env)
    assert res.returncode == 0, res.stderr
    run = "toy_l2_muon_lr4e-3_hfn_rev_d0.02_seed2"
    assert f"== toy chain {run}" in res.stdout
    cmds = [line for line in res.stdout.splitlines() if line.startswith("+ ")]
    assert cmds[0] == "+ env MODE=stable bash ladder/train.sh"
    assert cmds[1:] == ["+ bash ladder/sft/train_sft.sh"] * 2          # cond, canon; nothing to export yet
    # once checkpoints exist, export + probe them
    ck = tmp_path / "out" / "runs" / run / "ckpt" / "main"
    for it in (500, 1000):
        (ck / f"iter_{it:07d}").mkdir(parents=True)
    (ck / "latest_checkpointed_iteration.txt").write_text("1000")
    res = bash("ladder/toy/chain.sh", {**env, "STAGES": "export,probe", "PRE_ITERS": "all"})
    assert res.returncode == 0, res.stderr
    cmds = [line for line in res.stdout.splitlines() if line.startswith("+ ")]
    assert len(cmds) == 2 and "ITER=500" in cmds[0] and f"OUT={ck.parent.parent}/hf/main_iter1000" in cmds[1]
    # export_hf.py writes export_meta.json even when its logits check fails: such an export (passed false, or no
    # "passed" from before the near-tie check) is redone and never probed; a passed one is kept and probed
    hf = ck.parent.parent / "hf"
    for it, val in ((500, {"passed": True}), (1000, {"top1_agreement": 0.9785})):
        (hf / f"main_iter{it}").mkdir(parents=True)
        (hf / f"main_iter{it}" / "export_meta.json").write_text(json.dumps({"validation": val}))
    res = bash("ladder/toy/chain.sh", {**env, "STAGES": "export,probe", "PRE_ITERS": "all", "PYTHON": sys.executable})
    assert res.returncode == 0, res.stderr
    cmds = [line for line in res.stdout.splitlines() if line.startswith("+ ")]
    assert len(cmds) == 2 and "ITER=1000" in cmds[0] and "ladder/toy/probe.py" in cmds[1]
    assert "main_iter500" in cmds[1] and "main_iter1000" not in cmds[1]
    # decode: probe.py --update on the last probed, passed SFT export of each mode
    sft_hf = tmp_path / "out" / "sft" / f"{run}_main_sft_toy_cond" / "hf"
    for it in (122, 974):
        (sft_hf / f"iter_{it}").mkdir(parents=True)
        (sft_hf / f"iter_{it}" / "export_meta.json").write_text(json.dumps({"validation": {"passed": True}}))
        (sft_hf / f"iter_{it}" / "probe.json").write_text("{}")
    denv = {**env, "STAGES": "decode", "PYTHON": sys.executable}
    res = bash("ladder/toy/chain.sh", denv)
    assert res.returncode == 0, res.stderr
    cmds = [line for line in res.stdout.splitlines() if line.startswith("+ ")]
    assert len(cmds) == 1 and "--update" in cmds[0] and "--map-ms=-0.6,-1.2,-2,-3,-5" in cmds[0]
    assert "--sft-mode cond" in cmds[0] and "iter_974" in cmds[0] and "iter_122" not in cmds[0]
    assert "no probed SFT export" in res.stdout                               # canon has none yet
    res = bash("ladder/toy/chain.sh", {**denv, "DECODE_ITERS": "all", "DECODE_ARGS": "--map-ms=-3"})
    cmds = [line for line in res.stdout.splitlines() if line.startswith("+ ")]
    assert "iter_122" in cmds[0] and "iter_974" in cmds[0] and "--map-ms=-3 " in cmds[0]
    assert bash("ladder/toy/chain.sh", {**denv, "DECODE_ITERS": "first"}).returncode != 0
    # SFT protocol B: its own SFT directories (ladder/sft/common.sh), none exported yet
    res = bash("ladder/toy/chain.sh", {**denv, "SFT_MARGIN_D": "0.05", "SFT_MARGIN_REVERSE": "1"})
    assert res.returncode == 0, res.stderr
    assert not [line for line in res.stdout.splitlines() if line.startswith("+ ")]
    assert f"{run}_main_sft_toy_cond_B_rev_d0.05/hf" in res.stdout
    assert bash("ladder/toy/chain.sh", {**denv, "SFT_MARGIN_REVERSE": "1"}).returncode != 0   # needs SFT_MARGIN_D
    assert bash("ladder/toy/chain.sh", {**env, "STAGES": "pretrain,bogus"}).returncode != 0
    assert bash("ladder/toy/chain.sh", {**env, "HEAD": "xfn"}).returncode != 0


def test_pack_plan_and_validation(tmp_path):
    spec = tmp_path / "spec.txt"
    spec.write_text("# comment\nHEAD=plain LR=1e-3\n\nHEAD=hfn MARGIN_D=0.02 LR=1e-3  # trailing\n"
                    "HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=1 LR=1e-3\n")
    env = {"TOY_ROOT": str(tmp_path), "DRY_RUN": "1", "GPUS": "2", "RUNS_PER_GPU": "1", "BASE_PORT": "30000"}
    res = bash("ladder/toy/pack.sh", env, [str(spec)])
    assert res.returncode == 0, res.stderr
    slots = [line for line in res.stdout.splitlines() if line.strip().startswith("slot")]
    assert len(slots) == 3
    assert slots[0].strip().startswith("slot 0 (GPU 0, port 30000): HEAD=plain LR=1e-3")
    assert slots[2].strip().startswith("slot 0 (GPU 0, port 30000): HEAD=hfn MARGIN_D=0.02 MARGIN_REVERSE=1")
    spec.write_text("HEAD=plain LR=1e-3\nHEAD=plain  LR=1e-3\n")
    assert "same chain twice" in bash("ladder/toy/pack.sh", env, [str(spec)]).stderr
    spec.write_text("HEAD=plain lr=1e-3\n")
    assert "not a KEY=VALUE" in bash("ladder/toy/pack.sh", env, [str(spec)]).stderr


def test_run_toy_dry_run(tmp_path):
    env = {"TOY_ROOT": str(tmp_path), "DRY_RUN": "1"}
    res = bash("ladder/toy/run_toy.sh", {**env, "PHASE": "lr"})
    assert res.returncode == 0, res.stderr
    chains = [line.strip() for line in res.stdout.splitlines() if line.startswith("   CONFIG=")]
    assert len(chains) == 5 and all("HEAD=plain" in c and "STAGES=pretrain,export,probe" in c for c in chains)
    assert res.stdout.count("+ sbatch") == 1
    assert bash("ladder/toy/run_toy.sh", {**env, "PHASE": "grid"}).returncode != 0      # needs LR
    res = bash("ladder/toy/run_toy.sh", {**env, "PHASE": "grid", "LR": "4e-3"})
    assert res.returncode == 0, res.stderr
    chains = [line.strip() for line in res.stdout.splitlines() if line.startswith("   CONFIG=")]
    assert len(chains) == 45 and len(set(chains)) == 45 and res.stdout.count("+ sbatch") == 3
    assert all(" LR=4e-3 " in c for c in chains)
    res = bash("ladder/toy/run_toy.sh", {**env, "PHASE": "decode", "LR": "4e-3"})
    assert res.returncode == 0, res.stderr
    dec = [line.strip() for line in res.stdout.splitlines() if line.startswith("   CONFIG=")]
    assert [c.replace(" STAGES=decode", "") for c in dec] == chains and res.stdout.count("+ sbatch") == 1
    assert bash("ladder/toy/run_toy.sh", {**env, "PHASE": "sftB"}).returncode != 0      # needs LR
    res = bash("ladder/toy/run_toy.sh", {**env, "PHASE": "sftB", "LR": "4e-3"})
    assert res.returncode == 0, res.stderr
    sb = [line.strip() for line in res.stdout.splitlines() if line.startswith("   CONFIG=")]
    assert len(sb) == 12 and len(set(sb)) == 12 and res.stdout.count("+ sbatch") == 1
    assert all(c.endswith("STAGES=sft,decode") and "SFT_MARGIN_REVERSE=1" in c and "HEAD=hfn" in c for c in sb)
    assert sum("MARGIN_D=0.05 MARGIN_REVERSE=1 " in c for c in sb) == 6                 # B1 starts
    assert not (tmp_path / "specs").exists(), "a dry run writes nothing"


# ---- summarize ------------------------------------------------------------------------------------------
def fake_probe(kl, pass1=None):
    cls = {c: {"n": 0} for c in ("start", "det", "end", "doc")}
    cls["multi"] = {"n": 10, "kl": kl, "ce": 1.0 + kl, "h_p": 1.0, "h_q": 1.0, "top1_p": 0.5, "junk": 0.01, "m": 0.0,
                    "hnorm": 16.0, "kl_theory": kl, "kl_q_theory": 0.0, "h_theory": 1.0, "top1_theory": 0.5,
                    "supp_kl": kl}
    sweep = [{"value": v, "multi": {"kl": kl * v, "h_p": v, "junk": 0.01 * v, "supp_kl": kl, "top1_p": 0.5},
              "det": {"top1_p": 1.0, "ce": 0.01}} for v in (1.0, 2.0)]
    res = {"reference": {"entropy_rate": 0.9},
           "walk": {"classes": cls, "quantiles": {}, "head": {}, "s_fit": {"s": 20.0}, "temperature": sweep,
                    "posthoc_margin": sweep[:1]}}
    if pass1 is not None:
        res["paths"] = {"mode": "cond", "tf": {},
                        "sample": {"T1": {"heldout": {"pass@1": pass1, "cover_count": 0.5}},
                                   "T1.3": {"heldout": {"pass@1": pass1 / 2, "cover_count": 0.7}},
                                   "M-1.2": {"heldout": {"pass@1": pass1, "cover_count": 0.9,
                                                         "fail": {"ok": 0.8, "bad_edge": 0.2}}}}}
    return res


def test_summarize(tmp_path):
    out = tmp_path / "out"

    def put(path, res):
        path.mkdir(parents=True)
        (path / "probe.json").write_text(json.dumps(res))

    for lr, kl in (("1e-3", 0.05), ("2e-3", 0.02), ("4e-3", 0.03)):
        put(out / "runs" / f"toy_l2_muon_lr{lr}_seed1" / "hf" / "main_iter6250", fake_probe(kl))
    put(out / "runs" / "toy_l2_muon_lr2e-3_seed1" / "hf" / "main_iter500", fake_probe(0.9))
    for seed, kl in ((1, 0.04), (2, 0.06)):
        run = f"toy_l2_muon_lr2e-3_hfn_rev_d0.02_seed{seed}"
        put(out / "runs" / run / "hf" / "main_iter6250", fake_probe(kl))
        put(out / "sft" / f"{run}_main_sft_toy_cond" / "hf" / "iter_1100", fake_probe(kl, pass1=0.3 + seed / 10))
    put(out / "sft" / "toy_l2_muon_lr2e-3_hfn_rev_d0.02_seed1_main_sft_toy_cond_B_rev_d0.05" / "hf" / "iter_974",
        fake_probe(0.04, pass1=0.9))                                         # SFT protocol B
    pre, sft = summarize.collect(str(tmp_path))
    assert len(pre) == 6 and len(sft) == 3
    assert sorted(r["sft_protocol"] for r in sft) == ["A", "A", "B_rev_d0.05"]
    last = summarize.last_iteration(pre)
    assert {r["iteration"] for r in last} == {6250}
    g = summarize.group([r for r in last if r["variant"] == "hfn_rev_d0.02"], ["variant", "lr"], summarize.PRE_COLS)
    assert g[0]["seeds"] == 2 and g[0]["kl"] == pytest.approx(0.05) and g[0]["kl_sd"] == pytest.approx(math.sqrt(2e-4))
    assert g[0]["kl_all"] == pytest.approx(0.05)
    gs = {g["sft_protocol"]: g for g in summarize.group(sft, ["sft_mode", "sft_protocol", "variant", "lr"],
                                                         summarize.SFT_COLS)}
    assert gs["A"]["heldout_pass@1"] == pytest.approx(0.45) and gs["A"]["seeds"] == 2
    assert gs["B_rev_d0.05"]["heldout_pass@1"] == pytest.approx(0.9) and gs["B_rev_d0.05"]["seeds"] == 1
    res = subprocess.run([sys.executable, "ladder/toy/summarize.py", "lr", "--root", str(tmp_path)], cwd=REPO,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "lowest walk KL: LR=2e-3" in res.stdout
    res = subprocess.run([sys.executable, "ladder/toy/summarize.py", "table", "--root", str(tmp_path)], cwd=REPO,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    report = tmp_path / "report"
    assert (report / "pretrain_summary.csv").exists() and (report / "sft_rows.csv").exists()
    assert len(list(csv.DictReader(open(report / "pretrain_iters.csv")))) == 6          # every checkpoint
    dec = list(csv.DictReader(open(report / "pretrain_decode_summary.csv")))
    hfn = {(r["decode"], r["value"]): r for r in dec if r["variant"] == "hfn_rev_d0.02"}
    assert set(hfn) == {("T", "1.0"), ("T", "2.0"), ("map_m", "1.0")}
    assert float(hfn["T", "2.0"]["kl"]) == pytest.approx(0.1) and hfn["T", "2.0"]["seeds"] == "2"
    allrows = list(csv.DictReader(open(report / "sft_decode_summary.csv")))
    rows = [r for r in allrows if r["sft_protocol"] == "A"]
    assert [r["decode"] for r in rows] == ["T1", "T1.3", "M-1.2"]
    assert [r["decode"] for r in allrows if r["sft_protocol"] == "B_rev_d0.05"] == ["T1", "T1.3", "M-1.2"]
    dec = {r["decode"]: r for r in rows}
    assert float(dec["T1"]["heldout_pass@1"]) == pytest.approx(0.45)
    assert float(dec["T1.3"]["heldout_pass@1"]) == pytest.approx(0.225)
    assert float(dec["M-1.2"]["heldout_fail_bad_edge"]) == pytest.approx(0.2)
    assert float(dec["M-1.2"]["heldout_fail_no_eos"]) == 0.0
    order = sorted(["hfn_rev_d0.05", "plain", "sfn_d0.02", "hfn_d0.02", "hfn_rev_d0.02", "hfn_d0"],
                   key=summarize.variant_order)
    assert order == ["plain", "hfn_d0", "hfn_d0.02", "hfn_rev_d0.02", "hfn_rev_d0.05", "sfn_d0.02"]
