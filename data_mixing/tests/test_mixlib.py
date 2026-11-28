"""CPU tests for data_mixing/mixlib.py (stdlib only, no parquet).

    python -m pytest data_mixing/tests/test_mixlib.py -p no:cacheprovider -o addopts=""
    python3 data_mixing/tests/test_mixlib.py          # without pytest

Sources are in-memory lists of documents; every plan is checked against a brute-force reading of the
stream with the v1 builder's rules (skip null/empty, head = first k documents, quota crossing
document included, restart after the head when a source runs out).
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mixlib  # noqa: E402

SPEC = HERE.parent / "configs" / "mixes" / "climbmix_m1_200b.json"


class FakeReader:
    """Reader over in-memory files: data[path] = list of row values (str, '' or None)."""

    def __init__(self, data):
        self.data = data
        self.index_calls = []

    def lengths(self, f):
        return [len(t.encode("utf-8")) if t else 0 for t in self.data[f.path]]

    def index(self, f):
        self.index_calls.append(f.path)
        lens = self.lengths(f)
        return sum(1 for n in lens if n > 0), sum(lens)

    def texts(self, f, lo, hi):
        return self.data[f.path][lo:hi]


def make_source(rng, n_files, max_rows=12, empty_p=0.15):
    data, files = {}, []
    for i in range(n_files):
        rows = []
        for _ in range(rng.randint(0, max_rows)):
            u = rng.random()
            if u < empty_p / 2:
                rows.append(None)
            elif u < empty_p:
                rows.append("")
            else:
                rows.append("".join(rng.choice("abcdé漢") for _ in range(rng.randint(1, 30))))
        path = f"src/f{i:03d}.parquet"
        data[path] = rows
        files.append(mixlib.SourceFile(name=f"f{i:03d}.parquet", path=path, size=1000 + i, rows=len(rows),
                                       footer_bytes=sum(len(t.encode()) for t in rows if t)))
    return files, FakeReader(data)


def brute_force(files, reader, k, tail_bytes, quota):
    """Head docs, tail docs and the training document sequence, by reading the whole stream."""
    stream = [t for f in files for t in reader.data[f.path] if t]
    head = stream[:k]
    tail, acc = [], 0
    if tail_bytes > 0:
        for t in reversed(stream):
            tail.insert(0, t)
            acc += len(t.encode())
            if acc >= tail_bytes:
                break
        else:
            raise ValueError("source too small for the tail")
    region = stream[k: len(stream) - len(tail)]
    if len(head) + len(tail) > len(stream) or (k and len(head) < k):
        raise ValueError("overlap")
    out, consumed = [], 0
    if quota > 0:
        if not region:
            raise ValueError("empty region")
        while consumed < quota:
            for t in region:
                out.append(t)
                consumed += len(t.encode())
                if consumed >= quota:
                    break
    return head, tail, out


def materialize(files, reader, ops):
    by_path = {f.path: f for f in files}
    out = []
    for o in ops:
        f = by_path[o["src"]]
        assert (o["kind"] == "copy") == (o["lo"] == 0 and o["hi"] == f.rows)
        out.extend(mixlib.nonempty(reader.texts(f, o["lo"], o["hi"])))
    return out


def test_plan_matches_brute_force_randomized():
    rng = random.Random(1234)
    checked = 0
    for case in range(600):
        files, reader = make_source(rng, rng.randint(1, 6))
        stream = [t for f in files for t in reader.data[f.path] if t]
        total = sum(len(t.encode()) for t in stream)
        if not stream:
            continue
        k = rng.randint(0, max(0, len(stream) // 3))
        tail_bytes = rng.choice([0, rng.randint(1, max(1, total // 4))])
        region_guess = max(1, total - sum(len(t.encode()) for t in stream[:k]) - tail_bytes)
        quota = rng.choice([0, rng.randint(1, region_guess), region_guess, rng.randint(region_guess, 3 * region_guess),
                            rng.random() * 3.5 * region_guess])
        try:
            expected = brute_force(files, reader, k, tail_bytes, quota)
        except ValueError:
            expected = None
        try:
            plan = mixlib.plan_source(7, files, reader, head_docs=k, tail_bytes=tail_bytes, quota_bytes=quota,
                                      max_epochs=100)
        except ValueError:
            assert expected is None or (quota > 0 and not expected[2]), f"case {case}: plan failed, brute force did not"
            continue
        assert expected is not None, f"case {case}: brute force failed, plan did not"
        head, tail, train = expected
        assert mixlib.cut_texts(files, reader, plan["head"]) == head, f"case {case}: head"
        assert mixlib.cut_texts(files, reader, plan["tail"]) == tail, f"case {case}: tail"
        assert materialize(files, reader, plan["ops"]) == train, f"case {case}: train documents"
        assert plan["train_docs"] == len(train)
        assert plan["train_bytes"] == sum(len(t.encode()) for t in train)
        assert sum(o["bytes"] for o in plan["ops"]) == plan["train_bytes"]
        dsts = [o["dst"] for o in plan["ops"]]
        assert len(dsts) == len(set(dsts)), f"case {case}: duplicate output names"
        checked += 1
    assert checked > 300


def test_exact_quota_and_repeats():
    files, reader = make_source(random.Random(5), 4, max_rows=20, empty_p=0.0)
    stream = [t for f in files for t in reader.data[f.path] if t]
    k, region = 3, stream[3:]
    rb = sum(len(t.encode()) for t in region)
    # quota == region bytes: exactly one full pass, the last document is the crossing one
    p = mixlib.plan_source(1, files, reader, head_docs=k, tail_bytes=0, quota_bytes=rb, max_epochs=4)
    assert p["passes"] == 1 and p["train_docs"] == len(region) and p["region_bytes"] is None
    # 2.5 passes: e0 and e1 complete, e2 partial
    p = mixlib.plan_source(1, files, reader, head_docs=k, tail_bytes=0, quota_bytes=2.5 * rb, max_epochs=4)
    assert p["passes"] == 3 and p["region_bytes"] == rb and p["region_docs"] == len(region)
    assert {o["dst"].split("/")[-1][:3] for o in p["ops"]} == {"e0_", "e1_", "e2_"}
    assert p["train_bytes"] >= 2.5 * rb
    # more passes than max_epochs
    try:
        mixlib.plan_source(1, files, reader, head_docs=k, tail_bytes=0, quota_bytes=4.5 * rb, max_epochs=4)
        raise AssertionError("expected max_epochs error")
    except ValueError as e:
        assert "max_epochs" in str(e)


def test_reference_docs_must_be_in_pass_e0():
    files, reader = make_source(random.Random(9), 3, max_rows=15, empty_p=0.0)
    p = mixlib.plan_source(1, files, reader, head_docs=2, tail_bytes=0, quota_bytes=10, max_epochs=4)
    try:
        mixlib.plan_source(1, files, reader, head_docs=2, tail_bytes=0, quota_bytes=10, max_epochs=4,
                           must_contain_docs=p["pass0_docs"] + 1)
        raise AssertionError("expected an error")
    except ValueError as e:
        assert "must contain" in str(e)


def test_no_index_reads_past_the_crossing():
    files, reader = make_source(random.Random(3), 30, max_rows=10, empty_p=0.0)
    first_two = sum(f.footer_bytes for f in files[:2])
    p = mixlib.plan_source(1, files, reader, head_docs=1, tail_bytes=5, quota_bytes=first_two, max_epochs=4)
    last_used = max(int(Path(o["src"]).stem[1:]) for o in p["ops"])
    assert all(int(Path(c).stem[1:]) <= last_used for c in reader.index_calls)
    assert last_used <= 3
    assert mixlib.likely_files(files, first_two, 5) == list(range(len(mixlib.likely_files(files, first_two, 5))))
    assert len(mixlib.likely_files(files, 10 * sum(f.footer_bytes for f in files), 5)) == len(files)


def test_overlapping_validation_sets_rejected():
    files, reader = make_source(random.Random(11), 2, max_rows=5, empty_p=0.0)
    total = sum(f.footer_bytes for f in files)
    n_docs = sum(f.rows for f in files)
    try:
        mixlib.plan_source(1, files, reader, head_docs=n_docs - 1, tail_bytes=total, quota_bytes=1, max_epochs=4)
        raise AssertionError("expected an error")
    except ValueError:
        pass


def test_texts_sha256_definition():
    # the v1 builder's hash: utf-8 text followed by a NUL byte, per document, in order
    assert mixlib.texts_sha256(["a", "bc"]) == hashlib.sha256(b"a\0bc\0").hexdigest()
    assert mixlib.texts_sha256([]) == hashlib.sha256(b"").hexdigest()


def test_shuffle_is_frozen():
    # group validation files are shuffled with random.Random(str seed); a change here changes their hashes
    assert mixlib.shuffled(list(range(10)), "20261128:consumer") == [3, 0, 2, 6, 5, 7, 1, 4, 9, 8]


def test_m1_200b_estimate_frozen():
    spec = mixlib.load_spec(str(SPEC))
    w = mixlib.weights_from_spec(spec)
    assert len(w) == 20 and abs(sum(w.values()) - 1) < 1e-12
    assert abs(w[12] - 0.12883374) < 1e-8 and abs(w[15] - 0.01310219) < 1e-8
    est = mixlib.estimate(spec)
    src = est["sources"]
    assert abs(src[15]["epochs"] - 3.347) < 2e-3 and abs(src[14]["epochs"] - 3.286) < 2e-3
    assert sorted(est["repeated_sources"]) == [1, 2, 3, 5, 8, 9, 11, 13, 14, 15, 18, 19, 20]
    assert abs(est["repeated_tokens"] / 1e9 - 23.03) < 0.1
    assert all(r["contains_v1_50B"] for r in src.values())
    # the tail validation documents are never read by the v1 50B build (smallest unused tail: c15, ~133M)
    assert min(r["tail_unused_by_v1_tokens"] for r in src.values()) > 100e6
    assert abs(est["val_all_tokens"] / 1e6 - 10.01) < 0.05 and est["val_group_tokens"] == 100e6
    assert spec["_reference"]["val_all"]["docs"] == sum(c["val_docs"] for c in spec["_reference"]["clusters"].values())


def test_config_checks():
    spec = mixlib.load_spec(str(SPEC))
    bad = dict(spec)
    bad["_groups"] = [dict(g) for g in spec["_groups"]]
    bad["_groups"][0] = dict(bad["_groups"][0], clusters=[6, 7, 9, 16, 12])
    try:
        mixlib.check_config(bad)
        raise AssertionError("duplicate cluster not rejected")
    except ValueError:
        pass
    alpha = dict(spec, weights={"alpha": 0.5})
    wa = mixlib.weights_from_spec(alpha)
    wr = mixlib.weights_from_spec(spec)
    # the reference weights were computed with alpha 0.5 from the same sizes
    assert max(abs(wa[c] - wr[c]) for c in wr) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
