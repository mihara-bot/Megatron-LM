"""Core of the text-mixing tool: which documents of every source go into a mix (stdlib only).

Model
-----
A *source* (one ClimbMix cluster) is an ordered list of parquet files (sorted by name). Its *stream*
is the non-empty values of the text column in file order: null and empty texts are skipped
everywhere, as ladder/data/tokenize_dataset.py skips them. A position is (file index, row).

A mix cuts every stream into three parts:

    [head val: the first k documents] [train region] [tail val: the last documents, >= T bytes]

k comes from the reference build (the v1 validation set; the v1 builder used the same stream
definition). The tail is cut backwards from the end, and the document that reaches T is included.
Training reads the train region from its start, document by document, until the quota Q bytes is
reached. The document that crosses Q is included, as in the v1 builder. When the region holds
fewer than Q bytes, the reading starts over from the region start (pass e1, e2, ...), so a source
is repeated only when its quota needs it, and its validation documents are never read. The v1
training documents are therefore the first documents of pass e0.

A plan turns each pass into whole-file copies plus row slices (only at the head, tail and crossing
files), which parquet_io executes. Token counts are estimates: bytes / bytes-per-token of the
reference tokenizer. tokenize_dataset.py gives the exact counts.

Nothing here reads parquet. Data comes through a reader with
    index(f) -> (docs, bytes), lengths(f) -> per-row text byte lengths (0 = null/empty),
    texts(f, lo, hi) -> raw row values,
so the logic is testable with in-memory files (tests/test_mixlib.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PLAN_VERSION = 1

Range = Tuple[int, int, int]  # (file index, first row, end row), rows [lo, hi) of one file


@dataclass
class SourceFile:
    """One parquet file of a source, as its footer describes it."""

    name: str
    path: str
    size: int
    rows: int
    rg_rows: List[int] = field(default_factory=list)
    footer_bytes: int = 0  # text bytes estimated from the footer (uncompressed size - 4 bytes per row)

    @property
    def stem(self) -> str:
        return self.name[: -len(".parquet")] if self.name.endswith(".parquet") else self.name


@dataclass
class Cut:
    """A run of documents at the head or tail of a stream."""

    pos: Tuple[int, int]  # head: first position after the run; tail: first position of the run
    ranges: List[Range]
    docs: int
    bytes: int


# ----------------------------------------------------------------------------- small helpers


def texts_sha256(texts: Sequence[str]) -> str:
    """Content hash of documents in order (the v1 builder's definition: utf-8 text + NUL per document)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def nonempty(values: Sequence[Optional[str]]) -> List[str]:
    return [t for t in values if t]


def source_dirname(sid: int) -> str:
    return f"cluster_{int(sid):02d}"


def human(x: float) -> str:
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.3g}{unit}"
    return f"{x:.3g}"


def write_json_atomic(path: Path, obj: Any) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def shuffled(items: List[Any], seed: str) -> List[Any]:
    """Deterministic shuffle (random.Random with a str seed is hashed with SHA-512, so it does not
    depend on PYTHONHASHSEED)."""
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


# ----------------------------------------------------------------------------- configuration


def load_spec(path: str) -> Dict[str, Any]:
    """The mix spec, with its reference and groups files loaded (paths relative to the spec)."""
    spec_path = Path(path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = spec_path.parent
    ref_path = (base / spec["reference"]).resolve()
    groups_path = (base / spec["groups"]).resolve()
    ref_text = ref_path.read_text(encoding="utf-8")
    groups_text = groups_path.read_text(encoding="utf-8")
    spec["_paths"] = {"spec": str(spec_path), "reference": str(ref_path), "groups": str(groups_path)}
    spec["_reference"] = json.loads(ref_text)
    spec["_groups"] = json.loads(groups_text)["groups"]
    spec["_hashes"] = {
        "spec": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "reference": hashlib.sha256(ref_text.encode("utf-8")).hexdigest(),
        "groups": hashlib.sha256(groups_text.encode("utf-8")).hexdigest(),
    }
    check_config(spec)
    return spec


def check_config(spec: Dict[str, Any]) -> None:
    ref = spec["_reference"]["clusters"]
    sids = sorted(int(c) for c in ref)
    seen: Dict[int, str] = {}
    slugs = set()
    for g in spec["_groups"]:
        if not re.fullmatch(r"[a-z0-9_]+", g["slug"]) or g["slug"] == "all" or g["slug"] in slugs:
            raise ValueError(f"group slug {g['slug']!r} must be unique, [a-z0-9_]+ and not 'all'")
        slugs.add(g["slug"])
        for c in g["clusters"]:
            if int(c) in seen:
                raise ValueError(f"cluster {c} is in groups {seen[int(c)]!r} and {g['slug']!r}")
            seen[int(c)] = g["slug"]
    if sorted(seen) != sids:
        raise ValueError(f"groups must cover the reference clusters {sids} once each, got {sorted(seen)}")
    w = weights_from_spec(spec)
    if abs(sum(w.values()) - 1.0) > 1e-9:
        raise ValueError("weights do not sum to 1")
    if spec["train_tokens"] <= 0 or spec.get("val_tail_tokens_per_cluster", 0) < 0:
        raise ValueError("train_tokens must be positive and val_tail_tokens_per_cluster non-negative")


def weights_from_spec(spec: Dict[str, Any]) -> Dict[int, float]:
    """Training-token share of every source: the reference weights, n_i^alpha, or explicit weights."""
    ref = spec["_reference"]["clusters"]
    rule = spec["weights"]
    if rule == "reference":
        raw = {int(c): float(v["weight"]) for c, v in ref.items()}
    elif isinstance(rule, dict) and "alpha" in rule:
        a = float(rule["alpha"])
        raw = {int(c): (v["source_bytes"] / v["bytes_per_token"]) ** a for c, v in ref.items()}
    elif isinstance(rule, dict) and "explicit" in rule:
        raw = {int(c): 0.0 for c in ref}
        for c, v in rule["explicit"].items():
            if int(c) not in raw:
                raise ValueError(f"explicit weight for unknown cluster {c}")
            raw[int(c)] = float(v)
    else:
        raise ValueError(f"weights must be 'reference', {{'alpha': a}} or {{'explicit': {{...}}}}, got {rule!r}")
    if any(v < 0 for v in raw.values()) or sum(raw.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    total = sum(raw.values())
    return {c: v / total for c, v in sorted(raw.items())}


def group_of(spec: Dict[str, Any]) -> Dict[int, str]:
    return {int(c): g["slug"] for g in spec["_groups"] for c in g["clusters"]}


# ----------------------------------------------------------------------------- estimate (no data)


def estimate(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Quotas, passes and validation sizes from the reference numbers alone (footer byte estimates)."""
    ref = spec["_reference"]["clusters"]
    w = weights_from_spec(spec)
    n_tok = float(spec["train_tokens"])
    tail_tok = float(spec.get("val_tail_tokens_per_cluster", 0))
    rows = {}
    for c in sorted(w):
        r = ref[str(c)]
        bpt = r["bytes_per_token"]
        quota = w[c] * n_tok * bpt
        avail = r["source_bytes"] - r["val_bytes"] - tail_tok * bpt
        rows[c] = {
            "weight": w[c], "bytes_per_token": bpt, "quota_tokens": quota / bpt,
            "available_tokens": avail / bpt, "epochs": quota / avail if avail > 0 else math.inf,
            "repeated_tokens": max(0.0, quota - avail) / bpt,
            "contains_v1_50B": quota >= r.get("train_bytes_50B", 0),
            "tail_unused_by_v1_tokens": (avail + tail_tok * bpt - r.get("train_bytes_50B", 0)) / bpt,
        }
    groups = []
    for g in spec["_groups"]:
        groups.append({"slug": g["slug"], "clusters": g["clusters"], "tokens": tail_tok * len(g["clusters"])})
    return {
        "sources": rows,
        "train_tokens": n_tok,
        "repeated_tokens": sum(r["repeated_tokens"] for r in rows.values()),
        "repeated_sources": sorted((c for c, r in rows.items() if r["epochs"] > 1), key=lambda c: -rows[c]["epochs"]),
        "val_all_tokens": sum(spec["_reference"]["clusters"][str(c)]["val_bytes"] / rows[c]["bytes_per_token"]
                              for c in rows),
        "val_groups": groups,
        "val_group_tokens": tail_tok * len(rows),
    }


def print_estimate(spec: Dict[str, Any], est: Dict[str, Any]) -> None:
    ref = spec["_reference"]["clusters"]
    grp = group_of(spec)
    print(f"[estimate] {spec['name']}: {human(est['train_tokens'])} training tokens, "
          f"{len(est['sources'])} sources (reference footer bytes; exact numbers come from `plan`)")
    print(f"{'src':>4} {'name':34} {'group':15} {'weight%':>8} {'quota':>8} {'avail':>8} {'passes':>7} "
          f"{'repeat':>8} {'v1-50B':>7}")
    for c, r in est["sources"].items():
        print(f"{c:4d} {ref[str(c)]['name'][:34]:34} {grp[c]:15} {100 * r['weight']:8.3f} "
              f"{human(r['quota_tokens']):>8} {human(r['available_tokens']):>8} {r['epochs']:7.3f} "
              f"{human(r['repeated_tokens']):>8} {'in' if r['contains_v1_50B'] else 'NOT in':>7}")
    print(f"[estimate] repeated tokens {human(est['repeated_tokens'])} "
          f"({100 * est['repeated_tokens'] / est['train_tokens']:.1f}%), sources read more than once: "
          f"{len(est['repeated_sources'])}")
    print(f"[estimate] val/all (v1) {human(est['val_all_tokens'])} tokens; per-group val "
          f"{human(est['val_group_tokens'])} tokens: "
          + ", ".join(f"{g['slug']} {human(g['tokens'])}" for g in est["val_groups"]))


# ----------------------------------------------------------------------------- stream cuts


def head_cut(files: Sequence[SourceFile], reader: Any, k: int) -> Cut:
    """The first k non-empty documents; pos = the position right after the k-th."""
    if k <= 0:
        return Cut((0, 0), [], 0, 0)
    need, nbytes, ranges = k, 0, []
    for i, f in enumerate(files):
        lens = reader.lengths(f)
        for r, n in enumerate(lens):
            if n > 0:
                need -= 1
                nbytes += n
                if need == 0:
                    ranges.append((i, 0, r + 1))
                    return Cut((i, r + 1), ranges, k, nbytes)
        ranges.append((i, 0, f.rows))
    raise ValueError(f"the source has fewer than {k} non-empty documents")


def tail_cut(files: Sequence[SourceFile], reader: Any, target_bytes: float) -> Cut:
    """The last documents holding at least target_bytes (the one that reaches it included);
    pos = the position of the first of them."""
    if target_bytes <= 0:
        return Cut((len(files) - 1, files[-1].rows), [], 0, 0)
    acc, docs, ranges = 0, 0, []
    for i in range(len(files) - 1, -1, -1):
        lens = reader.lengths(files[i])
        for r in range(len(lens) - 1, -1, -1):
            n = lens[r]
            if n > 0:
                acc += n
                docs += 1
                if acc >= target_bytes:
                    ranges.insert(0, (i, r, files[i].rows))
                    return Cut((i, r), ranges, docs, acc)
        ranges.insert(0, (i, 0, files[i].rows))
    raise ValueError(f"the source holds fewer than {target_bytes:.0f} text bytes")


def region_ranges(files: Sequence[SourceFile], start: Tuple[int, int], end: Tuple[int, int]) -> List[Range]:
    """Row ranges from position start (included) to end (excluded), one per file, empty ones dropped."""
    (fs, s), (fe, e) = start, end
    out = []
    for i in range(fs, fe + 1):
        lo = s if i == fs else 0
        hi = e if i == fe else files[i].rows
        if lo < hi:
            out.append((i, lo, hi))
    return out


def range_stats(files: Sequence[SourceFile], reader: Any, rng: Range) -> Tuple[int, int]:
    """(non-empty documents, text bytes) of a row range; whole files come from the index."""
    i, lo, hi = rng
    f = files[i]
    if lo == 0 and hi == f.rows:
        return reader.index(f)
    seg = reader.lengths(f)[lo:hi]
    return sum(1 for n in seg if n > 0), sum(seg)


def _walk(files, reader, region, consumed, quota):
    """One pass over the region from its start: whole ranges while below the quota, then the range
    holding the crossing document up to and including it. Returns (taken, consumed, reached)."""
    taken = []
    for i, lo, hi in region:
        d, b = range_stats(files, reader, (i, lo, hi))
        if consumed + b < quota:
            taken.append((i, lo, hi, d, b))
            consumed += b
            continue
        lens = reader.lengths(files[i])
        d2 = b2 = 0
        for r in range(lo, hi):
            n = lens[r]
            if n > 0:
                d2 += 1
                b2 += n
                if consumed + b2 >= quota:
                    taken.append((i, lo, r + 1, d2, b2))
                    return taken, consumed + b2, True
        raise AssertionError("range bytes and lengths disagree")
    return taken, consumed, False


def plan_source(sid: int, files: Sequence[SourceFile], reader: Any, *, head_docs: int, tail_bytes: float,
                quota_bytes: float, max_epochs: float, must_contain_docs: int = 0) -> Dict[str, Any]:
    """Cuts, passes and file operations of one source (see the module docstring)."""
    head = head_cut(files, reader, head_docs)
    tail = tail_cut(files, reader, tail_bytes)
    if head.pos > tail.pos:  # equal = no training documents between them
        raise ValueError(f"source {sid}: the head and tail validation sets overlap")
    region = region_ranges(files, head.pos, tail.pos)

    passes: List[List[Tuple[int, int, int, int, int]]] = []
    consumed, region_docs, region_bytes = 0, None, None
    if quota_bytes > 0:
        taken, consumed, reached = _walk(files, reader, region, 0, quota_bytes)
        passes.append(taken)
        if not reached:  # the whole region was read: repeat it
            region_bytes = consumed
            region_docs = sum(t[3] for t in taken)
            if region_bytes <= 0:
                raise ValueError(f"source {sid}: no training text between the validation sets")
            if quota_bytes / region_bytes > max_epochs * (1 + 1e-12):
                raise ValueError(f"source {sid}: quota needs {quota_bytes / region_bytes:.2f} passes over its "
                                 f"training documents, more than max_epochs {max_epochs:g}")
            while consumed + region_bytes < quota_bytes:
                passes.append(list(passes[0]))
                consumed += region_bytes
            taken, consumed, reached = _walk(files, reader, region, consumed, quota_bytes)
            assert reached
            passes.append(taken)
    if must_contain_docs and (not passes or sum(t[3] for t in passes[0]) < must_contain_docs):
        got = sum(t[3] for t in passes[0]) if passes else 0
        raise ValueError(f"source {sid}: pass e0 has {got} documents, fewer than the {must_contain_docs} "
                         "reference training documents it must contain (raise the quota)")

    ops = []
    dirname = source_dirname(sid)
    for p, taken in enumerate(passes):
        for i, lo, hi, d, b in taken:
            f = files[i]
            ops.append({
                "source": sid, "pass": p, "kind": "copy" if (lo == 0 and hi == f.rows) else "slice",
                "src": f.path, "src_name": f.name, "src_size": f.size, "src_rows": f.rows,
                "dst": f"train/{dirname}/e{p}_{f.stem}.parquet", "lo": lo, "hi": hi, "docs": d, "bytes": b,
            })
    return {
        "head": head, "tail": tail, "region_docs": region_docs, "region_bytes": region_bytes,
        "passes": len(passes), "train_docs": sum(o["docs"] for o in ops), "train_bytes": consumed,
        "pass0_docs": sum(t[3] for t in passes[0]) if passes else 0, "ops": ops,
    }


def likely_files(files: Sequence[SourceFile], quota_bytes: float, tail_bytes: float, margin: int = 2) -> List[int]:
    """Files whose exact index a plan will probably need, from footer estimates: everything when the
    source is likely repeated, else the files up to the estimated crossing plus `margin`."""
    total = sum(f.footer_bytes for f in files)
    if quota_bytes >= 0.95 * (total - tail_bytes):
        return list(range(len(files)))
    acc, out = 0, []
    for i, f in enumerate(files):
        out.append(i)
        acc += f.footer_bytes
        if acc >= quota_bytes:
            break
    last = out[-1]
    return list(range(min(len(files), last + 1 + margin)))


def cut_texts(files: Sequence[SourceFile], reader: Any, cut: Cut) -> List[str]:
    """The documents of a cut, in stream order."""
    out: List[str] = []
    for i, lo, hi in cut.ranges:
        out.extend(nonempty(reader.texts(files[i], lo, hi)))
    if len(out) != cut.docs:
        raise AssertionError(f"read {len(out)} documents, the cut has {cut.docs}")
    return out


def cut_json(files: Sequence[SourceFile], cut: Cut) -> Dict[str, Any]:
    return {"ranges": [{"file": files[i].name, "src": files[i].path, "lo": lo, "hi": hi} for i, lo, hi in cut.ranges],
            "docs": cut.docs, "bytes": cut.bytes}


def fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
