"""Check a copied text dataset against its files.sha256 (after the transfer, before tokenizing).

    python data_mixing/verify_files.py /fast/users/zhuangxinlin/data/climbmix_mix/m1_200b --workers 128

Checks that _DONE exists (verify finished on the source side), that every listed file exists with its
sha256, and that train/ and val/ hold no unlisted files. Stdlib only; hashes in parallel processes.
Exit status 1 on any problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def sha256_file(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_listing(root: str):
    path = os.path.join(root, "files.sha256")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found: was `mix_text.py verify` run before the copy?")
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digest, rel = line.rstrip("\n").split("  ", 1)
                entries.append((rel, digest))
    return entries


def verify(root: str, workers: int) -> list:
    problems = []
    done = os.path.join(root, "_DONE")
    if not os.path.isfile(done):
        problems.append("_DONE missing: the source-side verify did not finish, or the copy is incomplete")
    entries = read_listing(root)
    listed = {rel for rel, _ in entries}
    for d in ("train", "val"):
        for r, _, names in os.walk(os.path.join(root, d)):
            for n in names:
                rel = os.path.relpath(os.path.join(r, n), root)
                if rel not in listed:
                    problems.append(f"not listed: {rel}")
    present = []
    for rel, digest in entries:
        if os.path.isfile(os.path.join(root, rel)):
            present.append((rel, digest))
        else:
            problems.append(f"missing: {rel}")
    total = sum(os.path.getsize(os.path.join(root, rel)) for rel, _ in present)
    t0, last, done_bytes = time.monotonic(), time.monotonic(), 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(sha256_file, os.path.join(root, rel)): (rel, digest) for rel, digest in present}
        for fut in as_completed(futures):
            rel, digest = futures[fut]
            if fut.result() != digest:
                problems.append(f"sha256 mismatch: {rel}")
            done_bytes += os.path.getsize(os.path.join(root, rel))
            if time.monotonic() - last > 30:
                last = time.monotonic()
                rate = done_bytes / max(last - t0, 1e-9)
                print(f"[verify_files] {done_bytes / 1e9:.1f} / {total / 1e9:.1f} GB, {rate / 1e6:.0f} MB/s", flush=True)
    if os.path.isfile(done):
        with open(done) as f:
            info = json.load(f)
        if info.get("files") != len(entries):
            problems.append(f"_DONE lists {info.get('files')} files, files.sha256 {len(entries)}")
    print(f"[verify_files] {len(entries)} files, {total / 1e9:.1f} GB checked in {time.monotonic() - t0:.0f}s, "
          f"{len(problems)} problems", flush=True)
    return problems


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="the copied dataset directory (holds files.sha256 and _DONE)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args(argv)
    problems = verify(args.root, args.workers)
    for x in problems[:30]:
        print(f"[verify_files] {x}")
    if problems:
        sys.exit(1)
    print("[verify_files] ok")


if __name__ == "__main__":
    main()
