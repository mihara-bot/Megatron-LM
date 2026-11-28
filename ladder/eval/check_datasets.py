"""Check that lm-eval tasks load offline from EVAL_DATA_DIR, the way ladder/eval/lm_eval.sh runs them.

Run through ladder/eval/check_datasets.sh, which uses the vLLM environment's python and runs from EVAL_DATA_DIR with the
Hub offline (no GPU needed). The installed lm-eval decides which repo each task reads, and that can differ between
lm-eval versions: social_iqa reads allenai/social_i_qa in one and lighteval/siqa in another. The check works like this:
  1. Expand the requested tasks into leaf tasks: groups and tags go through the installed lm-eval's task index.
  2. Build each leaf task exactly as lm-eval does. Building a task loads its dataset.
  3. Report each task as one of:
     OK       the dataset loaded (with its eval docs count)
     MISSING  its repo is not under EVAL_DATA_DIR, i.e. the Hub was needed
     FAILED   any other error: a script-only repo, a missing config, a truncated file, ...
  4. For each repo directory, report whether hfd.sh finished it:
     - no *.aria2 control files;
     - aria2c's saved session is empty;
     - every file in wget's list is present.
It ends with the download_datasets.sh command for the missing and unfinished repos. Exit status is 1 unless all is well.

The helpers below use only the standard library, so the tests import this file without lm-eval. main() imports lm-eval.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MISSING_PATTERNS = (
    r"Couldn't reach '([^']+)' on the Hub",
    r"Dataset '([^']+)' doesn't exist on the Hub",
)
EMPTY_DIR_PATTERN = r"Couldn't find (?:a dataset script|any data files?) at (\S+?)\.?(?:\s|'|$)"


def missing_repo(message: str, data_dir: str = "") -> Optional[str]:
    """The repo id in a datasets error that means "not available locally" (absent from EVAL_DATA_DIR, or a directory
    with no data in it), else None."""
    for pat in MISSING_PATTERNS:
        m = re.search(pat, message)
        if m:
            return m.group(1)
    m = re.search(EMPTY_DIR_PATTERN, message)
    if m:
        path = m.group(1)
        if data_dir and os.path.isabs(path):
            rel = os.path.relpath(path, data_dir)
            if not rel.startswith(".."):
                path = rel
        parts = [x for x in path.split("/") if x]
        return "/".join(parts[:2]) if len(parts) >= 2 else path
    return None


def hint(message: str) -> str:
    """A short explanation of common datasets failures."""
    if "scripts are no longer supported" in message or "dataset script" in message.lower():
        return ("script-only repo (datasets >= 4 cannot run it): download its refs/convert/parquet branch, as "
                "download_datasets.sh does for allenai/social_i_qa")
    if "BuilderConfig" in message and "not found" in message:
        return "the local copy lacks this config (README configs?)"
    if "parquet" in message.lower() or "ArrowInvalid" in message:
        return "unreadable data file (truncated download?)"
    return ""


def hfd_state(repo_dir: str) -> Optional[str]:
    """Why hfd.sh's download of a repo directory is unfinished, or None (finished, or not an hfd download)."""
    root = Path(repo_dir)
    if not root.is_dir():
        return "absent"
    problems = []
    partial = [p for p in root.rglob("*.aria2") if ".hfd" not in p.parts]
    if partial:
        problems.append(f"{len(partial)} partial file(s) (*.aria2), e.g. {partial[0].relative_to(root)}")
    session = root / ".hfd" / "aria2c_urls.txt"
    if session.is_file():
        unfinished = [line for line in session.read_text().splitlines() if line.startswith("http")]
        if unfinished:   # aria2c --save-session keeps only what did not finish
            problems.append(f"aria2c session lists {len(unfinished)} unfinished file(s)")
    listing = root / ".hfd" / "wget_urls.txt"
    if listing.is_file():
        lost = []
        for url in listing.read_text().split():
            if "/resolve/" not in url:
                continue
            rel = url.split("/resolve/", 1)[1].split("/", 1)[-1]   # <revision>/<path in repo>
            f = root / rel
            if not f.is_file() or f.stat().st_size == 0:
                lost.append(rel)
        if lost:
            problems.append(f"{len(lost)} listed file(s) missing or empty, e.g. {lost[0]}")
    return "; ".join(problems) or None


def _entry_info(entry: Any) -> Tuple[str, Any]:
    """(kind, cfg) of a task-index entry: an Entry dataclass (new lm-eval) or a dict (older)."""
    if isinstance(entry, dict):
        kind = str(entry.get("type", "")).lower()
        cfg = entry
    else:
        kind = str(getattr(getattr(entry, "kind", None), "name", getattr(entry, "kind", ""))).lower()
        cfg = getattr(entry, "cfg", None) or {}
    return kind, cfg


def leaves(index: Dict[str, Any], name: str, depth: int = 0) -> List[str]:
    """Leaf task names of a task, group or tag, from lm-eval's task index (best effort: a group whose members are
    inline configs stays one item, loaded as a whole)."""
    entry = index.get(name)
    if entry is None or depth > 8:
        return [name]
    kind, cfg = _entry_info(entry)
    if "group" in kind:
        members = cfg.get("task") if isinstance(cfg, dict) else None
        if not isinstance(members, list) or not all(isinstance(m, str) for m in members):
            return [name]
        out: List[str] = []
        for m in members:
            out.extend(leaves(index, m, depth + 1))
        return out
    if "tag" in kind:
        members = []
        for other, e in index.items():
            tags = e.get("tags", ()) if isinstance(e, dict) else getattr(e, "tags", ())
            if isinstance(tags, str):
                tags = [tags]
            if name in (tags or ()) and other != name:
                members.append(other)
        return sorted(members) or [name]
    return [name]


def task_objects(obj: Any) -> Iterable[Any]:
    """Task instances inside what TaskManager.load_task_or_group returns (nested dicts / lists)."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from task_objects(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from task_objects(v)
    elif hasattr(obj, "config"):
        yield obj


def describe(task: Any) -> str:
    cfg = getattr(task, "config", None)
    path = getattr(cfg, "dataset_path", None) or getattr(task, "DATASET_PATH", None) or "?"
    name = getattr(cfg, "dataset_name", None) or getattr(task, "DATASET_NAME", None)
    try:
        docs = f", {len(task.eval_docs)} eval docs"
    except Exception:  # noqa: BLE001
        docs = ""
    return f"{path}{'/' + name if name else ''}{docs}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", required=True, help="comma-separated lm-eval tasks / groups / tags")
    p.add_argument("--data-dir", required=True, help="EVAL_DATA_DIR (download_datasets.sh)")
    args = p.parse_args(argv)
    data_dir = os.path.abspath(args.data_dir)
    os.chdir(data_dir)   # load_dataset("<org>/<name>") takes the local directory at that relative path
    for var in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(var) != "1":
            print(f"note: {var} is not 1; a repo missing here could be fetched from the Hub instead of reported",
                  flush=True)

    logging.basicConfig(level=logging.ERROR)
    for name in ("lm_eval", "lm-eval", "datasets", "evaluate", "absl"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import datasets

        datasets.logging.set_verbosity_error()
        datasets.disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass
    import lm_eval
    from lm_eval.tasks import TaskManager

    try:
        from importlib.metadata import version

        lm_eval_version = version("lm_eval")
    except Exception:  # noqa: BLE001
        lm_eval_version = "?"
    print(f"lm_eval {lm_eval_version} from {os.path.dirname(os.path.dirname(lm_eval.__file__))}; "
          f"data {data_dir}", flush=True)
    tm = TaskManager()
    index = getattr(tm, "task_index", None) or {}
    requested = [t for t in args.tasks.split(",") if t]
    missing: Dict[str, List[str]] = {}
    failed: List[Tuple[str, str]] = []
    repos_seen: Dict[str, None] = {}
    n_ok = 0
    for top in requested:
        if index and top not in index:
            print(f"[UNKN] {top}: not a task, group or tag of this lm-eval", flush=True)
            failed.append((top, "unknown task"))
            continue
        for leaf in leaves(index, top):
            t0 = time.time()
            try:
                built = tm.load_task_or_group([leaf])
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}".strip()
                repo = missing_repo(msg, data_dir)
                label = leaf if leaf == top else f"{top}/{leaf}"
                if repo:
                    missing.setdefault(repo, []).append(label)
                    repos_seen[repo] = None
                    print(f"[MISS] {label}: {repo} is not under {data_dir}", flush=True)
                else:
                    first = msg.splitlines()[0][:300]
                    extra = hint(msg)
                    failed.append((label, first))
                    print(f"[FAIL] {label}: {first}" + (f"\n       -> {extra}" if extra else ""), flush=True)
                continue
            for task in task_objects(built):
                path = getattr(getattr(task, "config", None), "dataset_path", None)
                if path:
                    repos_seen[path] = None
            descs = sorted({describe(t) for t in task_objects(built)})
            n_ok += 1
            label = leaf if leaf == top else f"{top}/{leaf}"
            print(f"[ OK ] {label}: {'; '.join(descs) if len(descs) <= 3 else f'{len(descs)} subtasks'} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if any(t.startswith(("humaneval", "mbpp")) for t in requested):
        if not (Path(data_dir) / "code_eval" / "code_eval.py").is_file():
            failed.append(("code_eval", "metric script missing"))
            print(f"[MISS] code_eval metric: {data_dir}/code_eval/code_eval.py (EVAL_METRICS=code_eval)", flush=True)

    unfinished = {}
    for repo in repos_seen:
        d = Path(data_dir) / repo
        if d.is_dir():
            state = hfd_state(str(d))
            if state:
                unfinished[repo] = state
    for repo, state in unfinished.items():
        print(f"[PART] {repo}: hfd.sh download unfinished: {state}", flush=True)

    print(f"\n{n_ok} task(s) OK, {sum(len(v) for v in missing.values())} missing, {len(failed)} failed, "
          f"{len(unfinished)} unfinished repo(s)")
    redo = list(missing) + [r for r in unfinished if r not in missing]
    if redo:
        print("download / resume them with:\n"
              f"  EVAL_DATASETS=\"{' '.join(redo)}\" bash ladder/eval/download_datasets.sh")
    if failed:
        print("failed tasks need a look (see the messages above): " + ", ".join(t for t, _ in failed))
    return 1 if (missing or failed or unfinished) else 0


if __name__ == "__main__":
    sys.exit(main())
