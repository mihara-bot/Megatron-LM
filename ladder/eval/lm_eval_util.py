"""Helpers of ladder/eval/lm_eval.sh. Standard library only, so any python3 runs them (preflight: the vLLM one).

    python3 ladder/eval/lm_eval_util.py model-args <model dir> [--tokens auto|eos|default]   -> "<convention> [<args>]"
    <vllm python> ladder/eval/lm_eval_util.py preflight <model dir> [--allow-hf-version-mismatch]
    python3 ladder/eval/lm_eval_util.py collect --model <dir> --out <eval dir> --tasks a,b [--summary <jsonl>] ...

model-args: the token convention, as extra lm-eval --model_args. Ladder pretraining data is tokenized EOS-only
(text + EOS, no BOS; ladder/data/tokenize_dataset.py) and SFT data the same way, so with "eos" lm-eval adds no BOS
(the Llama-3 tokenizer would prepend <|begin_of_text|>, a token these models never saw) and uses EOS, the token
that precedes every document in training, as its prefix token: the context of empty-context requests and the
first token of loglikelihood_rolling (wikitext). Without prefix_token_id the vLLM backend takes bos_token_id.
The EOS id is config.json's eos_token_id, which ladder/export/export_hf.py writes from the training tokenizer.
"auto" (default): "eos" for a ladder export (it has export_meta.json), lm-eval's own default for other models.

preflight: lm_eval and vllm are installed in this interpreter's environment, and its transformers / tokenizers
are the versions of the Megatron environment the model was exported in (export_meta.json hf_versions;
ladder/env/setup_vllm_env.sh pins them, and a later `pip install -e` of lm-eval could move them).

collect: the newest results_<date>.json lm-eval wrote in <eval dir> -> <eval dir>/results.json, and one line in
--summary (default <eval dir>/../summary.jsonl): model, run / tag / iteration (ladder exports), LM head, settings,
every metric as "<task>/<metric>" (the ",none" filter suffix dropped), "score/<task>" for each requested task
(acc_norm, else acc, else exact_match, else the first metric) and "avg_score": the mean of the requested tasks'
scores that are accuracies.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Tuple

HF_PACKAGES = ("transformers", "tokenizers")
ACCURACY_METRICS = ("acc_norm", "acc", "exact_match")


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---- model-args ---------------------------------------------------------------------------------
def token_args(model_dir: str, tokens: str = "auto") -> Tuple[str, str]:
    """(convention used, extra --model_args) for a model directory."""
    if tokens == "auto":
        tokens = "eos" if os.path.isfile(os.path.join(model_dir, "export_meta.json")) else "default"
    if tokens == "default":
        return "default", ""
    if tokens != "eos":
        raise SystemExit(f"--tokens must be auto, eos or default, got {tokens!r}")
    config = read_json(os.path.join(model_dir, "config.json"))
    if config is None:
        raise SystemExit(f"{model_dir}/config.json not found")
    eos = config.get("eos_token_id")
    if isinstance(eos, list) and len(eos) == 1:
        eos = eos[0]
    if not isinstance(eos, int):
        raise SystemExit(f"{model_dir}/config.json: eos_token_id is {eos!r}; the eos convention needs one id")
    return "eos", f"add_bos_token=False,prefix_token_id={eos}"


# ---- preflight ----------------------------------------------------------------------------------
def local_versions(packages=HF_PACKAGES) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def version_diff(model_dir: str, local: Dict[str, Optional[str]]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """(the export's hf_versions or None, ["pkg A (export) vs B (here)", ...]); None for a model without the record."""
    meta = read_json(os.path.join(model_dir, "export_meta.json")) or {}
    exported = meta.get("hf_versions")
    if not exported:
        return None, []
    return dict(exported), [f"{pkg} {exported.get(pkg)} (export) vs {local.get(pkg)} (here)"
                            for pkg in HF_PACKAGES if exported.get(pkg) != local.get(pkg)]


def preflight(model_dir: str, allow_mismatch: bool) -> str:
    import importlib.util

    problems = []
    for mod in ("lm_eval", "vllm"):
        if importlib.util.find_spec(mod) is None:
            problems.append(mod)
    if problems:
        hint = ""
        if "lm_eval" in problems:
            lock = os.path.join(sys.prefix, "requirements.lock")
            hint = (f" Install lm-eval into this environment (never the Megatron one): {sys.executable} -m pip install "
                    f"-e <lm-evaluation-harness> -c {lock}  (ladder/EVAL.md)")
        raise SystemExit(f"{', '.join(problems)} not installed in {sys.prefix} ({sys.executable}).{hint}")
    local = local_versions()
    exported, diff = version_diff(model_dir, local)
    if diff:
        msg = ("transformers / tokenizers differ from the Megatron environment the model was exported in: "
               + "; ".join(diff) + ". Rebuild the vLLM environment (ladder/env/setup_vllm_env.sh), reinstall lm-eval "
               "with -c <vllm env>/requirements.lock, or set ALLOW_HF_MISMATCH=1")
        if not allow_mismatch:
            raise SystemExit(msg)
        print(f"[lm_eval] warning: {msg}", flush=True)
    spec = importlib.util.find_spec("lm_eval")
    where = os.path.dirname(os.path.dirname(spec.origin)) if spec and spec.origin else "?"
    vers = local_versions(("lm_eval", "vllm"))
    note = "" if exported else " (no hf_versions in export_meta.json: reference model or older export)"
    return (f"lm_eval {vers['lm_eval']} ({where}), vllm {vers['vllm']}, transformers {local['transformers']}, "
            f"tokenizers {local['tokenizers']}{note}")


# ---- collect ------------------------------------------------------------------------------------
def flatten_metrics(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """{"<task>/<metric>": value} over lm-eval's results["results"]; stderr kept as <metric>_stderr."""
    out: Dict[str, float] = {}
    for task, entry in results.items():
        for key, value in entry.items():
            if key == "alias" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            metric, _, filt = key.partition(",")
            name = metric if filt in ("", "none") else f"{metric},{filt}"
            out[f"{task}/{name}"] = float(value)
    return out


def task_score(metrics: Dict[str, float], task: str) -> Optional[Tuple[str, float]]:
    """(metric, value): acc_norm, else acc, else exact_match (any filter), else the task's first metric."""
    names = [k.split("/", 1)[1] for k in metrics if k.startswith(task + "/") and "_stderr" not in k]
    for want in ACCURACY_METRICS:
        for name in names:
            if name == want or name.startswith(want + ","):
                return name, metrics[f"{task}/{name}"]
    return (names[0], metrics[f"{task}/{names[0]}"]) if names else None


def name_tag_iter(model_dir: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """(run, tag, iteration) of <run dir>/hf/<tag>_iter<N> (ladder/eval/export.sh) or <sft dir>/hf/iter_<N>."""
    parent, name = os.path.split(model_dir.rstrip("/"))
    run = os.path.basename(os.path.dirname(parent)) if os.path.basename(parent) == "hf" else None
    m = re.fullmatch(r"(?:(.+)_)?iter_?(\d+)", name)
    if not m:
        return run, None, None
    return run, m.group(1), int(m.group(2))


def collect(model_dir: str, out_dir: str, tasks: List[str], summary: Optional[str], extra: Dict[str, Any]) -> Dict:
    files = sorted(glob.glob(os.path.join(out_dir, "results_*.json")))
    if not files:
        raise SystemExit(f"no results_*.json in {out_dir} (lm-eval failed?)")
    with open(files[-1]) as f:
        res = json.load(f)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    metrics = flatten_metrics(res.get("results", {}))
    for name, entry in (res.get("groups") or {}).items():   # older lm-eval keeps group scores apart
        metrics.update({k: v for k, v in flatten_metrics({name: entry}).items() if k not in metrics})
    scores: Dict[str, float] = {}
    score_metric: Dict[str, str] = {}
    missing = []
    for task in tasks:
        got = task_score(metrics, task)
        if got is None:
            missing.append(task)
            continue
        score_metric[task], scores[task] = got
    accuracies = [v for t, v in scores.items() if score_metric[t].split(",")[0] in ACCURACY_METRICS]
    meta = read_json(os.path.join(model_dir, "export_meta.json")) or {}
    run, tag, iteration = name_tag_iter(model_dir)
    head = meta.get("lm_head") or {}
    row = {
        "model": os.path.abspath(model_dir),
        "name": os.path.basename(model_dir.rstrip("/")),
        "run": run,
        "tag": tag,
        "iteration": meta.get("iteration", iteration),
        "source_checkpoint": meta.get("source_checkpoint"),
        "lm_head": head.get("lm_head_feature_norm"),
        "token_margin": head.get("token_margin"),
        "token_margin_reverse": head.get("token_margin_reverse"),
        "output": os.path.abspath(out_dir),
        "results_file": os.path.basename(files[-1]),
        "tasks": tasks,
        **extra,
        "avg_score": sum(accuracies) / len(accuracies) if accuracies else None,
        "avg_score_tasks": len(accuracies),
        "score_metric": score_metric,
        **{f"score/{t}": v for t, v in scores.items()},
        **metrics,
    }
    if missing:
        row["missing_tasks"] = missing
        print(f"[lm_eval] warning: no metric for {missing} in {files[-1]}", file=sys.stderr)
    dest = summary or os.path.join(os.path.dirname(os.path.abspath(out_dir)), "summary.jsonl")
    with open(dest, "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("model-args")
    a.add_argument("model")
    a.add_argument("--tokens", default="auto")
    b = sub.add_parser("preflight")
    b.add_argument("model")
    b.add_argument("--allow-hf-version-mismatch", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tasks", required=True, help="comma-separated, as given to lm-eval")
    c.add_argument("--summary", default=None)
    c.add_argument("--task-set", default=None)
    c.add_argument("--num-fewshot", default=None)
    c.add_argument("--limit", default=None)
    c.add_argument("--tokens", default=None, help="token convention used (recorded)")
    args = p.parse_args(argv)
    if args.cmd == "model-args":
        convention, extra = token_args(args.model, args.tokens)
        print(f"{convention} {extra}".rstrip())
    elif args.cmd == "preflight":
        print(f"[lm_eval] {preflight(args.model, args.allow_hf_version_mismatch)}", flush=True)
    else:
        extra = {"task_set": args.task_set, "num_fewshot": None if args.num_fewshot in (None, "") else int(args.num_fewshot),
                 "limit": None if args.limit in (None, "") else float(args.limit), "tokens": args.tokens}
        row = collect(args.model, args.out, [t for t in args.tasks.split(",") if t], args.summary, extra)
        scores = {k[len("score/"):]: round(v, 4) for k, v in row.items() if k.startswith("score/")}
        avg = row["avg_score"]
        print(f"[lm_eval] {row['name']}: avg_score {avg if avg is None else round(avg, 4)} over "
              f"{row['avg_score_tasks']} tasks; {scores}")


if __name__ == "__main__":
    main()
