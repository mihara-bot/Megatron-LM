"""Offline pass@k on GSM8K / MATH-500 / AMC-2023 / Countdown of an exported HF model, with vLLM.

The vLLM counterpart of pretrain_LLM_pipeline's offline eval (src/eval/math_passk.py, src/cli/eval_math_passk.py):
same tasks, prompts, graders (math_grader.py, copied), pass@k estimator and summary keys; settings in
math_passk.yaml next to this file. Run on the HF directory written by ladder/export/export_hf.sh (a plain
Llama / Qwen3; a feature-norm head is already folded into lm_head):

    python ladder/sft/eval/eval_vllm.py --model <hf dir> --data-root /fast/users/zhuangxinlin/data \\
        --output-dir <dir> [--tasks gsm8k math500] [--num-samples 16] [--max-problems 100]

Prompts are tokenized here without special tokens (no BOS), as in SFT training and in the pipeline, and given
to vLLM as token ids, so the tokenizer's own BOS default never applies. Each problem gets num_samples sampled
completions (temperature / top_p) and, with greedy, one greedy completion. A task's stop strings end the
completion (vLLM leaves them out of the text, as the pipeline's cut does); a completion is finished when it hit
EOS or a stop string within max_new_tokens. vLLM samples differently from HF generate, so per-sample results
are not identical to the pipeline's, only equal in distribution.

Output (--output-dir): <task>.jsonl (one record per completion, with gold), summary.json (metrics and settings).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from math_grader import GRADERS, countdown_gold, gsm8k_gold, pass_at_k, pass_k_values, sympy_available  # noqa: E402
from prepare_sft_data import PROMPT_TEMPLATE, render_template  # noqa: E402


@dataclass(frozen=True)
class EvalTask:
    name: str
    path: str
    grader: str
    problem_field: str
    gold_field: str
    expected_size: int
    max_new_tokens: int
    problem_template: str = ""
    gold_fields: Tuple[str, ...] = ()
    prompt_template: str = ""
    stop: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MathProblem:
    task: str
    index: int
    prompt: str
    gold: str


def load_config(path: str, data_root: str) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        raw = yaml.safe_load(f)
    tasks = []
    for name, t in raw["tasks"].items():
        grader = str(t.get("grader", "math"))
        if grader not in GRADERS:
            raise ValueError(f"task {name}: grader must be one of {sorted(GRADERS)}, got {grader!r}")
        gold_fields = tuple(str(x) for x in (t.get("gold_fields") or ()))
        if grader == "countdown" and set(gold_fields) != {"nums", "target"}:
            raise ValueError(f"task {name}: the countdown grader needs gold_fields: [nums, target]")
        tasks.append(EvalTask(
            name=str(name),
            path=str(t["path"]).replace("{data_root}", data_root),
            grader=grader,
            problem_field=str(t.get("problem_field", "problem")),
            gold_field=str(t.get("gold_field", "answer")),
            expected_size=int(t.get("expected_size", 0)),
            max_new_tokens=int(t.get("max_new_tokens", 1024)),
            problem_template=str(t.get("problem_template") or ""),
            gold_fields=gold_fields,
            prompt_template=str(t.get("prompt_template") or ""),
            stop=tuple(str(s) for s in (t.get("stop") or ()) if str(s)),
        ))
    sampling = raw.get("sampling") or {}
    return {
        "tasks": tasks,
        "prompt_template": str(raw.get("prompt_template") or PROMPT_TEMPLATE),
        "temperature": float(sampling.get("temperature", 0.6)),
        "top_p": float(sampling.get("top_p", 0.95)),
        "seed": int(sampling.get("seed", 1234)),
        "greedy": bool(raw.get("greedy", True)),
        "num_samples": int(raw.get("num_samples", 64)),
        "max_problems": int(raw.get("max_problems", 0)),
    }


def read_docs(path: str) -> List[Dict]:
    """A .jsonl / .json / .parquet file, or every such file under a directory (hidden entries skipped)."""
    if os.path.isdir(path):
        files = []
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files.extend(os.path.join(root, n) for n in names
                         if not n.startswith(".") and n.endswith((".jsonl", ".json", ".parquet")))
        return [doc for f in sorted(files) for doc in read_docs(f)]
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if text.lstrip().startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def problems_from_docs(task: EvalTask, docs: Sequence[Mapping], prompt_template: str,
                       max_problems: int) -> List[MathProblem]:
    """pretrain_LLM_pipeline src/eval/math_passk.py problems_from_docs."""
    if task.expected_size and len(docs) != task.expected_size:
        raise ValueError(f"task {task.name}: {task.path} has {len(docs)} problems, expected {task.expected_size}")
    if max_problems > 0:
        docs = docs[:max_problems]
    needed = set(task.gold_fields) if task.gold_fields else {task.gold_field}
    if not task.problem_template:
        needed.add(task.problem_field)
    out = []
    for i, doc in enumerate(docs):
        missing = sorted(f for f in needed if f not in doc)
        if missing:
            raise KeyError(f"task {task.name}: row {i} of {task.path} lacks {missing}; it has {sorted(doc)}")
        if task.grader == "countdown":
            gold = countdown_gold(doc["nums"], doc["target"])
        elif task.gold_fields:
            gold = json.dumps({f: doc[f] for f in task.gold_fields}, sort_keys=True, default=str)
        elif task.grader == "gsm8k":
            gold = gsm8k_gold(str(doc[task.gold_field]))
        else:
            gold = str(doc[task.gold_field])
        problem = render_template(task.problem_template, doc) if task.problem_template else doc[task.problem_field]
        prompt = render_template(task.prompt_template or prompt_template, {"problem": problem})
        out.append(MathProblem(task=task.name, index=i, prompt=prompt, gold=gold))
    return out


def summarize(results: Sequence[Mapping[str, Any]], problems: Mapping[str, Sequence[MathProblem]],
              num_samples: int, greedy: bool) -> Dict[str, float]:
    """pretrain_LLM_pipeline src/eval/math_passk.py summarize (same keys)."""
    metrics: Dict[str, float] = {}
    for task, task_problems in problems.items():
        rows = [r for r in results if r["task"] == task]
        n_prob = len(task_problems)
        metrics[f"{task}/num_problems"] = float(n_prob)
        sampled = [r for r in rows if r["sample"] >= 0]
        if num_samples > 0:
            correct = {p.index: 0 for p in task_problems}
            counts = {p.index: 0 for p in task_problems}
            for r in sampled:
                counts[r["index"]] += 1
                correct[r["index"]] += int(bool(r["correct"]))
            if any(c != num_samples for c in counts.values()):
                raise RuntimeError(f"eval {task}: expected {num_samples} samples for every problem")
            for k in pass_k_values(num_samples):
                metrics[f"{task}/pass@{k}"] = sum(pass_at_k(num_samples, c, k) for c in correct.values()) / n_prob
            if any("fallback" in r for r in sampled):
                strict = sum(int(bool(r["correct"]) and not r.get("fallback")) for r in sampled)
                metrics[f"{task}/strict_pass@1"] = strict / (n_prob * num_samples)
        if greedy:
            g = [r for r in rows if r["sample"] < 0]
            if len(g) != n_prob:
                raise RuntimeError(f"eval {task}: expected one greedy completion per problem")
            metrics[f"{task}/greedy_acc"] = sum(int(bool(r["correct"])) for r in g) / n_prob
        if rows:
            metrics[f"{task}/finish_rate"] = sum(int(bool(r["finished"])) for r in rows) / len(rows)
            metrics[f"{task}/boxed_rate"] = sum(int(bool(r["boxed"])) for r in rows) / len(rows)
            metrics[f"{task}/mean_tokens"] = sum(int(r["num_tokens"]) for r in rows) / len(rows)
            if any("fallback" in r for r in rows):
                metrics[f"{task}/fallback_rate"] = sum(int(bool(r.get("fallback"))) for r in rows) / len(rows)
    return metrics


def grade_completion(task: EvalTask, problem: MathProblem, sample: int, text: str, num_tokens: int,
                     finish_reason: Optional[str]) -> Dict[str, Any]:
    graded = GRADERS[task.grader](text, problem.gold)
    return {"task": task.name, "index": problem.index, "sample": sample, "num_tokens": int(num_tokens),
            "finished": finish_reason == "stop", **graded, "text": text}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="exported HF model directory")
    parser.add_argument("--config", default=os.path.join(HERE, "math_passk.yaml"))
    parser.add_argument("--data-root", required=True, help="replaces {data_root} in the task paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=None, help="subset of the config's tasks")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--no-greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.data_root)
    tasks = [t for t in cfg["tasks"] if args.tasks is None or t.name in args.tasks]
    if args.tasks and len(tasks) != len(set(args.tasks)):
        raise SystemExit(f"unknown task in {args.tasks}; the config has {[t.name for t in cfg['tasks']]}")
    num_samples = cfg["num_samples"] if args.num_samples is None else args.num_samples
    max_problems = cfg["max_problems"] if args.max_problems is None else args.max_problems
    greedy = cfg["greedy"] and not args.no_greedy
    seed = cfg["seed"] if args.seed is None else args.seed
    problems = {t.name: problems_from_docs(t, read_docs(t.path), cfg["prompt_template"], max_problems)
                for t in tasks}

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tokenizer=args.model, dtype=args.dtype, seed=seed,
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True)
    start = time.time()
    results: List[Dict[str, Any]] = []
    for task in tasks:
        task_problems = problems[task.name]
        prompts = [{"prompt_token_ids": tokenizer(p.prompt, add_special_tokens=False)["input_ids"]}
                   for p in task_problems]
        stop = list(task.stop) or None
        runs = []
        if greedy:
            runs.append((-1, SamplingParams(n=1, temperature=0.0, max_tokens=task.max_new_tokens, stop=stop)))
        if num_samples > 0:
            runs.append((0, SamplingParams(n=num_samples, temperature=cfg["temperature"], top_p=cfg["top_p"],
                                           max_tokens=task.max_new_tokens, stop=stop)))
        for first_sample, params in runs:
            t0 = time.time()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            for problem, out in zip(task_problems, outputs):
                for j, comp in enumerate(out.outputs):
                    sample = -1 if first_sample < 0 else j
                    results.append(grade_completion(task, problem, sample, comp.text, len(comp.token_ids),
                                                    comp.finish_reason))
            kind = "greedy" if first_sample < 0 else f"{num_samples} samples"
            print(f"[eval] {task.name} {kind}: {len(task_problems)} problems, {time.time() - t0:.0f}s", flush=True)

    metrics = summarize(results, problems, num_samples, greedy)
    metrics["eval_seconds"] = time.time() - start
    metrics["sympy_equivalence"] = float(sympy_available())
    os.makedirs(args.output_dir, exist_ok=True)
    by_key = {(p.task, p.index): p for ps in problems.values() for p in ps}
    for task in tasks:
        rows = sorted((r for r in results if r["task"] == task.name), key=lambda r: (r["index"], r["sample"]))
        with open(os.path.join(args.output_dir, f"{task.name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({**r, "gold": by_key[(task.name, r["index"])].gold}, ensure_ascii=False) + "\n")
    import vllm

    summary = {"metrics": metrics, "model": os.path.abspath(args.model), "backend": "vllm",
               "vllm_version": getattr(vllm, "__version__", "?"), "num_samples": num_samples, "greedy": greedy,
               "max_problems": max_problems, "temperature": cfg["temperature"], "top_p": cfg["top_p"],
               "seed": seed, "tasks": [t.name for t in tasks]}
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.4f}")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
