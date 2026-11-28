"""Check the vLLM eval environment (ladder/env/setup_vllm_env.sh) as ladder/sft/eval.sh uses it.

    bash ladder/sft/eval/run_isolated.sh <vllm env>/bin/python ladder/env/check_vllm_env.py \
        [--gpu] [--tokenizer <HF tokenizer dir> | --model <HF model dir>]

Always: versions; torch / vllm / transformers / sympy imported from this environment (not from the NGC image or
the Megatron venv); sympy's LaTeX parser works (without it the MATH grader silently falls back to string
matching, visible only as sympy_equivalence = 0 in summary.json); eval_vllm.py and math_passk.yaml load.
--gpu: CUDA, a bf16 matmul, and with --tokenizer (a tiny random Llama built with that tokenizer) or --model a
short vLLM generation with eval_vllm.py's engine settings, greedy and sampled.
--lm-eval: lm-eval (ladder/eval/lm_eval.sh) is installed in this venv (an editable install points at the clone), its
vLLM backend imports, and transformers / tokenizers are still the versions the setup pinned (requirements.lock).
Last line: "vLLM env OK" or the failures; exit status 1 on failure.
"""

import argparse
import importlib
import os
import sys
import tempfile
import traceback
from importlib.metadata import version

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "ladder", "sft", "eval"))
sys.path.insert(0, os.path.join(REPO, "ladder", "sft"))
FAILED = []


def check(name, fn):
    try:
        detail = fn()
        print(f"[ OK ] {name}" + (f": {detail}" if detail else ""), flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        FAILED.append(name)


def own_modules():
    """Every package eval needs comes from this environment, not a leaked path."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(f"{sys.executable} is not a venv interpreter")
    cfg = open(os.path.join(sys.prefix, "pyvenv.cfg")).read().replace(" ", "").lower()
    if "include-system-site-packages=true" in cfg:
        raise RuntimeError("the venv sees the system site-packages (built with --system-site-packages)")
    out, bad, missing = [], [], []
    for mod in ["torch", "triton", "vllm", "transformers", "sympy", "yaml", "pyarrow"]:
        try:
            m = importlib.import_module(mod)
        except ImportError as e:
            missing.append(f"{mod} ({e})")
            continue
        path = os.path.realpath(m.__file__)
        if not path.startswith(os.path.realpath(sys.prefix) + os.sep):
            bad.append(f"{mod} from {path}")
        out.append(f"{mod} {getattr(m, '__version__', '?')}")
    if missing:
        raise RuntimeError("cannot import: " + "; ".join(missing) + (f" (found: {', '.join(out)})" if out else ""))
    if bad:
        raise RuntimeError("imported from outside the venv (PYTHONPATH leak?): " + "; ".join(bad))
    import torch

    return ", ".join(out) + f"; torch CUDA {torch.version.cuda}"


def sympy_latex():
    import sympy
    from math_grader import sympy_available
    from sympy.parsing.latex import parse_latex

    if not sympy_available():
        raise RuntimeError("math_grader.sympy_available() is False (antlr4-python3-runtime missing or not 4.11)")
    if sympy.simplify(parse_latex(r"\frac{1}{2}") - sympy.Rational(1, 2)) != 0:
        raise RuntimeError(r"parse_latex(\frac{1}{2}) != 1/2")
    return f"antlr4-python3-runtime {version('antlr4-python3-runtime')}"


def eval_config():
    import eval_vllm

    cfg = eval_vllm.load_config(os.path.join(REPO, "ladder/sft/eval/math_passk.yaml"), "/data")
    return f"{len(cfg['tasks'])} tasks in math_passk.yaml"


def lm_eval_pkg():
    """lm_eval from this venv's site-packages (editable: the clone), its vLLM backend, pins unchanged."""
    import importlib.util

    if importlib.util.find_spec("lm_eval") is None:
        raise RuntimeError(f"lm_eval not installed: cd ../lm-evaluation-harness && {sys.executable} -m pip install -e . "
                           f"-c {os.path.join(sys.prefix, 'requirements.lock')}")
    importlib.import_module("lm_eval.models.vllm_causallms")
    lock = os.path.join(sys.prefix, "requirements.lock")
    moved = []
    if os.path.isfile(lock):
        pins = dict(line.strip().lower().split("==", 1) for line in open(lock) if "==" in line)
        for pkg in ("transformers", "tokenizers", "torch", "vllm"):
            if pkg in pins and version(pkg) != pins[pkg]:
                moved.append(f"{pkg} {pins[pkg]} -> {version(pkg)}")
    if moved:
        raise RuntimeError("installing lm-eval moved pinned packages (reinstall with -c requirements.lock): "
                           + "; ".join(moved))
    spec = importlib.util.find_spec("lm_eval")
    return f"lm_eval {version('lm_eval')} from {os.path.dirname(os.path.dirname(spec.origin))}, vllm backend imports"


def gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    err = ((a @ a).float() - a.float() @ a.float()).abs().max().item()
    if not err < 1.0:
        raise RuntimeError(f"bf16 matmul error {err}")
    try:
        import pynvml  # nvidia-ml-py, a vLLM dependency

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
    except Exception:  # noqa: BLE001
        driver = "?"
    detail = f"{torch.cuda.get_device_name()} (driver {driver}), bf16 matmul ok"
    # run_isolated.sh replaced the launcher's GPU UUIDs by indices: device 0 must still be the allocated GPU.
    expected = os.environ.get("CUDA_VISIBLE_DEVICES_UUIDS", "").split(",")[0].strip()
    if expected:
        got = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
        norm = lambda u: u.lower().removeprefix("gpu-")  # noqa: E731
        if not got:
            detail += f"; allocated {expected} (torch reports no uuid to compare)"
        elif norm(got) != norm(expected):
            raise RuntimeError(f"device 0 is GPU-{norm(got)}, but the job was given {expected} "
                               f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        else:
            detail += f"; device 0 = allocated {expected} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
    return detail


def tiny_llama(tokenizer_dir, out_dir):
    """A 2-layer random Llama (head_dim 128, GQA, untied head) with the given tokenizer, saved in bf16."""
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=256, intermediate_size=896, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                      rope_theta=500000.0, rms_norm_eps=1e-5, tie_word_embeddings=False,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def generate(model_dir):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, tokenizer=model_dir, dtype="bfloat16", seed=0, tensor_parallel_size=1,
              gpu_memory_utilization=0.5, enable_prefix_caching=True)
    prompts = [{"prompt_token_ids": tok(p, add_special_tokens=False)["input_ids"]}
               for p in ["Question: what is 2 + 3?\nAnswer:", "The capital of France is"]]
    greedy = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=16, ignore_eos=True),
                          use_tqdm=False)
    sampled = llm.generate(prompts, SamplingParams(n=4, temperature=0.6, top_p=0.95, max_tokens=16, ignore_eos=True),
                           use_tqdm=False)
    for out in greedy + sampled:
        if not out.outputs or any(len(c.token_ids) != 16 for c in out.outputs):
            raise RuntimeError("a completion does not have 16 tokens")
    if len(sampled[0].outputs) != 4:
        raise RuntimeError("n=4 returned a different number of samples")
    import vllm

    return f"vllm {vllm.__version__}: greedy {greedy[0].outputs[0].text!r}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--lm-eval", action="store_true", help="also check lm-eval (ladder/eval/lm_eval.sh)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tokenizer", help="build a tiny random Llama with this HF tokenizer and generate with it")
    g.add_argument("--model", help="an HF model directory to generate with (e.g. an export of ladder/sft/export.sh)")
    args = p.parse_args()

    print(f"python {sys.version.split()[0]} ({sys.executable})")
    leaked = [k for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV") if os.environ.get(k)]
    if leaked:
        print(f"note: {', '.join(leaked)} set; ladder/sft/eval/run_isolated.sh removes them (eval.sh uses it)")
    check("packages from this venv", own_modules)
    check("sympy LaTeX parser (MATH equivalence)", sympy_latex)
    check("eval_vllm.py + math_passk.yaml", eval_config)
    if args.lm_eval:
        check("lm-eval", lm_eval_pkg)
    if args.gpu:
        check("CUDA", gpu)
        if args.model:
            check(f"vLLM generation ({args.model})", lambda: generate(args.model))
        elif args.tokenizer:
            with tempfile.TemporaryDirectory(prefix="tiny_llama_") as d:
                check("tiny random Llama", lambda: tiny_llama(args.tokenizer, d))
                if "tiny random Llama" not in FAILED:
                    check("vLLM generation (tiny random Llama)", lambda: generate(d))
    elif args.model or args.tokenizer:
        print("note: --model / --tokenizer only run with --gpu")

    if FAILED:
        print("vLLM env FAILED: " + "; ".join(FAILED))
        sys.exit(1)
    print("vLLM env OK" + ("" if args.gpu else " (CPU checks only; add --gpu in a GPU job)"))


if __name__ == "__main__":
    main()
