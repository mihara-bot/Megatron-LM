"""Check that the environment can run Megatron training for the ladder.

Prints versions, then checks: GPU / NCCL / bf16, Megatron Core import and the compiled dataset
helper, TransformerEngine (a bf16 TE Linear forward/backward on GPU), Muon from emerging_optimizers
as Megatron sees it, and optionally a HF tokenizer.

    python ladder/env/check_env.py [--tokenizer /path/to/Llama-3.1-8B]
"""

import argparse
import importlib
import sys
import traceback

RESULTS = []


def check(name):
    def wrap(fn):
        try:
            detail = fn()
            RESULTS.append((name, True, detail or ""))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn

    return wrap


def version(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        return f"missing ({type(e).__name__})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default=None)
    args = p.parse_args()

    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    for mod in ["torch", "transformer_engine", "megatron.core", "emerging_optimizers", "apex",
                "triton", "flash_attn", "transformers", "numpy", "pyarrow", "nvidia_resiliency_ext"]:
        print(f"  {mod:24s} {version(mod)}")

    @check("cuda / nccl / bf16")
    def _():
        import torch

        assert torch.cuda.is_available(), "no CUDA device visible"
        n = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0)
        return (f"{n} x {torch.cuda.get_device_name(0)} (sm{cap[0]}{cap[1]}), CUDA {torch.version.cuda}, "
                f"NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}, bf16 {torch.cuda.is_bf16_supported()}")

    @check("megatron.core + dataset helper")
    def _():
        import megatron.core
        from megatron.core.datasets import helpers  # noqa: F401  (compiled helpers_cpp)

        return f"megatron.core {megatron.core.__version__} from {megatron.core.__file__}"

    @check("transformer_engine bf16 Linear on GPU")
    def _():
        import torch
        import transformer_engine.pytorch as te

        layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16).cuda()
        x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        layer(x).float().pow(2).mean().backward()
        return "forward/backward ok"

    @check("triton CUDA driver (needs cuda.h)")
    def _():
        # Builds triton's cuda_utils helper, the step torch.compile hits first (Megatron's jit warmup).
        from triton.runtime import driver

        return str(driver.active.get_current_target())

    @check("Muon available to Megatron")
    def _():
        mod = importlib.import_module("megatron.core.optimizer.emerging_optimizers")
        have = [k for k in dir(mod) if k.startswith("HAVE_")]
        flags = {k: getattr(mod, k) for k in have}
        assert all(flags.values()), f"emerging_optimizers import failed inside Megatron: {flags}"
        from emerging_optimizers.orthogonalized_optimizers.muon import Muon  # noqa: F401

        return str(flags)

    if args.tokenizer:
        @check("HF tokenizer")
        def _():
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            ids = tok("hello world", add_special_tokens=False).input_ids
            return f"vocab {len(tok)}, eos {tok.eos_token!r}={tok.eos_token_id}, 'hello world' -> {ids}"

    print()
    for name, ok, detail in RESULTS:
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)


if __name__ == "__main__":
    main()
