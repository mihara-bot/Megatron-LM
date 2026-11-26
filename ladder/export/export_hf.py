"""Export a dense ladder checkpoint (Llama3 / Qwen3 shapes) to a Hugging Face model directory.

Launch through ladder/export/export_hf.sh (one GPU, TP = PP = 1). Steps:
  1. build the GPTModel from the Megatron arguments and load the checkpoint (torch_dist reshards);
     load_checkpoint refuses an LM-head setting that differs from the checkpoint's
  2. map the weights to Llama / Qwen3 names, folding a SphereFace-R head (lm_head_feature_norm)
     into a plain lm_head (ladder/export/hf_mapping.py)
  3. write safetensors, config.json, generation_config.json, the tokenizer and export_meta.json
  4. reload the directory with transformers and compare its logits with the Megatron model's on
     random token sequences (KL, top-1 agreement, max |logit diff|); fail above the thresholds
The result loads in transformers / vLLM as a plain model, no remote code.
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from hf_mapping import DenseSpec, hf_config, megatron_to_hf, save_safetensors, spec_dict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import unwrap_model
from megatron.inference.utils import get_model_builder, inference_config_from_args
from megatron.training import get_args
from megatron.training.arguments import parse_and_validate_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.global_vars import get_run_config, set_run_config
from megatron.training.initialize import initialize_megatron


def add_export_args(parser):
    group = parser.add_argument_group(title="HF export")
    group.add_argument("--hf-export-dir", type=str, required=True, help="Output directory.")
    group.add_argument("--hf-validate-seqs", type=int, default=4,
                       help="Random sequences for the Megatron vs HF logits check (0 skips it).")
    group.add_argument("--hf-validate-len", type=int, default=512, help="Their length in tokens.")
    group.add_argument("--hf-max-mean-kl", type=float, default=1e-2,
                       help="Fail if the mean per-token KL(Megatron || HF) exceeds this.")
    group.add_argument("--hf-min-top1", type=float, default=0.98,
                       help="Fail if the top-1 agreement is below this.")
    group.add_argument("--hf-max-shard-gb", type=float, default=5.0, help="Safetensors shard size.")
    return parser


def dense_spec(model, args) -> DenseSpec:
    """Read the shape from the built model and refuse anything this exporter does not map."""
    c = model.config
    problems = []
    if c.normalization != "RMSNorm":
        problems.append(f"normalization={c.normalization}")
    if c.layernorm_zero_centered_gamma:
        problems.append("apply_layernorm_1p")
    if not c.gated_linear_unit or c.activation_func is not F.silu:
        problems.append("MLP is not SwiGLU")
    if c.add_bias_linear or c.add_qkv_bias:
        problems.append("linear biases")
    if c.num_moe_experts or c.multi_latent_attention or c.mtp_num_layers:
        problems.append("MoE / MLA / MTP (use Megatron-Bridge)")
    if getattr(c, "attention_output_gate", False) or getattr(c, "qk_l2_norm", False):
        problems.append("gated attention / qk_l2_norm")
    if c.softmax_scale is not None or c.rotary_interleaved:
        problems.append("custom softmax_scale / interleaved RoPE")
    if args.position_embedding_type != "rope" or args.rotary_percent != 1.0:
        problems.append(f"position embedding {args.position_embedding_type} x {args.rotary_percent}")
    if getattr(args, "use_rope_scaling", False):
        problems.append("Llama-3.1 RoPE scaling")
    if model.share_embeddings_and_output_weights:
        problems.append("tied embeddings")
    if problems:
        raise SystemExit("export_hf.py maps plain dense Llama / Qwen3 only: " + ", ".join(problems))
    return DenseSpec(
        num_layers=c.num_layers,
        hidden_size=c.hidden_size,
        num_attention_heads=c.num_attention_heads,
        num_query_groups=c.num_query_groups or c.num_attention_heads,
        head_dim=c.kv_channels,
        ffn_hidden_size=c.ffn_hidden_size,
        vocab_size=model.embedding.word_embeddings.weight.shape[0],
        norm_eps=c.layernorm_epsilon,
        rope_theta=float(args.rotary_base),
        max_position_embeddings=args.max_position_embeddings,
        qk_layernorm=c.qk_layernorm,
        head_mode=c.lm_head_feature_norm,
        head_scale=c.lm_head_feature_norm_scale if c.lm_head_feature_norm == "hfn" else None,
    )


@torch.no_grad()
def compare_logits(megatron_model, out_dir, vocab, args) -> dict:
    """Megatron (with its own head) vs the exported HF model on the same random tokens."""
    from transformers import AutoModelForCausalLM

    gen = torch.Generator().manual_seed(1234)
    tokens = torch.randint(0, vocab, (args.hf_validate_seqs, args.hf_validate_len), generator=gen)
    tokens = tokens.cuda()
    position_ids = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
    meg = torch.cat([megatron_model(t[None], p[None], None).float() for t, p in zip(tokens, position_ids)])

    hf = AutoModelForCausalLM.from_pretrained(out_dir, torch_dtype=torch.bfloat16).cuda().eval()
    ref = torch.cat([hf(input_ids=t[None]).logits.float() for t in tokens])
    del hf
    ref = ref[..., : meg.shape[-1]]
    lp_m, lp_h = meg.log_softmax(-1), ref.log_softmax(-1)
    kl = (lp_m.exp() * (lp_m - lp_h)).sum(-1)
    return {
        "sequences": args.hf_validate_seqs,
        "length": args.hf_validate_len,
        "mean_kl": kl.mean().item(),
        "max_kl": kl.max().item(),
        "top1_agreement": (meg.argmax(-1) == ref.argmax(-1)).float().mean().item(),
        "max_abs_logit_diff": (meg - ref).abs().max().item(),
        "mean_abs_logit": meg.abs().mean().item(),
    }


def main():
    args = parse_and_validate_args(
        extra_args_provider=add_export_args,
        args_defaults={"no_load_rng": True, "no_load_optim": True, "exit_on_missing_checkpoint": True},
    )
    cfg = inference_config_from_args(args)
    set_run_config(cfg)
    initialize_megatron()
    args = get_args()
    if torch.distributed.get_world_size() != 1:
        raise SystemExit("run export_hf.py on one GPU (TP = PP = 1); torch_dist reshards on load")

    builder = get_model_builder(args, cfg.tokenizer, provider="gpt")
    model = builder.build_distributed_models(
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(), wrap_with_ddp=False
    )
    checkpoint = get_run_config().checkpoint
    checkpoint.exit_on_missing_checkpoint = True
    iteration, _ = load_checkpoint(model, None, None, strict=True)
    wrapped = model[0]
    wrapped.eval()
    gpt = unwrap_model(model)[0]
    spec = dense_spec(gpt, args)
    print(f"> loaded {checkpoint.load} iteration {iteration}: {spec}", flush=True)

    state = {k: v.detach().to("cpu", copy=True) for k, v in gpt.state_dict().items() if torch.is_tensor(v)}
    tensors = {k: v.to(torch.bfloat16) for k, v in megatron_to_hf(state, spec).items()}

    from transformers import AutoTokenizer

    out = args.hf_export_dir
    os.makedirs(out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    if len(tokenizer) > spec.vocab_size:
        raise SystemExit(f"tokenizer has {len(tokenizer)} tokens, the embedding {spec.vocab_size}")
    save_safetensors(tensors, out, int(args.hf_max_shard_gb * 2**30))
    config = hf_config(spec, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id}, f, indent=2)
    tokenizer.save_pretrained(out)

    meta = {
        "source_checkpoint": checkpoint.load,
        "iteration": iteration,
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": config["architectures"][0],
        "spec": spec_dict(spec),
        "lm_head": {
            "lm_head_feature_norm": args.lm_head_feature_norm,
            "lm_head_feature_norm_scale": args.lm_head_feature_norm_scale,
            "token_margin": args.token_margin,
            "token_margin_reverse": args.token_margin_reverse,
            "soft_feature_norm_weight": args.soft_feature_norm_weight,
            "fold": {
                None: "plain head, copied",
                "nfn": "lm_head = W / ||W|| (exact)",
                "sfn": "lm_head = W / ||W|| (exact)",
                "hfn": f"lm_head = (s / sqrt(D)) W / ||W|| = {spec.head_scale or 0:g} / "
                       f"{math.sqrt(spec.hidden_size):g} W / ||W|| (final norm weight frozen at 1)",
            }[args.lm_head_feature_norm],
            "note": "token_margin / soft_feature_norm_weight are the export call's settings; the "
                    "training run's are in its cmd_*.sh. The margin is training-only.",
        },
        "tokenizer_note": "ladder data is tokenized EOS-only (no BOS); the HF tokenizer may add BOS "
                          "by default. Evaluate every run with the same convention.",
    }
    if args.hf_validate_seqs > 0:
        check = compare_logits(wrapped, out, spec.vocab_size, args)
        meta["validation"] = check
        print(f"> Megatron vs HF logits: {json.dumps(check)}", flush=True)
    with open(os.path.join(out, "export_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if args.hf_validate_seqs > 0:
        check = meta["validation"]
        if check["mean_kl"] > args.hf_max_mean_kl or check["top1_agreement"] < args.hf_min_top1:
            raise SystemExit(f"logits check failed (see {out}/export_meta.json): {check}")
    print(f"> exported to {out}", flush=True)


if __name__ == "__main__":
    main()
