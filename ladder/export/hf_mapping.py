"""Megatron GPTModel state dict -> Hugging Face Llama / Qwen3 state dict and config (dense only).

Pure torch + stdlib so the layout code is testable without Megatron (ladder/tests/test_hf_export.py).
ladder/export/export_hf.py builds the Megatron model, loads the checkpoint and calls this.

Megatron layouts (TP = 1, as loaded by export_hf.py):
  linear_qkv.weight  [ng * (np/ng + 2) * hn, D]: per query group, its np/ng query heads, then k, then v
  linear_fc1.weight  [2 F, D] with SwiGLU: [gate; up] (the activation applies to the first half)
  norms              fused into the next linear by Transformer Engine (linear_qkv.layer_norm_weight,
                     linear_fc1.layer_norm_weight) or separate modules with the local spec
A SphereFace-R head (lm_head_feature_norm) is folded into lm_head.weight, so the HF model is a plain
LlamaForCausalLM / Qwen3ForCausalLM that transformers and vLLM load without remote code.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch

SKIP_SUFFIXES = ("_extra_state",)
SKIP_PREFIXES = ("rotary_pos_emb.",)


@dataclass(frozen=True)
class DenseSpec:
    """What the exporter needs from the Megatron config (TransformerConfig + args)."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    head_dim: int
    ffn_hidden_size: int
    vocab_size: int                  # rows of the (padded) embedding
    norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    qk_layernorm: bool               # Qwen3 (per-head RMSNorm on q and k) vs Llama
    head_mode: Optional[str] = None  # lm_head_feature_norm: None | nfn | hfn | sfn
    head_scale: Optional[float] = None


def split_qkv(weight: torch.Tensor, spec: DenseSpec):
    """Megatron fused QKV weight -> (q [np*hn, D], k [ng*hn, D], v [ng*hn, D])."""
    ng, hn = spec.num_query_groups, spec.head_dim
    per_group = spec.num_attention_heads // ng
    expected = (ng * (per_group + 2) * hn, spec.hidden_size)
    if tuple(weight.shape) != expected:
        raise ValueError(f"linear_qkv.weight has shape {tuple(weight.shape)}, expected {expected}")
    w = weight.reshape(ng, (per_group + 2) * hn, spec.hidden_size)
    q, k, v = torch.split(w, [per_group * hn, hn, hn], dim=1)
    return (
        q.reshape(ng * per_group * hn, spec.hidden_size),
        k.reshape(ng * hn, spec.hidden_size),
        v.reshape(ng * hn, spec.hidden_size),
    )


def merge_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: DenseSpec) -> torch.Tensor:
    """Inverse of split_qkv (for tests and HF -> Megatron)."""
    ng, hn = spec.num_query_groups, spec.head_dim
    per_group = spec.num_attention_heads // ng
    parts = (
        q.reshape(ng, per_group * hn, -1),
        k.reshape(ng, hn, -1),
        v.reshape(ng, hn, -1),
    )
    return torch.cat(parts, dim=1).reshape(-1, q.shape[-1])


def _pop_one(sd: Dict[str, torch.Tensor], *names: str) -> torch.Tensor:
    """Pop the first present key of ``names`` (Transformer Engine and local-spec names)."""
    for name in names:
        if name in sd:
            return sd.pop(name)
    raise KeyError(f"none of {names} in the Megatron state dict")


def megatron_to_hf(sd: Dict[str, torch.Tensor], spec: DenseSpec) -> Dict[str, torch.Tensor]:
    """Map a Megatron GPTModel state dict (TP = PP = 1) to Hugging Face Llama / Qwen3 names.

    The head is folded (feature-norm heads become plain rows). Every Megatron tensor must be
    consumed: unexpected parameters (biases, MoE, MTP, ...) raise instead of being dropped.
    """
    sd = {
        k: v
        for k, v in sd.items()
        if not k.endswith(SKIP_SUFFIXES) and not k.startswith(SKIP_PREFIXES) and v is not None
    }
    out = {"model.embed_tokens.weight": sd.pop("embedding.word_embeddings.weight")}
    for i in range(spec.num_layers):
        m = f"decoder.layers.{i}."
        h = f"model.layers.{i}."
        out[h + "input_layernorm.weight"] = _pop_one(
            sd, m + "self_attention.linear_qkv.layer_norm_weight", m + "input_layernorm.weight"
        )
        q, k, v = split_qkv(sd.pop(m + "self_attention.linear_qkv.weight"), spec)
        out[h + "self_attn.q_proj.weight"] = q
        out[h + "self_attn.k_proj.weight"] = k
        out[h + "self_attn.v_proj.weight"] = v
        out[h + "self_attn.o_proj.weight"] = sd.pop(m + "self_attention.linear_proj.weight")
        if spec.qk_layernorm:
            out[h + "self_attn.q_norm.weight"] = sd.pop(m + "self_attention.q_layernorm.weight")
            out[h + "self_attn.k_norm.weight"] = sd.pop(m + "self_attention.k_layernorm.weight")
        out[h + "post_attention_layernorm.weight"] = _pop_one(
            sd, m + "mlp.linear_fc1.layer_norm_weight", m + "pre_mlp_layernorm.weight"
        )
        fc1 = sd.pop(m + "mlp.linear_fc1.weight")
        if tuple(fc1.shape) != (2 * spec.ffn_hidden_size, spec.hidden_size):
            raise ValueError(f"{m}mlp.linear_fc1.weight has shape {tuple(fc1.shape)}, expected SwiGLU")
        gate, up = torch.chunk(fc1, 2, dim=0)
        out[h + "mlp.gate_proj.weight"] = gate
        out[h + "mlp.up_proj.weight"] = up
        out[h + "mlp.down_proj.weight"] = sd.pop(m + "mlp.linear_fc2.weight")
    final_norm = sd.pop("decoder.final_layernorm.weight")
    out["model.norm.weight"] = final_norm
    lm_head = sd.pop("output_layer.weight")
    if spec.head_mode is not None:
        lm_head = fold_head(lm_head, final_norm, spec)
    out["lm_head.weight"] = lm_head
    if sd:
        raise ValueError(f"Megatron tensors with no Hugging Face counterpart: {sorted(sd)}")
    if out["model.embed_tokens.weight"].shape[0] != spec.vocab_size:
        raise ValueError("embedding rows differ from spec.vocab_size")
    return out


def fold_head(weight: torch.Tensor, final_norm: torch.Tensor, spec: DenseSpec) -> torch.Tensor:
    """Plain lm_head rows with the logits of the feature-norm head (see token_margin.py).

    nfn / sfn: W_i / ||W_i|| (exact). hfn: (s / sqrt(D)) W_i / ||W_i||, exact up to the RMSNorm
    epsilon because ||h|| = sqrt(D) when the final norm weight is 1, which GPTModel enforces
    under hfn by freezing it; a weight that moved means the checkpoint was not trained that way.
    """
    rows = weight.float()
    rows = rows / torch.linalg.vector_norm(rows, dim=-1, keepdim=True).clamp_min(1.0e-6)
    if spec.head_mode == "hfn":
        drift = (final_norm.float() - 1.0).abs().max().item()
        if drift != 0.0:
            raise ValueError(
                f"hfn export needs the final norm weight at exactly 1 (max |w - 1| = {drift:g}); "
                "this checkpoint did not freeze it, so the head does not fold into a plain layer."
            )
        rows = rows * (spec.head_scale / math.sqrt(spec.hidden_size))
    elif spec.head_mode not in ("nfn", "sfn"):
        raise ValueError(f"unknown head mode {spec.head_mode!r}")
    return rows.to(weight.dtype)


def hf_config(spec: DenseSpec, *, bos_token_id, eos_token_id, dtype: str = "bfloat16") -> dict:
    """config.json for LlamaForCausalLM / Qwen3ForCausalLM.

    Written as a plain dict so it does not depend on the local transformers version. The RoPE base
    is given both as rope_theta (transformers < 5, vLLM) and rope_parameters (transformers >= 5).
    """
    cfg = {
        "vocab_size": spec.vocab_size,
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.ffn_hidden_size,
        "num_hidden_layers": spec.num_layers,
        "num_attention_heads": spec.num_attention_heads,
        "num_key_value_heads": spec.num_query_groups,
        "head_dim": spec.head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": spec.max_position_embeddings,
        "rms_norm_eps": spec.norm_eps,
        "rope_theta": spec.rope_theta,
        "rope_scaling": None,
        "rope_parameters": {"rope_type": "default", "rope_theta": spec.rope_theta},
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "initializer_range": 0.02,
        "use_cache": True,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "torch_dtype": dtype,
    }
    if spec.qk_layernorm:
        cfg.update(
            architectures=["Qwen3ForCausalLM"],
            model_type="qwen3",
            use_sliding_window=False,
            sliding_window=None,
            max_window_layers=spec.num_layers,
        )
    else:
        cfg.update(
            architectures=["LlamaForCausalLM"], model_type="llama", mlp_bias=False, pretraining_tp=1
        )
    return cfg


def save_safetensors(tensors: Dict[str, torch.Tensor], out_dir: str, max_shard_bytes: int) -> None:
    """Write tensors as (sharded) safetensors with the index file transformers expects."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    shards, current, size = [], {}, 0
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        nbytes = t.numel() * t.element_size()
        if current and size + nbytes > max_shard_bytes:
            shards.append(current)
            current, size = {}, 0
        current[name] = t
        size += nbytes
    shards.append(current)
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    if len(shards) == 1:
        save_file(shards[0], os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
        return
    weight_map = {}
    for i, shard in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(out_dir, fname), metadata={"format": "pt"})
        weight_map.update({name: fname for name in shard})
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)


def spec_dict(spec: DenseSpec) -> dict:
    return asdict(spec)
