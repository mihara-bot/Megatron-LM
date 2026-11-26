"""SphereFace-R LM head + token-margin loss (megatron/core/models/gpt/token_margin.py).

    pytest tests/unit_tests/models/test_gpt_token_margin.py -p no:cacheprovider -o addopts=""

The closed-form tests run on the CPU. The GPTModel tests need one GPU (Transformer Engine spec):
- eval logits equal the paper's head written out from the final-norm output h and the raw weight;
- the training loss per token and the gradients of every parameter equal the explicit formula
  (target logit lowered by m, plus the sfn penalty), for nfn / hfn / sfn, d = 0, d > 0, reverse,
  with the unfused and the fused cross entropy;
- hfn freezes the final norm weight; the output layer runs without gradient-accumulation fusion;
- the metrics report exists after a training forward only, and is cleared on read;
- ladder/export/hf_mapping.py turns the model into a Hugging Face Llama / Qwen3 with the same
  logits, with the feature-norm head folded into lm_head (needs transformers + safetensors).
"""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt.token_margin import (
    head_geometry,
    normalize_rows,
    pop_token_margin_report,
    token_margin_loss,
    token_margin_objective,
    token_margin_scale,
)
from megatron.core.transformer.transformer_config import TransformerConfig

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


# ---- closed form (CPU) ----------------------------------------------------------------------------
def _modified_logit_ce(logits, labels, margin):
    """Cross entropy after lowering each target logit by its margin (the definition)."""
    onehot = F.one_hot(labels, logits.shape[-1]).to(logits.dtype)
    return F.cross_entropy(logits - margin[:, None] * onehot, labels, reduction="none")


@pytest.mark.parametrize("scale", [0.0, 0.5, 3.0, -2.0, 10.0])
def test_closed_form_matches_modified_logits(scale):
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(64, 50, generator=gen, dtype=torch.float64) * 3).requires_grad_()
    labels = torch.randint(0, 50, (64,), generator=gen)
    margin = scale * (0.5 + torch.rand(64, generator=gen, dtype=torch.float64))
    ref = _modified_logit_ce(logits, labels, margin)
    (g_ref,) = torch.autograd.grad(ref.sum(), logits)
    ce = F.cross_entropy(logits, labels, reduction="none")
    closed = token_margin_loss(ce, margin)
    (g_closed,) = torch.autograd.grad(closed.sum(), logits)
    torch.testing.assert_close(closed, ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(g_closed, g_ref, rtol=1e-8, atol=1e-10)
    # r scales the cross-entropy gradient per token
    (g_ce,) = torch.autograd.grad(
        F.cross_entropy(logits, labels, reduction="none").sum(), logits
    )
    r = token_margin_scale(ce.detach(), margin)
    torch.testing.assert_close(g_closed, r[:, None] * g_ce, rtol=1e-8, atol=1e-10)


def test_closed_form_at_zero_loss_is_finite():
    losses = torch.tensor([0.0, 1e-12, -1e-9], dtype=torch.float64, requires_grad=True)
    out = token_margin_loss(losses, 2.0)
    (grad,) = torch.autograd.grad(out.sum(), losses)
    assert torch.isfinite(out).all() and torch.isfinite(grad).all()
    assert out[0].item() == 0.0


def _cfg(mode, d=0.0, reverse=False, scale=60.0, t=1e-3):
    return SimpleNamespace(
        lm_head_feature_norm=mode,
        lm_head_feature_norm_scale=scale,
        token_margin=d,
        token_margin_reverse=reverse,
        soft_feature_norm_weight=t,
    )


@pytest.mark.parametrize("mode", ["nfn", "hfn", "sfn"])
@pytest.mark.parametrize("d,reverse", [(0.0, False), (0.05, False), (0.05, True)])
def test_objective_and_report(mode, d, reverse):
    gen = torch.Generator().manual_seed(1)
    ce = torch.rand(2, 8, generator=gen) * 4
    norm = 30 + 20 * torch.rand(2, 8, generator=gen)
    mask = (torch.rand(2, 8, generator=gen) > 0.3).float()
    cfg = _cfg(mode, d, reverse, scale=50.0, t=2e-3)
    loss, report = token_margin_objective(ce, norm, mask, cfg)
    m = (-d if reverse else d) * (50.0 if mode == "hfn" else norm.double())
    expect = token_margin_loss(ce.double(), m)
    if mode == "sfn":
        expect = expect + 2e-3 * (norm.double() - 50.0) ** 2
    torch.testing.assert_close(loss, expect.float())
    n = mask.sum()
    torch.testing.assert_close(report["ce loss"], torch.stack([(ce * mask).sum(), n]))
    m_mean = (torch.as_tensor(m).expand_as(ce).float() * mask).sum()
    torch.testing.assert_close(report["margin logit m"], torch.stack([m_mean, n]))
    assert ("sfn sq dev" in report) == (mode == "sfn")
    if d == 0.0:
        torch.testing.assert_close(report["margin r"], torch.stack([n, n]))


def test_head_geometry():
    assert head_geometry("nfn", 60) == head_geometry("sfn", 30) == ("cos",)
    assert head_geometry("hfn", 60) != head_geometry("hfn", 40)
    assert head_geometry(None, 60) == ("plain",) != head_geometry("nfn", None)


def _tcfg(**kw):
    base = dict(num_layers=2, hidden_size=64, num_attention_heads=4, normalization="RMSNorm")
    base.update(kw)
    return TransformerConfig(**base)


@pytest.mark.parametrize(
    "kw",
    [
        dict(token_margin=0.1),                                  # margin without a head
        dict(lm_head_feature_norm="hfn", token_margin_reverse=True),  # reverse with d = 0
        dict(lm_head_feature_norm="nfn", token_margin=-0.1),
        dict(lm_head_feature_norm="hfn", lm_head_feature_norm_scale=0.0),
        dict(lm_head_feature_norm="hfn", normalization="LayerNorm"),
        dict(lm_head_feature_norm="sfn", mtp_num_layers=1),
    ],
)
def test_config_rejects(kw):
    with pytest.raises(ValueError):
        _tcfg(**kw)


# ---- GPTModel (GPU) ----------------------------------------------------------------------------
VOCAB, SEQ, BATCH = 96, 16, 2


def _build(mode, d=0.0, reverse=False, scale=20.0, t=1e-3, qk_layernorm=False, fused_ce=False):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        qk_layernorm=qk_layernorm,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        embedding_init_method_std=1.0,   # residual rms ~ 1, so the hfn fold's eps error is ~1e-6
        cross_entropy_loss_fusion=fused_ce,
        lm_head_feature_norm=mode,
        lm_head_feature_norm_scale=scale,
        token_margin=d,
        token_margin_reverse=reverse,
        soft_feature_norm_weight=t,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=qk_layernorm),
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        position_embedding_type="rope",
        rotary_base=10000,
    ).cuda()
    with torch.no_grad():   # spread the output rows' norms so row normalization matters
        model.output_layer.weight.mul_(
            torch.rand(VOCAB, 1, device="cuda", generator=torch.Generator("cuda").manual_seed(5)) + 0.5
        )
    return model


def _inputs():
    gen = torch.Generator("cuda").manual_seed(7)
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda", generator=gen)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), device="cuda", generator=gen)
    mask = (torch.rand(BATCH, SEQ, device="cuda", generator=gen) > 0.25).float()
    position_ids = torch.arange(SEQ, device="cuda").expand(BATCH, SEQ)
    return tokens, position_ids, labels, mask


def _reference_logits(model, h):
    """Paper's head from the final-norm output h [s, b, D] and the raw weight: [s, b, V]."""
    cfg = model.config
    w = model.output_layer.weight
    h_norm = torch.linalg.vector_norm(h.float(), dim=-1, keepdim=True)
    cos = (h.float() / h_norm) @ (w.float() / torch.linalg.vector_norm(w.float(), dim=-1, keepdim=True)).T
    scale = cfg.lm_head_feature_norm_scale if cfg.lm_head_feature_norm == "hfn" else h_norm
    return scale * cos, h_norm.squeeze(-1)


@pytest.fixture
def model_parallel():
    from tests.unit_tests.test_utilities import Utils

    # The head and the reference multiply in different orders; keep TF32 out of the comparison
    # (NGC images may still force it, hence the tolerances: real bugs here are O(0.1)).
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    Utils.initialize_model_parallel(1, 1)
    yield
    Utils.destroy_model_parallel()
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def _capture_final_norm(model):
    store = {}
    handle = model.decoder.final_layernorm.register_forward_hook(lambda m, i, o: store.update(h=o))
    return store, handle


@needs_cuda
@pytest.mark.parametrize("mode", ["nfn", "hfn", "sfn"])
def test_eval_logits_are_the_feature_norm_head(model_parallel, mode):
    model = _build(mode, d=0.1)
    model.eval()
    tokens, position_ids, _, _ = _inputs()
    store, handle = _capture_final_norm(model)
    with torch.no_grad():
        logits = model(tokens, position_ids, None)
    handle.remove()
    ref, _ = _reference_logits(model, store["h"])
    torch.testing.assert_close(logits.float(), ref.transpose(0, 1), rtol=2e-3, atol=2e-3)
    assert getattr(model, "_token_margin_report") is None   # eval leaves no report


@needs_cuda
@pytest.mark.parametrize("mode", ["nfn", "hfn", "sfn"])
@pytest.mark.parametrize("d,reverse", [(0.0, False), (0.1, False), (0.1, True)])
@pytest.mark.parametrize("fused_ce", [False, True])
def test_training_loss_and_grads_match_definition(model_parallel, mode, d, reverse, fused_ce):
    scale, t = 20.0, 1e-2
    model = _build(mode, d=d, reverse=reverse, scale=scale, t=t, fused_ce=fused_ce)
    model.train()
    assert model.output_layer.gradient_accumulation_fusion is False
    final_w = model.decoder.final_layernorm.weight
    assert final_w.requires_grad == (mode != "hfn")
    tokens, position_ids, labels, mask = _inputs()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]

    # Transformer Engine's fused ops free their saved tensors after one backward (retain_graph
    # does not keep them), so the model path and the reference each get their own forward and a
    # single backward. Dropout is 0: both forwards compute the same h.
    loss = model(tokens, position_ids, None, labels=labels, loss_mask=mask)   # [b, s]
    report = pop_token_margin_report(model)
    g_model = torch.autograd.grad((loss * mask).sum(), params, allow_unused=True)

    store, handle = _capture_final_norm(model)
    model(tokens, position_ids, None)          # labels None: logits only, no margin, no report
    handle.remove()
    logits, h_norm = _reference_logits(model, store["h"])                   # [s, b, V], [s, b]
    signed_d = -d if reverse else d
    margin = signed_d * (scale if mode == "hfn" else h_norm)
    margin = torch.as_tensor(margin, device=logits.device).expand_as(h_norm)
    ref = _modified_logit_ce(
        logits.reshape(-1, VOCAB), labels.T.reshape(-1), margin.reshape(-1)
    ).reshape(SEQ, BATCH)
    if mode == "sfn":
        ref = ref + t * (h_norm - scale) ** 2
    ref = ref.T
    torch.testing.assert_close(loss.detach(), ref.detach(), rtol=2e-3, atol=2e-3)
    g_ref = torch.autograd.grad((ref * mask).sum(), params, allow_unused=True)

    for (name, _), gm, gr in zip(named, g_model, g_ref):
        if gr is None:
            assert gm is None or gm.abs().max() == 0, name
            continue
        torch.testing.assert_close(gm, gr, rtol=5e-3, atol=1e-4, msg=lambda m: f"{name}: {m}")

    expected = {"ce loss", "margin r", "margin logit m", "lm head input norm", "logit max", "logit rms"}
    assert set(report) == expected | ({"sfn sq dev"} if mode == "sfn" else set())
    with torch.no_grad():
        plain_ce = F.cross_entropy(logits.reshape(-1, VOCAB), labels.T.reshape(-1), reduction="none")
        plain_ce = plain_ce.reshape(SEQ, BATCH).T
    torch.testing.assert_close(
        report["ce loss"], torch.stack([(plain_ce * mask).sum(), mask.sum()]).float(), rtol=2e-3, atol=2e-2
    )
    assert pop_token_margin_report(model) == {}


@needs_cuda
def test_normalize_rows_unit_norm():
    w = torch.randn(10, 7, device="cuda", dtype=torch.bfloat16) * 3
    n = torch.linalg.vector_norm(normalize_rows(w).float(), dim=-1)
    torch.testing.assert_close(n, torch.ones_like(n), rtol=1e-2, atol=1e-2)


# ---- Hugging Face export (GPU + transformers) --------------------------------------------------
@needs_cuda
@pytest.mark.parametrize("mode", [None, "nfn", "hfn", "sfn"])
@pytest.mark.parametrize("qk_layernorm", [False, True])
def test_hf_export_matches_megatron(model_parallel, mode, qk_layernorm):
    transformers = pytest.importorskip("transformers")
    sys.path.insert(0, os.path.join(REPO, "ladder", "export"))
    from hf_mapping import DenseSpec, hf_config, megatron_to_hf

    model = _build(mode, d=0.1 if mode else 0.0, scale=20.0, qk_layernorm=qk_layernorm)
    with torch.no_grad():   # random norm weights, so a swapped norm would show
        for name, p in model.named_parameters():
            if "layer_norm_weight" in name or "layernorm" in name:
                if mode == "hfn" and name.startswith("decoder.final_layernorm"):
                    continue   # frozen at 1 under hfn
                p.uniform_(0.5, 1.5)
    model.eval()
    tokens, position_ids, _, _ = _inputs()
    with torch.no_grad():
        meg = model(tokens, position_ids, None).float()

    spec = DenseSpec(
        num_layers=2, hidden_size=64, num_attention_heads=4, num_query_groups=2, head_dim=16,
        ffn_hidden_size=128, vocab_size=VOCAB, norm_eps=1e-5, rope_theta=10000.0,
        max_position_embeddings=SEQ, qk_layernorm=qk_layernorm, head_mode=mode,
        head_scale=20.0 if mode == "hfn" else None,
    )
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if torch.is_tensor(v)}
    hf_sd = megatron_to_hf(state, spec)
    cfg = hf_config(spec, bos_token_id=None, eos_token_id=None, dtype="float32")
    model_type = cfg.pop("model_type")
    for key in ("architectures", "torch_dtype"):
        cfg.pop(key)
    hf_cfg = transformers.AutoConfig.for_model(model_type, **cfg)
    hf_cfg._attn_implementation = "eager"
    hf = transformers.AutoModelForCausalLM.from_config(hf_cfg).float()
    missing, unexpected = hf.load_state_dict(hf_sd, strict=False)
    assert not unexpected and all("rotary" in k for k in missing), (missing, unexpected)
    hf = hf.cuda().eval()
    with torch.no_grad():
        ref = hf(input_ids=tokens).logits.float()
    torch.testing.assert_close(ref, meg, rtol=2e-3, atol=2e-3)
