"""SphereFace-R LM head and token-margin loss for GPTModel.

Port of pretrain_LLM_pipeline's ``train.loss.margin`` (SphereFace-R, Liu et al., TPAMI 2022,
arXiv 2109.05565, Eq. 3/5/6/18, additive cosine margin ``psi(theta) = cos(theta) - d``).
Configured by ``TransformerConfig.lm_head_feature_norm`` / ``lm_head_feature_norm_scale`` /
``soft_feature_norm_weight`` / ``token_margin`` / ``token_margin_reverse``.

The output-layer rows always have unit norm, so every logit is a feature norm times
``cos(h, W_i)`` (``h`` = output-layer input, ``s`` = scale, ``t`` = sfn weight):

- ``nfn``: logits ``||h|| cos(h, W_i)``
- ``hfn``: logits ``s cos(h, W_i)`` (every token's ``h`` rescaled to norm ``s``)
- ``sfn``: logits as nfn; the training loss adds ``t (||h|| - s)^2`` per token

In training the target logit uses ``cos_y - d`` (``cos_y + d`` with reverse), i.e. a logit margin
``m = d s`` (hfn) or ``m = d ||h||`` (nfn, sfn; the gradient flows through ``||h||``). The margin
loss depends on the logits only through the per-token cross entropy ``L`` (``p_y = e^{-L}``):

    L_margin = L + m + log((1 - p_y) + p_y e^{-m})

so it is applied to the output of the vocab-parallel cross entropy, whose backward already
scales the logit gradient per token: the logit gradient is ``r dL/dz`` with
``r = 1 / ((1 - p_y) + p_y e^{-m})``. No modified copy of the logits is made and the TP vocab
split is untouched. The log term is a ``logaddexp`` in fp64, finite for any ``m`` and ``L >= 0``.
The fp32 cross entropy resolves ``L`` to ~6e-8 near 0 and the closed form scales that error by
up to ``e^|m|``: keep ``|m|`` below ~12. Evaluation (``model.eval()``) returns the plain cross
entropy of the normalized head, without margin or penalty.

Export (ladder/export/hf_mapping.py): nfn and sfn fold exactly into a plain output layer with rows
``W_i / ||W_i||``. hfn folds into ``(s / sqrt(D)) W_i / ||W_i||``: GPTModel freezes the final
RMSNorm weight at 1 under hfn, so ``||h|| = sqrt(D)`` up to the norm's epsilon.
"""

from typing import Dict, Optional, Tuple

import torch

from megatron.core.tensor_parallel import ColumnParallelLinear

FEATURE_NORM_MODES = ("nfn", "hfn", "sfn")
NORM_EPS = 1.0e-6
LOGIT_STATS_ROWS = 256
REPORT_ATTR = "_token_margin_report"


def head_geometry(mode: Optional[str], scale: Optional[float]) -> Tuple:
    """What the head computes from (h, W): equal geometries give equal logits from equal weights.

    nfn and sfn differ only in the training loss, so a checkpoint of one is a valid start for
    the other (e.g. SFT of an sfn run with nfn and no penalty).
    """
    if mode is None:
        return ("plain",)
    if mode == "hfn":
        return ("hfn", float(scale))
    if mode in ("nfn", "sfn"):
        return ("cos",)
    raise ValueError(f"unknown lm_head_feature_norm {mode!r}")


def normalize_rows(weight: torch.Tensor) -> torch.Tensor:
    """``weight`` with every row rescaled to unit L2 norm (norm reduced in fp32)."""
    norm = torch.linalg.vector_norm(weight, dim=-1, keepdim=True, dtype=torch.float32)
    return weight * (1.0 / norm.clamp_min(NORM_EPS)).to(weight.dtype)


class FeatureNormColumnParallelLinear(ColumnParallelLinear):
    """Output layer of the SphereFace-R head: multiplies by the row-normalized weight.

    Same parameters (and checkpoint keys) as ColumnParallelLinear. The rows are normalized inside
    this module's forward, after its forward pre-hooks: with --overlap-param-gather, DDP finishes a
    module's parameter all-gather in that module's own pre-hook, so the weight must not be read
    before. Gradient-accumulation fusion is off for this layer only: the GEMM's weight is the
    normalized copy, which has no main_grad; autograd carries the gradient through the
    normalization to weight.grad, and the DDP hook adds it to main_grad.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_fusion = False

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None, **kwargs):
        if weight is not None:
            raise ValueError("FeatureNormColumnParallelLinear normalizes its own weight.")
        return super().forward(input_, weight=normalize_rows(self.weight), **kwargs)


def feature_norm_head_input(
    hidden: torch.Tensor, mode: str, scale: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Output-layer input of the head and the per-token ``||h||`` (fp32, with graph).

    ``hidden`` is [..., D]; the norm has the leading shape of ``hidden``. Rows are local, so
    this needs no communication under sequence parallelism (the norm is then also local).
    """
    norm = torch.linalg.vector_norm(hidden, dim=-1, dtype=torch.float32)
    if mode == "hfn":
        hidden = hidden * (scale / norm.clamp_min(NORM_EPS)).unsqueeze(-1).to(hidden.dtype)
    return hidden, norm


def _log_margin_denominator(losses: torch.Tensor, margin) -> torch.Tensor:
    """``log((1 - p_y) + p_y e^{-m})`` with ``p_y = e^{-L}``, stable for any ``m`` and ``L >= 0``."""
    margin = torch.as_tensor(margin, dtype=losses.dtype, device=losses.device)
    # log(1 - p_y) is -inf at L = 0 (p_y = 1); the where keeps the infinite derivative of
    # log(-expm1) at L = 0 out of the graph. L <= 0 (rounding) is treated as L = 0.
    positive = losses > 0
    safe = torch.where(positive, losses, torch.ones_like(losses))
    log_rest = torch.where(
        positive, torch.log(-torch.expm1(-safe)), torch.full_like(losses, float("-inf"))
    )
    return torch.logaddexp(log_rest, -(losses + margin))


def token_margin_loss(losses: torch.Tensor, margin) -> torch.Tensor:
    """Per-token margin loss from per-token cross entropy; ``margin`` is a float or tensor.

    ``margin = 0`` returns ``losses``; a negative margin adds ``-margin`` to the target logit.
    """
    return losses + margin + _log_margin_denominator(losses, margin)


def token_margin_scale(losses: torch.Tensor, margin) -> torch.Tensor:
    """Per-token factor ``r`` by which the margin loss scales the cross-entropy gradient."""
    return torch.exp(-_log_margin_denominator(losses, margin))


def _report_entry(total: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """A ``[sum, count]`` report entry: the training loop sums these over microbatches and DP."""
    return torch.stack([total.detach().float(), count.detach().float()])


def token_margin_objective(
    ce: torch.Tensor, feature_norm: torch.Tensor, loss_mask: Optional[torch.Tensor], config
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Per-token training loss of the feature-norm head, and its report entries.

    Args:
        ce: [b, s] per-token cross entropy of the feature-norm head (with graph).
        feature_norm: [b, s] fp32 per-token ``||h||`` of the output-layer input (with graph).
        loss_mask: [b, s] mask of the tokens the loss counts, or None for all; used only for
            the report (the loss function applies it to the returned per-token loss).
        config: TransformerConfig with the lm_head_feature_norm / token_margin settings.

    Returns:
        The [b, s] fp32 per-token loss and ``{name: [masked sum, token count]}`` report entries.
    """
    mode = config.lm_head_feature_norm
    scale = float(config.lm_head_feature_norm_scale)
    d = float(config.token_margin)
    signed_d = -d if config.token_margin_reverse else d

    losses = ce.double()
    norm = feature_norm.double()
    # hfn: m = s d; nfn/sfn: m = ||h|| d (gradient through ||h||). Negative under reverse.
    margin = signed_d * (scale if mode == "hfn" else norm)
    objective = losses if d == 0.0 else token_margin_loss(losses, margin)
    if mode == "sfn":
        sq_dev = (norm - scale).square()
        objective = objective + float(config.soft_feature_norm_weight) * sq_dev

    with torch.no_grad():
        mask = torch.ones_like(losses) if loss_mask is None else loss_mask.to(losses.dtype)
        count = mask.sum()
        margin_tok = torch.as_tensor(margin, dtype=losses.dtype, device=losses.device).detach()
        if d == 0.0:
            ratio = torch.ones_like(losses)
        else:
            ratio = token_margin_scale(losses.detach(), margin_tok)
        report = {
            # Plain cross entropy of the normalized head: comparable to a baseline's lm loss.
            "ce loss": _report_entry((losses.detach() * mask).sum(), count),
            "margin r": _report_entry((ratio * mask).sum(), count),
            "margin logit m": _report_entry((margin_tok.expand_as(losses) * mask).sum(), count),
            "lm head input norm": _report_entry((norm.detach() * mask).sum(), count),
        }
        if mode == "sfn":
            report["sfn sq dev"] = _report_entry((sq_dev.detach() * mask).sum(), count)
    return objective.float(), report


@torch.no_grad()
def sampled_logit_stats(
    logits: torch.Tensor, tp_group: Optional[torch.distributed.ProcessGroup] = None
) -> Dict[str, torch.Tensor]:
    """Max logit and logit RMS (``||z|| / sqrt(V)``) over a strided sample of token rows.

    ``logits`` is [s, b, V/tp], vocab-split over ``tp_group``. Call before the cross entropy,
    which consumes the logits in place. The sample is a fixed stride over all positions, so it
    needs no host sync (masked positions are included).
    """
    flat = logits.reshape(-1, logits.size(-1))
    step = max(1, flat.size(0) // LOGIT_STATS_ROWS)
    rows = flat[::step][:LOGIT_STATS_ROWS].float()
    row_max = rows.amax(dim=-1)
    row_sq = rows.square().sum(dim=-1)
    vocab = rows.size(-1)
    if tp_group is not None and tp_group.size() > 1:
        torch.distributed.all_reduce(row_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
        torch.distributed.all_reduce(row_sq, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        vocab *= tp_group.size()
    count = torch.tensor(float(rows.size(0)), device=rows.device)
    return {
        "logit max": _report_entry(row_max.sum(), count),
        "logit rms": _report_entry((row_sq / vocab).sqrt().sum(), count),
    }


def pop_token_margin_report(model) -> Dict[str, torch.Tensor]:
    """Report entries of the model's last training forward (empty if none), cleared on read.

    ``model`` may be wrapped (DDP, Float16Module); the loss function calls this right after
    the forward of the same microbatch on the last pipeline stage.
    """
    module = model
    while not hasattr(module, REPORT_ATTR) and hasattr(module, "module"):
        module = module.module
    report = getattr(module, REPORT_ATTR, None)
    if report is None:
        return {}
    setattr(module, REPORT_ATTR, None)
    return report


__all__ = [
    "FEATURE_NORM_MODES",
    "FeatureNormColumnParallelLinear",
    "LOGIT_STATS_ROWS",
    "NORM_EPS",
    "REPORT_ATTR",
    "feature_norm_head_input",
    "head_geometry",
    "normalize_rows",
    "pop_token_margin_report",
    "sampled_logit_stats",
    "token_margin_loss",
    "token_margin_objective",
    "token_margin_scale",
]
