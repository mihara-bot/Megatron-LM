# Muon for the 2D hidden weights, AdamW for the rest. Sourced by ladder/train.sh (OPT=muon).
#
# Megatron routes parameters itself (megatron/core/optimizer/emerging_optimizers.py):
#   Muon : every 2D weight inside the transformer layers (QKV, attention out, fc1, fc2)
#   AdamW: embedding, lm_head (is_embedding_or_output_parameter) and every non-2D tensor (norms)
# Both parts use the same --lr / --min-lr and the same schedule.
#
# Update of a Muon weight W (m x n): W -= lr * (0.2 * sqrt(max(m, n)) * NS5(momentum) + wd * W)
#   --muon-scale-mode spectral gives sqrt(max(m, n)); --muon-extra-scale-factor 0.2 brings the update
#   RMS to about AdamW's (Moonlight, arXiv 2502.16982), so AdamW's LR is a sensible starting point
#   and one LR can serve both parts.
# QKV is orthogonalized per Q / K / V block (--muon-split-qkv is on by default). The fused SwiGLU fc1
# [gate; up] is still orthogonalized as one matrix; upstream has no option to split it.
#
# With --use-distributed-optimizer (set in train.sh) Megatron switches Muon to the layer-wise
# distributed optimizer: each data-parallel rank runs Newton-Schulz for its own share of the layers.
OPT_ARGS=(
    --optimizer muon
    --muon-momentum 0.95             # CLI default is 0.9; Moonlight / Keller Jordan use 0.95
    --muon-nesterov
    --muon-num-ns-steps 5
    --muon-coefficient-type quintic
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
    --muon-scalar-optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --weight-decay 0.1
    --clip-grad 1.0
)
