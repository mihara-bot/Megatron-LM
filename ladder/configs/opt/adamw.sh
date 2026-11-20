# AdamW for all parameters. Sourced by ladder/train.sh (OPT=adamw); LR comes from the rung config.
# Megatron's "adam" is AdamW: decoupled weight decay is on by default.
OPT_ARGS=(
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --adam-eps 1e-8
    --weight-decay 0.1
    --clip-grad 1.0
)
