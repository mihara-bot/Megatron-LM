# Llama3 family, rung R1. Sourced by ladder/train.sh.
#
# Shape rule of the Llama3 family: head_dim 128 (heads = d/128), KV groups ~ heads/4, SwiGLU FFN 3.5d,
# RMSNorm eps 1e-5, RoPE base 500k, untied embeddings, Llama-3.1 tokenizer (128256, no padding).
#
# N_EFF = non-embedding params + lm_head (the input embedding is a lookup, not compute):
#   per layer: q 1024*1024 + kv 2*1024*256 + o 1024*1024 + mlp 3*1024*3584 + 2 norms 2*1024 = 13,633,536
#   18 layers 245,403,648 + final norm 1,024 + lm_head 1024*128256 131,334,144 = 376,738,816
N_EFF=376738816

SEQ_LEN=4096
GBS=256                       # 256 x 4096 = 1M tokens per iteration
MBS=${MBS:-8}                 # per-GPU micro batch; GBS / (MBS * data-parallel size) = grad-accum steps

# Schedule (see train.sh): one stable WSD run to TPP_MAX, decay branches at BRANCH_TPPS.
TPP_MAX=50
BRANCH_TPPS="20"              # cheap extra data point (1500 iters from the stable ckpt at 6000)
SAVE_INTERVAL=2000            # checkpoint every N iterations (resume points and branch starts); multiple of 4
EVAL_INTERVAL=500             # validation every N iterations
# Branch ends E(T) are rounded to multiples of lcm(5/4*SAVE_INTERVAL, EVAL_INTERVAL) = 2500 here.
# Fix both intervals before the first run of a rung: they decide E(TPP_MAX), and with it the data order.
WARMUP_ITERS=500              # ~0.5B tokens

# Peak / final LR; the optimizer itself comes from ladder/configs/opt/<OPT>.sh. LR is swept.
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-0}

MODEL_ARGS=(
    --num-layers 18
    --hidden-size 1024
    --ffn-hidden-size 3584
    --num-attention-heads 8
    --group-query-attention
    --num-query-groups 2
    --kv-channels 128
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --position-embedding-type rope
    --rotary-base 500000
    --rotary-percent 1.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.02
)
