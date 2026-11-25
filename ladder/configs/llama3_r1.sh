# Llama3 family, rung R1. Sourced by ladder/train.sh; ladder/sweep.sh also reads NODES and LR_GRID.
#
# Shape rule of the Llama3 family: head_dim 128 (heads = d/128), KV groups ~ heads/4, SwiGLU FFN 3.5d,
# RMSNorm eps 1e-5, RoPE base 500k, untied embeddings, Llama-3.1 tokenizer (128256, no padding).
#
# N_EFF = non-embedding params + lm_head (the input embedding is a lookup, not compute):
#   per layer: q 1024*1024 + kv 2*1024*256 + o 1024*1024 + mlp 3*1024*3584 + 2 norms 2*1024 = 13,633,536
#   18 layers 245,403,648 + final norm 1,024 + lm_head 1024*128256 131,334,144 = 376,738,816
N_EFF=376738816

TOKENIZER_VOCAB=128256     # len(Llama-3.1 tokenizer); ladder/data/check_tokenizer.py checks it before launch
SEQ_LEN=4096
GBS=256                   # 256 x 4096 = 1M tokens per iteration
MBS=${MBS:-8}             # per-GPU micro batch; GBS / (MBS * data-parallel size) = grad-accum steps
NODES=1                   # 8-GPU nodes per job (ladder/sweep.sh), 8 GPUs: 4 grad-accum steps
# GBS is near the DeepSeek-LLM B_opt(C) = 0.292 C^0.327 at TPP 50, the middle of the branch range.

# Schedule (see train.sh): one stable WSD run to TPP_MAX, decay branches at BRANCH_TPPS.
TPP_MAX=100
BRANCH_TPPS="20 50"
SAVE_INTERVAL=2800          # checkpoint every N iterations (resume points and branch starts); multiple of 4
EVAL_INTERVAL=500           # validation every N iterations
# Branch ends E(T) are multiples of lcm(5/4*SAVE_INTERVAL, EVAL_INTERVAL) = 3500:
#   E = 7000 / 17500 / 35000 iterations = TPP 19.5 / 48.7 / 97.4; decay starts 5600 / 14000 / 28000;
#   the stable run keeps the checkpoints at multiples of 2800.
# Fix both intervals before the first run of a rung: they decide E(TPP_MAX), and with it the data order.
WARMUP_ITERS=500

# Peak / final LR; the optimizer itself comes from ladder/configs/opt/<OPT>.sh.
# Default = DeepSeek-LLM eta_opt(C) = 0.3118 C^-0.125 at TPP 100; LR_GRID is what ladder/sweep.sh runs.
LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-0}
LR_GRID="5e-4 7e-4 1e-3 1.4e-3 2e-3"

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
