# Llama3 family, rung R2. Sourced by ladder/train.sh; ladder/sweep.sh also reads NODES and LR_GRID.
#
# Shape rule of the Llama3 family: head_dim 128 (heads = d/128), KV groups ~ heads/4, SwiGLU FFN 3.5d,
# RMSNorm eps 1e-5, RoPE base 500k, untied embeddings, Llama-3.1 tokenizer (128256, no padding).
#
# N_EFF = non-embedding params + lm_head (the input embedding is a lookup, not compute):
#   per layer: q 1280*1280 + kv 2*1280*256 + o 1280*1280 + mlp 3*1280*4480 + 2 norms 2*1280 = 21,137,920
#   20 layers 422,758,400 + final norm 1,280 + lm_head 1280*128256 164,167,680 = 586,927,360
N_EFF=586927360

TOKENIZER_VOCAB=128256     # len(Llama-3.1 tokenizer); ladder/data/check_tokenizer.py checks it before launch
SEQ_LEN=4096
GBS=256                   # 256 x 4096 = 1M tokens per iteration
MBS=${MBS:-8}             # per-GPU micro batch; GBS / (MBS * data-parallel size) = grad-accum steps
NODES=2                   # 8-GPU nodes per job (ladder/sweep.sh), 16 GPUs: 2 grad-accum steps
# GBS is near the DeepSeek-LLM B_opt(C) = 0.292 C^0.327 at TPP 50, the middle of the branch range.

# Schedule (see train.sh): one stable WSD run to TPP_MAX, decay branches at BRANCH_TPPS.
TPP_MAX=100
BRANCH_TPPS="20 50"
SAVE_INTERVAL=2200          # checkpoint every N iterations (resume points and branch starts); multiple of 4
EVAL_INTERVAL=500           # validation every N iterations
# Branch ends E(T) are multiples of lcm(5/4*SAVE_INTERVAL, EVAL_INTERVAL) = 5500:
#   E = 11000 / 27500 / 55000 iterations = TPP 19.7 / 49.1 / 98.3; decay starts 8800 / 22000 / 44000;
#   the stable run keeps the checkpoints at multiples of 4400.
# Fix both intervals before the first run of a rung: they decide E(TPP_MAX), and with it the data order.
WARMUP_ITERS=500

# Peak / final LR; the optimizer itself comes from ladder/configs/opt/<OPT>.sh.
# Default = DeepSeek-LLM eta_opt(C) = 0.3118 C^-0.125 at TPP 100; LR_GRID is what ladder/sweep.sh runs.
LR=${LR:-9e-4}
MIN_LR=${MIN_LR:-0}
LR_GRID="6e-4 9e-4 1.3e-3"

MODEL_ARGS=(
    --num-layers 20
    --hidden-size 1280
    --ffn-hidden-size 4480
    --num-attention-heads 10
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
