# Llama3 family, rung R3. Sourced by ladder/train.sh; ladder/sweep.sh also reads NODES and LR_GRID.
#
# Shape rule of the Llama3 family: head_dim 128 (heads = d/128), KV groups ~ heads/4, SwiGLU FFN 3.5d,
# RMSNorm eps 1e-5, RoPE base 500k, untied embeddings, Llama-3.1 tokenizer (128256, no padding).
#
# N_EFF = non-embedding params + lm_head (the input embedding is a lookup, not compute):
#   per layer: q 1536*1536 + kv 2*1536*384 + o 1536*1536 + mlp 3*1536*5376 + 2 norms 2*1536 = 30,673,920
#   24 layers 736,174,080 + final norm 1,536 + lm_head 1536*128256 197,001,216 = 933,176,832
N_EFF=933176832

SEQ_LEN=4096
GBS=384                   # 384 x 4096 = 1.5M tokens per iteration
MBS=${MBS:-4}             # per-GPU micro batch; GBS / (MBS * data-parallel size) = grad-accum steps
NODES=4                   # 8-GPU nodes per job (ladder/sweep.sh), 32 GPUs: 3 grad-accum steps
# GBS is near the DeepSeek-LLM B_opt(C) = 0.292 C^0.327 at TPP 50, the middle of the branch range.

# Schedule (see train.sh): one stable WSD run to TPP_MAX, decay branches at BRANCH_TPPS.
TPP_MAX=100
BRANCH_TPPS="20 50"
SAVE_INTERVAL=2400          # checkpoint every N iterations (resume points and branch starts); multiple of 4
EVAL_INTERVAL=1000          # validation every N iterations
# Branch ends E(T) are multiples of lcm(5/4*SAVE_INTERVAL, EVAL_INTERVAL) = 3000:
#   E = 12000 / 30000 / 60000 iterations = TPP 20.2 / 50.6 / 101.1; decay starts 9600 / 24000 / 48000;
#   the stable run keeps the checkpoints at multiples of 4800.
# Fix both intervals before the first run of a rung: they decide E(TPP_MAX), and with it the data order.
WARMUP_ITERS=500

# Peak / final LR; the optimizer itself comes from ladder/configs/opt/<OPT>.sh.
# Default = DeepSeek-LLM eta_opt(C) = 0.3118 C^-0.125 at TPP 100; LR_GRID is what ladder/sweep.sh runs.
LR=${LR:-8e-4}
MIN_LR=${MIN_LR:-0}
LR_GRID="5.5e-4 8e-4 1.1e-3"

MODEL_ARGS=(
    --num-layers 24
    --hidden-size 1536
    --ffn-hidden-size 5376
    --num-attention-heads 12
    --group-query-attention
    --num-query-groups 3
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
