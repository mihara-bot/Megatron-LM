# Qwen3 family, rung R2 (the official Qwen3-0.6B shape, with untied embeddings). Sourced by
# ladder/train.sh; ladder/sweep.sh also reads NODES and LR_GRID.
#
# Shape rule of the Qwen3 family: heads = max(16, d/128), 8 KV groups, kv_channels 128 (attention width
# 2048 = 2d at this rung), SwiGLU FFN 3d, QK RMSNorm, RMSNorm eps 1e-6, RoPE base 1e6, 28 layers,
# untied embeddings, Qwen3 base tokenizer (len 151669, padded by Megatron to 151680 = 1185*128).
#
# N_EFF = non-embedding params + lm_head (the input embedding is a lookup, not compute):
#   per layer: q 1024*2048 + kv 2*1024*1024 + o 2048*1024 + mlp 3*1024*3072 + 2 norms 2*1024
#              + q/k norms 2*128 = 15,730,944
#   28 layers 440,466,432 + final norm 1,024 + lm_head 1024*151680 155,320,320 = 595,787,776
# (Llama3-R2: 586,927,360.) Training FLOPs/token (Megatron, causal) 4.98e9 = 1.20x Llama3-R2.
N_EFF=595787776

TOKENIZER_VOCAB=151669     # len(Qwen3 base tokenizer); ladder/data/check_tokenizer.py checks it before launch
SEQ_LEN=4096
GBS=256                   # 256 x 4096 = 1M tokens per iteration (same as Llama3-R2)
MBS=${MBS:-8}             # per-GPU micro batch; GBS / (MBS * data-parallel size) = grad-accum steps
NODES=2                   # 8-GPU nodes per job (ladder/sweep.sh), 16 GPUs: 2 grad-accum steps

# Schedule (see train.sh): one stable WSD run to TPP_MAX, decay branches at BRANCH_TPPS.
TPP_MAX=100
BRANCH_TPPS="20 50"
SAVE_INTERVAL=2200          # checkpoint every N iterations (resume points and branch starts); multiple of 4
EVAL_INTERVAL=500           # validation every N iterations
# Same intervals as Llama3-R2, so both rungs run the same iterations (= the same token counts).
# Branch ends E(T) are multiples of lcm(5/4*SAVE_INTERVAL, EVAL_INTERVAL) = 5500:
#   E = 11000 / 27500 / 55000 iterations = TPP 19.4 / 48.4 / 96.8; decay starts 8800 / 22000 / 44000;
#   the stable run keeps the checkpoints at multiples of 4400.
# Fix both intervals before the first run of a rung: they decide E(TPP_MAX), and with it the data order.
WARMUP_ITERS=500

# Peak / final LR; the optimizer itself comes from ladder/configs/opt/<OPT>.sh.
# Default and grid come from the Llama3-R2 Muon sweep (2026-11-25, N_eff within 2%): lr* = 2.1e-3 /
# 1.8e-3 / 1.6e-3 at TPP 20 / 50 / 100. Same grid points as Llama3-R2 (see ladder/run_qwen3_r2_lr_sweep.sh).
LR=${LR:-2e-3}
MIN_LR=${MIN_LR:-0}
LR_GRID="1e-3 1.4e-3 2e-3 2.8e-3"

MODEL_ARGS=(
    --num-layers 28
    --hidden-size 1024
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --qk-layernorm
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.02
)
