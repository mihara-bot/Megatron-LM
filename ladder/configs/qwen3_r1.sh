# Qwen3 family, rung R1. Sourced by ladder/train.sh; ladder/sweep.sh also reads NODES and LR_GRID.
#
# Shape rule of the Qwen3 family: heads = max(16, d/128), 8 KV groups, kv_channels 128 (so the attention
# width 16*128 = 2048 is wider than d at this rung), SwiGLU FFN 3d, QK RMSNorm, RMSNorm eps 1e-6,
# RoPE base 1e6, 28 layers, untied embeddings (the official small Qwen3 models tie them),
# Qwen3 base tokenizer (len 151669, padded by Megatron to 151680 = 1185*128).
#
# N_EFF = non-embedding params + lm_head (the input embedding is a lookup, not compute):
#   per layer: q 768*2048 + kv 2*768*1024 + o 2048*768 + mlp 3*768*2304 + 2 norms 2*768
#              + q/k norms 2*128 = 10,028,800
#   28 layers 280,806,400 + final norm 768 + lm_head 768*151680 116,490,240 = 397,297,408
# (Llama3-R1: 376,738,816.) Training FLOPs/token (Megatron, causal) 3.79e9 = 1.40x Llama3-R1.
N_EFF=397297408

TOKENIZER_VOCAB=151669     # len(Qwen3 base tokenizer); ladder/data/check_tokenizer.py checks it before launch
SEQ_LEN=4096
GBS=256                   # 256 x 4096 = 1M tokens per iteration (same as Llama3-R1)
MBS=${MBS:-8}             # per-GPU micro batch; GBS / (MBS * data-parallel size) = grad-accum steps
NODES=1                   # 8-GPU nodes per job (ladder/sweep.sh), 8 GPUs: 4 grad-accum steps

# Schedule (see train.sh): one stable WSD run to TPP_MAX, decay branches at BRANCH_TPPS.
TPP_MAX=100
BRANCH_TPPS="20 50"
SAVE_INTERVAL=1500          # checkpoint every N iterations (resume points and branch starts); multiple of 4
EVAL_INTERVAL=750           # validation every N iterations
# Branch ends E(T) are multiples of lcm(5/4*SAVE_INTERVAL, EVAL_INTERVAL) = 3750:
#   E = 7500 / 18750 / 37500 iterations = TPP 19.8 / 49.5 / 99.0; decay starts 6000 / 15000 / 30000;
#   the stable run keeps the checkpoints at multiples of 3000.
# (Llama3-R1's 2800 / 500 would put the branch ends at TPP 18.5 / 46.2 / 101.6 with this N_EFF.)
# Fix both intervals before the first run of a rung: they decide E(TPP_MAX), and with it the data order.
WARMUP_ITERS=500

# Peak / final LR; the optimizer itself comes from ladder/configs/opt/<OPT>.sh.
# Default and grid come from the Llama3-R1 Muon sweep (2026-11-25, N_eff within 5%): lr* = 3.4e-3 /
# 2.9e-3 / 2.7e-3 at TPP 20 / 50 / 100, and 2.8e-3 has regret <= 0.002 at every TPP. Same grid points as
# Llama3-R1 so that both families are fitted on the same LRs (see ladder/run_qwen3_r1_lr_sweep.sh).
LR=${LR:-2.8e-3}
MIN_LR=${MIN_LR:-0}
LR_GRID="1.4e-3 2e-3 2.8e-3 4e-3 5.6e-3"

MODEL_ARGS=(
    --num-layers 28
    --hidden-size 768
    --ffn-hidden-size 2304
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
