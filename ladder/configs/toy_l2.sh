# Margin toy (ladder/toy/TOY.md): a 2-layer Llama-style model on the 100-token synthetic graph language of
# ladder/toy/graph.py. Sourced by ladder/train.sh, ladder/sft/train_sft.sh and ladder/export/export_hf.sh like the
# ladder rungs; ladder/toy/chain.sh runs it on one GPU (GPUS_PER_NODE=1).
#
# Shape: d 256, 4 heads x 64 (no GQA), SwiGLU FFN 3.5d = 896, RMSNorm, RoPE base 10k, untied embeddings.
# --make-vocab-size-divisible-by 4 keeps the 100-token vocabulary unpadded (the default 128 would add 28 rows to
# the softmax that no data ever uses).
#
# N_EFF = non-embedding params + lm_head:
#   per layer: q 256*256 + kv 2*256*256 + o 256*256 + mlp 3*256*896 + 2 norms 2*256 = 950,784
#   2 layers 1,901,568 + final norm 256 + lm_head 256*100 25,600 = 1,927,424
N_EFF=1927424

TOKENIZER_VOCAB=100        # ladder/toy/make_tokenizer.py
SEQ_LEN=256                # 25 walk documents (10 tokens each) per sequence; SFT packs examples into it
GBS=128                    # 128 x 256 = 32768 tokens per iteration
MBS=${MBS:-128}            # one micro-batch per step on one GPU
NODES=1

# One stable WSD run, no decay branches: E(TPP 100) = 6250 iterations (205M tokens), decay from 5000.
# Every checkpoint is kept (no branches: retain interval = save interval), so training can be probed over time.
TPP_MAX=100
BRANCH_TPPS=""
SAVE_INTERVAL=500
EVAL_INTERVAL=250
WARMUP_ITERS=200

# Placeholder until the plain-head sweep (PHASE=lr in ladder/toy/run_toy.sh) picks the LR.
LR=${LR:-4e-3}
MIN_LR=${MIN_LR:-0}
LR_GRID="1e-3 2e-3 4e-3 8e-3 1.6e-2"

MODEL_ARGS=(
    --num-layers 2
    --hidden-size 256
    --ffn-hidden-size 896
    --num-attention-heads 4
    --kv-channels 64
    --seq-length ${SEQ_LEN}
    --max-position-embeddings ${SEQ_LEN}
    --position-embedding-type rope
    --rotary-base 10000
    --rotary-percent 1.0
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --init-method-std 0.02
    --make-vocab-size-divisible-by 4
)
