# TOK=/mnt/users/zhuangxinlin/ckpt/Llama3.1-8B        # base 版本的 tokenizer
# SRC=/mnt/oss-zhuangxinlin/climbmix_v1                # build_mix.sh 的输出目录
# OUT=/mnt/oss-zhuangxinlin/megatron_data/llama3       # Megatron 格式数据的输出目录
# WORKERS=64                                           # 按节点的 CPU 核数调整

# FAMILY 选择 tokenizer 和输出目录（llama3 | qwen3），与 ladder/launch/*.sh 按 config 前缀取的默认路径一致。
# Qwen3 必须用 base 版 tokenizer：追加的 EOS 是 tokenizer.eos_token，base 为 <|endoftext|>，instruct 为 <|im_end|>。
FAMILY=${FAMILY:-llama3}
case $FAMILY in
    llama3) TOK=/fast/users/zhuangxinlin/tokenizers/llama-3 ;;
    qwen3)  TOK=/fast/users/zhuangxinlin/tokenizers/qwen3 ;;
    *) echo "FAMILY 须为 llama3 或 qwen3，当前为 $FAMILY" >&2; exit 1 ;;
esac
SRC=/fast/users/zhuangxinlin/data/climbmix_v1
OUT=/fast/users/zhuangxinlin/megatron_data/$FAMILY
WORKERS=64


python ladder/data/tokenize_dataset.py \
    --input-root $SRC/m1 --input-glob '**/*.parquet' \
    --output-root $OUT/climbmix_v1_m1 --tokenizer $TOK --workers $WORKERS


# 合并：把 train/ 下的所有分片合成一个 prefix
python ladder/data/merge_indexed.py \
    --input-root $OUT/climbmix_v1_m1/train \
    --output-prefix $OUT/climbmix_v1_m1/train_merged

# 检查
python ladder/data/inspect_indexed.py $OUT/climbmix_v1_m1/train_merged --tokenizer $TOK
python ladder/data/inspect_indexed.py $OUT/climbmix_v1_m1/val_all/val --tokenizer $TOK

# Slurm CPU 提交：在跳板机激活 megatron 环境，进入共享的 Megatron-LM 仓库根目录，
# 将下面的 sbatch 命令复制到终端执行。申请的 CPU 数与上面的 WORKERS 对齐。
# sbatch --partition=polaris --job-name=prepare-m1 \
#     --nodes=1 --ntasks=1 --cpus-per-task=64 --mem=128G --time=04:00:00 \
#     --chdir="$PWD" --export=ALL --output="prepare-m1-%j.out" \
#     --wrap='srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" bash -e ladder/data/prepare_dataset.sh'