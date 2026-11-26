export HF_ENDPOINT=https://hf-mirror.com
DATA=/fast/users/zhuangxinlin/data

# SFT 源数据：OMI2 完整训练集（data/train-000xx-of-00032.parquet）
bash hfd.sh nvidia/OpenMathInstruct-2 --dataset --include "data/train-*" --local-dir $DATA/OpenMathInstruct-2

# 评测集
bash hfd.sh openai/gsm8k --dataset --include "main/test-*" --local-dir $DATA/gsm8k
bash hfd.sh HuggingFaceH4/MATH-500 --dataset --local-dir $DATA/MATH-500
bash hfd.sh math-ai/amc23 --dataset --local-dir $DATA/amc23
bash hfd.sh Jiayi-Pan/Countdown-Tasks-3to4 --dataset --local-dir $DATA/Countdown-Tasks-3to4
python ladder/sft/eval/prepare_countdown_eval.py --input $DATA/Countdown-Tasks-3to4 \
    --output $DATA/countdown/test_tinyzero_1024.jsonl