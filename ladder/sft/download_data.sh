export HF_ENDPOINT=https://hf-mirror.com
DATA=/fast/users/zhuangxinlin/data

# Step 1. Download

# SFT training set OMI2 full (/train-000xx-of-00032.parquet）
bash hfd.sh nvidia/OpenMathInstruct-2 --dataset --include "data/train-*" --local-dir $DATA/OpenMathInstruct-2

# SFT evaluation set (gsm8k, math-500, amc23, Countdown)
bash hfd.sh openai/gsm8k --dataset --include "main/test-*" --local-dir $DATA/gsm8k
bash hfd.sh HuggingFaceH4/MATH-500 --dataset --local-dir $DATA/MATH-500
bash hfd.sh math-ai/amc23 --dataset --local-dir $DATA/amc23
bash hfd.sh Jiayi-Pan/Countdown-Tasks-3to4 --dataset --local-dir $DATA/Countdown-Tasks-3to4
python ladder/sft/eval/prepare_countdown_eval.py --input $DATA/Countdown-Tasks-3to4 \
    --output $DATA/countdown/test_tinyzero_1024.jsonl


# Step 2. Preprocess

# extract 500k for training and 2k for validation

python -m src.cli.prepare_sft_subset --input $DATA/OpenMathInstruct-2/data --file_glob 'train-*' \
  --output_dir $DATA/sft/openmathinstruct2_500k --num_train 500000 --num_validation 2000 --seed 42 \
  --columns problem generated_solution expected_answer problem_source --disjoint_column problem


# process into Megatron format
python ladder/sft/prepare_sft_data.py --input $DATA/sft/openmathinstruct2_500k/train \
  --tokenizer /fast/users/zhuangxinlin/tokenizers/llama-3 \
  --output-dir $DATA/sft_megatron/omi2_500k_ep10_llama3 --epochs 10