CONFIG=llama3_r1 OPT=muon LR=2.8e-3 HEAD=hfn MARGIN_D=0.02 \
  sbatch -J pre-export -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=128G -t 02:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/eval/export.sh'

# 评测（1 卡），第一次建议先用 LIMIT 做冒烟
MODEL="/fast/users/zhuangxinlin/ckpts/runs/llama3_r1_muon_lr2.8e-3_hfn_d0.02/hf/*_iter*" \
VLLM_PYTHON=/fast/users/zhuangxinlin/envs/vllm/bin/python TASKS=arc_easy,hellaswag LIMIT=200 \
  sbatch -J lm-eval -A polaris -p polaris -N 1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=16 \
  --mem=128G -t 01:00:00 -o logs/%x-%j.out --wrap 'srun --ntasks=1 ht-run bash ladder/eval/lm_eval.sh'