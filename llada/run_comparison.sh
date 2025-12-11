#!/bin/bash

# ==================== 环境变量 ====================
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ==================== 通用参数 ====================
task=gsm8k
num_fewshot=5
length=256
block_length=32
steps_per_block=$((length / block_length))  # 8
total_steps=${length}                       # LLaDA 需要 steps=gen_length
model_path='GSAI-ML/LLaDA-8B-Instruct'

# # ==================== 1. 官方最快配置 (dual cache + parallel factor) ====================
# echo "=========================================="
# echo "Running OFFICIAL FASTEST: dual_cache + factor=1.0"
# echo "=========================================="

# accelerate launch --num_processes 1 eval_llada.py \
#   --tasks ${task} --num_fewshot ${num_fewshot} \
#   --confirm_run_unsafe_code --model llada_dist \
#   --model_args model_path=${model_path},gen_length=${length},steps=${total_steps},block_length=${block_length},use_cache=True,dual_cache=True,factor=1.0,show_speed=True \
#   --batch_size 1 --limit 100   # 先用 100 条快速对比，全量去掉 --limit

# ==================== 2. 你的 SlowFast 配置 (动态阈值) ====================
echo "=========================================="
echo "Running YOUR SLOWFAST VERSION"
echo "=========================================="

accelerate launch --num_processes 1 eval_llada.py \
  --tasks ${task} --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args model_path=${model_path},gen_length=${length},steps=${total_steps},block_length=${block_length},use_cache=True,dual_cache=True,show_speed=True \
  --slowfast True \
  --slow_steps 6 \
    --slow_threshold 0.98 \
    --fast_threshold 0.45 \
    --converge_window 2 \
    --min_converge_var 0.02\
  --batch_size 4 --limit 100

echo "=========================================="
echo "All done! Compare Tokens per second and avg nfe above."
echo "=========================================="