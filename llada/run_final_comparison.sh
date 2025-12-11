#!/bin/bash

# ==================== 环境变量 ====================
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# ==================== 通用参数 ====================
task=gsm8k
num_fewshot=5
length=256
block_length=32
total_steps=${length}
model_path='GSAI-ML/LLaDA-8B-Instruct'

# ==================== 结果记录 ====================
MAIN_LOG="final_full_comparison.log"
mkdir -p logs

echo "========== Baseline vs SlowFast 全量 GSM8K 对比（$(date)） ==========" > "${MAIN_LOG}"
echo "任务: ${task}, fewshot: ${num_fewshot}, gen_length: ${length}, block_length: ${block_length}" >> "${MAIN_LOG}"
echo "" >> "${MAIN_LOG}"

#

# ==================== 2. SlowFast Combo 1: 4,0.97,0.6,3,0.02 (2 轮) ====================
echo "==========================================" | tee -a "${MAIN_LOG}"
echo "Running SlowFast Combo 1: 4,0.97,0.6,3,0.02 (2 轮，全量)" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"

for run in 1 2; do
  echo ">>> SlowFast Combo 1 Run ${run}/2 开始" | tee -a "${MAIN_LOG}"
  log_file="logs/slowfast_combo1_run${run}.log"

  accelerate launch  --num_processes 4  eval_llada.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${total_steps},block_length=${block_length},use_cache=True,dual_cache=True,show_speed=True \
    --slowfast True \
    --slow_steps 4 \
    --slow_threshold 0.97 \
    --fast_threshold 0.60 \
    --converge_window 3 \
    --min_converge_var 0.02 \
    --batch_size 1 2>&1 | tee "${log_file}"

  echo ">>> SlowFast Combo 1 Run ${run}/2 完成" | tee -a "${MAIN_LOG}"
  echo "=== SlowFast Combo 1 Run ${run}/2 结果摘要 ===" >> "${MAIN_LOG}"
  grep -E "Tokens per second|Total NFE|exact_match|Stderr" "${log_file}" >> "${MAIN_LOG}"
  echo "" >> "${MAIN_LOG}"
done

# ==================== 3. SlowFast Combo 2: 4,0.99,0.7,3,0.06 (推荐，质量最稳) ====================
echo "==========================================" | tee -a "${MAIN_LOG}"
echo "Running SlowFast Combo 2: 4,0.99,0.7,3,0.06 (2 轮，全量) - 推荐" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"

for run in 1 2; do
  echo ">>> SlowFast Combo 2 Run ${run}/2 开始" | tee -a "${MAIN_LOG}"
  log_file="logs/slowfast_combo2_run${run}.log"

  accelerate launch  --num_processes 4 eval_llada.py \
    --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path=${model_path},gen_length=${length},steps=${total_steps},block_length=${block_length},use_cache=True,dual_cache=True,show_speed=True \
    --slowfast True \
    --slow_steps 4 \
    --slow_threshold 0.99 \
    --fast_threshold 0.70 \
    --converge_window 3 \
    --min_converge_var 0.06 \
    --batch_size 1 2>&1 | tee "${log_file}"

  echo ">>> SlowFast Combo 2 Run ${run}/2 完成" | tee -a "${MAIN_LOG}"
  echo "=== SlowFast Combo 2 Run ${run}/2 结果摘要 ===" >> "${MAIN_LOG}"
  grep -E "Tokens per second|Total NFE|exact_match|Stderr" "${log_file}" >> "${MAIN_LOG}"
  echo "" >> "${MAIN_LOG}"
done

echo "==========================================" | tee -a "${MAIN_LOG}"
echo "所有对比全部完成！" | tee -a "${MAIN_LOG}"
echo "完整结果见 ${MAIN_LOG} 和 logs/ 目录" | tee -a "${MAIN_LOG}"
echo "==========================================" | tee -a "${MAIN_LOG}"