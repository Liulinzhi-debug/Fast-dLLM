import os
import subprocess
import re
import itertools
import csv
import time

# ==================== 1. 配置参数网格 ====================
# 这里填入你想要搜索的参数范围
SLOW_STEPS = [4, 5]
SLOW_THRESH = [0.97, 0.99]
FAST_THRESH = [0.50, 0.60, 0.70, 0.80]
CONVERGE_WINDOW = [3, 4]
MIN_CONVERGE_VAR = [0.02, 0.04, 0.06]

# 固定参数
TASK = "gsm8k"
NUM_FEWSHOT = 5
LENGTH = 256
BLOCK_LENGTH = 32
TOTAL_STEPS = LENGTH
MODEL_PATH = 'GSAI-ML/LLaDA-8B-Instruct'
LIMIT = 100  # 快速验证

# 结果保存文件
OUTPUT_CSV = "grid_search_results.csv"

# ==================== 2. 环境设置 ====================
env = os.environ.copy()
env["HF_ALLOW_CODE_EVAL"] = "1"
env["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

def parse_logs(log_output):
    """从日志输出中提取关键指标 (修正版)"""
    metrics = {
        "tps": 0.0,
        "nfe": 0,
        "acc": 0.0,
        "stderr": 0.0
    }
    
    # 1. 提取 Tokens per second
    tps_match = re.search(r"Tokens per second:\s+([\d\.]+)", log_output)
    if tps_match:
        metrics["tps"] = float(tps_match.group(1))
        
    # 2. 提取 Total NFE
    nfe_match = re.search(r"Total NFE is\s+(\d+)", log_output)
    if nfe_match:
        metrics["nfe"] = int(nfe_match.group(1))
        
    # 3. 提取 Flexible Extract 的 Value (使用字符串分割法，更稳健)
    # 表格结构: |gsm8k| 3|flexible-extract| 5|exact_match|↑ | 0.67|± |0.0473|
    # 分割后索引: [0]  [1] [2]      [3]      [4]     [5]    [6]  [7]  [8]   [9]
    lines = log_output.split('\n')
    for line in lines:
        # 锁定目标行
        if "gsm8k" in line and "flexible-extract" in line:
            try:
                # 按 | 分割并去除首尾空格
                parts = [p.strip() for p in line.split('|')]
                
                # parts[7] 对应 Value (0.67)
                if len(parts) > 7 and parts[7]:
                    metrics["acc"] = float(parts[7])
                
                # parts[9] 对应 Stderr (0.0473)
                if len(parts) > 9 and parts[9]:
                    metrics["stderr"] = float(parts[9])
                    
            except (ValueError, IndexError):
                print(f"Warning: Parse error on line: {line}")
                continue
                
    return metrics
    
def run_grid_search():
    # 生成所有参数组合
    combinations = list(itertools.product(
        SLOW_STEPS, SLOW_THRESH, FAST_THRESH, CONVERGE_WINDOW, MIN_CONVERGE_VAR
    ))
    
    print(f"Total combinations to test: {len(combinations)}")
    
    # 初始化 CSV 文件头
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Slow Steps", "Slow Thresh", "Fast Thresh", "Conv Window", "Min Var",
            "Accuracy", "StdErr", "TPS", "NFE", "Status"
        ])

    best_config = None
    best_acc = 0.0

    # 开始循环
    for idx, (s_step, s_thresh, f_thresh, c_win, min_var) in enumerate(combinations):
        print(f"\n[{idx+1}/{len(combinations)}] Running: s_step={s_step}, s_th={s_thresh}, f_th={f_thresh}, win={c_win}, var={min_var}")
        
        # 构造命令
        # 注意：这里模拟了你脚本中的调用方式
        cmd = [
            "accelerate", "launch", "--num_processes", "1", "eval_llada.py",
            "--tasks", TASK,
            "--num_fewshot", str(NUM_FEWSHOT),
            "--confirm_run_unsafe_code",
            "--model", "llada_dist",
            "--model_args", f"model_path={MODEL_PATH},gen_length={LENGTH},steps={TOTAL_STEPS},block_length={BLOCK_LENGTH},use_cache=True,dual_cache=True,show_speed=True",
            "--slowfast", "True",
            "--slow_steps", str(s_step),
            "--slow_threshold", str(s_thresh),
            "--fast_threshold", str(f_thresh),
            "--converge_window", str(c_win),
            "--min_converge_var", str(min_var),
            "--batch_size", "1",
            "--limit", str(LIMIT)
        ]

        try:
            # 执行命令并捕获输出
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                env=env, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, # 将 stderr 合并到 stdout 以便捕获
                text=True
            )
            duration = time.time() - start_time
            
            output_log = result.stdout
            
            # 解析日志
            metrics = parse_logs(output_log)
            
            # 打印简报
            print(f"  -> Done in {duration:.1f}s. Acc: {metrics['acc']}, TPS: {metrics['tps']}, NFE: {metrics['nfe']}")
            
            # 判断是否满足目标
            status = "FAIL"
            if metrics['acc'] > 0.75:
                status = "PASS (>0.75)"
                print(f"  Found valid config! Acc: {metrics['acc']}")
                
                # 记录最优
                if metrics['acc'] > best_acc:
                    best_acc = metrics['acc']
                    best_config = (s_step, s_thresh, f_thresh, c_win, min_var)

            # 写入 CSV
            with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    s_step, s_thresh, f_thresh, c_win, min_var,
                    metrics['acc'], metrics['stderr'], metrics['tps'], metrics['nfe'], status
                ])

        except Exception as e:
            print(f"  Error running config: {e}")

    print("\n==========================================")
    print("Grid Search Finished!")
    print(f"Results saved to: {OUTPUT_CSV}")
    if best_config:
        print(f"Best Accuracy: {best_acc}")
        print(f"Best Config: {best_config}")
    else:
        print("No config exceeded 0.75 accuracy.")
    print("==========================================")

if __name__ == "__main__":
    run_grid_search()