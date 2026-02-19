import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import time
import glob
from tqdm import tqdm
from pathlib import Path
import json
import csv
from math import ceil
import re

from Industrial_Pipeline_Functions import (
    LightweightIndustrialDiffusion,
    load_ipps_problem_from_json,
    get_ipps_problem_data,
    validate_constraints
)
from Evaluate import (
    load_problem_definitions,
    simulate_complete_scheduling,
    graph_to_simulation_input
)

model_weight_path = 'SL_Run_Fix_Batching_121'
MODEL_PATH = f"SPT_checkpoints/{model_weight_path}/sl_model_1749.pth"
base_dir = f"SPT_checkpoints/{model_weight_path}"
config_path = os.path.join(base_dir, "config.json")


if os.path.exists(config_path):
    print(f"Loading config file: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    print(f"⚠️ Warning: Config file not found at {config_path}, using defaults.")
    config = {}


TEST_DATA_DIR = "Problem_TestSet_GA_large"
OUTPUT_FILE = f"test_results_{model_weight_path}.json" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNCERTAINTY_LEVEL = 0
TIME_GUIDANCE_SCALE = 0
POS_SCALER = 0

T_STEPS = config.get("T_STEPS", 8)
HIDDEN_DIM = config.get("HIDDEN_DIMENSION", 128)
NUM_LAYERS = config.get("NUM_LAYERS", 6)
N_HEADS = config.get("N_HEADS", 4)
TEMPERATURE_METHOD = 'cosine'

print(f'Model Config: T={T_STEPS}, Hidden={HIDDEN_DIM}, Layers={NUM_LAYERS}')
print(f'Test Directory: {TEST_DATA_DIR}')

# ===========================================

def save_to_csv(data_list, filename):
    """
    保存结果到CSV
    data_list: list of tuples (filename, makespan, time, valid)
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['Filename', 'Makespan', 'Energy', 'Inference_Time(s)', 'Valid']) 
            writer.writerows(data_list)
        print(f"✅ CSV saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to save CSV {filename}: {e}")

def create_gantt_chart(completed_operations, title="Gantt Chart", filename=None):
    """
    绘制甘特图
    """
    if not completed_operations:
        print("⚠️ No operations to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    raw_workpieces = list(set(op['workpiece'] for op in completed_operations))
    
    def extract_number(text):
        match = re.search(r'\d+', str(text))
        return int(match.group()) if match else 0

    workpieces = sorted(raw_workpieces, key=extract_number)
    colors = plt.cm.tab20(np.linspace(0, 1, len(workpieces)))
    color_map = {wp: colors[i] for i, wp in enumerate(workpieces)}

    for operation in completed_operations:
        m_id = operation['machine']
        wp = operation['workpiece']
        start = operation['start_time']
        dur = operation['processing_time']
        
        wp_num = extract_number(wp)
        label_text = f"J{wp_num-1}"
        
        ax.barh(y=m_id, width=dur, left=start, 
                height=0.6, align='center', 
                color=color_map[wp], edgecolor='black', alpha=0.9)
        
        ax.text(start + dur / 2, m_id, label_text, 
                ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    machines = sorted(list(set(op['machine'] for op in completed_operations)))
    ax.set_yticks(machines)
    ax.set_yticklabels([f"M-{m}" for m in machines])
    
    ax.set_ylabel("Machines")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        # print(f"📊 Gantt chart saved to {filename}")
        plt.close(fig)
    else:
        plt.show()

def run_ai_solver(model, problem_file, workpieces_objs, machine_power_data, device, batch_size=4):

    # 1. 构建画布
    raw_wp_dicts, raw_machines = load_ipps_problem_from_json(problem_file)
    ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, device)

    # 2. 模型推理 (计时开始)
    start_time = time.time()
    
    generated_edges_batch, _, _, priorities_batch = model.reverse_diffusion_with_logprob(
        ipps_canvas, 
        device, 
        num_samples=batch_size,
        time_guidance_scale=TIME_GUIDANCE_SCALE, 
        position_guidance_scale=POS_SCALER, 
        temperature_method=TEMPERATURE_METHOD,
        greedy=False     # 贪婪模式
    )
    
    best_mk = float('inf')
    best_eng = float('inf')
    best_schedule = None
    valid_count = 0
    
    # 3. 验证结构合法性
    node_labels = ipps_canvas.x.argmax(dim=1)
    
    edges_indices_batch = generated_edges_batch.argmax(dim=-1).detach().cpu()
    priorities_batch = priorities_batch.detach().cpu() # [B, N]

    for i in range(batch_size):
        # A. 提取单个解
        edges_matrix = edges_indices_batch[i]
        priorities = priorities_batch[i]
        
        # B. 验证合法性
        is_valid = validate_constraints(edges_matrix, node_labels, device, exact=True, data=ipps_canvas)
        
        if not is_valid:
            continue
            
        valid_count += 1
        
        try:
            wp_cycles = graph_to_simulation_input(edges_matrix, ipps_canvas, workpieces_objs, priorities)
            _, energy_report, completed_ops = simulate_complete_scheduling(
                wp_cycles, 
                machine_power_data, 
                time_uncertainty=UNCERTAINTY_LEVEL
            )
            mk = energy_report['total']['makespan']
            eng = energy_report['total']['total_energy']
            
            # D. 更新最优解
            if mk < best_mk:
                best_mk = mk
                best_eng = eng
                best_schedule = completed_ops
                
        except Exception as e:
            # print(f"Sample {i} Sim Error: {e}")
            continue
    end_time = time.time()
    inference_time = end_time - start_time
    
    return best_mk, best_eng, True, inference_time, best_schedule
    

def plot_results(results_df):
    """
    绘制结果分布直方图 (代替之前的Bar Chart，因为现在没有固定的Size组)
    """
    if results_df.empty or 'Makespan' not in results_df.columns:
        print("No valid data to plot.")
        return

    # 过滤掉无效解 (-1)
    valid_df = results_df[results_df['Valid'] == "True"]
    
    if valid_df.empty:
        print("No valid solutions found.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(valid_df['Makespan'].astype(float), bins=20, color='royalblue', alpha=0.7, edgecolor='black')
    plt.xlabel('Makespan')
    plt.ylabel('Count')
    plt.title(f'Makespan Distribution (Valid Solutions: {len(valid_df)}/{len(results_df)})')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f"Test_Distribution_{model_weight_path}.png", dpi=300)
    print("\n✅ Distribution plot saved.")
    plt.show()

def main():
    
    csv_data_model = []
    
    gantt_dir = Path("Gantt_Charts_Trainset_GA_Trained_Diffusion_1749_Checkpoint121_BS4_large")
    gantt_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载模型
    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = LightweightIndustrialDiffusion(
        T=T_STEPS, hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, nhead=N_HEADS, device=DEVICE
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # 开启评估模式
    except FileNotFoundError:
        print(f"❌ Model file not found at {MODEL_PATH}. Please check the path.")
        return

    # 2. 获取测试文件列表
    if not os.path.exists(TEST_DATA_DIR):
        print(f"❌ Test directory not found: {TEST_DATA_DIR}")
        return
        
    test_files = glob.glob(os.path.join(TEST_DATA_DIR, "*.json"))
    test_files.sort() # 排序，保证顺序一致
    
    if not test_files:
        print(f"⚠️ No JSON files found in {TEST_DATA_DIR}")
        return
        
    print(f"\n🚀 Starting Test on {len(test_files)} files from '{TEST_DATA_DIR}'...")

    total_mk = 0
    valid_count = 0
    
    # 3. 循环测试
    for i, problem_path in enumerate(tqdm(test_files)):
        filename = os.path.basename(problem_path)
        
        try:
            workpieces_objs, machine_power_data = load_problem_definitions(problem_path)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            csv_data_model.append((filename, -1, -1, 0, "LoadError"))
            continue

        # B. 运行模型
        with torch.no_grad():
            mk, eng, is_valid, inf_time, schedule = run_ai_solver(
                model, 
                problem_path, 
                workpieces_objs, 
                machine_power_data, 
                DEVICE
            )

        # C. 记录结果
        status = "True" if is_valid else "False"
        csv_data_model.append((filename, mk, eng, f"{inf_time:.4f}", status))
        
        if is_valid:
            valid_count += 1
            total_mk += mk
            
            # (可选) 画前5个有效解的甘特图，或者你可以根据文件名画特定的
            if valid_count <= 5: 
                gantt_path = gantt_dir / f"Gantt_{filename.replace('.json', '')}_MK{int(mk)}.png"
                create_gantt_chart(
                    schedule, 
                    title=f"{filename}: MK={mk:.1f}, Time={inf_time:.3f}s", 
                    filename=str(gantt_path)
                )

    # 4. 统计与保存
    print(f"\n📊 Test Finished!")
    print(f"Total Files: {len(test_files)}")
    print(f"Valid Solutions: {valid_count} ({valid_count/len(test_files)*100:.1f}%)")
    if valid_count > 0:
        print(f"Average Makespan (Valid): {total_mk / valid_count:.2f}")

    print(f"💾 Saving results to CSV...")
    save_to_csv(csv_data_model, f"result_trainset_{model_weight_path}_1749_BS4_large.csv")

    # 简单的可视化
    df = pd.DataFrame(csv_data_model, columns=['Filename', 'Makespan', 'Energy', 'Inference_Time', 'Valid'])
    plot_results(df)

if __name__ == "__main__":
    main()