import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from pathlib import Path
import json


# --- 导入你的模块 ---
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
# 假设 generate_random_ipps_problem 在 Generate_Problem.py 中
from Generate_random_problem_instances import generate_random_ipps_problem

# ================= 配置区域 =================
model_weight_path = 'rl_adding_temperature_BS32_T16_Layer6_HIDDEN_DIMENSION256'
MODEL_PATH = f"rl_checkpoints/{model_weight_path}/model_ep1999.pth"
TEST_SIZES = [10, 30, 50, 100]  # 要测试的工件数量 (Job sizes)
NUM_MACHINES = [5, 5, 10, 10]  # 固定机器数量，模拟车间规模
OUTPUT_FILE = f"generalization_test_results_{model_weight_path}.json"
NUM_INSTANCES = 10  # 每个尺寸生成多少个问题
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

T_STEPS = 16
HIDDEN_DIM = 256
NUM_LAYERS = 6
N_HEADS = 4
TIME_GUIDANCE_SCALE = 0.001
TEMPERATURE_METHOD = 'cosine'

# ===========================================

def run_random_baseline(workpieces_objs, machine_power_data):
    """
    随机基线：随机选择机器，随机生成优先级
    """
    workpiece_cycles = []

    for wp in workpieces_objs:
        selected_machines = []
        processing_times = []

        for feat_idx in range(len(wp.optional_machines)):
            options = wp.optional_machines[feat_idx]
            times = wp.processing_time[feat_idx]

            # 1. 随机选择一个合法机器
            rand_idx = random.randint(0, len(options) - 1)
            machine_id = options[rand_idx]
            proc_time = times[rand_idx]

            selected_machines.append(machine_id)
            processing_times.append(proc_time)

        # 2. 优先级将在 simulate_complete_scheduling 内部随机生成 (传入3元组)
        workpiece_cycles.append((wp.name, selected_machines, processing_times))

    # 运行模拟
    _, energy_report, _ = simulate_complete_scheduling(workpiece_cycles, machine_power_data)
    return energy_report['total']['makespan'], energy_report['total']['total_energy']


def run_ai_solver(model, problem_file, workpieces_objs, machine_power_data, device):

    # 1. 构建画布
    raw_wp_dicts, raw_machines = load_ipps_problem_from_json(problem_file)
    ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, device)

    # 2. 模型推理
    # 注意：推理时可以调高 time_guidance_scale 来增强引导
    generated_edges, _, _, priorities = model.reverse_diffusion_with_logprob(
        ipps_canvas, device, time_guidance_scale=TIME_GUIDANCE_SCALE, temperature_method=TEMPERATURE_METHOD,
    )

    edges_matrix = generated_edges.argmax(dim=-1).detach().cpu()

    # 3. 验证结构合法性
    node_labels = ipps_canvas.x.argmax(dim=1)
    is_valid = validate_constraints(edges_matrix, node_labels, device, exact=True, data=ipps_canvas)

    if not is_valid:
        print('invalid graph')
        return float('inf'), float('inf')  # 标记为失败

    # 4. 转换并模拟
    try:
        wp_cycles = graph_to_simulation_input(edges_matrix, ipps_canvas, workpieces_objs, priorities)
        _, energy_report, _ = simulate_complete_scheduling(wp_cycles, machine_power_data)
        return energy_report['total']['makespan'], energy_report['total']['total_energy']
    except Exception as e:
        print(f"Sim Error: {e}")
        return float('inf'), float('inf')


def main():
    # --- 1. 加载模型 ---
    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = LightweightIndustrialDiffusion(T=T_STEPS, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, nhead=N_HEADS, dropout=0.1, device=DEVICE).to(
        DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print("❌ Model file not found. Please check the path.")
        return

    results = []

    # --- 2. 测试循环 ---
    print("\n🚀 Starting Generalization Test...")

    temp_dir = Path("TestSet/Generalization_Temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    for n_jobs, n_machines in zip(TEST_SIZES, NUM_MACHINES):
        print(f"\n📦 Testing Problem Size: {n_jobs} Jobs (generating {NUM_INSTANCES} instances)...")

        model_makespans = []
        random_makespans = []

        for i in tqdm(range(NUM_INSTANCES)):
            # A. 生成随机问题
            problem_file = temp_dir / f"gen_{n_jobs}_job_{i}.json"
            if problem_file.exists():
                pass
            else:
                generate_random_ipps_problem(
                    filename=str(problem_file),
                    num_machines=n_machines,
                    num_workpieces=n_jobs,
                    min_ops=4, max_ops=8,  # 随机工序长度
                    min_opts=2, max_opts=4,  # 柔性程度
                    seed=None  # 不设种子以保证随机性
                )

            # B. 加载问题定义
            workpieces_objs, machine_power_data = load_problem_definitions(str(problem_file))

            # C. 运行 Random Baseline
            # 运行 3 次取最好，作为强一点的 Baseline
            rand_mk_sum = 0
            for _ in range(3):
                r_mk, _ = run_random_baseline(workpieces_objs, machine_power_data)
                rand_mk_sum += r_mk

            avg_rand_mk = rand_mk_sum / 3
            random_makespans.append(avg_rand_mk)

            model_mk_sum = 0
            with torch.no_grad():
                for _ in range(3):
                    mk, _ = run_ai_solver(model, str(problem_file), workpieces_objs, machine_power_data, DEVICE)
                    model_mk_sum += mk

            model_mk = model_mk_sum / 3
            model_makespans.append(model_mk)

        avg_model = np.mean(model_makespans)
        avg_rand = np.mean(random_makespans)
        improvement = (avg_rand - avg_model) / avg_rand * 100

        print(f"   👉 Size {n_jobs}: Random Avg={avg_rand:.1f}, Model Avg={avg_model:.1f}, Improv={improvement:.1f}%")

        results.append({
            "Size": n_jobs,
            "Random_Makespan": avg_rand,
            "AI_Makespan": avg_model,
            "Improvement_Pct": improvement,
            "AI_Raw": model_makespans,
            "Random_Raw": random_makespans
        })


    print(f"\n💾 Saving results to {OUTPUT_FILE}...")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print("✅ Save successful!")
    except Exception as e:
        print(f"❌ Save failed: {e}")

    # --- 3. 可视化结果 ---
    plot_results(results)


def plot_results(results):
    sizes = [r["Size"] for r in results]
    rand_means = [r["Random_Makespan"] for r in results]
    ai_means = [r["AI_Makespan"] for r in results]

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, rand_means, width, label='Random (Avg. of 3 runs)', color='gray', alpha=0.7)
    rects2 = ax.bar(x + width / 2, ai_means, width, label='Model (Avg. of 3 runs)', color='royalblue', alpha=0.9)

    ax.set_xlabel('Problem Size (Number of Jobs)')
    ax.set_ylabel('Average Makespan (Lower is Better)')
    ax.set_title(f'Tested on 10 instances per problem size')
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend()

    # 标注提升百分比
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        improv = results[i]["Improvement_Pct"]
        ax.annotate(f'-{improv:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(f"Generalization_Test_Result_{model_weight_path}.png", dpi=300)
    print("\n✅ Test finished! Result saved to 'Generalization_Test_Result.png'")
    plt.show()


if __name__ == "__main__":
    # json_file = "generalization_test_results_1000.json"
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     results_data = json.load(f)

    # plot_results(results_data)
    main()