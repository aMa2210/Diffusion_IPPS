import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import os
import json

# --- 1. 导入你的函数 ---
from Industrial_Pipeline_Functions import (
    load_ipps_problem_from_json,
    get_ipps_problem_data,
    LightweightIndustrialDiffusion
)
from Evaluate import (
    load_problem_definitions,
    graph_to_simulation_input,
    simulate_complete_scheduling
)

# --- 2. 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径模板
FILE_TEMPLATE = "TestSet/Generalization_Temp/gen_{size}_job_{idx}.json"
MODEL_PATH = "rl_checkpoints/rl_multi_generalization_BATCH_SIZE16_T_STEPS4/model_ep1999.pth"
OUTPUT_DIR = "Analysis_Stepwise_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 实验设置
PROBLEM_SIZES = [10, 30, 50, 100]
NUM_INSTANCES = 10  # 每个规模测多少个任务

T_STEPS_INFERENCE = 8
HIDDEN_DIM = 256
NUM_LAYERS = 6
NHEAD = 4
DROPOUT = 0.1 
TIME_GUIDANCE_SCALE = 0.001
NUM_REPEATS_PER_JOB = 10


# --- 3. 核心辅助函数 ---

def get_trajectory_makespans(model, json_path):
    """
    对单个文件运行推理，并返回每一步的 Makespan 列表
    """
    # A. 加载数据
    if not os.path.exists(json_path):
        print(f'{json_path} does not exist')
        return None

    all_workpieces_objs, machine_power_data = load_problem_definitions(json_path)
    raw_wp_dicts, raw_machines = load_ipps_problem_from_json(json_path)
    ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, DEVICE)

    # B. 获取轨迹
    with torch.no_grad():
        _, _, _, _, trajectory = model.reverse_diffusion_with_logprob(
            ipps_canvas,
            DEVICE,
            time_guidance_scale=TIME_GUIDANCE_SCALE,
            return_trajectory=True
        )

    # C. 遍历轨迹计算 Makespan
    step_makespans = []

    # trajectory 包含 T_STEPS_INFERENCE 个状态 (从噪声到清晰)
    for edge_indices, priorities in trajectory:
        try:
            # 转换图结构
            wp_cycles = graph_to_simulation_input(
                edge_indices,
                ipps_canvas,
                all_workpieces_objs,
                priorities
            )
            # 运行仿真
            _, energy_report, _ = simulate_complete_scheduling(wp_cycles, machine_power_data)
            mk = energy_report['total']['makespan']
            step_makespans.append(mk)
        except Exception:
            print('invalid graph')
            step_makespans.append(np.nan)

    return step_makespans


def plot_aggregated_evolution(size, all_trajectories):
    """
    绘制聚合后的进化曲线 (均值 + 标准差阴影)
    """
    # all_trajectories shape: [Num_Instances, T_Steps]
    data_matrix = np.array(all_trajectories)

    # 计算均值和标准差 (忽略 NaN)
    means = np.nanmean(data_matrix, axis=0)
    stds = np.nanstd(data_matrix, axis=0)

    steps = np.arange(1, len(means) + 1)

    plt.figure(figsize=(10, 6))

    # 绘制主曲线
    plt.plot(steps, means, color='royalblue', linewidth=2, label='Average Makespan')

    # 绘制阴影区域 (标准差范围)
    plt.fill_between(steps, means - stds, means + stds, color='royalblue', alpha=0.2, label='Standard Deviation')

    # 标注起点和终点
    if not np.isnan(means[0]):
        plt.scatter(steps[0], means[0], color='red', zorder=5, label='Start (Noisy)')
    if not np.isnan(means[-1]):
        plt.scatter(steps[-1], means[-1], color='green', zorder=5, label='End (Refined)')

    plt.xlabel(f"Denoising Step (1 to {len(steps)})")
    plt.ylabel("Makespan")
    plt.title(f"Evolution of Solution Quality - Problem Size {size}\n(Averaged over {len(all_trajectories)} Instances)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    save_path = os.path.join(OUTPUT_DIR, f"Evolution_Size_{size}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Plot saved to {save_path}")


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":

    print(f"Loading model from {MODEL_PATH}...")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    keys_to_remove = [
        "alpha_bar",
        "beta_schedule"
    ]
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    model = LightweightIndustrialDiffusion(
        T=T_STEPS_INFERENCE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        nhead=NHEAD,
        dropout=DROPOUT,
        device=DEVICE
    ).to(DEVICE)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. 遍历每个规模
    for size in PROBLEM_SIZES:
        print(f"\n🚀 Analyzing Problem Size: {size} ...")

        trajectories_buffer = []  # 存储该 Size 下所有任务的轨迹

        for i in tqdm(range(NUM_INSTANCES), desc=f"Simulating Size {size}"):
            json_file = FILE_TEMPLATE.format(size=size, idx=i)

            # traj = get_trajectory_makespans(model, json_file)
            # if traj is not None:
            #     trajectories_buffer.append(traj)
            # else:
            #     print('trajectory is None')
            single_job_trajectories = []
            for _ in range(NUM_REPEATS_PER_JOB):
                traj = get_trajectory_makespans(model, json_file)
                if traj is not None:
                    single_job_trajectories.append(traj)
                else:
                    print('traj is None')
            
            if single_job_trajectories:
                # 转换为 numpy 数组处理
                job_data = np.array(single_job_trajectories)
                # 计算均值，忽略某次失败导致的 NaN
                job_mean_traj = np.nanmean(job_data, axis=0)
                
                # 将这个"平均轨迹"加入到总 buffer
                trajectories_buffer.append(job_mean_traj.tolist())
            else:
                print(f"⚠️ Job {i} failed all {NUM_REPEATS_PER_JOB} attempts.")

        if trajectories_buffer:
            save_data = []
            for traj in trajectories_buffer:
                clean_traj = [None if np.isnan(x) else float(x) for x in traj]
                save_data.append(clean_traj)

            traj_file = os.path.join(OUTPUT_DIR, f"Trajectories_Size_{size}.json")
            with open(traj_file, 'w') as f:
                json.dump(save_data, f, indent=4)
            print(f"💾 Trajectory data saved to {traj_file}")

            plot_aggregated_evolution(size, trajectories_buffer)
        else:
            print(f"⚠️ No valid data found for Size {size}")

    print("\n✅ All analyses completed.")