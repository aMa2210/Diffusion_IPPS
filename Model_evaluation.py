import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
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
from Generate_random_problem_instances import generate_random_ipps_problem

model_weight_path = 'rl_new0110_decay_scaler(TScaler0.01)_introduce_advantage_resume_from_ep999(LR1e-7)'
MODEL_PATH = f"rl_checkpoints/{model_weight_path}/model_ep599.pth"
base_dir = f"rl_checkpoints/{model_weight_path}"
config_path = os.path.join(base_dir, "config.json")
if os.path.exists(config_path):
    print(f"Loading config file: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    raise FileNotFoundError(f"config file does not exist: {config_path}")
TEST_SIZES = [10, 30, 50, 100]  # 要测试的工件数量 (Job sizes)
NUM_MACHINES = [5, 5, 10, 10]  # 固定机器数量，模拟车间规模
OUTPUT_FILE = f"generalization_test_results_{model_weight_path}.json"
NUM_INSTANCES = 10  # 每个尺寸生成多少个问题
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNCERTAINTY_LEVEL = 0

T_STEPS = config.get("T_STEPS", 8)
print(f'T_STEPS{T_STEPS}')
HIDDEN_DIM = config.get("HIDDEN_DIMENSION", 128)
NUM_LAYERS = config.get("NUM_LAYERS", 6)
N_HEADS = config.get("N_HEADS", 4)
TIME_GUIDANCE_SCALE = config.get("T_SCALER", config.get("T_SCALER_END", 0.001))
POS_SCALER = config.get("Pos_SCALER", config.get("POS_SCALER_END", 2.0))

print(TIME_GUIDANCE_SCALE)
print(POS_SCALER)
TEMPERATURE_METHOD = 'cosine'


def save_to_csv(data_list, filename):
    """
    将结果列表保存为 CSV 文件
    data_list: list of tuples (filename, makespan)
    filename: 输出文件名
    """
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['Filename', 'Best_Makespan']) # 表头
            writer.writerows(data_list)
        print(f"✅ CSV saved: {filename}")
    except Exception as e:
        print(f"❌ Failed to save CSV {filename}: {e}")
        
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
        ipps_canvas, device, time_guidance_scale=TIME_GUIDANCE_SCALE, position_guidance_scale=POS_SCALER, temperature_method=TEMPERATURE_METHOD,
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
        _, energy_report, _ = simulate_complete_scheduling(wp_cycles, machine_power_data, time_uncertainty = UNCERTAINTY_LEVEL)
        return energy_report['total']['makespan'], energy_report['total']['total_energy']
    except Exception as e:
        print(f"Sim Error: {e}")
        return float('inf'), float('inf')

def crossover_batch(elites, num_offspring, data, device):
    """
    【回滚版】节点级(Node-wise)交叉：
    对图中的每一个节点（工序），独立随机决定继承父代A还是父代B的连接策略。
    这虽然破坏了工件连贯性，但能打散全局资源分配，配合 Diffusion Refine 可能效果更好。
    """
    elite_edges = torch.stack([p["edges_onehot"] for p in elites], dim=0) 
    elite_prios = torch.stack([p["priorities"] for p in elites], dim=0)
    
    K = len(elites)
    # num_nodes = elite_edges.size(1) # 这行没用到，可以注释掉

    # 1. 随机选择父代索引
    parent1_idx = torch.randint(0, K, (num_offspring,), device=device)
    parent2_idx = torch.randint(0, K, (num_offspring,), device=device)

    # === A. 优先级交叉 (Blend Crossover) - 保持不变 ===
    # 优先级的混合总是好的，因为它提供了连续空间的搜索
    alpha = torch.rand((num_offspring, 1), device=device)
    child_prios = alpha * elite_prios[parent1_idx] + (1 - alpha) * elite_prios[parent2_idx]

    # === B. 节点级图结构交叉 (Node-wise Random Exchange) ===
    
    e1 = elite_edges[parent1_idx] # [B, N, N, C]
    e2 = elite_edges[parent2_idx]
    
    # 生成 Mask: [B, N, 1, 1]
    # 对每一个节点(行)，抛硬币决定选 P1 还是 P2
    # 0.5 的概率
    mask = (torch.rand((num_offspring, e1.size(1), 1, 1), device=device) > 0.5).float()
    
    # 混合
    child_edges = mask * e1 + (1 - mask) * e2
    
    return child_edges, child_prios

    
def run_evolutionary_solver(model, problem_file, workpieces_objs, machine_power_data, device,
                            pop_size=30, keep_size=10, num_generations=3,
                            rollback_t=2):
    """
    进化求解器：集成生成、筛选、变异(热力学对齐)、修复过程。
    """
    # 1. 准备数据
    raw_wp_dicts, raw_machines = load_ipps_problem_from_json(problem_file)
    ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, device)
    node_labels = ipps_canvas.x.argmax(dim=1)

    # 2. 初始种群生成 (Generation 0)
    # 使用全局配置的 TIME_GUIDANCE_SCALE 和 POS_SCALER
    edges_batch, _, _, priorities_batch = model.reverse_diffusion_with_logprob(
        ipps_canvas, device, num_samples=pop_size,
        time_guidance_scale=TIME_GUIDANCE_SCALE, position_guidance_scale=POS_SCALER,
        temperature_method=TEMPERATURE_METHOD, greedy=True
    )

    best_solution = {"makespan": float('inf'), "energy": float('inf')}

    # 3. 进化循环
    for gen in range(num_generations):
        # --- A. 评估 (Evaluation) ---
        population = []
        e_indices_cpu = edges_batch.argmax(dim=-1).detach().cpu()
        prio_cpu = priorities_batch.detach().cpu()

        for i in range(pop_size):
            mk, eng = float('inf'), float('inf')
            # 验证有效性
            if validate_constraints(e_indices_cpu[i], node_labels, device, exact=True, data=ipps_canvas):
                wp_cycles = graph_to_simulation_input(e_indices_cpu[i], ipps_canvas, workpieces_objs, prio_cpu[i])
                _, rep, completed_ops = simulate_complete_scheduling(wp_cycles, machine_power_data, time_uncertainty = UNCERTAINTY_LEVEL)
                mk = rep['total']['makespan']
                eng = rep['total']['total_energy']
            else:
                print('error code 9685')

            population.append({
                "makespan": mk,
                "energy": eng,
                "edges_onehot": edges_batch[i],
                "priorities": priorities_batch[i],
                "schedule": completed_ops
            })

        # --- B. 筛选 (Selection) ---
        population.sort(key=lambda x: x["makespan"])
        elites = population[:keep_size]

        # 记录本代最优
        current_best = elites[0]
        if current_best['makespan'] < best_solution['makespan']:
            best_solution = current_best

        print(best_solution['makespan'])
        # 如果是最后一轮，直接结束循环
        if gen == num_generations - 1:
            break

        # --- C. 变异 (Mutation) ---
        elite_edges_stack = torch.stack([p["edges_onehot"] for p in elites], dim=0)
        elite_priorities_stack = torch.stack([p["priorities"] for p in elites], dim=0)

        # mutated_input_list = []
        num_offspring = pop_size - keep_size
        # offspring_edges, offspring_prios = crossover_batch(elites, num_new_offspring, device)
        offspring_edges, offspring_prios = crossover_batch(elites, num_offspring, ipps_canvas, device)
        # num_repeats = ceil(num_new_offspring / keep_size)

        # for _ in range(num_repeats):
        #     mutated_edges, _, rate = model.rl_structural_mutation(
        #         elite_edges_stack,
        #         elite_priorities_stack,
        #         t=rollback_t,
        #         temperature_method=TEMPERATURE_METHOD
        #     )
        #     mutated_input_list.append(mutated_edges)

        # num_repeats = ceil(num_new_offspring / keep_size)
        # large_edge_batch = elite_edges_stack.repeat(num_repeats, 1, 1, 1)
        # large_prio_batch = elite_priorities_stack.repeat(num_repeats, 1)
        # input_edges = large_edge_batch[:num_new_offspring]
        # input_prios = large_prio_batch[:num_new_offspring]
        
        # input_edges = torch.cat([elite_edges_stack, offspring_edges], dim=0)
        # input_prios = torch.cat([elite_priorities_stack, offspring_prios], dim=0)
        
        next_gen_input, mutated_prios, rate = model.rl_structural_mutation(
            offspring_edges,
            offspring_prios,
            t=rollback_t,
            temperature_method=TEMPERATURE_METHOD
        )
        
        # next_gen_input = torch.cat(mutated_input_list, dim=0)[:num_new_offspring]

        # --- D. 修复 (Refinement) ---
        refined_edges, refined_priorities = model.refine_from_intermediate(
            noisy_e=next_gen_input,
            data=ipps_canvas,
            device=device,
            start_t=rollback_t,
            hint_priorities=mutated_prios,
            time_guidance_scale=TIME_GUIDANCE_SCALE,
            temperature_method=TEMPERATURE_METHOD
        )
        elite_edges_stack = torch.stack([p["edges_onehot"] for p in elites], dim=0)
        elite_priorities_stack = torch.stack([p["priorities"] for p in elites], dim=0)
        edges_batch = torch.cat([elite_edges_stack, refined_edges], dim=0)
        priorities_batch = torch.cat([elite_priorities_stack, refined_priorities], dim=0)
    # 返回格式与 run_ai_solver 保持一致 (MakeSpan, Energy)
    return best_solution['makespan'], best_solution['energy'], best_solution['schedule']


def create_gantt_chart(completed_operations, title="Gantt Chart", filename=None):
    """
    修正版甘特图：
    1. 修复机器 ID 偏移 (显示 M-1 到 M-5)
    2. 标签对齐 JSON (显示 J1 对应 Workpiece1)
    """
    if not completed_operations:
        print("⚠️ No operations to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # ==========================================
    # 1. 数据解析与排序
    # ==========================================
    
    # 获取所有唯一的工件名称
    raw_workpieces = list(set(op['workpiece'] for op in completed_operations))
    
    # 智能排序：从字符串中提取数字进行排序 (Workpiece1, Workpiece2, ..., Workpiece10)
    # 如果只是字符串排序，Workpiece10 会排在 Workpiece2 前面
    def extract_number(text):
        match = re.search(r'\d+', str(text))
        return int(match.group()) if match else 0

    workpieces = sorted(raw_workpieces, key=extract_number)
    
    # 建立映射：Workpiece Name -> Color
    # 使用 tab20 颜色板，颜色更丰富
    colors = plt.cm.tab20(np.linspace(0, 1, len(workpieces)))
    color_map = {wp: colors[i] for i, wp in enumerate(workpieces)}

    # ==========================================
    # 2. 绘制条形图
    # ==========================================
    for operation in completed_operations:
        m_id = operation['machine']      # 假设这里是 1, 2, 3, 4, 5
        wp = operation['workpiece']      # e.g., "Workpiece1"
        start = operation['start_time']
        dur = operation['processing_time']
        
        # 提取工件编号用于显示 (Workpiece1 -> 1)
        wp_num = extract_number(wp)
        label_text = f"J{wp_num-1}" # 显示为 J1, J2...
        
        # 绘制矩形
        # 注意：直接在 y=m_id 的位置画图
        ax.barh(y=m_id, width=dur, left=start, 
                height=0.6, align='center', 
                color=color_map[wp], edgecolor='black', alpha=0.9)
        
        # 添加文字标签 (白色，加粗)
        ax.text(start + dur / 2, m_id, label_text, 
                ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # ==========================================
    # 3. 设置坐标轴
    # ==========================================
    
    # 获取所有出现过的机器ID并排序
    machines = sorted(list(set(op['machine'] for op in completed_operations)))
    
    # 设置 Y 轴刻度位置
    ax.set_yticks(machines)
    
    # 【核心修改】：直接使用机器ID显示，不加1
    # 如果你的数据里机器是 1-5，这里就显示 M-1...M-5
    ax.set_yticklabels([f"M-{m}" for m in machines])
    
    ax.set_ylabel("Machines")
    ax.set_xlabel("Time")
    ax.set_title(title)
    
    # 添加网格线方便查看时间对齐
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"📊 Gantt chart saved to {filename}")
        plt.close(fig)
    else:
        plt.show()
        

def main():
    
    need_random = False
    csv_data_random = []
    csv_data_model = []

    results = []
    gantt_dir = Path("Gantt_Charts_Diffusion_stochastic")
    gantt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Loading model from {MODEL_PATH}...")
    model = LightweightIndustrialDiffusion(T=T_STEPS, input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, nhead=N_HEADS, dropout=0.1, device=DEVICE).to(
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
            problem_filename = f"gen_{n_jobs}_job_{i}.json"
            
            problem_file = temp_dir / f"gen_{n_jobs}_job_{i}.json"
            if problem_file.exists():
                pass
            else:
                generate_random_ipps_problem(
                    filename=str(problem_file),
                    num_machines=n_machines,
                    num_workpieces=n_jobs,
                    min_ops=4, max_ops=8,  # 随机工序长度
                    min_opts=1, max_opts=3,  # 柔性程度
                    seed=None  # 不设种子以保证随机性
                )

            # B. 加载问题定义
            workpieces_objs, machine_power_data = load_problem_definitions(str(problem_file))

            #############不跑random以节省时间
            avg_rand_mk = 1000.0  # <--- 强制设置为 1000
            random_makespans.append(avg_rand_mk)
            # 即使不跑，也存入CSV数据列表以防索引报错，或者你可以选择不存
            csv_data_random.append((problem_filename, int(avg_rand_mk)))
            # C. 运行 Random Baseline
            # 运行 3 次取平均值
            # rand_mk_sum = 0
            # for _ in range(3):
            #     r_mk, _ = run_random_baseline(workpieces_objs, machine_power_data)
            #     rand_mk_sum += r_mk

            # avg_rand_mk = rand_mk_sum / 3
            # random_makespans.append(avg_rand_mk)
            # csv_data_random.append((problem_filename, int(avg_rand_mk)))
    
            with torch.no_grad():
                # current_rollback = max(1, T_STEPS // 2)
                current_rollback = max(4, 1)
                evo_mk, evo_energy, best_schedule = run_evolutionary_solver(
                    model,
                    str(problem_file),
                    workpieces_objs,
                    machine_power_data,
                    DEVICE,
                    pop_size=40,
                    keep_size=5,
                    num_generations=1,
                    rollback_t=current_rollback
                )
            if evo_mk == float('inf'):
                print(f"   ⚠️ Evo Solver failed for {problem_filename}, using inf.")
            else:
                if best_schedule is not None:
                    gantt_filename = gantt_dir / f"Gantt_{n_jobs}Jobs_{n_machines}M_Instance{i}_MK{int(evo_mk)}.png"
                    create_gantt_chart(
                        best_schedule, 
                        title=f"Schedule: {n_jobs} Jobs, Makespan: {evo_mk:.1f}", 
                        filename=str(gantt_filename)
                    )
            model_mk = evo_mk
            
            # model_mk_sum = 0
            # with torch.no_grad():
            #     for _ in range(3):
            #         mk, _ = run_ai_solver(model, str(problem_file), workpieces_objs, machine_power_data, DEVICE)
            #         model_mk_sum += mk

            # model_mk = model_mk_sum / 3
            model_makespans.append(model_mk)
            csv_data_model.append((problem_filename, int(model_mk)))

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
    print(f"\n💾 Saving CSV files...")
    if need_random:
        save_to_csv(csv_data_random, f"result_random.csv")
    save_to_csv(csv_data_model, f"result_model_{model_weight_path}_evo1_passPriority_cross.csv")

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
    plt.savefig(f"Generalization_Test_Result_{model_weight_path}_passPriority.png", dpi=300)
    print("\n✅ Test finished! Result saved to 'Generalization_Test_Result.png'")
    plt.show()


if __name__ == "__main__":
    # json_file = "generalization_test_results_1000.json"
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     results_data = json.load(f)

    # plot_results(results_data)
    main()