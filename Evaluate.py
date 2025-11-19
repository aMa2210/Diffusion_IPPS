import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import copy
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
from Industrial_Pipeline_Functions import load_ipps_problem_from_json, get_ipps_problem_data
import random

@dataclass
class Workpiece:
    name: str
    optional_machines: List[List[int]]
    processing_time: List[List[int]]


def load_problem_definitions(json_path: str) -> Tuple[List[Workpiece], Dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    workpieces_objs = []
    for wp_data in data['workpieces']:

        new_wp = Workpiece(
            name=wp_data['name'],
            optional_machines=wp_data['optional_machines'],
            processing_time=wp_data['processing_time']
        )
        workpieces_objs.append(new_wp)

    machine_power = {
        1: {'no_load': 0.8, 'processing': 2.4},
        2: {'no_load': 1.2, 'processing': 3.6},
        3: {'no_load': 1.4, 'processing': 4.1},
        4: {'no_load': 0.8, 'processing': 3.8},
        5: {'no_load': 1.3, 'processing': 2.1},
        6: {'no_load': 1.5, 'processing': 4.3}
    }

    print(f"Task loaded from {json_path} with {len(workpieces_objs)} workpieces")
    return workpieces_objs, machine_power


def calculate_energy_consumption(completed_operations: List[Dict], makespan: float, machine_power: Dict) -> Dict:
    energy_consumption = {}
    total_energy = 0

    for machine_id, power_info in machine_power.items():
        machine_ops = [op for op in completed_operations if op['machine'] == machine_id]
        if machine_ops:
            total_processing_time = sum(op['processing_time'] for op in machine_ops)
            total_no_load_time = makespan - total_processing_time
            processing_energy = total_processing_time * power_info['processing']
            no_load_energy = total_no_load_time * power_info['no_load']
            total_machine_energy = processing_energy + no_load_energy
            energy_consumption[machine_id] = {
                'processing_time': total_processing_time, 'no_load_time': total_no_load_time,
                'processing_energy': processing_energy, 'no_load_energy': no_load_energy,
                'total_energy': total_machine_energy, 'utilization': (total_processing_time / makespan) * 100
            }
            total_energy += total_machine_energy
        else:
            no_load_energy = makespan * power_info['no_load']
            energy_consumption[machine_id] = {
                'processing_time': 0, 'no_load_time': makespan, 'processing_energy': 0,
                'no_load_energy': no_load_energy, 'total_energy': no_load_energy, 'utilization': 0.0
            }
            total_energy += no_load_energy

    energy_consumption['total'] = {
        'processing_energy': sum(v['processing_energy'] for k, v in energy_consumption.items() if k != 'total'),
        'no_load_energy': sum(v['no_load_energy'] for k, v in energy_consumption.items() if k != 'total'),
        'total_energy': total_energy, 'makespan': makespan
    }
    return energy_consumption


def simulate_complete_scheduling(workpiece_cycles: List[Tuple], machine_power_data: Dict) -> Tuple[
    Dict, Dict, List[Dict]]:
    # ==========================================
    # 1. 解析输入并生成 "固定" 的机器队列
    # ==========================================
    all_ops_map = {}
    machine_queues = {m: [] for m in machine_power_data.keys()}
    total_ops_count = 0

    for item in workpiece_cycles:
        if len(item) == 4:
            wp_name, selected_machines, processing_times, priorities = item
        else:
            wp_name, selected_machines, processing_times = item
            priorities = [random.random() for _ in range(len(selected_machines))]

        for feature_idx, (mid, proc_time, priority) in enumerate(zip(selected_machines, processing_times, priorities)):
            current_feat = feature_idx + 1
            operation = {
                'workpiece': wp_name,
                'feature': current_feat,
                'machine': mid,
                'processing_time': proc_time,
                'priority': priority,
                'start_time': None,
                'end_time': None
            }

            # 记录到全局查找表，方便查前置状态
            all_ops_map[(wp_name, current_feat)] = operation

            if mid in machine_queues:
                machine_queues[mid].append(operation)

            total_ops_count += 1

    # 核心：预先按优先级排序 (High -> Low)
    for mid in machine_queues:
        machine_queues[mid].sort(key=lambda x: (-x['priority'], x['workpiece'], x['feature']))

    # ==========================================
    # 2. 执行 List Scheduling (允许跳过不可行任务)
    # ==========================================

    machine_available_time = {m: 0 for m in machine_power_data.keys()}
    job_available_time = {name: 0 for name in [c[0] for c in workpiece_cycles]}
    completed_operations = []

    # 安全计数器 (虽然 List Scheduling 不易死锁，但保留以防图结构本身非法)
    steps = 0
    max_steps = total_ops_count * 5  # 给多一点宽容度

    while len(completed_operations) < total_ops_count:
        if steps > max_steps:
            raise RuntimeError("Simulation stuck! (Possible invalid graph structure or extreme deadlock)")

        progress_made = False

        # 遍历每一台机器
        for mid in list(machine_queues.keys()):
            queue = machine_queues[mid]
            if not queue:
                continue

            # --- 🚀 核心修改区域 START ---

            target_op_index = -1
            target_op = None

            # 遍历队列，寻找 *第一个* 满足物理条件 (前置已完成) 的任务
            for idx, op in enumerate(queue):
                wp_name = op['workpiece']
                feat_idx = op['feature']

                # 检查前置工序是否完成
                is_ready = True
                if feat_idx > 1:
                    prev_op_key = (wp_name, feat_idx - 1)
                    # 如果前置工序的 end_time 还是 None，说明没做完
                    if all_ops_map[prev_op_key]['end_time'] is None:
                        is_ready = False

                if is_ready:
                    # 找到了！这是当前队列中优先级最高且 *能做* 的任务
                    target_op = op
                    target_op_index = idx
                    break  # 停止扫描，锁定这个任务

            # 如果找到了能做的任务，执行它
            if target_op is not None:
                op = target_op
                wp_name = op['workpiece']

                job_ready_t = job_available_time[wp_name]
                machine_ready_t = machine_available_time[mid]

                # 计算时间 (取最大值：体现了如果机器空闲但工件没来，机器会空转等待)
                start_t = max(job_ready_t, machine_ready_t)
                end_t = start_t + op['processing_time']

                # 更新 Operation 状态
                op['start_time'] = start_t
                op['end_time'] = end_t

                # 更新全局时钟
                machine_available_time[mid] = end_t
                job_available_time[wp_name] = end_t

                # 提交结果
                completed_operations.append(op)

                # 从队列中移除 (注意是用 index 移除，不仅仅是 pop(0))
                queue.pop(target_op_index)

                progress_made = True

            # --- 🚀 核心修改区域 END ---

        if not progress_made:
            steps += 1
        else:
            steps = 0

    completion_times = job_available_time
    makespan = max(completion_times.values()) if completion_times else 0
    energy = calculate_energy_consumption(completed_operations, makespan, machine_power_data)

    return completion_times, energy, completed_operations

def create_gantt_chart(completed_operations, title="Gantt Chart"):
    fig, ax = plt.subplots(figsize=(14, 8))
    machines = sorted(list(set(op['machine'] for op in completed_operations)))
    workpieces = sorted(list(set(op['workpiece'] for op in completed_operations)))
    colors = plt.cm.Set3(np.linspace(0, 1, len(workpieces)))
    color_map = {wp: colors[i] for i, wp in enumerate(workpieces)}

    for operation in completed_operations:
        m_idx = machines.index(operation['machine'])
        start, dur = operation['start_time'], operation['processing_time']
        ax.barh(m_idx, dur, left=start, height=0.6, color=color_map[operation['workpiece']], edgecolor='black')
        ax.text(start + dur / 2, m_idx, f"F{operation['feature']}", ha='center', va='center', fontsize=8)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'Machine {m}' for m in machines])
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return plt


def graph_to_simulation_input(edge_matrix, ipps_canvas, all_workpieces_objs):

    num_ops = ipps_canvas.op_info.size(0)
    assignments = {}

    for op_node_idx in range(num_ops):
        wp_idx, feat_idx = ipps_canvas.op_info[op_node_idx].tolist()
        if wp_idx not in assignments: assignments[wp_idx] = {}

        connected_nodes = edge_matrix[op_node_idx].nonzero(as_tuple=True)[0]

        machine_node_idx = -1
        for node_idx in connected_nodes:
            if node_idx >= num_ops:
                machine_node_idx = node_idx.item()
                break

        if machine_node_idx != -1:
            real_machine_id = ipps_canvas.machine_map[machine_node_idx - num_ops].item()
            assignments[wp_idx][feat_idx] = real_machine_id
        else:
            assignments[wp_idx][feat_idx] = None

    workpiece_cycles = []

    for wp_idx, wp_obj in enumerate(all_workpieces_objs):
        wp_assignments = assignments.get(wp_idx, {})
        selected_machines = []
        processing_times = []

        num_features = len(wp_obj.optional_machines)

        for feat_idx in range(num_features):
            chosen_machine = wp_assignments.get(feat_idx)

            if chosen_machine is not None:
                selected_machines.append(chosen_machine)
                try:
                    option_index = wp_obj.optional_machines[feat_idx].index(chosen_machine) # to find the corresponding process time
                    time = wp_obj.processing_time[feat_idx][option_index]
                    processing_times.append(time)
                except ValueError:
                    print(f"❌ Mistake: {chosen_machine} is not a valid choice for workpiece: {wp_obj.name} process: {feat_idx}")
                    processing_times.append(0)
            else:
                print(f"❌No chosen machine for workpiece{wp_idx} process{feat_idx}")
                sys.exit(1)


        workpiece_cycles.append((wp_obj.name, selected_machines, processing_times))

    return workpiece_cycles



if __name__ == "__main__":
    RUN_NAME = "baseline"
    PROBLEM_FILE = "TestSet/1.json"
    ablation_dir = Path("ablation_runs_11_19_for_RL")
    samples_path = ablation_dir / RUN_NAME / "samples.pt"

    print(f"📂 Loading problem definitions from: {PROBLEM_FILE}")
    all_workpieces, machine_power_data = load_problem_definitions(PROBLEM_FILE)

    raw_wp_dicts, raw_machines = load_ipps_problem_from_json(PROBLEM_FILE)
    ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, "cpu")

    if not samples_path.exists():
        print(f"❌ Path not found: {samples_path}")
        sys.exit(1)

    print(f"Loading samples...")
    samples = torch.load(samples_path)
    print(f"Found {len(samples)} samples.")

    makespan_list = []
    energy_list = []
    best_makespan = float('inf')
    best_sample_idx = -1
    best_results = None

    print("Running simulations...")
    for idx, sample in enumerate(samples):
        _, edge_matrix = sample

        wp_cycles = graph_to_simulation_input(edge_matrix, ipps_canvas, all_workpieces)

        try:
            completion, energy_report, ops = simulate_complete_scheduling(wp_cycles, machine_power_data)

            makespan = energy_report['total']['makespan']
            total_energy = energy_report['total']['total_energy']

            makespan_list.append(makespan)
            energy_list.append(total_energy)

            if makespan < best_makespan:
                best_makespan = makespan
                best_sample_idx = idx
                best_results = (completion, energy_report, ops)

        except Exception as e:
            print(f"⚠️ Sample {idx} simulation failed: {e}")

    if makespan_list:
        avg_makespan = np.mean(makespan_list)
        avg_energy = np.mean(energy_list)

        print("\n" + "=" * 40)
        print("📊 EVALUATION RESULTS")
        print("=" * 40)
        print(f"Problem File:     {PROBLEM_FILE}")
        print(f"Total Samples:    {len(makespan_list)}")
        print(f"Average Makespan: {avg_makespan:.2f}")
        print(f"Best Makespan:    {best_makespan:.2f} (Sample {best_sample_idx})")
        print(f"Average Energy:   {avg_energy:.2f}")
        print("=" * 40)

        if best_results:
            print(f"\n🎨 Plotting Gantt chart for Best Sample ({best_sample_idx})...")
            _, _, best_ops = best_results
            plt_obj = create_gantt_chart(best_ops,
                                         title=f"Best Solution (Makespan: {best_makespan}) - Sample {best_sample_idx}")

            out_file = f"{RUN_NAME}_best_gantt.png"
            plt_obj.savefig(out_file)
            print(f"✅ Gantt chart saved to: {out_file}")
            plt_obj.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(makespan_list, energy_list, alpha=0.6, c='blue', edgecolors='k')
        plt.xlabel("Makespan")
        plt.ylabel("Total Energy")
        plt.title(f"Distribution of Generated Solutions ({len(samples)} samples)\nFile: {PROBLEM_FILE}")
        plt.grid(True, alpha=0.3)

        dist_file = f"{RUN_NAME}_distribution.png"
        plt.savefig(dist_file)
        print(f"✅ Distribution plot saved to: {dist_file}")
        plt.show()
    else:
        print("❌ No valid simulations completed.")