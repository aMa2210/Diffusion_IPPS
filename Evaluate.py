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
    # 1. 初始化与严格定序
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
            all_ops_map[(wp_name, current_feat)] = operation
            if mid in machine_queues:
                machine_queues[mid].append(operation)
            total_ops_count += 1

    # 🔒 严格定序：完全按照 AI 的优先级排队
    for mid in machine_queues:
        machine_queues[mid].sort(key=lambda x: (-x['priority'], x['workpiece'], x['feature']))

    # ==========================================
    # 2. 仿真执行 (带死锁恢复)
    # ==========================================

    machine_available_time = {m: 0 for m in machine_power_data.keys()}
    job_available_time = {name: 0 for name in [c[0] for c in workpiece_cycles]}
    completed_operations = []

    while len(completed_operations) < total_ops_count:

        progress_made = False

        # --- 🅰️ 阶段 A: 尝试严格执行 (Strict Execution) ---
        # 遍历每台机器，只检查它的队首 (Queue Head)
        for mid in list(machine_queues.keys()):
            queue = machine_queues[mid]
            if not queue: continue

            op = queue[0]  # 只看第一个！

            # 检查前置条件
            is_ready = True
            if op['feature'] > 1:
                prev_op = all_ops_map[(op['workpiece'], op['feature'] - 1)]
                if prev_op['end_time'] is None:
                    is_ready = False

            if is_ready:
                # 执行任务
                job_ready_t = job_available_time[op['workpiece']]
                machine_ready_t = machine_available_time[mid]
                start_t = max(job_ready_t, machine_ready_t)
                end_t = start_t + op['processing_time']

                op['start_time'] = start_t
                op['end_time'] = end_t
                machine_available_time[mid] = end_t
                job_available_time[op['workpiece']] = end_t

                completed_operations.append(op)
                queue.pop(0)  # 移除队首

                progress_made = True
                # 这里的策略是：一旦有机器动了，系统状态就变了，
                # 我们立刻重新开始循环，看看这个变动是否解锁了其他机器的队首
                # (这有助于保持严格顺序)

        if progress_made:
            continue  # 继续下一轮严格检查

        # --- 🅱️ 阶段 B: 死锁恢复 (Deadlock Recovery) ---
        # 如果代码走到这里，说明：所有机器的队首任务都卡住了 (progress_made = False)
        # 这就是死锁。我们需要打破它。

        # 策略：扫描所有队列中的 *非队首* 任务，找到优先级最高的 *可行* 任务插队
        best_rescue_op = None
        best_rescue_mid = -1
        best_rescue_idx = -1

        for mid, queue in machine_queues.items():
            # 从第2个任务开始看 (因为第1个已经确诊卡住了)
            for idx, op in enumerate(queue):
                if idx == 0: continue

                # 检查是否可行
                is_ready = True
                if op['feature'] > 1:
                    prev_op = all_ops_map[(op['workpiece'], op['feature'] - 1)]
                    if prev_op['end_time'] is None:
                        is_ready = False

                if is_ready:
                    # 找到一个能动的！
                    # 我们挑选 priority 最高的那个来 "救火"
                    if best_rescue_op is None or op['priority'] > best_rescue_op['priority']:
                        best_rescue_op = op
                        best_rescue_mid = mid
                        best_rescue_idx = idx

        if best_rescue_op:
            # 🚑 执行救援任务 (插队执行)
            op = best_rescue_op
            mid = best_rescue_mid

            # 标准执行逻辑
            job_ready_t = job_available_time[op['workpiece']]
            machine_ready_t = machine_available_time[mid]
            start_t = max(job_ready_t, machine_ready_t)
            end_t = start_t + op['processing_time']

            op['start_time'] = start_t
            op['end_time'] = end_t
            machine_available_time[mid] = end_t
            job_available_time[op['workpiece']] = end_t

            completed_operations.append(op)
            machine_queues[mid].pop(best_rescue_idx)  # 从队列中间移除

            # print(f"⚠️ Deadlock resolved by swapping: {op['workpiece']}-F{op['feature']} on Machine {mid}")

        else:
            # 如果连救援任务都找不到，说明图本身不连通或有逻辑错误
            raise RuntimeError("Unresolvable Deadlock! The graph structure might be invalid.")

    # ==========================================
    # 3. 结算
    # ==========================================
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


def graph_to_simulation_input(edge_matrix, ipps_canvas, all_workpieces_objs, priorities_tensor=None):

    num_ops = ipps_canvas.op_info.size(0)
    assignments = {}
    priority_map = {}

    for op_node_idx in range(num_ops):
        wp_idx, feat_idx = ipps_canvas.op_info[op_node_idx].tolist()
        if wp_idx not in assignments:
            assignments[wp_idx] = {}
            priority_map[wp_idx] = {}
        if priorities_tensor is not None:
            priority_val = priorities_tensor[op_node_idx].item()
            priority_map[wp_idx][feat_idx] = priority_val
        else:
            priority_map[wp_idx][feat_idx] = 0.0
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
        wp_priorities_map = priority_map.get(wp_idx, {})

        selected_machines = []
        processing_times = []
        current_priorities = []
        num_features = len(wp_obj.optional_machines)

        for feat_idx in range(num_features):
            chosen_machine = wp_assignments.get(feat_idx)
            p_val = wp_priorities_map.get(feat_idx, 0.0)
            current_priorities.append(p_val)

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


        workpiece_cycles.append((wp_obj.name, selected_machines, processing_times, current_priorities))

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