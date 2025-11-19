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
    to_process_operations = []
    workpiece_progress = {}

    for wp_name, selected_machines, processing_times in workpiece_cycles:
        operations = []
        for feature_idx, (machine, proc_time) in enumerate(zip(selected_machines, processing_times)):
            operation = {'workpiece': wp_name, 'feature': feature_idx + 1, 'machine': machine,
                         'processing_time': proc_time, 'start_time': None, 'end_time': None}
            operations.append(operation)
        operations.sort(key=lambda x: x['feature'])
        to_process_operations.extend(operations)
        workpiece_progress[wp_name] = {'next_operation_index': 0, 'operations': operations, 'last_end_time': 0}

    machine_ids = list(machine_power_data.keys())
    machine_status = {m: {'current_operation': None, 'available_from': 0} for m in machine_ids}

    completed_operations = []
    time = 0
    max_time = 100000

    while any(progress['next_operation_index'] < len(progress['operations']) for progress in
              workpiece_progress.values()) and time < max_time:
        machines_freed = False
        for machine_id, status in machine_status.items():
            if status['current_operation'] is not None and status['available_from'] <= time:
                status['current_operation'] = None
                machines_freed = True

        assigned_any = False
        available_operations = []
        for wp_name, progress in workpiece_progress.items():
            if progress['next_operation_index'] < len(progress['operations']):
                operation = progress['operations'][progress['next_operation_index']]
                if progress['last_end_time'] <= time:
                    available_operations.append(operation)

        while True:
            machine_candidates = {m: [] for m in machine_status}
            for operation in available_operations:
                mid = operation['machine']
                if mid in machine_status and machine_status[mid]['current_operation'] is None and machine_status[mid][
                    'available_from'] <= time:
                    wp_p = workpiece_progress[operation['workpiece']]
                    start_t = max(time, wp_p['last_end_time'])
                    machine_candidates[mid].append((operation, start_t))

            new_assignments = []
            for mid, candidates in machine_candidates.items():
                if candidates:
                    candidates.sort(key=lambda x: (x[1], x[0]['processing_time']))
                    new_assignments.append((mid, candidates[0][0], candidates[0][1]))

            if not new_assignments: break

            for mid, op, start_t in new_assignments:
                end_t = start_t + op['processing_time']
                op['start_time'], op['end_time'] = start_t, end_t
                machine_status[mid]['current_operation'] = op
                machine_status[mid]['available_from'] = end_t
                workpiece_progress[op['workpiece']]['next_operation_index'] += 1
                workpiece_progress[op['workpiece']]['last_end_time'] = end_t
                completed_operations.append(copy.deepcopy(op))
                if op in available_operations: available_operations.remove(op)
                assigned_any = True

        if not assigned_any and not machines_freed:
            next_completion = float('inf')
            for status in machine_status.values():
                if status['current_operation'] is not None and status['available_from'] < next_completion:
                    next_completion = status['available_from']
            if next_completion == float('inf'):
                next_avail = min(status['available_from'] for status in machine_status.values())
                time = max(time + 1, next_avail)
                print('something is wrong')
            else:
                time = next_completion
        else:
            continue

    completion_times = {name: p['last_end_time'] for name, p in workpiece_progress.items()}
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
    ablation_dir = Path("ablation_runs_11_17_posterior_randomChooseIfInvalid_classifier_free_guidance_withoutKL")
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