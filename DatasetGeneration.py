import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import copy
import os
import sys
import json

@dataclass
class Workpiece:
    name: str
    optional_machines: List[List[int]]
    processing_time: List[List[int]]
    
    def print_info(self):
        print(f"Workpiece: {self.name}")
        for feature in range(len(self.optional_machines)):
            print(f"Feature {feature+1}:")
            print(f"- Optional Machines: {self.optional_machines[feature]}")
            print(f"- Processing Time: {self.processing_time[feature]}")
        print()


def load_workpieces_from_json(json_path: str) -> List[Workpiece]:
    if not os.path.exists(json_path):
        print(f"❌ Error: File not found at {json_path}")
        sys.exit(1)

    with open(json_path, 'r') as f:
        data = json.load(f)

    loaded_workpieces = []
    for wp_data in data['workpieces']:
        new_wp = Workpiece(
            name=wp_data['name'],
            optional_machines=wp_data['optional_machines'],
            processing_time=wp_data['processing_time']
        )
        loaded_workpieces.append(new_wp)

    print(f"✅ Successfully loaded {len(loaded_workpieces)} workpieces from {json_path}")
    return loaded_workpieces

machine_power = {
    1: {'no_load': 0.8, 'processing': 2.4},
    2: {'no_load': 1.2, 'processing': 3.6},
    3: {'no_load': 1.4, 'processing': 4.1},
    4: {'no_load': 0.8, 'processing': 3.8},
    5: {'no_load': 1.3, 'processing': 2.1},
    6: {'no_load': 1.5, 'processing': 4.3}
}

def plot_makespan_vs_energy(dataset: pd.DataFrame, filename: str = "makespan_vs_energy.png"):
    """
    Create a scatter plot of makespan vs energy consumption for all simulations
    
    Args:
        dataset: DataFrame with simulation results
        filename: Name of the output image file
    """
    # Create Images folder if it doesn't exist
    images_folder = "Images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    # Create full path
    full_path = os.path.join(images_folder, filename)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    makespan = dataset['makespan']
    energy = dataset['total_energy']
    
    # Create scatter plot
    scatter = ax.scatter(makespan, energy, alpha=0.7, s=50, c=dataset['simulation_id'], 
                        cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Add labels and title
    ax.set_xlabel('Makespan (time units)', fontsize=12)
    ax.set_ylabel('Total Energy Consumption (kWh)', fontsize=12)
    ax.set_title('Makespan vs Energy Consumption\n(All Simulations)', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics annotations
    avg_makespan = makespan.mean()
    avg_energy = energy.mean()
    corr_coef = np.corrcoef(makespan, energy)[0, 1]
    
    stats_text = f'Statistics:\n'
    stats_text += f'Average Makespan: {avg_makespan:.1f}\n'
    stats_text += f'Average Energy: {avg_energy:.1f} kWh\n'
    stats_text += f'Correlation: {corr_coef:.3f}\n'
    stats_text += f'Total Simulations: {len(dataset)}'
    
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    # Add trend line
    z = np.polyfit(makespan, energy, 1)
    p = np.poly1d(z)
    ax.plot(makespan, p(makespan), "r--", alpha=0.8, linewidth=2, label='Trend line')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    # Check if file was created
    if os.path.exists(full_path):
        print(f"✅ Scatter plot saved as: {full_path}")
    else:
        print(f"❌ ERROR: Scatter plot was not created!")
    
    plt.show()
    
    # Print additional statistics
    print(f"\n📊 MAKESPAN VS ENERGY STATISTICS:")
    print(f"   Correlation coefficient: {corr_coef:.3f}")
    print(f"   Makespan range: {makespan.min():.1f} - {makespan.max():.1f}")
    print(f"   Energy range: {energy.min():.1f} - {energy.max():.1f} kWh")

def generate_multiple_simulations(workpieces: List[Workpiece], n_simulations: int, seed: int = None) -> pd.DataFrame:
    """
    Generate multiple production cycles and save results to Excel
    
    Args:
        n_simulations: Number of simulations to run
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with all simulation results
    """
    if seed is not None:
        random.seed(seed)
    
    # List to store all simulation results
    all_results = []
    
    print(f"🚀 Generating {n_simulations} simulations...")
    
    for sim_id in range(1, n_simulations + 1):
        # Generate random cycles for each workpiece
        workpiece_cycles = []
        cycle_details = {}

        for workpiece in workpieces:
            selected_machines, processing_times = generate_random_cycle(workpiece)
            workpiece_cycles.append((workpiece.name, selected_machines, processing_times))
            cycle_details[workpiece.name] = {
                'machines': selected_machines,
                'processing_times': processing_times
            }
        
        # Simulate scheduling
        completion_times, energy_consumption, completed_operations = simulate_complete_scheduling(workpiece_cycles)
        
        makespan = max(completion_times.values())
        total_energy = energy_consumption['total']['total_energy']
        
        # Create result entry
        result = {
            'simulation_id': sim_id,
            'makespan': makespan,
            'total_energy': total_energy,
            'processing_energy': energy_consumption['total']['processing_energy'],
            'no_load_energy': energy_consumption['total']['no_load_energy']
        }
        
        # Add cycle details for each workpiece
        for wp in workpieces:
            wp_name = wp.name
            cycle_info = cycle_details[wp_name]
            result[f'{wp_name}_machines'] = str(cycle_info['machines'])
            result[f'{wp_name}_processing_times'] = str(cycle_info['processing_times'])
            result[f'{wp_name}_completion_time'] = completion_times[wp_name]
        
        # Add machine utilization and energy details
        for machine_id in range(1, 7):
            machine_info = energy_consumption[machine_id]
            result[f'machine_{machine_id}_utilization'] = machine_info['utilization']
            result[f'machine_{machine_id}_energy'] = machine_info['total_energy']
        
        all_results.append(result)
        
        # Progress update
        if sim_id % 10 == 0:
            print(f"📊 Completed {sim_id}/{n_simulations} simulations")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to Excel
    excel_filename = "Dataset.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Main results sheet
        df.to_excel(writer, sheet_name='Simulation_Results', index=False)
        
        # Summary statistics sheet
        summary_data = {
            'Metric': ['Total Simulations', 'Average Makespan', 'Std Makespan', 
                      'Average Energy', 'Std Energy', 'Min Makespan', 'Max Makespan',
                      'Min Energy', 'Max Energy'],
            'Value': [
                n_simulations,
                df['makespan'].mean(),
                df['makespan'].std(),
                df['total_energy'].mean(),
                df['total_energy'].std(),
                df['makespan'].min(),
                df['makespan'].max(),
                df['total_energy'].min(),
                df['total_energy'].max()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        # Machine power settings sheet
        power_data = []
        for machine_id, power_info in machine_power.items():
            power_data.append({
                'Machine': machine_id,
                'No-load Power': power_info['no_load'],
                'Processing Power': power_info['processing']
            })
        power_df = pd.DataFrame(power_data)
        power_df.to_excel(writer, sheet_name='Machine_Power_Settings', index=False)
    
    print(f"✅ Dataset saved as '{excel_filename}'")
    print(f"📁 File location: {os.path.abspath(excel_filename)}")
    print(f"📊 Total simulations: {n_simulations}")
    print(f"📈 Average makespan: {df['makespan'].mean():.2f}")
    print(f"⚡ Average energy: {df['total_energy'].mean():.2f} kWh")
    
    # Create makespan vs energy plot
    plot_makespan_vs_energy(df)
    
    return df

def calculate_energy_consumption(completed_operations: List[Dict], makespan: float) -> Dict:
    """
    Calculate energy consumption for all machines considering the makespan as total time.
    
    Energy consumption = Processing Energy + No-load Energy
    - Processing Energy: sum(processing_time * processing_power) for each machine
    - No-load Energy: (makespan - total_processing_time) * no_load_power for each machine
    
    Args:
        completed_operations: List of completed operations with start/end times
        makespan: Total schedule time
    
    Returns:
        Dictionary with energy consumption details for each machine and total
    """
    energy_consumption = {}
    total_energy = 0
    
    for machine_id, power_info in machine_power.items():
        # Get all operations for this machine
        machine_ops = [op for op in completed_operations if op['machine'] == machine_id]
        
        if machine_ops:
            # Calculate total processing time for this machine
            total_processing_time = sum(op['processing_time'] for op in machine_ops)
            
            # Calculate no-load time (from time 0 to makespan, minus processing time)
            # This assumes machines are available from time 0 and remain on until makespan
            total_no_load_time = makespan - total_processing_time
            
            # Calculate energy consumption
            processing_energy = total_processing_time * power_info['processing']
            no_load_energy = total_no_load_time * power_info['no_load']
            total_machine_energy = processing_energy + no_load_energy
            
            energy_consumption[machine_id] = {
                'processing_time': total_processing_time,
                'no_load_time': total_no_load_time,
                'processing_energy': processing_energy,
                'no_load_energy': no_load_energy,
                'total_energy': total_machine_energy,
                'utilization': (total_processing_time / makespan) * 100  # Percentage utilization
            }
            total_energy += total_machine_energy
        else:
            # Machine had no operations - only no-load energy for entire makespan
            no_load_energy = makespan * power_info['no_load']
            
            energy_consumption[machine_id] = {
                'processing_time': 0,
                'no_load_time': makespan,
                'processing_energy': 0,
                'no_load_energy': no_load_energy,
                'total_energy': no_load_energy,
                'utilization': 0.0
            }
            total_energy += no_load_energy
    
    # Add total energy and summary statistics
    energy_consumption['total'] = {
        'processing_energy': sum(energy_consumption[m]['processing_energy'] for m in range(1, 7)),
        'no_load_energy': sum(energy_consumption[m]['no_load_energy'] for m in range(1, 7)),
        'total_energy': total_energy,
        'makespan': makespan
    }
    
    return energy_consumption

def print_energy_consumption(energy_consumption: Dict):
    """
    Print detailed energy consumption report
    """
    print("\n" + "="*80)
    print("ENERGY CONSUMPTION REPORT")
    print("="*80)
    
    total_info = energy_consumption['total']
    
    # Print machine details
    for machine_id in range(1, 7):
        machine_info = energy_consumption[machine_id]
        print(f"\nMachine {machine_id}:")
        print(f"  Processing Time: {machine_info['processing_time']:6.1f} units")
        print(f"  No-load Time:    {machine_info['no_load_time']:6.1f} units")
        print(f"  Utilization:     {machine_info['utilization']:6.1f}%")
        print(f"  Processing Energy: {machine_info['processing_energy']:8.2f} kWh")
        print(f"  No-load Energy:    {machine_info['no_load_energy']:8.2f} kWh")
        print(f"  Total Energy:      {machine_info['total_energy']:8.2f} kWh")
    
    # Print summary
    print("\n" + "-"*80)
    print("SUMMARY:")
    print(f"Makespan: {total_info['makespan']} time units")
    print(f"Total Processing Energy: {total_info['processing_energy']:8.2f} kWh")
    print(f"Total No-load Energy:    {total_info['no_load_energy']:8.2f} kWh")
    print(f"TOTAL ENERGY CONSUMPTION: {total_info['total_energy']:8.2f} kWh")
    print("="*80)

def simulate_complete_scheduling(workpiece_cycles: List[Tuple]) -> Tuple[Dict, Dict, List[Dict]]:
    to_process_operations = []
    workpiece_progress = {}  
    
    for wp_name, selected_machines, processing_times in workpiece_cycles:
        operations = []
        for feature_idx, (machine, proc_time) in enumerate(zip(selected_machines, processing_times)):
            operation = {
                'workpiece': wp_name,
                'feature': feature_idx + 1,
                'machine': machine,
                'processing_time': proc_time,
                'start_time': None,
                'end_time': None
            }
            operations.append(operation)
        
        operations.sort(key=lambda x: x['feature'])
        to_process_operations.extend(operations)
        workpiece_progress[wp_name] = {
            'next_operation_index': 0,
            'operations': operations,
            'last_end_time': 0
        }
    
    machine_status = {
        1: {'current_operation': None, 'available_from': 0},
        2: {'current_operation': None, 'available_from': 0},
        3: {'current_operation': None, 'available_from': 0},
        4: {'current_operation': None, 'available_from': 0},
        5: {'current_operation': None, 'available_from': 0},
        6: {'current_operation': None, 'available_from': 0}
    }
    
    completed_operations = []
    
    def get_available_operations():
        available = []
        for wp_name, progress in workpiece_progress.items():
            if progress['next_operation_index'] < len(progress['operations']):
                operation = progress['operations'][progress['next_operation_index']]
                if operation['feature'] == 1 or progress['last_end_time'] > 0:
                    available.append(operation)
        return available
    
    # Scheduling
    time = 0
    max_time = 1000  

    while any(progress['next_operation_index'] < len(progress['operations']) 
              for progress in workpiece_progress.values()) and time < max_time:
        
        # PHASE 1: Free all machines that have completed at current time
        machines_freed = False
        for machine_id, status in machine_status.items():
            if (status['current_operation'] is not None and 
                status['available_from'] <= time):
                status['current_operation'] = None
                machines_freed = True
        
        # PHASE 2: Assign ALL available operations to free machines
        assigned_any = False
        available_operations = get_available_operations()
        
        # Continue assigning while there are available operations and free machines
        while True:
            machine_candidates = {machine_id: [] for machine_id in machine_status.keys()}
            
            # Collect candidates for each machine
            for operation in available_operations:
                machine_id = operation['machine']
                if (machine_status[machine_id]['current_operation'] is None and 
                    machine_status[machine_id]['available_from'] <= time):
                    
                    wp_name = operation['workpiece']
                    wp_progress = workpiece_progress[wp_name]
                    start_time = max(time, wp_progress['last_end_time'])
                    
                    machine_candidates[machine_id].append((operation, start_time))
            
            # Assign for each machine (priority: operations with lower start_time)
            new_assignments = []
            for machine_id, candidates in machine_candidates.items():
                if candidates:
                    # Sort by start_time (lowest first) and then by processing time (shortest first)
                    candidates.sort(key=lambda x: (x[1], x[0]['processing_time']))
                    selected_operation, start_time = candidates[0]
                    new_assignments.append((machine_id, selected_operation, start_time))
            
            # Execute assignments
            for machine_id, operation, start_time in new_assignments:
                end_time = start_time + operation['processing_time']
                
                # Update operation
                operation['start_time'] = start_time
                operation['end_time'] = end_time
                
                # Update machine status
                machine_status[machine_id]['current_operation'] = operation
                machine_status[machine_id]['available_from'] = end_time
                
                # Update workpiece progress
                wp_name = operation['workpiece']
                workpiece_progress[wp_name]['next_operation_index'] += 1
                workpiece_progress[wp_name]['last_end_time'] = end_time
                
                # Add to completed operations
                completed_operations.append(copy.deepcopy(operation))
                
                # Remove from available operations list
                if operation in available_operations:
                    available_operations.remove(operation)
                    to_process_operations.remove(operation)
                
                assigned_any = True
            
            # If no new assignments, exit the loop
            if not new_assignments:
                break
        
        # PHASE 3: Advance to next event only if no assignments were made
        if not assigned_any and not machines_freed:
            # Find next completion event
            next_completion_time = float('inf')
            for machine_id, status in machine_status.items():
                if status['current_operation'] is not None and status['available_from'] < next_completion_time:
                    next_completion_time = status['available_from']
            
            # If no imminent completions, advance to next available time
            if next_completion_time == float('inf'):
                next_available_time = min(status['available_from'] for status in machine_status.values())
                time = next_available_time
            else:
                time = next_completion_time
        else:
            # If assignments were made or machines were freed,
            # stay at same time to process other opportunities
            continue
    
    # Calculate performance metrics
    completion_times = {wp_name: progress['last_end_time'] 
                       for wp_name, progress in workpiece_progress.items()}
    
    makespan = max(completion_times.values()) if completion_times else 0
    
    # Calculate energy consumption
    energy_consumption = calculate_energy_consumption(completed_operations, makespan)
    
    return completion_times, energy_consumption, completed_operations

def generate_random_cycle(workpiece: Workpiece) -> Tuple[List[int], List[int]]:
    selected_machines = []
    processing_times = []
    
    for feature_idx in range(len(workpiece.optional_machines)):
        machine_options = workpiece.optional_machines[feature_idx]
        time_options = workpiece.processing_time[feature_idx]
        
        selected_idx = random.randint(0, len(machine_options) - 1)
        selected_machine = machine_options[selected_idx]
        selected_time = time_options[selected_idx]
        
        selected_machines.append(selected_machine)
        processing_times.append(selected_time)
    
    return selected_machines, processing_times

def create_gantt_chart(completed_operations: List[Dict], filename: str = "optimal_solution_gantt.png"):
    """
    Create a Gantt chart showing the scheduling of operations on machines
    """
    # Create Images folder if it doesn't exist
    images_folder = "Images"
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    # Create full path
    full_path = os.path.join(images_folder, filename)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique machines and workpieces
    machines = sorted(list(set(op['machine'] for op in completed_operations)))
    workpieces = sorted(list(set(op['workpiece'] for op in completed_operations)))
    
    # Define colors for different workpieces
    colors = plt.cm.Set3(np.linspace(0, 1, len(workpieces)))
    color_map = {wp: colors[i] for i, wp in enumerate(workpieces)}
    
    # Plot each operation as a horizontal bar
    for operation in completed_operations:
        machine_idx = machines.index(operation['machine'])
        start = operation['start_time']
        duration = operation['processing_time']
        
        # Create the bar
        bar = plt.barh(machine_idx, duration, left=start, 
                       height=0.6, color=color_map[operation['workpiece']],
                       edgecolor='black', alpha=0.8)
        
        # Add label inside the bar (operation details)
        label_x = start + duration / 2
        label_y = machine_idx
        label_text = f"F{operation['feature']}"
        
        ax.text(label_x, label_y, label_text, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='black')
        
        # Add start time label on the left
        ax.text(start, machine_idx - 0.25, f"{start}", 
                ha='left', va='top', fontsize=7, color='blue')
        
        # Add end time label on the right
        end_time = start + duration
        ax.text(end_time, machine_idx - 0.25, f"{end_time}", 
                ha='right', va='top', fontsize=7, color='red')
    
    # Customize the chart
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'Machine {m}' for m in machines])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart - Production Scheduling')
    
    # Add grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[wp], 
                                    edgecolor='black', label=wp) 
                      for wp in workpieces]
    ax.legend(handles=legend_elements, title='Workpieces', 
              loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Get absolute path for debugging
    absolute_path = os.path.abspath(full_path)
    print(f"💾 Saving Gantt chart to: {absolute_path}")
    
    # Save the figure
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    
    # Check if file was created
    if os.path.exists(full_path):
        print(f"✅ SUCCESS: Gantt chart saved in Images folder!")
    else:
        print(f"❌ ERROR: File was not created!")
    
    # plt.show()

def print_schedule_details(completed_operations: List[Dict]):
    """
    Print detailed schedule information
    """
    print("SCHEDULE DETAILS:")
    print("=" * 80)
    
    # Sort operations by machine and start time
    sorted_ops = sorted(completed_operations, key=lambda x: (x['machine'], x['start_time']))
    
    for machine in sorted(set(op['machine'] for op in sorted_ops)):
        machine_ops = [op for op in sorted_ops if op['machine'] == machine]
        print(f"\nMachine {machine}:")
        print("-" * 40)
        
        for op in machine_ops:
            print(f"  Time [{op['start_time']:3d}-{op['end_time']:3d}]: "
                  f"{op['workpiece']}-F{op['feature']} "
                  f"(Duration: {op['processing_time']:2d})")

# Main execution
if __name__ == "__main__":
    # Option 1: Run single simulation (as before)
    print("🎯 SINGLE SIMULATION")
    print("=" * 50)
    PROBLEM_FILE = "TestSet/1.json"
    loaded_workpieces = load_workpieces_from_json(PROBLEM_FILE)
    # Generate random cycles for each workpiece
    random.seed(42)  # For reproducible results
    
    workpiece_cycles = []
    for workpiece in loaded_workpieces:
        selected_machines, processing_times = generate_random_cycle(workpiece)
        workpiece_cycles.append((workpiece.name, selected_machines, processing_times))
    
    # Simulate scheduling
    completion_times, energy_consumption, completed_operations = simulate_complete_scheduling(workpiece_cycles)
    
    # Print results
    print("COMPLETION TIMES:")
    for workpiece, time in completion_times.items():
        print(f"{workpiece}: {time} time units")
    
    print(f"\nMAKESPAN: {max(completion_times.values())} time units")
    
    # Print detailed schedule
    print_schedule_details(completed_operations)
    
    # Print energy consumption report
    print_energy_consumption(energy_consumption)
    
    # Create Gantt chart
    create_gantt_chart(completed_operations, "optimal_solution_gantt.png")
    
    # Option 2: Generate multiple simulations dataset
    print("\n🎯 MULTIPLE SIMULATIONS DATASET")
    print("=" * 50)
    
    # Generate 100 simulations and save to Excel
    dataset = generate_multiple_simulations(workpieces=loaded_workpieces, n_simulations=500, seed=42)