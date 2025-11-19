import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- 1. Import necessary modules ---
from Industrial_Pipeline_Functions import load_ipps_problem_from_json, get_ipps_problem_data

# Import simulation and plotting logic from the Evaluate script
# (Ensure Evaluate.py is in the same directory)
from Evaluate import (
    load_problem_definitions,
    graph_to_simulation_input,
    simulate_complete_scheduling,
    create_gantt_chart
)

# --- 2. Configuration ---
RUN_NAME = "baseline"  # Run name (e.g., 'baseline' or 'rl_finetuned')
PROBLEM_FILE = "TestSet/1.json"  # Problem definition file
SAMPLE_INDEXS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]  # Which sample index to visualize (0, 1, 2...)

for SAMPLE_INDEX in SAMPLE_INDEXS:
    # --- 3. Load Data and Environment ---
    print(f"📂 Loading problem: {PROBLEM_FILE}")

    # A. Load object data required for simulation (Workpiece Objects)
    all_workpieces_objs, machine_power_data = load_problem_definitions(PROBLEM_FILE)

    # B. Load canvas data required for plotting
    raw_wp_dicts, raw_machines = load_ipps_problem_from_json(PROBLEM_FILE)
    ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, "cpu")

    # C. Load generated samples
    ablation_dir = Path("rl_checkpoints")  # Or "ablation_runs_...", adjust based on actual path
    # Adjust this path if your checkpoint directory structure differs.
    # Here we assume we are looking at samples generated previously.
    samples_path = Path(f"ablation_runs_11_19_for_RL/{RUN_NAME}/samples.pt")

    if not samples_path.exists():
        # Try alternative path
        samples_path = Path(f"ablation_runs/{RUN_NAME}/samples.pt")

    if not samples_path.exists():
        print(f"❌ Error: Cannot find samples.pt at {samples_path}")
        sys.exit(1)

    print(f"📂 Loading samples from {samples_path}...")
    samples = torch.load(samples_path)

    if not samples or SAMPLE_INDEX >= len(samples):
        print(f"❌ Error: Index {SAMPLE_INDEX} out of range. Total samples: {len(samples)}")
        sys.exit(1)

    # Get the specified sample
    _, edge_matrix = samples[SAMPLE_INDEX]

    # ==============================================================================
    # PART A: Run Simulation and Plot Gantt Chart
    # ==============================================================================
    print("\n🚀 Running Simulation for Gantt Chart...")

    try:
        # 1. Convert graph to simulation input
        wp_cycles = graph_to_simulation_input(edge_matrix, ipps_canvas, all_workpieces_objs)

        # 2. Run simulation
        completion_times, energy_report, completed_ops = simulate_complete_scheduling(wp_cycles, machine_power_data)

        makespan = energy_report['total']['makespan']
        print(f"✅ Simulation Successful!")
        print(f"📊 Makespan: {makespan}")
        print(f"⚡ Total Energy: {energy_report['total']['total_energy']:.2f}")

        # 3. Plot Gantt chart
        plt_gantt = create_gantt_chart(
            completed_ops,
            title=f"Gantt Chart - {RUN_NAME} (Sample {SAMPLE_INDEX})\nMakespan: {makespan}"
        )
        plt_gantt.show()

    except Exception as e:
        print(f"❌ Simulation Failed: {e}")
        print("Skipping Gantt chart generation.")

    # # ==============================================================================
    # # PART B: Draw Network Graph Structure
    # # ==============================================================================
    # print("\n🌐 Drawing Network Graph structure...")
    #
    # node_labels_tensor = ipps_canvas.x.argmax(dim=1)
    # num_nodes = ipps_canvas.num_nodes
    # num_ops = ipps_canvas.op_info.size(0)
    #
    # G = nx.DiGraph()
    # node_colors = []
    # node_labels_dict = {}
    #
    # for i in range(num_nodes):
    #     node_type = node_labels_tensor[i].item()
    #     layer_id = 0
    #
    #     if node_type == 0:  # Operation
    #         wp_idx, feat_idx = ipps_canvas.op_info[i].tolist()
    #         label = f"W{wp_idx + 1}-F{feat_idx + 1}"
    #         node_colors.append("skyblue")
    #         layer_id = 0
    #     else:  # Machine
    #         machine_id = ipps_canvas.machine_map[i - num_ops].item()
    #         label = f"M{machine_id}"
    #         node_colors.append("salmon")
    #         layer_id = 1
    #
    #     G.add_node(i, layer=layer_id)
    #     node_labels_dict[i] = label
    #
    # # Add edges
    # src_nodes, dst_nodes = edge_matrix.nonzero(as_tuple=True)
    # edge_colors = []
    #
    # for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
    #     G.add_edge(src, dst)
    #     if node_labels_tensor[src] == 0 and node_labels_tensor[dst] == 0:
    #         edge_colors.append("blue")  # Op -> Op (Sequential edge)
    #     else:
    #         edge_colors.append("black")  # Op -> Machine (Assignment edge)
    #
    # # Drawing
    # plt.figure(figsize=(20, 15))
    # pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')
    #
    # nx.draw_networkx(
    #     G,
    #     pos=pos,
    #     labels=node_labels_dict,
    #     node_color=node_colors,
    #     node_size=2000,
    #     edge_color=edge_colors,
    #     font_size=8,
    #     font_weight="bold",
    #     arrows=True,
    #     arrowstyle="-|>",
    #     arrowsize=20,
    # )
    #
    # plt.xlabel("Machines (Layer 1)")
    # plt.ylabel("Operations (Layer 0)")
    # plt.title(f"Graph Structure - {RUN_NAME} (Sample {SAMPLE_INDEX})")
    # plt.box(False)
    # plt.show()