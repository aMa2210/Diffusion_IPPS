import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from Industrial_Pipeline_Functions import load_ipps_problem_from_json, get_ipps_problem_data


RUN_NAME = "baseline"
PROBLEM_FILE = "TestSet/1.json"
SAMPLE_INDEX = 0

problem_workpieces, problem_machines = load_ipps_problem_from_json(PROBLEM_FILE)
ipps_canvas = get_ipps_problem_data(problem_workpieces, problem_machines, "cpu")

ablation_dir = Path("ablation_runs_11_17_posterior_randomChooseIfInvalid")
samples_path = ablation_dir / RUN_NAME / "samples.pt"

if not samples_path.exists():
    print(f"Wrong path")
    sys.exit(1)

print(f"Loading samples from {samples_path}...")
samples = torch.load(samples_path)

if not samples or SAMPLE_INDEX >= len(samples):
    print(f"Index {SAMPLE_INDEX} is out of range. There are only {len(samples)} samples")
    sys.exit(1)

_, edge_matrix = samples[SAMPLE_INDEX]
node_labels_tensor = ipps_canvas.x.argmax(dim=1)
num_nodes = ipps_canvas.num_nodes
num_ops = ipps_canvas.op_info.size(0)

print("Building graph...")
G = nx.DiGraph()
node_colors = []
node_labels = {}

for i in range(num_nodes):
    node_type = node_labels_tensor[i].item()
    layer_id = 0

    if node_type == 0:
        wp_idx, feat_idx = ipps_canvas.op_info[i].tolist()
        label = f"W{wp_idx + 1}-F{feat_idx + 1}"
        node_colors.append("skyblue")
        layer_id = 0
    else:
        machine_id = ipps_canvas.machine_map[i - num_ops].item()
        label = f"M{machine_id}"
        node_colors.append("salmon")
        layer_id = 1

    G.add_node(i, layer=layer_id)
    node_labels[i] = label

src_nodes, dst_nodes = edge_matrix.nonzero(as_tuple=True)
edge_colors = []

for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
    G.add_edge(src, dst)
    if node_labels_tensor[src] == 0 and node_labels_tensor[dst] == 0:
        edge_colors.append("blue")  # Op -> Op
    else:
        edge_colors.append("black")  # Op -> Machine

print("Drawing graph...")
plt.figure(figsize=(20, 15))

pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')

nx.draw_networkx(
    G,
    pos=pos,
    labels=node_labels,
    node_color=node_colors,
    node_size=2000,
    edge_color=edge_colors,
    font_size=8,
    font_weight="bold",
    arrows=True,
    arrowstyle="-|>",
    arrowsize=20,
)

plt.xlabel("Machines")
plt.ylabel("Operations")
plt.box(False)
# output_filename = f"{RUN_NAME}_sample_{SAMPLE_INDEX}.png"
# plt.savefig(output_filename)
# print(f"✅ Figure save to: {output_filename}")
plt.show()