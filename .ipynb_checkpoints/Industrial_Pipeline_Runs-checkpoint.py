# ===== run_and_plot.py =====
import os, random, collections
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
from pathlib import Path
from glob import glob
import numpy as np
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Industrial_Pipeline_Functions import (
    experiment_free, experiment_allpinned, experiment_partial,
    wl_hash, plant_valid
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1) GENERATION ----------n
# Choose ONE experiment:

# experiment_settings = ['E1', 'E2', 'E3']
experiment_settings = ['E2']
# plant_model_path = 'ablation_runs_new_guidance_30epoch_repeat1-3/baseline/model.pth'
# plant_model_path = 'ablation_runs_new_guidance_30epoch_repeat1-3/baseline/model.pth'
plant_model_path = 'ablation_runs_new_no_guidance_epoch_60/baseline/model.pth'
for experiment_setting in experiment_settings:
    if experiment_setting == 'E1':
        file_name_output = experiment_free(n_samples=300, n_nodes=15, plant_model_path = plant_model_path)              # creates E1_*.pt
    elif experiment_setting == 'E2':
        file_name_output = experiment_allpinned(n_samples=300, inv=(3,4,2,1), plant_model_path = plant_model_path)          # creates E2_*.pt
    elif experiment_setting == 'E3':
        file_name_output = experiment_partial(n_samples=300, n_nodes=15, pin_ratio=0.3, plant_model_path = plant_model_path)  # creates E3_*.pt

# Save the generated graphs with a specific name
import datetime


# Assuming experiment_free returns a batch of dicts with 'nodes' and 'edges'
# If experiment_free does not return, but saves internally, skip this block
# Otherwise, collect and save manually:
# batch = experiment_free(n_samples=10, n_nodes=15)
# torch.save(batch, save_path)
print(f"Graphs just generated have been saved as: {os.path.abspath(file_name_output)}")

# # ---------- 2) CHOOSE SAVED FILE ----------
# TAG = "E1"  # "E1" if you ran free, "E3" if partial
# matches = sorted(Path("exp_outputs").glob(f"{TAG}_*.pt"))
# if not matches:
#     raise FileNotFoundError(f"No file {TAG}_*.pt found in exp_outputs/. Did you run the corresponding experiment?")
# file_path = str(matches[-1])  # most recent
print("Selected file for visualization:", file_name_output)

# ---------- 3) PLOTTING ----------
node_styles = {
    0: {'marker': 's', 'color': 'blue',   'label': 'MACHINE'},
    1: {'marker': 'v', 'color': 'red',    'label': 'BUFFER'},
    2: {'marker': 'D', 'color': 'green',  'label': 'ASSEMBLY'},
    3: {'marker': 'h', 'color': 'orange', 'label': 'DISASSEMBLY'},
}

def to_numpy_adj(A):
    if isinstance(A, torch.Tensor):
        A = A.cpu()
        return A.numpy()
    return np.asarray(A)

def normalize_types(types, label2id):
    if len(types) == 0:
        return []
    first = types[0]
    if isinstance(first, str):
        return [label2id[t] for t in types]
    elif isinstance(first, (int, np.integer)) or (isinstance(first, torch.Tensor) and first.ndim == 0):
        return [int(t) for t in types]
    else:
        return [label2id[t] for t in types]

data = torch.load(file_name_output, map_location="cpu")
adj_matrices = data["adjacency_matrices"]
node_types   = data["node_types"]
label2id     = data["label2id"]

output_folder = f"exp_outputs/{experiment_setting}/Figures"
os.makedirs(output_folder, exist_ok=True)


num_graphs = len(adj_matrices)  # limit to first 5 graphs
print("Number of graphs to plot:", len(adj_matrices))
for i in range(num_graphs):
    print(f"Plotting graph {i}...")
    A = adj_matrices[i]
    types = node_types[i]

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    handles = []
    for ntype, props in node_styles.items():
        nodelist = [idx for idx, t in enumerate(types) if t == ntype]
        if not nodelist:
            continue
        col = nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist,
            node_shape=props['marker'],
            node_color=props['color'],
            node_size=400, label=props['label']
        )
        handles.append(col)

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrows=True, arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(f"Industrial Graph {i}")
    plt.axis('off')
    if handles:
        plt.legend(scatterpoints=1)

    out_png = os.path.join(output_folder, f"industrial_graph_{i}.png")
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[{i}] Saved: {os.path.abspath(out_png)}")
