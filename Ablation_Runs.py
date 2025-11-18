"""
Launches several trainings with different loss switches and collects validity, uniqueness, and novelty metrics.

Folder structure created:
./ablation_runs/
    └── <run_name>/
          ├── model.pth
          ├── samples.pt
          └── metrics.json

A ./ablation_summary.csv file is also generated containing all rows.
"""

import json, csv, os, shutil, torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch_geometric                       
from torch_geometric.data import Data        # clase Data
from torch_geometric.utils import to_dense_adj
from datetime import datetime
from pathlib import Path
from torch_geometric.loader import DataLoader
from Industrial_Pipeline_Functions import IndustrialGraphDataset
from Industrial_Pipeline_Functions import (
    LightweightIndustrialDiffusion,
    train_model, validate_constraints                               # <-- sí vive en industrial_diffusion.py
)
from tqdm import tqdm
from random import randint
from Industrial_Pipeline_Functions import (
    compute_edge_weights,
    compute_marginal_probs,
    load_ipps_problem_from_json,  # <-- 修改
    get_ipps_problem_data,     # <-- 修改
    validate_constraints
)



import networkx as nx
ROOT_OUT = Path("ablation_runs_11_17_posterior_randomChooseIfInvalid")
ROOT_OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = IndustrialGraphDataset(root="Trainset")
# ### WEIGHTED TRAINING
# import pandas as pd # get the metric file for the training dataset to determine which data should be repeated
# from sklearn.preprocessing import MinMaxScaler
# df = pd.read_csv('Figure_Simulation/20250929_175625/training_labels.csv')
# df['throughput_per_node'] = df['throughput'] / df['num_non_buffer_nodes']
# df['energy_per_node'] = df['total_energy'] / df['num_non_buffer_nodes']
# scaler = MinMaxScaler()
# df[['throughput_norm', 'energy_norm']] = scaler.fit_transform(df[['throughput_per_node', 'energy_per_node']])
# weight_throughput = 0.5
# weight_energy = 1-weight_throughput
# df['score'] = weight_throughput * df['throughput_norm'] - weight_energy * df['energy_norm'] # using '-' the minus before df['energy_norm'] because the less it is the better
#
# min_repeat = 1
# max_repeat = 3
# score_norm = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
# df['repeat'] = (score_norm * (max_repeat - min_repeat) + min_repeat).round().astype(int)
#
# repeat_list = df['repeat'].tolist()
# DATASET.weighted_repeat_inplace(repeat_list=repeat_list)
# ### WEIGHTED TRAINING END

LOADER  = DataLoader(DATASET, batch_size=4, shuffle=True)

EDGE_W  = compute_edge_weights(DATASET, DEVICE)
NODE_M, EDGE_M = compute_marginal_probs(DATASET, DEVICE)

PROBLEM_FILE = "TestSet/1.json"
problem_workpieces, problem_machines = load_ipps_problem_from_json(PROBLEM_FILE)
IPPS_CANVAS = get_ipps_problem_data(problem_workpieces, problem_machines, DEVICE)

EPOCHS = 30
# EPOCHS = 60
T_STEPS = 100
N_SAMP  = 300        



class GraphEncoder(torch.nn.Module):
    """Mini-GIN → mean-pool → linear  (128-D por defecto)."""
    def __init__(self, in_dim=2, hid=64, out=128):
        super().__init__()
        mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hid, hid))
        self.conv = GINConv(mlp)
        self.lin  = torch.nn.Linear(hid, out)

    def forward(self, batch):
        h = self.conv(batch.x, batch.edge_index)
        h = global_mean_pool(h, batch.batch)      # (B, hid)
        return self.lin(h)                        # (B, out)


@torch.no_grad()
def encode_graphs(list_dicts, encoder, device='cpu', bs=64):
    """Convierte tu lista de dicts {'nodes','edges'} en embeddings."""
    data_objs = []
    for g in list_dicts:
        x = torch.nn.functional.one_hot(g["nodes"], num_classes=2).float()
        edge_idx = (g["edges"] > 0).nonzero(as_tuple=False).t().contiguous()
        from torch_geometric.data import Data
        data_objs.append(Data(x=x, edge_index=edge_idx))
    loader = DataLoader(data_objs, bs, shuffle=False,
                        collate_fn=Batch.from_data_list)
    Z = []
    for batch in loader:
        Z.append(encoder(batch.to(device)).cpu())
    return torch.cat(Z, 0)                # (N, d)


def frechet(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean = cov_sqrt(cov1 @ cov2)
    return diff.dot(diff) + torch.trace(cov1 + cov2 - 2 * covmean)

def cov_sqrt(mat, eps=1e-8):
    # mat: (d,d) simétrica PSD
    evals, evecs = torch.linalg.eigh(mat)
    evals_clamped = torch.clamp(evals, min=0.)          # num. safety
    return (evecs * evals_clamped.sqrt()) @ evecs.t()



def mmd_rbf(X, Y):
    Z = torch.cat([X, Y], 0)
    sq = torch.cdist(Z, Z, p=2.0)**2
    sigma = torch.sqrt(0.5*sq[sq>0].median())
    k = lambda A,B: torch.exp(-torch.cdist(A,B,p=2.0)**2 / (2*sigma**2))
    m, n = len(X), len(Y)
    return (k(X,X).sum() - m)/(m*(m-1)) \
         + (k(Y,Y).sum() - n)/(n*(n-1)) \
         - 2*k(X,Y).mean()

from torch_geometric.data import Batch
from torch_geometric.nn   import GINConv, global_mean_pool
import numpy as np

device = DEVICE  

ENC = GraphEncoder(in_dim=2).to(device).eval()

def _embed_dataset(dataset):
    objs = []
    for d in dataset:
        x_lbl = d.x.argmax(1)
        dense = to_dense_adj(d.edge_index, max_num_nodes=d.x.size(0))[0]
        objs.append({"nodes": x_lbl, "edges": dense})
    return encode_graphs(objs, ENC, device)

_Z_TRAIN = _embed_dataset(DATASET)
MU_T, COV_T = _Z_TRAIN.mean(0), torch.from_numpy(np.cov(_Z_TRAIN.T.numpy()))

# ---------- 2. List of Variants ----------
ABLATIONS = [
    # Complete Model (24 dim)
    {"name": "baseline",
     "hidden_dim": 12, "kl": 0.1, "constraint": 1.0, "use_projector": True},

    # # same size but without the projector
    # {"name": "no_projector",
    #  "hidden_dim": 12, "kl": 0.1, "constraint": 0.0, "use_projector": False},
    #
    # # smaller network: 32 → half of 64
    # {"name": "half_hidden",
    #  "hidden_dim": 6, "kl": 0.1, "constraint": 1.0, "use_projector": True},
    #
    # # random weights + projector
    # {"name": "random_projector",
    #  "hidden_dim": 12, "kl": 0.0, "constraint": 0.0,
    #  "use_projector": True, "skip_training": True},
    #
    # {"name": "hidden_24",
    #  "hidden_dim": 24, "kl": 0.1, "constraint": 1.0, "use_projector": True},
]

def model_id_from_cfg(cfg: dict) -> str:
    return str(cfg.get("name", "run"))

# ---------- 3. Simple Metrics ----------

def wl_hash(nodes, edges):
    G = nx.Graph()
    for i, t in enumerate(nodes.tolist()):
        G.add_node(int(i), label=int(t))
    for i, j in (edges.nonzero(as_tuple=False)):
        G.add_edge(int(i), int(j))
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="label")


def compute_metrics(model, out_dir, ipps_canvas, n_samples=N_SAMP, reuse_samples=True):
    model.eval()
    samples_path = out_dir / "samples.pt"

    # --- load or generate samples ---
    if reuse_samples and samples_path.exists():
        print(f"🔁  Found samples at {samples_path}. Reusing them.")
        samples = torch.load(samples_path, map_location="cpu")
    else:
        print("🎲  Generating new samples...")
        samples = []
        data_canvas = ipps_canvas.to(DEVICE)
        with torch.no_grad():
            for _ in tqdm(range(n_samples)):
                # n = 15
                # edge_idx = torch.tensor([(i, j) for i in range(n) for j in range(n) if i != j],
                #                         dtype=torch.long).t().contiguous().to(DEVICE)
                # data = Data(x=torch.zeros(n, model.node_num_classes, device=DEVICE),
                #             edge_index=edge_idx)
                # data.batch = torch.zeros(n, dtype=torch.long, device=DEVICE)
                nodes, edges, _ = model.reverse_diffusion_single(
                    data_canvas, DEVICE, save_intermediate=False)
                samples.append((nodes.cpu(), edges.squeeze(0).cpu()))
        torch.save(samples, samples_path)
        print(f"💾  Samples saved to: {samples_path}")

    # --- metrics from (possibly reused) samples ---
    hashes = [wl_hash(n_lbls.cpu(), e_mat.cpu()) for (n_lbls, e_mat) in samples]

    data_canvas_cpu = ipps_canvas.to("cpu")  # <-- 修改
    viol = sum(not validate_constraints(e_mat, n_lbls, "cpu", data=data_canvas_cpu)
               for (n_lbls, e_mat) in samples)

    validity = 1 - viol / len(samples) if len(samples) > 0 else 0.0

    # uniqueness (WL)
    uniqueness = len(set(hashes)) / len(samples) if len(samples) > 0 else 0.0

    # novelty (WL) vs dataset
    train_hashes = {
        wl_hash(d.x.argmax(1).cpu(),
                to_dense_adj(d.edge_index, max_num_nodes=d.x.size(0))[0].cpu())
        for d in DATASET
    }
    novelty = sum(h not in train_hashes for h in hashes) / len(samples) if len(samples) > 0 else 0.0

    # FID / MMD
    batch_objs = [{"nodes": n_lbls, "edges": e_mat} for (n_lbls, e_mat) in samples]
    Z_samp = encode_graphs(batch_objs, ENC, device)
    mu_s  = Z_samp.mean(0)
    cov_s = torch.from_numpy(np.cov(Z_samp.T.numpy())) if Z_samp.shape[0] > 1 else torch.eye(Z_samp.shape[1])
    fid = frechet(MU_T, COV_T, mu_s, cov_s).item()
    mmd = mmd_rbf(_Z_TRAIN, Z_samp).item()

    return {"validity": validity, "uniqueness": uniqueness,
            "novelty": novelty, "fid": fid, "mmd": mmd}


# ---------- 4. Main Loop ----------
summary_rows = []
from Industrial_Pipeline_Functions import compute_batch_loss

def train_loop(model, loader, device,
               edge_w, node_m, edge_m,
               epochs, T, kl, c_lam):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = compute_batch_loss(model, batch, T, device,
                                      edge_w, node_m, edge_m,
                                      kl_lambda=kl,
                                      constraint_lambda=c_lam)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {ep+1}/{epochs}, Loss: {tot/len(loader):.4f}")

for cfg in ABLATIONS:
    # fixed per-variant folder (exactly the names you showed)
    model_id = model_id_from_cfg(cfg)           # "baseline", "half_hidden", ...
    run_dir  = ROOT_OUT / model_id
    run_dir.mkdir(exist_ok=True)

    weights_path = run_dir / "model.pth"

    print(f"▶️  Run: {model_id}  (dir: {run_dir})")

    # instantiate model
    hidden = cfg.get("hidden_dim", 12)
    model = LightweightIndustrialDiffusion(
        T=T_STEPS,
        hidden_dim=hidden,
        use_projector=cfg.get("use_projector", True)
    ).to(DEVICE)

    # load weights if present; otherwise train+save
    if weights_path.exists():
        print(f"🔁  Weights found: {weights_path}. Loading and skipping training.")
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)
        did_train = False
    else:
        if cfg.get("skip_training", False):
            print("⚠️  skip_training=True but no saved weights found; proceeding WITHOUT training.")
            did_train = False
        else:
            print("🧠  No weights found: running training...")
            train_loop(model, LOADER, DEVICE,
                       EDGE_W, NODE_M, EDGE_M,
                       EPOCHS, T_STEPS,
                       cfg.get("kl", 0.1),
                       cfg.get("constraint", 1.0))
            torch.save(model.state_dict(), weights_path)
            print(f"💾  Weights saved to: {weights_path}")
            did_train = True

    # metrics (reuses samples.pt if present)
    metrics = compute_metrics(model, run_dir, IPPS_CANVAS, n_samples=N_SAMP, reuse_samples=True)
    metrics.update({
        "run_dir": str(run_dir),
        "model_id": model_id,
        "weights_path": str(weights_path),
        "did_train": int(did_train),
        **cfg
    })

    # also persist metrics.json (overwrites with latest metrics)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # print compact
    pretty = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    print("→ Results:", pretty)

    summary_rows.append(metrics)

    # free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n=== Final Resume ===")
for row in summary_rows:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

# --- CSV ---
if len(summary_rows) > 0:
    preferred = ["timestamp", "model_id", "run_dir", "weights_path", "did_train",
                 "name", "hidden_dim", "kl", "constraint", "use_projector",
                 "validity", "uniqueness", "novelty", "fid", "mmd", "skip_training"]

    for r in summary_rows:
        r.setdefault("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    all_keys = set().union(*[set(r.keys()) for r in summary_rows])
    fieldnames = [k for k in preferred if k in all_keys] + [k for k in sorted(all_keys) if k not in preferred]

    csv_path = ROOT_OUT / "ablation_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in summary_rows:
            clean = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}
            w.writerow(clean)

    print(f"📄  CSV updated: {csv_path}")
else:
    print("⚠️  No summary rows: CSV not written.")
