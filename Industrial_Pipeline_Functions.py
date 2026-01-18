# Industrial Graph Pipeline

# 1 Industrial Dataset Creation
from tqdm import tqdm
import ast

import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
import json
import random
from torch.distributions import Normal
from torch_geometric.nn import TransformerConv, GlobalAttention

# 2 Industrial Diffusion Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data
import os
from torch_geometric.data import Batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPERATION, MACHINE = 0, 1


def load_ipps_problem_from_json(filepath):
    with open(filepath, 'r') as f:
        problem_def = json.load(f)

    workpieces = problem_def.get("workpieces", [])
    for wp in workpieces:
        wp["optional_machines"] = [[int(m) for m in op] for op in wp.get("optional_machines", [])]

    machines = [int(m) for m in problem_def.get("machines", [])]

    print(f"✅ Loaded task from {filepath} with {len(workpieces)} workpieces {len(machines)} machines")
    return workpieces, machines


def get_ipps_problem_data(problem_workpieces, problem_machines, device):
    op_info_list = []

    for wp_idx, wp in enumerate(problem_workpieces):
        for feat_idx in range(len(wp["optional_machines"])):
            op_info_list.append([wp_idx, feat_idx])

    num_ops = len(op_info_list)
    num_machines = len(problem_machines)
    num_nodes = num_ops + num_machines
    time_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    max_time = 1.0
    for wp in problem_workpieces:
        for time_list in wp["processing_time"]:
            if time_list:
                max_time = max(max_time, max(time_list))
    for i in range(num_ops):
        wp_idx, feat_idx = op_info_list[i]
        machines = problem_workpieces[wp_idx]["optional_machines"][feat_idx]
        times = problem_workpieces[wp_idx]["processing_time"][feat_idx]

        for m_id, t_val in zip(machines, times):
            m_idx_in_list = problem_machines.index(m_id)
            machine_graph_idx = num_ops + m_idx_in_list

            time_matrix[i, machine_graph_idx] = t_val / max_time
    op_labels = torch.full((num_ops,), 0, dtype=torch.long)
    machine_labels = torch.full((num_machines,), 1, dtype=torch.long)
    all_labels = torch.cat([op_labels, machine_labels]).to(device)
    type_onehot = F.one_hot(all_labels, num_classes=2).float()

    # Position, Workload, Connectivity
    extra_feats = torch.zeros((num_nodes, 5), device=device)

    extra_feats[num_ops:, 0] = -1.0
    extra_feats[num_ops:, 3:] = -1.0
    
    for i in range(num_ops):  # step number
        wp_idx, feat_idx = op_info_list[i]
        total_feats = len(problem_workpieces[wp_idx]["optional_machines"])
        if total_feats > 1:  # to prevent there is only one step
            norm_pos = feat_idx / (total_feats - 1)
        else:
            norm_pos = 0.0
        extra_feats[i, 0] = norm_pos

    # extract Op-Machine sub matrix
    sub_matrix = time_matrix[:num_ops, num_ops:]

    # connection for every op
    op_conn = (sub_matrix > 0).float().sum(dim=1)
    # average process time for every op
    # op_load = sub_matrix.sum(dim=1) / op_conn.clamp(min=1.0)
    
    temp_sub = sub_matrix.clone()
    temp_sub[temp_sub == 0] = float('inf')
    op_min_time, _ = temp_sub.min(dim=1) # [Num_Ops]
    # # 把之前无穷大的改回 0 (针对那些没有任何机器可选的极端情况，虽不常见)
    # op_min_time[op_min_time == float('inf')] = 0.0
    op_max_time, _ = sub_matrix.max(dim=1) # [Num_Ops]
    op_avg_time = sub_matrix.sum(dim=1) / op_conn.clamp(min=1.0)
    
    # how many operations can be processed in this machine
    m_conn = (sub_matrix > 0).float().sum(dim=0)
    # average process time for every machine
    m_load = sub_matrix.sum(dim=0) / m_conn.clamp(min=1.0)

    # Op Features 0. type 1. average processtime 2.
    extra_feats[:num_ops, 1] = op_avg_time  # Workload
    extra_feats[:num_ops, 2] = op_conn / num_machines  # Connectivity (归一化)
    extra_feats[:num_ops, 3] = op_min_time
    extra_feats[:num_ops, 4] = op_max_time
    
    extra_feats[num_ops:, 1] = m_load  # Workload
    extra_feats[num_ops:, 2] = m_conn / num_ops  # Connectivity (归一化)
   

    x = torch.cat([type_onehot, extra_feats], dim=1)

    op_info = torch.tensor(op_info_list, dtype=torch.long).to(device)
    machine_map = torch.tensor(problem_machines, dtype=torch.long).to(device)

    source_edges = []
    target_edges = []
    op_map = {tuple(info): idx for idx, info in enumerate(op_info_list)}

    for op_idx, (wp_idx, feat_idx) in enumerate(op_info_list):
        next_op_key = (wp_idx, feat_idx + 1)
        if next_op_key in op_map:
            source_edges.append(op_idx)
            target_edges.append(op_map[next_op_key])

    edge_index = torch.tensor([source_edges, target_edges], dtype=torch.long).to(device)

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    data.problem_workpieces = problem_workpieces
    data.op_info = op_info  # [N_ops, 2] (wp_idx, feat_idx)
    data.machine_map = machine_map  # [N_machines] (machine_id)
    data.time_matrix = time_matrix

    min_time_expanded = op_min_time.unsqueeze(1).expand(-1, num_machines)
    
    # 计算优势矩阵 (Op-Machine部分)
    # 注意：只针对 sub_matrix > 0 的部分计算
    advantage_sub_matrix = torch.zeros_like(sub_matrix)
    mask = sub_matrix > 0
    # 相对偏差： (Time - Min) / Min
    advantage_sub_matrix[mask] = (sub_matrix[mask] - min_time_expanded[mask]) / (min_time_expanded[mask] + 1e-6)
    
    # 放入完整的 advantage_matrix
    advantage_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    advantage_matrix[:num_ops, num_ops:] = advantage_sub_matrix
    
    data.advantage_matrix = advantage_matrix
    
    return data


def get_ipps_allowed_mask(node_labels, data, device):
    n_nodes = node_labels.size(0)
    op_info = data.op_info
    machine_map = data.machine_map
    problem_workpieces = data.problem_workpieces

    n_ops = op_info.size(0)
    n_machines = machine_map.size(0)

    op_indices = (node_labels == 0).nonzero(as_tuple=True)[0]
    machine_indices = (node_labels == 1).nonzero(as_tuple=True)[0]

    allowed_mask = torch.zeros((n_nodes, n_nodes), dtype=torch.bool, device=device)

    op_map = {tuple(info.tolist()): idx.item() for idx, info in zip(op_indices, op_info)}
    for i in range(n_ops):
        op_graph_idx = op_indices[i].item()
        wp_idx, feat_idx = op_info[i].tolist()

        next_op_key = (wp_idx, feat_idx + 1)
        if next_op_key in op_map:
            next_op_graph_idx = op_map[next_op_key]
            allowed_mask[op_graph_idx, next_op_graph_idx] = True

    for i in range(n_ops):
        op_graph_idx = op_indices[i].item()
        wp_idx, feat_idx = op_info[i].tolist()

        allowed_machine_ids = problem_workpieces[wp_idx]["optional_machines"][feat_idx]

        for j in range(n_machines):
            machine_graph_idx = machine_indices[j].item()
            original_machine_id = machine_map[j].item()

            if original_machine_id in allowed_machine_ids:
                allowed_mask[op_graph_idx, machine_graph_idx] = True

    return allowed_mask


def ipps_projector(node_labels, candidate_matrix, data, device):
    n_nodes = node_labels.size(0)
    projected_edges = torch.zeros((n_nodes, n_nodes), dtype=torch.long, device=device)

    op_info = data.op_info
    machine_map = data.machine_map
    problem_workpieces = data.problem_workpieces

    op_indices = (node_labels == 0).nonzero(as_tuple=True)[0]
    machine_indices = (node_labels == 1).nonzero(as_tuple=True)[0]
    n_ops = op_indices.size(0)
    n_machines = machine_indices.size(0)

    for i in range(n_ops):
        op_graph_idx = op_indices[i].item()

        wp_idx, feat_idx = op_info[i].tolist()
        allowed_machine_ids = problem_workpieces[wp_idx]["optional_machines"][feat_idx]

        allowed_graph_indices = []
        for j in range(n_machines):
            if machine_map[j].item() in allowed_machine_ids:
                allowed_graph_indices.append(machine_indices[j].item())

        proposed_machines = []
        for machine_idx in allowed_graph_indices:
            if candidate_matrix[op_graph_idx, machine_idx] == 1:
                proposed_machines.append(machine_idx)

        if proposed_machines:
            chosen_machine = random.choice(proposed_machines)

        elif allowed_graph_indices:

            chosen_machine = random.choice(allowed_graph_indices)

        projected_edges[op_graph_idx, chosen_machine] = 1

    return projected_edges


def validate_constraints(edge_matrix, node_labels, device, exact=True, data=None):
    E = torch.as_tensor(edge_matrix, dtype=torch.long, device=device)

    op_info = data.op_info
    machine_map = data.machine_map

    op_indices = (node_labels == 0).nonzero(as_tuple=True)[0]
    machine_indices = (node_labels == 1).nonzero(as_tuple=True)[0]
    n_ops = op_indices.size(0)

    if torch.any(torch.diag(E) != 0):
        return False

    if E[machine_indices][:, machine_indices].sum() > 0:
        return False

    allowed_mask = get_ipps_allowed_mask(node_labels, data, device)
    for i in range(n_ops):
        op_graph_idx = op_indices[i].item()
        op_to_machine_edges = E[op_graph_idx, machine_indices]
        if op_to_machine_edges.sum() != 1:
            return False

        chosen_machine_graph_idx = machine_indices[op_to_machine_edges.argmax()]
        if not allowed_mask[op_graph_idx, chosen_machine_graph_idx]:
            return False

    op_map = {tuple(info.tolist()): idx.item() for idx, info in zip(op_indices, op_info)}
    for i in range(n_ops):
        op_graph_idx = op_indices[i].item()
        wp_idx, feat_idx = op_info[i].tolist()

        next_op_key = (wp_idx, feat_idx + 1)
        if next_op_key in op_map:
            next_op_graph_idx = op_map[next_op_key]
            if E[op_graph_idx, next_op_graph_idx] != 1:
                return False

        op_to_op_edges = E[op_graph_idx, op_indices]
        if op_to_op_edges.sum() > 1:
            return False

    return True


def get_sinusoidal_embedding(t, embedding_dim):
    if t.dim() == 1:
        t = t.unsqueeze(1)
    device = t.device
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    dims = torch.arange(half_dim, device=device).float()
    dims = torch.exp(-dims * emb_scale)
    emb = t * dims.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=device)], dim=1)
    return emb


def kl_divergence(pred_probs, marginal_probs):
    marginal_probs = marginal_probs.unsqueeze(0)
    kl = torch.sum(pred_probs * (torch.log(pred_probs + 1e-8) - torch.log(marginal_probs + 1e-8)), dim=1)
    return kl.mean()



class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :return: an (N, dim) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResGNNBlock(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.1, edge_dim=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)  # Affine由AdaLN处理
        self.attn = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads,
                                    concat=True, dropout=dropout, edge_dim=edge_dim)

        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        # AdaLN Modulation: 预测 (scale, shift) x 2 (for norm1 and norm2)
        # 输入是 time_emb，输出是 4 * hidden_dim 参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 4 * hidden_dim)
        )

    def forward(self, x, t_emb, edge_index, edge_attr):
        # 1. 计算 AdaLN 参数
        scale_shift = self.adaLN_modulation(t_emb)  # [Batch, 4*Dim] -> 需要扩展到 [Num_Nodes, 4*Dim]

        # 处理 Batch 维度对齐问题 (Graph Batching 中 x 是堆叠的)
        # 假设 t_emb 已经根据 batch 扩展好了 或者在这里进行 gather
        # 简单起见，假设传入 forward 的 t_emb 已经是 [Num_Nodes, Dim]

        shift_msa, scale_msa, shift_mlp, scale_mlp = scale_shift.chunk(4, dim=1)

        # 2. Attention Block (Pre-Norm + Residual)
        x_norm = self.norm1(x) * (1 + scale_msa) + shift_msa
        x_attn = self.attn(x_norm, edge_index, edge_attr)
        x = x + x_attn  # Residual

        # 3. FFN Block (Pre-Norm + Residual)
        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x_ffn = self.ffn(x_norm)
        x = x + x_ffn  # Residual

        return x


# ----------------------------------------------------------------
# 3. 主模型：ComplexIndustrialDiffusion
# ----------------------------------------------------------------
class LightweightIndustrialDiffusion(nn.Module):
    def __init__(self, T=100, input_dim=7, hidden_dim=128, num_layers=6,
                 beta_start=0.0001, beta_end=0.02, nhead=4, dropout=0.1,
                 device='cuda', edge_dim=16):
        super().__init__()
        self.device = torch.device(device)
        self.T = T
        self.hidden_dim = hidden_dim

        self.beta_schedule = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta_schedule
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

        # Input Projection
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(3, edge_dim)  # edge_attr 是时间和priority

        # Time Embedding
        self.time_embedder = TimestepEmbedder(hidden_dim)

        # Backbone: Stack of ResGNNBlocks
        self.layers = nn.ModuleList([
            ResGNNBlock(hidden_dim, heads=nhead, dropout=dropout, edge_dim=edge_dim)
            for _ in range(num_layers)  # 加深到 num_layers 层
        ])

        # Global Pooling (用于提取图级别的上下文信息)
        self.global_pool = GlobalAttention(nn.Sequential(
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        ))

        # Output Heads
        # self.node_out = nn.Linear(hidden_dim, 2)  # Node Class

        # 增强的 Edge Decoder (Bilinear or Deeper MLP)
        self.edge_num_classes = 2
        self.edge_out_dim = self.edge_num_classes + 2

        # 这里使用一个 Bilinear 层来增强节点对之间的交互建模
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.edge_out_dim)
        )

    def forward(self, x, edge_index, batch, t, time_matrix=None, priorities=None, advantage_matrix=None):

        h = self.node_encoder(x.float())

        # Time Embedding
        t_tensor = torch.tensor([t], dtype=torch.float, device=x.device)
        t_emb = self.time_embedder(t_tensor)  # [1, Hidden]
        # 将 t_emb 扩展到每个节点: [Num_Nodes, Hidden]
        t_emb_node = t_emb.repeat(h.size(0), 1)
        src, dst = edge_index
        
        # Edge Attributes 处理
        if time_matrix is not None:
            if time_matrix.size(0) < x.size(0):
                num_nodes_per_graph = time_matrix.size(0)
                edge_times = time_matrix[src % num_nodes_per_graph, dst % num_nodes_per_graph].unsqueeze(-1)
            else:
                edge_times = time_matrix[src, dst].unsqueeze(-1)
                # === [Batch Patch End] ===
        else:
            print('no time metrix!!!')
            edge_times = torch.zeros((edge_index.size(1), 1), device=x.device)

        if advantage_matrix is not None:
            edge_advs = advantage_matrix[src, dst].unsqueeze(-1)
        else:
            edge_advs = torch.zeros_like(edge_times)
            
        if priorities is not None:
            # 假设 priorities 是 [Batch, Num_Nodes] 或者 [Num_Nodes]
            # 我们需要获取每条边源节点(src)对应的优先级
            # 如果 priorities 是 [Num_Nodes]，直接索引
            if priorities.dim() == 1:
                edge_prios = priorities[src].unsqueeze(-1) 
            elif priorities.dim() == 2 and priorities.size(0) == x.size(0):
                 # 如果 priorities 是 [N, 1]
                 edge_prios = priorities[src]
            else:
                 print('error code3745')
                 edge_prios = torch.zeros_like(edge_times)
        else:
            # 默认优先级为 0.5 (中性) 或 0
            edge_prios = torch.zeros_like(edge_times)
            
        raw_edge_attr = torch.cat([edge_times, edge_advs, edge_prios], dim=1)
        edge_attr = self.edge_encoder(raw_edge_attr)
        # 2. Backbone Processing
        for layer in self.layers:
            # 传入 t_emb_node 用于 AdaLN
            h = layer(h, t_emb_node, edge_index, edge_attr)

        # 3. Output Heads
        # node_logits = self.node_out(h)

        # 4. Dense Edge Prediction with Global Context
        h_dense, mask = to_dense_batch(h, batch)  # [Batch, Max_Nodes, Hidden]
        batch_size, max_nodes, _ = h_dense.shape

        edge_logits_list = []

        for i in range(batch_size):
            num_nodes = int(mask[i].sum().item())
            h_i = h_dense[i, :num_nodes, :]  # [N, Hidden]

            # 显式构建节点对:
            # src: [N, N, Hidden], dst: [N, N, Hidden]
            h_src = h_i.unsqueeze(1).expand(-1, num_nodes, -1)
            h_dst = h_i.unsqueeze(0).expand(num_nodes, -1, -1)

            # 使用 Bilinear Layer 捕捉更强的交互: src^T * W * dst + b
            # [N, N, Hidden]
            pair_embed = self.bilinear(h_src, h_dst)

            # Final projection
            edge_logits = self.edge_mlp(pair_embed)
            edge_logits_list.append(edge_logits)

        return edge_logits_list

    @staticmethod
    def get_temperature(t, T_total, start_temp, end_temp, method='linear'):

        progress = t / T_total

        if method == 'linear':
            return end_temp + (start_temp - end_temp) * progress

        elif method == 'cosine':
            import math
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            return end_temp + (start_temp - end_temp) * cosine_decay

        elif method == 'exp':
            return end_temp * (start_temp / end_temp) ** progress

        else:
            return end_temp + (start_temp - end_temp) * progress

    def _precompute_log_matrices(self):

        K = self.node_num_classes
        T = self.T
        device = self.alpha_bar.device

        log_Q_matrices = torch.zeros((T, K, K), device=device)
        log_Q_bar_matrices = torch.zeros((T, K, K), device=device)

        for t in range(T):
            p_keep_t = self.alpha_bar[t]
            beta_bar_t = 1.0 - p_keep_t
            off_diag_val = beta_bar_t / K
            diag_val = p_keep_t + off_diag_val

            Q_bar_t = torch.full((K, K), fill_value=off_diag_val, device=device)
            Q_bar_t.fill_diagonal_(diag_val)

            log_Q_bar_matrices[t] = torch.log(Q_bar_t + 1e-12)

        for t in range(1, T):
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_t_minus_1 = self.alpha_bar[t - 1]

            if alpha_bar_t_minus_1 == 0:

                beta_t = 1.0
            else:
                beta_t = (alpha_bar_t_minus_1 - alpha_bar_t) / alpha_bar_t_minus_1

            if beta_t < 0: beta_t = 0.0

            off_diag_val_q = beta_t / K
            diag_val_q = (1.0 - beta_t) + off_diag_val_q

            Q_t = torch.full((K, K), fill_value=off_diag_val_q, device=device)
            Q_t.fill_diagonal_(diag_val_q)

            log_Q_matrices[t] = torch.log(Q_t + 1e-12)
            return log_Q_matrices, log_Q_bar_matrices

    def _get_posterior_logits(self, x_t_onehot, x_0_pred_labels, t):

        if t == 0:
            raise ValueError("Error, t equals to zero, and the posterior is not defined")

        log_q_xt_given_xt_minus_1 = self.log_Q_matrices[t]  # [K, K]
        log_q_xt_minus_1_given_x0 = self.log_Q_bar_matrices[t - 1]  # [K, K]

        x_t_int = x_t_onehot.argmax(dim=1)  # [N]
        log_term_1 = log_q_xt_given_xt_minus_1.T  # [K, K]
        log_term_1_gathered = log_term_1[:, x_t_int].T  # [N, K]
        log_term_2_gathered = log_q_xt_minus_1_given_x0[x_0_pred_labels, :]  # [N, K]
        log_logits = log_term_1_gathered + log_term_2_gathered

        return log_logits

    def reverse_diffusion_with_logprob(self, data, device, num_samples=1,
                                       time_guidance_scale=0.1, position_guidance_scale=0.0,
                                       return_trajectory=False, temperature_method='cosine',
                                       start_temp=2.0, end_temp=0.1, greedy=False):
        """
        For RL sampling
        """
        data_list = [data.clone() for _ in range(num_samples)]
        batch_data = Batch.from_data_list(data_list).to(device)
        num_nodes_per_graph = data.x.size(0)
        total_nodes = batch_data.x.size(0)
        B = num_samples
        x_single = data.x
        node_types = x_single.argmax(dim=1)
        allowed_mask_single = get_ipps_allowed_mask(node_types, data, device)
        allowed_mask_batch = allowed_mask_single.unsqueeze(0).expand(B, -1, -1)

        seq_edges_src = data.edge_index[0]
        seq_edges_tgt = data.edge_index[1]
        pinned_mask_single = torch.zeros((num_nodes_per_graph, num_nodes_per_graph), dtype=torch.bool, device=device)
        pinned_mask_single[seq_edges_src, seq_edges_tgt] = True
        pinned_mask_batch = pinned_mask_single.unsqueeze(0).expand(B, -1, -1)

        op_indices = (node_types == 0).nonzero(as_tuple=True)[0]
        machine_indices = (node_types == 1).nonzero(as_tuple=True)[0]
        e = torch.zeros((B, num_nodes_per_graph, num_nodes_per_graph, self.edge_num_classes), device=device)
        e[:, :, :, 0] = 1  # Default NoEdge
        e[pinned_mask_batch] = torch.tensor([0.0, 1.0], device=device)

        total_log_prob = torch.zeros(B, device=device)
        total_entropy = torch.zeros(B, device=device)
        trajectory = []


        x_dense, _ = to_dense_batch(batch_data.x, batch_data.batch) 
    
        # x_dense[:, :, 2] 是 norm_pos (0.0=首工序, 1.0=尾工序, -1.0=机器)
        # 我们只关心工序节点 (type==0, 即 x_dense[:,:,0]==1)
        is_op_node = (x_dense[:, :, 0] == 1).float() # operation node
        norm_pos = x_dense[:, :, 2]  # operation position
        
        pos_bias_raw = (1.0 - norm_pos) * is_op_node
        pos_bias_tensor = pos_bias_raw.unsqueeze(-1) * position_guidance_scale
        current_priorities = torch.randn(total_nodes, device=device) * 0.5 + 0.5
        current_priorities = torch.clamp(current_priorities, 0.0, 1.0)
                                           
        for t in range(self.T - 1, -1, -1):

            current_temp = self.get_temperature(t, self.T, start_temp, end_temp, method=temperature_method)

            current_edge_labels = e.argmax(dim=-1)
            b_idx, u, v = (current_edge_labels > 0).nonzero(as_tuple=True)
            global_u = u + b_idx * num_nodes_per_graph
            global_v = v + b_idx * num_nodes_per_graph
            edge_index_t = torch.stack([global_u, global_v], dim=0)
            edge_outputs_list = self.forward(batch_data.x, edge_index_t, batch_data.batch, t, data.time_matrix, priorities=current_priorities)
##########################################
            if edge_outputs_list:
                # edge_output = edge_outputs_list[0]  # [B, N, N, 4]
                edge_output = torch.stack(edge_outputs_list, dim=0)
                # --- Priorities ---
                prio_mean = edge_output[:, :, :, 2]
                if position_guidance_scale > 0:
                    prio_mean = prio_mean + pos_bias_tensor
                
                prio_log_std = edge_output[:, :, :, 3]
                prio_std = torch.exp(torch.clamp(prio_log_std, min=-20, max=2))
                scaled_prio_std = prio_std * current_temp + 1e-6

                prio_dist = torch.distributions.Normal(prio_mean, scaled_prio_std)
                if greedy:
                    # Greedy模式：直接使用均值 (正态分布的均值即众数/概率最大处)
                    raw_priority_sample = prio_mean
                else:
                    # Stochastic模式：从分布采样
                    raw_priority_sample = prio_dist.sample()

                priority_scores = torch.sigmoid(raw_priority_sample)

                # --- Routing ---
                edge_logits = edge_output[:, :, :, :2]
                score_matrix = edge_logits[:, :, :, 1]  # [B, N, N]

                # Time Guidance (广播: [B, N, N] - [1, N, N])
                score_matrix = score_matrix - (data.time_matrix.unsqueeze(0) * time_guidance_scale)

                new_e_indices = torch.zeros((B, num_nodes_per_graph, num_nodes_per_graph), dtype=torch.long,
                                            device=device)
                new_e_indices[pinned_mask_batch] = 1

                op_machine_scores = score_matrix.clone() / current_temp

                # Masking (Batch)
                op_machine_scores[~allowed_mask_batch] = -1e9
                valid_col_mask = torch.zeros_like(op_machine_scores, dtype=torch.bool)
                valid_col_mask[:, :, machine_indices] = True
                op_machine_scores[~valid_col_mask] = -1e9

                # Target Scores [B, Num_Ops, Num_Nodes]
                target_scores = op_machine_scores[:, op_indices, :]

                # Sample
                dist = torch.distributions.Categorical(logits=target_scores)
                if greedy:
                    # Greedy模式：选择概率最大的动作 (Argmax)
                    actions = torch.argmax(target_scores, dim=-1)
                else:
                    # Stochastic模式：按概率采样
                    actions = dist.sample()
                # actions = dist.sample()  # [B, Num_Ops]

                # --- Metrics ---
                # Log Prob
                selected_prio_log_prob = prio_dist.log_prob(raw_priority_sample)
                relevant_prio_log_prob = selected_prio_log_prob[:, op_indices, :]  # [B, Ops, N]
                chosen_prio_log_prob = relevant_prio_log_prob.gather(2, actions.unsqueeze(-1)).squeeze(-1)

                step_log_prob = dist.log_prob(actions).sum(dim=1) + chosen_prio_log_prob.sum(dim=1)

                # Entropy
                entropy_routing = dist.entropy().mean(dim=1)
                entropy_prio = prio_dist.entropy()[:, op_indices, :].mean(dim=(1, 2))

                total_log_prob += step_log_prob
                total_entropy += (entropy_routing + entropy_prio)


                batch_idx_expanded = torch.arange(B, device=device).unsqueeze(1).expand(-1, len(op_indices))
                op_indices_expanded = op_indices.unsqueeze(0).expand(B, -1)

                new_e_indices[batch_idx_expanded, op_indices_expanded, actions] = 1
                new_e_indices[pinned_mask_batch] = 1

                if return_trajectory:
                    trajectory.append(new_e_indices.detach().cpu().clone())

                e = torch.nn.functional.one_hot(new_e_indices, num_classes=self.edge_num_classes).float()

            relevant_priorities = priority_scores[:, op_indices, :]
            final_priorities = relevant_priorities.gather(2, actions.unsqueeze(-1)).squeeze(-1)
            temp_prio_matrix = torch.zeros((B, num_nodes_per_graph), device=device)
            temp_prio_matrix[:, op_indices] = final_priorities
            current_priorities = temp_prio_matrix.view(-1)
            
        if num_samples == 1:
            e = e.squeeze(0)  # [N, N, C]
            total_log_prob = total_log_prob.squeeze(0)
            total_entropy = total_entropy.squeeze(0)
            final_priorities = final_priorities.squeeze(0)
            if return_trajectory:
                trajectory = [t.squeeze(0) for t in trajectory]

        if return_trajectory:
            return e, total_log_prob, total_entropy, final_priorities, trajectory
        else:
            return e, total_log_prob, total_entropy, final_priorities

    def refine_from_intermediate(self, noisy_e, data, device, start_t,
                                 hint_priorities=None,
                                 time_guidance_scale=0.1, position_guidance_scale=0.0,
                                 temperature_method='cosine', start_temp=2.0, end_temp=0.1):
        """
        [Refinement Step]
        接力 mutation：从中间时刻 start_t 开始，把变异后的粗糙图修补成完整解。
        """
        # 1. 初始化 Batch 信息
        B = noisy_e.shape[0]
        data_list = [data.clone() for _ in range(B)]
        batch_data = Batch.from_data_list(data_list).to(device)
        num_nodes_per_graph = data.x.size(0)
        total_nodes = batch_data.x.size(0)
        # 2. 继承变异后的边状态 (而不是像原函数那样初始化为全0)
        e = noisy_e.clone()

        # 3. 准备各种 Mask (直接复制原函数的逻辑)
        x_single = data.x
        node_types = x_single.argmax(dim=1)
        allowed_mask_single = get_ipps_allowed_mask(node_types, data, device)
        allowed_mask_batch = allowed_mask_single.unsqueeze(0).expand(B, -1, -1)

        seq_edges_src = data.edge_index[0]
        seq_edges_tgt = data.edge_index[1]
        pinned_mask_single = torch.zeros((num_nodes_per_graph, num_nodes_per_graph), dtype=torch.bool, device=device)
        pinned_mask_single[seq_edges_src, seq_edges_tgt] = True
        pinned_mask_batch = pinned_mask_single.unsqueeze(0).expand(B, -1, -1)

        op_indices = (node_types == 0).nonzero(as_tuple=True)[0]
        machine_indices = (node_types == 1).nonzero(as_tuple=True)[0]

        # 准备 Position Bias
        x_dense, _ = to_dense_batch(batch_data.x, batch_data.batch)
        is_op_node = (x_dense[:, :, 0] == 1).float()
        norm_pos = x_dense[:, :, 2]
        pos_bias_raw = (1.0 - norm_pos) * is_op_node
        pos_bias_tensor = pos_bias_raw.unsqueeze(-1) * position_guidance_scale

        if hint_priorities is not None:
            # hint_priorities 是 [B, Num_Ops]
            # 我们需要把它扩展到 [B, N] 并展平
            temp_prio = torch.zeros((B, num_nodes_per_graph), device=device)
            temp_prio[:, op_indices] = hint_priorities
            current_priorities = temp_prio.view(-1) # [B*N]
        else:
            # 如果没有 Hint，就随机初始化
            current_priorities = torch.randn(total_nodes, device=device) * 0.5 + 0.5
        
        current_priorities = torch.clamp(current_priorities, 0.0, 1.0)
                                     
        final_priorities = None

        # 4. 循环：从 start_t 开始倒数，而不是从 self.T 开始
        #    这是唯一的区别！
        for t in range(start_t, -1, -1):

            # --- 以下逻辑与 reverse_diffusion_with_logprob 完全一致 ---
            current_temp = self.get_temperature(t, self.T, start_temp, end_temp, method=temperature_method)

            current_edge_labels = e.argmax(dim=-1)
            b_idx, u, v = (current_edge_labels > 0).nonzero(as_tuple=True)
            global_u = u + b_idx * num_nodes_per_graph
            global_v = v + b_idx * num_nodes_per_graph
            edge_index_t = torch.stack([global_u, global_v], dim=0)
            # edge_outputs_list = self.forward(batch_data.x, edge_index_t, batch_data.batch, t, data.time_matrix)
            edge_outputs_list = self.forward(batch_data.x, edge_index_t, batch_data.batch, 
                                             t, data.time_matrix, priorities=current_priorities)
            if edge_outputs_list:
                edge_output = torch.stack(edge_outputs_list, dim=0)

                # Priority Logic
                prio_mean = edge_output[:, :, :, 2]
                if position_guidance_scale > 0:
                    prio_mean = prio_mean + pos_bias_tensor
                prio_log_std = edge_output[:, :, :, 3]
                prio_std = torch.exp(torch.clamp(prio_log_std, min=-20, max=2))
                scaled_prio_std = prio_std * current_temp + 1e-6
                prio_dist = torch.distributions.Normal(prio_mean, scaled_prio_std)
                raw_priority_sample = prio_dist.sample()
                priority_scores = torch.sigmoid(raw_priority_sample)

                # Routing Logic
                edge_logits = edge_output[:, :, :, :2]
                score_matrix = edge_logits[:, :, :, 1]
                score_matrix = score_matrix - (data.time_matrix.unsqueeze(0) * time_guidance_scale)

                new_e_indices = torch.zeros((B, num_nodes_per_graph, num_nodes_per_graph), dtype=torch.long,
                                            device=device)
                new_e_indices[pinned_mask_batch] = 1

                op_machine_scores = score_matrix.clone() / current_temp
                op_machine_scores[~allowed_mask_batch] = -1e9
                valid_col_mask = torch.zeros_like(op_machine_scores, dtype=torch.bool)
                valid_col_mask[:, :, machine_indices] = True
                op_machine_scores[~valid_col_mask] = -1e9

                target_scores = op_machine_scores[:, op_indices, :]
                dist = torch.distributions.Categorical(logits=target_scores)
                actions = dist.sample()

                # 更新边状态 e
                batch_idx_expanded = torch.arange(B, device=device).unsqueeze(1).expand(-1, len(op_indices))
                op_indices_expanded = op_indices.unsqueeze(0).expand(B, -1)
                new_e_indices[batch_idx_expanded, op_indices_expanded, actions] = 1
                new_e_indices[pinned_mask_batch] = 1
                e = torch.nn.functional.one_hot(new_e_indices, num_classes=self.edge_num_classes).float()

                relevant_priorities = priority_scores[:, op_indices, :]
                final_priorities = relevant_priorities.gather(2, actions.unsqueeze(-1)).squeeze(-1)
                
                temp_prio_matrix = torch.zeros((B, num_nodes_per_graph), device=device)
                temp_prio_matrix[:, op_indices] = final_priorities
                current_priorities = temp_prio_matrix.view(-1)

        return e, final_priorities

    def rl_structural_mutation(self, elite_edges_onehot, elite_priorities, t,
                               base_mutation_scale=1.0,
                               priority_noise_scale=0.3,  # 新增：控制优先级的变异幅度
                               start_temp=2.0, end_temp=0.1,
                               temperature_method='cosine'):
        """
        [Theory-Aligned Dual Mutation]
        同时对 图结构(离散) 和 优先级(连续) 进行符合热力学的变异。
        """

        # 1. 计算归一化温度 (Entropy Level)
        current_temp = self.get_temperature(
            t, self.T, start_temp, end_temp, method=temperature_method
        )
        tau_min = end_temp
        tau_max = start_temp
        normalized_temp = (current_temp - tau_min) / (tau_max - tau_min + 1e-8)

        # 限制范围 [0, 1]
        mutation_intensity = torch.clamp(torch.tensor(normalized_temp), 0.0, 1.0).item()

        # ==========================================
        # Part A: 边的离散变异 (Discrete Mutation)
        # ==========================================
        edge_mutation_rate = mutation_intensity * base_mutation_scale
        edge_mutation_rate = min(edge_mutation_rate, 1.0)  # 钳位

        current_decisions = elite_edges_onehot.argmax(dim=-1)
        rand_mask = torch.rand_like(current_decisions.float()) < edge_mutation_rate
        random_decisions = torch.randint(0, self.edge_num_classes, current_decisions.shape, device=self.device)

        mutated_indices = torch.where(rand_mask, random_decisions, current_decisions)
        mutated_edges_onehot = F.one_hot(mutated_indices.long(), num_classes=self.edge_num_classes).float()

        # ==========================================
        # Part B: 优先级的连续变异 (Continuous Mutation)
        # ==========================================
        # 假设 elite_priorities 是 [B, Num_Ops] 且范围在 [0, 1] 之间
        # 我们添加均值为 0，标准差与 Temperature 成正比的高斯噪声

        # noise_std 随着温度升高而变大
        current_noise_std = priority_noise_scale * mutation_intensity

        # 生成高斯噪声
        gaussian_noise = torch.randn_like(elite_priorities) * current_noise_std

        # 叠加噪声
        mutated_priorities = elite_priorities + gaussian_noise

        # 重新钳位回有效范围 (例如 sigmoid 后的 0~1)
        mutated_priorities = torch.clamp(mutated_priorities, 0.0, 1.0)

        return mutated_edges_onehot, mutated_priorities, mutation_intensity

    def forward_diffusion(self, x0, e0, t, device):
        x_t_onehot = F.one_hot(x0, num_classes=self.node_num_classes).float()

        p_keep = self.alpha_bar[t].item()
        rand_vals = torch.rand(x0.shape, device=device)
        # random_node = torch.randint(0, self.node_num_classes, x0.shape, device=device)
        # x_t = torch.where(rand_vals < p_keep, x0, random_node)
        # x_t_onehot = F.one_hot(x_t, num_classes=self.node_num_classes).float()

        rand_vals_e = torch.rand(e0.shape, device=device)
        random_edge = torch.randint(0, self.edge_num_classes, e0.shape, device=device)
        e_t_raw = torch.where(rand_vals_e < p_keep, e0, random_edge)

        # if self.use_projector:
        #     projected_edges = ipps_projector(x_t, e_t_raw, device)
        # else:
        #     projected_edges = e_t_raw # No se aplica projector

        e_t_onehot = F.one_hot(e_t_raw.long(), num_classes=self.edge_num_classes).float()

        return x_t_onehot, e_t_onehot

    def reverse_diffusion_single(self, data, device, save_intermediate=True, time_guidance_scale=0.1):
        num_nodes = data.x.size(0)
        x = data.x.clone()
        seq_edges_src = data.edge_index[0]
        seq_edges_tgt = data.edge_index[1]
        pinned_edge_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        pinned_edge_mask[seq_edges_src, seq_edges_tgt] = True

        e = torch.zeros((num_nodes, num_nodes, self.edge_num_classes), device=device)
        e[:, :, 0] = 1
        e[pinned_edge_mask] = torch.tensor([0.0, 1.0], device=device)

        intermediate_graphs = []
        # max_attempts = 20

        for t in range(self.T - 1, -1, -1):

            current_edge_labels = e.argmax(dim=-1)  # [N, N]
            edge_index_t = (current_edge_labels > 0).nonzero(as_tuple=False).t().contiguous()

            node_logits, edge_logits_list = self.forward(x, edge_index_t, data.batch, t)
            # if t > 0: tmp disable posterior for now, since we do not sample node anymore tbdTBD
            #     node_probs = F.softmax(node_logits, dim=1)  # p_theta(x0 | x_t) [cite: 127]
            #     x_0_pred_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)  # [N]
            #     posterior_logits = self._get_posterior_logits(x, x_0_pred_labels, t)  # [cite: 128, 82]
            #     posterior_probs = F.softmax(posterior_logits, dim=1)
            #     x_labels = torch.multinomial(posterior_probs, num_samples=1).squeeze(1)
            #     x = F.one_hot(x_labels, num_classes=self.node_num_classes).float()
            # else:
            #     node_probs = F.softmax(node_logits, dim=1)
            #     x_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)
            #     x = F.one_hot(x_labels, num_classes=self.node_num_classes).float()

            if edge_logits_list and edge_logits_list[0].numel() > 0:
                edge_logits = edge_logits_list[0]
                if hasattr(data, 'time_matrix'):
                    time_penalty = data.time_matrix * time_guidance_scale
                    edge_logits[:, :, 1] -= time_penalty

                current_node_labels = x.argmax(dim=1)

                allowed_mask = get_ipps_allowed_mask(current_node_labels, data, device)
                forbidden_mask = ~allowed_mask
                edge_logits[:, :, 1][forbidden_mask] = -torch.inf
                large_val = 1e10
                edge_logits[:, :, 0][pinned_edge_mask] = -torch.inf
                edge_logits[:, :, 1][pinned_edge_mask] = large_val

                edge_probs = F.softmax(edge_logits, dim=-1)
                flat_probs = edge_probs.view(-1, self.edge_num_classes)
                sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
                candidate_edge_matrix = sampled_flat.view(num_nodes, num_nodes)

                projected_op_machine = ipps_projector(current_node_labels, candidate_edge_matrix, data, device)
                projected = projected_op_machine
                projected[pinned_edge_mask] = 1

                e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()

            if save_intermediate:
                intermediate_graphs.append(Data(
                    x=x.clone(),
                    edge_index=(e.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
                ))

        final_node_labels = x.argmax(dim=1)
        final_edge_labels = e.argmax(dim=-1)

        return final_node_labels, final_edge_labels.unsqueeze(0), intermediate_graphs

    def generate_global_graph(self, n_nodes):
        edge_list = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
        x = torch.zeros(n_nodes, self.node_num_classes, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.reverse_diffusion_single(data, self.device, False)
        node_types = final_nodes
        return node_types, final_edges
