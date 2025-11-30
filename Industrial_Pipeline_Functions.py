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

class IndustrialGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Dataset.xlsx']

    @property
    def processed_file_names(self):
        return ['processed_graphs.pt']

    def download(self):
        pass

    def process(self):

        df = pd.read_excel(self.raw_paths[0])

        machine_cols = sorted(
            [col for col in df.columns if col.startswith('Workpiece') and col.endswith('machines')],
            key=lambda x: int(x.split('_')[0].replace('Workpiece', ''))
        )

        node_type_to_idx = {'OPERATION': 0, 'MACHINE': 1}
        num_node_classes = 2

        data_list = []

        for _, row in df.iterrows():

            op_node_index_counter = 0
            all_ops_info = []  # (op_node_idx, machine_id_from_excel)
            all_machine_ids_in_graph = set()

            source_edges = []
            target_edges = []

            for col_name in machine_cols:
                if pd.isna(row[col_name]):
                    continue

                machine_list_str = str(row[col_name])

                try:
                    # 使用 ast.literal_eval 安全解析 "[6, 2, 1]"
                    machine_indices = ast.literal_eval(machine_list_str)
                except Exception as e:
                    print(f"error: {_.name}, column {col_name} Error: {e} skipped")
                    continue

                ops_for_this_workpiece = []

                for machine_idx in machine_indices:
                    op_idx = op_node_index_counter

                    ops_for_this_workpiece.append(op_idx)
                    all_ops_info.append((op_idx, machine_idx))
                    all_machine_ids_in_graph.add(machine_idx)

                    op_node_index_counter += 1

                # add edges among operations inside the same workpiece
                for i in range(len(ops_for_this_workpiece) - 1):
                    u = ops_for_this_workpiece[i]
                    v = ops_for_this_workpiece[i + 1]
                    source_edges.append(u)
                    target_edges.append(v)

            num_operation_nodes = op_node_index_counter

            unique_machine_ids = sorted(list(all_machine_ids_in_graph))
            num_machine_nodes = len(unique_machine_ids)

            machine_id_to_graph_idx_map = {
                machine_id: i + num_operation_nodes
                for i, machine_id in enumerate(unique_machine_ids)
            }

            op_labels = torch.full((num_operation_nodes,),
                                   fill_value=node_type_to_idx['OPERATION'],
                                   dtype=torch.long)

            machine_labels = torch.full((num_machine_nodes,),
                                        fill_value=node_type_to_idx['MACHINE'],
                                        dtype=torch.long)

            all_labels = torch.cat([op_labels, machine_labels])
            x = F.one_hot(all_labels, num_classes=num_node_classes).float()

            for op_idx, machine_id in all_ops_info:
                u = op_idx
                if machine_id not in machine_id_to_graph_idx_map:
                    continue
                v = machine_id_to_graph_idx_map[machine_id]
                source_edges.append(u)
                target_edges.append(v)

            edge_index = torch.tensor([source_edges, target_edges], dtype=torch.long)

            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

            y = None
            if 'makespan' in df.columns:
                y = torch.tensor([row['makespan']], dtype=torch.float)

            data = Data(x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        n_nodes=x.size(0))

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def weighted_repeat_inplace(self, repeat_list):
        assert len(repeat_list) == len(self), "length of repeat_list must equal to the length of the dataset"

        data_list = [self.get(i) for i in range(len(self))]
        new_data_list = []
        for data, repeat_times in zip(data_list, repeat_list):
            for _ in range(repeat_times):
                new_data_list.append(data.clone())

        self.data, self.slices = self.collate(new_data_list)
        print(f"Dataset length changed from {len(data_list)} to {len(new_data_list)}")


# 2 Industrial Diffusion Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Definir constantes para claridad
# MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY = 0, 1, 2, 3

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


    # op_labels = torch.full((num_ops,), 0, dtype=torch.long)
    # machine_labels = torch.full((num_machines,), 1, dtype=torch.long)
    # all_labels = torch.cat([op_labels, machine_labels])
    # x = F.one_hot(all_labels, num_classes=2).float().to(device)
    op_labels = torch.full((num_ops,), 0, dtype=torch.long)
    machine_labels = torch.full((num_machines,), 1, dtype=torch.long)
    all_labels = torch.cat([op_labels, machine_labels]).to(device)
    type_onehot = F.one_hot(all_labels, num_classes=2).float()
    
    # Position, Workload, Connectivity
    extra_feats = torch.zeros((num_nodes, 3), device=device) 

    extra_feats[num_ops:, 0] = -1.0 
    
    for i in range(num_ops): # step number
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
    op_load = sub_matrix.sum(dim=1) / op_conn.clamp(min=1.0)
    
    # how many operations can be processed in this machine
    m_conn = (sub_matrix > 0).float().sum(dim=0)
    # average process time for every machine
    m_load = sub_matrix.sum(dim=0) / m_conn.clamp(min=1.0)


    # Op Features 0. type 1. average processtime 2.
    extra_feats[:num_ops, 1] = op_load  # Workload
    extra_feats[:num_ops, 2] = op_conn / num_machines # Connectivity (归一化)
    
    extra_feats[num_ops:, 1] = m_load   # Workload
    extra_feats[num_ops:, 2] = m_conn / num_ops # Connectivity (归一化)

    # x shape: [Num_Nodes, 2 + 3] = [Num_Nodes, 5]
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

def compute_batch_loss(model, batch_data, T, device, edge_weight, node_marginal, edge_marginal, kl_lambda=0.1, constraint_lambda=1.0):
    data_list = batch_data.to_data_list()
    total_loss = 0.0
    count = 0
    for data in data_list:
        true_n = data.n_nodes.item() if hasattr(data, 'n_nodes') else data.x.size(0)
        if true_n == 0:
            continue

        x0 = data.x[:true_n].argmax(dim=1)
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=true_n)[0]
        e0 = (dense_adj > 0).long()

        t_i = torch.randint(0, T, (1,)).item()
        x_t, e_t = model.forward_diffusion(x0, e0, t_i, device)

        edge_index_noisy = (e_t.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
        data_i = Data(x=x_t, edge_index=edge_index_noisy)
        data_i.batch = torch.zeros(x_t.size(0), dtype=torch.long, device=device)
        node_logits, edge_logits_list = model(data_i.x, data_i.edge_index, data_i.batch, t=t_i)

        loss_node = F.cross_entropy(node_logits, x0.to(device))
        
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_logits = edge_logits_list[0]
            loss_edge = F.cross_entropy(edge_logits.view(-1, model.edge_num_classes),
                                        e0.to(device).view(-1),
                                        weight=edge_weight)
        else:
            loss_edge = 0.0

        ### !!!!!!!!!!!tbd
        ### here the calculation may has problems, the kl divergence is calculated one by one instead of for the whome distribution
        ## tbd tmp needless for now, since every operation node must have exactly one connection, this is promised by projector
        # node_probs = F.softmax(node_logits, dim=1)
        # kl_node = kl_divergence(node_probs, node_marginal.to(device))
        # if edge_logits_list and edge_logits_list[0].numel() > 0:
        #     edge_logits = edge_logits_list[0]
        #     edge_probs = F.softmax(edge_logits, dim=-1)
        #     edge_probs_avg = edge_probs.view(-1, model.edge_num_classes)
        #     kl_edge = kl_divergence(edge_probs_avg, edge_marginal.to(device))
        # else:
        #     kl_edge = 0.0
        kl_node = 0.0
        kl_edge = 0.0


#         ### !!!!!!!!!!!tbd
#         ### change to cross-entropy too
#         if edge_logits_list and edge_logits_list[0].numel() > 0:
#             # edge_probs = F.softmax(edge_logits, dim=-1)
#             # pred_edge_prob = edge_probs[..., 1]
#             # forbidden_mask = get_forbidden_mask(x0, device)
#             # forbidden_mask = forbidden_mask[:true_n, :true_n]
#             # constraint_loss = F.mse_loss(pred_edge_prob * forbidden_mask, torch.zeros_like(pred_edge_prob))
#
#             edge_logits = edge_logits_list[0]
#             forbidden_mask = get_forbidden_mask(x0, device)
#             forbidden_mask = forbidden_mask[:true_n, :true_n]
#             target_labels = torch.zeros(true_n, true_n, dtype=torch.long, device=device)
#             ce_loss_all_positions = F.cross_entropy(
#                 edge_logits.view(-1, model.edge_num_classes),
#                 target_labels.view(-1),
#                 reduction='none'
#             )
#             ce_loss_all_positions = ce_loss_all_positions.view(true_n, true_n)
#             ce_loss_masked = ce_loss_all_positions * forbidden_mask.float()
#             constraint_loss = ce_loss_masked.sum() / (forbidden_mask.sum() + 1e-8)
#         else:
#             constraint_loss = 0.0
# #############################################
#
#         constraint_validate_loss = torch.tensor(0.0, device=device)
#
#         if edge_logits_list and edge_logits_list[0].numel() > 0:
#             edge_logits = edge_logits_list[0]
#             edge_probs = F.softmax(edge_logits, dim=-1)
#             flat_probs = edge_probs.view(-1, model.edge_num_classes)
#
#             # x_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)
#             x_labels = torch.argmax(node_probs, dim=1)
#             # current_node_labels 就是 x_labels，我们直接用 x_labels
#
#             # sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
#             sampled_flat = torch.argmax(flat_probs, dim=-1)
#             candidate_edge_matrix = sampled_flat.view(true_n, true_n)
#             projected = candidate_edge_matrix
#
#             if not validate_constraints(projected, x_labels, device, exact=True):
#                 reward = -0.01
#             else:
#                 reward = 0.01
#
#
#             node_log_probs = F.log_softmax(node_logits, dim=1)
#             edge_log_probs = F.log_softmax(edge_logits.view(-1, model.edge_num_classes), dim=-1)
#
#             picked_node_log_probs = node_log_probs.gather(1, x_labels.unsqueeze(1)).sum()
#             picked_edge_log_probs = edge_log_probs.gather(1, sampled_flat.unsqueeze(1)).sum()
#
#             constraint_validate_loss = - (picked_node_log_probs + picked_edge_log_probs) * torch.tensor(reward,
#                                                                                                         device=device).detach()

        constraint_loss = torch.tensor(0.0, device=device)
        constraint_validate_loss = torch.tensor(0.0, device=device)

        loss = loss_node + loss_edge + kl_lambda * (kl_node + kl_edge) + constraint_lambda * constraint_loss + constraint_lambda * constraint_validate_loss
        total_loss += loss
        count += 1

    avg_loss = total_loss / count if count > 0 else torch.tensor(0.0, device=device)
    return avg_loss


def train_model(model, dataloader, optimizer, device, edge_weight, node_marginal, edge_marginal, epochs=20, T=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = compute_batch_loss(model, batch, T, device, edge_weight, node_marginal, edge_marginal)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


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
    def __init__(self, T=100, input_dim=5, hidden_dim=128, num_layers=6,
                 beta_start=0.0001, beta_end=0.02, nhead=4, dropout=0.1,
                 device='cuda', edge_dim=1):
        super().__init__()
        self.device = torch.device(device)
        self.T = T
        self.hidden_dim = hidden_dim

        self.beta_schedule = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta_schedule
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

        # Input Projection
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(1, edge_dim)  # 如果 edge_attr 只是时间

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

    def forward(self, x, edge_index, batch, t, time_matrix=None):
        # 1. Embedding Inputs
        # x shape: [Num_Nodes, Feature_Dim] (One-hot or similar)
        h = self.node_encoder(x.float())

        # Time Embedding
        t_tensor = torch.tensor([t], dtype=torch.float, device=x.device)
        t_emb = self.time_embedder(t_tensor)  # [1, Hidden]
        # 将 t_emb 扩展到每个节点: [Num_Nodes, Hidden]
        t_emb_node = t_emb.repeat(h.size(0), 1)

        # Edge Attributes 处理
        if time_matrix is not None:
            src, dst = edge_index
            edge_times = time_matrix[src, dst].unsqueeze(-1)
            edge_attr = self.edge_encoder(edge_times)
        else:
            print('no time metrix!!!')
            edge_attr = torch.zeros((edge_index.size(1), 1), device=x.device)
            edge_attr = self.edge_encoder(edge_attr)

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
# class LightweightIndustrialDiffusion(nn.Module):
#     def __init__(self, T=100, hidden_dim=64, beta_start=0.0001, beta_end=0.02, time_embed_dim=32, nhead=4, dropout=0.1, use_projector=True, device=device, edge_dim=1):
#         super().__init__()
#         self.device = torch.device(device)
#         self.T = T
#         self.beta_schedule = torch.linspace(beta_start, beta_end, T)
#         self.alpha = 1 - self.beta_schedule
#         self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
#         self.use_projector = use_projector
#
#         self.node_num_classes = 2  # operation machine
#         self.edge_num_classes = 2
#
#         self.time_linear = nn.Linear(time_embed_dim, time_embed_dim)
#         in_channels = self.node_num_classes + time_embed_dim
#         self.transformer1 = TransformerConv(in_channels, hidden_dim, heads=nhead, concat=False, dropout=dropout, edge_dim=edge_dim)
#         self.transformer2 = TransformerConv(hidden_dim, hidden_dim, heads=nhead, concat=False, dropout=dropout, edge_dim=edge_dim)
#         self.node_out = nn.Linear(hidden_dim, self.node_num_classes)
#         self.edge_out_dim = self.edge_num_classes + 2  #0:NoEdge, 1:Edge, 2:Prio_Mean, 3:Prio_LogStd
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, self.edge_out_dim)
#         )
#         log_Q, log_Q_bar = self._precompute_log_matrices()
#         self.register_buffer('log_Q_matrices', log_Q)
#         self.register_buffer('log_Q_bar_matrices', log_Q_bar)

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

    # def forward(self, x, edge_index, batch, t, time_matrix=None):
    #     t_tensor = torch.tensor([t], dtype=torch.float, device=x.device) / self.T
    #     t_embed = get_sinusoidal_embedding(t_tensor, self.time_linear.in_features)
    #     t_embed = self.time_linear(t_embed).repeat(x.size(0), 1)
    #
    #     x_input = torch.cat([x, t_embed], dim=1)
    #     if time_matrix is not None:
    #         src, dst = edge_index
    #         edge_times = time_matrix[src, dst].unsqueeze(-1) # 变成 [Num_Edges, 1]
    #         edge_attr = edge_times
    #     elif edge_attr is None:
    #         edge_attr = torch.zeros((edge_index.size(1), 1), device=x.device)
    #
    #     h = F.relu(self.transformer1(x_input, edge_index, edge_attr))
    #     h = F.relu(self.transformer2(h, edge_index, edge_attr))
    #     node_logits = self.node_out(h)
    #
    #     h_dense, mask = to_dense_batch(h, batch)
    #     batch_size, max_nodes, _ = h_dense.shape
    #     edge_logits_list = []
    #     for i in range(batch_size):
    #         num_nodes = int(mask[i].sum().item())
    #         h_i = h_dense[i, :num_nodes, :]
    #         edge_input = torch.cat([h_i.unsqueeze(1).expand(-1, num_nodes, -1), h_i.unsqueeze(0).expand(num_nodes, -1, -1)], dim=-1)
    #         edge_logits = self.edge_mlp(edge_input)
    #         edge_logits_list.append(edge_logits)
    #
    #     return node_logits, edge_logits_list

    def reverse_diffusion_with_logprob(self, data, device, time_guidance_scale=0.1, return_trajectory=False):
        """
        For RL sampling specifically
        """
        num_nodes = data.x.size(0)
        x = data.x.clone()

        seq_edges_src = data.edge_index[0]
        seq_edges_tgt = data.edge_index[1]
        pinned_edge_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
        pinned_edge_mask[seq_edges_src, seq_edges_tgt] = True

        node_types = x.argmax(dim=1)
        op_indices = (node_types == 0).nonzero(as_tuple=True)[0]
        machine_indices = (node_types == 1).nonzero(as_tuple=True)[0]

        allowed_mask = get_ipps_allowed_mask(node_types, data, device)

        e = torch.zeros((num_nodes, num_nodes, self.edge_num_classes), device=device)
        e[:, :, 0] = 1
        e[pinned_edge_mask] = torch.tensor([0.0, 1.0], device=device)

        total_log_prob = 0.0
        total_entropy = 0.0
        trajectory = []

        for t in range(self.T - 1, -1, -1):

            current_edge_labels = e.argmax(dim=-1)
            edge_index_t = (current_edge_labels > 0).nonzero(as_tuple=False).t().contiguous()

            # Forward Pass
            tm = data.time_matrix
            edge_outputs_list = self.forward(x, edge_index_t, data.batch, t, tm)

            if edge_outputs_list:
                edge_output = edge_outputs_list[0]
                # edge_logits = edge_logits_list[0]
                edge_logits = edge_output[:, :, :2]
                score_matrix = edge_logits[:, :, 1]  # shape: [N, N]
                
                prio_mean = edge_output[:, :, 2]
                prio_log_std = edge_output[:, :, 3]
                prio_std = torch.exp(torch.clamp(prio_log_std, min=-20, max=2))
                prio_dist = Normal(prio_mean, prio_std)
                raw_priority_sample = prio_dist.sample() 
                priority_scores = torch.sigmoid(raw_priority_sample)
                
                # priority_scores = torch.sigmoid(raw_priority)

                score_matrix = score_matrix - (data.time_matrix * time_guidance_scale)

                new_e_indices = torch.zeros((num_nodes, num_nodes), dtype=torch.long, device=device)
                new_e_indices[pinned_edge_mask] = 1

                op_machine_scores = score_matrix.clone()
                op_machine_scores[~allowed_mask] = -1e9

                valid_col_mask = torch.zeros_like(op_machine_scores, dtype=torch.bool)
                valid_col_mask[:, machine_indices] = True
                op_machine_scores[~valid_col_mask] = -1e9

                target_scores = op_machine_scores[op_indices]  # [Num_Ops, Num_Nodes] prevent machine-machine connections

                dist = torch.distributions.Categorical(logits=target_scores)

                actions = dist.sample()
                selected_prio_log_prob = prio_dist.log_prob(raw_priority_sample) # [N, N]
                relevant_prio_log_prob = selected_prio_log_prob[op_indices]
                chosen_prio_log_prob = relevant_prio_log_prob.gather(1, actions.unsqueeze(1)).squeeze(1)

                step_log_prob = dist.log_prob(actions).sum() + chosen_prio_log_prob.sum()
                
                relevant_priorities = priority_scores[op_indices]
                selected_priorities = relevant_priorities.gather(1, actions.unsqueeze(1)).squeeze(1)

                # step_log_prob = dist.log_prob(actions).sum()
                entropy_routing = dist.entropy().mean()
                entropy_prio = prio_dist.entropy().mean()
                
                step_entropy = entropy_routing + entropy_prio
                total_log_prob += step_log_prob
                total_entropy += step_entropy

                new_e_indices[op_indices, actions] = 1
                new_e_indices[pinned_edge_mask] = 1
                if return_trajectory:
                    step_snapshot = (
                        new_e_indices.detach().cpu().clone(),
                        selected_priorities.detach().cpu().clone()
                    )
                    trajectory.append(step_snapshot)

                e = F.one_hot(new_e_indices, num_classes=self.edge_num_classes).float()

        if return_trajectory:
            return e, total_log_prob, total_entropy, selected_priorities, trajectory
        else:
            return e, total_log_prob, total_entropy, selected_priorities

    
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



# 3 Industrial Training Script
# train_industrial.py  ── versión cronometrada
import os, time, argparse, torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj


# ---------- utilidades para pesos y marginales ----------
def compute_edge_weights(dataset, device):
    total_edges = 0
    class_counts = torch.zeros(2, device=device)
    for data in dataset:
        dense = to_dense_adj(data.edge_index,
                             max_num_nodes=data.x.size(0))[0]
        e0 = (dense > 0).long()
        class_counts += torch.bincount(e0.view(-1), minlength=2).to(device)
        total_edges  += e0.numel()
    class_counts[class_counts == 0] = 1.0
    w = total_edges / (2.0 * class_counts)
    return w / w.sum()

def compute_marginal_probs(dataset, device):
    node_counts = torch.zeros(2, device=device)   # 4 tipos de nodo
    edge_counts = torch.zeros(2, device=device)
    n_nodes = n_edges = 0
    for data in dataset:
        labels = data.x.argmax(dim=1)
        node_counts += torch.bincount(labels, minlength=2).float().to(device)
        n_nodes += data.x.size(0)
        dense = to_dense_adj(data.edge_index,
                             max_num_nodes=data.x.size(0))[0]
        e0 = (dense > 0).long()
        edge_counts += torch.bincount(e0.view(-1), minlength=2).float().to(device)
        n_edges += e0.numel()
    return node_counts / n_nodes, edge_counts / n_edges
# --------------------------------------------------------


def run_training(epochs=30, batch=4, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = 'industrial_model.pth'
    if os.path.exists(ckpt):
        print(f"⚠️  Found existing weights: {ckpt}. Skipping training.")
        return
    dataset  = IndustrialGraphDataset(root='industrial_dataset')
    loader   = DataLoader(dataset, batch_size=batch, shuffle=True)

    edge_w   = compute_edge_weights(dataset, device)
    node_m, edge_m = compute_marginal_probs(dataset, device)

    model     = LightweightIndustrialDiffusion(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n▶ Training INDUSTRIAL model  ({len(dataset)} graphs)")
    start = time.perf_counter()

    train_model(model, loader, optimizer, device,
                edge_weight=edge_w,
                node_marginal=node_m,
                edge_marginal=edge_m,
                epochs=epochs, T=100)

    elapsed = time.perf_counter() - start
    print(f"⏱  Finished in {elapsed/60:.1f} min  ({elapsed:.1f} s)\n")

    torch.save(model.state_dict(), 'industrial_model.pth')
    print("✅ Weights saved to  industrial_model.pth")


# -------------- entry point con argparse -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch',  type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr',     type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    run_training(epochs=args.epochs,
                 batch=args.batch,
                 lr=args.lr)
    

# 4 Industrial Running functions
import time, random, collections
import torch, networkx as nx
import numpy as np
from torch_geometric.data     import Batch
from torch_geometric.nn       import global_mean_pool, GINConv
from torch.utils.data         import DataLoader
# --------------------------------------------------------------------------
# Helpers for hashing and validity
# --------------------------------------------------------------------------
def plant_valid(node_labels: torch.Tensor, edge_mat: torch.Tensor, device):
    """True iff plant-level constraints C1–C6 hold."""
    return validate_constraints(edge_mat, node_labels, device)


def wl_hash(node_labels: torch.Tensor, edge_mat: torch.Tensor) -> str:
    """Deterministic hash ( Weisfeiler-Lehman ) for isomorphism tests."""
    G = nx.DiGraph()
    n = len(node_labels)
    for i in range(n):
        G.add_node(i, t=int(node_labels[i]))
    src, dst = torch.nonzero(edge_mat, as_tuple=True)
    for s, d in zip(src.tolist(), dst.tolist()):
        G.add_edge(s, d)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='t')


# LABEL2ID = {"MACHINE": 0,
#             "BUFFER":  1,
#             "ASSEMBLY":2,
#             "DISASSEMBLY":3}
LABEL2ID = {"OPERATION": 0, "MACHINE": 1}

from pathlib import Path
import datetime as dt
import numpy as np
from typing import Union


def _save_graphs_pt(tag: str, batch: list[dict], save_dir: Union[str, Path]) -> None:
    """
    Save a batch of graphs to a .pt file with the same schema as graphs_data_int.pt
    Keys:
      ├ adjacency_matrices : list[np.ndarray]  (int8/uint8)
      ├ node_types         : list[np.ndarray]  (int8)
      └ label2id           : dict[str,int]
    """
    from pathlib import Path
    import datetime as dt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    adj_list, node_list = [], []
    for g in batch:
        adj_list.append(g["edges"].cpu().numpy().astype(np.uint8))
        node_list.append(g["nodes"].cpu().numpy().astype(np.int8))

    payload = {
        "adjacency_matrices": adj_list,
        "node_types":         node_list,
        "label2id":           LABEL2ID
    }

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = save_dir / f"{tag}_{stamp}.pt"
    torch.save(payload, fname)
    print(f"   ↳ Graphs saved in {fname}")
    return fname


# --------------------------------------------------------------------------
# Experiment E1 – free generation
# --------------------------------------------------------------------------
def experiment_free(n_samples=300, n_nodes=15, plant_model_path = "ablation_runs_new/baseline/model.pth"):
    batch = []
    t0 = time.time()
    for _ in tqdm(range(n_samples)):
        model = LightweightIndustrialDiffusion(device=device).to(device)
        model.load_state_dict(torch.load(plant_model_path,
                                               map_location=device))
        nodes, edges = model.generate_global_graph(n_nodes)
        batch.append({"nodes": nodes,
                      "edges": edges.squeeze(0)})
    runtime = time.time() - t0
    print(f"[E1-Free]   {n_samples} samples in {runtime:.1f}s")
    #print(evaluate(batch), "\n")
    #extra_metrics(batch, tag="[E1]")
    file_name =_save_graphs_pt("E1", batch, save_dir="exp_outputs/E1/pt_file")
    return file_name


# --------------------------------------------------------------------------
# Experiment E2 – all-pinned inventory
# --------------------------------------------------------------------------
def experiment_allpinned(n_samples=300,
                         inv=(3,4,2,1), plant_model_path = "ablation_runs_new/baseline/model.pth"):   # (M, B, A, D)
    numM,numB,numA,numD = inv
    batch = []
    t0 = time.time()
    for _ in tqdm(range(n_samples)):
        model = LightweightIndustrialDiffusion(device=device).to(device)
        model.load_state_dict(torch.load(plant_model_path,
                                               map_location=device))
        nodes, edges = model.generate_global_graph_all_pinned(
            num_machines=numM,
            num_buffers=numB,
            num_assemblies=numA,
            num_disassemblies=numD)
        ok_inv = ( (nodes==0).sum()==numM and
                   (nodes==1).sum()==numB and
                   (nodes==2).sum()==numA and
                   (nodes==3).sum()==numD )
        batch.append({"nodes": nodes,
                      "edges": edges.squeeze(0),
                      "success": ok_inv})
    runtime = time.time() - t0
    print(f"[E2-AllPinned] {n_samples} samples in {runtime:.1f}s")
    #print(evaluate(batch), "\n")
    #extra_metrics(batch, tag="[E2]")
    file_name =_save_graphs_pt("E2", batch, save_dir="exp_outputs/E2/pt_file")
    return file_name


# --------------------------------------------------------------------------
# Experiment E3 – partial-pinned (30 % nodes)
# --------------------------------------------------------------------------
def experiment_partial(n_samples=300, n_nodes=20, pin_ratio=0.3, plant_model_path = "ablation_runs_new/baseline/model.pth"):
    batch = []
    t0 = time.time()
    for _ in tqdm(range(n_samples)):
        pin_counts = {"MACHINE": 1,
                      "ASSEMBLY": 1,
                      "BUFFER": int(pin_ratio*n_nodes) - 2}
        model = LightweightIndustrialDiffusion(device=device).to(device)
        model.load_state_dict(torch.load(plant_model_path,
                                               map_location=device))
        nodes, edges = model.generate_global_graph_partial_pinned(
            num_nodes=n_nodes,
            pinned_info=pin_counts)
        batch.append({"nodes": nodes,
                      "edges": edges.squeeze(0)})
    runtime = time.time() - t0
    print(f"[E3-Partial] {n_samples} samples in {runtime:.1f}s")
    #print(evaluate(batch), "\n")
    #extra_metrics(batch, tag="[E3]")
    file_name =_save_graphs_pt("E3", batch, save_dir="exp_outputs/E3/pt_file")
    return file_name


# ───────────── extra: FID / MMD ───────────────────────────────

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
    # bandwidth heurístico (mediana)
    Z = torch.cat([X, Y], 0)
    sq = torch.cdist(Z, Z, p=2.0)**2
    sigma = torch.sqrt(0.5*sq[sq>0].median())
    k = lambda A,B: torch.exp(-torch.cdist(A,B,p=2.0)**2 / (2*sigma**2))
    m, n = len(X), len(Y)
    return (k(X,X).sum() - m)/(m*(m-1)) \
         + (k(Y,Y).sum() - n)/(n*(n-1)) \
         - 2*k(X,Y).mean()

