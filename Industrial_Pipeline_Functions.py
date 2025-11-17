# Industrial Graph Pipeline

# 1 Industrial Dataset Creation 
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as pyg_utils
from tqdm import tqdm

class IndustrialGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graphs_data.pt']

    @property
    def processed_file_names(self):
        return ['processed_graphs.pt']

    def download(self):
        pass  

    def process(self):
        raw_data = torch.load(self.raw_paths[0])
        adj_matrices = raw_data['adjacency_matrices']
        node_types = raw_data['node_types']

        node_type_to_idx = {'MACHINE': 0, 'BUFFER': 1, 'ASSEMBLY': 2, 'DISASSEMBLY': 3}

        data_list = []
        for adj, types in zip(adj_matrices, node_types):
            node_labels = torch.tensor([node_type_to_idx[t] for t in types])
            x = F.one_hot(node_labels, num_classes=4).float()

            edge_index, _ = pyg_utils.dense_to_sparse(torch.tensor(adj))
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=x.size(0))
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
MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY = 0, 1, 2, 3


# def get_forbidden_mask(node_labels, device):
#     n = node_labels.size(0)
#     diag_mask = torch.eye(n, dtype=torch.bool, device=device)
#
#     allowed_mask = torch.ones((n, n), dtype=torch.bool, device=device)
#     for i in range(n):
#         for j in range(n):
#             if node_labels[i] == 1 and node_labels[j] == 1:  # BUFFER-BUFFER forbidden
#                 allowed_mask[i, j] = False
#
#
#     buf = (node_labels == BUFFER)
#     m_idx = torch.where(node_labels == MACHINE)[0]
#     asm_idx = torch.where(node_labels == ASSEMBLY)[0]
#     dis_idx = torch.where(node_labels == DISASSEMBLY)[0]
#     for i in m_idx:
#         buf_indices = torch.where(buf)[0]
#         if len(buf_indices) > 0:
#             allowed_mask[i, buf_indices[0]] = True
#
#
#
#     forbidden_mask = diag_mask | (~allowed_mask)
#     return forbidden_mask.float()


def get_forbidden_mask(node_labels, device):
    n = node_labels.size(0)
    diag_mask = torch.eye(n, dtype=torch.bool, device=device)  # 自环禁止

    buf = (node_labels == BUFFER)
    allowed_mask = torch.zeros((n, n), dtype=torch.bool, device=device)

    for i in range(n):
        for j in range(n):
            if buf[i] or buf[j]:
                if buf[i] != buf[j]:
                    allowed_mask[i, j] = True

    # 禁止自环
    forbidden_mask = diag_mask | (~allowed_mask)

    return forbidden_mask.float()

# def validate_constraints(edge_matrix, node_labels, device):
#     """
#     Devuelve True si `edge_matrix` cumple todas las restricciones industriales:
#     - No self‑loops.
#     - No Buffer→Buffer.
#     - Máximos de conexiones según tipo.
#     - No bidireccionales.
#     """
#     forbidden = get_forbidden_mask(node_labels, device)
#     # 1) Ninguna arista donde forbidden==1
#     if (edge_matrix * forbidden).any():
#         return False
#     # 2) Máximos por tipo (ejemplo MACHINE→BUFFER <=1)
#     n = node_labels.size(0)
#     # Cuenta salidas MACHINE→BUFFER
#     for i in range(n):
#         if node_labels[i]==0:
#             if edge_matrix[i, node_labels==1].sum() > 1:
#                 return False
#     # (añade aquí otras comprobaciones específicas si lo deseas)
#     return True


def validate_constraints(edge_matrix, node_labels, device, exact=True):
    E = torch.as_tensor(edge_matrix, dtype=torch.long, device=device)
    y = torch.as_tensor(node_labels, dtype=torch.long, device=device)

    # --- base hard constraints ---
    # no self-loops
    if torch.any(torch.diag(E) != 0):
        return False
    # no BUFFER-BUFFER edges
    buf = (y == BUFFER)
    if E[buf][:, buf].sum() > 0:
        return False
    # no bidirectional edges anywhere
    if torch.any(E.bool() & E.t().bool()):
        return False

    # --- MACHINE constraints ---
    m_idx = torch.where(y == MACHINE)[0]
    if exact:
        # exactly 1 outgoing MACHINE -> BUFFER
        for i in m_idx:
            if E[i, buf].sum() != 1:
                return False
        # exactly 1 incoming BUFFER -> MACHINE
        for j in m_idx:
            if E[buf, j].sum() != 1:
                return False
    else:
        for i in m_idx:
            if E[i, buf].sum() > 1:
                return False
        for j in m_idx:
            if E[buf, j].sum() > 1:
                return False

    # --- ASSEMBLY constraints ---
    asm_idx = torch.where(y == ASSEMBLY)[0]
    if exact:
        # exactly 2 inputs BUFFER -> ASSEMBLY
        for j in asm_idx:
            if E[buf, j].sum() != 2:
                return False
        # exactly 1 output ASSEMBLY -> BUFFER
        for i in asm_idx:
            if E[i, buf].sum() != 1:
                return False
    else:
        for j in asm_idx:
            if E[buf, j].sum() > 2:
                return False
        for i in asm_idx:
            if E[i, buf].sum() > 1:
                return False

    # --- DISASSEMBLY constraints ---
    dis_idx = torch.where(y == DISASSEMBLY)[0]
    if exact:
        # exactly 1 input BUFFER -> DISASSEMBLY
        for j in dis_idx:
            if E[buf, j].sum() != 1:
                return False
        # exactly 2 outputs DISASSEMBLY -> BUFFER
        for i in dis_idx:
            if E[i, buf].sum() != 2:
                return False
    else:
        for j in dis_idx:
            if E[buf, j].sum() > 1:
                return False
        for i in dis_idx:
            if E[i, buf].sum() > 2:
                return False
    return True

    
def strict_projector_industrial(node_labels, candidate_edge_matrix, device, add_additional_edges=True):
    """
    Project a sampled binary candidate adjacency matrix into one that satisfies:

      • No self-loops.
      • No bidirectional edges (if j→i already exists, i→j is not allowed).
      • MACHINE: exactly 1 outgoing edge to BUFFER (M→B) and exactly 1 incoming edge from BUFFER (B→M).
      • ASSEMBLY: exactly 2 incoming edges from BUFFER (B→A) and exactly 1 outgoing edge to BUFFER (A→B).
      • DISASSEMBLY: exactly 1 incoming edge from BUFFER (B→D) and exactly 2 outgoing edges to BUFFER (D→B).
      • All other edge types are disallowed (e.g., BUFFER→BUFFER, MACHINE→MACHINE, etc.).

    IMPORTANT:
      - This projector never creates edges that the candidate did not propose (it only keeps a subset of the
        candidate 1s). If the candidate does not contain enough valid 1s to satisfy exact cardinalities,
        the final validation step should reject the graph and the sampler should re-try.
    """
    # Ensure tensor types/shapes
    n = node_labels.size(0)
    cand = candidate_edge_matrix.to(device).long()

    # Start with empty adjacency
    projected_edges = torch.zeros((n, n), dtype=torch.long, device=device)

    # Shortcuts for types
    isM = (node_labels == MACHINE)
    isB = (node_labels == BUFFER)
    isA = (node_labels == ASSEMBLY)
    isD = (node_labels == DISASSEMBLY)

    # ---------------------------------------------------------------------------------
    # First pass: "at most" constraints with no self-loops and no bidirectionals.
    # We only accept edges that are 1 in the candidate, and we do not exceed local maxima.
    # ---------------------------------------------------------------------------------
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # no self-loops

            # avoid bidirectional: if j->i already accepted, skip i->j
            if projected_edges[j, i] == 1:
                continue

            if cand[i, j] != 1:
                continue  # only consider candidate positives

            si = node_labels[i].item()
            sj = node_labels[j].item()

            # Allowed patterns with local maxima (≤):
            # MACHINE -> BUFFER  (max 1 out per MACHINE)
            if si == MACHINE and sj == BUFFER:
                if projected_edges[i, isB].sum() == 0:
                    projected_edges[i, j] = 1

            # BUFFER -> MACHINE  (max 1 in per MACHINE)
            elif si == BUFFER and sj == MACHINE:
                if projected_edges[isB, j].sum() == 0:
                    projected_edges[i, j] = 1

            # BUFFER -> ASSEMBLY (max 2 in per ASSEMBLY)
            elif si == BUFFER and sj == ASSEMBLY:
                if projected_edges[isB, j].sum() < 2:
                    projected_edges[i, j] = 1

            # ASSEMBLY -> BUFFER (max 1 out per ASSEMBLY)
            elif si == ASSEMBLY and sj == BUFFER:
                if projected_edges[i, isB].sum() < 1:
                    projected_edges[i, j] = 1

            # BUFFER -> DISASSEMBLY (max 1 in per DISASSEMBLY)
            elif si == BUFFER and sj == DISASSEMBLY:
                if projected_edges[isB, j].sum() < 1:
                    projected_edges[i, j] = 1

            # DISASSEMBLY -> BUFFER (max 2 out per DISASSEMBLY)
            elif si == DISASSEMBLY and sj == BUFFER:
                if projected_edges[i, isB].sum() < 2:
                    projected_edges[i, j] = 1

    return projected_edges


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
        node_probs = F.softmax(node_logits, dim=1)
        kl_node = kl_divergence(node_probs, node_marginal.to(device))
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_logits = edge_logits_list[0]
            edge_probs = F.softmax(edge_logits, dim=-1)
            edge_probs_avg = edge_probs.view(-1, model.edge_num_classes)
            kl_edge = kl_divergence(edge_probs_avg, edge_marginal.to(device))
        else:
            kl_edge = 0.0

        ### !!!!!!!!!!!tbd
        ### change to cross-entropy too
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            # edge_probs = F.softmax(edge_logits, dim=-1)
            # pred_edge_prob = edge_probs[..., 1]
            # forbidden_mask = get_forbidden_mask(x0, device)
            # forbidden_mask = forbidden_mask[:true_n, :true_n]
            # constraint_loss = F.mse_loss(pred_edge_prob * forbidden_mask, torch.zeros_like(pred_edge_prob))

            edge_logits = edge_logits_list[0]
            forbidden_mask = get_forbidden_mask(x0, device)
            forbidden_mask = forbidden_mask[:true_n, :true_n]
            target_labels = torch.zeros(true_n, true_n, dtype=torch.long, device=device)
            ce_loss_all_positions = F.cross_entropy(
                edge_logits.view(-1, model.edge_num_classes),
                target_labels.view(-1),
                reduction='none'
            )
            ce_loss_all_positions = ce_loss_all_positions.view(true_n, true_n)
            ce_loss_masked = ce_loss_all_positions * forbidden_mask.float()
            constraint_loss = ce_loss_masked.sum() / (forbidden_mask.sum() + 1e-8)
        else:
            constraint_loss = 0.0
#############################################

        constraint_validate_loss = torch.tensor(0.0, device=device)

        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_logits = edge_logits_list[0]
            edge_probs = F.softmax(edge_logits, dim=-1)
            flat_probs = edge_probs.view(-1, model.edge_num_classes)

            # x_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)
            x_labels = torch.argmax(node_probs, dim=1)
            # current_node_labels 就是 x_labels，我们直接用 x_labels

            # sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
            sampled_flat = torch.argmax(flat_probs, dim=-1)
            candidate_edge_matrix = sampled_flat.view(true_n, true_n)
            projected = candidate_edge_matrix

            if not validate_constraints(projected, x_labels, device, exact=True):
                reward = -0.01
            else:
                reward = 0.01


            node_log_probs = F.log_softmax(node_logits, dim=1)
            edge_log_probs = F.log_softmax(edge_logits.view(-1, model.edge_num_classes), dim=-1)

            picked_node_log_probs = node_log_probs.gather(1, x_labels.unsqueeze(1)).sum()
            picked_edge_log_probs = edge_log_probs.gather(1, sampled_flat.unsqueeze(1)).sum()

            constraint_validate_loss = - (picked_node_log_probs + picked_edge_log_probs) * torch.tensor(reward,
                                                                                                        device=device).detach()


        # x_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)
        # x = F.one_hot(x_labels, num_classes=model.node_num_classes).float()
        # if edge_logits_list and edge_logits_list[0].numel() > 0:
        #     edge_logits = edge_logits_list[0]
        #     edge_probs = F.softmax(edge_logits, dim=-1)
        #     flat_probs = edge_probs.view(-1, model.edge_num_classes)
        #     current_node_labels = x.argmax(dim=1)
        #     sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
        #     candidate_edge_matrix = sampled_flat.view(true_n, true_n)
        #     projected = candidate_edge_matrix
        #     if not validate_constraints(projected, current_node_labels, device, exact=True):
        #         constraint_validate_loss = torch.tensor(0.01)
        #     else:
        #         constraint_validate_loss = torch.tensor(0)

        # constraint_validate_loss = 0


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

       
class LightweightIndustrialDiffusion(nn.Module):
    def __init__(self, T=100, hidden_dim=12, beta_start=0.0001, beta_end=0.02, time_embed_dim=16, nhead=4, dropout=0.1, use_projector=True, device=device):
        super().__init__()
        self.device = torch.device(device)  
        self.T = T
        self.beta_schedule = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta_schedule
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.use_projector = use_projector
    
        self.node_num_classes = 4  # MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY
        self.edge_num_classes = 2

        self.time_linear = nn.Linear(time_embed_dim, time_embed_dim)
        in_channels = self.node_num_classes + time_embed_dim
        self.transformer1 = TransformerConv(in_channels, hidden_dim, heads=nhead, concat=False, dropout=dropout)
        self.transformer2 = TransformerConv(hidden_dim, hidden_dim, heads=nhead, concat=False, dropout=dropout)
        self.node_out = nn.Linear(hidden_dim, self.node_num_classes)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.edge_num_classes)
        )
        log_Q, log_Q_bar = self._precompute_log_matrices()
        self.register_buffer('log_Q_matrices', log_Q)
        self.register_buffer('log_Q_bar_matrices', log_Q_bar)

    def _precompute_log_matrices(self):
        """
        根据 self.alpha_bar (即 p_keep) 预计算 Q_t 和 Q_bar_t 的对数矩阵。
        self.alpha_bar[t] 被解释为 1 - beta_bar_t
        (基于 D3PM 论文附录 A.2.1 和 A.4.1)
        """

        K = self.node_num_classes
        T = self.T  # 假设 self.T 存储了总步数
        device = self.alpha_bar.device

        # 初始化存储矩阵
        log_Q_matrices = torch.zeros((T, K, K), device=device)
        log_Q_bar_matrices = torch.zeros((T, K, K), device=device)

        # 1. 预计算 log(Q_bar_t) = log(q(x_t | x_0))
        for t in range(T):
            # p_keep = self.alpha_bar[t]
            # 在D3PM论文的均匀扩散公式中, 这对应 1 - beta_bar_t
            # [cite: 530]
            p_keep_t = self.alpha_bar[t]
            beta_bar_t = 1.0 - p_keep_t

            # 构建 Q_bar_t 矩阵
            # (q(xt=i | x0=i) = p_keep + (1-p_keep)/K)
            # (q(xt=j | x0=i) = (1-p_keep)/K)
            # 这等价于 (1 - beta_bar_t) * I + (beta_bar_t / K) * 11^T

            off_diag_val = beta_bar_t / K
            diag_val = p_keep_t + off_diag_val

            Q_bar_t = torch.full((K, K), fill_value=off_diag_val, device=device)
            Q_bar_t.fill_diagonal_(diag_val)

            # 存储 log 矩阵, 增加 1e-12 避免 log(0)
            log_Q_bar_matrices[t] = torch.log(Q_bar_t + 1e-12)

        # 2. 预计算 log(Q_t) = log(q(x_t | x_{t-1}))
        # 我们使用 Q_t = Q_bar_t * (Q_bar_{t-1})^{-1}
        # 对于均匀扩散, 这可以简化计算
        for t in range(1, T):  # t=0 时 Q_0 未定义 (或不需要)
            alpha_bar_t = self.alpha_bar[t]
            alpha_bar_t_minus_1 = self.alpha_bar[t - 1]

            # p_keep_t = 1 - beta_t
            # p_keep_t = alpha_bar_t / alpha_bar_t_minus_1
            # beta_t = 1.0 - (alpha_bar_t / alpha_bar_t_minus_1)
            # (这来自 D3PM 论文附录 A.4.1 [cite: 530] 和 DDPM 的推导)

            # 更稳健的推导 (如我们之前讨论):
            # beta_bar_t = 1.0 - alpha_bar_t
            # beta_bar_t_minus_1 = 1.0 - alpha_bar_t_minus_1
            # beta_t = (beta_bar_t - beta_bar_t_minus_1) / (1.0 - beta_bar_t_minus_1)

            # 确保 alpha_bar_t_minus_1 不是 0, 避免除零
            if alpha_bar_t_minus_1 == 0:
                # 这发生在 t 接近 T 时, 噪声已满
                beta_t = 1.0  # 完全随机化
            else:
                beta_t = (alpha_bar_t_minus_1 - alpha_bar_t) / alpha_bar_t_minus_1

            if beta_t < 0: beta_t = 0.0  # 剪裁, 避免数值误差

            off_diag_val_q = beta_t / K
            diag_val_q = (1.0 - beta_t) + off_diag_val_q

            Q_t = torch.full((K, K), fill_value=off_diag_val_q, device=device)
            Q_t.fill_diagonal_(diag_val_q)

            log_Q_matrices[t] = torch.log(Q_t + 1e-12)
            return log_Q_matrices, log_Q_bar_matrices

    def _get_posterior_logits(self, x_t_onehot, x_0_pred_labels, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0_pred) 的 logits。
        基于 D3PM 论文, Equation (3) [cite: 82]。

        参数:
        x_t_onehot (torch.Tensor): 当前的 x_t (one-hot), 形状 [N, K]
        x_0_pred_labels (torch.Tensor): 采样的预测 x_0 (整数标签), 形状 [N]
        t (int): 当前时间步 (从 T-1 到 1)

        返回:
        torch.Tensor: log_logits, 形状 [N, K]
        """

        if t == 0:
            raise ValueError("后验 q(x_{t-1} | ...) 在 t=0 时未定义。")

        # --- 获取预先计算好的 log 转移矩阵 ---
        # log q(x_t | x_{t-1})
        log_q_xt_given_xt_minus_1 = self.log_Q_matrices[t]  # [K, K]

        # log q(x_{t-1} | x_0)
        log_q_xt_minus_1_given_x0 = self.log_Q_bar_matrices[t - 1]  # [K, K]

        x_t_int = x_t_onehot.argmax(dim=1)  # [N]

        # --- 实现贝叶斯定理: log p(A|B,C) = log p(B|A,C) + log p(A|C) ---

        # 1. log q(x_t | x_{t-1})
        # 我们需要 p(x_t=x_t_int[n] | x_{t-1}=k)
        # log_q_xt_given_xt_minus_1.T 的形状是 [K_out, K_in] = [xt, xt-1]
        # .T[k, j] = log q(x_t=j | x_{t-1}=k)
        log_term_1 = log_q_xt_given_xt_minus_1.T  # [K, K]
        # log_term_1_gathered[n, k] = log q(x_t=x_t_int[n] | x_{t-1}=k)
        log_term_1_gathered = log_term_1[:, x_t_int].T  # [N, K]

        # 2. log q(x_{t-1} | x_0)
        # 我们需要 p(x_{t-1}=k | x_0=x_0_pred_int[n])
        # log_q_xt_minus_1_given_x0[i, j] = log q(x_{t-1}=j | x_0=i)
        # log_term_2_gathered[n, k] = log q(x_{t-1}=k | x_0=x_0_pred_int[n])
        log_term_2_gathered = log_q_xt_minus_1_given_x0[x_0_pred_labels, :]  # [N, K]

        # log_logits = log_term_1_gathered + log_term_2_gathered
        log_logits = log_term_1_gathered + log_term_2_gathered

        # 返回 logits (未归一化的 log 概率)
        return log_logits
    def forward(self, x, edge_index, batch, t):
        t_tensor = torch.tensor([t], dtype=torch.float, device=x.device) / self.T
        t_embed = get_sinusoidal_embedding(t_tensor, self.time_linear.in_features)
        t_embed = self.time_linear(t_embed).repeat(x.size(0), 1)

        x_input = torch.cat([x, t_embed], dim=1)
        h = F.relu(self.transformer1(x_input, edge_index))
        h = F.relu(self.transformer2(h, edge_index))
        node_logits = self.node_out(h)

        h_dense, mask = to_dense_batch(h, batch)
        batch_size, max_nodes, _ = h_dense.shape
        edge_logits_list = []
        for i in range(batch_size):
            num_nodes = int(mask[i].sum().item())
            h_i = h_dense[i, :num_nodes, :]
            edge_input = torch.cat([h_i.unsqueeze(1).expand(-1, num_nodes, -1), h_i.unsqueeze(0).expand(num_nodes, -1, -1)], dim=-1)
            edge_logits = self.edge_mlp(edge_input)
            edge_logits_list.append(edge_logits)

        return node_logits, edge_logits_list


    def forward_diffusion(self, x0, e0, t, device):
        p_keep = self.alpha_bar[t].item()
        rand_vals = torch.rand(x0.shape, device=device)
        random_node = torch.randint(0, self.node_num_classes, x0.shape, device=device)
        x_t = torch.where(rand_vals < p_keep, x0, random_node)
        x_t_onehot = F.one_hot(x_t, num_classes=self.node_num_classes).float()

        rand_vals_e = torch.rand(e0.shape, device=device)
        random_edge = torch.randint(0, self.edge_num_classes, e0.shape, device=device)
        e_t_raw = torch.where(rand_vals_e < p_keep, e0, random_edge)

        if self.use_projector:
            projected_edges = strict_projector_industrial(x_t, e_t_raw, device)
        else:
            projected_edges = e_t_raw # No se aplica projector

        e_t_onehot = F.one_hot(projected_edges.long(), num_classes=self.edge_num_classes).float()

        return x_t_onehot, e_t_onehot


    def reverse_diffusion_single(self, data, device, save_intermediate=True):
        num_nodes = data.x.size(0)

        x = torch.randint(0, self.node_num_classes, (num_nodes,), device=device)
        x = F.one_hot(x, num_classes=self.node_num_classes).float()
        # e = torch.randint(0, self.edge_num_classes, (num_nodes, num_nodes), device=device)
        # e = F.one_hot(e, num_classes=self.edge_num_classes).float()
        e = torch.zeros((num_nodes, num_nodes, self.edge_num_classes), device=device)
        e[:, :, 0] = 1

        intermediate_graphs = []
        max_attempts = 20

        for t in range(self.T - 1, -1, -1):

            current_edge_labels = e.argmax(dim=-1)  # [N, N]
            edge_index_t = (current_edge_labels > 0).nonzero(as_tuple=False).t().contiguous()

            node_logits, edge_logits_list = self.forward(x, edge_index_t, data.batch, t)
            # node_probs = F.softmax(node_logits, dim=1)
            # x_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)
            # x = F.one_hot(x_labels, num_classes=self.node_num_classes).float()
            if t > 0:
                node_probs = F.softmax(node_logits, dim=1)  # p_theta(x0 | x_t) [cite: 127]

                # 1.2 采样一个 x0_pred (D3PM 论文中采样一个 x0_tilde [cite: 128])
                x_0_pred_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)  # [N]

                # 1.3 计算后验 q(x_{t-1} | x_t, x0_pred) 的 logits
                # (这里 x 是 x_t, one-hot 格式)
                posterior_logits = self._get_posterior_logits(x, x_0_pred_labels, t)  # [cite: 128, 82]

                # 1.4 从后验 q 中采样 x_{t-1}
                posterior_probs = F.softmax(posterior_logits, dim=1)
                x_labels = torch.multinomial(posterior_probs, num_samples=1).squeeze(1)
                x = F.one_hot(x_labels, num_classes=self.node_num_classes).float()

            else:  # t == 0 时
                node_probs = F.softmax(node_logits, dim=1)
                x_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)
                x = F.one_hot(x_labels, num_classes=self.node_num_classes).float()


            if edge_logits_list and edge_logits_list[0].numel() > 0:
                edge_logits = edge_logits_list[0]
                edge_probs  = F.softmax(edge_logits, dim=-1)
                flat_probs  = edge_probs.view(-1, self.edge_num_classes)
                current_node_labels = x.argmax(dim=1)

                found = False
                if t in (1, 0):
                    for attempt in range(max_attempts):
                        sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
                        candidate_edge_matrix = sampled_flat.view(num_nodes, num_nodes)

                        projected = strict_projector_industrial(current_node_labels, candidate_edge_matrix, device) \
                                    if self.use_projector else candidate_edge_matrix
                        if validate_constraints(projected, current_node_labels, device, exact=True):
                            e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()
                            found = True
                            break
                        # e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()
                        # 💡 qui il pezzo che mi chiedevi dove mettere:
                else:
                    sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
                    candidate_edge_matrix = sampled_flat.view(num_nodes, num_nodes)

                    projected = strict_projector_industrial(current_node_labels, candidate_edge_matrix, device) \
                        if self.use_projector else candidate_edge_matrix
                    e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()

            if save_intermediate:
                if validate_constraints(projected, current_node_labels, device, exact=True):
                    e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()
                    found = True
                    break


            if save_intermediate:
                intermediate_graphs.append(Data(
                    x=x.clone(),
                    edge_index=(e.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
                ))

        final_node_labels = x.argmax(dim=1)
        final_edge_labels = e.argmax(dim=-1)

        return final_node_labels, final_edge_labels.unsqueeze(0), intermediate_graphs

    def reverse_diffusion_single_counts(self, data, pinned_mask, device, save_intermediate=True):
        """
        Similar to reverse_diffusion_single, but it ONLY re-samples those nodes that are NOT fixed (pinned_mask[i] = False).
        pinned_mask is a boolean tensor of size [num_nodes].

        Example:
        pinned_mask = [True, True, False, ...] ⇒ the types of nodes 0 and 1 are preserved, and the rest are re-sampled.
        """
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data

        num_nodes = data.x.size(0)

        # Inicializa e sin aristas (todas clase 0 = no-edge).
        e = torch.zeros((num_nodes, num_nodes, self.edge_num_classes), device=device)
        e[:, :, 0] = 1

        # x actual (arranca en data.x, que ya contiene la info pineada)
        x = data.x.clone()

        intermediate_graphs = []

        for t in range(self.T - 1, -1, -1):
            node_logits, edge_logits_list = self.forward(x, data.edge_index, data.batch, t)

            node_probs = F.softmax(node_logits, dim=1)
            sampled_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)  # (num_nodes,)

            # Actualizamos x SOLO para nodos que no estén pineados
            current_x_int = x.argmax(dim=1)  # entero
            for i in range(num_nodes):
                if not pinned_mask[i]:
                    current_x_int[i] = sampled_labels[i]

            x = F.one_hot(current_x_int, num_classes=self.node_num_classes).float()

            # Muestreo de edges
            if edge_logits_list and edge_logits_list[0].numel() > 0:
                edge_logits = edge_logits_list[0]  # (num_nodes, num_nodes, 2)
                # edge_probs = F.softmax(edge_logits, dim=-1)
                # candidate_edge_matrix = torch.multinomial(edge_probs.view(-1, self.edge_num_classes), 1).view(num_nodes,
                #                                                                                               num_nodes)

                # current_node_labels = x.argmax(dim=1)
                # projected_edges = strict_projector_industrial(current_node_labels, candidate_edge_matrix, device)
                # e = F.one_hot(projected_edges.long(), num_classes=self.edge_num_classes).float()
                ######################
                edge_probs = F.softmax(edge_logits, dim=-1)
                max_attempts = 20
                current_node_labels = x.argmax(dim=1)

                found = False
                if t in (1, 0):
                    for attempt in range(max_attempts):
                        candidate_edge_matrix = torch.multinomial(edge_probs.view(-1, self.edge_num_classes), 1).view(
                            num_nodes,
                            num_nodes)

                        projected = strict_projector_industrial(current_node_labels, candidate_edge_matrix, device) \
                            if self.use_projector else candidate_edge_matrix
                        if validate_constraints(projected, current_node_labels, device, exact=True):
                            e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()
                            found = True
                            break
                        # e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()
                        # 💡 qui il pezzo che mi chiedevi dove mettere:
                else:
                    candidate_edge_matrix = torch.multinomial(edge_probs.view(-1, self.edge_num_classes), 1).view(
                        num_nodes,
                        num_nodes)

                    projected = strict_projector_industrial(current_node_labels, candidate_edge_matrix, device) \
                        if self.use_projector else candidate_edge_matrix
                    e = F.one_hot(projected.long(), num_classes=self.edge_num_classes).float()
                ######################

            if save_intermediate:
                new_data = Data(
                    x=x.clone(),
                    edge_index=(e.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
                )
                intermediate_graphs.append(new_data)

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
        
   
    def generate_global_graph_all_pinned(self, num_machines, num_buffers, num_assemblies, num_disassemblies):
        total_nodes = num_machines + num_buffers + num_assemblies + num_disassemblies
        node_types_list = ([MACHINE] * num_machines +
                        [BUFFER] * num_buffers +
                        [ASSEMBLY] * num_assemblies +
                        [DISASSEMBLY] * num_disassemblies)

        perm = torch.randperm(total_nodes)
        pinned_types = torch.tensor(node_types_list, device=self.device)[perm]
        pinned_x = F.one_hot(pinned_types, num_classes=self.node_num_classes).float()
        pinned_mask = torch.ones(total_nodes, dtype=torch.bool, device=self.device)

        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        data = Data(x=pinned_x, edge_index=edge_index)
        data.batch = torch.zeros(total_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.reverse_diffusion_single_counts(data, pinned_mask, self.device, False)
        return final_nodes, final_edges


    def generate_global_graph_partial_pinned(self, num_nodes, pinned_info):
        type_map = {"MACHINE": MACHINE, "BUFFER": BUFFER, "ASSEMBLY": ASSEMBLY, "DISASSEMBLY": DISASSEMBLY}
        pinned_list = []
        for type_str, count in pinned_info.items():
            pinned_list.extend([type_map[type_str]] * count)

        pinned_count = len(pinned_list)
        if pinned_count > num_nodes:
            raise ValueError("Has pineado más nodos de los existentes.")

        free_count = num_nodes - pinned_count
        pinned_list = torch.tensor(pinned_list, device=self.device)

        perm_pinned = torch.randperm(pinned_count, device=self.device)
        pinned_list_shuf = pinned_list[perm_pinned]

        node_labels = torch.full((num_nodes,), -1, device=self.device, dtype=torch.long)
        pos_pinned = torch.randperm(num_nodes, device=self.device)[:pinned_count]
        node_labels[pos_pinned] = pinned_list_shuf
        pinned_mask = (node_labels != -1)

        for i in range(num_nodes):
            if node_labels[i] == -1:
                node_labels[i] = torch.randint(0, self.node_num_classes, (1,), device=self.device)

        pinned_x = F.one_hot(node_labels, num_classes=self.node_num_classes).float()

        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        data = Data(x=pinned_x, edge_index=edge_index)
        data.batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.reverse_diffusion_single_counts(data, pinned_mask, self.device, False)
        return final_nodes, final_edges


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
    node_counts = torch.zeros(4, device=device)   # 4 tipos de nodo
    edge_counts = torch.zeros(2, device=device)
    n_nodes = n_edges = 0
    for data in dataset:
        labels = data.x.argmax(dim=1)
        node_counts += torch.bincount(labels, minlength=4).float().to(device)
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


# 1-solo: mapping fijo que ya usamos en el script
LABEL2ID = {"MACHINE": 0,
            "BUFFER":  1,
            "ASSEMBLY":2,
            "DISASSEMBLY":3}


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
    def __init__(self, in_dim=4, hid=64, out=128):
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
        x = torch.nn.functional.one_hot(g["nodes"], num_classes=4).float()
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


def extra_metrics(batch, tag=""):
    """Calcula FID y MMD de esta tanda frente al set de training."""
    # 1)  preparar encoder (si ya lo tienes entrenado, cárgale pesos)
    enc = GraphEncoder(in_dim=4).to(device).eval()

    # 2)  embeddings de training   (usa los que ya cargaste al principio)
    global _Z_train          # cache en memoria
    if '_Z_train' not in globals():
        train_objs = []
        lmap = {"MACHINE":0,"BUFFER":1,"ASSEMBLY":2,"DISASSEMBLY":3}
        for A, types in zip(train["adjacency_matrices"], train["node_types"]):
            xlabs = torch.tensor([lmap[t] for t in types])
            train_objs.append({"nodes": xlabs,
                               "edges": torch.tensor(A)})
        _Z_train = encode_graphs(train_objs, enc, device)

    # 3)  embeddings de la muestra
    Z_samp = encode_graphs(batch, enc, device)

    # 4)  FID
    mu_t, cov_t = _Z_train.mean(0), torch.from_numpy(np.cov(_Z_train.T.numpy()))
    mu_s, cov_s = Z_samp.mean(0), torch.from_numpy(np.cov(Z_samp.T.numpy()))
    fid  = frechet(mu_t, cov_t, mu_s, cov_s).item()

    # 5)  MMD
    mmd = mmd_rbf(_Z_train, Z_samp).item()

    print(f"   ↳ FID={fid:7.2f}   MMD={mmd:7.4f}   {tag}")