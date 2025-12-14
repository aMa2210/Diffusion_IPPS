import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import Batch
from torch_geometric.utils import softmax as geo_softmax
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool
import random
import os
import glob

def load_data_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    machines_list = data.get("machines", [])
    jobs_data = {}
    all_processing_times = []
    
    for idx, wp in enumerate(data["workpieces"]):
        job_id = idx
        ops_list = []
        opt_machines = wp["optional_machines"]
        proc_times = wp["processing_time"]
        for m_list, t_list in zip(opt_machines, proc_times):
            machine_dict = {}
            for m_id, t_val in zip(m_list, t_list):
                machine_dict[m_id] = t_val
                all_processing_times.append(t_val)
            ops_list.append({'machines': machine_dict})
        jobs_data[job_id] = ops_list
    global_max_time = max(all_processing_times) if all_processing_times else 1.0
    return machines_list, jobs_data, global_max_time


# ==========================================
# 2. 图构建 (Graph Builder)
# ==========================================
def build_hetero_graph(machines_list, jobs_data, norm_factor):
    graph = HeteroData()
    num_machines = len(machines_list)
    num_jobs = len(jobs_data)
    total_ops = sum(len(ops) for ops in jobs_data.values())

    # --- Node Feature Init (Placeholders) ---
    graph['machine'].x = torch.zeros(num_machines, 6)  # [cite: 535-541]
    graph['job'].x = torch.zeros(num_jobs, 3)  # [cite: 543-546]
    graph['operation'].x = torch.zeros(total_ops, 9)  # [cite: 529-534]
    graph['combination'].x = torch.zeros(num_jobs, 2)

    op_to_job = torch.zeros(total_ops, dtype=torch.long)

    # --- Edge Construction ---
    edge_index_o_o, edge_index_o_m, edge_attr_o_m = [], [], []  # operation to operation/ operation to doable machine/ operation to machine process time
    edge_index_o_c, edge_index_c_j = [], []  # operation to combination/ combination to job

    op_global_id = 0
    sorted_job_ids = sorted(jobs_data.keys())

    for j_idx, job_id in enumerate(sorted_job_ids):
        ops_list = jobs_data[job_id]
        c_idx = j_idx  # FJSP: 1 Comb per Job

        edge_index_c_j.append([c_idx, j_idx])
        current_job_op_ids = []

        for op_data in ops_list:
            curr_op_id = op_global_id
            current_job_op_ids.append(curr_op_id)
            op_to_job[curr_op_id] = j_idx  # Store mapping

            edge_index_o_c.append([curr_op_id, c_idx])

            for m_id, proc_time in op_data['machines'].items():
                m_idx = m_id - 1
                edge_index_o_m.append([curr_op_id, m_idx])
                edge_attr_o_m.append([proc_time / norm_factor])

            op_global_id += 1

        for i in range(len(current_job_op_ids) - 1):
            edge_index_o_o.append([current_job_op_ids[i], current_job_op_ids[i + 1]])

    # To Tensors
    graph['operation', 'precedes', 'operation'].edge_index = torch.tensor(edge_index_o_o,
                                                                          dtype=torch.long).t().contiguous()

    om_index = torch.tensor(edge_index_o_m, dtype=torch.long).t().contiguous()
    om_attr = torch.tensor(edge_attr_o_m, dtype=torch.float)
    graph['operation', 'processed_by', 'machine'].edge_index = om_index
    graph['operation', 'processed_by', 'machine'].edge_attr = om_attr
    graph['machine', 'processes', 'operation'].edge_index = om_index.flip(0)
    graph['machine', 'processes', 'operation'].edge_attr = om_attr

    oc_index = torch.tensor(edge_index_o_c, dtype=torch.long).t().contiguous()
    graph['operation', 'in_combination', 'combination'].edge_index = oc_index
    graph['combination', 'has_operation', 'operation'].edge_index = oc_index.flip(0)

    cj_index = torch.tensor(edge_index_c_j, dtype=torch.long).t().contiguous()
    graph['combination', 'belongs_to', 'job'].edge_index = cj_index
    graph['job', 'has_combination', 'combination'].edge_index = cj_index.flip(0)

    # Store auxiliary data
    graph.op_to_job = op_to_job

    return graph


# ==========================================
# 3. 神经网络模型 (HGAT + Actor-Critic)
# ==========================================
class HGATModel(nn.Module):
    def __init__(self, metadata, hidden_dim=64, num_heads=2, num_layers=2):
        super().__init__()
        self.feature_proj = nn.ModuleDict({
            'operation': nn.Linear(9, hidden_dim),
            'machine': nn.Linear(6, hidden_dim),
            'job': nn.Linear(3, hidden_dim),
            'combination': nn.Linear(2, hidden_dim)
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src_type, rel, dst_type = edge_type

                # 【关键修改】辨别哪些边有时间属性
                # 我们只给 O-M 和 M-O 边设置 edge_dim=1，其他边 edge_dim=None
                if rel == 'processed_by' or rel == 'processes':
                    # 注意：GATv2Conv 的 edge_dim 参数用于接收边特征
                    conv_dict[edge_type] = GATv2Conv(
                        hidden_dim, hidden_dim // num_heads,
                        heads=num_heads,
                        add_self_loops=False,
                        edge_dim=1  # <--- 告诉网络这里有 1 维的边特征(加工时间)
                    )
                else:
                    conv_dict[edge_type] = GATv2Conv(
                        hidden_dim, hidden_dim // num_heads,
                        heads=num_heads,
                        add_self_loops=False,
                        edge_dim=None  # 其他边没有特征
                    )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # Actor: H_ijk = h_op || h_m || h_j || h_t (Total 6 * hidden) [cite: 285]
        self.actor_mlp = nn.Sequential(
            nn.Linear(6 * hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Critic: h_t (Total 3 * hidden)
        self.critic_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, graph, valid_action_indices, action_batch_index=None):
        x_dict = {key: self.feature_proj[key](graph[key].x) for key in graph.node_types}
        edge_index_dict = graph.edge_index_dict

        # 构建边特征
        edge_attr_dict = {}
        for edge_type in graph.edge_types:
            src, rel, dst = edge_type
            if rel == 'processed_by' or rel == 'processes':
                edge_attr_dict[edge_type] = graph[edge_type].edge_attr
            else:
                edge_attr_dict[edge_type] = None

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(v) for k, v in x_dict.items()}

        h_ops, h_mchs, h_jobs = x_dict['operation'], x_dict['machine'], x_dict['job']

        # --- 【关键修复开始】 ---
        # 使用 global_mean_pool 分别计算 Batch 中每个图的 h_t
        # graph['operation'].batch 存储了每个节点属于 Batch 中的哪个图 (例如 [0, 0, 0, 1, 1, 1...])
        
        if hasattr(graph['operation'], 'batch') and graph['operation'].batch is not None:
            # Batch 模式 (训练时)
            batch_ops = graph['operation'].batch
            batch_mchs = graph['machine'].batch
            batch_jobs = graph['job'].batch
            
            # 池化后形状: [batch_size, hidden_dim]
            h_ops_pooled = global_mean_pool(h_ops, batch_ops)
            h_mchs_pooled = global_mean_pool(h_mchs, batch_mchs)
            h_jobs_pooled = global_mean_pool(h_jobs, batch_jobs)
            
            h_t = torch.cat([h_ops_pooled, h_mchs_pooled, h_jobs_pooled], dim=1)
        else:
            # 单图模式 (推理时)
            # 形状: [1, hidden_dim]
            h_t = torch.cat([torch.mean(h_ops, 0, keepdim=True), 
                             torch.mean(h_mchs, 0, keepdim=True),
                             torch.mean(h_jobs, 0, keepdim=True)], dim=1)
        # --- 【关键修复结束】 ---

        # Critic Value (h_t 现在对于 Batch 输入是 [batch_size, dim]，对于单图是 [1, dim])
        value = self.critic_mlp(h_t)

        if len(valid_action_indices) == 0:
            return None, value
            
        om_edge_index = graph['operation', 'processed_by', 'machine'].edge_index
        valid_edges = om_edge_index[:, valid_action_indices]
        valid_op_ids, valid_m_ids = valid_edges[0], valid_edges[1]
        valid_job_ids = graph.op_to_job[valid_op_ids]

        if action_batch_index is not None:
            # 现在 h_t 是 [batch_size, dim]，可以用 action_batch_index 正确索引了
            h_t_expanded = h_t[action_batch_index] 
        else:
            # 单图模式
            h_t_expanded = h_t.expand(len(valid_op_ids), -1)

        pair_embeds = torch.cat([
            h_ops[valid_op_ids], 
            h_mchs[valid_m_ids], 
            h_jobs[valid_job_ids],
            h_t_expanded
        ], dim=1)

        scores = self.actor_mlp(pair_embeds).squeeze(-1)

        if action_batch_index is not None:
            probs = geo_softmax(scores, action_batch_index)
        else:
            probs = F.softmax(scores, dim=0)
            
        return probs, value


# ==========================================
# 4. 环境逻辑 (Environment)
# ==========================================
class FJSPEnv:
    def __init__(self, machines_list, jobs_data, graph_template, time_scale):
        self.jobs_data = jobs_data
        self.machines_list = machines_list
        self.graph_template = graph_template
        self.num_jobs = len(jobs_data)
        self.avg_op_times = self._precompute_avg_times()
        # 计算一个粗略的最大时间用于归一化 (例如所有工序时间总和)
        self.max_time_scale = sum(self.avg_op_times.values())
        self.time_scale = time_scale

    def _precompute_avg_times(self):
        avg = {}
        idx = 0
        for jid in sorted(self.jobs_data):
            for op in self.jobs_data[jid]:
                vals = list(op['machines'].values())
                avg[idx] = sum(vals) / len(vals)
                idx += 1
        return avg

    def reset(self):
        self.state_op_status = {i: 0 for i in range(self.graph_template['operation'].num_nodes)}
        self.state_m_free_time = np.zeros(len(self.machines_list))
        self.current_time = 0.0
        self.finished_cnt = 0
        self.total_ops = len(self.state_op_status)
        self.jobs_next_op_time = {jid: 0.0 for jid in self.jobs_data.keys()} # [新增] 记录每个Job的可用时间

        curr = 0
        for jid in sorted(self.jobs_data):
            self.state_op_status[curr] = 1
            curr += len(self.jobs_data[jid])

        self.graph = self.graph_template.clone()
        # 初始时，last_est 设为理论下界
        self.last_est_makespan = self._calc_est_makespan()
        self._update_features()
        return self.graph

    def step(self, action_idx):
        edge_index = self.graph['operation', 'processed_by', 'machine'].edge_index
        edge_attr = self.graph['operation', 'processed_by', 'machine'].edge_attr

        op_id = edge_index[0, action_idx].item()
        m_id = edge_index[1, action_idx].item()
        proc_time = edge_attr[action_idx].item() # 注意：这里要是原始时间，不要归一化后的
        job_id = self.graph.op_to_job[op_id].item()
        # --- 【关键修改 1】记录动作执行前的最大完工时间 E(t) ---
        # E(t) 是当前所有机器空闲时间中的最大值
        # 注意：初始状态时 state_m_free_time 全为0，E(0)=0
        old_max_makespan = self.state_m_free_time.max()
        job_avail_time = self.jobs_next_op_time[job_id]
        machine_free_time = self.state_m_free_time[m_id]
        start_time = max(machine_free_time, job_avail_time)
        end_time = start_time + proc_time

        
        self.state_m_free_time[m_id] = end_time
        self.jobs_next_op_time[job_id] = end_time
        
        self.state_op_status[op_id] = 2
        self.finished_cnt += 1

        precedes = self.graph['operation', 'precedes', 'operation'].edge_index
        succ = precedes[1, precedes[0] == op_id]
        if succ.numel() > 0:
            self.state_op_status[succ[0].item()] = 1

        self.current_time = min(self.state_m_free_time)

        # --- 【关键修改 2】计算动作执行后的最大完工时间 E(t+1) ---
        new_max_makespan = self.state_m_free_time.max()

        # --- 【关键修改 3】实现论文中的奖励公式 ---
        # Reward = E(t) - E(t+1) 
        # 含义：如果你让最大完工时间变长了（通常会发生），奖励就是负的（惩罚）。
        # 如果你填补了空隙没有推高最大完工时间，奖励就是 0。
        reward = old_max_makespan - new_max_makespan
        
        # 归一化奖励（可选但推荐）：为了防止奖励数值过大导致训练不稳定
        # 建议除以一个时间常数，比如加工时间均值或 max_time_scale
        # reward /= 10.0  

        done = (self.finished_cnt == self.total_ops)

        self._update_features()
        return self.graph, reward, done

    def _update_features(self):


        for i in range(self.total_ops):
            s = self.state_op_status[i]
            self.graph['operation'].x[i, 0] = 1.0 if s == 2 else 0.0
            self.graph['operation'].x[i, 1] = 1.0 if s == 1 else 0.0
            # 加入平均工时特征 (归一化)
            self.graph['operation'].x[i, 2] = self.avg_op_times[i] / self.time_scale
            
        for m in range(len(self.machines_list)):
            free = self.state_m_free_time[m]
            is_work = 1.0 if free > self.current_time else 0.0
            rem = max(0, free - self.current_time)

            self.graph['machine'].x[m, 0] = is_work
            self.graph['machine'].x[m, 1] = np.log1p(rem)

    def _calc_est_makespan(self):
        # 修正估算逻辑：不仅仅是 current_time + rem，而是考虑 max(free_time)
        # 这里为了保持密集奖励的平滑性，我们还是用 current_time + rem
        # 但在 step 中通过 done 时的惩罚来校准
        est = 0
        gid = 0
        for jid in sorted(self.jobs_data):
            rem = 0
            for _ in self.jobs_data[jid]:
                if self.state_op_status[gid] != 2:
                    rem += self.avg_op_times[gid]
                gid += 1
            if self.current_time + rem > est:
                est = self.current_time + rem
        return est

    # [新增] 必须暴露给 Masking 使用
    def get_valid_actions(self):
        feasible_ops = [k for k, v in self.state_op_status.items() if v == 1]
        edge_index = self.graph['operation', 'processed_by', 'machine'].edge_index
        valid_indices = []
        for i in range(edge_index.shape[1]):
            if edge_index[0, i].item() in feasible_ops:
                valid_indices.append(i)
        return valid_indices


# ==========================================
# 5. PPO 训练逻辑
# ==========================================
def main():
    HIDDEN_DIM = 64
    ##################
    train_folder = 'Problem_TrainSet'
    json_files = glob.glob(os.path.join(train_folder, '*.json'))
    dataset = []
    for fpath in json_files:
        m, j, mt = load_data_from_json(fpath)
        dataset.append({'machines': m, 'jobs': j, 'max_time': mt, 'name': fpath})
    temp_graph = build_hetero_graph(dataset[0]['machines'], dataset[0]['jobs'], dataset[0]['max_time'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGATModel(temp_graph.metadata(), hidden_dim=HIDDEN_DIM).to(device)
    
    ###################


    
    # 学习率建议调小一点，0.0005 对 GAT 稍大，0.0002 比较稳
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    MAX_EPISODES = 5000
    UPDATE_TIMESTEP = 2000
    GAMMA = 0.99
    EPS_CLIP = 0.2
    K_EPOCHS = 4

    print(f"Start Training on {device}...")

    # Buffer
    memory_graphs = []
    memory_valid_indices = []
    memory_logprobs = []
    memory_values = []
    memory_rewards = []
    memory_actions = []
    memory_is_terminals = []

    time_step = 0
    log_makespans = []

    for ep in range(MAX_EPISODES):
        data = random.choice(dataset)
        machines, jobs, max_time = data['machines'], data['jobs'], float(data['max_time'])
        current_norm_factor = max_time
        graph_template = build_hetero_graph(machines, jobs, max_time)
        env = FJSPEnv(machines, jobs, graph_template, time_scale=max_time)
    
        graph = env.reset()
        done = False
        ep_reward = 0
        while not done:
            time_step += 1
            valid_idx = env.get_valid_actions()
            if not valid_idx: break

            # 1. 准备数据
            # 注意：这里存入 memory 的 graph 最好是没有任何梯度的副本
            # 为了安全，我们在 inference 时用 graph_dev，存 memory 时存 graph.clone()
            graph_dev = graph.clone().to(device)
            valid_idx_dev = torch.tensor(valid_idx, dtype=torch.long).to(device)

            # 2. 推理 (No Grad 以防万一，虽然最后要 backward，但这里不需要)
            # 在 pytorch 中，收集数据阶段最好不要带梯度，除非是 reparameterization
            # 但 PPO 是 on-policy，收集的数据是旧策略的，不需要梯度
            with torch.no_grad():
                probs, val = model(graph_dev, valid_idx_dev)
                dist = Categorical(probs)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor)
            
            action_idx_in_valid = action_tensor.item()
            actual_edge_idx = valid_idx[action_idx_in_valid]

            # 3. 环境步进
            next_graph, reward, done = env.step(actual_edge_idx)

            # 4. 存入 Buffer (关键：全部 detach 放入 CPU 或 GPU，确保不带计算图)
            memory_graphs.append(graph_dev) # graph_dev 本身由 graph.clone() 来，没有梯度
            memory_valid_indices.append(valid_idx_dev)
            memory_logprobs.append(log_prob)
            memory_values.append(val)
            memory_rewards.append(reward)
            memory_actions.append(action_tensor)
            memory_is_terminals.append(done)

            ep_reward += reward
            graph = next_graph

            # --- PPO Update ---
            if time_step % UPDATE_TIMESTEP == 0:
                print(f"  [Update] Updating PPO at step {time_step}...")

                # A. 计算 Monte Carlo Returns (倒序)
                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(memory_rewards), reversed(memory_is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + GAMMA * discounted_reward
                    rewards.insert(0, discounted_reward)

                # B. 转换为 Tensor 并归一化
                returns = torch.tensor(rewards, dtype=torch.float32).to(device)
                returns = (returns - returns.mean()) / (returns.std() + 1e-7)

                # C. 整理旧数据 (Stack)
                old_logprobs = torch.stack(memory_logprobs).to(device).detach()
                old_values = torch.stack(memory_values).to(device).detach().squeeze()
                old_actions = torch.stack(memory_actions).to(device).detach()

                # D. 计算 Advantages (固定值，不带梯度！)
                advantages = returns - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7) # 再次归一化 Advantage 有助于收敛

                # E. PPO 迭代更新
                # 使用 Mini-Batch 模拟 (Gradient Accumulation) 以减少显存压力
                # 如果显存够，batch_size 可以大一点；如果不够，调小
                batch_size = 64 
                dataset_size = len(memory_graphs)
                indices = np.arange(dataset_size)

                for _ in range(K_EPOCHS):
                    np.random.shuffle(indices) # 打乱数据
                    
                    for start_idx in range(0, dataset_size, batch_size):
                        end_idx = min(start_idx + batch_size, dataset_size)
                        batch_indices = indices[start_idx:end_idx]
                        
                        loss_accum = 0
                        optimizer.zero_grad()

                        # 内层循环：逐个图计算 Loss (因为没有实现 Graph Batching)
                        batch_graphs_list = []
                        batch_valid_indices = []
                        batch_action_indices = [] 
                        batch_action_offsets = [] 
                        
                        cumulative_edges = 0
                        cumulative_jobs = 0
                        cumulative_action_count = 0
                        
                        batch_op_to_job_list = []

                        for local_i, idx in enumerate(batch_indices):
                            g = memory_graphs[idx]
                            v_idx = memory_valid_indices[idx]
                            
                            batch_graphs_list.append(g)
                            
                            # 处理边索引偏移
                            num_edges = g['operation', 'processed_by', 'machine'].num_edges
                            batch_valid_indices.append(v_idx + cumulative_edges)
                            cumulative_edges += num_edges

                            # 处理 Softmax 分组索引
                            num_valid = v_idx.size(0)
                            batch_action_indices.append(torch.full((num_valid,), local_i, dtype=torch.long, device=device))
                            
                            # 处理动作 LogProb 偏移
                            batch_action_offsets.append(cumulative_action_count)
                            cumulative_action_count += num_valid

                            # 处理 Job ID 偏移 (修复 op_to_job)
                            batch_op_to_job_list.append(g.op_to_job + cumulative_jobs)
                            cumulative_jobs += g['job'].num_nodes

                        # 合并成大图
                        big_graph = Batch.from_data_list(batch_graphs_list).to(device)
                        big_valid_indices = torch.cat(batch_valid_indices)
                        big_action_batch_idx = torch.cat(batch_action_indices)
                        big_graph.op_to_job = torch.cat(batch_op_to_job_list) # 注入修正后的 op_to_job

                        # 获取 Batch 对应的旧数据
                        mb_old_logprobs = old_logprobs[batch_indices]
                        mb_old_actions = old_actions[batch_indices]
                        mb_returns = returns[batch_indices]
                        mb_adv = advantages[batch_indices]

                        # 2. 一次性前向传播 (速度提升的关键)
                        optimizer.zero_grad()
                        
                        # 传入 action_batch_index，激活模型内的 Batch Softmax
                        probs, values = model(big_graph, big_valid_indices, action_batch_index=big_action_batch_idx)
                        values = values.squeeze()

                        # 3. 提取选中动作的概率
                        action_offsets_tensor = torch.tensor(batch_action_offsets, device=device)
                        # 计算在扁平 probs 向量中的真实位置
                        flat_action_indices = action_offsets_tensor + mb_old_actions
                        
                        selected_probs = probs[flat_action_indices]
                        new_logprobs = torch.log(selected_probs + 1e-10)

                        # 4. 计算 Entropy (按图分组求和)
                        flat_entropy = - probs * torch.log(probs + 1e-10)
                        dist_entropy = scatter(flat_entropy, big_action_batch_idx, dim=0, reduce='sum').mean()

                        # 5. Loss 计算
                        ratio = torch.exp(new_logprobs - mb_old_logprobs)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * mb_adv

                        loss_actor = -torch.min(surr1, surr2).mean()
                        loss_critic = F.mse_loss(values, mb_returns)
                        
                        loss = loss_actor + 0.5 * loss_critic - 0.01 * dist_entropy

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()

                # F. 清空 Buffer
                memory_graphs = []
                memory_valid_indices = []
                memory_logprobs = []
                memory_values = []
                memory_rewards = []
                memory_actions = []
                memory_is_terminals = []

        # Logging
        final_makespan_norm = max(env.state_m_free_time)
        real_makespan = final_makespan_norm * current_norm_factor # 还原真实时间
        log_makespans.append(real_makespan)

        if (ep + 1) % 10 == 0:
            avg_makespan = sum(log_makespans[-10:]) / 10.0
            print(f"Episode {ep + 1}: Makespan = {real_makespan:.2f} (Avg: {avg_makespan:.2f})")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # 可选：如果想以后接着训练
        'metadata': temp_graph.metadata(),              # 关键：用于初始化模型结构
        'hidden_dim': HIDDEN_DIM                                # 关键：模型超参数
    }
    
    save_path = "Comparison_RL_checkpoint/fjsp_ppo_checkpoint.pth"
    torch.save(checkpoint, save_path)
    print(f"Model and metadata saved to {save_path}")
    print("Training Finished.")

if __name__ == "__main__":
    main()
