import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
# ==========================================
# 1. 数据准备 (Your Data)
# ==========================================
json_data_str = """
{
  "machines": [1, 2, 3, 4, 5, 6],
  "workpieces": [
    {
      "name": "Workpiece1",
      "optional_machines": [[1, 3, 6], [2, 3, 4, 5], [2, 3, 5, 6], [1, 6], [1, 4, 5], [3, 4, 5, 6], [1, 2, 4]],
      "processing_time": [[6, 8, 7], [5, 7, 6, 9], [8, 7, 6, 6], [8, 9], [7, 5, 8], [9, 9, 7, 8], [5, 7, 6]]
    },
    {
      "name": "Workpiece2",
      "optional_machines": [[1, 3], [2, 4, 6], [1, 5, 6], [3, 4, 5, 6], [1, 2, 3, 4, 5]],
      "processing_time": [[7, 6], [5, 8, 7], [9, 8, 8], [7, 7, 7, 6], [8, 8, 9, 9, 9]]
    },
    {
      "name": "Workpiece3",
      "optional_machines": [[1, 2, 3, 4, 5], [1, 4, 6], [3, 5], [2, 6], [3, 4, 5], [1, 3, 4, 6]],
      "processing_time": [[3, 4, 5, 5, 4], [5, 4, 4], [4, 3], [5, 3], [6, 8, 7], [6, 7, 7, 5]]
    },
    {
      "name": "Workpiece4",
      "optional_machines": [[5, 6], [4, 5], [1, 2, 3, 4], [2, 3, 4, 5], [1, 2, 6], [2, 5], [1, 2, 3, 4, 6], [2, 5, 6], [2, 3, 4, 6]],
      "processing_time": [[7, 6], [6, 5], [5, 6, 4, 5], [5, 6, 4, 4], [7, 7, 7], [8, 8], [12, 10, 9, 11, 9], [3, 3, 4], [4, 5, 7, 4]]
    },
    {
      "name": "Workpiece5",
      "optional_machines": [[2, 4], [1, 3, 5], [1, 2, 5], [3, 4, 5, 6], [2, 3], [3, 5, 6], [1, 2, 4, 6], [1, 2]],
      "processing_time": [[8, 7], [6, 8, 9], [7, 7, 8], [5, 8, 6, 5], [6, 7], [2, 5, 3], [9, 9, 4, 4], [5, 5]]
    }
  ]
}
"""


def load_data_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    machines_list = data.get("machines", [])
    jobs_data = {}
    for idx, wp in enumerate(data["workpieces"]):
        job_id = idx + 1
        ops_list = []
        opt_machines = wp["optional_machines"]
        proc_times = wp["processing_time"]
        for m_list, t_list in zip(opt_machines, proc_times):
            machine_dict = {}
            for m_id, t_val in zip(m_list, t_list):
                machine_dict[m_id] = t_val
            ops_list.append({'machines': machine_dict})
        jobs_data[job_id] = ops_list
    return machines_list, jobs_data


# ==========================================
# 2. 图构建 (Graph Builder)
# ==========================================
def build_hetero_graph(machines_list, jobs_data):
    graph = HeteroData()
    num_machines = len(machines_list)
    num_jobs = len(jobs_data)
    total_ops = sum(len(ops) for ops in jobs_data.values())

    # --- Node Feature Init (Placeholders) ---
    graph['machine'].x = torch.zeros(num_machines, 6)  # [cite: 535-541]
    graph['job'].x = torch.zeros(num_jobs, 3)  # [cite: 543-546]
    graph['operation'].x = torch.zeros(total_ops, 9)  # [cite: 529-534]
    graph['combination'].x = torch.zeros(num_jobs, 2)

    # 辅助：记录 op 属于哪个 job，用于模型中快速索引
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
                edge_attr_o_m.append([proc_time])

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

    def forward(self, graph, valid_action_indices):
        x_dict = {key: self.feature_proj[key](graph[key].x) for key in graph.node_types}
        edge_index_dict = graph.edge_index_dict

        # 【关键修改】构建边特征字典
        # 只有 O-M 和 M-O 边有时间数据，其他边没有
        edge_attr_dict = {}
        for edge_type in graph.edge_types:
            src, rel, dst = edge_type
            if rel == 'processed_by' or rel == 'processes':
                # 归一化时间特征！这非常重要，防止梯度爆炸
                # 假设最大加工时间大约是 20，我们除以 20 或者 100 把它缩放到 0-1 之间
                # 这里简单除以 10.0，实际可以用数据集中最大值
                edge_attr_dict[edge_type] = graph[edge_type].edge_attr / 10.0
            else:
                edge_attr_dict[edge_type] = None

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(v) for k, v in x_dict.items()}

        h_ops, h_mchs, h_jobs = x_dict['operation'], x_dict['machine'], x_dict['job']

        # Global State h_t [cite: 283]
        h_t = torch.cat([torch.mean(h_ops, 0, keepdim=True), torch.mean(h_mchs, 0, keepdim=True),
                         torch.mean(h_jobs, 0, keepdim=True)], dim=1)

        # Critic Value
        value = self.critic_mlp(h_t)

        # Actor (Masking)
        if len(valid_action_indices) == 0:
            return None, value  # No actions available

        om_edge_index = graph['operation', 'processed_by', 'machine'].edge_index
        valid_edges = om_edge_index[:, valid_action_indices]
        valid_op_ids, valid_m_ids = valid_edges[0], valid_edges[1]
        valid_job_ids = graph.op_to_job[valid_op_ids]

        # Construct Pair Embeddings
        pair_embeds = torch.cat([
            h_ops[valid_op_ids], h_mchs[valid_m_ids], h_jobs[valid_job_ids],
            h_t.expand(len(valid_op_ids), -1)
        ], dim=1)

        scores = self.actor_mlp(pair_embeds).squeeze(-1)
        return F.softmax(scores, dim=0), value


# ==========================================
# 4. 环境逻辑 (Environment)
# ==========================================
class FJSPEnv:
    def __init__(self, machines_list, jobs_data, graph_template):
        self.jobs_data = jobs_data
        self.machines_list = machines_list
        self.graph_template = graph_template
        self.num_jobs = len(jobs_data)
        self.avg_op_times = self._precompute_avg_times()
        # 计算一个粗略的最大时间用于归一化 (例如所有工序时间总和)
        self.max_time_scale = sum(self.avg_op_times.values())

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
        proc_time = edge_attr[action_idx].item()

        # Update time
        start_time = max(self.current_time, self.state_m_free_time[m_id])
        end_time = start_time + proc_time
        self.state_m_free_time[m_id] = end_time

        self.state_op_status[op_id] = 2
        self.finished_cnt += 1

        precedes = self.graph['operation', 'precedes', 'operation'].edge_index
        succ = precedes[1, precedes[0] == op_id]
        if succ.numel() > 0:
            self.state_op_status[succ[0].item()] = 1

        self.current_time = min(self.state_m_free_time)

        # --- 奖励修正 ---
        cur_est = self._calc_est_makespan()

        # 1. 密集奖励：基于估计时间的减少
        dense_reward = self.last_est_makespan - cur_est
        self.last_est_makespan = cur_est

        # 2. 稀疏惩罚：如果所有工序都做完了，额外给一个基于实际 Makespan 的惩罚
        # 这迫使模型关注 Max Free Time 而不是 Min Free Time
        done = (self.finished_cnt == self.total_ops)

        final_makespan_penalty = 0
        if done:
            real_makespan = max(self.state_m_free_time)
            # 这是一个简单的技巧：Makespan 越小，惩罚越小（或奖励越大）
            # 我们给一个基于 Baseline 的奖励，比如期望它小于 100
            final_makespan_penalty = (100.0 - real_makespan) / 10.0

        # 总奖励
        reward = dense_reward + (1.0 if done else 0.0) * final_makespan_penalty

        self._update_features()
        return self.graph, reward, done

    def _update_features(self):
        # 归一化因子
        time_norm = 100.0

        for i in range(self.total_ops):
            s = self.state_op_status[i]
            self.graph['operation'].x[i, 0] = 1.0 if s == 2 else 0.0
            self.graph['operation'].x[i, 1] = 1.0 if s == 1 else 0.0
            # 加入平均工时特征 (归一化)
            self.graph['operation'].x[i, 2] = self.avg_op_times[i] / 20.0

        for m in range(len(self.machines_list)):
            free = self.state_m_free_time[m]
            is_work = 1.0 if free > self.current_time else 0.0
            rem = max(0, free - self.current_time)

            self.graph['machine'].x[m, 0] = is_work
            # 归一化剩余时间
            self.graph['machine'].x[m, 1] = rem / time_norm

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
    # ... (Load Data 部分不变) ...
    # 这里直接使用字符串加载方便调试
    # file_path = 'Problem_TrainSet/1.json'
    file_path = 'TestSet/Generalization_Temp/gen_30_job_0.json'
    machines, jobs = load_data_from_json(file_path)

    graph_template = build_hetero_graph(machines, jobs)
    env = FJSPEnv(machines, jobs, graph_template)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HGATModel(graph_template.metadata(), hidden_dim=64).to(device)

    # 降低学习率，增加稳定性
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # 训练参数调整
    MAX_EPISODES = 3000  # 增加总回合数
    UPDATE_TIMESTEP = 2000  # 每收集 2000 个 step 更新一次 (Batch Update)
    GAMMA = 0.99
    EPS_CLIP = 0.2
    K_EPOCHS = 4  # 每次更新多学几轮

    print(f"Start Training on {device}...")

    # 存储 Batch 数据的 Buffer
    memory_graphs = []
    memory_valid_indices = []
    memory_logprobs = []
    memory_values = []
    memory_rewards = []
    memory_action_indices = []
    memory_is_terminals = []

    time_step = 0

    # 记录日志
    log_makespans = []

    for ep in range(MAX_EPISODES):
        graph = env.reset()
        done = False
        ep_reward = 0

        while not done:
            time_step += 1

            valid_idx = env.get_valid_actions()
            if not valid_idx: break

            # Prepare inputs
            graph_dev = graph.clone().to(device)
            valid_idx_dev = torch.tensor(valid_idx, dtype=torch.long).to(device)

            # Inference
            probs, val = model(graph_dev, valid_idx_dev)
            dist = Categorical(probs)
            action_tensor = dist.sample()
            action_idx_in_valid = action_tensor.item()
            actual_edge_idx = valid_idx[action_idx_in_valid]

            # Env Step
            next_graph, reward, done = env.step(actual_edge_idx)

            # Store to Buffer
            memory_graphs.append(graph_dev)
            memory_valid_indices.append(valid_idx_dev)
            memory_logprobs.append(dist.log_prob(action_tensor))
            memory_values.append(val)
            memory_rewards.append(reward)
            memory_action_indices.append(action_tensor)
            memory_is_terminals.append(done)

            ep_reward += reward
            graph = next_graph

            # --- PPO Update (Triggered by TimeStep) ---
            if time_step % UPDATE_TIMESTEP == 0:
                print(f"  [Update] Updating PPO at step {time_step}...")

                # 1. 计算 Monte Carlo Returns & Advantages
                rewards = []
                discounted_reward = 0
                # 注意：这里需要正确处理多个 Episode 的边界
                # 我们倒序遍历 Buffer
                for reward, is_terminal in zip(reversed(memory_rewards), reversed(memory_is_terminals)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + GAMMA * discounted_reward
                    rewards.insert(0, discounted_reward)

                # 归一化 Returns
                returns = torch.tensor(rewards, dtype=torch.float32).to(device)
                returns = (returns - returns.mean()) / (returns.std() + 1e-7)

                # 准备旧数据
                old_logprobs = torch.stack(memory_logprobs).detach()
                old_values = torch.stack(memory_values).detach().squeeze()
                old_actions = torch.stack(memory_action_indices).detach()

                # 2. PPO Iterations
                # 为了省显存，这里不建议一次性把 2000 个图喂进去，最好用 Mini-batch
                # 但为了代码简单，我们先整批做 (如果显存爆了请减小 UPDATE_TIMESTEP)
                for _ in range(K_EPOCHS):
                    # Re-evaluate
                    new_logprobs_list = []
                    new_values_list = []
                    entropy_list = []

                    # 这里的循环是性能瓶颈，实际工程中需优化为 DataLoader
                    for i, g in enumerate(memory_graphs):
                        p, v = model(g, memory_valid_indices[i])
                        d = Categorical(p)
                        new_logprobs_list.append(d.log_prob(old_actions[i]))
                        new_values_list.append(v.squeeze())
                        entropy_list.append(d.entropy())

                    new_logprobs_tensor = torch.stack(new_logprobs_list)
                    new_values_tensor = torch.stack(new_values_list)
                    entropy_tensor = torch.stack(entropy_list).mean()

                    ratios = torch.exp(new_logprobs_tensor - old_logprobs)
                    advantages = returns - old_values

                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

                    loss_actor = -torch.min(surr1, surr2).mean()
                    loss_critic = F.mse_loss(new_values_tensor, returns)

                    loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy_tensor

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 3. Clear Buffer
                memory_graphs = []
                memory_valid_indices = []
                memory_logprobs = []
                memory_values = []
                memory_rewards = []
                memory_action_indices = []
                memory_is_terminals = []

        # Logging
        final_makespan = max(env.state_m_free_time)
        log_makespans.append(final_makespan)

        if (ep + 1) % 10 == 0:
            avg_makespan = sum(log_makespans[-10:]) / 10.0
            print(f"Episode {ep + 1}: Makespan = {final_makespan:.2f} (Avg: {avg_makespan:.2f})")

    print("Training Finished.")


if __name__ == "__main__":
    main()