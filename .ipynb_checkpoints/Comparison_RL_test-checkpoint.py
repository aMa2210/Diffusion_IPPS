import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.utils import softmax as geo_softmax
from torch_geometric.nn import global_mean_pool
import os
import glob
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_gantt_chart(schedule, filename, makespan, save_dir="Gantt_Charts_RL"):
    """
    绘制 Gantt 图 (RL版本 - 风格已对齐)
    schedule: list of dict, 每个元素包含 {job, op, machine, start, end}
    """
    if not schedule:
        print(f"⚠️ No schedule data for {filename}")
        return

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fig, ax = plt.subplots(figsize=(14, 8)) # 尺寸调整为 14, 8 与 Diffusion 版本一致
    
    # 1. 设置颜色：为每个工件(Job)分配不同的颜色
    # RL 环境中 job_id 通常是 0, 1, 2... 已经是整数，直接排序即可
    unique_jobs = sorted(list(set(item['job'] for item in schedule)))
    
    # 使用 tab20 颜色映射 (与 Diffusion 版本一致)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_jobs)))
    job_color_map = {job: color for job, color in zip(unique_jobs, colors)}

    # 2. 绘制条形图
    for item in schedule:
        m_id = item['machine'] # RL环境内部通常是 0-based (0, 1, 2, 3, 4)
        job_id = item['job']   # RL环境内部通常是 0-based
        start = item['start']
        duration = item['end'] - item['start']
        
        # 绘制矩形 (barh)
        # 注意：这里直接用 m_id 画在 y 轴上
        ax.barh(y=m_id, width=duration, left=start, 
                height=0.6, align='center', 
                color=job_color_map[job_id], edgecolor='black', alpha=0.9)
        
        # 在条形中间添加文本
        # 【修改点】：显示为 J{job_id + 1} 以匹配 Workpiece1 -> J1 的风格
        # 如果你的 RL job_id 确实是从 0 开始的，这里 +1 就是 J1, J2...
        label_text = f"J{job_id}"
        
        ax.text(start + duration/2, m_id, label_text, 
                ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # 3. 设置坐标轴
    # Y轴：显示机器编号
    machines = sorted(list(set(item['machine'] for item in schedule)))
    ax.set_yticks(machines)
    
    # 【关键点】：RL环境通常将机器ID映射为 0~N-1
    # 所以为了显示 M-1, M-2... 这里需要 m+1
    ax.set_yticklabels([f"M-{m+1}" for m in machines])
    
    ax.set_ylabel("Machines")
    ax.set_xlabel("Time")
    ax.set_title(f"Gantt Chart - {filename} (Makespan: {makespan})")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # 4. 保存
    plt.tight_layout()
    
    # 构造文件名
    save_name = f"Gantt_{filename.replace('.json', '')}_MK{makespan}.png"
    save_path = os.path.join(save_dir, save_name)
    
    plt.savefig(save_path, dpi=150)
    print(f"   📊 Chart saved to: {save_path}")
    
    plt.close(fig) # 关闭画布防止内存溢出
    
# ==========================================
# 1. 必须包含原有的类定义以加载模型结构
# ==========================================

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

def build_hetero_graph(machines_list, jobs_data, norm_factor):
    graph = HeteroData()
    num_machines = len(machines_list)
    num_jobs = len(jobs_data)
    total_ops = sum(len(ops) for ops in jobs_data.values())

    graph['machine'].x = torch.zeros(num_machines, 6)
    graph['job'].x = torch.zeros(num_jobs, 3)
    graph['operation'].x = torch.zeros(total_ops, 9)
    graph['combination'].x = torch.zeros(num_jobs, 2)
    op_to_job = torch.zeros(total_ops, dtype=torch.long)

    edge_index_o_o, edge_index_o_m, edge_attr_o_m = [], [], []
    edge_index_o_c, edge_index_c_j = [], []

    op_global_id = 0
    sorted_job_ids = sorted(jobs_data.keys())

    for j_idx, job_id in enumerate(sorted_job_ids):
        ops_list = jobs_data[job_id]
        c_idx = j_idx
        edge_index_c_j.append([c_idx, j_idx])
        current_job_op_ids = []

        for op_data in ops_list:
            curr_op_id = op_global_id
            current_job_op_ids.append(curr_op_id)
            op_to_job[curr_op_id] = j_idx
            edge_index_o_c.append([curr_op_id, c_idx])

            for m_id, proc_time in op_data['machines'].items():
                m_idx = m_id - 1
                edge_index_o_m.append([curr_op_id, m_idx])
                edge_attr_o_m.append([proc_time / norm_factor])
            op_global_id += 1

        for i in range(len(current_job_op_ids) - 1):
            edge_index_o_o.append([current_job_op_ids[i], current_job_op_ids[i + 1]])

    graph['operation', 'precedes', 'operation'].edge_index = torch.tensor(edge_index_o_o, dtype=torch.long).t().contiguous()
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
    graph.op_to_job = op_to_job
    return graph

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
                if rel == 'processed_by' or rel == 'processes':
                    conv_dict[edge_type] = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, add_self_loops=False, edge_dim=1)
                else:
                    conv_dict[edge_type] = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, add_self_loops=False, edge_dim=None)
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        self.actor_mlp = nn.Sequential(
            nn.Linear(6 * hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.critic_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, graph, valid_action_indices, action_batch_index=None):
        x_dict = {key: self.feature_proj[key](graph[key].x) for key in graph.node_types}
        edge_index_dict = graph.edge_index_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(v) for k, v in x_dict.items()}

        h_ops, h_mchs, h_jobs = x_dict['operation'], x_dict['machine'], x_dict['job']
        
        # Inference mode (single graph)
        if hasattr(graph['operation'], 'batch') and graph['operation'].batch is not None:
            batch_ops = graph['operation'].batch
            batch_mchs = graph['machine'].batch
            batch_jobs = graph['job'].batch
            h_ops_pooled = global_mean_pool(h_ops, batch_ops)
            h_mchs_pooled = global_mean_pool(h_mchs, batch_mchs)
            h_jobs_pooled = global_mean_pool(h_jobs, batch_jobs)
            h_t = torch.cat([h_ops_pooled, h_mchs_pooled, h_jobs_pooled], dim=1)
        else:
            h_t = torch.cat([torch.mean(h_ops, 0, keepdim=True), 
                             torch.mean(h_mchs, 0, keepdim=True),
                             torch.mean(h_jobs, 0, keepdim=True)], dim=1)

        value = self.critic_mlp(h_t)
        if len(valid_action_indices) == 0:
            return None, value
            
        om_edge_index = graph['operation', 'processed_by', 'machine'].edge_index
        valid_edges = om_edge_index[:, valid_action_indices]
        valid_op_ids, valid_m_ids = valid_edges[0], valid_edges[1]
        valid_job_ids = graph.op_to_job[valid_op_ids]

        if action_batch_index is not None:
             h_t_expanded = h_t[action_batch_index] 
        else:
            h_t_expanded = h_t.expand(len(valid_op_ids), -1)

        pair_embeds = torch.cat([
            h_ops[valid_op_ids], h_mchs[valid_m_ids], h_jobs[valid_job_ids], h_t_expanded
        ], dim=1)

        scores = self.actor_mlp(pair_embeds).squeeze(-1)
        
        if action_batch_index is not None:
            probs = geo_softmax(scores, action_batch_index)
        else:
            probs = F.softmax(scores, dim=0)
        return probs, value

class FJSPEnv:
    def __init__(self, machines_list, jobs_data, graph_template, time_scale):
        self.jobs_data = jobs_data
        self.machines_list = machines_list
        self.graph_template = graph_template
        self.num_jobs = len(jobs_data)
        self.avg_op_times = self._precompute_avg_times()
        self.time_scale = time_scale
        self.schedule = []
        
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
        self.jobs_next_op_time = {jid: 0.0 for jid in self.jobs_data.keys()}
        curr = 0
        for jid in sorted(self.jobs_data):
            self.state_op_status[curr] = 1
            curr += len(self.jobs_data[jid])
        self.graph = self.graph_template.clone()
        self._update_features()
        self.schedule = []
        
        return self.graph

    def step(self, action_idx):
        edge_index = self.graph['operation', 'processed_by', 'machine'].edge_index
        edge_attr = self.graph['operation', 'processed_by', 'machine'].edge_attr
        op_id = edge_index[0, action_idx].item()
        m_id = edge_index[1, action_idx].item()
        proc_time = edge_attr[action_idx].item() # normalized time
        # Convert back to real time for calculation if edge_attr was normalized
        # BUT: In build_graph, edge_attr is normalized. 
        # The env logic in your training code used raw edge_attr which is proc_time/norm_factor.
        # However, to get REAL makespan, we need to be careful. 
        # Let's assume the Env step logic uses the normalized time for state tracking,
        # but we will rely on self.state_m_free_time * self.time_scale for the final output.
        
        real_proc_time = proc_time * self.time_scale

        job_id = self.graph.op_to_job[op_id].item()
        job_avail_time = self.jobs_next_op_time[job_id]
        machine_free_time = self.state_m_free_time[m_id] # This is in real time now? 
        # WAIT: In the training code, env.state_m_free_time was storing values.
        # If edge_attr is normalized, state_m_free_time accumulates normalized values.
        
        start_time = max(machine_free_time, job_avail_time)
        end_time = start_time + proc_time # adding normalized time

        self.schedule.append({
            'job': job_id,
            'op': op_id, # 全局 op id
            'machine': m_id,
            'start': start_time * self.time_scale,
            'end': end_time * self.time_scale
        })
        
        self.state_m_free_time[m_id] = end_time
        self.jobs_next_op_time[job_id] = end_time
        self.state_op_status[op_id] = 2
        self.finished_cnt += 1

        precedes = self.graph['operation', 'precedes', 'operation'].edge_index
        succ = precedes[1, precedes[0] == op_id]
        if succ.numel() > 0:
            self.state_op_status[succ[0].item()] = 1

        self.current_time = min(self.state_m_free_time)
        done = (self.finished_cnt == self.total_ops)
        self._update_features()
        return self.graph, 0, done

    def _update_features(self):
        for i in range(self.total_ops):
            s = self.state_op_status[i]
            self.graph['operation'].x[i, 0] = 1.0 if s == 2 else 0.0
            self.graph['operation'].x[i, 1] = 1.0 if s == 1 else 0.0
            self.graph['operation'].x[i, 2] = self.avg_op_times[i] / self.time_scale
        for m in range(len(self.machines_list)):
            free = self.state_m_free_time[m]
            is_work = 1.0 if free > self.current_time else 0.0
            rem = max(0, free - self.current_time)
            self.graph['machine'].x[m, 0] = is_work
            self.graph['machine'].x[m, 1] = np.log1p(rem)

    def get_valid_actions(self):
        feasible_ops = [k for k, v in self.state_op_status.items() if v == 1]
        edge_index = self.graph['operation', 'processed_by', 'machine'].edge_index
        valid_indices = []
        for i in range(edge_index.shape[1]):
            if edge_index[0, i].item() in feasible_ops:
                valid_indices.append(i)
        return valid_indices

# ==========================================
# 2. 推理主逻辑
# ==========================================

def evaluate_folder(model_path, data_folder, output_csv):
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载 Checkpoint
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint['metadata']
    hidden_dim = checkpoint['hidden_dim']
    
    # 2. 初始化模型
    model = HGATModel(metadata, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 切换到评估模式 (Dropout 关闭等)

    print("Model loaded successfully.")

    # 3. 搜索文件
    json_files = sorted(glob.glob(os.path.join(data_folder, '*.json')), reverse=True)
    print(f"Found {len(json_files)} files in {data_folder}")

    results = []

    # 4. 逐个文件处理
    for filepath in json_files:
        filename = os.path.basename(filepath)
        try:
            # 加载数据
            machines, jobs, max_time = load_data_from_json(filepath)
            
            # 构建环境
            graph_template = build_hetero_graph(machines, jobs, max_time)
            env = FJSPEnv(machines, jobs, graph_template, time_scale=max_time)
            
            graph = env.reset()
            done = False
            
            # 仿真 Loop
            while not done:
                valid_idx = env.get_valid_actions()
                if not valid_idx:
                    break
                
                # Tensor 准备
                graph_dev = graph.clone().to(device)
                valid_idx_dev = torch.tensor(valid_idx, dtype=torch.long).to(device)
                
                with torch.no_grad():
                    # 只有 1 个图，不需要 action_batch_index
                    probs, _ = model(graph_dev, valid_idx_dev)
                    
                    # 【核心策略】 测试时使用 Greedy (ArgMax)，而不是采样
                    # 这样可以保证结果是确定性的，通常也是所谓"Best"的表现
                    action_idx_in_valid = torch.argmax(probs).item()
                
                actual_edge_idx = valid_idx[action_idx_in_valid]
                graph, _, done = env.step(actual_edge_idx)

            # 计算最终 Makespan (还原归一化)
            final_makespan_norm = max(env.state_m_free_time)
            real_makespan = final_makespan_norm * max_time
            int_makespan = int(round(real_makespan))
            
            # 记录结果 (保留整数还是小数看需求，这里保留整数)
            results.append({
                "Filename": filename,
                "Best_Makespan": int(round(real_makespan))
            })
            
            print(f"Processed {filename}: {int(round(real_makespan))}")
            draw_gantt_chart(env.schedule, filename, int_makespan)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                "Filename": filename,
                "Best_Makespan": -1 # Error flag
            })

    # 5. 保存到 CSV
    df = pd.DataFrame(results)
    # 确保列顺序
    df = df[['Filename', 'Best_Makespan']]
    df.to_csv(output_csv, index=False) # 使用逗号分隔，类似你给的 tab 示例可以用 sep='\t'
    print(f"\nSaved results to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    # --- 用户配置 ---
    MODEL_CHECKPOINT = "Comparison_RL_checkpoint/fjsp_ppo_checkpoint.pth"
    TEST_DATA_DIR = "TestSet/Generalization_Temp"
    # TEST_DATA_DIR = "Problem_TrainSet"
    OUTPUT_FILE = "results_RL_None.csv"
    
    evaluate_folder(MODEL_CHECKPOINT, TEST_DATA_DIR, OUTPUT_FILE)