import glob
import os
import json
import random
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
# 如果你的 build_hetero_graph 函数里用了这些，也需要保留：
from torch_geometric.data import Batch 
from torch_geometric.utils import softmax as geo_softmax
# ==========================================
# 0. 必要的函数定义 (请确保这些在你的代码中)
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

class FJSPEnv:
    def __init__(self, machines_list, jobs_data, graph_template, time_scale):
        self.jobs_data = jobs_data
        self.machines_list = machines_list
        self.graph_template = graph_template
        self.num_jobs = len(jobs_data)
        self.avg_op_times = self._precompute_avg_times()
        self.time_scale = time_scale
        self.schedule = []
        self.state_m_free_time = np.zeros(len(self.machines_list)) # 初始化防止未重置报错
        
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
        proc_time = edge_attr[action_idx].item() 
        real_proc_time = proc_time * self.time_scale

        job_id = self.graph.op_to_job[op_id].item()
        job_avail_time = self.jobs_next_op_time[job_id]
        machine_free_time = self.state_m_free_time[m_id] 
        
        start_time = max(machine_free_time, job_avail_time)
        end_time = start_time + proc_time 

        self.schedule.append({
            'job': job_id,
            'op': op_id,
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
def evaluate_folder(model_path, data_folder, output_csv): # model_path 参数其实可以忽略了
    # 配置
    print(f"Running Random Baseline (No Model)...")

    # 1. 跳过模型加载
    # 不需要 checkpoint，也不需要初始化 HGATModel

    # 2. 搜索文件
    json_files = sorted(glob.glob(os.path.join(data_folder, '*.json')), reverse=True)
    print(f"Found {len(json_files)} files in {data_folder}")

    results = []

    # 3. 逐个文件处理
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
                
                # --- 【核心修改】 ---
                # 不再使用模型推理，直接随机选择一个动作
                # 这种 "纯随机" 才是标准的 Baseline
                action_idx_in_valid = random.randint(0, len(valid_idx) - 1)
                
                actual_edge_idx = valid_idx[action_idx_in_valid]
                graph, _, done = env.step(actual_edge_idx)

            # 计算最终 Makespan
            final_makespan_norm = max(env.state_m_free_time)
            real_makespan = final_makespan_norm * max_time
            
            # 记录结果
            results.append({
                "Filename": filename,
                "Best_Makespan": int(round(real_makespan))
            })
            
            print(f"Processed {filename}: {int(round(real_makespan))}")
            # draw_gantt_chart(env.schedule, filename, int(round(real_makespan))) # 需要画图取消注释
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                "Filename": filename,
                "Best_Makespan": -1
            })

    # 4. 保存到 CSV
    df = pd.DataFrame(results)
    df = df[['Filename', 'Best_Makespan']]
    df.to_csv(output_csv, index=False)
    print(f"\nSaved Random Baseline results to {output_csv}")
    print(df.head())

if __name__ == "__main__":
    # --- 用户配置 ---
    # MODEL_CHECKPOINT = "..."  # 随机模式不需要这个路径
    TEST_DATA_DIR = "TestSet/Generalization_Temp"
    OUTPUT_FILE = "results_RL_Random.csv" # 建议改个名，避免覆盖训练结果
    
    # 第一个参数传 None 即可
    evaluate_folder(None, TEST_DATA_DIR, OUTPUT_FILE)