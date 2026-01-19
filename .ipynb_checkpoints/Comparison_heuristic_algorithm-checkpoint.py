import json
import random
import copy
import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
import torch

class GA_to_Diffusion_Converter:
    def __init__(self, jobs_data, machine_ids, device='cpu'):
        self.jobs_data = jobs_data
        self.machine_ids = machine_ids
        self.device = device
        
        self.num_machines = len(machine_ids)
        self.total_ops = sum(len(ops) for ops in jobs_data.values())
        self.num_nodes = self.total_ops + self.num_machines
        
        # 建立 Machine ID -> Graph Node Index 的映射
        self.machine_id_to_node_idx = {
            m_id: i + self.total_ops 
            for i, m_id in enumerate(machine_ids)
        }
        
        # 预计算：Job ID -> 全局工序起始索引 的映射
        # 例如: Job1从0开始, Job2从5开始...
        self.job_start_indices = {}
        curr = 0
        sorted_job_ids = sorted(self.jobs_data.keys())
        for j in sorted_job_ids:
            self.job_start_indices[j] = curr
            curr += len(self.jobs_data[j])

    def convert(self, individual, completed_operations):
        """
        修改版: 
        1. Routing (边) 依然来自 individual.machine_gene (这是决策源头)
        2. Priority (值) 改为来自 completed_operations (这是仿真结果，包含真实顺序)
        """
        
        # --- 1. 转换 Edges (Routing) ---
        # (这部分逻辑不变，因为机器分配是由基因决定的)
        source_nodes = []
        target_nodes = []
        global_op_idx = 0
        gene_idx = 0
        sorted_job_ids = sorted(self.jobs_data.keys())
        
        for jid in sorted_job_ids:
            ops = self.jobs_data[jid]
            for op_data in ops:
                possible_machines = list(op_data['machines'].keys())
                choice_idx = individual.machine_gene[gene_idx] % len(possible_machines)
                selected_m_id = possible_machines[choice_idx]
                
                u = global_op_idx 
                v = self.machine_id_to_node_idx[selected_m_id]
                
                source_nodes.append(u)
                target_nodes.append(v)
                
                global_op_idx += 1
                gene_idx += 1
                
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # --- 2. 转换 Priorities (Machine-Level Sequencing) ---
        # 🔥🔥🔥 核心修改 🔥🔥🔥
        
        priorities = torch.zeros(self.total_ops)
        
        # A. 按机器分组收集工序
        # 结构: machine_queues = { m_id: [ (start_time, global_op_idx), ... ] }
        machine_queues = {m: [] for m in self.machine_ids}
        
        for op_record in completed_operations:
            # 解析记录: {'workpiece': 'Workpiece1', 'machine': 1, 'start_time': 10, 'feature': 1}
            wp_str = op_record['workpiece']
            # 从 "Workpiece1" 提取 ID 1
            # 注意：你的代码里 extract_number 逻辑可能有变，这里写稳健一点
            job_id = int(re.search(r'\d+', str(wp_str)).group())
            
            # feature 是工序在工件内的序号 (1-based)，转为 0-based
            op_internal_idx = op_record['feature'] - 1
            
            # 计算全局节点索引
            g_idx = self.job_start_indices[job_id] + op_internal_idx
            
            m_id = op_record['machine']
            start_t = op_record['start_time']
            
            machine_queues[m_id].append((start_t, g_idx))
            
        # B. 对每台机器的队列按时间排序，并赋值优先级
        for m_id, queue in machine_queues.items():
            if not queue:
                continue
                
            # 按开始时间从小到大排序
            # 如果开始时间相同，保持原序 (虽然在甘特图中不太可能完全相同)
            queue.sort(key=lambda x: x[0])
            
            # 赋值逻辑:
            # 越早开始 -> 优先级越高 (1.0)
            # 越晚开始 -> 优先级越低 (0.0)
            n = len(queue)
            for rank, (st, g_idx) in enumerate(queue):
                if n > 1:
                    # 线性插值: Rank 0 -> 1.0, Rank N-1 -> 0.0
                    val = 1.0 - (rank / (n - 1))
                else:
                    val = 1.0 # 只有一个工序，优先级拉满
                
                priorities[g_idx] = val

        return edge_index, priorities
class Individual:
    def __init__(self):
        self.process_gene = []  # 工序编码
        self.machine_gene = []  # 机器编码
        self.makespan = float('inf')  # 适应度值 (Makespan)


class SingleObjectiveGA:
    def __init__(self, jobs_data, machine_ids, pop_size=100, max_gen=100, pc=0.8, pm=0.1):
        self.jobs_data = jobs_data
        self.machine_ids = machine_ids
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.pc = pc
        self.pm = pm
        self.population = []

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = Individual()

            # 1. 生成工序编码
            p_gene = []
            for jid, ops in self.jobs_data.items():
                p_gene.extend([jid] * len(ops))
            random.shuffle(p_gene)
            ind.process_gene = p_gene

            # 2. 生成机器编码
            m_gene = []
            for jid in sorted(self.jobs_data.keys()):
                for op in self.jobs_data[jid]:
                    num_options = len(op['machines'])
                    m_gene.append(random.randint(0, num_options - 1))
            ind.machine_gene = m_gene

            self.calculate_fitness(ind)
            self.population.append(ind)

    def calculate_fitness(self, ind):
        machine_timeline = {m: [] for m in self.machine_ids}
        job_next_available = {j: 0 for j in self.jobs_data.keys()}
        job_op_counter = {j: 0 for j in self.jobs_data.keys()}

        for job_id in ind.process_gene:
            op_idx = job_op_counter[job_id]
            op_data = self.jobs_data[job_id][op_idx]

            # 计算机器基因索引偏移
            gene_offset = 0
            for j in sorted(self.jobs_data.keys()):
                if j == job_id: break
                gene_offset += len(self.jobs_data[j])
            machine_gene_idx = gene_offset + op_idx

            possible_machines = list(op_data['machines'].keys())
            choice_idx = ind.machine_gene[machine_gene_idx] % len(possible_machines)
            machine_id = possible_machines[choice_idx]
            proc_time = op_data['machines'][machine_id]

            # 计算开始时间
            start_time_job = job_next_available[job_id]
            m_log = machine_timeline[machine_id]
            start_time_machine = m_log[-1][1] if m_log else 0

            real_start = max(start_time_job, start_time_machine)
            real_end = real_start + proc_time

            machine_timeline[machine_id].append((real_start, real_end))
            job_next_available[job_id] = real_end
            job_op_counter[job_id] += 1

        ind.makespan = max(job_next_available.values())

    def selection(self):
        p1 = random.choice(self.population)
        p2 = random.choice(self.population)
        return p1 if p1.makespan < p2.makespan else p2

    def crossover(self, p1, p2):
        # POX Crossover
        all_jobs = list(self.jobs_data.keys())
        job_set1 = set(random.sample(all_jobs, random.randint(1, max(1, len(all_jobs) - 1))))

        c1_proc = [-1] * len(p1.process_gene)
        c2_proc = [-1] * len(p2.process_gene)

        for i, gene in enumerate(p1.process_gene):
            if gene in job_set1: c1_proc[i] = gene
        for i, gene in enumerate(p2.process_gene):
            if gene in job_set1: c2_proc[i] = gene

        p2_idx = 0
        for i in range(len(c1_proc)):
            if c1_proc[i] == -1:
                while p2.process_gene[p2_idx] in job_set1: p2_idx += 1
                c1_proc[i] = p2.process_gene[p2_idx]
                p2_idx += 1

        p1_idx = 0
        for i in range(len(c2_proc)):
            if c2_proc[i] == -1:
                while p1.process_gene[p1_idx] in job_set1: p1_idx += 1
                c2_proc[i] = p1.process_gene[p1_idx]
                p1_idx += 1

        # Uniform Crossover (Machines)
        c1_mach, c2_mach = [], []
        for i in range(len(p1.machine_gene)):
            if random.random() < 0.5:
                c1_mach.append(p1.machine_gene[i])
                c2_mach.append(p2.machine_gene[i])
            else:
                c1_mach.append(p2.machine_gene[i])
                c2_mach.append(p1.machine_gene[i])

        off1, off2 = Individual(), Individual()
        off1.process_gene, off1.machine_gene = c1_proc, c1_mach
        off2.process_gene, off2.machine_gene = c2_proc, c2_mach
        return off1, off2

    def mutation(self, ind):
        if random.random() < self.pm and len(ind.process_gene) > 1:
            idx1, idx2 = sorted(random.sample(range(len(ind.process_gene)), 2))
            ind.process_gene[idx1:idx2 + 1] = ind.process_gene[idx1:idx2 + 1][::-1]

        if random.random() < self.pm:
            idx = random.randint(0, len(ind.machine_gene) - 1)
            ind.machine_gene[idx] = random.randint(0, 50)

    def run(self):
        self.initialize_population()
        # 初始最优
        best_so_far = min(self.population, key=lambda x: x.makespan).makespan

        for gen in range(self.max_gen):
            offspring_pop = []
            while len(offspring_pop) < self.pop_size:
                p1 = self.selection()
                p2 = self.selection()
                if random.random() < self.pc:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                self.mutation(c1)
                self.mutation(c2)
                self.calculate_fitness(c1)
                self.calculate_fitness(c2)
                offspring_pop.append(c1)
                offspring_pop.append(c2)

            combined = self.population + offspring_pop
            combined.sort(key=lambda x: x.makespan)
            self.population = combined[:self.pop_size]

            # 可选：如果收敛太快可以提前退出，这里为了稳定性跑满
            current_best = self.population[0].makespan

        # return self.population[0].makespan
        return self.population[0]


def evaluate_stochastic_with_log(ind: Individual, jobs_data, machine_ids, uncertainty=0.1, seed=None):
    """
    修改后的评估函数：除了返回 makespan，还返回详细的工序列表供绘图使用。
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    machine_timeline = {m: [] for m in machine_ids}
    job_next_available = {j: 0 for j in jobs_data.keys()}
    job_op_counter = {j: 0 for j in jobs_data.keys()}
    
    # 🔥 新增：用于存储详细操作记录的列表
    completed_operations = []

    for job_id in ind.process_gene:
        op_idx = job_op_counter[job_id]
        op_data = jobs_data[job_id][op_idx]

        gene_offset = 0
        for j in sorted(jobs_data.keys()):
            if j == job_id: break
            gene_offset += len(jobs_data[j])
        machine_gene_idx = gene_offset + op_idx

        possible_machines = list(op_data['machines'].keys())
        choice_idx = ind.machine_gene[machine_gene_idx] % len(possible_machines)
        machine_id = possible_machines[choice_idx]
        
        base_time = op_data['machines'][machine_id]
        
        if uncertainty > 0:
            fluctuation = rng.uniform(-uncertainty, uncertainty)
            real_proc_time = base_time * (1 + fluctuation)
            real_proc_time = max(0.1, real_proc_time)
        else:
            real_proc_time = base_time

        start_time_job = job_next_available[job_id]
        m_log = machine_timeline[machine_id]
        start_time_machine = m_log[-1][1] if m_log else 0

        real_start = max(start_time_job, start_time_machine)
        real_end = real_start + real_proc_time

        machine_timeline[machine_id].append((real_start, real_end))
        job_next_available[job_id] = real_end
        job_op_counter[job_id] += 1
        
        # 🔥 记录操作详情 (格式需匹配 create_gantt_chart)
        # 注意：这里构造的 workpiece 名字要符合 "Workpiece{ID}" 的格式
        completed_operations.append({
            'workpiece': f"Workpiece{job_id}", 
            'machine': machine_id,
            'start_time': real_start,
            'processing_time': real_proc_time,
            'feature': op_idx + 1
        })

    return max(job_next_available.values()), completed_operations


def create_gantt_chart(completed_operations, title="Gantt Chart", filename=None):
    """
    绘制甘特图
    """
    if not completed_operations:
        print("⚠️ No operations to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 1. 数据解析与排序
    raw_workpieces = list(set(op['workpiece'] for op in completed_operations))
    
    def extract_number(text):
        match = re.search(r'\d+', str(text))
        return int(match.group()) if match else 0

    workpieces = sorted(raw_workpieces, key=extract_number)
    colors = plt.cm.tab20(np.linspace(0, 1, len(workpieces)))
    color_map = {wp: colors[i] for i, wp in enumerate(workpieces)}

    # 2. 绘制条形图
    for operation in completed_operations:
        m_id = operation['machine']      
        wp = operation['workpiece']      
        start = operation['start_time']
        dur = operation['processing_time']
        
        wp_num = extract_number(wp)
        label_text = f"J{wp_num-1}" # 显示为 J0, J1... 建议改为 J{wp_num} 看个人习惯
        
        ax.barh(y=m_id, width=dur, left=start, 
                height=0.6, align='center', 
                color=color_map[wp], edgecolor='black', alpha=0.9)
        
        ax.text(start + dur / 2, m_id, label_text, 
                ha='center', va='center', color='white', fontweight='bold', fontsize=8)

    # 3. 设置坐标轴
    machines = sorted(list(set(op['machine'] for op in completed_operations)))
    ax.set_yticks(machines)
    ax.set_yticklabels([f"M-{m}" for m in machines])
    
    ax.set_ylabel("Machines")
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"📊 Gantt chart saved to {filename}")
        plt.close(fig)
    else:
        plt.show()

# ==========================================
# 2. 数据加载函数
# ==========================================

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
# 3. 批量处理主程序
# ==========================================

if __name__ == "__main__":

    # folder_path = os.path.join("TestSet", "Generalization_Temp")
    
    folder_path = 'Problem_TrainSet_GA'
    dataset_save_path = "ga_expert_data.pt"
    output_csv = "results_GA_for_training_Diffusion.csv"
    
    
    gantt_dir = Path("Gantt_Charts_GA_for_training_DM")
    gantt_dir.mkdir(parents=True, exist_ok=True)
    UNCERTAINTY_LEVEL = 0  # 10% 的时间波动
    EVAL_SEED = 42
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 找不到文件夹路径: {folder_path}")
        print("请确认代码所在的目录下是否存在 TestSet/Generalization_Temp 文件夹。")
        exit()

    # 获取所有json文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    # 按照文件名排序，防止乱序
    files.sort()

    print(f"检测到 {len(files)} 个任务文件，准备开始计算...")
    print("-" * 50)

    results = []

    start_time_total = time.time()
    expert_dataset = []
    
    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)

        try:
            # 1. 加载数据
            machine_ids, jobs_data = load_data_from_json(file_path)

            # 2. 运行算法 (针对批量运行，稍微调小了参数以加快速度，可按需调大)
            # pop_size=100, max_gen=100
            ga = SingleObjectiveGA(jobs_data, machine_ids, pop_size=100, max_gen=500)
            best_ind = ga.run()
            det_makespan = best_ind.makespan
            stochastic_mk, ops_list = evaluate_stochastic_with_log(
                best_ind, 
                jobs_data, 
                machine_ids, 
                uncertainty=UNCERTAINTY_LEVEL, 
                seed=EVAL_SEED
            )
            # stochastic_mk = evaluate_stochastic(
            #     best_ind, 
            #     jobs_data, 
            #     machine_ids, 
            #     uncertainty=UNCERTAINTY_LEVEL, 
            #     seed=EVAL_SEED
            # )
            # 3. 记录结果
            base_name = os.path.splitext(filename)[0] 
            gantt_filename = gantt_dir / f"GA_{base_name}_MK{int(stochastic_mk)}.png"
            
            create_gantt_chart(
                ops_list,
                title=f"GA Solution: {base_name} (MK: {stochastic_mk:.1f})",
                filename=str(gantt_filename)
            )
            
            results.append([filename, f"{det_makespan:.2f}", f"{stochastic_mk:.2f}"])
            print(f"[{i+1}/{len(files)}] {filename} -> Det: {det_makespan:.1f} | Stoch(±{UNCERTAINTY_LEVEL*100}%): {stochastic_mk:.1f}")
            
            # results.append([filename, best_makespan])
            # print(f"[{i + 1}/{len(files)}] 完成: {filename} -> Makespan: {best_makespan}")
            converter = GA_to_Diffusion_Converter(jobs_data, machine_ids)
            edge_index, priorities = converter.convert(best_ind, ops_list)
            
            # 构造一个 Data 对象 (或者字典)
            data_sample = {
                "problem_file": filename,      # 记录对应的文件名
                "machine_ids": machine_ids,    # 记录机器ID列表防止顺序混淆
                "expert_edges": edge_index,    # [2, N_ops] 只有 Op->Machine 的边
                "expert_priorities": priorities, # [N_ops]
                "makespan": best_ind.makespan
            }
            expert_dataset.append(data_sample)
            
        except Exception as e:
            print(f"[{i + 1}/{len(files)}] 出错: {filename} -> Error: {e}")
            results.append([filename, "Error"])

    # 保存到 CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Deterministic_MK", "Stochastic_MK"])
        writer.writerows(results)

    total_time = time.time() - start_time_total
    torch.save(expert_dataset, dataset_save_path)
    print("-" * 50)
    print(f"所有任务已完成。总耗时: {total_time:.2f} 秒")
    print(f"结果已保存至: {os.path.abspath(output_csv)}")