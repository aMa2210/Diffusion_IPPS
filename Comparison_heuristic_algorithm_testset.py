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
from Evaluate import simulate_complete_scheduling

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

    # def calculate_fitness(self, ind):
    #     machine_timeline = {m: [] for m in self.machine_ids}
    #     job_next_available = {j: 0 for j in self.jobs_data.keys()}
    #     job_op_counter = {j: 0 for j in self.jobs_data.keys()}

    #     for job_id in ind.process_gene:
    #         op_idx = job_op_counter[job_id]
    #         op_data = self.jobs_data[job_id][op_idx]

    #         # 计算机器基因索引偏移
    #         gene_offset = 0
    #         for j in sorted(self.jobs_data.keys()):
    #             if j == job_id: break
    #             gene_offset += len(self.jobs_data[j])
    #         machine_gene_idx = gene_offset + op_idx

    #         possible_machines = list(op_data['machines'].keys())
    #         choice_idx = ind.machine_gene[machine_gene_idx] % len(possible_machines)
    #         machine_id = possible_machines[choice_idx]
    #         proc_time = op_data['machines'][machine_id]

    #         # 计算开始时间
    #         start_time_job = job_next_available[job_id]
    #         m_log = machine_timeline[machine_id]
    #         start_time_machine = m_log[-1][1] if m_log else 0

    #         real_start = max(start_time_job, start_time_machine)
    #         real_end = real_start + proc_time

    #         machine_timeline[machine_id].append((real_start, real_end))
    #         job_next_available[job_id] = real_end
    #         job_op_counter[job_id] += 1

    #     ind.makespan = max(job_next_available.values())
    def calculate_fitness(self, ind):
        """
        使用统一的 simulate_complete_scheduling 函数进行评估
        """
        # 1. 转换格式
        wp_cycles = decode_ga_to_cycles(ind, self.jobs_data)
        
        # 2. 构造 machine_power_data 
        # (GA 可能只有 machine_ids，我们需要构造一个假的 power data 传给仿真器)
        # 仿真器需要 keys 来初始化队列，power values 用于算能耗(虽然GA可能只看Makespan)
        fake_power_data = {m: {'processing': 100, 'no_load': 10} for m in self.machine_ids}
        
        # 3. 调用复杂评估器
        # time_uncertainty=0.0: 训练/搜索时通常使用确定性环境
        # seed=None: 确保不重置随机数生成器 (虽然 time_uncertainty=0 时 seed 不重要)
        try:
            completion_times, energy_report, _ = simulate_complete_scheduling(
                workpiece_cycles=wp_cycles, 
                machine_power_data=fake_power_data, 
                time_uncertainty=0.0, 
                seed=None
            )
            
            # 4. 获取 Makespan
            if completion_times:
                ind.makespan = max(completion_times.values())
            else:
                print('error code 67654')
                ind.makespan = float('inf') # 防御性编程
                
        except RuntimeError as e:
            # simulate_complete_scheduling 可能会抛出 "Unresolvable Deadlock"
            # 如果 GA 生成了无法修复的死锁结构，我们给它一个极差的适应度
            # print(f"GA Deadlock caught: {e}")
            print('error code9132')
            ind.makespan = float('inf')

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
        history = []
        start_time = time.time()
        
        self.initialize_population()

        self.population.sort(key=lambda x: x.makespan)
        current_best = self.population[0].makespan
        history.append((0, time.time() - start_time, current_best))
        
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
            elapsed_time = time.time() - start_time
            history.append((gen + 1, elapsed_time, current_best))
            
        # return self.population[0].makespan
        return self.population[0], history


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


def decode_ga_to_cycles(ind, jobs_data):
    """
    将 GA 个体解码为 simulate_complete_scheduling 所需的 workpiece_cycles 格式。
    
    关键逻辑：
    利用 process_gene 的顺序来生成 Implicit Priority。
    排在基因序列前面的工序，赋予更高的 Priority 值。
    """
    # 1. 初始化容器
    # num_jobs = len(jobs_data)
    # job_cycles = {job_id: {'machines': [], 'times': [], 'priorities': []}}
    # 使用字典暂存，最后转列表
    # 假设 job_id 是 1-based (1, 2, 3...)
    cycles_dict = {
        jid: {
            'machines': [0] * len(ops), # 预分配长度
            'times': [0.0] * len(ops),
            'priorities': [0.0] * len(ops)
        } 
        for jid, ops in jobs_data.items()
    }
    
    # 辅助计数器：记录我们在处理该工件的第几个工序
    job_op_counter = {jid: 0 for jid in jobs_data.keys()}
    
    # 2. 遍历 process_gene (决定优先级)
    total_genes = len(ind.process_gene)
    
    for rank, job_id in enumerate(ind.process_gene):
        # rank: 当前基因在序列中的位置 (0, 1, 2...)
        # job_id: 当前工件ID
        
        op_idx = job_op_counter[job_id]
        op_data = jobs_data[job_id][op_idx]
        
        # --- A. 解码机器选择 ---
        # 计算 machine_gene 的全局索引
        gene_offset = 0
        for j in sorted(jobs_data.keys()):
            if j == job_id: break
            gene_offset += len(jobs_data[j])
        machine_gene_idx = gene_offset + op_idx
        
        possible_machines = list(op_data['machines'].keys())
        choice_idx = ind.machine_gene[machine_gene_idx] % len(possible_machines)
        machine_id = possible_machines[choice_idx]
        proc_time = op_data['machines'][machine_id]
        
        # --- B. 计算优先级 ---
        # 越早出现的基因，优先级越高
        # Priority = 1.0 - (rank / total_genes)
        priority = 1.0 - (rank / total_genes)
        
        # --- C. 填入数据 ---
        cycles_dict[job_id]['machines'][op_idx] = machine_id
        cycles_dict[job_id]['times'][op_idx] = proc_time
        cycles_dict[job_id]['priorities'][op_idx] = priority
        
        job_op_counter[job_id] += 1
        
    # 3. 转换为列表格式
    workpiece_cycles = []
    # 确保按 Job ID 顺序添加 (Workpiece1, Workpiece2...)
    for jid in sorted(jobs_data.keys()):
        wp_name = f"Workpiece{jid}"
        data = cycles_dict[jid]
        workpiece_cycles.append((
            wp_name,
            data['machines'],
            data['times'],
            data['priorities']
        ))
        
    return workpiece_cycles
# ==========================================
# 3. 批量处理主程序
# ==========================================

if __name__ == "__main__":

    # folder_path = os.path.join("TestSet", "Generalization_Temp")
    
    folder_path = 'Problem_TestSet_GA_large'
    output_csv = "result_benchmark_testset_large_GA_summary.csv"
    history_dir = "GA_Benchmark_testset_large_History"
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 找不到文件夹路径: {folder_path}")
        exit()

    # 获取所有json文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    # 按照文件名排序，防止乱序
    files.sort()

    print(f"检测到 {len(files)} 个任务文件，准备开始计算...")
    print("-" * 50)

    summary_results = []
    POP_SIZE = 20
    MAX_GEN = 500
    total_makespan = 0
    total_time = 0
    
    start_time_total = time.time()
    
    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)

        try:
            # 1. 加载数据
            machine_ids, jobs_data = load_data_from_json(file_path)

            # 2. 运行算法 (针对批量运行，稍微调小了参数以加快速度，可按需调大)
            # pop_size=100, max_gen=100
            ga = SingleObjectiveGA(jobs_data, machine_ids, pop_size=POP_SIZE, max_gen=MAX_GEN)
            best_ind, history = ga.run()
            
            makespan = best_ind.makespan
            final_time = history[-1][1]
            summary_results.append([filename, int(makespan), -1, f"{final_time:.4f}", "True"])
            
            total_makespan += makespan
            total_time += final_time
            
            history_csv_name = os.path.join(history_dir, f"{os.path.splitext(filename)[0]}_history.csv")
            with open(history_csv_name, 'w', newline='', encoding='utf-8') as hf:
                h_writer = csv.writer(hf)
                h_writer.writerow(['Generation', 'Time(s)', 'Best_Makespan'])
                h_writer.writerows(history)
                
            # results.append([filename, best_makespan])
            # print(f"[{i + 1}/{len(files)}] 完成: {filename} -> Makespan: {best_makespan}")
            print(f"[{i + 1}/{len(files)}] {filename} -> MK: {int(makespan)} | Time: {final_time:.2f}s | History Saved")
        except Exception as e:
            print(f"[{i + 1}/{len(files)}] 出错: {filename} -> Error: {e}")
            summary_results.append([filename, "Error"])
    print("-" * 50)
    print(f"📊 Benchmark Finished!")
    if len(summary_results) > 0:
        print(f"Average Makespan: {total_makespan / len(summary_results):.2f}")
        print(f"Average Time: {total_time / len(summary_results):.4f} s")
        
    # 保存到 CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Makespan', 'Energy', 'Inference_Time(s)', 'Valid'])
        writer.writerows(summary_results)
