import json
import random
import copy
import os
import csv
import time

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

        return self.population[0].makespan


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
    folder_path = 'Problem_TrainSet'
    output_csv = "batch_results_trainset.csv"

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

    for i, filename in enumerate(files):
        file_path = os.path.join(folder_path, filename)

        try:
            # 1. 加载数据
            machine_ids, jobs_data = load_data_from_json(file_path)

            # 2. 运行算法 (针对批量运行，稍微调小了参数以加快速度，可按需调大)
            # pop_size=100, max_gen=100
            ga = SingleObjectiveGA(jobs_data, machine_ids, pop_size=100, max_gen=100)
            best_makespan = ga.run()

            # 3. 记录结果
            results.append([filename, best_makespan])

            print(f"[{i + 1}/{len(files)}] 完成: {filename} -> Makespan: {best_makespan}")

        except Exception as e:
            print(f"[{i + 1}/{len(files)}] 出错: {filename} -> Error: {e}")
            results.append([filename, "Error"])

    # 保存到 CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Best_Makespan"])  # 表头
        writer.writerows(results)

    total_time = time.time() - start_time_total
    print("-" * 50)
    print(f"所有任务已完成。总耗时: {total_time:.2f} 秒")
    print(f"结果已保存至: {os.path.abspath(output_csv)}")