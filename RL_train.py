import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

from Industrial_Pipeline_Functions import (
    LightweightIndustrialDiffusion,
    load_ipps_problem_from_json,
    get_ipps_problem_data,
    validate_constraints,
)

from Evaluate import (
    load_problem_definitions,
    simulate_complete_scheduling,
    graph_to_simulation_input
)

PROBLEM_FILE = "TestSet/1.json"
RUN_NAME = "rl_finetuned"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LR = 1e-3   #learning rate
EPOCHS = 1000
BATCH_SIZE = 16
T_STEPS = 8
ENTROPY_START = 0.005
ENTROPY_END = 0.0001
DECAY_STEPS = 500
T_SCALER = 0.01

all_workpieces_objs, machine_power_data = load_problem_definitions(PROBLEM_FILE)

raw_wp_dicts, raw_machines = load_ipps_problem_from_json(PROBLEM_FILE)
ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, DEVICE)

model = LightweightIndustrialDiffusion(T=T_STEPS, hidden_dim=128, num_layers=6, nhead=4, dropout=0.1,device=DEVICE).to(DEVICE)

# model.load_state_dict(torch.load("ablation_runs_11_19_for_RL/baseline/model.pth"))

optimizer = optim.Adam(model.parameters(), lr=LR)
global_best_makespan = float('inf')
moving_avg_makespan = 0

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    progress = min(1.0, epoch / DECAY_STEPS)
    current_entropy_coef = ENTROPY_START - (ENTROPY_START - ENTROPY_END) * progress

    batch_loss = 0
    batch_makespans = []
    batch_rewards = []
    batch_log_probs = []
    batch_entropies = []
    node_labels = ipps_canvas.x.argmax(dim=1)


    for b in range(BATCH_SIZE):

        generated_edges, log_prob, entropy, priorities = model.reverse_diffusion_with_logprob(ipps_canvas, DEVICE, time_guidance_scale=T_SCALER)
        edges_matrix = generated_edges.argmax(dim=-1).detach().cpu()
        is_structurally_valid = validate_constraints(
            edges_matrix,
            node_labels,
            DEVICE,
            exact=True,
            data=ipps_canvas
        )
        if not is_structurally_valid:
            raise ValueError("Invalid Graph")
        else:
            wp_cycles = graph_to_simulation_input(edges_matrix, ipps_canvas, all_workpieces_objs, priorities)
            _, energy_report, _ = simulate_complete_scheduling(wp_cycles, machine_power_data)
            makespan = energy_report['total']['makespan']

            if makespan <= 0: raise ValueError("Invalid Graph")


        batch_makespans.append(makespan)
        batch_log_probs.append(log_prob)
        batch_entropies.append(entropy)


    current_avg_makespan = np.mean(batch_makespans)
    
    # 如果是第一轮，初始化 Baseline
    if epoch == 0:
        moving_avg_makespan = current_avg_makespan

    # --- 【核心修改 1】计算绝对值 Advantage ---
    # 不再除以 Baseline，直接用差值，信号更强
    raw_advantages = np.array([moving_avg_makespan - ms for ms in batch_makespans])
    
    # 转换为 Tensor
    advantages = torch.tensor(raw_advantages, dtype=torch.float32).to(DEVICE)

    # --- 【核心修改 2】只向“好样本”学习 (Top 50% Filtering) ---
    # 我们只保留 Advantage > 0 的样本，或者保留前 50% 的样本
    # 这样 Loss 就不会相互抵消为 0 了
    
    # 找出前 50% 好的索引 (makespan 最小的)
    k = max(1, BATCH_SIZE // 2) # 例如保留 8 个
    topk_indices = torch.topk(advantages, k).indices # 找出 advantage 最大的 k 个索引

    # 筛选出 Top-K 的 LogProb, Advantage, Entropy
    batch_log_probs_tensor = torch.stack(batch_log_probs) # [B]
    batch_entropies_tensor = torch.stack(batch_entropies) # [B]

    selected_log_probs = batch_log_probs_tensor[topk_indices]
    selected_advantages = advantages[topk_indices]
    selected_entropies = batch_entropies_tensor[topk_indices]

    # 对选出来的 Advantage 进行归一化（可选，但建议加上防止梯度爆炸）
    if selected_advantages.std() > 1e-8:
        selected_advantages = (selected_advantages - selected_advantages.mean()) / (selected_advantages.std() + 1e-8)

    # --- 计算 Loss (仅针对 Top-K) ---
    loss_policy = 0
    loss_entropy = 0
    
    # 向量化计算，比 for 循环更快
    loss_policy = -(selected_advantages * selected_log_probs).mean()
    loss_entropy = -selected_entropies.mean()

    # 总 Loss
    loss = loss_policy + current_entropy_coef * loss_entropy

    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # 更新 Baseline
    moving_avg_makespan = 0.9 * moving_avg_makespan + 0.1 * current_avg_makespan
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss.item():.2f} | Avg Makespan: {current_avg_makespan:.1f} | Best: {min(batch_makespans):.1f} | Baseline: {moving_avg_makespan:.1f}")

    if epoch % 100 == 0:
        save_dir = Path(f"rl_checkpoints/{RUN_NAME}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"model_ep{epoch}.pth")

print("#######################Finished#######################")