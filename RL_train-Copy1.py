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
LR = 1e-5   #learning rate
EPOCHS = 2000
BATCH_SIZE = 32
T_STEPS = 2
# ENTROPY_START = 0.005
# ENTROPY_END = 0.0001
# DECAY_STEPS = 500
ENTROPY_START = 0.1
ENTROPY_END = 0.01
DECAY_STEPS = 300
T_SCALER = 0.001

all_workpieces_objs, machine_power_data = load_problem_definitions(PROBLEM_FILE)

raw_wp_dicts, raw_machines = load_ipps_problem_from_json(PROBLEM_FILE)
ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, DEVICE)

model = LightweightIndustrialDiffusion(T=T_STEPS, hidden_dim=256, num_layers=6, nhead=4, dropout=0.1,device=DEVICE).to(DEVICE)

# model.load_state_dict(torch.load("ablation_runs_11_19_for_RL/baseline/model.pth"))

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

global_best_makespan = float('inf')
moving_avg_makespan = 0

log_dir = Path(f"rl_checkpoints/{RUN_NAME}")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "training_log_copy.txt"

with open(log_path, "w") as f:
    f.write(f"Training Log for {RUN_NAME}\n")
    f.write("==================================================\n")
    
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


    # current_avg_makespan = np.mean(batch_makespans)
    sorted_makespans = np.sort(batch_makespans)
    current_elite_avg = np.mean(sorted_makespans[:4])
    batch_mean = np.mean(batch_makespans)
    
    if epoch == 0: 
        moving_avg_makespan = batch_mean
            
    
    
    adv_local = batch_mean - np.array(batch_makespans)
    adv_global = moving_avg_makespan - np.array(batch_makespans)
    raw_advantages = 0.5 * adv_local + 0.5 * adv_global
    advantages = torch.tensor(raw_advantages, dtype=torch.float32).to(DEVICE)
    
    # raw_advantages = np.array([batch_mean - ms for ms in batch_makespans])
    # advantages = torch.tensor(raw_advantages, dtype=torch.float32).to(DEVICE)
    if advantages.std() > 1e-8:
        advantages = advantages / (advantages.std() + 1e-8)
    advantages = torch.clamp(advantages, min=-5.0, max=5.0)
    
    moving_avg_makespan = 0.9 * moving_avg_makespan + 0.1 * batch_mean
    # raw_advantages = np.array([moving_avg_makespan - ms for ms in batch_makespans])
    # advantages = torch.tensor(raw_advantages, dtype=torch.float32).to(DEVICE)

    # k = max(1, BATCH_SIZE // 2) # 例如保留 8 个
    k = max(1, BATCH_SIZE)
    topk_indices = torch.topk(advantages, k).indices # 找出 advantage 最大的 k 个索引

    batch_log_probs_tensor = torch.stack(batch_log_probs) # [B]
    batch_entropies_tensor = torch.stack(batch_entropies) # [B]

    selected_log_probs = batch_log_probs_tensor[topk_indices]
    selected_advantages = advantages[topk_indices]
    selected_entropies = batch_entropies_tensor[topk_indices]
    scale_factor = 100.0
    selected_log_probs = selected_log_probs / scale_factor
    
    # if selected_advantages.std() > 1e-8:
    #     selected_advantages = (selected_advantages) / (selected_advantages.std() + 1e-8)
    
    high_performance_mask = selected_advantages > 1.0
    selected_advantages[high_performance_mask] *= 1.5
    loss_policy = -(selected_advantages * selected_log_probs).mean()
    loss_entropy = -selected_entropies.mean()
    print(f'loss_policy: {loss_policy}')
    print(f'loss_entropy: {current_entropy_coef * loss_entropy}')
    # 总 Loss
    loss = loss_policy + current_entropy_coef * loss_entropy

    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 1 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (f"Epoch {epoch} | "
                   f"Loss: {loss.item():.2f} | "
                   f"Avg Makespan: {current_elite_avg:.1f} | "
                   f"Best: {min(batch_makespans):.1f} | "
                   f"Baseline: {moving_avg_makespan:.1f} | "
                   f"LR: {current_lr:.6e}")
        print(log_msg)
        
        with open(log_path, "a") as f:
            f.write(log_msg + "\n") 

    # if epoch % 100 == 0:
    #     save_dir = Path(f"rl_checkpoints/{RUN_NAME}")
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     torch.save(model.state_dict(), save_dir / f"model_ep{epoch}.pth")

print("#######################Finished#######################")