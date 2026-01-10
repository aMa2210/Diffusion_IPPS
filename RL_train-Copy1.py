import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import os
import random
import json

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

SEED = 65
random.seed(SEED)
TRAIN_DIR = "Problem_TrainSet"
VAL_DIR = "Problem_ValidationSet"
# PROBLEM_FILE = "Problem_TrainSet/1.json"
RUN_NAME = "rl_new1219_2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LR = 2e-5   #learning rate
EPOCHS = 1000
BATCH_SIZE = 32
T_STEPS = 8
# ENTROPY_START = 0.005
# ENTROPY_END = 0.0001
# DECAY_STEPS = 500
ENTROPY_START = 0.1
ENTROPY_END = 0.001
DECAY_STEPS = 500
T_SCALER = 0.001
POS_SCALER = 2.0
VALIDATE_STEP = 1  #validate the model every {VALIDATE_STEP} steps
VALIDATE_BS = 4     #how many samples are generated when validating the model, then choose the best one
HIDDEN_DIMENSION = 128
NUM_LAYERS = 6
N_HEADS = 4
TEMPERATURE_METHOD = 'cosine'
def load_dataset(directory):
    """
    read all the json files in {directory} and generate PyG Data based on them
    """
    dataset = []
    files = glob.glob(os.path.join(directory, "*.json"))

    if not files:
        print(f"Warning: No json files found in {directory}")
        return []

    print(f"Loading {len(files)} problems from {directory}...")

    for filepath in tqdm(files):

        all_workpieces_objs, machine_power_data = load_problem_definitions(filepath)
        raw_wp_dicts, raw_machines = load_ipps_problem_from_json(filepath)
        ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, DEVICE)

        # x[:, :2] only the type of the node without considering other properties
        node_labels_single = ipps_canvas.x[:, :2].argmax(dim=1)

        problem_data = {
            "id": os.path.basename(filepath),
            "canvas": ipps_canvas,
            "wp_objs": all_workpieces_objs,
            "power_data": machine_power_data,
            "node_labels": node_labels_single
        }
        dataset.append(problem_data)

    return dataset



train_set = load_dataset(TRAIN_DIR)
val_set = load_dataset(VAL_DIR)


model = LightweightIndustrialDiffusion(T=T_STEPS, hidden_dim=HIDDEN_DIMENSION, num_layers=NUM_LAYERS, nhead=N_HEADS, dropout=0.1,device=DEVICE).to(DEVICE)

# model.load_state_dict(torch.load("ablation_runs_11_19_for_RL/baseline/model.pth"))

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# baseline for each problem
baseline_registry = {}
best_makespan_registry = {}

for prob in train_set:
    baseline_registry[prob['id']] = None
    best_makespan_registry[prob['id']] = float('inf')


log_dir = Path(f"rl_checkpoints/{RUN_NAME}")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "training_log.txt"
config = {
    "RUN_NAME": RUN_NAME,
    "SEED": SEED,
    "TRAIN_DIR": TRAIN_DIR,
    "VAL_DIR": VAL_DIR,
    "LR": LR,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "T_STEPS": T_STEPS,
    "ENTROPY_START": ENTROPY_START,
    "ENTROPY_END": ENTROPY_END,
    "DECAY_STEPS": DECAY_STEPS,
    "T_SCALER": T_SCALER,
    "Pos_SCALER": POS_SCALER,
    "VALIDATE_STEP": VALIDATE_STEP,
    "VALIDATE_BS": VALIDATE_BS,
    "HIDDEN_DIMENSION": HIDDEN_DIMENSION,
    "NUM_LAYERS": NUM_LAYERS,
    "N_HEADS": N_HEADS,
    "DEVICE": str(DEVICE),
    "Description": f"RL training with {TEMPERATURE_METHOD} temperature annealing"
}
config_path = log_dir / "config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Configuration saved to {config_path}")

with open(log_path, "w") as f:
    f.write(f"Training Log for {RUN_NAME}\n")
    f.write("==================================================\n")

PROBLEMS_PER_EPOCH = 5
#################tbd
# train_set = [train_set[0]]  # <--- 只练这一个！
# PROBLEMS_PER_EPOCH = 1
#################tbd
for epoch in range(EPOCHS):
    model.train()

    progress = min(1.0, epoch / DECAY_STEPS)
    current_entropy_coef = ENTROPY_START - (ENTROPY_START - ENTROPY_END) * progress

    random.shuffle(train_set)

    if len(train_set) > PROBLEMS_PER_EPOCH:
        sampled_problems = random.sample(train_set, PROBLEMS_PER_EPOCH)
    else:
        random.shuffle(train_set)
        sampled_problems = train_set
    
    epoch_loss_sum = 0
    epoch_makespan_sum = 0

    pbar = tqdm(sampled_problems, desc=f"Epoch {epoch}", leave=False)
    for prob in pbar:
        optimizer.zero_grad()

        prob_id = prob['id']
        single_canvas = prob['canvas']

        (e_batch_onehot,  # [B, N, N, C]
         batch_log_probs,  # [B]
         batch_entropies,  # [B]
         batch_priorities  # [B, Num_Ops]
         ) = model.reverse_diffusion_with_logprob(
            single_canvas,
            DEVICE,
            num_samples=BATCH_SIZE,
            time_guidance_scale=T_SCALER,
            position_guidance_scale=POS_SCALER,
            temperature_method=TEMPERATURE_METHOD,
        )

        batch_makespans = []
        batch_flow_times = []
        
        e_batch_indices = e_batch_onehot.argmax(dim=-1).detach().cpu()  # [B, N, N]
        batch_priorities_cpu = batch_priorities.detach().cpu()  # [B, Num_Ops]

        for i in range(BATCH_SIZE):
            edges_matrix = e_batch_indices[i]  # [N, N]
            priorities = batch_priorities_cpu[i]

            is_valid = validate_constraints(
                edges_matrix,
                prob['node_labels'],
                DEVICE,
                exact=True,
                data=single_canvas
            )

            if not is_valid:
                raise ValueError("Invalid Graph Generated")

            else:
                wp_cycles = graph_to_simulation_input(edges_matrix, single_canvas, prob['wp_objs'], priorities)
                completion_times, energy_report, _ = simulate_complete_scheduling(wp_cycles, prob['power_data'])
                makespan = energy_report['total']['makespan']
                
                ft = sum(completion_times.values())
                if makespan <= 0: raise ValueError("Invalid Graph")

            batch_makespans.append(makespan)
            batch_flow_times.append(ft)
            
        batch_makespans_np = np.array(batch_makespans)
        batch_ft_np = np.array(batch_flow_times)
        batch_mean = np.mean(batch_makespans_np)

        # Baseline 更新
        if baseline_registry[prob_id] is None:
            baseline_registry[prob_id] = batch_mean
        moving_avg = baseline_registry[prob_id]

        current_best = best_makespan_registry[prob_id]
        min_ms_in_batch = batch_makespans_np.min()
        if min_ms_in_batch < current_best:
            best_makespan_registry[prob_id] = min_ms_in_batch
            current_best = min_ms_in_batch
        
        # 计算 Advantage
        adv_local = batch_mean - batch_makespans_np
        
        adv_global = moving_avg - batch_makespans_np
        adv_ms = 0.5 * adv_local + 0.5 * adv_global
        if adv_ms.std() > 1e-8:
            adv_ms = adv_ms / (adv_ms.std() + 1e-8)
            
        batch_ft_mean = np.mean(batch_ft_np)
        adv_ft = (batch_ft_mean - batch_ft_np)
        if adv_ft.std() > 1e-8:
            adv_ft = adv_ft / (adv_ft.std() + 1e-8)
            
        raw_advantages = adv_ms + 0.1 * adv_ft
        advantages = torch.tensor(raw_advantages, dtype=torch.float32).to(DEVICE)
        is_record_breaking = batch_makespans_np <= (current_best + 1e-5)
        
        if is_record_breaking.any():

            bonus_mask = torch.tensor(is_record_breaking, device=DEVICE)
            advantages[bonus_mask] += 1.0 

        if advantages.std() > 1e-8:
             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # Normalize Advantage
        
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)

        baseline_registry[prob_id] = 0.9 * moving_avg + 0.1 * batch_mean

        k = BATCH_SIZE
        topk_indices = torch.topk(advantages, k).indices

        selected_log_probs = batch_log_probs[topk_indices]
        selected_advantages = advantages[topk_indices]
        selected_entropies = batch_entropies[topk_indices]

        selected_log_probs = selected_log_probs / 100.0

        # High Performance Bonus
        high_perf = selected_advantages > 1.0
        selected_advantages[high_perf] *= 1.5

        # Loss Calculation
        loss_policy = -(selected_advantages * selected_log_probs).mean()
        loss_entropy = -selected_entropies.mean()

        if epoch % 50 == 0:
            loss_policy_magnitude = (selected_advantages * selected_log_probs).abs().mean()
            print(f'   >>> [Debug] Net Policy Loss: {loss_policy.item():.4f} | '
                  f'Abs Magnitude: {loss_policy_magnitude.item():.4f} | ' # <--- 关注这个！
                  f'Weighted Entropy: {(current_entropy_coef * loss_entropy).item():.4f}')
        
        loss = loss_policy + current_entropy_coef * loss_entropy

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss_sum += loss.item()
        epoch_makespan_sum += batch_mean
        gap = (batch_mean - current_best) / (current_best + 1e-5)
        
        pbar.set_postfix({'L': f"{loss.item():.2f}", 'Avg': f"{batch_mean:.1f}", 'Gap': f"{gap:.1%}"})

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    avg_train_loss = epoch_loss_sum / len(train_set)
    avg_train_makespan = epoch_makespan_sum / len(train_set)

    val_msg = ""
    if len(val_set) > 0 and (epoch % VALIDATE_STEP == 0):
        model.eval()
        val_makespans = []
        with torch.no_grad():
            for v_prob in val_set:
                e_batch_onehot, _, _, batch_priorities = model.reverse_diffusion_with_logprob(
                    v_prob['canvas'],
                    DEVICE,
                    num_samples=VALIDATE_BS,
                    time_guidance_scale=T_SCALER,
                    position_guidance_scale=POS_SCALER,
                    temperature_method=TEMPERATURE_METHOD,
                )

                e_indices = e_batch_onehot.argmax(dim=-1).cpu()
                prio_cpu = batch_priorities.cpu()

                best_ms_in_batch = float('inf')

                for i in range(VALIDATE_BS):
                    e_mat = e_indices[i]
                    prio = prio_cpu[i]

                    if validate_constraints(e_mat, v_prob['node_labels'], DEVICE, exact=True, data=v_prob['canvas']):
                        wp = graph_to_simulation_input(e_mat, v_prob['canvas'], v_prob['wp_objs'], prio)
                        _, rep, _ = simulate_complete_scheduling(wp, v_prob['power_data'])
                        ms = rep['total']['makespan']
                        if ms < best_ms_in_batch:
                            best_ms_in_batch = ms
                    else:
                        raise ValueError("Invalid Graph for Validation")
                if best_ms_in_batch != float('inf'):
                    val_makespans.append(best_ms_in_batch)

        if len(val_makespans) > 0:
            avg_val_ms = np.mean(val_makespans)
            val_msg = f" | Val Best Avg: {avg_val_ms:.1f}"
        else:
            val_msg = " | Val Failed"

    # Logging
    log_msg = (f"Epoch {epoch} | "
               f"Loss: {avg_train_loss:.2f} | "
               f"Train Avg MS: {avg_train_makespan:.1f} | "
               f"LR: {current_lr:.6e}"
               f"{val_msg}")

    print(log_msg)
    with open(log_path, "a") as f:
        f.write(log_msg + "\n")

    if (epoch + 1) % 100 == 0 and epoch != 0:
        torch.save(model.state_dict(), log_dir / f"model_ep{epoch}.pth")

print("Done.")
