import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob
import os
import random
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


random.seed(42)
TRAIN_DIR = "Problem_TrainSet"
VAL_DIR = "Problem_ValidationSet"
# PROBLEM_FILE = "Problem_TrainSet/1.json"
RUN_NAME = "rl_multi_generalization_BATCH_SIZE16_T_STEPS4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
LR = 1e-5   #learning rate
EPOCHS = 2000
BATCH_SIZE = 16
T_STEPS = 4
# ENTROPY_START = 0.005
# ENTROPY_END = 0.0001
# DECAY_STEPS = 500
ENTROPY_START = 0.1
ENTROPY_END = 0.01
DECAY_STEPS = 300
T_SCALER = 0.001
VALIDATE_STEP = 1  #validate the model every {VALIDATE_STEP} steps
VALIDATE_BS = 4     #how many samples are generated when validating the model, then choose the best one
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


model = LightweightIndustrialDiffusion(T=T_STEPS, hidden_dim=256, num_layers=6, nhead=4, dropout=0.1,device=DEVICE).to(DEVICE)

# model.load_state_dict(torch.load("ablation_runs_11_19_for_RL/baseline/model.pth"))

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

# baseline for each problem
baseline_registry = {}
for prob in train_set:
    baseline_registry[prob['id']] = None


log_dir = Path(f"rl_checkpoints/{RUN_NAME}")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "training_log.txt"

with open(log_path, "w") as f:
    f.write(f"Training Log for {RUN_NAME}\n")
    f.write("==================================================\n")
    
for epoch in range(EPOCHS):
    model.train()
    progress = min(1.0, epoch / DECAY_STEPS)
    current_entropy_coef = ENTROPY_START - (ENTROPY_START - ENTROPY_END) * progress
    random.shuffle(train_set)

    epoch_loss_sum = 0
    epoch_makespan_sum = 0

    pbar = tqdm(train_set, desc=f"Epoch {epoch}", leave=False)
    for prob in pbar:
        optimizer.zero_grad()

        prob_id = prob['id']
        single_canvas = prob['canvas']

        batch_makespans = []
        batch_log_probs = []
        batch_entropies = []

        for b in range(BATCH_SIZE):
            generated_edges, log_prob, entropy, priorities = model.reverse_diffusion_with_logprob(
                single_canvas, DEVICE, time_guidance_scale=T_SCALER
            )

            edges_matrix = generated_edges.argmax(dim=-1).detach().cpu()

            is_valid = validate_constraints(
                edges_matrix,
                prob['node_labels'],
                DEVICE,
                exact=True,
                data=single_canvas
            )
            if not is_valid:
                raise ValueError("Invalid Graph")
            else:
                wp_cycles = graph_to_simulation_input(edges_matrix, single_canvas, prob['wp_objs'], priorities)
                _, energy_report, _ = simulate_complete_scheduling(wp_cycles, prob['power_data'])
                makespan = energy_report['total']['makespan']

                if makespan <= 0: raise ValueError("Invalid Graph")
            batch_makespans.append(makespan)
            batch_log_probs.append(log_prob)
            batch_entropies.append(entropy)
        batch_makespans_np = np.array(batch_makespans)
        batch_mean = np.mean(batch_makespans_np)

        if baseline_registry[prob_id] is None:
            baseline_registry[prob_id] = batch_mean
        moving_avg = baseline_registry[prob_id]

        adv_local = batch_mean - batch_makespans_np
        adv_global = moving_avg - batch_makespans_np
        raw_advantages = 0.5 * adv_local + 0.5 * adv_global

        advantages = torch.tensor(raw_advantages, dtype=torch.float32).to(DEVICE)

        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, min=-5.0, max=5.0)

        baseline_registry[prob_id] = 0.9 * moving_avg + 0.1 * batch_mean

        k = BATCH_SIZE  # tbd change the top k to control the loss backward logic
        topk_indices = torch.topk(advantages, k).indices

        batch_log_probs_tensor = torch.stack(batch_log_probs)
        batch_entropies_tensor = torch.stack(batch_entropies)

        selected_log_probs = batch_log_probs_tensor[topk_indices]
        selected_advantages = advantages[topk_indices]
        selected_entropies = batch_entropies_tensor[topk_indices]

        selected_log_probs = selected_log_probs / 100.0

        high_perf = selected_advantages > 1.0
        selected_advantages[high_perf] *= 1.5

        loss_policy = -(selected_advantages * selected_log_probs).mean()
        loss_entropy = -selected_entropies.mean()

        loss = loss_policy + current_entropy_coef * loss_entropy

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss_sum += loss.item()
        epoch_makespan_sum += batch_mean
        pbar.set_postfix({'L': f"{loss.item():.2f}", 'Avg': f"{batch_mean:.1f}"})

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
                TEST_BATCH = VALIDATE_BS
                best_ms = float('inf')
                for _ in range(TEST_BATCH):
                    e_onehot, _, _, prio = model.reverse_diffusion_with_logprob(
                        v_prob['canvas'], DEVICE, time_guidance_scale=T_SCALER
                    )
                    e_mat = e_onehot.argmax(dim=-1).cpu()

                    if validate_constraints(e_mat, v_prob['node_labels'], DEVICE, exact=True,
                                            data=v_prob['canvas']):
                        wp = graph_to_simulation_input(e_mat, v_prob['canvas'], v_prob['wp_objs'], prio)
                        _, rep, _ = simulate_complete_scheduling(wp, v_prob['power_data'])
                        ms = rep['total']['makespan']
                        if ms < best_ms:
                            best_ms = ms
                    else: raise ValueError("Invalid Graph for Validation")

                val_makespans.append(best_ms)

        if len(val_makespans) > 0:
            avg_val_ms = np.mean(val_makespans)
            val_msg = f" | Val Best Avg: {avg_val_ms:.1f}"
        else:
            val_msg = " | Val Failed"

    log_msg = (f"Epoch {epoch} | "
               f"Loss: {avg_train_loss:.2f} | "
               f"Train Avg MS: {avg_train_makespan:.1f} | "
               f"LR: {current_lr:.6e}"
               f"{val_msg}")

    print(log_msg)
    with open(log_path, "a") as f:
        f.write(log_msg + "\n")

    if epoch % 100 == 0:
        torch.save(model.state_dict(), log_dir / f"model_ep{epoch}.pth")
print("Done.")
