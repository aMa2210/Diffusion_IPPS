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
EPOCHS = 1000
BATCH_SIZE = 16
T_STEPS = 30
ENTROPY_START = 0.05
ENTROPY_END = 0.001
DECAY_STEPS = 500
T_SCALER = 0.01
all_workpieces_objs, machine_power_data = load_problem_definitions(PROBLEM_FILE)

raw_wp_dicts, raw_machines = load_ipps_problem_from_json(PROBLEM_FILE)
ipps_canvas = get_ipps_problem_data(raw_wp_dicts, raw_machines, DEVICE)

model = LightweightIndustrialDiffusion(T=T_STEPS, device=DEVICE).to(DEVICE)

# model.load_state_dict(torch.load("ablation_runs_11_19_for_RL/baseline/model.pth"))

optimizer = optim.Adam(model.parameters(), lr=LR)

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

    if epoch == 0:
        moving_avg_makespan = current_avg_makespan
    else:
        moving_avg_makespan = 0.9 * moving_avg_makespan + 0.1 * current_avg_makespan


    advantages = []
    for ms in batch_makespans:
        adv = (moving_avg_makespan - ms)
        # Optional: Scale by dividing by the standard deviation.
        # adv = adv / (np.std(batch_makespans) + 1e-8)
        advantages.append(adv)

    advantages = torch.tensor(advantages).to(DEVICE)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Loss = - (Advantage * LogProb) - (Entropy_Coef * Entropy)
    loss_policy = 0
    loss_entropy = 0

    for log_p, adv, ent in zip(batch_log_probs, advantages, batch_entropies):
        loss_policy += -(adv * log_p)
        loss_entropy += -ent

    loss = (loss_policy / BATCH_SIZE) + (current_entropy_coef * (loss_entropy / BATCH_SIZE))

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} | Loss: {loss.item():.2f} | Avg Makespan: {current_avg_makespan:.1f} | Best: {min(batch_makespans):.1f} | Baseline: {moving_avg_makespan:.1f}")

    if epoch % 100 == 0:
        save_dir = Path(f"rl_checkpoints/{RUN_NAME}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / f"model_ep{epoch}.pth")

print("#######################Finished#######################")