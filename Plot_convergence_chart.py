import matplotlib.pyplot as plt

# filename = 'rl_checkpoints/rl_finetuned/training_log_copy.txt'
filename = 'training_log_P_Guidance.txt'

epochs = []
avg_makespans = []
baselines = []

def moving_average(data, window_size=9):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        smoothed.append(sum(window) / len(window))
    return smoothed

with open(filename, 'r') as f:
    for line in f:
        if line.startswith('Epoch'):
            # example: Epoch 0 | Loss: -0.38 | Avg Makespan: 124.5 | Best: 104.0 | Baseline: 160.6 |
            #          Epoch 0 | Loss: 2.72 | Train Avg MS: 152.4 | LR: 9.999994e-06 | Val Best Avg: 164.0
            parts = line.split('|')

            epoch_part = parts[0].strip()
            avg_makespan_part = parts[2].strip()
            baseline_part = parts[4].strip()

            try:
                epoch = int(epoch_part.split()[1])
                avg_makespan = float(avg_makespan_part.split(':')[1].strip())
                baseline = float(baseline_part.split(':')[1].strip())

                epochs.append(epoch)
                avg_makespans.append(avg_makespan)
                baselines.append(baseline)
            except (ValueError, IndexError) as e:
                print(f"Skipping line due to parsing error: {line.strip()} - {e}")
                continue

baselines = moving_average(baselines)
plt.figure(figsize=(10, 6))

plt.plot(epochs, avg_makespans, label='Train Average')

plt.plot(epochs, baselines, label='Test(Best out of 4)', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Makespan')
plt.title('Convergence Plot: Avg Makespan vs Baseline')

plt.legend()
plt.grid(True)

plt.savefig(f'Figures/{filename.replace(".txt","")}.png')

plt.show()