import matplotlib.pyplot as plt

filename = 'training_log (1).txt'

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

last_baseline = None  # 用来保存上一次验证结果

with open(filename, 'r') as f:
    for line in f:
        if not line.startswith('Epoch'):
            continue

        parts = [p.strip() for p in line.split('|')]

        try:
            # ---- Epoch ----
            epoch = int(parts[0].split()[1])

            # ---- Train Avg MS ----
            train_avg = None
            baseline = None

            for p in parts:
                if 'Train Avg MS' in p:
                    train_avg = float(p.split(':')[1])
                elif 'Val Best Avg' in p:
                    baseline = float(p.split(':')[1])

            if train_avg is None:
                continue

            epochs.append(epoch)
            avg_makespans.append(train_avg)

            # ---- baseline 处理逻辑 ----
            if baseline is not None:
                last_baseline = baseline

            # 如果你希望“没有验证的 epoch 也画 baseline”，就用上一值
            baselines.append(last_baseline)

        except Exception as e:
            print(f"Skipping line: {line.strip()} | Error: {e}")

# 如果前几个 epoch 从未验证，baseline 会是 None，过滤掉
valid_idx = [i for i, b in enumerate(baselines) if b is not None]

epochs_plot = [epochs[i] for i in valid_idx]
avg_plot = [avg_makespans[i] for i in valid_idx]
baseline_plot = [baselines[i] for i in valid_idx]

baseline_plot = moving_average(baseline_plot)

plt.figure(figsize=(10, 6))
plt.plot(epochs_plot, avg_plot, label='Train Average')
plt.plot(epochs_plot, baseline_plot, label='Val Best Avg', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Makespan')
plt.title('Convergence Plot: Avg Makespan vs Baseline')
plt.legend()
plt.grid(True)

plt.savefig(f'Figures/{filename.replace(".txt","")}.png')
plt.show()
