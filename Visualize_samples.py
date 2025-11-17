import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from Industrial_Pipeline_Functions import load_ipps_problem_from_json, get_ipps_problem_data


# --- 1. (你需要修改这里) ---
# 指定你想要可视化的运行结果
RUN_NAME = "baseline"  # (例如: "baseline", "no_projector", ...)
PROBLEM_FILE = "TestSet/1.json"  # (必须与 Ablation_Runs.py 中使用的文件 *完全相同*)
SAMPLE_INDEX = 0  # (你想看哪个样本？0 = 第一个)
# ------------------------------

# --- 2. 加载 IPPS "画布" (Canvas) ---
# 我们需要它来获取节点的 *含义* (例如 "W1-F1" 或 "M3")
try:
    problem_workpieces, problem_machines = load_ipps_problem_from_json(PROBLEM_FILE)
    ipps_canvas = get_ipps_problem_data(problem_workpieces, problem_machines, "cpu")
except FileNotFoundError:
    print(f"错误: 找不到问题文件 '{PROBLEM_FILE}'")
    sys.exit(1)
except Exception as e:
    print(f"加载 IPPS 画布时出错: {e}")
    sys.exit(1)

# --- 3. 加载 samples.pt 文件 ---
ablation_dir = Path("ablation_runs_11_17_posterior_randomChooseIfInvalid")  # (确保这与你的 ROOT_OUT 匹配)
samples_path = ablation_dir / RUN_NAME / "samples.pt"

if not samples_path.exists():
    print(f"错误: 找不到 samples.pt 文件。")
    print(f"路径: {samples_path}")
    sys.exit(1)

print(f"Loading samples from {samples_path}...")
samples = torch.load(samples_path)

if not samples or SAMPLE_INDEX >= len(samples):
    print(f"错误: 样本索引 {SAMPLE_INDEX} 超出范围。文件只包含 {len(samples)} 个样本。")
    sys.exit(1)

# --- 4. 提取节点和边 ---
# node_labels_tensor, edge_matrix = samples[SAMPLE_INDEX]
# (更正：node_labels 是固定的，我们可以直接从 ipps_canvas 获取)
_, edge_matrix = samples[SAMPLE_INDEX]
node_labels_tensor = ipps_canvas.x.argmax(dim=1)  # (0=Op, 1=Machine)
num_nodes = ipps_canvas.num_nodes
num_ops = ipps_canvas.op_info.size(0)

# --- 5. 构建 NetworkX 图 ---
print("Building graph...")
G = nx.DiGraph()
node_colors = []
node_labels = {}

# 添加节点
for i in range(num_nodes):
    node_type = node_labels_tensor[i].item()
    layer_id = 0  # 默认为第 0 层 (Op)

    if node_type == 0:  # 节点是 Operation
        wp_idx, feat_idx = ipps_canvas.op_info[i].tolist()
        label = f"W{wp_idx + 1}-F{feat_idx + 1}"
        node_colors.append("skyblue")
        layer_id = 0  # 放在左侧
    else:  # 节点是 Machine
        machine_id = ipps_canvas.machine_map[i - num_ops].item()
        label = f"M{machine_id}"
        node_colors.append("salmon")
        layer_id = 1  # 放在右侧

    G.add_node(i, layer=layer_id)  # <-- 修正：在这里添加 "layer" 属性
    node_labels[i] = label

# 添加边
src_nodes, dst_nodes = edge_matrix.nonzero(as_tuple=True)
edge_colors = []

for src, dst in zip(src_nodes.tolist(), dst_nodes.tolist()):
    G.add_edge(src, dst)
    # 给边上色
    if node_labels_tensor[src] == 0 and node_labels_tensor[dst] == 0:
        edge_colors.append("blue")  # Op -> Op
    else:
        edge_colors.append("black")  # Op -> Machine

# --- 6. 绘图 ---
print("Drawing graph...")
plt.figure(figsize=(20, 15))

# 使用 multipartite 布局将 Op 和 Machine 分开
pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')

nx.draw_networkx(
    G,
    pos=pos,
    labels=node_labels,
    node_color=node_colors,
    node_size=2000,
    edge_color=edge_colors,
    font_size=8,
    font_weight="bold",
    arrows=True,
    arrowstyle="-|>",
    arrowsize=20,
)

plt.title(f"可视化: {RUN_NAME} / Sample {SAMPLE_INDEX}\nProblem: {PROBLEM_FILE}", fontsize=20)
plt.xlabel("Machines")  # 右侧是 partition 1
plt.ylabel("Operations")  # 左侧是 partition 0
plt.box(False)
# output_filename = f"{RUN_NAME}_sample_{SAMPLE_INDEX}.png"
# plt.savefig(output_filename)
# print(f"✅ Figure save to: {output_filename}")
plt.show()