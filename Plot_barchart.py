import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# 假设您的四个文件名为 file1.csv, file2.csv, file3.csv, file4.csv
# 在实际使用时，请修改为您真实的文件名
file_names = ['results_GA.csv', 'results_RL.csv',
              'result_model_rl_adding_temperature_BS32_T16_Layer6_HIDDEN_DIMENSION256.csv',
              'result_model_rl_adding_temperature_BS32_T4_Layer6_HIDDEN_DIMENSION128_Trainset20_P_Guidance.csv',
              'result_model_rl_adding_temperature_BS32_T4_Layer6_HIDDEN_DIMENSION128_Trainset20_P_Guidance_evo3.csv',
              'result_model_rl_adding_temperature_BS32_T4_Layer6_HIDDEN_DIMENSION128_Trainset20_P_Guidance_evo10.csv',
              'result_model_rl_adding_temperature_BS32_T4_Layer6_HIDDEN_DIMENSION128_Trainset20_P_Guidance_evo60_Pguidance5.csv',
              'result_model_rl_new1219_2_evo_Pguidance5.csv',
              # 'result_model_rl_new1219_2_evo_Pguidance5_passPriority.csv',
              # 'result_model_rl_new1219_2_evo_Pguidance5_passPriority_step2.csv',
              # 'result_model_rl_new1219_2_evo20_Pguidance5_passPriority.csv',
              # 'result_model_rl_new1219_2_evo_Pguidance5_passPriority_step4.csv',
              # 'result_model_rl_new1219_2_evo100_Pguidance5_passPriority.csv',
              'result_random.csv']
data_frames = []
legend_name = ['GA', 'RL', 'Diffusion without P Guidance', 'Diffusion', 'Diffusion with EVO3', 'Diffusion with EVO10',
               'Diffusion with EVO10_sample60','Diffusion_correct with EVO10_sample40',
               # 'Diffusion_correct_pass_Priority with EVO10_sample40',
               # 'Diffusion_correct_pass_Priority with EVO10_sample40_mutationStep2',
               # 'Diffusion_correct_pass_Priority with EVO20_sample40_mutationStep1',
               # 'Diffusion_correct_pass_Priority with EVO20_sample40_mutationStep4',
               # 'Diffusion_correct_pass_Priority with EVO100_sample40_mutationStep4',
               'Random']
figname = 'grouped_bar_chart_8.png'
# 定义提取Job数量的函数
def extract_job_count(filename):
    # 正则表达式：匹配 gen_ 和 _job 之间的数字
    match = re.search(r'gen_(\d+)_job', str(filename))
    if match:
        return int(match.group(1))
    return None


# 循环读取文件并处理
for i, file in enumerate(file_names):
    # 读取CSV (根据实际情况，如果不含表头可能需要 header=None)
    # 这里假设您的CSV和描述一致，第一行是表头
    try:
        df = pd.read_csv(file)

        # 提取Job数量
        df['Job_Count'] = df['Filename'].apply(extract_job_count)

        # 添加来源标记 (例如: File 1, File 2...)
        df['Source'] = legend_name[i]

        data_frames.append(df)
    except FileNotFoundError:
        print(f"警告: 找不到文件 {file}，请确保文件在当前目录下。")

# 如果读取到了数据，开始绘图
if data_frames:
    # 合并所有数据
    all_data = pd.concat(data_frames, ignore_index=True)

    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 绘制分组柱状图
    # x: Job数量, y: Makespan, hue: 来源文件(区分颜色)
    sns.barplot(
        data=all_data,
        x='Job_Count',
        y='Best_Makespan',
        hue='Source',
        # errorbar='sd'  # 显示标准差误差线，如果不需要可以设为 None
        errorbar=None  # 显示标准差误差线，如果不需要可以设为 None
    )

    plt.title('Comparison of Average Makespan by Job Count')
    plt.xlabel('Job Count')
    plt.ylabel('Best Makespan')
    plt.legend(title='File Source')

    # 保存或显示
    plt.savefig(figname)
    plt.show()