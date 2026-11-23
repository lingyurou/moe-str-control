import matplotlib.pyplot as plt
import numpy as np

# 1. 设置顶刊级全局科学美学样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#333333'  # 柔和的深灰边框，避免死黑
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['grid.color'] = '#e5e5e5'      # 极淡的网格线
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

# 2. 真实填入你刚才跑出的 10 大数据集核心数据
datasets = [
    'CUTE80', 'IC03_860', 'IC03_867', 'IC13_1015', 'IC13_857',
    'IC15_1811', 'IC15_2077', 'IIIT5k_3000', 'SVT', 'SVTP'
]

# 严格对应日志大表中的百分比数字
exp1 = [47.23, 46.18, 45.96, 45.93, 45.58, 44.51, 44.69, 46.73, 44.30, 44.78]
exp2 = [10.06, 12.64, 12.71, 12.03, 13.98, 11.91, 11.45, 11.40, 14.04, 13.01]
exp3 = [30.54, 31.27, 31.46, 30.76, 31.19, 33.77, 33.42, 30.86, 32.01, 32.11]
exp4 = [12.16,  9.91,  9.87, 11.29,  9.25,  9.82, 10.45, 11.01,  9.64, 10.10]

x = np.arange(len(datasets))  # 基准位置
width = 0.2  # 柱子宽度，4个专家并列刚好占满

# 3. 创建画布（标准单栏/双栏比例，适合顶刊排版）
fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

# 开启底层网格，方便审稿人定量比对
ax.set_axisbelow(True)
ax.grid(axis='y')

# 4. 选用学术高级莫兰迪/哑光配色（学术界最喜欢的莫兰迪蓝灰调，严禁红绿大高光）
colors = ['#4A7BB0', '#8FB9A8', '#E9A97E', '#C38D9E']

# 5. 矩阵式物理绘制
rects1 = ax.bar(x - 1.5*width, exp1, width, label='Expert 1 (Primary)', color=colors[0], edgecolor='none')
rects2 = ax.bar(x - 0.5*width, exp2, width, label='Expert 2 (Regularizer)', color=colors[1], edgecolor='none')
rects3 = ax.bar(x + 0.5*width, exp3, width, label='Expert 3 (Secondary)', color=colors[2], edgecolor='none')
rects4 = ax.bar(x + 1.5*width, exp4, width, label='Expert 4 (Regularizer)', color=colors[3], edgecolor='none')

# 6. 细节雕刻：标签、刻度与界限
ax.set_ylabel('Routing Allocation Proportion (%)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Decoupled Semantic Routing Trajectory across Ten Evaluation Benchmarks', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10, rotation=15, ha='right') # 微微倾斜避免字体重叠
ax.set_ylim(0, 60) # 留出顶部空间放图例

# 7. 优雅的图例设计（单行平铺居中，绝对不遮挡任何高柱子）
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True,
          facecolor='white', edgecolor='#e5e5e5', fontsize=10)

# 8. 消除上方和右侧的冗余边框（现代学术排版极简美学）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# 9. 双通道保存：PNG供快速查看，PDF是投递IEEE TIP等顶刊系统必须使用的“无限放大不模糊”矢量格式
plt.savefig('moe_routing_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('moe_routing_distribution.pdf', format='pdf', bbox_inches='tight')

print("🎉 [SUCCESS] 顶刊级矢量图已成功渲染并保存至当前目录！")
print("👉 查看 moe_routing_distribution.png 获取预览，LaTeX 排版请直接挂载 moe_routing_distribution.pdf")