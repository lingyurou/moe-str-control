import pandas as pd
import matplotlib.pyplot as plt

# 读取数据，指定无表头，并将列名强制设为 epoch 和 acc
df = pd.read_csv("eval_history.csv", header=None, names=['epoch', 'acc'])

# 按 epoch 排序，防止读取顺序错乱
df = df.sort_values(by='epoch')

plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['acc'], marker='o', linestyle='-', color='b', linewidth=2)

# 添加标签，让图表符合学术规范
plt.title("G-MoE-STR Performance Progress", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("CUTE80 Accuracy (%)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 100) # 准确率范围通常是 0-100

plt.savefig("accuracy_trend.png")
print("✅ 性能演化趋势图已生成: accuracy_trend.png")