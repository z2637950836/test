import pandas as pd
import matplotlib.pyplot as plt

# 读取文件 - target_data.csv 没有列名，需要指定
sim = pd.read_csv("NumberOfHousehold.csv")  # 这个有列名
target = pd.read_csv("data/target_data.csv", header=None, names=['Year', 'Number-of-Households'])  # 这个没有列名

# 查看前几行，确认列名
print("模拟数据 (sim):")
print(sim.head())
print("\n目标数据 (target):")
print(target.head())

# 绘制对比图
plt.figure(figsize=(12, 7))
plt.plot(sim['Year'], sim['Number-of-Households'], label='Simulated (Model)', linewidth=2.5, marker='o', markersize=4)
plt.plot(target['Year'], target['Number-of-Households'], label='Real Data (Target)', linewidth=2, linestyle='--', marker='s', markersize=4)

# 添加标题和标签
plt.title("Anasazi Model: Comparison of Simulated vs Real Household Numbers", fontsize=14, fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Households", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片（避免虚拟机图形显示问题）
plt.savefig('anasazi_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 'anasazi_comparison.png'")

# 显示图表
plt.show()
