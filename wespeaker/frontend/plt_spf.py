import matplotlib.pyplot as plt

# 数据
prune_rates = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
val_data = [0.276, 0.239, 0.237, 0.339, 0.290, 0.400, 0.465, 1.293, 2.678, 6.279]
eval_data = [0.096, 0.054, 0.077, 0.077, 0.043, 0.110, 0.099, 0.295, 0.505, 5.884]

# 绘图
plt.figure(figsize=(10, 6))

# 定义颜色
color1 = '#377eb8'  # 蓝色
color2 = '#4daf4a'  # 绿色

# 绘制Val数据
plt.plot(prune_rates, val_data, marker='o', color=color1, label='Val')
# 标注每个点
for x, y in zip(prune_rates, val_data):
    plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=12)

# 绘制Eval数据
plt.plot(prune_rates, eval_data, marker='s', color=color2, label='Eval')
# 标注每个点
for x, y in zip(prune_rates, eval_data):
    plt.text(x, y, f"{y:.2f}", ha='center', va='top', fontsize=12)


# 轴标签和标题
plt.xlabel('Pruning Ratio (%)', fontsize=12)
plt.ylabel('EER (%)', fontsize=12)
# plt.title('Effect of Pruning on Val and Eval', fontsize=14)

# 美化
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()

# 保存
plt.savefig("val_eval_vs_pruning.png", dpi=300, bbox_inches='tight')