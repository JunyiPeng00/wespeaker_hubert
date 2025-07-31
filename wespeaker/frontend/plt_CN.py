import matplotlib.pyplot as plt

# 数据
prune_rates = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
eer_cnceleb_e = [8.009, 7.97, 7.953, 7.823, 7.711, 8.268, 8.395, 8.426, 8.77, 9.747]

# 绘图
plt.figure(figsize=(8,5))

# 用和之前一致的颜色系
color = '#984ea3'  # 紫色

plt.plot(prune_rates, eer_cnceleb_e, marker='o', color=color, label='CNCeleb-E')

# 标注每个点
for x, y in zip(prune_rates, eer_cnceleb_e):
    plt.text(x, y+0.05, f"{y:.2f}", ha='center', va='bottom', fontsize=12)

# 轴标签
plt.xlabel('Pruning Ratio (%)', fontsize=12)
plt.ylabel('EER (%)', fontsize=12)
# plt.title('Effect of Pruning on EER (CNCeleb-E)', fontsize=14)

# 美化
plt.grid(alpha=0.3, linestyle='--')
plt.legend()
plt.tight_layout()

# 保存
plt.savefig("eer_vs_pruning_cnceleb_e.png", dpi=300, facecolor='white')
plt.show()
