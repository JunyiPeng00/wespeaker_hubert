import numpy as np
import matplotlib.pyplot as plt

# 数据
prune_rates = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# JiangYu's pruning
jiangyu_dev_EER = [1.697, 1.504, 2.077, 2.439, 3.820, 3.102, 6.517, 5.374, 12.090, 20.977]
jiangyu_eval_EER = [4.670, 5.575, 5.331, 5.624, 11.187, 10.223, 14.944, 15.101, 21.008, 26.973]

# DP HuBERT
dp_dev_EER = [1.697, 1.487, 1.650, 1.874, 2.483, 3.942, 6.750, 7.871, 9.488, 20.961]
dp_eval_EER = [4.670, 5.131, 6.311, 7.230, 7.971, 11.727, 13.561, 14.971, 19.746, 31.542]

# Baseline WavLM Base (94M)
wavlm_dev_EER = [2.394, 2.738, 3.293, 3.463, 4.726, 4.248, None, 15.628, None, 25.569]
wavlm_eval_EER = [4.566, 3.758, 4.829, 5.657, 6.049, 8.746, 11.456, 21.135, 24.291, 28.64]

# 转为 numpy，方便掩蔽
wavlm_dev_EER = np.array(wavlm_dev_EER, dtype=np.float64)
wavlm_eval_EER = np.array(wavlm_eval_EER, dtype=np.float64)

# Colors
colors = ['#4575b4', '#e08214', '#7fbf7b']

# -------------------
# Plot Eval
# -------------------
plt.figure(figsize=(8,5))
plt.plot(prune_rates, jiangyu_eval_EER, marker='o', color=colors[1], label="Structured Pruning")
plt.plot(prune_rates, dp_eval_EER, marker='s', color=colors[2], label='DPHuBERT')
plt.plot(prune_rates, wavlm_eval_EER, marker='^', color=colors[0], label='Proposed')

# 标注每个点
for x, y in zip(prune_rates, jiangyu_eval_EER):
    plt.text(x, y+0.5, f"{y:.2f}", ha='center', fontsize=7)
for x, y in zip(prune_rates, dp_eval_EER):
    plt.text(x, y+0.5, f"{y:.2f}", ha='center', fontsize=7)
for x, y in zip(prune_rates, wavlm_eval_EER):
    if not np.isnan(y):
        plt.text(x, y+0.5, f"{y:.2f}", ha='center', fontsize=7)

plt.xlabel('Pruning Ratio (%)')
plt.ylabel('EER (%)')
plt.title('EER vs Pruning Ratio (Eval)')
plt.legend()
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('eer_vs_pruning_eval.png', dpi=300, facecolor='white')
plt.show()
