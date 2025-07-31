import numpy as np
import matplotlib.pyplot as plt

# --------- 数据定义（原样） ----------
layers = ['FeatureExtractor'] + [f'Layer {i+1:02}' for i in range(12)]

feature_extractor_0 = 1939113
self_attention_0 = np.array([0, 2366740, 2362900, 2166100, 2166100, 2362900, 2362900, 2362900, 1969300, 1575700, 591700, 198100, 0])
feed_forward_0 = np.array([1939113, 3297633, 3017899, 3251523, 0, 3664976, 3439037, 2650556, 0, 358889, 0, 103747, 54563])

feature_extractor_A = 1940261
self_attention_A = np.array([0, 1579540, 1182100, 1772500, 2362900, 788500, 1182100, 1575700, 2166100, 0, 198100, 591700, 1575700])
feed_forward_A = np.array([1940261, 2140272, 1739115, 2244788, 3789473, 1313366, 1848242, 3113193, 4545677, 192893, 385018, 1648432, 2389266])

feature_extractor_B = 4200448
self_attention_B = np.array([0, 2366740, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900])
feed_forward_B = np.array([4200448, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432])

feature_extractor_C = 1819868
self_attention_C = np.array([0, 1382740, 1182100, 1969300, 2166100, 985300, 985300, 1772500, 2166100, 0, 0, 788500, 1182100])
feed_forward_C = np.array([1819868, 2051126, 1799058, 2400025, 3906285, 1216535, 1831335, 3166988, 4590250, 219022, 438813, 1746800, 2587539])

systems_raw = {
    'Proposed': (feature_extractor_0, self_attention_0, feed_forward_0),
    'Structured Pruning': (feature_extractor_A, self_attention_A, feed_forward_A),
    'DPHubert': (feature_extractor_C, self_attention_C, feed_forward_C),
}

# --------- 计算百分比 ---------
systems_ratio = {}
for name, (fe, sa, ffn) in systems_raw.items():
    sa_ratio = np.divide(sa, self_attention_B, out=np.zeros_like(sa, dtype=float), where=self_attention_B > 0)
    ffn_ratio = np.divide(ffn, feed_forward_B, out=np.zeros_like(ffn, dtype=float), where=feed_forward_B > 0)
    fe_ratio = fe / feature_extractor_B
    systems_ratio[name] = (fe_ratio, sa_ratio, ffn_ratio)

plt.rcParams.update({
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
})

system_names = list(systems_ratio.keys())
n_layers = len(layers)
bar_width = 0.18
indices = np.arange(n_layers)

plt.figure(figsize=(13, 5.5), dpi=120)

# Nature风格主色
nature_colors = [
    "#4575b4", # 蓝色 System 0
    "#e08214", # 橙色 System A
    "#7fbf7b", # 绿色 System C
]
gray = "#bdbdbd"

for i, name in enumerate(system_names):
    fe_ratio, sa_ratio, ffn_ratio = systems_ratio[name]
    sa_all = np.copy(sa_ratio) * 100
    ffn_all = np.copy(ffn_ratio) * 100

    pos = indices + (i - 1) * bar_width

    # FeatureExtractor
    fe_height = fe_ratio * 100
    plt.bar(pos[0], fe_height, width=bar_width, color=nature_colors[i],
            edgecolor=gray, linewidth=0.75, zorder=3)
    # 百分比横向在柱内
    if fe_height > 0:
        color = "white" if fe_height > 16 else "black"
        plt.text(pos[0], fe_height/2, f'{fe_height:.1f}%', ha='center', va='center', rotation=90,
                 fontsize=10, color=color, fontweight='bold')

    # SelfAttention
    sa = plt.bar(pos[1:], sa_all[1:], width=bar_width, color=nature_colors[i],
                 alpha=0.9, edgecolor=gray, linewidth=0.75, zorder=3, label=name if i==0 else None)
    # FeedForward
    ffn = plt.bar(pos[1:], ffn_all[1:], bottom=sa_all[1:], width=bar_width, color=nature_colors[i],
                  alpha=0.48, edgecolor=gray, linewidth=0.75, zorder=3)

    # 内部百分比（SelfAttention）
    for x, h in zip(pos[1:], sa_all[1:]):
        if h > 0:
            color = "white" if h > 18 else "black"
            plt.text(x, h/2, f'{h:.1f}%', ha='center', va='center', fontsize=9, color=color, fontweight='bold', rotation=90)
    # 内部百分比（FeedForward）
    for x, sa_h, ffn_h in zip(pos[1:], sa_all[1:], ffn_all[1:]):
        if ffn_h > 0:
            color = "white" if ffn_h > 18 else "black"
            plt.text(x, sa_h + ffn_h/2, f'{ffn_h:.1f}%', ha='center', va='center', fontsize=9, color=color, fontweight='bold', rotation=90)

plt.xticks(indices, layers, rotation=28, ha='right')
plt.ylabel('Relative Parameters (%)', labelpad=7)
# plt.title('Layer-wise Parameter Ratio', pad=7)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.1)
ax.spines['bottom'].set_linewidth(1.1)
plt.ylim(0, 180)
plt.yticks(np.arange(0, 180, 25))
plt.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6, zorder=0)

from matplotlib.patches import Patch

from matplotlib.patches import Patch

# 3个系统的主色（深色）
system_handles = [
    Patch(facecolor=nature_colors[0], edgecolor=gray, label='Proposed'),
    Patch(facecolor=nature_colors[1], edgecolor=gray, label='Structured Pruning'),
    Patch(facecolor=nature_colors[2], edgecolor=gray, label='DPHubert'),
]

# SelfAttention/FeedForward示意（深/浅色，与主色系一致）
sa_handle = Patch(facecolor="#666666", edgecolor='none', label="SelfAttention (darker)")
ffn_handle = Patch(facecolor="#bbbbbb", edgecolor='none', label="FeedForward (lighter)")

# 系统和成分分别分组
legend_handles = system_handles + [sa_handle, ffn_handle]
legend_labels = [
    'Proposed', 'Structured Pruning', 'DPHubert',
    'SelfAttention (darker)', 'FeedForward (lighter)'
]

plt.legend(legend_handles, legend_labels, frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(1.01, 1.0))

plt.tight_layout()
plt.savefig("layerwise_relative_percentage_nature.png", bbox_inches='tight')
plt.show()
