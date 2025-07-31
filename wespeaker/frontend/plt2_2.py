import numpy as np
import matplotlib.pyplot as plt

# Layer labels remain the same
layers = ['CNN Encoder'] + [f'Layer {i+1:02d}' for i in range(12)]
layer_labels = layers[::-1]

# System data remains unchanged
systems = [
    {
        'name': 'Proposed',
        'ffn': np.array([1939113, 3297633, 3017899, 3251523, 0, 3664976, 3439037, 2650556, 0, 358889, 0, 103747, 54563]),
        'sa':  np.array([0, 2366740, 2362900, 2166100, 2166100, 2362900, 2362900, 2362900, 1969300, 1575700, 591700, 198100, 0]),
        'fe':  1939113
    },
    {
        'name': 'Structured Pruning',
        'ffn': np.array([1940261, 2140272, 1739115, 2244788, 3789473, 1313366, 1848242, 3113193, 4545677, 192893, 385018, 1648432, 2389266]),
        'sa':  np.array([0, 1579540, 1182100, 1772500, 2362900, 788500, 1182100, 1575700, 2166100, 0, 198100, 591700, 1575700]),
        'fe':  1940261
    },
    {
        'name': 'DP-HUBERT',
        'ffn': np.array([1819868, 2051126, 1799058, 2400025, 3906285, 1216535, 1831335, 3166988, 4590250, 219022, 438813, 1746800, 2587539]),
        'sa':  np.array([0, 1382740, 1182100, 1969300, 2166100, 985300, 985300, 1772500, 2166100, 0, 0, 788500, 1182100]),
        'fe':  1819868
    }
]

# System B reference remains unchanged
feature_extractor_B = 4200448
self_attention_B = np.array([0, 2366740, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900])
feed_forward_B = np.array([4200448, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432])

n_systems = len(systems)
n_layers = len(layers)

# Parameters for vertical bars
bar_width = 2.5
group_gap = 1

# x-coordinate mapping
x_pos = []
for i in range(n_layers):
    start = i * (n_systems + group_gap)
    x_pos.append([start + j for j in range(n_systems)])

all_x_pos = [item for sublist in x_pos for item in sublist]
max_x = all_x_pos[-1] + 1

# Setup canvas (using the wider figure size)
plt.figure(figsize=(20, 10))

colors = ['#4575b4', '#e08214', '#7fbf7b']
gray = "#bdbdbd"

for sys_idx, sys in enumerate(systems):
    ffn_percent = []
    sa_percent = []
    fe_percent = sys['fe'] / feature_extractor_B * 100 if feature_extractor_B > 0 else 0
    ffn_percent.append(fe_percent)
    sa_percent.append(0)
    for i in range(1, n_layers):
        ffn_p = sys['ffn'][i] / feed_forward_B[i] * 100 if feed_forward_B[i] > 0 else 0
        sa_p = sys['sa'][i] / self_attention_B[i] * 100 if self_attention_B[i] > 0 else 0
        ffn_percent.append(ffn_p)
        sa_percent.append(sa_p)
        
    ffn_percent = np.array(ffn_percent[::-1])
    sa_percent = np.array(sa_percent[::-1])

    for layer_idx in range(n_layers):
        xpos = x_pos[layer_idx][sys_idx]
        
        plt.bar(
            xpos, -ffn_percent[layer_idx], width=bar_width/n_systems, color=colors[sys_idx], edgecolor=gray, alpha=0.72
        )
        plt.bar(
            xpos, sa_percent[layer_idx], width=bar_width/n_systems, color=colors[sys_idx], edgecolor=gray, alpha=0.7
        )
        
        if abs(ffn_percent[layer_idx]) > 0:
            if abs(ffn_percent[layer_idx]) < 10:
                plt.text(xpos, -ffn_percent[layer_idx] - 1, f'{ffn_percent[layer_idx]:.1f}%', 
                         ha='center', va='top', color='black', fontsize=10)
            else:
                plt.text(xpos, -ffn_percent[layer_idx]/2, f'{ffn_percent[layer_idx]:.1f}%', 
                         ha='center', va='center', color='black', fontsize=10)
        
        if abs(sa_percent[layer_idx]) > 0:
            if abs(sa_percent[layer_idx]) < 10:
                plt.text(xpos, sa_percent[layer_idx] + 1, f'{sa_percent[layer_idx]:.1f}%', 
                         ha='center', va='bottom', color='black', fontsize=10)
            else:
                plt.text(xpos, sa_percent[layer_idx]/2, f'{sa_percent[layer_idx]:.1f}%', 
                         ha='center', va='center', color='black', fontsize=10)

xticks_pos = [x_pos[i][n_systems//2] for i in range(n_layers)]
plt.xticks(xticks_pos, layer_labels, rotation=45, ha='right', fontsize=12)

yticks = np.arange(-100, 101, 25)
yticklabels = [str(abs(y)) for y in yticks]
plt.yticks(yticks, yticklabels)
plt.ylabel('Parameter ratio relative to unpruned WavLM Base (%)', fontsize=12)

plt.text(max_x / 2, -115, "FFN", ha='center', va='center', fontsize=16, fontweight='bold')
plt.text(max_x / 2, 115, "MHSA", ha='center', va='center', fontsize=16, fontweight='bold')

plt.ylim(-125, 125)
plt.xlim(-group_gap, max_x)

# --- 新增代码：反转X轴 ---
plt.gca().invert_xaxis()
# --- 修改结束 ---

plt.axhline(0, color='black', linewidth=1)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.tight_layout(rect=[0, 0, 0.85, 1]) 

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=colors[0], label='Proposed', edgecolor='black', alpha=1),
    Patch(facecolor=colors[1], label='Structured Pruning', edgecolor='black', alpha=1),
    Patch(facecolor=colors[2], label='DPHubert', edgecolor='black', alpha=1),
]
plt.legend(
    handles=legend_handles,
    frameon=False, ncol=1, loc='upper right', bbox_to_anchor=(1.0, 0.9)
)

plt.savefig("./layerwise_population_pyramid_vertical_inverted.png", bbox_inches='tight', dpi=300)
plt.show()