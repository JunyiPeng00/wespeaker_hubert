import numpy as np
import matplotlib.pyplot as plt

# 层标签
layers = ['CNN Encoder'] + [f'Layer {i+1:02}' for i in range(12)]
layer_labels = layers[::-1]  # 金字塔风格，自下而上

# 系统数据
systems = [
    {
        'name': 'CNCeleb',
        'ffn': np.array([ 2560224, 3923192, 3663439, 3552775, 3600422, 3260745, 2329323, 1370235, 604809, 0, 0, 0, 128339 ]),
        'sa':  np.array([0, 1973140, 2362900, 2362900, 2362900, 2362900, 2166100, 2362900, 1378900, 0, 0, 0, 0]),
        'fe':  2560224
    },
    {
        'name': 'VoxCeleb',
        'ffn': np.array([1712952, 3389853, 2999455, 2996381, 3259208,3188506, 2965641, 2137198, 1285700,670900, 437276, 0, 203652]),
        'sa':  np.array([ 0, 989140, 2362900, 2362900, 2166100, 2362900, 1969300, 1969300, 1772500, 788500, 394900, 0, 0]),
        'fe':  1712952
    },
        {
        'name': 'ASVSpoof5',
        'ffn': np.array([1939113, 3297633, 3017899, 3251523, 0, 3664976, 3439037, 2650556, 0, 358889, 0, 103747, 54563]),
        'sa':  np.array([0, 2366740, 2362900, 2166100, 2166100, 2362900, 2362900, 2362900, 1969300, 1575700, 591700, 198100, 0]),
        'fe':  1939113
    },
    {
        'name': 'SpoofCeleb',
        'ffn': np.array([
            0, # Placeholder for index 0, as 'fe' is used
            3067083, 2836533, 3188506, 0, 3651143, 3457481,
            2610594, 1082816, 217485, 0, 97599, 45341
        ]),
        'sa':  np.array([
            0, # Placeholder for index 0
            2366740, 2166100, 2362900, 0, 2362900, 2362900,
            2362900, 2362900, 1772500, 788500, 394900, 0
        ]),
        'fe':  1875942
    }
]

# System B reference
feature_extractor_B = 4200448
self_attention_B = np.array([0, 2366740, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900])
feed_forward_B = np.array([4200448, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432])

n_systems = len(systems)
n_layers = len(layers)

# 新参数
bar_width = 2.7     # 单个柱子的高度
group_gap = 1       # 层与层之间的空行数

# y坐标映射，每一层有三组，层与层之间空group_gap行
y_pos = []
for i in range(n_layers):
    start = i * (n_systems + group_gap)
    y_pos.append([start + j for j in range(n_systems)])

# 展开成所有实际y坐标（方便画分隔线用）
all_y_pos = [item for sublist in y_pos for item in sublist]
max_y = all_y_pos[-1] + 1

# 画布
plt.figure(figsize=(13, 10))
colors = ['#e31a1c', '#1f9e89','#4575b4', '#fee08b']
gray = "#bdbdbd"

for sys_idx, sys in enumerate(systems):
    ffn_percent = []
    sa_percent = []
    fe_percent = sys['fe'] / feature_extractor_B * 100 if feature_extractor_B > 0 else 0
    ffn_percent.append(fe_percent)
    sa_percent.append(0)  # FeatureExtractor无SA
    for i in range(1, n_layers):
        ffn_p = sys['ffn'][i] / feed_forward_B[i] * 100 if feed_forward_B[i] > 0 else 0
        sa_p = sys['sa'][i] / self_attention_B[i] * 100 if self_attention_B[i] > 0 else 0
        ffn_percent.append(ffn_p)
        sa_percent.append(sa_p)
    # 反转顺序：自下而上
    ffn_percent = np.array(ffn_percent[::-1])
    sa_percent = np.array(sa_percent[::-1])

    # 按新y坐标画
    for layer_idx in range(n_layers):
        ypos = y_pos[layer_idx][sys_idx]
        # FFN左
        b1 = plt.barh(
            ypos, -ffn_percent[layer_idx], height=bar_width/n_systems, color=colors[sys_idx], edgecolor=gray, alpha=0.72
        )
        # SA右
        b2 = plt.barh(
            ypos, sa_percent[layer_idx], height=bar_width/n_systems, color=colors[sys_idx], edgecolor=gray, alpha=0.7
        )
        # 数字标注
        # if abs(ffn_percent[layer_idx]) > 0:
        #     if abs(ffn_percent[layer_idx]) < 10:
        #         # 放在外侧
        #         plt.text(-ffn_percent[layer_idx] - 1, ypos, f'{ffn_percent[layer_idx]:.1f}%', 
        #                 ha='right', va='center', color='black', fontsize=8)
        #     else:
        #         # 柱子中央
        #         plt.text(-ffn_percent[layer_idx]/2, ypos, f'{ffn_percent[layer_idx]:.1f}%', 
        #                 ha='center', va='center', color='black', fontsize=8)
        # # SA标签
        # if abs(sa_percent[layer_idx]) > 0:
        #     if abs(sa_percent[layer_idx]) < 10:
        #         plt.text(sa_percent[layer_idx] + 1, ypos, f'{sa_percent[layer_idx]:.1f}%', 
        #                 ha='left', va='center', color='black', fontsize=8)
        #     else:
        #         plt.text(sa_percent[layer_idx]/2, ypos, f'{sa_percent[layer_idx]:.1f}%', 
        #                 ha='center', va='center', color='black', fontsize=8)
# y轴标签：每组中间那条放label，其余空
yticks = [y_pos[i][n_systems//2] for i in range(n_layers)]
plt.yticks(yticks, layer_labels)

# x轴
xticks = np.arange(-100, 101, 25)
xticklabels = [str(abs(x)) for x in xticks]
plt.xticks(xticks, xticklabels)
plt.xlabel('Parameter ratio relative to unpruned WavLM Base (%)')

# X轴上下标题
plt.text(-50, max_y+0.5, "FeedForward", ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(50, max_y+0.5, "SelfAttention", ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.xlim(-105, 105)
plt.ylim(-1, max_y+2)
plt.axvline(0, color='black', linewidth=1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout(rect=[0, 0, 0.82, 1])

# 图例
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=colors[2], label='ASD-ASVSpoof5', edgecolor='black', alpha=1),
    Patch(facecolor=colors[3], label='ASD-SpoofCeleb', edgecolor='black', alpha=1),
    Patch(facecolor=colors[0], label='SV-CNCeleb', edgecolor='black', alpha=1),
    Patch(facecolor=colors[1], label='SV-VoxCeleb', edgecolor='black', alpha=1),
]
plt.legend(
    handles=legend_handles,
    frameon=False, 
    ncol=2,  # Changed to 2 columns
    loc='upper right', 
    bbox_to_anchor=(0.95, 0.95), # Adjusted bbox slightly for better spacing
    columnspacing=1.0 # Adjust spacing between columns
)

plt.savefig("./layerwise_population_pyramid_legend_2col.png", bbox_inches='tight', dpi=300)
plt.show()