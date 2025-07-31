import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Data Preparation ---
# Layer labels
layers = ['CNN'] + [f'L {i+1:02}' for i in range(12)]
layer_labels = layers[::-1]  # For pyramid style (bottom to top)

# All systems data
all_systems = {
    'SV-CNCeleb': {
        'ffn': np.array([ 2560224, 3923192, 3663439, 3552775, 3600422, 3260745, 2329323, 1370235, 604809, 0, 0, 0, 128339 ]),
        'sa':  np.array([0, 1973140, 2362900, 2362900, 2362900, 2362900, 2166100, 2362900, 1378900, 0, 0, 0, 0]),
        'fe':  2560224,
        'color': '#e31a1c' # Red
    },
    'SV-VoxCeleb': {
        'ffn': np.array([1712952, 3389853, 2999455, 2996381, 3259208,3188506, 2965641, 2137198, 1285700,670900, 437276, 0, 203652]),
        'sa':  np.array([ 0, 989140, 2362900, 2362900, 2166100, 2362900, 1969300, 1969300, 1772500, 788500, 394900, 0, 0]),
        'fe':  1712952,
        'color': '#1f9e89' # Green
    },
    'Antispoofing-ASVSpoof5': {
        'ffn': np.array([1939113, 3297633, 3017899, 3251523, 0, 3664976, 3439037, 2650556, 0, 358889, 0, 103747, 54563]),
        'sa':  np.array([0, 2366740, 2362900, 2166100, 2166100, 2362900, 2362900, 2362900, 1969300, 1575700, 591700, 198100, 0]),
        'fe':  1939113,
        'color': '#4575b4' # Blue
    },
    'ASD-SpoofCeleb': {
        'ffn': np.array([0, 3067083, 2836533, 3188506, 0, 3651143, 3457481, 2610594, 1082816, 217485, 0, 97599, 45341]),
        'sa':  np.array([0, 2366740, 2166100, 2362900, 0, 2362900, 2362900, 2362900, 2362900, 1772500, 788500, 394900, 0]),
        'fe':  1875942,
        'color': '#fee08b' # Yellow
    }
}

# Unpruned WavLM Base model reference
feature_extractor_B = 4200448
self_attention_B = np.array([0, 2366740, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900, 2362900])
feed_forward_B = np.array([4200448, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432, 4722432])

# --- Plotting Function ---
def plot_pyramid(ax, systems_to_plot):
    n_systems = len(systems_to_plot)
    n_layers = len(layers)
    bar_width = 2.7
    group_gap = 1.5
    gray = "#bdbdbd"

    y_pos = []
    for i in range(n_layers):
        start = i * (n_systems + group_gap)
        y_pos.append([start + j for j in range(n_systems)])

    all_y_pos = [item for sublist in y_pos for item in sublist]
    max_y = all_y_pos[-1] + 1

    for sys_idx, sys_name in enumerate(systems_to_plot):
        sys = all_systems[sys_name]
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
            ypos = y_pos[layer_idx][sys_idx]
            ax.barh(ypos, -ffn_percent[layer_idx], height=bar_width/n_systems, color=sys['color'], edgecolor=gray, alpha=0.8)
            ax.barh(ypos, sa_percent[layer_idx], height=bar_width/n_systems, color=sys['color'], edgecolor=gray, alpha=0.8)
    
    yticks = [np.mean(y_pos[i]) for i in range(len(y_pos))]
    ax.set_yticks(yticks)
    
    if ax.get_subplotspec().is_first_col():
        ax.set_yticklabels(layer_labels, fontsize=18) # Increased font size
    
    xticks = np.arange(-100, 101, 25)
    xticklabels = [str(abs(x)) for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=20) # Increased font size
    
    ax.set_xlim(-105, 105)
    ax.set_ylim(-2, max_y + 2)
    ax.axvline(0, color='black', linewidth=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # --- ADD LEGEND INSIDE SUBPLOT ---
    legend_handles = []
    for sys_name in systems_to_plot:
        sys = all_systems[sys_name]
        legend_handles.append(Patch(facecolor=sys['color'], label=sys_name, edgecolor='black', alpha=1))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=18, frameon=False, facecolor='white', framealpha=0.7, bbox_to_anchor=(0.90, 0.99))
    
    return max_y

# --- Create Figure with 1x2 Subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
fig.subplots_adjust(wspace=0.05)

# --- Subplot (a) on the Left: Task-Specificity ---
systems_a = ['SV-VoxCeleb', 'Antispoofing-ASVSpoof5']
max_y_a = plot_pyramid(ax1, systems_a)
ax1.set_title('(a) Task-Specific Pruning Patterns (SV vs. Antispoofing)', y=-0.1, fontsize=20)
ax1.set_xlabel('Parameter ratio relative to unpruned WavLM Base (%)', fontsize=18)
ax1.text(-50, max_y_a + 1.5, "FFN", ha='center', va='bottom', fontsize=18, fontweight='bold')
ax1.text(50, max_y_a + 1.5, "MHSA", ha='center', va='bottom', fontsize=18, fontweight='bold')


# --- Subplot (b) on the Right: Domain-Specificity ---
systems_b = ['SV-VoxCeleb', 'SV-CNCeleb']
max_y_b = plot_pyramid(ax2, systems_b)
ax2.set_title('(b) Domain-Specific Pruning Patterns (VoxCeleb vs. CNCeleb)', y=-0.1, fontsize=20)
ax2.set_xlabel('Parameter ratio relative to unpruned WavLM Base (%)', fontsize=18)
ax2.text(-50, max_y_b + 1.5, "FFN", ha='center', va='bottom', fontsize=18, fontweight='bold')
ax2.text(50, max_y_b + 1.5, "MHSA", ha='center', va='bottom', fontsize=18, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("./layerwise_pyramid_subplots_internal_legend.png", dpi=300, bbox_inches='tight')
plt.show()
