import matplotlib.pyplot as plt
import numpy as np

# --- Data Preparation ---
# Data for WavLM Base+ (converted to NumPy arrays)
params_base_m = np.array([95.63, 86.25, 76.74, 67.25, 57.92, 48.44, 39.01, 29.57, 20.13, 10.55])
eer_base_o = np.array([0.761, 0.702, 0.739, 0.675, 0.745, 0.760, 0.771, 0.803, 1.005, 1.596])
eer_base_e = np.array([0.753, 0.793, 0.799, 0.791, 0.824, 0.824, 0.851, 0.913, 1.109, 1.785])
eer_base_h = np.array([1.512, 1.561, 1.572, 1.538, 1.613, 1.592, 1.639, 1.757, 2.071, 3.028])

# Data for WavLM Large (converted to NumPy arrays)
params_large_m = np.array([316.87, 286.38, 254.70, 223.01, 191.32, 159.63, 127.95, 96.26, 64.57, 32.89])
eer_large_o = np.array([0.633, 0.718, 0.686, 0.702, 1.154, 1.319, 1.489, 2.032, 2.447, 7.435])
eer_large_e = np.array([0.654, 0.664, 0.661, 0.716, 1.235, 1.358, 1.554, 2.104, 2.394, 7.091])
eer_large_h = np.array([1.348, 1.353, 1.349, 1.424, 2.542, 2.754, 3.107, 4.051, 4.026, 10.143])

# Group data for easier plotting
plot_data = {
    'Vox1-O': {'base': eer_base_o, 'large': eer_large_o, 'ylim': [0.5, 8.0]},
    'Vox1-E': {'base': eer_base_e, 'large': eer_large_e, 'ylim': [0.5, 8.0]},
    'Vox1-H': {'base': eer_base_h, 'large': eer_large_h, 'ylim': [1.0, 11.0]}
}

# --- Plotting ---
plt.style.use('seaborn-v0_8-paper')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

base_color = '#0072B2'
large_color = '#D55E00'

for i, (ax, (title, data)) in enumerate(zip(axes, plot_data.items())):
    # Plot Base+ and Large model data
    ax.plot(params_base_m, data['base'], color=base_color, linestyle='-', marker='o', label='WavLM Base Plus')
    ax.plot(params_large_m, data['large'], color=large_color, linestyle='--', marker='s', label='WavLM Large')
    
    # --- Insightful Annotations ---
    # Add vertical line for unpruned Base+ model size
    unpruned_base_params = 95.63
    # Only add the label for the first subplot to avoid duplicate legend entries
    vline_label = 'Unpruned Base Plus Size' if i == 0 else None
    ax.axvline(x=unpruned_base_params, color='gray', linestyle=':', linewidth=2, label=vline_label)
    
    # Highlight the "sweet spot" where pruned Large is better than unpruned Base
    unpruned_base_eer = data['base'][0]
    ax.axhline(y=unpruned_base_eer, color=base_color, linestyle=':', linewidth=1)
    ax.fill_between(params_large_m, data['large'], unpruned_base_eer, 
                    where=(data['large'] < unpruned_base_eer), 
                    color='green', alpha=0.2, interpolate=True)

    # Formatting
    ax.set_title(title, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(data['ylim'])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Reverse the shared x-axis
axes[0].set_xlim(325, 0)

# Shared and individual labels
fig.text(0.5, -0.04, 'Number of Parameters (M)', ha='center', fontsize=16)
axes[0].set_ylabel('EER (%)', fontsize=16)

# --- Create the legend only in the first subplot ---
axes[0].legend(loc='upper left', fontsize=14, frameon=True, facecolor='white', framealpha=0.8)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("model_comparison_optimized.png", dpi=300, bbox_inches='tight')
plt.show()
