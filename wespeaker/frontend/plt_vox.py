import matplotlib.pyplot as plt
import numpy as np

# Data
# x-axis data: Number of parameters, converted to millions (M)
params = [95631530, 86246262, 76738995, 67251224, 57916106, 48444596, 39011330, 29567034, 20129817, 10554866]
params_m = [p / 1e6 for p in params]

# y-axis data: EER values
eer_vox1_o = [0.761, 0.702, 0.739, 0.675, 0.745, 0.76, 0.771, 0.803, 1.005, 1.596]
eer_vox1_e = [0.753, 0.793, 0.799, 0.791, 0.824, 0.824, 0.851, 0.913, 1.109, 1.785]
eer_vox1_h = [1.512, 1.561, 1.572, 1.538, 1.613, 1.592, 1.639, 1.757, 2.071, 3.028]
eer_cnceleb = [10.911, 10.724, 10.897, 10.885, 10.784, 10.893, 10.634, 11.028, 11.148, 12.351]

# Colors
colors = ['#4575b4', '#e08214', '#7fbf7b', '#d73027']

# Create two subplots for the broken y-axis, sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
fig.subplots_adjust(hspace=0.1)

# Plot data on both axes using parameter count for the x-axis
for ax in [ax1, ax2]:
    ax.plot(params_m, eer_vox1_o, marker='o', color=colors[0], label='Vox1-O')
    ax.plot(params_m, eer_vox1_e, marker='s', color=colors[1], label='Vox1-E')
    ax.plot(params_m, eer_vox1_h, marker='^', color=colors[2], label='Vox1-H')
    ax.plot(params_m, eer_cnceleb, marker='d', color=colors[3], label='CNCeleb-Eval')
    ax.grid(alpha=0.3, linestyle='--')

# Set the y-axis limits for each subplot to create the "break"
ax1.set_ylim(10.5, 12.5)  # Upper part of the plot
ax2.set_ylim(0.6, 3.2)   # Lower part of the plot

# Reverse the x-axis to show performance as model size decreases
ax2.set_xlim(max(params_m) + 5, min(params_m) - 5)


# Hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# Add the diagonal "cut" marks
d = .015
kwargs = dict(transform=ax1.transAxes, color='black', clip_on=False)
ax1.plot((-d, +d), (-d*2, +d*2), **kwargs)
ax1.plot((1 - d, 1 + d), (-d*2, +d*2), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)


# Annotations for the lower plot (ax2)
y_offset_low = 0.05
for x, y in zip(params_m, eer_vox1_o):
    ax2.text(x, y + y_offset_low, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
for x, y in zip(params_m, eer_vox1_e):
    ax2.text(x, y + y_offset_low, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
for x, y in zip(params_m, eer_vox1_h):
    ax2.text(x, y + y_offset_low, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

# Annotations for the upper plot (ax1)
y_offset_high = 0.05
for x, y in zip(params_m, eer_cnceleb):
    ax1.text(x, y + y_offset_high, f"{y:.2f}", ha='center', va='bottom', fontsize=8)


# Set labels and title
ax2.set_xlabel('Number of Parameters (M)', fontsize=12)
fig.text(0.04, 0.5, 'EER (%)', va='center', rotation='vertical', fontsize=12)
# ax1.set_title('EER vs. Number of Parameters', fontsize=14)

# Add legend to the top plot
ax1.legend(loc='upper left')

# Save the figure
plt.savefig("eer_vs_params_broken_axis.png", dpi=300, facecolor='white', bbox_inches='tight')
plt.show()