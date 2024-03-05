import json
import matplotlib.pyplot as plt
import numpy as np

# List of model identifiers
# models_identifiers = ['n', 's', 'm', 'l', 'x']
# files = [f"../output/results_yolov8{n}.pt.json" for n in models_identifiers]
models_identifiers = ['yolov8m', 'best_all']
files = [f"../output/results_{n}.pt.json" for n in models_identifiers]

# Initialize lists to hold data
models = []
map50s = []
map70s = []
total_times = []
aps_50 = []
aps_70 = []

# Load data from each file
for file in files:
    with open(file, 'r') as f:
        data = json.load(f)
        model = data['params']['model']
        models.append(model)
        map50s.append(data['results']['mAP50'])
        map70s.append(data['results']['mAP70'])
        total_times.append(data['results']['total_time'])
        aps_50.append(data['results']['aps_50'])
        aps_70.append(data['results']['aps_70'])

# Create a model_pretty list for plotting
model_pretty = ['Medium', 'Trained medium']
for i, model in enumerate(models):
    models[i] = model_pretty[i] + ' YOLOv8'

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Improved color palette
colors_map = ['#FFA07A', '#FA8072']  # LightSalmon, Salmon
colors_time = ['#2E8B57']  # SeaGreen

# Bar plots for mAP50 and mAP70
indices = np.arange(len(models))
width = 0.2
bars1 = ax1.bar(indices - width, map50s, width, label='mAP50', color=colors_map[0], edgecolor='gray')
bars2 = ax1.bar(indices, map70s, width, label='mAP70', color=colors_map[1], edgecolor='gray')

# Set the y-axis for mAP scores
ax1.set_ylabel('mAP Scores', fontsize=12)
ax1.set_title('Model Performance and Inference Time', fontsize=16, fontweight='bold')
ax1.set_xticks(indices)
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=10)  # Rotate for better label readability

# Create a second y-axis for total time
ax2 = ax1.twinx()
bars3 = ax2.bar(indices + width, total_times, width, label='Total Time (s)', color=colors_time[0], hatch='//', edgecolor='white')
ax2.set_ylabel('Total Time (s)', fontsize=12)
ax2.tick_params(axis='y')

# Place the legend outside the plot area
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), title='Metrics')
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.85), title='Time')

# Adding data labels
def add_labels(bars, axis):
    for bar in bars:
        height = bar.get_height()
        axis.text(bar.get_x() + bar.get_width()/2., 1.02*height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

add_labels(bars1, ax1)
add_labels(bars2, ax1)
add_labels(bars3, ax2)

ax1.set_ylim(0, 1.1 * max(max(map50s), max(map70s)))
ax2.set_ylim(0, 1.1 * max(total_times))
# Adjust subplot params for better fit and to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.2, right=0.8)
# Save the plot
plt.savefig('../output/model_performance_improvement.png', bbox_inches='tight')

# Define shades of red for different models
reds = plt.cm.Reds(np.linspace(0.5, 1, len(models)))

# Define your model_pretty and aps_50, aps_70 lists here
# model_pretty = ['Nano', 'Small', 'Medium', 'Large', 'X-Large']
# aps_50 = [...]
# aps_70 = [...]

# Plot for AP50 evolution
plt.figure(figsize=(12, 6))
for i, ap in enumerate(aps_50):
    plt.plot(ap, label=f'{model_pretty[i]} AP50', color=reds[i], linewidth=2)
plt.xlabel('Frame', fontsize=12)
plt.ylabel('AP50', fontsize=12)
plt.title('AP50 Evolution During Frames for Each Model', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Model')
plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right margin to fit the legend
# Save the plot
plt.savefig('../output/ap50_evolution_improvement.png', bbox_inches='tight')

# Plot for AP70 evolution
plt.figure(figsize=(12, 6))
for i, ap in enumerate(aps_70):
    plt.plot(ap, label=f'{model_pretty[i]} AP70', color=reds[i], linewidth=2)
plt.xlabel('Frame', fontsize=12)
plt.ylabel('AP70', fontsize=12)
plt.title('AP70 Evolution During Frames for Each Model', fontsize=16, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Model')
plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right margin to fit the legend
# Save the plot
plt.savefig('../output/ap70_evolution_improvement.png', bbox_inches='tight')

plt.show()