import matplotlib.pyplot as plt
import numpy as np

# Metrics for the laptop and smartwatch models
metrics_laptop = {
    'Accuracy': 0.88,
    'Precision': 0.85,
    'Recall': 0.83,
    'F1 Score': 0.84
}

metrics_watch = {
    'Accuracy': 0.80,
    'Precision': 0.78,
    'Recall': 0.77,
    'F1 Score': 0.77
}

# Convert the values to percentages
metrics_laptop_values = [v * 100 for v in metrics_laptop.values()]
metrics_watch_values = [v * 100 for v in metrics_watch.values()]

# X-axis labels
metric_names = list(metrics_laptop.keys())
x = np.arange(len(metric_names))  # Label locations

# Width of the bars
width = 0.35  

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Plotting both laptop and smartwatch bars
bars_laptop = ax.bar(x - width/2, metrics_laptop_values, width, label='Laptop', color='blue')
bars_watch = ax.bar(x + width/2, metrics_watch_values, width, label='Smartwatch', color='green')

# Adding labels on top of the bars
for bars in [bars_laptop, bars_watch]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

# Labels, title, and custom x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Percentage (%)')
ax.set_title('Performance Comparison: Laptop vs. Smartwatch')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.legend()

# Set y-axis limit from 0 to 100
ax.set_ylim(0, 100)

# Show the plot
plt.grid(axis='y')
plt.savefig('Plots/laptop_vs_smartwatch_metrics.png')
plt.show()
