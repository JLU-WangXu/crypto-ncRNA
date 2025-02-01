# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Construct simulated CSV data
data = {
    'Data Length': [50, 100, 1000, 100000, 1000000],
    'Total Runs': [10, 10, 10, 10, 10],
    'Encryption Success': [10, 10, 10, 10, 10],
    'Encryption Failure': [0, 0, 0, 0, 0],
    'Decryption Success': [10, 10, 10, 10, 10],
    'Decryption Failure': [0, 0, 0, 0, 0],
    'Encryption Success Rate (%)': [100.00, 100.00, 100.00, 100.00, 100.00],
    'Decryption Success Rate (%)': [100.00, 100.00, 100.00, 100.00, 100.00]
}
df = pd.DataFrame(data)

# Convert data to long format for plotting
df_melted = df.melt(
    id_vars=['Data Length'],
    value_vars=['Encryption Success Rate (%)', 'Decryption Success Rate (%)'],
    var_name='Type',
    value_name='Success Rate (%)'
)

# Set Seaborn theme
sns.set(style="whitegrid")
# Use the viridis palette for scientific coloring
scientific_palette = sns.color_palette("viridis", n_colors=2)

# Create a figure, set the size to 8 x 5 inches
plt.figure(figsize=(6, 4))

# Draw a bar plot: x-axis is data length, y-axis is success rate, different types are distinguished by color
ax = sns.barplot(
    x='Data Length',
    y='Success Rate (%)',
    hue='Type',
    data=df_melted,
    palette=scientific_palette
)

# Add data labels to each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f'{height:.0f}%',
        (p.get_x() + p.get_width() / 2., height),
        ha='center', va='bottom',
        fontsize=10, color='black',
        xytext=(0, 5), textcoords='offset points'
    )

# Adjust chart title and axis labels
plt.title("Encryption and Decryption Success Rates", fontsize=14, fontweight='bold',pad=15)
plt.xlabel("Data Length", fontweight='bold', fontsize=12)
plt.ylabel("Success Rate (%)", fontweight='bold', fontsize=12)
plt.ylim(97, 101)  # Set y-axis limits to focus on the success rate range
plt.yticks(range(97, 101, 1))  # Set y-axis ticks to show every 1% increment
plt.legend(title="Operation Type", loc='lower left')  # Move legend outside the plot
plt.savefig(os.path.join('./results/img/success_result.png'),
           dpi=1200, bbox_inches='tight')
# Adjust layout for better spacing
plt.tight_layout()
plt.show()