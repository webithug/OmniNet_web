import matplotlib.pyplot as plt
import numpy as np

# Data for the bar plot
categories = ['TT1L', 'TT1L_mlm', 'TT2L', 'TT2L_mlm', 'TTHad', 'TTHad_mlm', 'ttW_FullHad', 'WJetsToLNu', 'WJetsToQQ', 'ZJetsToQQ']
omninet = [0.46, 0.47, 0.64, 0.63, 0.37, 0.33, 0.23, 1, 0.82, 0.81]
spanet = [0.34, 0.33, 0.58, 0.56, 0.36, 0.18, 0.21, 1, 0.79, 0.79]

# Number of categories
x = np.arange(len(categories))

# Bar width
width = 0.35

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, omninet, width, label='OmniNet', color="orangered")
plt.bar(x + width/2, spanet, width, label='SPANet', color = "dodgerblue")

# Add labels, title, and legend
plt.xlabel('Process')
plt.ylabel('Purity')
plt.title('Single process: OmniNet vs. SPANet')
plt.xticks(x, categories, rotation=30)  # Set x-axis labels
plt.legend()

# Show the plot
plt.savefig("/global/homes/w/weipow/OmniNet_web/output_web/all_process_bar_1115.png")

mean_omninet = np.mean(omninet)
mean_spanet = np.mean(spanet)

print(f"Mean Purity - OmniNet: {mean_omninet}, SPANet: {mean_spanet}")
