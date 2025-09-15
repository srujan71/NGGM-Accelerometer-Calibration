import numpy as np
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure
import os
import seaborn as sns
import pandas as pd

# Define paths and parameters
current_dir = os.path.dirname(__file__)
axis = 'x'
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Frequency_analysis/"
seeds = list(range(1, 101))
frequencies = [0.1, 0.01]

# Collect data
data = []
for freq in frequencies:
    for seed in seeds:
        # Load scaled data (for all frequencies, including freq == 0.01)
        scaled_folder_path = path + f"f_{freq}/Seed_{seed}/"
        scaled_ratio = float(np.loadtxt(scaled_folder_path + "PSD_ratio.txt"))
        data.append({'Frequency': freq, 'Seed': seed, 'Ratio': scaled_ratio, 'Scaling': 'Scaled'})
        if freq == 0.1:
            ratio = float(np.loadtxt(scaled_folder_path + "PSD_ratio.txt"))
            data.append({'Frequency': freq, 'Seed': seed, 'Ratio': ratio, 'Scaling': 'Not Scaled'})

        # Load not scaled data (only for freq == 0.01)
        if freq == 0.01:
            not_scaled_folder_path = path + f"wo_factor/f_{freq}/Seed_{seed}/"
            not_scaled_ratio = float(np.loadtxt(not_scaled_folder_path + "PSD_ratio.txt"))
            data.append({'Frequency': freq, 'Seed': seed, 'Ratio': not_scaled_ratio, 'Scaling': 'Not Scaled'})


# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)


fig, ax = plt.subplots()
sns.boxplot(df, x='Scaling', y='Ratio', hue='Frequency', palette="tab10", ax=ax)
ax.hlines(1, -0.5, 1.5, colors='r', linestyles='--', label='Requirement')
ax.set_xlabel('Thrust Scaling [-]', fontsize=14)
ax.set_ylabel('Ratio [-]', fontsize=14)
ax.set_title('Box plot of Ratio vs Thrust scaling', fontsize=16)
ax.set_yscale('log')
ax.legend(title='Frequency [Hz]')
ax.grid()
fig.tight_layout()

save_figure(fig, "Sensitivity_analysis/One_at_a_time/Boxplots/Thrust_scaling_boxplot.svg")
# plt.show()




# Create histograms for "Scaled" and "Not Scaled" data
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 6))

binwidth_scaled = 1 - min(df[df['Scaling'] == 'Scaled']['Ratio'])
binwidth_not_scaled = 1 - min(df[df['Scaling'] == 'Not Scaled']['Ratio'])
# Histogram for Scaled data
sns.histplot(
    data=df[df['Scaling'] == 'Scaled'],
    x='Ratio',
    hue='Frequency',
    palette="viridis",
    ax=ax2,
    multiple="layer",
    common_norm=False,
    log_scale=False,
    stat='percent',
    binwidth=binwidth_scaled
)
ax2.set_xlabel('Ratio', fontsize=14)
ax2.set_ylabel('Percentage %', fontsize=14)
ax2.set_xscale('log')
ax2.set_title('Histogram of Ratios (Scaled)', fontsize=16)
ax2.grid()

# Histogram for Not Scaled data
sns.histplot(
    data=df[df['Scaling'] == 'Not Scaled'],
    x='Ratio',
    hue='Frequency',
    palette="viridis",
    ax=ax3,
    multiple="layer",
    common_norm=False,
    log_scale=False,
    stat='percent',
    binwidth=binwidth_not_scaled
)
ax3.set_xlabel('Ratio', fontsize=14)
ax3.set_ylabel('Percentage %', fontsize=14)
ax3.set_xscale('log')
ax3.set_title('Histogram of Ratios (Not Scaled)', fontsize=16)
ax3.grid()

# Adjust layout
fig2.tight_layout()



# Show the plot
# plt.show()