import numpy as np
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure
import os
import seaborn as sns
import pandas as pd

current_dir = os.path.dirname(__file__)
axis = 'x'

path = current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Layout_2_runs_modified/Layout_2/Axis_{axis}/Thrust_2e-06/"

seeds = list(range(1, 201))

durations = [6, 12, 18, 24, 30, 36, 42]

data = []

for duration in durations:
    fuel_consumption_lst = []
    for seed in seeds:
        folder_path = path + f"Duration_{duration}/Seed_{seed}/"
        ratio = float(np.loadtxt(folder_path + "PSD_ratio.txt"))
        fuel_consumption = float(np.loadtxt(folder_path + "response_variables.txt")[0])
        data.append({'Duration': duration, 'Seed': seed, 'Ratio': ratio, 'Fuel_consumption': fuel_consumption})
        fuel_consumption_lst.append(fuel_consumption)
    print(f"Duration: {duration}, Mean fuel consumption: {np.mean(fuel_consumption_lst)}")

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Calculate the mean and median PSD ratios for each duration
mean_ratios = df.groupby('Duration')['Ratio'].mean().reset_index()
median_ratios = df.groupby('Duration')['Ratio'].median().reset_index()

# Normalize the mean and median to start at 1 for the smallest duration
mean_ratios['Normalized'] = np.sqrt(mean_ratios['Ratio'] / mean_ratios['Ratio'].iloc[0])
median_ratios['Normalized'] = np.sqrt(median_ratios['Ratio'] / median_ratios['Ratio'].iloc[0])


# Calculate the theoretical sqrt(n) law, starting from 1 for the smallest duration
sqrt_n_values = 1 / np.sqrt((np.array(durations) / durations[0]))

# percentage difference between the theoretical sqrt(n) law and the normalized mean PSD ratios
mean_percentage_diff = 100 * (mean_ratios['Normalized'] - sqrt_n_values) / sqrt_n_values
median_percentage_diff = 100 * (median_ratios['Normalized'] - sqrt_n_values) / sqrt_n_values

print(mean_percentage_diff)
print(median_percentage_diff)

fig, ax = plt.subplots()

# Plot the normalized mean PSD ratios
# ax.plot(mean_ratios['Duration'], mean_ratios['Normalized'], marker='o', linestyle='-', color='b', label='Normalized Mean Ratio')

# Plot the normalized median PSD ratios
ax.plot(median_ratios['Duration'], median_ratios['Normalized'], marker='s', linestyle='--', color='g', label='Normalized Median Ratio')

# Plot the theoretical sqrt(n) law
ax.plot(durations, sqrt_n_values, marker='^', linestyle=':', color='r', label=r'$1 / \sqrt{n}$ Law')

# Add labels and title
ax.set_xlabel('Duration [hrs]', fontsize=14)
ax.set_ylabel('Normalized Ratio', fontsize=14)


# Add grid for better readability
ax.grid(True)

# Show the legend
ax.legend()



# Create a boxplot to visualize the distribution of ratios for each duration
fig1, ax1 = plt.subplots()
sns.boxplot(x='Duration', y='Ratio', data=df, palette="viridis", ax=ax1)
ax1.axhline(1, color='r', linestyle='--', label='Requirement')
ax1.legend()

# Add labels and title
ax1.set_xlabel('Duration [hrs]', fontsize=14)
ax1.set_ylabel('Ratio [-]', fontsize=14)
ax1.set_title('Box plot of Ratio vs Duration', fontsize=16)
ax1.set_yscale('log')
ax1.grid()

fig_hist, ax_hist = plt.subplots()
binwidth = 1 - np.min(df['Ratio'])
sns.histplot(df, x='Ratio', hue='Duration', palette='viridis', stat='percent', binwidth=binwidth, ax=ax_hist, multiple='layer', common_norm=False)

ax_hist.set_xscale('log')
ax_hist.set_xlabel('Ratio [-]', fontsize=16)
ax_hist.set_ylabel('Percentage %', fontsize=16)
# ax_hist.legend(fontsize=14)

save_figure(fig_hist, f"Sensitivity_analysis/One_at_a_time/Boxplots/Layout_2/Axis_{axis}/Duration_hist.svg")
save_figure(fig1, f"Sensitivity_analysis/One_at_a_time/Boxplots/Layout_2/Axis_{axis}/Duration_boxplot_L2_x_200.svg")
save_figure(fig, f"Sensitivity_analysis/One_at_a_time/Boxplots/Layout_2/Axis_{axis}/Duration_normalized.svg")


# plt.show()

