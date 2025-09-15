import numpy as np
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure
import os
import seaborn as sns
import pandas as pd


def plot_frequency_boxplot(df, figsize=(6, 5), colormap='tab10', yscale='log'):
    """
    Plots a box plot for the frequency setting, with ratio on the y-axis, axis on the x-axis, and hue as frequency.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        figsize (tuple): Figure size (width, height). Default is (8, 6).
        colormap (str): Name of the colormap to use. Default is 'tab10'.
        yscale (str): Scale for the y-axis ('linear' or 'log'). Default is 'log'.
    """
    # Filter the DataFrame for the Frequency setting
    freq_df = df[df['Setting'] == 'Frequency']

    # Create the box plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x='Axis', y='Ratio', hue='Setting_value', data=freq_df, palette=colormap, ax = ax)

    # Customize the plot
    ax.set_title(fr'Ratio vs Axis for Thrust acceleration = {df["Thrust acceleration"][0]} $m/s^{{2}}$')
    ax.set_xlabel('Axis [-]')
    ax.set_ylabel('Ratio [-]')
    ax.set_yscale(yscale)  # Set y-axis scale
    ax.axhline(1, color='red', linestyle='--', label='Requirement')
    ax.legend(title='Frequency (Hz)')
    ax.grid()
    fig.tight_layout()
    # plt.close(fig)
    return fig, ax


current_dir = os.path.dirname(__file__)


seeds = list(range(1, 201))
axes = ['x', 'z']
thrusts = [4e-07, 6e-07, 8e-07, 1e-06]
frequencies = [0.01, 0.1]
durations = [6, 12, 18]

ratio_data = []

for axis in axes:
    path = current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Layout_3_runs/Layout_3/Axis_{axis}/"
    nominal_settings = {'Thrust acceleration': 2e-06, 'Duration': 24, 'Frequency': 0.01, 'Length': 0.6, 'Layout': 3, 'Axis': axis}

    for freq in frequencies:
        for seed in seeds:
            folder_path = path + f"Frequency_{freq}/Seed_{seed}/"
            ratio_freq = float(np.loadtxt(folder_path + "PSD_ratio.txt"))
            ratio_data.append({'Setting': 'Frequency', 'Setting_value': freq, 'Thrust acceleration': 2e-06, 'Ratio': ratio_freq, 'Axis': axis})

    if axis == 'x':
        for thrust in thrusts:
            for duration in durations:
                fuel_consumption_lst = []
                for seed in seeds:
                    folder_path = path + f"Thrust_{thrust}/f_0.01/Duration_{duration}/Seed_{seed}/"
                    ratio_th = float(np.loadtxt(folder_path + "PSD_ratio.txt"))
                    fuel_consumption = float(np.loadtxt(folder_path + "response_variables.txt")[0])
                    ratio_data.append({'Setting': 'Duration', 'Setting_value': duration, 'Thrust acceleration': thrust, 'Ratio': ratio_th, 'Axis': axis})
                    fuel_consumption_lst.append(fuel_consumption)
                print(f"Thrust acceleration: {thrust}, Duration: {duration}, Mean fuel consumption: {np.mean(fuel_consumption_lst)}")

# Convert to dataframe
df = pd.DataFrame(ratio_data)

# Define unit labels for settings
unit_labels = {
    'Frequency': 'Hz',
    'Duration': 'hrs',
    'Thrust acceleration': r'$m/s^2$'
}

fig_freq, ax_freq = plot_frequency_boxplot(df)

# Plot boxplots for Duration setting (only for axis 'x')
df_duration = df[df['Setting'] == 'Duration']
fig_duration, ax_duration = plot_setting_boxplots(df_duration, setting='Duration', unit_labels=unit_labels)
save_figure(fig_freq, "Sensitivity_analysis/One_at_a_time/Boxplots/Layout_3/Frequency_boxplot.svg")
save_figure(fig_duration, "Sensitivity_analysis/One_at_a_time/Boxplots/Layout_3/Duration_boxplot_L3_200.svg")
plt.show()

