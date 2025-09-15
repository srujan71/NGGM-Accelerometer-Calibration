import numpy as np
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure
import os
import seaborn as sns
import pandas as pd
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure



current_dir = os.path.dirname(__file__)
axis = 'y'

path = current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Layout_2_runs_modified/Layout_2/Axis_{axis}/"

seeds = list(range(1, 201))

durations = [24, 30]
thrusts = [4e-06, 6e-06, 8e-06, 1e-05, 2e-05]

data = []

for thrust in thrusts:
    for duration in durations:
        fuel_consumption_lst = []
        for seed in seeds:
            folder_path = path + f"Thrust_{thrust}/f_0.01/Duration_{duration}/Seed_{seed}/"
            ratio = float(np.loadtxt(folder_path + "PSD_ratio.txt"))
            fuel_consumption = float(np.loadtxt(folder_path + "response_variables.txt")[0])
            data.append({'Setting': 'Duration', 'Setting_value': duration, 'Thrust acceleration': thrust, 'Ratio': ratio, 'Axis': axis})
            fuel_consumption_lst.append(fuel_consumption)
        print(f"Thrust acceleration: {thrust}, Duration: {duration}, Mean fuel consumption: {np.mean(fuel_consumption_lst)}")

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Define unit labels for settings
unit_labels = {
    'Frequency': 'Hz',
    'Duration': 'hrs',
    'Thrust acceleration': r'$m/s^2/\sqrt{Hz}$'
}

# Plot boxplots for Duration setting (only for axis 'x')
df_duration = df[df['Setting'] == 'Duration']
fig_duration, ax_duration = plot_setting_boxplots(df_duration, setting='Duration', unit_labels=unit_labels)
save_figure(fig_duration, f"Sensitivity_analysis/One_at_a_time/Boxplots/Layout_2/Axis_{axis}/Duration_boxplot_L2_Y_200.svg")



# plt.show()

