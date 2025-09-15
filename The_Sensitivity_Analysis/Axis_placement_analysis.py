import numpy as np
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure
import os
import pandas as pd

current_dir = os.path.dirname(__file__)

columns = ['Thrust acceleration', 'Setting', 'Setting_value', 'Seed', 'Ratio']

seeds = list(range(1, 31))

data = []
path = current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Freq_0.1_30_seeds/y-axis/"
fig_save_path = f"Sensitivity_analysis/One_at_a_time/Boxplots/30_seeds/Freq_0.1/y_axis/"
thrusts = [2e-06, 4e-06, 6e-06]
# thrusts = [1e-06, 1e-05]
nominal_settings = {'Duration': 24, 'Frequency': 0.1, 'Length': 0.6, 'Layout': 2, 'Axis': 'y'}
settings = {'Axis': ['x', 'z']}

for th_idx, thrust in enumerate(thrusts):
    for var, var_setting in settings.items():
        for seed in seeds:
            nominal_path = path + f"Thrust_{thrust}/Seed_{seed}/Nominal/Run_0/"
            nominal_ratio = np.loadtxt(nominal_path + "PSD_ratio.txt")
            folder_path = path + f"Thrust_{thrust}/Seed_{seed}/{var}/"
            ratio_lst = []
            var_setting_copy = var_setting.copy()
            for setting_idx in range(len(var_setting)):
                ratio = np.loadtxt(folder_path + f"Run_{setting_idx}/PSD_ratio.txt")
                ratio_lst.append(float(ratio))

            ratio_lst.append(float(nominal_ratio))
            var_setting_copy.append(nominal_settings[var])

            # Sort the values
            var_setting_copy_arr = np.array(var_setting_copy)
            ratio_lst_arr = np.array(ratio_lst)

            sort_idx = np.argsort(var_setting_copy)
            var_setting_copy_arr = var_setting_copy_arr[sort_idx]
            ratio_lst_arr = ratio_lst_arr[sort_idx]


            # Append to the data list
            for setting_value, ratio in zip(var_setting_copy_arr, ratio_lst_arr):
                data.append([thrust, var, setting_value, seed, ratio])


# Create the dataframe
df = pd.DataFrame(data, columns=columns)
df_sorted = df.sort_values(by=['Thrust acceleration', 'Setting', 'Setting_value', 'Ratio'])

# duration_values_to_remove = [2, 36, 42, 48]
# length_values_to_remove = [0.2, 1]
# df_filtered = df[~((df['Setting'] == 'Duration') & (df['Setting_value'].isin(duration_values_to_remove)))]
# df_filtered = df_filtered[~((df_filtered['Setting'] == 'Length') & (df_filtered['Setting_value'].isin(length_values_to_remove)))]

unit_labels = {'Axis': ''}
for setting in df['Setting'].unique():
    fig, ax = plot_setting_boxplots(df, setting, unit_labels)
    # plt.show()
    save_figure(fig, fig_save_path + f"{setting}_boxplot.svg")








