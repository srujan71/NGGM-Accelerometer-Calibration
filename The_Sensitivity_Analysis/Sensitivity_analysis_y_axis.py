import numpy as np
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure
import os
import pandas as pd

current_dir = os.path.dirname(__file__)

columns = ['Thrust acceleration', 'Setting', 'Setting_value', 'Seed', 'Ratio']

seeds = list(range(1, 31))

data = []
path = current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Freq_0.1_30_seeds/y-axis/Layout_4/"
fig_save_path = f"Sensitivity_analysis/One_at_a_time/Boxplots/30_seeds/Freq_0.1/y_axis/Layout_4/"
# thrusts = [1e-06, 1e-05]
thrusts = [2e-06, 4e-06, 6e-06]
nominal_settings = {'Duration': 24, 'Frequency': 0.1, 'Length': 0.6, 'Layout': 3, 'Axis': 'y'}
# settings = {'Duration': [2, 6, 12, 18, 30, 36, 42, 48],
#             'Frequency': [0.001, 0.01],
#             'Length': [0.2, 0.4, 0.8, 1],
#             'Layout': [1, 2, 3],
#             'Axis': ['x', 'y', 'z']}

settings = {'Axis': ['x', 'z']}


for th_idx, thrust in enumerate(thrusts):
    for var, var_setting in settings.items():
        for seed in seeds:
            nominal_path = path + f"Thrust_{thrust}/Seed_{seed}/Nominal/Run_0/"
            nominal_ratio = np.loadtxt(nominal_path + "PSD_ratio.txt")
            # if var != 'Layout':
            #     nominal_ratio_modified = np.loadtxt(nominal_path + "PSD_ratio_modified.txt")
            # else:
            #     nominal_ratio_modified = 0
            folder_path = path + f"Thrust_{thrust}/Seed_{seed}/{var}/"
            ratio_lst = []
            ratio_lst_modified = []
            var_setting_copy = var_setting.copy()
            for setting_idx in range(len(var_setting)):
                ratio = np.loadtxt(folder_path + f"Run_{setting_idx}/PSD_ratio.txt")
                # if var != 'Layout':
                #     ratio_modified = np.loadtxt(folder_path + f"Run_{setting_idx}/PSD_ratio_modified.txt")
                #     ratio_lst_modified.append(float(ratio_modified))
                # else:
                #     ratio_lst_modified.append(0)
                ratio_lst.append(float(ratio))

            if var != 'Layout':
                ratio_lst.append(float(nominal_ratio))
                # ratio_lst_modified.append(float(nominal_ratio_modified))
                var_setting_copy.append(nominal_settings[var])

                # Sort the values
                var_setting_copy_arr = np.array(var_setting_copy)
                ratio_lst_arr = np.array(ratio_lst)
                # ratio_lst_modified_arr = np.array(ratio_lst_modified)
                sort_idx = np.argsort(var_setting_copy)
                var_setting_copy_arr = var_setting_copy_arr[sort_idx]
                ratio_lst_arr = ratio_lst_arr[sort_idx]
                # ratio_lst_modified_arr = ratio_lst_modified_arr[sort_idx]
            else:
                var_setting_copy_arr = np.array(var_setting_copy)
                ratio_lst_arr = np.array(ratio_lst)
                # ratio_lst_modified_arr = np.array(ratio_lst_modified)

            # Append to the data list
            for setting_value, ratio in zip(var_setting_copy_arr, ratio_lst_arr):
                data.append([thrust, var, setting_value, seed, ratio])


# Create the dataframe
df = pd.DataFrame(data, columns=columns)

# duration_values_to_remove = [2, 36, 42, 48]
# length_values_to_remove = [0.2, 1]
# df_filtered = df[~((df['Setting'] == 'Duration') & (df['Setting_value'].isin(duration_values_to_remove)))]
# df_filtered = df_filtered[~((df_filtered['Setting'] == 'Length') & (df_filtered['Setting_value'].isin(length_values_to_remove)))]

unit_labels = {'Duration': 'hrs', 'Frequency': 'Hz', 'Length': 'm', 'Layout': '', 'Axis': ''}
for setting in df['Setting'].unique():
    fig, ax = plot_setting_boxplots(df, setting, unit_labels)
    save_figure(fig, fig_save_path + f"{setting}_boxplot.svg")
    # save_figure(fig_mod, fig_save_path + f"{setting}_boxplot_modified.svg")







