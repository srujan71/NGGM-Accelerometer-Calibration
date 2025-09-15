import numpy as np
from Tools.PlotTools import plt, plot_setting_boxplots, save_figure
import os
import pandas as pd


def calculate_standard_error(group):
    return group['Ratio'].std()

current_dir = os.path.dirname(__file__)

columns = ['Thrust acceleration', 'Setting', 'Setting_value', 'Seed', 'Ratio']

seeds = list(range(1, 31))
frequencies = [0.1, 0.01]

for freq in frequencies:
    data = []
    path = current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Freq_{freq}_30_seeds/"
    fig_save_path = f"Sensitivity_analysis/One_at_a_time/Boxplots/30_seeds/Freq_{freq}/x_axis/"
    thrusts = [2e-06, 4e-06, 6e-06]
    nominal_settings = {'Duration': 24, 'Frequency': freq, 'Length': 0.6, 'Layout': 2, 'Axis': 'x'}
    if freq == 0.1:
        settings = {'Duration': [6, 12, 18],
                    'Length': [0.4, 0.8],
                    'Layout': [1, 3],
                    'Frequency': [0.01]}
    else:
        settings = {'Duration': [6, 12, 18],
                    'Length': [0.4, 0.8],
                    'Layout': [1, 3]}

    for th_idx, thrust in enumerate(thrusts):
        for var, var_setting in settings.items():
            for seed in seeds:
                nominal_path = path + f"Thrust_{thrust}/Seed_{seed}/Nominal/Run_0/"
                nominal_ratio = float(np.loadtxt(nominal_path + "PSD_ratio.txt"))
                folder_path = path + f"Thrust_{thrust}/Seed_{seed}/{var}/"
                ratio_lst = []
                var_setting_copy = var_setting.copy()
                for setting_idx in range(len(var_setting)):
                    if var == 'Frequency':
                        setting_idx += 1
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

    # Filter the DataFrame for 'Duration' setting
    df_duration = df[(df['Setting'] == 'Duration') & (df['Thrust acceleration'] == 2e-06)]

    standard_errors = df_duration.groupby(['Thrust acceleration', 'Setting_value']).apply(calculate_standard_error).reset_index()
    standard_errors.columns = ['Thrust acceleration', 'Setting_value', 'Standard_Error']

    # Print or save the standard errors
    print(standard_errors)

    # Optionally, save the standard errors to a CSV file
    standard_errors.to_csv(f"duration_standard_errors_freq_{freq}.csv", index=False)


    unit_labels = {'Duration': 'hrs', 'Frequency': 'Hz', 'Length': 'm', 'Layout': '', 'Axis': ''}
    for setting in df['Setting'].unique():
        fig, ax = plot_setting_boxplots(df, setting, unit_labels)
        save_figure(fig, fig_save_path + f"{setting}_boxplot.svg")







