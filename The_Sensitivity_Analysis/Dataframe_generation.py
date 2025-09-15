import numpy as np
import os
import pandas as pd

# Define paths
current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/low_freq/"

# Define parameters
seeds = list(range(1, 31))

unit_labels = {'Thrust': r'$m/s^{{2}}$', 'Duration': 'hrs', 'Frequency': 'Hz', 'Length': 'm',
               'Bandwidth': 'units', 'Layout': '', 'Axis': ''}

sheets_data = {}


nominal_thrust = [2e-6, 4e-6, 6e-6]
df_duration = pd.DataFrame(columns=['Thrust', 'Duration', 'Fuel_consumption', 'Ratio'])
df_frequency = pd.DataFrame(columns=['Thrust', 'Frequency', 'Fuel_consumption', 'Ratio'])
df_length = pd.DataFrame(columns=['Thrust', 'Length', 'Fuel_consumption', 'Ratio'])
df_layout = pd.DataFrame(columns=['Thrust', 'Layout', 'Fuel_consumption', 'Ratio'])
df_bandwidth = pd.DataFrame(columns=['Thrust', 'Bandwidth', 'Fuel_consumption', 'Ratio'])

for th_idx, th in enumerate(nominal_thrust):
    nominal_settings = {'Thrust': th, 'Duration': 24, 'Frequency': 1e-2, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'x'}
    if th_idx == 0:
        thrust_lst = [4e-6, 6e-6]
    elif th_idx == 1:
        thrust_lst = [2e-6, 6e-6]
    elif th_idx == 2:
        thrust_lst = [2e-6, 4e-6]
    else:
        thrust_lst = [2e-6, 4e-6]

    settings = {'Thrust': thrust_lst,
                'Duration': [6, 12, 18],
                'Length': [0.4],
                'Layout': [1, 3]
                }

    for var, var_setting in settings.items():
        if var == 'Thrust':
            continue
        fuel_consumption_seed_lst = []
        ratio_seed_lst = []
        for seed in seeds:
            var_setting_copy = var_setting.copy()
            folder_path = path + f"Thrust_{th}/Seed_{seed}/"
            nominal_folder_path = folder_path + "Nominal/Run_0/"

            nominal_response_variables = np.loadtxt(nominal_folder_path + "response_variables.txt")
            fuel_consumption_nominal = nominal_response_variables[0]
            ratio_nominal = np.loadtxt(nominal_folder_path + "PSD_ratio.txt")

            fuel_consumption_lst = []
            ratio_lst = []

            # Loop over different settings
            for setting_idx in range(len(var_setting_copy)):
                var_folder = folder_path + f"{var}/Run_{setting_idx}/"
                response_variables = np.loadtxt(var_folder + "response_variables.txt")
                ratio = np.loadtxt(var_folder + "PSD_ratio.txt")

                fuel_consumption_lst.append(response_variables[0])
                ratio_lst.append(ratio)

            if var not in ["Axis"]:
                # Append the nominal values
                fuel_consumption_lst.append(fuel_consumption_nominal)
                ratio_lst.append(ratio_nominal)
                var_setting_copy.append(nominal_settings[var])

                # Sort lists by variable setting
                idx = np.argsort(np.array(var_setting_copy))
                var_setting_copy = np.array(var_setting_copy)[idx]
                fuel_consumption_lst = np.array(fuel_consumption_lst)[idx]
                ratio_lst = np.array(ratio_lst)[idx]

            fuel_consumption_seed_lst.append(fuel_consumption_lst)
            ratio_seed_lst.append(ratio_lst)

        # Convert lists to NumPy arrays for easier indexing
        fuel_consumption_seed_lst = np.array(fuel_consumption_seed_lst)
        ratio_seed_lst = np.array(ratio_seed_lst)
        if var == "Duration" and th_idx == 0:
            print(ratio_seed_lst)

        # Compute RMS
        fuel_mean = np.mean(fuel_consumption_seed_lst, axis=0)
        ratio_mean = np.mean(ratio_seed_lst, axis=0)

        for i in range(len(var_setting_copy)):
            if var == 'Duration':
                df_duration.loc[len(df_duration)] = {'Thrust': th, 'Duration': var_setting_copy[i], 'Fuel_consumption': fuel_mean[i], 'Ratio': ratio_mean[i]}
            elif var == 'Frequency':
                df_frequency.loc[len(df_frequency)] = {'Thrust': th, 'Frequency': var_setting_copy[i], 'Fuel_consumption': fuel_mean[i], 'Ratio': ratio_mean[i]}
            elif var == 'Length':
                df_length.loc[len(df_length)] = {'Thrust': th, 'Length': var_setting_copy[i], 'Fuel_consumption': fuel_mean[i], 'Ratio': ratio_mean[i]}
            elif var == 'Layout':
                df_layout.loc[len(df_layout)] = {'Thrust': th, 'Layout': var_setting_copy[i], 'Fuel_consumption': fuel_mean[i], 'Ratio': ratio_mean[i]}
            elif var == 'Bandwidth':
                df_bandwidth.loc[len(df_bandwidth)] = {'Thrust': th, 'Bandwidth': var_setting_copy[i], 'Fuel_consumption': fuel_mean[i], 'Ratio': ratio_mean[i]}


sheets_data['Duration'] = df_duration
sheets_data['Frequency'] = df_frequency
sheets_data['Length'] = df_length
sheets_data['Layout'] = df_layout
sheets_data['Bandwidth'] = df_bandwidth

with pd.ExcelWriter('Mean_values_low_freq.xlsx') as writer:
    for sheet_name, data in sheets_data.items():
        data.to_excel(writer, sheet_name=sheet_name, index=False)


