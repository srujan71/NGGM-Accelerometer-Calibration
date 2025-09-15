import numpy as np
from Tools.PlotTools import create_plots, save_figure, plt
import os


def filter_iqr(data):
    """Returns mask of inlier indices based on the IQR method."""
    Q1, Q3 = np.percentile(data, [25, 75], axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data >= lower_bound) & (data <= upper_bound)

def progressive_rms(data):
    """Computes progressive RMS for a 1D array."""
    return np.sqrt(np.cumsum(data**2) / np.arange(1, len(data) + 1))

# Define paths
current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/x_axis/"
fig_save_path = f"Sensitivity_analysis/One_at_a_time/Seed_analysis/Outlier_filtered/x_axis/"

# Define parameters
nominal_thrust = [1e-6, 1e-5]
seeds = list(range(1, 31))
cmap = plt.get_cmap('brg')

unit_labels = {'Thrust': r'$m/s^{{2}}$', 'Duration': 'hrs', 'Frequency': 'Hz', 'Length': 'm',
               'Bandwidth': 'units', 'Layout': '', 'Axis': ''}

nominal_thrust = [2e-6, 4e-6, 6e-6, 8e-6]
for th_idx, th in enumerate(nominal_thrust):
    nominal_settings = {'Thrust': th, 'Duration': 24, 'Frequency': 1e-1, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'x'}
    if th_idx == 0:
        thrust_lst = [4e-6, 6e-6, 8e-6]
    elif th_idx == 1:
        thrust_lst = [2e-6, 6e-6, 8e-6]
    elif th_idx == 2:
        thrust_lst = [2e-6, 4e-6, 8e-6]
    else:
        thrust_lst = [2e-6, 4e-6, 6e-6]

    settings = {'Thrust': thrust_lst,
                'Duration': [6, 12, 18, 30],
                'Frequency': [1e-3, 1e-2],
                'Length': [0.4, 0.8],
                'Layout': [1, 3],
                }

    for var, var_setting in settings.items():
        fuel_consumption_seed_lst = []
        ratio_seed_lst = []

        # Create figures
        if var == "Thrust" or var == "Duration":
            fig, ax = create_plots("Seed [-]", "Fuel consumption [kg]", False, True, figsize=(13, 10))
            fig1, ax1 = create_plots("Seed [-]", "Ratio [-]", False, True, figsize=(13, 10))
        else:
            fig, ax = create_plots("Seed [-]", "Fuel consumption [kg]", False, False, figsize=(13, 10))
            fig1, ax1 = create_plots("Seed [-]", "Ratio [-]", False, False, figsize=(13, 10))

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

        # Convert lists to NumPy arrays
        fuel_consumption_seed_lst = np.array(fuel_consumption_seed_lst)
        ratio_seed_lst = np.array(ratio_seed_lst)

        # **Apply IQR filtering separately for Fuel and Ratio**


        fuel_inlier_mask = filter_iqr(fuel_consumption_seed_lst)
        ratio_inlier_mask = filter_iqr(ratio_seed_lst)

        # Define color map
        colors = cmap(np.linspace(0, 1, len(var_setting_copy)))
        colors[:, 3] = 0.8  # Set transparency

        fig_rms, ax_rms = create_plots("Seed [-]", "Progressive RMS Fuel Consumption", False, False, figsize=(13, 10))
        fig_rms1, ax_rms1 = create_plots("Seed [-]", "Progressive RMS Ratio", False, False, figsize=(13, 10))

        # **Plot fuel data using its mask**
        for i in range(len(var_setting_copy)):
            # Get valid data for fuel
            fuel_points = fuel_consumption_seed_lst[:, i][fuel_inlier_mask[:, i]]
            valid_fuel_seeds = np.array(seeds)[fuel_inlier_mask[:, i]]

            # Scatter plot for Fuel
            ax.scatter(valid_fuel_seeds, fuel_points, color=colors[i], label=f"{var}: {var_setting_copy[i]} {unit_labels[var]}")
            ax.plot(valid_fuel_seeds, fuel_points, color=colors[i], alpha=0.8)

            # Get valid data for ratio
            ratio_points = ratio_seed_lst[:, i][ratio_inlier_mask[:, i]]
            valid_ratio_seeds = np.array(seeds)[ratio_inlier_mask[:, i]]

            # Scatter plot for Ratio
            ax1.scatter(valid_ratio_seeds, ratio_points, color=colors[i], label=f"{var}: {var_setting_copy[i]} {unit_labels[var]}")
            ax1.plot(valid_ratio_seeds, ratio_points, color=colors[i], alpha=0.8)

            if len(fuel_points) > 1:
                rms_fuel = progressive_rms(fuel_points)
                ax_rms.scatter(valid_fuel_seeds[: len(rms_fuel)], rms_fuel, color=colors[i], label=f"{var}: {var_setting_copy[i]} {unit_labels[var]}")
                ax_rms.plot(valid_fuel_seeds[: len(rms_fuel)], rms_fuel, color=colors[i], alpha=0.8)

                # Compute and plot progressive RMS for ratio
            if len(ratio_points) > 1:
                rms_ratio = progressive_rms(ratio_points)
                ax_rms1.scatter(valid_ratio_seeds[: len(rms_ratio)], rms_ratio, color=colors[i], label=f"{var}: {var_setting_copy[i]} {unit_labels[var]}")
                ax_rms1.plot(valid_ratio_seeds[: len(rms_ratio)], rms_ratio, color=colors[i], alpha=0.8)


        ax.legend(fontsize=16)
        ax1.legend(fontsize=16)
        ax_rms.legend(fontsize=16)
        ax_rms1.legend(fontsize=16)

        # Save filtered plots separately
        save_figure(fig, fig_save_path + f"Thrust_{th}/Filtered_Fuel_consumption_{var}.png")
        save_figure(fig1, fig_save_path + f"Thrust_{th}/Filtered_Ratio_{var}.png")
        save_figure(fig_rms, fig_save_path + f"Thrust_{th}/Progressive_RMS_Fuel_{var}.png")
        save_figure(fig_rms1, fig_save_path + f"Thrust_{th}/Progressive_RMS_Ratio_{var}.png")

print("Filtered plots saved successfully!")
