import numpy as np
from Tools.PlotTools import create_plots, save_figure, plt
import os



# Define paths
current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Freq_0.01_30_seeds/"
fig_save_path = f"Sensitivity_analysis/One_at_a_time/Seed_analysis/Freq_0.01_30_seeds/"

# Define parameters
seeds = list(range(1, 51))
cmap = plt.get_cmap('brg')

unit_labels = {'Thrust': r'$m/s^{{2}}$', 'Duration': 'hrs', 'Frequency': 'Hz', 'Length': 'm',
               'Bandwidth': 'units', 'Layout': '', 'Axis': ''}

nominal_thrust = [2e-6, 4e-6, 6e-6]

for th_idx, th in enumerate(nominal_thrust):
    nominal_settings = {'Thrust': th, 'Duration': 24, 'Frequency': 1e-2, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'x'}

    settings = {'Duration': [6, 12, 18],
                'Length': [0.4, 0.8],
                'Layout': [1, 3],
                }
    thrust_ratio = []
    for var, var_setting in settings.items():
        fuel_consumption_seed_lst = []
        ratio_seed_lst = []

        fig1, ax1 = create_plots("Seed [-]", "Ratio [-]", False, False, figsize=(13, 10))

        for seed in seeds:
            var_setting_copy = var_setting.copy()
            folder_path = path + f"Thrust_{th}/Seed_{seed}/"
            nominal_folder_path = folder_path + "Nominal/Run_0/"

            nominal_response_variables = np.loadtxt(nominal_folder_path + "response_variables.txt")
            fuel_consumption_nominal = nominal_response_variables[0]
            ratio_nominal = np.loadtxt(nominal_folder_path + "PSD_ratio.txt")

            if var == "Duration":
                thrust_ratio.append(ratio_nominal)

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

        # Compute RMS
        fuel_rms = np.sqrt(np.mean(fuel_consumption_seed_lst ** 2, axis=0))
        ratio_rms = np.sqrt(np.mean(ratio_seed_lst ** 2, axis=0))

        # Define color map
        colors = cmap(np.linspace(0, 1, len(var_setting_copy)))
        colors[:, 3] = 0.8  # Set transparency

        # Connect points across seeds
        for i in range(len(var_setting_copy)):
            fuel_points = [fuel_consumption_seed_lst[seed_idx][i] for seed_idx in range(len(seeds))]
            ratio_points = [ratio_seed_lst[seed_idx][i] for seed_idx in range(len(seeds))]

            # Scatter plot for individual points
            ax1.scatter(seeds, ratio_points, color=colors[i], label=f"{var}: {var_setting_copy[i]} {unit_labels[var]}")

            # Line plot to connect points
            ax1.plot(seeds, ratio_points, color=colors[i], alpha=0.8)

        ax1.legend(fontsize=16)

        # Compute Progressive RMS
        progressive_fuel_rms = []
        progressive_ratio_rms = []
        included_seeds = np.arange(1, len(seeds) + 1)  # Number of included seeds
        progressive_fuel_median = []
        progressive_ratio_median = []

        for i in range(1, len(seeds) + 1):
            fuel_rms_progress = (np.mean(fuel_consumption_seed_lst[:i], axis=0))
            ratio_rms_progress = (np.mean(ratio_seed_lst[:i], axis=0))

            fuel_median = np.median(fuel_consumption_seed_lst[:i], axis=0)
            ratio_median = np.median(ratio_seed_lst[:i], axis=0)

            progressive_fuel_rms.append(fuel_rms_progress)
            progressive_ratio_rms.append(ratio_rms_progress)

            progressive_fuel_median.append(fuel_median)
            progressive_ratio_median.append(ratio_median)

        # Convert to numpy arrays for plotting
        progressive_fuel_rms = np.array(progressive_fuel_rms)
        progressive_ratio_rms = np.array(progressive_ratio_rms)
        progressive_fuel_median = np.array(progressive_fuel_median)
        progressive_ratio_median = np.array(progressive_ratio_median)

        # Create Mean vs. Included Seeds Plots
        fig5, ax5 = create_plots("Number of Included Seeds", "Mean Ratio [-]", False, False, figsize=(13, 10))
        fig7, ax7 = create_plots("Number of Included Seeds", "Median Ratio [-]", False, True, figsize=(13, 10))

        # Plot Progressive RMS
        for i in range(len(var_setting_copy)):
            ax5.plot(included_seeds, progressive_ratio_rms[:, i], marker='o', linestyle='-', color=colors[i], label=f"{var}: {var_setting_copy[i]} {unit_labels[var]}")
            ax7.plot(included_seeds, progressive_ratio_median[:, i], marker='o', linestyle='-', color=colors[i], label=f"{var}: {var_setting_copy[i]} {unit_labels[var]}")

        ax5.legend(fontsize=16)
        ax7.legend(fontsize=16)

        save_figure(fig1, fig_save_path + f"Thrust_{th}/Ratio_{var}.png")
        save_figure(fig5, fig_save_path + f"Thrust_{th}/Mean_Ratio_{var}.png")
        save_figure(fig7, fig_save_path + f"Thrust_{th}/Median_Ratio_{var}.png")


    thrust_ratio = np.array(thrust_ratio)
    np.savetxt(f"Thrust_ratio_{th}.txt", thrust_ratio)