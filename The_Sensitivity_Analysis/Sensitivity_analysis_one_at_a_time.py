import numpy as np
from Tools.PlotTools import create_plots, save_figure, plot_asd
import os
from scipy.fft import rfftfreq
from scipy.stats import norm

current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/x_axis/"

fig_save_path = "Sensitivity_analysis/One_at_a_time/x_axis/"  # The first half of the path is defined already in save figure function

NFFT = 27001
freq_acc = rfftfreq(NFFT, 1)
a_ng_requirement = (5e-12 * np.sqrt(1 + np.where(freq_acc != 0, 0.001 / freq_acc, np.inf) ** 2 + (100 * freq_acc ** 2) ** 2))

nominal_thrust = [1e-6, 1e-5]
seeds = list(range(1, 31))

for th_idx, th in enumerate(nominal_thrust):
    nominal_settings = {'Thrust': th, 'Duration': 24, 'Frequency': 1e-1, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'x'}
    if th_idx == 0:
        settings_th = 1e-5
    else:
        settings_th = 1e-6

    settings = {'Thrust': [1e-8, 1e-7, settings_th, 1e-4],
                'Duration': [2, 6, 12, 18, 30, 36, 42, 48],
                'Frequency': [1e-3, 1e-2],
                'Length': [0.2, 0.4, 0.8, 1],
                'Bandwidth': [2, 6, 8],
                'Layout': [1, 2, 3],
                'Axis': ['x', 'y', 'z']}
    var_units = {'Thrust': fr"[$m/s^{{2}}]]$", 'Duration': "[hrs]", 'Frequency': "[Hz]", 'Length': "[m]", "Bandwidth": "[-]", "Layout": "[-]", "Axis": "[-]"}
    for var, var_setting in settings.items():
        a_ng_seed_lst = []
        fuel_consumption_seed_lst = []
        error_seed_lst = []
        ratio_seed_lst = []

        # Create a figure for the sensitivity analysis
        if var == "Thrust":
            fig, ax = create_plots(f"{var} {var_units[var]}", "Fuel consumption [kg]", True, True, ylabel_color='b')
            ax.set_title(f"Seeds 1-{len(seeds)} averaged", fontsize=18)
            ax2 = ax.twinx()
            ax2.set_ylabel("Ratio [-]", fontsize=16, color='r')
            ax2.set_yscale('log')
        elif var == "Frequency":
            fig, ax = create_plots(f"{var} {var_units[var]}", "Fuel consumption [g]", True, False, ylabel_color='b')
            ax.set_title(f"Seeds 1-{len(seeds)} averaged", fontsize=18)
            ax2 = ax.twinx()
            ax2.set_ylabel("Ratio [-]", fontsize=16, color='r')
        elif var == "Duration":
            fig, ax = create_plots(f"{var} {var_units[var]}", "Fuel consumption [kg]", False, False, ylabel_color='b')
            ax.set_title(f"Seeds 1-{len(seeds)} averaged", fontsize=18)
            ax2 = ax.twinx()
            ax2.set_ylabel("Ratio [-]", fontsize=16, color='r')
            ax2.set_yscale('log')
        else:
            fig, ax = create_plots(f"{var} {var_units[var]}", "Fuel consumption [kg]", False, False, ylabel_color='b')
            ax.set_title(f"Seeds 1-{len(seeds)} averaged", fontsize=18)
            ax2 = ax.twinx()
            ax2.set_ylabel("Ratio [-]", fontsize=16, color='r')

        for seed in seeds:
            var_setting_copy = var_setting.copy()
            fuel_consumption_lst = []
            error_lst = []
            ratio_lst = []
#            a_ng_error_lst = []
            folder_path = path + f"Thrust_{th}/Seed_{seed}/"
            # Load the nominal response variables
            nominal_folder_path = folder_path + "Nominal/Run_0/"
            nominal_response_variables = np.loadtxt(nominal_folder_path + "response_variables.txt")
            fuel_consumption_nominal = nominal_response_variables[0]
            error_nominal = nominal_response_variables[1]
            ratio_nominal = np.loadtxt(nominal_folder_path + "PSD_ratio.txt")

#            a_ng_error_nominal = np.loadtxt(nominal_folder_path + "a_ng_error.txt")
#            a_ng_error_nominal -= np.mean(a_ng_error_nominal, axis=0)

            # Loop over the different settings
            for setting_idx in range(len(var_setting_copy)):
                var_folder = folder_path + f"{var}/Run_{setting_idx}/"
                response_variables = np.loadtxt(var_folder + "response_variables.txt")
                ratio = np.loadtxt(var_folder + "PSD_ratio.txt")

                # Store the response variables
                fuel_consumption_lst.append(response_variables[0])
                error_lst.append(response_variables[1])
                ratio_lst.append(ratio)

                # Load the a_ng error
#                a_ng_error = np.loadtxt(var_folder + "a_ng_error.txt")
#                a_ng_error -= np.mean(a_ng_error, axis=0)
#                a_ng_error_lst.append(a_ng_error)

            if var not in ['Layout', 'Axis']:
                # Append the nominal values
                var_setting_copy.append(nominal_settings[var])
                fuel_consumption_lst.append(fuel_consumption_nominal)
                error_lst.append(error_nominal)
                ratio_lst.append(ratio_nominal)
#                a_ng_error_lst.append(a_ng_error_nominal)

                # Sort the lists
                sorted_idx = np.argsort(var_setting_copy)
                var_setting_copy = np.array(var_setting_copy)[sorted_idx]
                fuel_consumption_lst = np.array(fuel_consumption_lst)[sorted_idx]
                error_lst = np.array(error_lst)[sorted_idx]
                ratio_lst = np.array(ratio_lst)[sorted_idx]
#                a_ng_error_lst = np.array(a_ng_error_lst)[sorted_idx]
            else:
#                a_ng_error_lst = np.array(a_ng_error_lst)
                pass

            # Append the a_ng time series to the seed list
#            a_ng_seed_lst.append(a_ng_error_lst)
            fuel_consumption_seed_lst.append(fuel_consumption_lst)
            error_seed_lst.append(error_lst)
            ratio_seed_lst.append(ratio_lst)

#        a_ng_seed_stacked = np.stack(a_ng_seed_lst, axis=0)
        # Mean and standard deviation of the a_ng error
#        a_ng_error_mean = np.mean(a_ng_seed_stacked, axis=0)
#        a_ng_error_std = np.std(a_ng_seed_stacked, axis=0)
#        a_ng_error_mean = a_ng_error_mean[:, :, np.newaxis]

        # Do it for fuel, ratio and error as well
        fuel_consumption_seed_stacked = np.stack(fuel_consumption_seed_lst, axis=0)
        fuel_consumption_mean = np.mean(fuel_consumption_seed_stacked, axis=0)
        fuel_consumption_std = np.std(fuel_consumption_seed_stacked, axis=0, ddof=1)

        error_seed_stacked = np.stack(error_seed_lst, axis=0)
        error_mean = np.mean(error_seed_stacked, axis=0)
        error_std = np.std(error_seed_stacked, axis=0)

        ratio_seed_stacked = np.stack(ratio_seed_lst, axis=0)
        ratio_mean = np.mean(ratio_seed_stacked, axis=0)
        ratio_std = np.std(ratio_seed_stacked, axis=0, ddof=1)

        # calculate the confidence interval of fuel and ratio
        confidence = 0.95
        z = norm.ppf((1 + confidence) / 2)  # z-score
        fuel_consumption_std = fuel_consumption_std / np.sqrt(len(seeds))
        ratio_std = ratio_std / np.sqrt(len(seeds))

        fuel_consumption_lb = fuel_consumption_mean - z * fuel_consumption_std
        fuel_consumption_ub = fuel_consumption_mean + z * fuel_consumption_std

        ratio_lb = ratio_mean - z * ratio_std
        ratio_ub = ratio_mean + z * ratio_std


        # Plot the results
        ax.plot(var_setting_copy, fuel_consumption_mean, color='b', marker='o')
        if var == "Thrust" or var == "Duration":
            ax.fill_between(var_setting_copy, fuel_consumption_lb, fuel_consumption_ub, color='b', alpha=0.3)

        ax2.plot(var_setting_copy, ratio_mean, color='r', marker='o')
        ax2.fill_between(var_setting_copy, ratio_lb, ratio_ub, color='r', alpha=0.3)
        fig.tight_layout()
        # Save the figures
        save_figure(fig, fig_save_path + f"Thrust_{th}/{var}_response_averaged.png")
#        fig_asd, ax_asd = plot_asd(a_ng_error_mean, NFFT, [f"{var} = "], path=fig_save_path + f"Thrust_{th}/{var}_psd_averaged.png",
#                                   series_labels=list(var_setting_copy), plot_all_axes=False, color='brg', save_fig=False)
#        ax_asd.set_title(f"Seeds 1-{len(seeds)} averaged. T_mag={th}", fontsize=18)
#        ax_asd.plot(freq_acc, a_ng_requirement, 'k', label='Requirement', marker='o')
#        ax_asd.legend()

#        save_figure(fig_asd, fig_save_path + f"Thrust_{th}/{var}_psd_averaged.png")

