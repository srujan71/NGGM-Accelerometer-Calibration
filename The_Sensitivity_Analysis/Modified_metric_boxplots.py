import numpy as np
from Tools.PlotTools import plt, save_figure, create_plots, plot_asd
import os
from scipy.fft import rfftfreq
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from Tools.Initialisation import acceleration_setup
from Tools.CalibrationTools import reconstruct_L2_modified_a_ng

current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Layout_2_runs_modified/Layout_2/Axis_y/Thrust_2e-6/"
fig_save_path = "Sensitivity_analysis/One_at_a_time/Boxplots/"  # The first half of the path is defined already in save figure function

L = 0.6
shake_bool = True
T = 2e-6
f_lower_bound = 6e-2
f_upper_bound = 1e-1
duration = 86400
layout = 2
axis = 'x'
NFFT = 27001
noise_switch = 1

freq_acc = rfftfreq(NFFT, 1)
a_ng_req = (5e-12 * np.sqrt(1 + np.where(freq_acc != 0, 0.001 / freq_acc, np.inf) ** 2 + (100 * freq_acc ** 2) ** 2))
# calculate power between 0.1 - 1 mHz
f_lb = 0.1e-3
f_ub = 1e-3

condition = (freq_acc >= f_lb) & (freq_acc <= f_ub)
a_ng_req_power = np.sum(a_ng_req[condition] ** 2) * (freq_acc[1] - freq_acc[0])

linear_shaking_dict = dict()
angular_shaking_dict = dict()
linear_shaking_dict['x'] = [shake_bool, T, f_lower_bound, f_upper_bound, duration]
linear_shaking_dict['y'] = [shake_bool, T, f_lower_bound, f_upper_bound, duration]
linear_shaking_dict['z'] = [shake_bool, T, f_lower_bound, f_upper_bound, duration]

angular_shaking_dict['x'] = [shake_bool, T, f_lower_bound, f_upper_bound, duration]
angular_shaking_dict['y'] = [shake_bool, T, f_lower_bound, f_upper_bound, duration]
angular_shaking_dict['z'] = [shake_bool, T, f_lower_bound, f_upper_bound, duration]

shaking_dictionary = {"linear": linear_shaking_dict, "angular": angular_shaking_dict}

# Create the position dictionary
position_dictionary = {'x': {1: {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([-L / 2, 0, 0])},
                             2: {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([-L / 2, 0, 0])},
                             3: {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([0, L / 2, 0]), "pos_acc3": np.array([-L / 2, 0, 0]),
                                 "pos_acc4": np.array([0, -L / 2, 0])}},
                       'y': {1: {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([0, -L / 2, 0])},
                             2: {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, -L / 2, 0])},
                             3: {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([0, 0, L / 2]), "pos_acc3": np.array([0, -L / 2, 0]),
                                 "pos_acc4": np.array([0, 0, -L / 2])}},
                       'z': {1: {"pos_acc1": np.array([0, 0, L / 2]), "pos_acc2": np.array([0, 0, -L / 2])},
                             2: {"pos_acc1": np.array([0, 0, L / 2]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, 0, -L / 2])},
                             3: {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([0, 0, L / 2]), "pos_acc3": np.array([-L / 2, 0, 0]),
                                 "pos_acc4": np.array([0, 0, -L / 2])}}}

freqs_sh = [0.01, 0.1]
seeds = list(range(1, 201))
combined_data = []
for f_sh in freqs_sh:
    # acc_lst, linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking = acceleration_setup(
    #     shaking_dictionary=shaking_dictionary,
    #     position_dictionary=
    #     position_dictionary[axis][layout],
    #     duration=duration, layout=layout,
    #     noise_switch=noise_switch,
    #     shake_bool=shake_bool,
    #     goce_parameters=False,
    #     load_true_data=False)
    ratio_arr = np.zeros((len(seeds), 2))
    for seed in seeds:
        folder_path = path + f"f_{f_sh}/Seed_{seed}/"
        x0 = np.loadtxt(folder_path + "x0.txt")
        x_true = np.loadtxt(folder_path + "x_true.txt")
        ratio = np.loadtxt(folder_path + "PSD_ratio.txt")
        ratio_modified = np.loadtxt(folder_path + "PSD_ratio_modified.txt")
        ratio_arr[seed - 1, 0] = ratio
        ratio_arr[seed - 1, 1] = ratio_modified

        # print(ratio_modified)
        # # Compute the normalized confidence intervals
        # cov_x = np.load(folder_path + "covariance_matrix.npy")
        # sd_x = np.sqrt(np.diag(cov_x))
        # sd_x[27:36] = sd_x[27:36] * 1e6
        # par = np.arange(0, 47, 1)
        # x_normalized = (x0 - x_true) / sd_x
        # x_true_normalized = np.zeros_like(x0)
        # confidence_level = 3
        # lower_bound = x_normalized - confidence_level
        # upper_bound = x_normalized + confidence_level
        # fig3, ax3 = create_plots("Parameter", "Value", x_scale_log=False, y_scale_log=False)
        # ax3.errorbar(par, x_normalized, yerr=confidence_level, fmt='o', label='Estimated Parameter (CI)', capsize=5)
        # ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # ax3.scatter(par, x_true_normalized, label='True Value', color='r')
        # ax3.legend()
        #
        # # Find the largest of the distance between the bounds and the true value
        # x_worst = np.zeros(len(x0))
        # for i in range(len(x_normalized)):
        #     if abs(lower_bound[i]) < abs(upper_bound[i]):
        #         x_worst[i] = x0[i] + (3 * sd_x[i])
        #     else:
        #         x_worst[i] = x0[i] - (3 * sd_x[i])
        #
        # acc_lst_science = acceleration_setup(shaking_dictionary=shaking_dictionary,
        #                                      position_dictionary=position_dictionary[axis][layout],
        #                                      layout=layout, duration=172800, noise_switch=noise_switch,
        #                                      shake_bool=False, goce_parameters=False,
        #                                      load_true_data=True, x_true=x_true, acc_lst_dummy=acc_lst)
        #
        # # Calculate the estimate of a_ng
        # a_ng_rcst = reconstruct_L2_modified_a_ng(x0, acc_lst_science)
        # a_ng_true = acc_lst_science[0].a_ng
        #
        # a_ng_residual = a_ng_rcst - a_ng_true
        # a_ng_residual_relative = np.sqrt(2) * a_ng_residual
        #
        # # LoS vector
        # LoS = np.array([1, 1e-5, 1e-5])
        # LoS_dir = LoS / np.linalg.norm(LoS)
        # a_ng_residual_relative_los = np.dot(a_ng_residual_relative, LoS_dir)
        #
        # # Remove the mean
        # a_ng_residual_relative_los -= np.mean(a_ng_residual_relative_los)
        #
        # fig, ax = plot_asd(a_ng_residual_relative_los, NFFT, [""], "", plot_all_axes=False)
        # ax.plot(freq_acc, a_ng_req, label='Requirement', color='red', linestyle='--')
        #
        #
        # plt.show()

    df = pd.DataFrame(ratio_arr, columns=['All accelerometers', 'Only central accelerometer'])
    df_melted = df.melt(var_name='Type', value_name='Ratio')

    df_melted['Frequency'] = f_sh
    combined_data.append(df_melted)
    color_palette = {'All accelerometers': 'red', 'Only central accelerometer': 'green'}
    # Plot the histogram of ratios, stacked. Y-axis as percentage
    fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
    sns.histplot(df_melted, x='Ratio', hue='Type',
                 binwidth=1-np.min(ratio_arr), ax=ax_hist, stat='percent', multiple='layer', palette=color_palette, common_norm=False)
    ax_hist.set_xlabel("Ratio [-]")
    ax_hist.set_ylabel("Percentage %")
    ax_hist.set_xscale('log')
    ax_hist.grid()
    plt.show()

df_final = pd.concat(combined_data, ignore_index=True)
df_final['Frequency'] = df_final['Frequency'].astype(str)
# Compute means for each combination of Type and Frequency
means = df_final.groupby(['Frequency', 'Type'])['Ratio'].mean().reset_index()

custom_palette = {'All accelerometers': 'green', 'Only central accelerometer': 'orange'}
custom_palette_scatter = {'All accelerometers': 'red', 'Only central accelerometer': 'red'}

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(x='Frequency', y='Ratio', data=df_final, hue='Type', ax=ax, palette=custom_palette)

# manually dodge scatter points
means['x_dodge'] = means.groupby('Frequency').cumcount()
means['x_dodge'] = means['x_dodge'].map({0: -0.2, 1: 0.2})
freq_map = {'0.01': 0, '0.1': 1}
means['x_pos'] = means['Frequency'].map(freq_map) + means['x_dodge']

# sns.scatterplot(x='x_pos', y='Ratio', data=means, hue='Type', ax=ax, palette=custom_palette_scatter, s=100, zorder=3,
#                 legend=False, edgecolor='black', linewidth=1)

ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Ratio [-]")
ax.set_yscale('log')

# mean_proxy_label = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Mean')
# handles, labels = ax.get_legend_handles_labels()
# handles.append(mean_proxy_label)
# labels.append('Mean')
# ax.legend(handles=handles, labels=labels, loc='lower right')
ax.axhline(1, color='red', linestyle='--', label='Requirement')
ax.legend(loc='lower right')
# Plot horizontal line at ratio = 1

ax.grid()
fig.tight_layout()
save_figure(fig, fig_save_path + "Modified_metric_boxplot.svg")

plt.show()
