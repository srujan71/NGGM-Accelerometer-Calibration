import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfftfreq, rfft, irfft
from Tools.SignalProcessingUtilities import hann_window, Noise
from Tools.Initialisation import acceleration_setup
from Tools.PlotTools import save_figure, plot_asd, create_plots
from Tools.CalibrationTools import reconstruct_a_ng
from Tools.SensitivityAnalysisTools import calculate_power_band, run_calibration

np.random.seed(42)
def calculate_power(psd, N, fs):
    return np.sum(psd) * (fs / N)

def create_shaking_psd(setting, N, fs):
    frequency = rfftfreq(N, fs)
    shaking_psd_arr = np.zeros((len(frequency)))

    condition_1 = (setting[2] < frequency) & (frequency < setting[3])
    condition_2 = frequency < setting[2]
    condition_3 = frequency > setting[3]
    shaking_psd_arr[condition_1] = setting[1] ** 2
    shaking_psd_arr[condition_2] = (setting[1] / 10) ** 2
    shaking_psd_arr[condition_3] = np.interp(frequency[condition_3], [setting[3], 0.5], [setting[1] / 10, 1e-10]) ** 2
    # Set the DC component to zero
    shaking_psd_arr[0] = 0

    hann = hann_window(9)
    shaking_psd_arr = np.convolve(shaking_psd_arr, hann / np.sum(hann), mode='same')

    return frequency, shaking_psd_arr

def create_time_signal(psd, N):
    linear_shaking = Noise(psd, fs=1)
    linear_shaking.cor_fil = linear_shaking.correlation_filter(N)

    linear_shaking.noise = linear_shaking.filter_matrix(linear_shaking.cor_fil, np.random.randn(86400 + N - 1))

    return linear_shaking.noise

def calculate_factor_frequency(nominal_setting, setting):
    # Define the nominal settings
    nom_thrust = nominal_setting["Thrust"]
    nom_f_upper_bound = nominal_setting["Frequency"]
    nom_bandwidth = nominal_setting["Bandwidth"]
    nom_bandwidth = nom_bandwidth * 10 ** (int(np.log10(nom_f_upper_bound) - 1))
    nom_bandwidth = np.trunc(nom_bandwidth * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors
    nom_f_lower_bound = nom_f_upper_bound - nom_bandwidth
    nom_f_lower_bound = np.trunc(nom_f_lower_bound * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors

    nom_duration = nominal_setting["Duration"]

    if nom_duration < 10000:
        NFFT = 5501
    elif 10000 < nom_duration < 27000:
        NFFT = 11001
    else:
        NFFT = 27001

    nominal_shaking_setting = [True, nom_thrust, nom_f_lower_bound, nom_f_upper_bound]

    nom_freq, nom_shaking_psd = create_shaking_psd(nominal_shaking_setting, NFFT, fs=1)
    # Calculate the power of the nominal shaking signal
    nom_power = calculate_power(nom_shaking_psd, NFFT, fs=1)
    print("Nominal power: ", nom_power)
    # create time signal
    nom_time_signal = create_time_signal(nom_shaking_psd, NFFT)
    plot_singals(nom_freq, nom_shaking_psd, nom_time_signal, "Nominal_frequency")
    # print max difference
    nom_max_diff = np.max(nom_time_signal) - np.min(nom_time_signal)
    print("Nominal max difference: ", nom_max_diff)
    ############################################################################################################
    # Define the settings for which the factor is calculated
    ############################################################################################################
    thrust = setting["Thrust"]
    f_upper_bound = setting["Frequency"]
    bandwidth = setting["Bandwidth"]
    bandwidth = bandwidth * 10 ** (int(np.log10(f_upper_bound) - 1))
    bandwidth = np.trunc(bandwidth * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors
    f_lower_bound = f_upper_bound - bandwidth
    f_lower_bound = np.trunc(f_lower_bound * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors

    shaking_setting = [True, thrust, f_lower_bound, f_upper_bound]

    freq, shaking_psd = create_shaking_psd(shaking_setting, NFFT, fs=1)
    # Calculate the power of the shaking signal
    power = calculate_power(shaking_psd, NFFT, fs=1)
    print("Old power: ", power)
    # Create the shaking signal
    shaking_signal = create_time_signal(shaking_psd, NFFT)
    plot_singals(freq, shaking_psd, shaking_signal, "Old_frequency")
    # print max difference
    max_diff = np.max(shaking_signal) - np.min(shaking_signal)
    print("Old max difference: ", max_diff)
    factor = np.sqrt(nom_power / power)
    print("Factor: ", factor)
    # Append the factor to the thrust setting
    setting["Thrust"] = factor * thrust

    # Create the shaking signal with the corrected thrust
    new_shaking_setting = [True, setting["Thrust"], f_lower_bound, f_upper_bound]
    new_freq, new_shaking_psd = create_shaking_psd(new_shaking_setting, NFFT, fs=1)
    new_power = calculate_power(new_shaking_psd, NFFT, fs=1)
    print("New power: ", new_power)
    new_shaking_signal = create_time_signal(new_shaking_psd, NFFT)

    plot_singals(new_freq, new_shaking_psd, new_shaking_signal, "Corrected_frequency")
    # print max difference
    new_max_diff = np.max(new_shaking_signal) - np.min(new_shaking_signal)
    print("New max difference: ", new_max_diff)

    return None

def plot_singals(freq, shaking_psd, time_signal, title):
    fig, ax = create_plots("Frequency (Hz)", r'ASD ($m/s^2/\sqrt{Hz}$)', True, True, fontsize=16)
    ax.plot(freq, np.sqrt(shaking_psd), marker='o')
    # ax.set_title(f"ASD {title}")

    fig1, ax1 = create_plots("Time [hrs]", r'Acceleration magnitude [$m/s^2$]', False, False, fontsize=16)
    t = np.arange(0, len(time_signal), dtype=np.float64)
    t /= 3600
    ax1.plot(t, time_signal)
    # ax1.set_title(f"Time signal {title}")

    fig2, ax2 = plot_asd(time_signal, 10000, ['x'], path=f'Sensitivity_analysis/Thrust_correction/ASD_{title}.svg',
                         plot_all_axes=False, save_fig=True, fontsize=16)

    # print the RMS
    print(f"RMS {title}: ", np.sqrt(np.mean(time_signal**2)))
    save_figure(fig, f'Sensitivity_analysis/Thrust_correction/ASD_{title}_filter.svg')
    save_figure(fig1, f'Sensitivity_analysis/Thrust_correction/Time_signal_{title}.svg')


def run_sim(combination, path):


    shake_bool = True
    noise_switch = 1

    layout = combination['Layout']
    axis = combination['Axis']
    T = combination['Thrust']
    duration = combination['Duration']
    f_upper_bound = combination['Frequency']
    L = combination['Length']
    bandwidth = combination['Bandwidth']

    bandwidth = bandwidth * 10 ** (int(np.log10(f_upper_bound) - 1))
    bandwidth = np.trunc(bandwidth * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors
    f_lower_bound = f_upper_bound - bandwidth
    f_lower_bound = np.trunc(f_lower_bound * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors

    if duration < 10000:
        NFFT = 5501
    elif 10000 < duration < 27000:
        NFFT = 11001
    else:
        NFFT = 27001

    # Create the shaking signal dictionary
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

    # Create the accelerometers with the shaking signal
    acc_lst, linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking = acceleration_setup(
        shaking_dictionary=shaking_dictionary,
        position_dictionary=
        position_dictionary[axis][layout],
        duration=duration, layout=layout,
        noise_switch=noise_switch,
        shake_bool=shake_bool,
        goce_parameters=False,
        load_true_data=False)



    # x0, x_true, digit_loss = run_calibration(acc_lst, layout, NFFT, noise_switch, output_path,
    #                                          linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking,
    #                                          inertia_tensor, thruster_arm)
    #
    # acc_lst_science = acceleration_setup(shaking_dictionary=shaking_dictionary,
    #                                      position_dictionary=position_dictionary[axis][layout],
    #                                      layout=layout, duration=172800, noise_switch=noise_switch,
    #                                      shake_bool=False, goce_parameters=False,
    #                                      load_true_data=True, x_true=x_true, acc_lst_dummy=acc_lst)
    #
    # # Calculate the estimate of a_ng
    # a_ng_rcst = reconstruct_a_ng(x0, acc_lst_science, layout)
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
    # a_ng_error_power = calculate_power_band(a_ng_residual_relative_los, power_NFFT, power_f_lb, power_f_ub)
    #
    # a_ng_error_power_ratio = a_ng_error_power / a_ng_requirement_power
    #
    # np.savetxt(output_path + "PSD_ratio.txt", np.array([a_ng_error_power_ratio]))

    return acc_lst, linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking

nominal_settings = {'Thrust': 2e-6, 'Duration': 86400, 'Frequency': 1e-1, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'y'}
setting = {'Thrust': 2e-6, 'Duration': 86400, 'Frequency': 1e-2, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'y'}


acc_lst, ang, dw, w = run_sim(nominal_settings, "")

fig, ax = create_plots("Time [hrs]", r'Acceleration magnitude [$m/s^2$]', False, False, fontsize=16)
t = np.arange(0, len(dw), dtype=np.float64)
t /= 3600
ax.plot(t, dw)

fig1, ax1 = create_plots("Time [hrs]", r'Angular rate magnitude [$rad/s$]', False, False, fontsize=16)
ax1.plot(t, w[:, 0], label=r'$\omega_{x}$')
ax1.plot(t, w[:, 1], label=r'$\omega_{y}$')
ax1.plot(t, w[:, 2], label=r'$\omega_{z}$')
ax1.legend(fontsize=14)
ax1.set_ylim([-6.5e-5, 6.5e-5])

# save_figure(fig, f'Sensitivity_analysis/Thrust_correction/Angular_acceleration_hf.svg')
save_figure(fig1, f'Sensitivity_analysis/Thrust_correction/Angular_rate_hf.svg')


calculate_factor_frequency(nominal_settings, setting)

acc_lst_l, angl, dwl, wl = run_sim(setting, "")

fig2, ax2 = create_plots("Time [hrs]", r'Acceleration magnitude [$m/s^2$]', False, False, fontsize=16)
ax2.plot(t, dwl)

#
fig3, ax3 = create_plots("Time [hrs]", r'Angular rate magnitude [$rad/s$]', False, False, fontsize=16)
ax3.plot(t, wl[:, 0], label=r'$\omega_{x}$')
ax3.plot(t, wl[:, 1], label=r'$\omega_{y}$')
ax3.plot(t, wl[:, 2], label=r'$\omega_{z}$')
ax3.legend(fontsize=14)
ax3.set_ylim([-6.5e-5, 6.5e-5])

# rms calculation
rms_wl_x = np.sqrt(np.mean(wl[:, 0]**2))
rms_wl_y = np.sqrt(np.mean(wl[:, 1]**2))
rms_wl_z = np.sqrt(np.mean(wl[:, 2]**2))

# save_figure(fig2, f'Sensitivity_analysis/Thrust_correction/Angular_acceleration_lf_wo_scaling.svg')
# save_figure(fig3, f'Sensitivity_analysis/Thrust_correction/Angular_rate_lf_wo_scaling.svg')
# print the rms of the angular acceleration and angular rate
print("RMS angular rate x: ", rms_wl_x)
print("RMS angular rate y: ", rms_wl_y)
print("RMS angular rate z: ", rms_wl_z)


# save_figure(fig2, f'Sensitivity_analysis/Thrust_correction/Angular_acceleration_lf_scaled.svg')
save_figure(fig3, f'Sensitivity_analysis/Thrust_correction/Angular_rate_lf_scaled.svg')



# plt.show()





