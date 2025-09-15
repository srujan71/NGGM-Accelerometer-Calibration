from Tools.CalibrationTools import calibrate, reconstruct_L2_modified_a_ng, reconstruct_a_ng, acc_cal_par_vec_to_mat
import os
import numpy as np
from Tools.SignalProcessingUtilities import Noise
from Tools.Initialisation import acceleration_setup
from Tools.PlotTools import create_plots, save_figure, draw_labels, plot_asd, plt

def run_calibration(acc_lst, layout, NFFT, noise_switch, path,
                    linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking, inertia_tensor, thruster_arm):
    """
    Run the calibration and calculate the response variables
    :param acc_lst: List of accelerometer
    :param layout: layout
    :param NFFT: Number of points for the decorelation filter
    :param noise_switch: Bool for noise
    :param path: path to save the results
    :param linear_acceleration_shaking: time series of linear acceleration of shaking
    :param angular_acceleration_shaking:  time series of angular acceleration of shaking
    :param angular_rates_shaking: time series of angular rates of shaking
    :param inertia_tensor: inertia tensor of the satellite
    :param thruster_arm: Arm length of the thrusters from COM
    :return:
    """
    ##########################################
    # Run calibration ########################
    ##########################################
    if not os.path.exists(path):
        os.makedirs(path)
    x_err_initial, x_err, residuals, cov_par, x0, x_true, sigma, digit_loss = calibrate(acc_lst, layout, NFFT, noise_switch)
    np.save(path + "x_err.npy", x_err)
    np.savetxt(path + "x_err_initial.txt", x_err_initial)
    np.save(path + "covariance_matrix.npy", cov_par)
    np.savetxt(path + "x0.txt", x0)
    np.savetxt(path + "x_true.txt", x_true)
    np.savetxt(path + "singular_values.txt", sigma)
    # np.save(path + 'residuals.npy', residuals)        # Only save it if required. Large file

    # calculate the response variables here
    #################################################################################################
    # 1) Fuel consumption
    #################################################################################################
    # Calculate the torque on the satellite
    M = np.empty((len(angular_acceleration_shaking), 3))
    for i in range(len(angular_rates_shaking)):
        Omega = np.array([[0, -angular_rates_shaking[i, 2], angular_rates_shaking[i, 1]],
                          [angular_rates_shaking[i, 2], 0, -angular_rates_shaking[i, 0]],
                          [-angular_rates_shaking[i, 1], angular_rates_shaking[i, 0], 0]])

        M[i, :] = inertia_tensor @ angular_acceleration_shaking[i, :] + Omega @ inertia_tensor @ angular_rates_shaking[i, :]

    # Assume that Mx is produced by Fz, My by Fx and Mz by Fy only
    Fy = (M[:, 0] + M[:, 2]) / (thruster_arm[0] - thruster_arm[2])
    Fz = -M[:, 1] / thruster_arm[0]

    F_angular = np.vstack((Fy, Fz)).T
    # Calculate the mass rate
    # m_dot = (linear_shaking_series + acceleration_angular) * 1000 / (350 * 9.81)
    mdot_linear = (linear_acceleration_shaking * 1000) / (60 * 9.81)
    mdot_angular = (F_angular) / (60 * 9.81)

    fuel_consumption_linear = np.sum(np.linalg.norm(mdot_linear, axis=1))
    fuel_consumption_angular = np.sum(np.linalg.norm(mdot_angular, axis=1))
    fuel_consumption = fuel_consumption_linear + fuel_consumption_angular

    response_variable = np.array([fuel_consumption])

    #################################################################################################
    # 2) Error between calibrated and true parameters
    #################################################################################################
    error = np.linalg.norm(x0 - x_true)
    response_variable = np.append(response_variable, error)

    np.savetxt(path + f"response_variables.txt", response_variable)

    return x0, x_true, digit_loss


def calculate_power_band(time_series, NFFT, f_lb, f_ub):
    """
    Calculate the power spectral density of a time series
    :param time_series: The time series
    :param NFFT: The number of points in the FFT
    :param f_lb: The lower bound of the frequency range
    :param f_ub: The upper bound of the frequency range
    :return: The power spectral density
    """
    fs = 1
    signal = Noise(1, fs=fs)
    ffx, Pxx = signal.psd_welch(time_series, NFFT)

    condition = (ffx >= f_lb) & (ffx <= f_ub)
    power = np.sum(Pxx[condition]) * (ffx[1] - ffx[0])

    return power


def calculate_a_ng_error(a_ng_true, a_ng_rcst):
    """
    Calculate the error between the true and reconstructed a_ng
    :param a_ng_true: True a_ng
    :param a_ng_rcst: Reconstructed a_ng
    :return: The error
    """
    a_ng_residual = a_ng_rcst - a_ng_true
    a_ng_residual_relative = np.sqrt(2) * a_ng_residual

    # LoS vector
    LoS = np.array([1, 1e-5, 1e-5])
    LoS_dir = LoS / np.linalg.norm(LoS)
    a_ng_residual_relative_los = np.dot(a_ng_residual_relative, LoS_dir)

    # Remove the mean
    a_ng_residual_relative_los -= np.mean(a_ng_residual_relative_los)

    return a_ng_residual_relative_los


def calculate_modified_L2_ratio(acc_lst_science, x0, requirement_power):
    a_ng_rcst = reconstruct_L2_modified_a_ng(x0, acc_lst_science)
    a_ng_true = acc_lst_science[0].a_ng

    a_ng_residual = a_ng_rcst - a_ng_true
    a_ng_residual_relative = np.sqrt(2) * a_ng_residual

    # LoS vector
    LoS = np.array([1, 1e-5, 1e-5])
    LoS_dir = LoS / np.linalg.norm(LoS)
    a_ng_residual_relative_los = np.dot(a_ng_residual_relative, LoS_dir)

    # Remove the mean
    a_ng_residual_relative_los -= np.mean(a_ng_residual_relative_los)

    a_ng_error_power = calculate_power_band(a_ng_residual_relative_los, 27001, 0.1e-3, 1e-3)

    return a_ng_error_power / requirement_power


def plot_parameters_error(path, name, freq, a_ng_req, layout=2):
    ##########################################################################################################
    # Estimated parameters vs True parameters
    ##########################################################################################################
    save_path = f'Test_figures/{name}/'

    if name == 'y_min':
        axis = 'y'
    else:
        axis = name

    x_err = np.load(path + "x_err.npy")
    x_err_initial = np.loadtxt(path + "x_err_initial.txt")
    cov_par = np.load(path + "covariance_matrix.npy")
    x0 = np.loadtxt(path + "x0.txt")
    x_true = np.loadtxt(path + "x_true.txt")
    # if name == 'y':
    #     x0[0:18] = x_true[0:18]
    print(f"Norm of M matrices error: {np.linalg.norm(x0 - x_true)}")

    # Calculate the standard deviation of the estimated parameters
    std_dev = np.sqrt(np.diag(cov_par))
    # print(std_dev)
    std_dev[27:36] = std_dev[27:36] * 1e6

    # Create the plot for errors between estimated and true parameters
    fig_err, ax_err = create_plots("Parameter [-]", "Estimation Error [-]", False, True, fontsize=16)
    ax_err.plot(abs(x_err_initial), marker='o', label="Initial guess", color='b')
    ax_err.plot(abs(x_err[-1][-1]), marker='o', label="Estimated Solution", color='r')
    # plot first step error
    # ax_err.plot(abs(x_err[0][0]), marker='o', label="First step error", color='orange')
    # ax_err.plot(abs(std_dev), marker='o', label="Standard deviation", color='g')
    ax_err.grid(alpha=0.5)
    ax_err.set_xticks(np.arange(0, 47, 1), minor=True)
    ax_err.legend(fontsize=12)
    const_x_lines = [-0.5, 8.5, 17.5, 26.5, 35.5, 41.5, 46.6]
    for i in const_x_lines:
        ax_err.axvline(x=i, color='k', linestyle='--', alpha=0.65)

    min_error = min(abs(x_err[-1][-1]))
    height = 1.4 * 3e-8
    label_y_coord = 0.5 * 3e-8
    fontsize = 14
    draw_labels(ax_err, const_x_lines, label_y_coord, height, fontsize, layout)

    ax_err.set_ylim(1e-8, 30)

    error, acc_lst = reconstruct_acc(x0, x_true, axis)
    #
    fig_asd, ax_asd = plot_asd(error, 27001, [""], save_path)
    ax_asd.plot(freq, a_ng_req, label="Requirement", color='r')

    sd_x = np.sqrt(np.diag(cov_par))
    # sd_x[27:36] = sd_x[27:36]
    par = np.arange(0, 47, 1)
    # cov_par[27:36, 27:36] = cov_par[27:36, 27:36]

    # Compute correlation matrix
    rho = cov_par / np.outer(sd_x, sd_x)

    fig2, ax2 = create_plots("Parameter", "Parameter", False, False)
    cax = ax2.matshow(rho, aspect='auto', cmap='seismic')
    fig2.colorbar(cax, ax=ax2)


    save_figure(fig_asd, save_path + f"ASD_{name}.svg")
    save_figure(fig_err, save_path + f"Estimation_error.svg")
    save_figure(fig2, save_path + f"Correlation_matrix.svg")

    M1, M2, M3, K1, K2, K3, W1, W2, W3, dr1, dr2, dr3 = acc_cal_par_vec_to_mat(x0, acc_lst, layout=layout)

    M1_true, M2_true, M3_true, K1_true, K2_true, K3_true, W1_true, W2_true, W3_true, dr1_true, dr2_true, dr3_true = acc_cal_par_vec_to_mat(x_true, acc_lst, layout=layout)

    M1_norm = np.linalg.norm(M1 - M1_true)
    M2_norm = np.linalg.norm(M2 - M2_true)
    M3_norm = np.linalg.norm(M3 - M3_true)

    # print(f"Norm of M1: {M1_norm}")
    # print(f"Norm of M2: {M2_norm}")
    # print(f"Norm of M3: {M3_norm}\n")




def reconstruct_acc(x0, x_true, axis):
    shaking_dictionary = {}
    L = 0.6
    layout = 2
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
    acc_lst = acceleration_setup(
        shaking_dictionary=shaking_dictionary,
        position_dictionary=
        position_dictionary[axis][layout],
        duration=86400, layout=layout,
        noise_switch=1,
        shake_bool=False,
        goce_parameters=False,
        load_true_data=False)

    acc_lst_science = acceleration_setup(shaking_dictionary=shaking_dictionary,
                                         position_dictionary=position_dictionary[axis][layout],
                                         layout=layout, duration=172800, noise_switch=1,
                                         shake_bool=False, goce_parameters=False,
                                         load_true_data=True, x_true=x_true, acc_lst_dummy=acc_lst)

    a_ng_rcst = reconstruct_a_ng(x0, acc_lst_science, layout)

    return calculate_a_ng_error(acc_lst_science[0].a_ng, a_ng_rcst), acc_lst_science

