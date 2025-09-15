import logging
import numpy as np
from Tools.Initialisation import acceleration_setup
from Tools.SensitivityAnalysisTools import run_calibration, calculate_power_band
from tudatpy.data import save2txt
import os
import time
from multiprocessing import Pool
from Tools.CalibrationTools import reconstruct_a_ng, reconstruct_L2_modified_a_ng
from Tools.Logging_config import setup_logging
from scipy.fft import rfftfreq
from Tools.SignalProcessingUtilities import hann_window


def run_sim(combination, path, inertia_tensor, thruster_arm, a_ng_requirement_power, power_NFFT, power_f_lb, power_f_ub):
    try:
        logging.info(f"Starting simulation: {combination['Folder']} Seed {combination['Seed']}")

        shake_bool = True
        noise_switch = 1

        layout = combination['Layout']
        axis = combination['Axis']
        T = combination['Thrust']
        duration = combination['Duration']
        f_upper_bound = combination['Frequency']
        L = combination['Length']
        bandwidth = combination['Bandwidth']
        seed = combination['Seed']
        # Set the seed:
        np.random.seed(seed)

        folder_name = combination['Folder']
        index = combination['Index']
        output_path = path + f"{folder_name}_{seed}/"

        logging.debug(f"Simulation parameters: {combination}")

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

        dict_to_write = {"Run": index,
                         "Layout": layout,
                         "Axis": axis,
                         "Thrust magnitude": T,
                         "Duration": duration,
                         "Upper bound": f_upper_bound,
                         "Arm length": L,
                         "Bandwidth": bandwidth,
                         "Seed": seed}

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

        start = time.time()
        x0, x_true, digit_loss = run_calibration(acc_lst, layout, NFFT, noise_switch, output_path,
                                                 linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking,
                                                 inertia_tensor, thruster_arm)
        end = time.time()

        acc_lst_science = acceleration_setup(shaking_dictionary=shaking_dictionary,
                                             position_dictionary=position_dictionary[axis][layout],
                                             layout=layout, duration=172800, noise_switch=noise_switch,
                                             shake_bool=False, goce_parameters=False,
                                             load_true_data=True, x_true=x_true, acc_lst_dummy=acc_lst)

        # Calculate the estimate of a_ng
        a_ng_rcst = reconstruct_a_ng(x0, acc_lst_science, layout)
        a_ng_rcst_modified = reconstruct_L2_modified_a_ng(x0, acc_lst_science)
        a_ng_true = acc_lst_science[0].a_ng

        a_ng_residual = a_ng_rcst - a_ng_true
        a_ng_residual_modified = a_ng_rcst_modified - a_ng_true
        a_ng_residual_relative = np.sqrt(2) * a_ng_residual
        a_ng_residual_relative_modified = np.sqrt(2) * a_ng_residual_modified

        # LoS vector
        LoS = np.array([1, 1e-5, 1e-5])
        LoS_dir = LoS / np.linalg.norm(LoS)
        a_ng_residual_relative_los = np.dot(a_ng_residual_relative, LoS_dir)
        a_ng_residual_relative_los_modified = np.dot(a_ng_residual_relative_modified, LoS_dir)

        # Remove the mean
        a_ng_residual_relative_los -= np.mean(a_ng_residual_relative_los)
        a_ng_residual_relative_los_modified -= np.mean(a_ng_residual_relative_los_modified)

        a_ng_error_power = calculate_power_band(a_ng_residual_relative_los, power_NFFT, power_f_lb, power_f_ub)
        a_ng_error_power_modified = calculate_power_band(a_ng_residual_relative_los_modified, power_NFFT, power_f_lb, power_f_ub)

        a_ng_error_power_ratio = a_ng_error_power / a_ng_requirement_power
        a_ng_error_power_ratio_modified = a_ng_error_power_modified / a_ng_requirement_power

        np.savetxt(output_path + "PSD_ratio.txt", np.array([a_ng_error_power_ratio]))
        np.savetxt(output_path + "PSD_ratio_modified.txt", np.array([a_ng_error_power_ratio_modified]))

        # np.savetxt(output_path + "a_ng_error.txt", a_ng_residual_relative_los)

        # Save the info dictionary
        dict_to_write["Digit loss"] = digit_loss
        dict_to_write["Calibration time"] = end - start
        save2txt(dict_to_write, "Info.txt", output_path)

        logging.info(f"Run {index} completed successfully and saved to {output_path}.")

        return 0

    except Exception as e:
        logging.error(f"Error in simulation {combination['Folder']} Run {combination['Index']}: {e}", exc_info=True)
        return 1


def generate_combinations(nominal_settings, settings):
    """
    Generate all combinations of settings by changing one variable at a time
    from the nominal settings.

    Args:
        nominal_settings (dict): Nominal values for each variable.
        settings (dict): List of settings for each variable.

    Returns:
        list[dict]: List of all combinations.
    """
    combinations = []

    # Add the nominal settings as the first simulation
    nominal_combination = nominal_settings.copy()
    nominal_combination['Folder'] = "Nominal"
    nominal_combination['Index'] = 0
    combinations.append(nominal_combination)

    # Iterate through each variable in the settings
    for var, var_settings in settings.items():
        for idx, setting in enumerate(var_settings):
            # Create a new combination based on the nominal settings
            new_combination = nominal_settings.copy()
            # Change only the current variable
            new_combination[var] = setting
            # Add the index
            new_combination['Index'] = idx
            # Add the folder name
            new_combination['Folder'] = var
            # Append the combination
            combinations.append(new_combination)

    return combinations


def run_sim_wrapper(args):
    """
        Wrapper function for `run_sim` to unpack arguments when using multiprocessing.
        """
    try:
        return run_sim(*args)
    except Exception as e:
        combination = args[0]
        logging.error(f"Error in multiprocessing for combination {combination['Index']}: {e}", exc_info=True)
        return 1


def find_setting(key, value, combinations):
    result = [d for d in combinations if d.get(key) == value]
    return result


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

    factor = np.sqrt(nom_power / power)
    print("Factor:", factor)
    # Append the factor to the thrust setting
    setting["Thrust"] = factor * thrust

    return None


if __name__ == "__main__":
    total_start_time = time.time()
    current_dir = os.path.dirname(__file__)
    setup_logging(log_directory=current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time", log_name="Simulation")
    logging.info("Starting sensitivity analysis one at a time simulations")

    num_processes = int(input("Enter the number of processes: "))
    if num_processes > os.cpu_count():
        logging.critical(f"Number of processes exceeds the number of available CPUs: {os.cpu_count()}")
        raise ValueError("Number of processes exceeds the number of available CPUs.")
    elif num_processes < 1:
        logging.critical("Number of processes must be greater than 0.")
        raise ValueError("Number of processes must be greater than 0.")
    else:
        logging.info(f"Number of processes: {num_processes}")

    J = np.array([[183.7798188, -1.703220484, 0.577165535],
                  [-1.703220484, 965.8341289, 0.059878845],
                  [0.577165535, 0.059878845, 1080.630192]])

    M_arm = np.array([1.5, 0.719, 0.450])

    NFFT = 27001
    freq_acc = rfftfreq(NFFT, 1)
    a_ng_req = (5e-12 * np.sqrt(1 + np.where(freq_acc != 0, 0.001 / freq_acc, np.inf) ** 2 + (100 * freq_acc ** 2) ** 2))
    # calculate power between 0.1 - 1 mHz
    f_lb = 0.1e-3
    f_ub = 1e-3
    condition = (freq_acc >= f_lb) & (freq_acc <= f_ub)
    a_ng_req_power = np.sum(a_ng_req[condition] ** 2) * (freq_acc[1] - freq_acc[0])

    frequencies = [0.01, 0.1]

    nominal_settings_hf = {'Thrust': 2e-6, 'Duration': 86400, 'Frequency': 0.1, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'y'}
#    for duration in durations:
    for freq_sh in frequencies:
        nominal_settings = {'Thrust': 2e-6, 'Duration': 86400, 'Frequency': freq_sh, 'Length': 0.6, 'Bandwidth': 4, 'Layout': 2, 'Axis': 'y'}

        settings = {'Seed': list(range(1, 201))}

        # Correct the thrust for the different frequencies and bandwidth
        if freq_sh != 0.1:
            calculate_factor_frequency(nominal_settings_hf, nominal_settings)

        # One at a time variation from nominal settings
        combinations_lst = generate_combinations(nominal_settings, settings)

        folder_path = current_dir + f"/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Frequency_analysis/avg_removed/Layout_2/Axis_y/Thrust_2e-6/f_{freq_sh}/"

        with Pool(num_processes) as pool:
            args = [(combination, folder_path, J, M_arm, a_ng_req_power, NFFT, f_lb, f_ub) for combination in combinations_lst[1:]]
            pool.map(run_sim_wrapper, args)

total_end_time = time.time()
total_time = total_end_time - total_start_time
logging.info(f"Total time for all simulations: {total_time} s")
logging.info("All simulations completed.")
