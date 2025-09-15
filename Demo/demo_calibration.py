import numpy as np
from Tools.Initialisation import acceleration_setup
from Tools.SensitivityAnalysisTools import run_calibration, calculate_power_band
from tudatpy.data import save2txt
import os
import time
from Tools.CalibrationTools import reconstruct_a_ng, reconstruct_L2_modified_a_ng
from scipy.fft import rfftfreq
from Tools.VerficationTools import verify_lstsq

current_dir = os.path.dirname(__file__)

shake_bool = True
noise_switch = 1

layout = 2  # 1, 2, 3
axis = 'x'  # x, y, z
T = 2e-06   # m/s^2
duration = 86400  # s
f_upper_bound = 0.1  # Hz
L = 0.6  # m
bandwidth = 4   # units
output_path = current_dir + f"/../SimulationOutput/Output/Demo_calibration/"

bandwidth = bandwidth * 10 ** (int(np.log10(f_upper_bound) - 1))
bandwidth = np.trunc(bandwidth * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors
f_lower_bound = f_upper_bound - bandwidth
f_lower_bound = np.trunc(f_lower_bound * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors

# Select the window size for the power spectral density calculation
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

dict_to_write = {"Layout": layout,
                 "Axis": axis,
                 "Thrust magnitude": T,
                 "Duration": duration,
                 "Upper bound": f_upper_bound,
                 "Arm length": L,
                 "Bandwidth": bandwidth}

# Inertia tensor and thruster arm
J = np.array([[183.7798188, -1.703220484, 0.577165535],
              [-1.703220484, 965.8341289, 0.059878845],
              [0.577165535, 0.059878845, 1080.630192]])

M_arm = np.array([1.5, 0.719, 0.450])

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

# Run the calibration
start = time.time()
x0, x_true, digit_loss = run_calibration(acc_lst, layout, NFFT, noise_switch, output_path,
                                         linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking,
                                         J, M_arm)
end = time.time()

# Create the science mode observations
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

power_NFFT = 27001
power_f_lb = 1e-4
power_f_ub = 1e-3
freq_acc = rfftfreq(NFFT, 1)
a_ng_req = (5e-12 * np.sqrt(1 + np.where(freq_acc != 0, 0.001 / freq_acc, np.inf) ** 2 + (100 * freq_acc ** 2) ** 2))
# calculate power between 0.1 - 1 mHz
condition = (freq_acc >= power_f_lb) & (freq_acc <= power_f_ub)
a_ng_req_power = np.sum(a_ng_req[condition] ** 2) * (freq_acc[1] - freq_acc[0])

a_ng_error_power = calculate_power_band(a_ng_residual_relative_los, power_NFFT, power_f_lb, power_f_ub)
a_ng_error_power_modified = calculate_power_band(a_ng_residual_relative_los_modified, power_NFFT, power_f_lb, power_f_ub)

a_ng_error_power_ratio = a_ng_error_power / a_ng_req_power
a_ng_error_power_ratio_modified = a_ng_error_power_modified / a_ng_req_power

np.savetxt(output_path + "PSD_ratio.txt", np.array([a_ng_error_power_ratio]))
np.savetxt(output_path + "PSD_ratio_modified.txt", np.array([a_ng_error_power_ratio_modified]))

# Save the info dictionary
dict_to_write["Digit loss"] = digit_loss
dict_to_write["Calibration time"] = end - start
save2txt(dict_to_write, "Info.txt", output_path)

######################################################################################################################################################
# Visualization ######################################################################################################################################
######################################################################################################################################################

verify_lstsq(acc_lst, layout=2, noise_switch=1, load_data=True, path=output_path)




