import pandas as pd
import numpy as np
from Tools.Initialisation import acceleration_setup
from Tools.CalibrationTools import reconstruct_a_ng
from Tools.SensitivityAnalysisTools import run_calibration
from tudatpy.data import save2txt
import os
import time

current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/Factorial/"
factorial_design = pd.read_excel('../Input_data/Factorial_design.xlsx', sheet_name='Factorial', usecols="B:F", keep_default_na=False, na_filter=False)

axes = ['x', 'y', 'z']
layouts = [1, 2, 3]
shake_bool = True
noise_switch = 1

total_start = time.time()
# Loop over the factor settings
for index, row in factorial_design.iterrows():
    # Extract the parameters
    T = row["A"]
    duration = int(row["B"])
    f_upper_bound = row["C"]
    L = row["D"]
    bandwidth = row["E"]
    bandwidth = bandwidth * 10 ** (int(np.log10(f_upper_bound) - 1))
    bandwidth = np.trunc(bandwidth * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors
    f_lower_bound = f_upper_bound - bandwidth
    f_lower_bound = np.trunc(f_lower_bound * 10 ** 6) / 10 ** 6  # Truncate to avoid floating point errors

    NFFT = duration // 3
    NFFT = duration if NFFT % 2 == 1 else NFFT + 1
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

    # Loop over the axes and layouts
    for axis in axes:
        for layout in layouts:
            output_path = path + f"Run_{index}/Layout_{layout}/Axis_{axis}/"
            print("------------------------------------------------------")
            print("------------------------------------------------------")
            print(f"RUN: {index}")
            print(f"LAYOUT: {layout}")
            print(f"AXIS: {axis}")
            print("------------------------------------------------------")
            print(f"Thrust magnitude: {T} m/s^2")
            print(f"Duration: {duration} s")
            print(f"Upper bound: {f_upper_bound} Hz")
            print(f"Arm length: {L} m")
            print(f"Bandwidth: {bandwidth} units")
            print("------------------------------------------------------")
            print("------------------------------------------------------")

            dict_to_write = {"Run": index,
                             "Layout": layout,
                             "Axis": axis,
                             "Thrust magnitude": T,
                             "Duration": duration,
                             "Upper bound": f_upper_bound,
                             "Arm length": L,
                             "Bandwidth": bandwidth}

            # Create the accelerometers with the shaking signal
            acc_lst, linear_shaking_series, angular_shaking_series = acceleration_setup(shaking_dictionary=shaking_dictionary,
                                                                                        position_dictionary=position_dictionary[axis][layout],
                                                                                        duration=duration, layout=layout, noise_switch=noise_switch,
                                                                                        shake_bool=shake_bool, goce_parameters=False,
                                                                                        load_true_data=False)

            start = time.time()
            x0, x_true, digit_loss = run_calibration(acc_lst, layout, NFFT, noise_switch, output_path)
            end = time.time()

            # Create the accelerometers without the shaking signal for two days
            acc_lst_science = acceleration_setup(shaking_dictionary=shaking_dictionary,
                                                 position_dictionary=position_dictionary[axis][layout],
                                                 layout=layout, duration=172800, noise_switch=noise_switch,
                                                 shake_bool=False, goce_parameters=False,
                                                 load_true_data=True, x_true=x_true, acc_lst_dummy=acc_lst)

            # Calculate the estimate of a_ng
            a_ng_rcst = reconstruct_a_ng(x0, acc_lst_science, layout)

            # Save true a_ng
            np.savetxt(output_path + "a_ng_true.txt", acc_lst_science[0].a_ng)
            # Save reconstructed a_ng
            np.savetxt(output_path + "a_ng_rcst.txt", a_ng_rcst)
            # Save the info dictionary
            dict_to_write["Digit loss"] = digit_loss
            dict_to_write["Calibration time"] = end - start
            save2txt(dict_to_write, "Info.txt", output_path)
            # Save linear and angular shaking series
            np.savetxt(output_path + "linear_shaking_series.txt", linear_shaking_series)
            np.savetxt(output_path + "angular_shaking_series.txt", angular_shaking_series)

total_end = time.time()
print(f"Total time: {total_end - total_start} s")
np.savetxt(path + "Total_time.txt", np.array([total_end - total_start]))

