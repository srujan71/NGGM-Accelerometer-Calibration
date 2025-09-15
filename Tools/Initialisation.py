import numpy as np
from scipy.io import loadmat
from scipy.fft import rfftfreq
from Tools.SignalProcessingUtilities import create_attitude_sensor_noise, create_thrust_noise, create_thrust_series, create_configuration, \
    create_validation_accelerometer_configuration, load_configuration
from Tools.CalibrationTools import acc_cal_par_vec_to_mat


def acceleration_setup(shaking_dictionary, position_dictionary, duration, layout, noise_switch, shake_bool,
                       goce_parameters=False, load_true_data=False, **kwargs):
    #########################
    # Load the simulation data
    #########################
    if load_true_data:
        acc_lst_dummy = kwargs["acc_lst_dummy"]
        x_true = kwargs["x_true"]
        match layout:
            case 1:
                M1, M2, K1, K2, W1, W2, dr1, dr2 = acc_cal_par_vec_to_mat(x_true, acc_lst_dummy, layout)
                PAR = dict()
                PAR['M1'] = M1
                PAR['M2'] = M2
                PAR['K1'] = K1
                PAR['K2'] = K2
                PAR['W1'] = W1
                PAR['W2'] = W2
                PAR['dr1'] = dr1
                PAR['dr2'] = dr2
            case 2:
                M1, M2, M3, K1, K2, K3, W1, W2, W3, dr1, dr2, dr3 = acc_cal_par_vec_to_mat(x_true, acc_lst_dummy, layout)
                PAR = dict()
                PAR['M1'] = M1
                PAR['M2'] = M2
                PAR['M3'] = M3
                PAR['K1'] = K1
                PAR['K2'] = K2
                PAR['K3'] = K3
                PAR['W1'] = W1
                PAR['W2'] = W2
                PAR['W3'] = W3
                PAR['dr1'] = dr1
                PAR['dr2'] = dr2
                PAR['dr3'] = dr3
            case 3:
                M1, M2, M3, M4, K1, K2, K3, K4, W1, W2, W3, W4, dr1, dr2, dr3, dr4 = acc_cal_par_vec_to_mat(x_true, acc_lst_dummy, layout)
                PAR = dict()
                PAR['M1'] = M1
                PAR['M2'] = M2
                PAR['M3'] = M3
                PAR['M4'] = M4
                PAR['K1'] = K1
                PAR['K2'] = K2
                PAR['K3'] = K3
                PAR['K4'] = K4
                PAR['W1'] = W1
                PAR['W2'] = W2
                PAR['W3'] = W3
                PAR['W4'] = W4
                PAR['dr1'] = dr1
                PAR['dr2'] = dr2
                PAR['dr3'] = dr3
                PAR['dr4'] = dr4


    state_history = np.loadtxt("../SimulationOutput/Output/Orbit_data/state_history.dat")[:duration+1, :]
    # deps_var = np.loadtxt("../SimulationOutput/Output/Orbit_data/dependent_variable_history.dat")[:duration+1, :]

    # Load gravity gradients and angular rates
    gravity_gradients = loadmat("../SimulationOutput/Output/Orbit_data/gravity_gradients_130.mat")['gravity_gradients']
    Vxx = gravity_gradients[0]['Vxx'][0][:duration+1]
    Vyy = gravity_gradients[0]['Vyy'][0][:duration+1]
    Vzz = gravity_gradients[0]['Vzz'][0][:duration+1]
    Vxy = gravity_gradients[0]['Vxy'][0][:duration+1]
    Vxz = gravity_gradients[0]['Vxz'][0][:duration+1]
    Vyz = gravity_gradients[0]['Vyz'][0][:duration+1]
    V = np.hstack((Vxx, Vyy, Vzz, Vxy, Vxz, Vyz))

    w_true = loadmat("../SimulationOutput/Output/Orbit_data/angular_rates.mat")['w'][:duration+1, :]
    dw_true = loadmat("../SimulationOutput/Output/Orbit_data/angular_accelerations.mat")['dw'][:duration+1, :]
    a_ng = np.loadtxt("../SimulationOutput/Output/Orbit_data/a_ng_body.txt")[:duration+1, :]

    # Extract time vector
    t = state_history[:, 0] - state_history[0, 0]
    # Convert time to hours
    t = t / 3600

    a_ng[:, 0] = 0  # Set the x component to zero. Drag compensated on the x-axis



    ##########################################
    # Generate thruster noise and shaking data
    ##########################################
    M_thruster = (duration//3 * 9)   # Filter length
    M_thruster = M_thruster if M_thruster % 2 == 1 else M_thruster + 1
    # 1) Thruster noise
    freq_noise = rfftfreq(M_thruster, 1)  # Frequency vector. Sampling frequency is 10 Hz. The noise is downsampled to 1 Hz
    thruster_noise_psd = np.zeros((len(freq_noise)))
    # For thruster noise, build the ASD first as it is given in Newtons. Then convert to m/s^2 and then to PSD.
    # Condition 1: freq_noise < 0.3e-3
    mask_1 = freq_noise < 0.3e-3
    thruster_noise_psd[mask_1] = 100e-6
    # Condition 2: 0.3e-3 < freq_noise < 0.03
    mask_2 = (freq_noise > 0.3e-3) & (freq_noise < 0.03)
    thruster_noise_psd[mask_2] = np.power(10, np.interp(np.log10(freq_noise[mask_2]), np.log10([0.3e-3, 0.03]), np.log10([100e-6, 1e-6])))
    # Condition 3: freq_noise >= 0.03
    mask_3 = freq_noise >= 0.03
    thruster_noise_psd[mask_3] = 1e-6

    thruster_noise_psd /= 1000  # Convert to m/s^2
    thruster_noise_psd = thruster_noise_psd ** 2  # Convert to PSD
    thruster_noise_psd[0] = 0  # Set the DC component to zero

    thruster_noise = create_thrust_noise(thruster_noise_psd, t, M_thruster)

    # 2) Create the shaking data. Sampling frequency is 1 Hz
    if duration < 10000:
        M_acc = 5500
    elif 10000 < duration < 27000:
        M_acc = 11000
    else:
        M_acc = 27000
    M_acc = M_acc if M_acc % 2 == 1 else M_acc + 1
    freq_shaking = rfftfreq(M_acc, 1)

    if shake_bool:
        # Parameters for shaking
        linear_shaking_dict = shaking_dictionary['linear']
        angular_shaking_dict = shaking_dictionary['angular']
        linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking = create_thrust_series(linear_shaking_dict,
                                                                                                                angular_shaking_dict,
                                                                                                                freq_shaking, M_acc, len(t))
        # Add the shaking to the non-gravitational acceleration.
        a_ng += linear_acceleration_shaking
        # Add the shaking to the true angular rate and angular acceleration
        angular_rates_shaking -= np.mean(angular_rates_shaking, axis=0)     # Remove the mean
        w_true += angular_rates_shaking
        dw_true += angular_acceleration_shaking

    # Add thruster noise in x-axis
    a_ng += thruster_noise.noise
    # Remove the mean
    a_ng = a_ng - np.mean(a_ng, axis=0)

    ###################################################################
    # Create accelerometer noise and angular rate noise
    # !!! Accelerometer noise is provided in "create_configuration"
    ###################################################################
    freq_acc = rfftfreq(M_acc, 1)
    # 1) Accelerometer noise
    acc_noise_psd = np.empty((len(freq_acc)))
    non_zero_idx = np.where(freq_acc != 0)
    acc_noise_psd[non_zero_idx] = (2e-12 * np.sqrt(1.2 + 0.002 / freq_acc[non_zero_idx] + 6000 * freq_acc[non_zero_idx] ** 4)) ** 2
    # Set the DC component to zero
    acc_noise_psd[0] = 0

    # 2) Angular rate and angular acceleration noise
    star_noise_psd = np.empty((len(freq_acc)))
    star_noise_psd[non_zero_idx] = (8.5e-6 / np.sqrt(freq_acc[non_zero_idx])) ** 2
    star_noise_psd[0] = 0

    acc_noise_psd_ang = np.empty((len(freq_acc)))
    acc_noise_psd_ang[non_zero_idx] = (1e-10 * np.sqrt(0.4 + 0.001 / freq_acc[non_zero_idx] + 2500 * freq_acc[non_zero_idx] ** 4)) ** 2
    acc_noise_psd_ang[0] = 0
    # Sensor fusion
    noise_psd_dw = 1 / (1 / acc_noise_psd_ang + 1 / (star_noise_psd * (2 * np.pi * freq_acc) ** 4))
    noise_psd_dw[0] = 0

    w_noise, dw_noise = create_attitude_sensor_noise(noise_psd_dw, len(t), M_acc)


    if noise_switch == 0:
        w_meas = w_true
        dw_meas = dw_true
    else:
        w_meas = w_true + w_noise
        dw_meas = dw_true + dw_noise

    w_lst = [w_true, w_meas]
    dw_lst = [dw_true, dw_meas]

    if goce_parameters:
        CAL = loadmat("../Input_data/goce_parameters.mat")['CAL']
        acc_lst = create_validation_accelerometer_configuration(CAL, V, w_lst, dw_lst, a_ng, position_dictionary,
                                                                acc_noise_psd, 1, M_acc, layout=layout, noise_switch=noise_switch)
    elif load_true_data:
        if 'gg' in kwargs:
            acc_lst = load_configuration(PAR, V, w_lst, dw_lst, a_ng, position_dictionary, acc_noise_psd, 1, M_acc, layout=layout, noise_switch=noise_switch, gg=True)

            return acc_lst

        elif 'gg_w_rate' in kwargs:
            acc_lst = load_configuration(PAR, V, w_lst, dw_lst, a_ng, position_dictionary, acc_noise_psd, 1, M_acc, layout=layout, noise_switch=noise_switch, gg_w_rate=True)
        else:
            acc_lst = load_configuration(PAR, V, w_lst, dw_lst, a_ng, position_dictionary, acc_noise_psd, 1, M_acc, layout=layout, noise_switch=noise_switch)
    else:
        acc_lst = create_configuration(V, w_lst, dw_lst, a_ng, position_dictionary, acc_noise_psd, 1, M_acc, layout=layout, noise_switch=noise_switch)

    if shake_bool:
        return acc_lst, linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking
    else:
        return acc_lst
