import numpy as np
from scipy.io import loadmat
from Tools.CalibrationTools import reconstruct_a_ng
from Tools.SignalProcessingUtilities import create_thrust_noise, create_attitude_sensor_noise,\
    create_thrust_series, create_validation_accelerometer_configuration
from scipy.fft import rfftfreq
from tudatpy.kernel import constants
from Tools.VerficationTools import verify_lstsq, parameter_estimation
from Tools.PlotTools import create_plots, save_figure

np.random.seed(11)
#########################
# Load the simulation data
#########################
t_len = 86400
CAL = loadmat("../Input_data/goce_parameters.mat")['CAL']
state_history = np.loadtxt("../SimulationOutput/Output/Orbit_data/state_history.dat")[0:t_len, :]
deps_var = np.loadtxt("../SimulationOutput/Output/Orbit_data/dependent_variable_history.dat")[0:t_len, :]

# Load gravity gradients and angular rates
gravity_gradients = loadmat("../SimulationOutput/Output/Orbit_data/gravity_gradients_130.mat")['gravity_gradients']
Vxx = gravity_gradients[0]['Vxx'][0][0:t_len]
Vyy = gravity_gradients[0]['Vyy'][0][0:t_len]
Vzz = gravity_gradients[0]['Vzz'][0][0:t_len]
Vxy = gravity_gradients[0]['Vxy'][0][0:t_len]
Vxz = gravity_gradients[0]['Vxz'][0][0:t_len]
Vyz = gravity_gradients[0]['Vyz'][0][0:t_len]
V = np.hstack((Vxx, Vyy, Vzz, Vxy, Vxz, Vyz))

w_true = loadmat("../SimulationOutput/Output/Orbit_data/angular_rates.mat")['w'][0:t_len, :]
dw_true = loadmat("../SimulationOutput/Output/Orbit_data/angular_accelerations.mat")['dw'][0:t_len, :]
a_ng = np.loadtxt("../SimulationOutput/Output/Orbit_data/a_ng_body.txt")[0:t_len, :]

# Extract time vector
t = state_history[:, 0] - state_history[0, 0]
# Convert time to hours
t = t / 3600

a_ng[:, 0] = 0  # Set the x component to zero. Drag compensated on the x-axis

# Parameters for shaking
linear_shaking_dict = {}
angular_shaking_dict = {}
linear_shaking_dict['x'] = [True, 3e-6, 0.06, 0.1, constants.JULIAN_DAY, 0]
linear_shaking_dict['y'] = [True, 3e-6, 0.06, 0.1, constants.JULIAN_DAY, 0]
linear_shaking_dict['z'] = [True, 3e-6, 0.06, 0.1, constants.JULIAN_DAY, 0]

angular_shaking_dict['x'] = [True, 3e-6, 0.06, 0.1, constants.JULIAN_DAY, 0]
angular_shaking_dict['y'] = [True, 3e-6, 0.06, 0.1, constants.JULIAN_DAY, 0]
angular_shaking_dict['z'] = [True, 3e-6, 0.06, 0.1, constants.JULIAN_DAY, 0]

##########################################
# Generate thruster noise and shaking data
##########################################
M_thruster = (86400//3 * 9)   # Filter length
M_thruster = M_thruster if M_thruster % 2 == 1 else M_thruster + 1
# 1) Thruster noise
freq_noise = rfftfreq(M_thruster, 1)    # Frequency vector. Sampling frequency is 10 Hz. The noise is downsampled to 1 Hz
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
thruster_noise_psd = thruster_noise_psd ** 2    # Convert to PSD
thruster_noise_psd[0] = 0   # Set the DC component to zero

thruster_noise = create_thrust_noise(thruster_noise_psd, t, M_thruster)

# 2) Create the shaking data. Sampling frequency is 1 Hz
M_acc = 86400//3
M_acc = M_acc if M_acc % 2 == 1 else M_acc + 1
freq_shaking = rfftfreq(M_acc, 1)

linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking = create_thrust_series(linear_shaking_dict,
                                                                                                        angular_shaking_dict,
                                                                                                        freq_shaking, M_acc, len(t))


# Add the shaking to the non-gravitational acceleration. Add thruster noise in x-axis
a_ng += thruster_noise.noise
a_ng += linear_acceleration_shaking
# Remove the mean
a_ng = a_ng - np.mean(a_ng, axis=0)
# Add the shaking to the true angular rate and angular acceleration
angular_rates_shaking -= np.mean(angular_rates_shaking, axis=0)
w_true += angular_rates_shaking
dw_true += angular_acceleration_shaking
###################################################################
# Create accelerometer noise and angular rate noise
# !!! Accelerometer noise is provided in "create_configuration"
###################################################################
freq_acc = rfftfreq(M_acc, 1)
# 1) Accelerometer noise
acc_noise_psd = (2e-12 * np.sqrt(1.2 + np.where(freq_acc != 0, 0.002 / freq_acc, np.inf) + 6000 * freq_acc**4))**2
# Set the DC component to zero
acc_noise_psd[0] = 0

# 2) Angular rate and angular acceleration noise
star_noise_psd = (8.5e-6 / np.sqrt(freq_acc)) ** 2
star_noise_psd[0] = 0
acc_noise_psd_ang = (1e-10 * np.sqrt(0.4 + 0.001/freq_acc + 2500*freq_acc**4))**2
acc_noise_psd_ang[0] = 0
# Sensor fusion
noise_psd_dw = 1 / (1/acc_noise_psd_ang + 1/(star_noise_psd*(2*np.pi*freq_acc)**4))

w_noise, dw_noise = create_attitude_sensor_noise(noise_psd_dw, len(t), M_acc)


load_data = False
L = 2*0.25
layouts = [2]
noise_switch = [1]
axis = ['x', 'y', 'z']
print("Least squares verification")
layout_counter = 0
for layout in layouts:
    if layout == 1:
        pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([-L / 2, 0, 0])}
        # pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([-L / 2, 0, 0])}
        # pos_dict = {"pos_acc1": np.array([L 0, 0, L/2]), "pos_acc2": np.array([0, 0, -L/2])}
    elif layout == 2:
        # pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([-L / 2, 0, 0])}
        pos_dict = {"pos_acc1": np.array([0, L/2, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, -L/2, 0])}
        # pos_dict = {"pos_acc1": np.array([L, 0, L/2]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, 0, -L/2])}
    else:
        pos_dict = {"pos_acc2": np.array([L / 2, 0, 0]), "pos_acc1": np.array([0, L/2, 0]), "pos_acc4": np.array([-L / 2, 0, 0]),
                    "pos_acc3": np.array([0, -L/2, 0])}
    for n in noise_switch:
        print(f"\nLayout {layout}, Noise switch {n}")
        # Add angular rate noise to true angular rate
        if n == 0:
            w_meas = w_true
            dw_meas = dw_true
        else:
            w_meas = w_true + w_noise
            dw_meas = dw_true + dw_noise
        w_lst = [w_true, w_meas]
        dw_lst = [dw_true, dw_meas]

        acc_lst = create_validation_accelerometer_configuration(CAL, V, w_lst, dw_lst, a_ng, pos_dict, acc_noise_psd, 1, M_acc, layout=layout,
                                                                noise_switch=n)

        verify_lstsq(acc_lst, layout=layout, noise_switch=n, load_data=load_data, validation=True)

        parameter_estimation(layout, noise_switch=True, validation=True, sensitivity_analysis=False)

    # Calculate the non-gravitational acceleration in science mode
    # Remove the shaking data
    if layout_counter == 0:
        a_ng_shakeless = a_ng - linear_acceleration_shaking
        w_shakeless = w_true - angular_rates_shaking
        dw_shakeless = dw_true - angular_acceleration_shaking

        w_meas_shakeless = w_shakeless + w_noise
        dw_meas_shakeless = dw_shakeless + dw_noise
        w_lst_shakeless = [w_shakeless, w_meas_shakeless]
        dw_lst_shakeless = [dw_shakeless, dw_meas_shakeless]

    acc_lst_shakeless = create_validation_accelerometer_configuration(CAL, V, w_lst_shakeless, dw_lst_shakeless, a_ng_shakeless, pos_dict,
                                                                      acc_noise_psd, 1, M_acc, layout=layout,
                                                                      noise_switch=1)

    # Calculate the reconstructed non-gravitational acceleration
    x0 = np.loadtxt(f"../SimulationOutput/Output/Validation/Least_squares/Configuration_{layout}/Noisy/x0.txt")

    acc1_shakeless = acc_lst_shakeless[0]
    psd_points = 5401 * 3

    a_ng_measured = reconstruct_a_ng(x0, acc_lst_shakeless, layout=layout)
    a_ng_residual = a_ng_measured - a_ng_shakeless

    # Project the error on the LoS vector cone. The body frame x-axis is already along the LoS vector. So only worst case scenarios apply
    a_ng_residual_relative = np.sqrt(2) * a_ng_residual

    a_ng_residual_relative_los = a_ng_residual_relative[:, 0]

    a_ng_worst_case = a_ng_residual_relative
    a_ng_worst_case[:, 1:3] = a_ng_worst_case[:, 1:3] * 1e-5
    a_ng_worst_case_norm = np.linalg.norm(a_ng_worst_case, axis=1)

    ff_ang, ang_nominal = acc1_shakeless.psd_welch(a_ng_residual[:, 0], psd_points)
    ff_ang_worst, ang_worst = acc1_shakeless.psd_welch(a_ng_worst_case[:, 0], psd_points)

    # Create the goal asd

    # 1) Accelerometer noise
    a_ng_requirement = (5e-12 * np.sqrt(1 + np.where(freq_acc != 0, 0.001 / freq_acc, np.inf) ** 2 + (100 * freq_acc ** 2) ** 2))
    # Set the DC component to zero
    a_ng_requirement[0] = 0
    fig3, ax3 = create_plots("Frequency [Hz]", "ASD [m/s^2/sqrt(Hz)]", x_scale_log=True, y_scale_log=True)
    ax3.plot(ff_ang, np.sqrt(ang_nominal), label='Nominal')
    ax3.plot(ff_ang_worst, np.sqrt(ang_worst), label='Worst case')
    ax3.plot(freq_acc, a_ng_requirement, label='Requirement')
    ax3.legend()
    save_figure(fig3, f"Validation/reconstructed_a_ng_ASD_layout_{layout}.svg")

    layout_counter += 1



acc1 = acc_lst[0]
psd_points = 5401*3
ffx, acx = acc1.psd_welch(acc1.a_meas[:, 0], psd_points)
ffy, acy = acc1.psd_welch(acc1.a_meas[:, 1], psd_points)
ffz, acz = acc1.psd_welch(acc1.a_meas[:, 2], psd_points)

fig, ax = create_plots("Frequency [Hz]", "ASD [m/s^2/sqrt(Hz)]", x_scale_log=True, y_scale_log=True)
ax.plot(ffx, np.sqrt(acx), label='ax')
ax.plot(ffy, np.sqrt(acy), label='ay')
ax.plot(ffz, np.sqrt(acz), label='az')
ax.legend()

ffx, dwx = acc1.psd_welch(acc1.dw_meas[:, 0], psd_points)
ffy, dwy = acc1.psd_welch(acc1.dw_meas[:, 1], psd_points)
ffz, dwz = acc1.psd_welch(acc1.dw_meas[:, 2], psd_points)

fig1, ax1 = create_plots("Frequency [Hz]", "ASD [rad/s^2/sqrt(Hz)]", x_scale_log=True, y_scale_log=True)
ax1.plot(ffx, np.sqrt(dwx), label='dwx')
ax1.plot(ffy, np.sqrt(dwy), label='dwy')
ax1.plot(ffz, np.sqrt(dwz), label='dwz')
ax1.legend()

ffx, wx = acc1.psd_welch(acc1.w_meas[:, 0], psd_points)
ffy, wy = acc1.psd_welch(acc1.w_meas[:, 1], psd_points)
ffz, wz = acc1.psd_welch(acc1.w_meas[:, 2], psd_points)

fig2, ax2 = create_plots("Frequency [Hz]", "ASD [rad/s/sqrt(Hz)]", x_scale_log=True, y_scale_log=True)
ax2.plot(ffx, np.sqrt(wx), label='wx')
ax2.plot(ffy, np.sqrt(wy), label='wy')
ax2.plot(ffz, np.sqrt(wz), label='wz')
ax2.legend()


save_figure(fig, f"Validation/linear_acceleration_ASD.svg")
save_figure(fig1, f"Validation/angular_acceleration_ASD.svg")
save_figure(fig2, f"Validation/angular_rate_ASD.svg")






