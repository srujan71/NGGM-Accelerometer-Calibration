from Tools.VerficationTools import verify_lstsq, verify_design_matrix, singular_matrix_analysis
import numpy as np
from scipy.io import loadmat
from scipy.fft import rfftfreq
from Tools.SignalProcessingUtilities import create_configuration, Noise, integrate_trapezoid, create_thrust_noise, create_thrust_series
from Tools.PlotTools import create_plots, save_figure
from tudatpy.kernel import constants


np.random.seed(1)
t_len = 86400
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
w_true += angular_rates_shaking
dw_true += angular_acceleration_shaking

M = 27005
freq_acc = rfftfreq(M, 1)

# Accelerometer noise
acc_noise_psd = (2e-12 * np.sqrt(1.2 + 0.002/freq_acc + 6000*freq_acc**4))**2
acc_noise_psd[0] = 0

# Angular rate
# Star sensor noise
star_noise_psd = (8.5e-6 / np.sqrt(freq_acc)) ** 2
star_noise_psd[0] = 0

acc_noise_psd_ang = (1e-10 * np.sqrt(0.4 + 0.001/freq_acc + 2500*freq_acc**4))**2
acc_noise_psd_ang[0] = 0
# Sensor fusion
noise_psd_dw = 1 / (1/acc_noise_psd_ang + 1/(star_noise_psd*(2*np.pi*freq_acc)**4))

# !!! Have to provide 2 sided spectrum as rfft and irfft ignore the negative frequencies, but it does not double the positive frequencies
dw_noise = Noise(noise_psd_dw, 1)
dw_fil = dw_noise.correlation_filter(M)
dw_noise = dw_noise.filter_matrix(dw_fil, np.random.randn(len(t) + M-1, 3))
w_noise = integrate_trapezoid(dw_noise)


design_matrix = False
least_squares = True
load_data = True
L = 0.6
if design_matrix:

    w_meas = w_true + w_noise
    dw_meas = dw_true + dw_noise
    w_lst = [w_true, w_meas]
    dw_lst = [dw_true, dw_meas]

    layout = [2]

    print("Design matrix verification")
    for i, l in enumerate(layout):
        print(f"\nLayout {l}")
        if l == 1:
            # pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([-L / 2, 0, 0])}
            pos_dict = {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([0, -L / 2, 0])}
            # pos_dict = {"pos_acc1": np.array([0, 0, L/2]), "pos_acc2": np.array([0, 0, -L/2])}
            x0_len = 29
        elif l == 2:
            # pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([-L / 2, 0, 0])}
            pos_dict = {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, -L / 2, 0])}
            # pos_dict = {"pos_acc1": np.array([0, 0, L/2]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, 0, -L/2])}
            x0_len = 47
        else:
            pos_dict = {"pos_acc1": np.array([L/2, 0, 0]), "pos_acc2": np.array([0, L/2, 0]), "pos_acc3": np.array([-L/2, 0, 0]),
                        "pos_acc4": np.array([0, -L/2, 0])}
            # pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([0, 0, L / 2]), "pos_acc3": np.array([-L / 2, 0, 0]),
            #             "pos_acc4": np.array([0, 0, -L / 2])}
            x0_len = 64

        acc_lst = create_configuration(V, w_lst, dw_lst, a_ng, pos_dict, acc_noise_psd, 1, M, layout=l,
                                       noise_switch=1)
        x0 = np.random.randn(x0_len)
        verify_design_matrix(x0, acc_lst, a_ng, layout=l, load_data=load_data)

if least_squares:
    layout = [2]
    noise_switch = [0]
    print("Least squares verification")
    for l in layout:
        if l == 1:
            # pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([-L / 2, 0, 0])}
            pos_dict = {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([0, -L / 2, 0])}
            # pos_dict = {"pos_acc1": np.array([L 0, 0, L/2]), "pos_acc2": np.array([0, 0, -L/2])}
        elif l == 2:
            # pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([-L / 2, 0, 0])}
            pos_dict = {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, -L / 2, 0])}
            # pos_dict = {"pos_acc1": np.array([L 0, 0, L/2]), "pos_acc2": np.array([0, 0, 0]), "pos_acc3": np.array([0, 0, -L/2])}
        else:
            pos_dict = {"pos_acc1": np.array([0, L / 2, 0]), "pos_acc2": np.array([L / 2, 0, 0]), "pos_acc3": np.array([0, -L / 2, 0]),
                        "pos_acc4": np.array([-L / 2, 0, 0])}
        for n in noise_switch:
            print(f"\nLayout {l}, Noise switch {n}")
            if n == 0:
                w_meas = w_true
                dw_meas = dw_true
            else:
                w_meas = w_true + w_noise
                dw_meas = dw_true + dw_noise
            w_lst = [w_true, w_meas]
            dw_lst = [dw_true, dw_meas]

            acc_lst = create_configuration(V, w_lst, dw_lst, a_ng, pos_dict, acc_noise_psd, 1, M, layout=l,
                                           noise_switch=n)

            verify_lstsq(acc_lst, layout=l, noise_switch=n, load_data=load_data)
            # singular_matrix_analysis(layout=l)



NFFT = 5*5401

acc1 = acc_lst[0]
ff_psd, Px = acc1.psd_welch(acc1.noise[:, 0], NFFT)
ff_psd, Py = acc1.psd_welch(acc1.noise[:, 1], NFFT)
ff_psd, Pz = acc1.psd_welch(acc1.noise[:, 2], NFFT)
ff_psd_dw, Px_dw = acc1.psd_welch(dw_noise[:, 0], NFFT)
ff_psd_dw, Py_dw = acc1.psd_welch(dw_noise[:, 1], NFFT)
ff_psd_dw, Pz_dw = acc1.psd_welch(dw_noise[:, 2], NFFT)

fig, ax = create_plots("Frequency[Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", True, True)
ax.plot(ff_psd[1:], np.sqrt(Px[1:]), label=r"$n_{x}$")
ax.plot(ff_psd[1:], np.sqrt(Py[1:]), label=r"$n_{y}$")
ax.plot(ff_psd[1:], np.sqrt(Pz[1:]), label=r"$n_{z}$")
ax.plot(freq_acc[1:], np.sqrt(acc_noise_psd[1:]), label="Desired ASD", color='k')
ax.legend(fontsize=14)
ax.set_ylim([None, 1e-10])
save_figure(fig, "Verification/AccelerometerNoise_ASD.svg")

fig1, ax1 = create_plots("Frequency[Hz]", r"ASD [$rad/s^{2}/\sqrt{Hz}$]", True, True)
ax1.plot(ff_psd_dw[1:], np.sqrt(Px_dw[1:]), label=r"$n_{dw,x}$")
ax1.plot(ff_psd_dw[1:], np.sqrt(Py_dw[1:]), label=r"$n_{dw,y}$")
ax1.plot(ff_psd_dw[1:], np.sqrt(Pz_dw[1:]), label=r"$n_{dw,z}$")

ax1.plot(freq_acc[1:], np.sqrt(star_noise_psd[1:]*(2*np.pi*freq_acc[1:])**4), label="Star sensor noise", color='purple')
ax1.plot(freq_acc[1:], np.sqrt(acc_noise_psd_ang[1:]), label="Accelerometer noise", color='k')
ax1.plot(freq_acc[1:], np.sqrt(noise_psd_dw[1:]), label="Reconstructed", color='r')
ax1.legend()
ax1.set_ylim([1e-11, 1e-7])
save_figure(fig1, "Verification/AngularAccNoise_ASD.svg")
