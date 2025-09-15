import numpy as np
from scipy.io import loadmat
from Tools.SignalProcessingUtilities import create_thrust_noise, create_attitude_sensor_noise, create_validation_accelerometer_configuration
from scipy.fft import rfftfreq
from Tools.PlotTools import create_plots, save_figure, plt

#########################
# Load the simulation data
#########################
CAL = loadmat("Input_data/goce_parameters.mat")['CAL']
state_history = np.loadtxt("SimulationOutput/Output/Orbit_data/state_history.dat")


w_true = loadmat("SimulationOutput/Output/Orbit_data/angular_rates.mat")['w']
dw_true = loadmat("SimulationOutput/Output/Orbit_data/angular_accelerations.mat")['dw']
a_ng = np.loadtxt("SimulationOutput/Output/Orbit_data/a_ng_body.txt")

# Extract time vector
t = state_history[:, 0] - state_history[0, 0]
# Convert time to hours
t = t / 3600

# Load the non-gravitational acceleration

# a_ng[:, 0] = 0  # Set the x component to zero. Drag compensated on the x-axis


##########################################
# Generate thruster noise and shaking data
##########################################
M_thruster = 27005 * 9   # Filter length
# 1) Thruster noise
freq_noise = rfftfreq(M_thruster, 1 / 10)    # Frequency vector. Sampling frequency is 10 Hz. The noise is downsampled to 1 Hz
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
M_acc = 5501*9
freq_shaking = rfftfreq(M_acc, 1)

# Add the shaking to the non-gravitational acceleration. Add thruster noise in x-axis
# a_ng[:, 0] += thruster_noise.noise[:, 0]
# Remove the mean
# a_ng = a_ng - np.mean(a_ng, axis=0)
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

w_meas = w_true + w_noise
dw_meas = dw_true + dw_noise
w_lst = [w_true, w_meas]
dw_lst = [dw_true, dw_meas]


load_data = False
L = 2*0.5
iter_quadratic = 3
degrees = [60, 70, 80, 90, 100, 110, 120, 130, 140]
psd_points = 5501 * 3
degrees.reverse()
axis = ["x", "y", "z"]


pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([-L / 2, 0, 0])}
V = np.zeros((len(t), 6))
acc_lst = create_validation_accelerometer_configuration(CAL, V, w_lst, dw_lst, a_ng, pos_dict, acc_noise_psd, 1, M_acc, layout=1, noise_switch=1)
acc1 = acc_lst[0]

cmap = plt.get_cmap('inferno')
colors = np.flip(cmap(np.linspace(0.1, 0.8, len(degrees))), axis=0)


fig1, ax1 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
fig2, ax2 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
fig3, ax3 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
fig4, ax4 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
fig5, ax5 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
fig6, ax6 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
ax_lst = [ax1, ax2, ax3, ax4, ax5, ax6]

fig9, ax9 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
ff_ang, ang_psd_x = acc1.psd_welch(a_ng[:, 0], psd_points)
_, ang_psd_y = acc1.psd_welch(a_ng[:, 1], psd_points)
_, ang_psd_z = acc1.psd_welch(a_ng[:, 2], psd_points)
ax9.plot(ff_ang, np.sqrt(ang_psd_x), label='Non-gravitational acceleration')
ax9.plot(ff_ang, np.sqrt(ang_psd_y), label='Non-gravitational acceleration')
ax9.plot(ff_ang, np.sqrt(ang_psd_z), label='Non-gravitational acceleration')
ax9.legend()
# ax9.set_ylim([1e-9, 1e-3])
save_figure(fig9, "Sensitivity_analysis/gravity_gradients/Non_gravitational_acceleration.svg")


for acc_ax in axis:
    fig0, ax0 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
    fig7, ax7 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
    fig8, ax8 = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True)
    if acc_ax == 'x':
        pos_dict = {"pos_acc1": np.array([L / 2, 0, 0]), "pos_acc2": np.array([-L / 2, 0, 0])}
    elif acc_ax == 'y':
        pos_dict = {"pos_acc1": np.array([0, L/2, 0]), "pos_acc2": np.array([0, -L/2, 0])}
    else:
        pos_dict = {"pos_acc1": np.array([0, 0, L/2]), "pos_acc2": np.array([0, 0, -L/2])}

    # Update the accelerometer positions
    acc1.r = pos_dict["pos_acc1"]
    ##########################################
    # Calculate the Euler Acceleration
    ##########################################

    Eu1 = -np.array([np.zeros(len(t)), -dw_true[:, 2], dw_true[:, 1]]).T
    Eu2 = -np.array([dw_true[:, 2], np.zeros(len(t)), -dw_true[:, 0]]).T
    Eu3 = -np.array([-dw_true[:, 1], dw_true[:, 0], np.zeros(len(t))]).T

    Ce1 = -np.array([w_true[:, 1] ** 2 + w_true[:, 2] ** 2, -(w_true[:, 0] * w_true[:, 1]), -(w_true[:, 0] * w_true[:, 2])]).T
    Ce2 = -np.array([-(w_true[:, 0] * w_true[:, 1]), w_true[:, 0] ** 2 + w_true[:, 2] ** 2, -(w_true[:, 1] * w_true[:, 2])]).T
    Ce3 = -np.array([-(w_true[:, 0] * w_true[:, 2]), -(w_true[:, 1] * w_true[:, 2]), w_true[:, 0] ** 2 + w_true[:, 1] ** 2]).T

    acceleration_euler = Eu1 * acc1.r[0] + Eu2 * acc1.r[1] + Eu3 * acc1.r[2]
    acceleration_centrifugal = Ce1 * acc1.r[0] + Ce2 * acc1.r[1] + Ce3 * acc1.r[2]

    ffe, aeu_psd_x = acc1.psd_welch(acceleration_euler[:, 0], psd_points)
    _, aeu_psd_y = acc1.psd_welch(acceleration_euler[:, 1], psd_points)
    _, aeu_psd_z = acc1.psd_welch(acceleration_euler[:, 2], psd_points)

    ffc, ace_psd_x = acc1.psd_welch(acceleration_centrifugal[:, 0], psd_points)
    _, ace_psd_y = acc1.psd_welch(acceleration_centrifugal[:, 1], psd_points)
    _, ace_psd_z = acc1.psd_welch(acceleration_centrifugal[:, 2], psd_points)

    ax7.plot(ffe, np.sqrt(aeu_psd_x), label='x')
    ax7.plot(ffe, np.sqrt(aeu_psd_y), label='y')
    ax7.plot(ffe, np.sqrt(aeu_psd_z), label='z')

    ax8.plot(ffc, np.sqrt(ace_psd_x), label='x')
    ax8.plot(ffc, np.sqrt(ace_psd_y), label='y')
    ax8.plot(ffc, np.sqrt(ace_psd_z), label='z')


    for i, deg in enumerate(degrees):
        # Load gravity gradients and angular rates
        gravity_gradients = loadmat(f"SimulationOutput/Output/Gravity_gradients/gravity_gradients_{deg}.mat")['gravity_gradients']
        Vxx = gravity_gradients[0]['Vxx'][0]
        Vyy = gravity_gradients[0]['Vyy'][0]
        Vzz = gravity_gradients[0]['Vzz'][0]
        Vxy = gravity_gradients[0]['Vxy'][0]
        Vxz = gravity_gradients[0]['Vxz'][0]
        Vyz = gravity_gradients[0]['Vyz'][0]
        V = np.hstack((Vxx, Vyy, Vzz, Vxy, Vxz, Vyz))

        # Calculate the contribution from just the gravity gradients
        G1 = -np.array([V[:, 0], V[:, 3], V[:, 4]]).T
        G2 = -np.array([V[:, 3], V[:, 1], V[:, 5]]).T
        G3 = -np.array([V[:, 4], V[:, 5], V[:, 2]]).T

        acceleration_gravity_gradient = G1 * acc1.r[0] + G2 * acc1.r[1] + G3 * acc1.r[2]

        ffx, acx = acc1.psd_welch(acceleration_gravity_gradient[:, 0], psd_points)

        ff_nx, ac_nx = acc1.psd_welch(acc1.noise[:, 0], psd_points)
        if deg == degrees[0]:
            ax0.plot(ff_nx, np.sqrt(ac_nx), label='Accelerometer Noise', color='dimgray')
        ax0.plot(ffx, np.sqrt(acx), label=f'D/O = {deg}', color=colors[i])

        if acc_ax == "x":
            axx = np.reshape(-Vxx * acc1.r[0], -1)
            axy = np.reshape(-Vxy * acc1.r[0], -1)
            axz = np.reshape(-Vxz * acc1.r[0], -1)

            ff, axx_psd = acc1.psd_welch(axx, psd_points)
            _, axy_psd = acc1.psd_welch(axy, psd_points)
            _, axz_psd = acc1.psd_welch(axz, psd_points)

            if deg == degrees[0]:
                ax1.plot(ff_nx, np.sqrt(ac_nx), label='Accelerometer Noise', color='dimgray')
                ax2.plot(ff_nx, np.sqrt(ac_nx), label='Accelerometer Noise', color='dimgray')
                ax3.plot(ff_nx, np.sqrt(ac_nx), label='Accelerometer Noise', color='dimgray')
            ax1.plot(ff, np.sqrt(axx_psd), label=f'D/O = {deg}', color=colors[i])
            ax2.plot(ff, np.sqrt(axy_psd), label=f'D/O = {deg}', color=colors[i])
            ax3.plot(ff, np.sqrt(axz_psd), label=f'D/O = {deg}', color=colors[i])
        elif acc_ax == "y":
            ayy = np.reshape(-Vyy * acc1.r[1], -1)
            ayz = np.reshape(-Vyz * acc1.r[1], -1)

            ff, ayy_psd = acc1.psd_welch(ayy, psd_points)
            _, ayz_psd = acc1.psd_welch(ayz, psd_points)

            if deg == degrees[0]:
                ax4.plot(ff_nx, np.sqrt(ac_nx), label='Accelerometer Noise', color='dimgray')
                ax5.plot(ff_nx, np.sqrt(ac_nx), label='Accelerometer Noise', color='dimgray')
            ax4.plot(ff, np.sqrt(ayy_psd), label=f'D/O = {deg}', color=colors[i])
            ax5.plot(ff, np.sqrt(ayz_psd), label=f'D/O = {deg}', color=colors[i])

        else:
            azz = np.reshape(-Vzz * acc1.r[2], -1)

            ff, azz_psd = acc1.psd_welch(azz, psd_points)

            if deg == degrees[0]:
                ax6.plot(ff_nx, np.sqrt(ac_nx), label='Accelerometer Noise', color='dimgray')
            ax6.plot(ff, np.sqrt(azz_psd), label=f'D/O = {deg}', color=colors[i])

        for acc in acc_lst:
            acc.w_noise = w_noise
            acc.dw_noise = dw_noise

    ax0.legend()
    ax7.legend()
    ax8.legend()
    save_figure(fig0, f"Sensitivity_analysis/gravity_gradients/DO_sensitivity_{acc_ax}.svg")
    save_figure(fig7, f"Sensitivity_analysis/gravity_gradients/Euler_acceleration_{acc_ax}.svg")
    save_figure(fig8, f"Sensitivity_analysis/gravity_gradients/Centrifugal_acceleration_{acc_ax}.svg")

for ax in ax_lst:
    ax.legend()
    ax.set_ylim([1e-15, 1e-9])
    ax.set_xlim([5e-3, 0.1])

save_figure(fig1, f"Sensitivity_analysis/gravity_gradients/DO_sensitivity_Vxx.svg")
save_figure(fig2, f"Sensitivity_analysis/gravity_gradients/DO_sensitivity_Vxy.svg")
save_figure(fig3, f"Sensitivity_analysis/gravity_gradients/DO_sensitivity_Vxz.svg")
save_figure(fig4, f"Sensitivity_analysis/gravity_gradients/DO_sensitivity_Vyy.svg")
save_figure(fig5, f"Sensitivity_analysis/gravity_gradients/DO_sensitivity_Vyz.svg")
save_figure(fig6, f"Sensitivity_analysis/gravity_gradients/DO_sensitivity_Vzz.svg")
