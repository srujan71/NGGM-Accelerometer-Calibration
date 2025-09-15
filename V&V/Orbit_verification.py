import numpy as np
from Tools.PlotTools import plot_3d, create_plots, save_figure
from scipy.spatial.transform import Rotation as RS

# Load the dependent variables
n = 86400

deps_var = np.loadtxt("../SimulationOutput/Output/Orbit_data/dependent_variable_history.dat")[0:n, :]
state = np.loadtxt("../SimulationOutput/Output/Orbit_data/state_history.dat")[0:n, :]

time = deps_var[:, 0] - deps_var[0, 0]
time = time / 3600

xv = np.zeros(len(time))
yv = np.zeros(len(time))
zv = np.zeros(len(time))

xr = np.zeros(len(time))
yr = np.zeros(len(time))
zr = np.zeros(len(time))

x_new = np.zeros(len(time))
y_new = np.zeros(len(time))
z_new = np.zeros(len(time))

x_rel = np.zeros(len(time))
y_rel = np.zeros(len(time))
z_rel = np.zeros(len(time))

v_vector = np.zeros((len(time), 3))
quaternion = np.zeros((len(time), 4))
# For every epoch, rotate x and z axis of the body frame to the inertial frame
for i in range(len(time)):
    rot_mat_vector = deps_var[i, 1:10]
    R_BI = np.reshape(rot_mat_vector, (3, 3))
    R_IB = R_BI.T

    x_body = np.array([1, 0, 0])
    y_body = np.array([0, 1, 0])
    z_body = np.array([0, 0, 1])

    # Convert the body frame to inertial frame
    x_iner = R_IB @ x_body
    y_iner = R_IB @ y_body
    z_iner = R_IB @ z_body

    sc_attitude = np.vstack((x_iner, y_iner, z_iner)).T
    sc_position = state[i, 7:10]
    # if i % 100 == 0:
    #     plot_3d(state[:, 7:10], sc_attitude, sc_position, f"Uncorrected/{i}")

    # Calculate the dot product of the velocity vector in inertial frame with the body frame
    v = state[i, 10:13]
    v_dir = v / np.linalg.norm(v)
    xv[i] = np.rad2deg(np.arccos(np.dot(v_dir, x_iner)))
    yv[i] = np.rad2deg(np.arccos(np.dot(v_dir, y_iner)))
    zv[i] = np.rad2deg(np.arccos(np.dot(v_dir, z_iner)))

    # Calculate the dot product of the position vector in inertial frame with the body frame
    r = state[i, 7:10]
    r_dir = r / np.linalg.norm(r)
    # calcualte the position in body frame
    r_body = R_BI @ r



    xr[i] = np.rad2deg(np.arccos(np.dot(r_dir, x_iner)))
    yr[i] = np.rad2deg(np.arccos(np.dot(r_dir, y_iner)))
    zr[i] = np.rad2deg(np.arccos(np.dot(r_dir, z_iner)))

    # Calculate the correction
    z_corr = (-np.pi - np.deg2rad(zr[i]))

    R_x = np.array([[1, 0, 0], [0, np.cos(z_corr), np.sin(z_corr)], [0, -np.sin(z_corr), np.cos(z_corr)]])

    R_IB_new = R_IB @ R_x

    x_iner_new = R_IB_new @ x_body
    y_iner_new = R_IB_new @ y_body
    z_iner_new = R_IB_new @ z_body

    x_new[i] = np.rad2deg(np.arccos(np.dot(r_dir, x_iner_new)))
    y_new[i] = np.rad2deg(np.arccos(np.dot(r_dir, y_iner_new)))
    z_new[i] = np.rad2deg(np.arccos(np.dot(r_dir, z_iner_new)))

    sc_attitude_new = np.vstack((x_iner_new, y_iner_new, z_iner_new)).T

    # Caluclate the relative position vector
    r21 = state[i, 1:4] - state[i, 7:10]
    r21_dir = r21 / np.linalg.norm(r21)
    x_rel[i] = np.rad2deg(np.arccos(np.dot(r21_dir, x_iner)))
    y_rel[i] = np.rad2deg(np.arccos(np.dot(r21_dir, y_iner)))
    z_rel[i] = np.rad2deg(np.arccos(np.dot(r21_dir, z_iner)))

    r_scipy = RS.from_matrix(R_IB_new)
    quaternion[i] = r_scipy.as_quat(scalar_first=True)
    # if i % 100 == 0:
    #     plot_3d(state[:, 7:10], sc_attitude_new, sc_position, f"Corrected/{i}")


fig, ax = create_plots("Time [hrs]", "Angle [deg]", x_scale_log=False, y_scale_log=False, fontsize=16)
# ax.plot(time, xv, label="x")
ax.plot(time, yv, label="y")
ax.plot(time, zv, label="z")
ax.legend()
save_figure(fig, "Verification/Orbit/Angle_between_axes_velocity.svg")

fig1, ax1 = create_plots("Time [hrs]", "Angle [deg]", x_scale_log=False, y_scale_log=False, fontsize=16)
ax1.plot(time, xr, label="x")
ax1.plot(time, yr, label="y")
ax1.plot(time, zr, label="z")
ax1.legend()
save_figure(fig1, "Verification/Orbit/Angle_between_axes_position.svg")

fig3, ax3 = create_plots("Time [hrs]", "Angle [deg]", x_scale_log=False, y_scale_log=False, fontsize=16)
# ax3.plot(time, x_new, label="x")
# ax3.plot(time, y_new, label="y")
ax3.plot(time, z_new, label=r"Angle between $z_{B}$ & $r_{2}$")
ax3.legend(fontsize=15)
ax3.set_ylim(179, 179.1)

ax.set_yticks([179, 179.01, 179.02, 179.03, 179.04, 179.05, 179.06, 179.07, 179.08, 179.09, 179.1])
save_figure(fig3, "Verification/Orbit/Angle_between_axes_position_corrected.svg")


fig4, ax4 = create_plots("Time [hrs]", "Angle [deg]", x_scale_log=False, y_scale_log=False, fontsize=16)
ax4.plot(time, x_rel, label="Angle between x axis and LoS")
# ax4.plot(time, y_rel, label="y", marker='o')
# ax4.plot(time, z_rel, label="z", marker='o')
# ax4.set_ylim(-0.5, 0.5)
ax4.legend()
save_figure(fig4, "Verification/Orbit/Angle_between_axes_relative_position.svg")
