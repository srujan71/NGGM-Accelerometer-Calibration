import numpy as np
from Tools.PlotTools import plot_asd, plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from functools import partial


def load_angular_velocity(file_path, file_name):
    omega = np.loadtxt(file_path + file_name)
    time = np.arange(0, len(omega))
    return time, omega


def derivative(t, theta, omega_interp):
    th1 = theta[0]
    th2 = theta[1]
    th3 = theta[2]
    mat = (1/np.cos(th2)) * np.array([[np.cos(th2), np.sin(th1)*np.sin(th2), np.cos(th1)*np.sin(th2)],
                                      [0, np.cos(th1)*np.cos(th2), -np.sin(th1)*np.cos(th2)],
                                      [0, np.sin(th1), np.cos(th1)]])

    w = omega_interp(t)

    dthdt = mat @ w

    return dthdt



current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Attitude_check/"

time, omega = load_angular_velocity(path, "w_1e-2.txt")
omega -= np.mean(omega, axis=0)

omega_interp = interp1d(time, omega, axis=0, kind='linear', fill_value='extrapolate')

# Initial condition
theta_0 = np.array([0, 0, 0])
t_span = (time[0], time[-1])
t_eval = time

solution = solve_ivp(derivative, t_span, theta_0, args=(omega_interp,), t_eval=t_eval, method='LSODA')

theta = solution.y.T
theta = np.rad2deg(theta)

# time = time[0:10000]
# theta = theta[0:10000, :]

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(time, theta[:, 0])
ax[0].set_title("Roll angle")
ax[0].set_xlabel("Time [s]")
ax[0].set_ylabel("Angle [deg]")
ax[1].plot(time, theta[:, 1])
ax[1].set_title("Pitch angle")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("Angle [deg]")
ax[2].plot(time, theta[:, 2])
ax[2].set_title("Yaw angle")
ax[2].set_xlabel("Time [s]")
ax[2].set_ylabel("Angle [deg]")

fig.tight_layout()
plt.show()

