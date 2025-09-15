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


def skew_symmetric_matrix(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def derivative(t, C_flat, omega_interp):
    C = C_flat.reshape((3, 3))
    w = omega_interp(t)
    Omega = skew_symmetric_matrix(w)
    dcdt = -Omega @ C
    return dcdt.flatten()


def update(frame, C_matrices, time, quiver, unit_axes):
    C = C_matrices[frame*500]
    transformed_axes = C @ unit_axes

    # Update the quiver
    quiver.set_segments([[[0, 0, 0], transformed_axes[:, i]] for i in range(3)])
    ax.set_title(f"Time: {time[frame*500]} s")  # Update title with time

    return quiver,


current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Attitude_check/"

time, omega = load_angular_velocity(path, "w_1e-1.txt")
# omega -= np.mean(omega, axis=0)

omega_interp = interp1d(time, omega, axis=0, kind='linear', fill_value='extrapolate')

# Initial condition
C0 = np.identity(3).flatten()
t_span = (time[0], time[-1])
t_eval = time

C_flat = solve_ivp(derivative, t_span, C0, args=(omega_interp,), t_eval=t_eval, method='LSODA')

C_mat = C_flat.y.T.reshape(-1, 3, 3)

# Plot and animate the attitude on a 3d graph with x, y, z axes
unit_axes = np.eye(3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Attitude animation")

quiver = ax.quiver(0, 0, 0, *unit_axes.T, color=['r', 'g', 'b'], linewidth=2)

update_wrapper = partial(update, C_matrices=C_mat, time=time, quiver=quiver, unit_axes=unit_axes)

fig2, ax2 = plt.subplots()
ax2.plot(time, omega)
ax2.grid()

# Convert to milliseconds

ani = animation.FuncAnimation(fig, update_wrapper, frames=len(C_mat)//500, interval=1, blit=False)

ani.save('Shaking.gif', writer='pillow', fps=30)

# plt.show()

# fig, ax = plot_asd(w, 27001, ['wx', 'wy', 'wz'], "", plot_all_axes=True)
#
# plt.show()
