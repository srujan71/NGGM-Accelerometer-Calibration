import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import matplotlib as mpl
from Tools.SignalProcessingUtilities import Noise
import seaborn as sns


def return_sparse_output(time_history, variable_history, datapoints=200):
    interp_function = interpolate.interp1d(time_history, variable_history)
    time_interp = np.linspace(time_history[0], time_history[-1], datapoints)
    interpolated_values = [interp_function(epoch) for epoch in time_interp]

    return time_interp, interpolated_values


def create_three_plots(xlabel, ylabel, fig_titles, x_scale_log=False, y_scale_log=False, sharey=False):
    fig, ax = plt.subplots(1, 3, figsize=(13, 6), sharey=sharey)

    for j in range(3):
        ax[j].grid()
        ax[j].minorticks_on()
        ax[j].tick_params(labelsize=14, width=3)
        ax[j].tick_params(which='minor', width=2)
        ax[j].set_title(fig_titles[j], fontsize=18)
        ax[j].set_xlabel(xlabel, fontsize=16)
        if j == 0:
            ax[j].set_ylabel(ylabel, fontsize=16)
        if x_scale_log:
            ax[j].set_xscale('log')
        if y_scale_log:
            ax[j].set_yscale('log')
    # fig.tight_layout()
    fig.constrained_layout = True
    plt.close(fig)
    return fig, ax


def plot_heatmap(data, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    sns.heatmap(data, ax=ax, cmap='turbo', annot=True, fmt=".5f")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig, ax


def plot_setting_boxplots(df, setting, unit_labels, figsize=(6, 5), colormap='tab10', fontsize=14, **kwargs):
    """
    Plots box plots for a given setting, with separate boxes for each setting value.
    """
    # Filter the DataFrame for the specific setting
    setting_df = df[df['Setting'] == setting]
    setting_df = setting_df.copy()
    # Modify 'Setting_value' for the 'Layout' setting
    if setting == 'Layout':
        setting_df = setting_df.copy()  # Avoid modifying the original DataFrame
        setting_df['Setting_value'] = setting_df['Setting_value'].replace({1: 2, 2: 3, 3: 4})
        setting_df['Setting_value'] = setting_df['Setting_value'].astype(int)

    unit = unit_labels[setting]

    # Dynamically generate a color palette based on the number of unique 'Setting_value' values
    num_unique_values = setting_df['Setting_value'].nunique()
    palette = sns.color_palette(colormap, n_colors=num_unique_values)

    # Calculate mean values
    means = setting_df.groupby(['Thrust acceleration', 'Setting_value'])['Ratio'].mean().reset_index()
    means['x_dodge'] = means.groupby('Thrust acceleration').cumcount()   # Assign a unique index for each group
    num_settings = means['x_dodge'].nunique()   # Number of unique 'Setting_value' values per 'Thrust'
    dodge_width = 0.8 / num_settings   # Width of each box
    means['x_dodge'] = means['x_dodge'].map(lambda x: (x - (num_settings - 1) / 2) * dodge_width)   # Center the dodge positions

    # Map Thrust to a numerical value for plotting
    thrust_map = {thrust: i for i, thrust in enumerate(means['Thrust acceleration'].unique())}
    means['x_pos'] = means['Thrust acceleration'].map(thrust_map) + means['x_dodge']

    # Create the box plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x='Thrust acceleration', y='Ratio', hue='Setting_value', data=setting_df, ax=ax, palette=palette)
    sns.scatterplot(x='x_pos', y='Ratio', hue='Setting_value', data=means, ax=ax, palette=palette, marker='X', s=100, legend=False, zorder=10)
    ax.set_title(f'Box Plot of Ratio vs Thrust acceleration for {setting} Setting')
    ax.set_xlabel(r'Thrust acceleration [$m/s^2/\sqrt{Hz}$]', fontsize=fontsize)
    ax.set_ylabel('Ratio [-]', fontsize=fontsize)
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.3)
    ax.axhline(1, color='red', linestyle='--', label='Requirement')
    legend_title = f'{setting} ({unit})' if unit else f'{setting}'
    ax.legend(title=legend_title)  # Add a legend
    ax.tick_params(labelsize=fontsize - 2)
    fig.tight_layout()
    plt.close(fig)

    if "ratio_modified" in kwargs:
        means = setting_df.groupby(['Thrust acceleration', 'Setting_value'])['Ratio_modified'].mean().reset_index()
        means['x_dodge'] = means.groupby('Thrust acceleration').cumcount()  # Assign a unique index for each group
        num_settings = means['x_dodge'].nunique()  # Number of unique 'Setting_value' values per 'Thrust'
        dodge_width = 0.8 / num_settings  # Width of each box
        means['x_dodge'] = means['x_dodge'].map(lambda x: (x - (num_settings - 1) / 2) * dodge_width)  # Center the dodge positions

        # Map Thrust to a numerical value for plotting
        thrust_map = {thrust: i for i, thrust in enumerate(means['Thrust acceleration'].unique())}
        means['x_pos'] = means['Thrust acceleration'].map(thrust_map) + means['x_dodge']

        # Create the box plot
        fig_mod, ax_mod = plt.subplots(figsize=figsize)
        sns.boxplot(x='Thrust acceleration', y='Ratio_modified', hue='Setting_value', data=setting_df, ax=ax_mod, palette=palette)
        sns.scatterplot(x='x_pos', y='Ratio_modified', hue='Setting_value', data=means, ax=ax_mod, palette=palette, marker='X', s=100, legend=False, zorder=10)
        ax_mod.set_title(f'Box Plot of modified Ratio vs Thrust acceleration for {setting} Setting')
        ax_mod.set_xlabel(r'Thrust acceleration [$m/s^2/\sqrt{Hz}$]', fontsize=fontsize)
        ax_mod.set_ylabel('Ratio [-]', fontsize=fontsize)
        ax_mod.set_yscale('log')
        ax_mod.grid()
        ax_mod.axhline(1, color='red', linestyle='--', label='Requirement')
        legend_title = f'{setting} ({unit})' if unit else f'{setting}'
        ax_mod.legend(title=legend_title)  # Add a legend
        ax_mod.tick_params(labelsize=fontsize - 2)
        fig_mod.tight_layout()
        plt.close(fig_mod)

        return fig, ax, fig_mod, ax_mod
    else:
        return fig, ax


def create_plots(xlabel, ylabel, x_scale_log=False, y_scale_log=False, figsize=(6, 5), fontsize=12, ylabel_color='k'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.minorticks_on()
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.2)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, color=ylabel_color)
    if x_scale_log:
        ax.set_xscale('log')
    if y_scale_log:
        ax.set_yscale('log')
    ax.tick_params(labelsize=fontsize-2)
    plt.close(fig)
    return fig, ax


def save_figure(fig, path, format='svg'):
    current_dir = os.path.dirname(__file__)
    # Go one directory back to get to the main directory
    output_path = current_dir + "/../SimulationOutput/figures/"
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break

    folders.reverse()
    directory = output_path + "/".join(folders[:-1]) + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.tight_layout()
    fig.savefig(f"{directory + folders[-1]}", format=format)


def plot_asd(time_series, psd_points, axis_labels, path, series_labels=None, plot_all_axes=False, colormap=None, fontsize=16, figsize=(6,5), **kwargs):
    """
        Plots the Amplitude Spectral Density (ASD) for one or more time series.

        Parameters:
            time_series (np.ndarray): A 2D or 3D array of shape (n_samples, n_axes) or (n_time_series, n_samples, n_axes).
            psd_points (int): Number of points for the Welch PSD computation.
            axis_labels (list): Labels for the axes (e.g., ['X', 'Y', 'Z']).
            path (str): Path to save the figure.
            series_labels (list, optional): Labels for each time series (e.g., ['Series 1', 'Series 2']).
            plot_all_axes (bool): Whether to plot all axes (default: True).
            **kwargs: Additional arguments (e.g., 'compare' for comparison with a requirement).
        """

    if time_series.ndim != 1:
        if time_series.shape[1] == 3:
            time_series = time_series[np.newaxis, :]
        else:
            time_series = time_series[:, :, np.newaxis]
    else:
        time_series = time_series[np.newaxis, :, np.newaxis]

    # Setup colors
    if colormap is None:
        cmap = plt.get_cmap('viridis')
        colors = cmap(np.linspace(0, 1, len(time_series)))
        colors[:, 3] = 0.8
    elif isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, len(time_series)))
        colors[:, 3] = 0.8
    else:
        raise ValueError("Invalid color argument.")

    fig, ax = create_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", x_scale_log=True, y_scale_log=True, fontsize=fontsize, figsize=figsize)

    for i, ts in enumerate(time_series):

        series_label = f"{series_labels[i]}" if series_labels else None
        signal = Noise(1, fs=1)
        ffx, Pxx = signal.psd_welch(ts[:, 0], psd_points)
        ax.plot(ffx[1:], np.sqrt(Pxx[1:]), label=f"{axis_labels[0]}{series_label}" if series_labels else axis_labels[0],
                color=colors[i])

        if "confidence_interval" in kwargs:
            ts_lb = kwargs["confidence_intervals"][0][i][:, 0]
            ff_lb, Pxx_lb = signal.psd_welch(ts_lb, psd_points)

            ts_ub = kwargs["confidence_intervals"][1][i][:, 0]
            ff_ub, Pxx_ub = signal.psd_welch(ts_ub, psd_points)

            ax.fill_between(ff_lb[1:], np.sqrt(Pxx_lb[1:]), np.sqrt(Pxx_ub[1:]), color=colors[i], alpha=0.4)

        if plot_all_axes:
            ffy, Pyy = signal.psd_welch(ts[:, 1], psd_points)
            ffz, Pzz = signal.psd_welch(ts[:, 2], psd_points)
            ax.plot(ffy[1:], np.sqrt(Pyy[1:]), label=f"{axis_labels[1]}{series_label}" if series_labels else axis_labels[1])
            ax.plot(ffz[1:], np.sqrt(Pzz[1:]), label=f"{axis_labels[2]}{series_label}" if series_labels else axis_labels[2])

    # ax.legend(fontsize=14)

    if "save_fig" in kwargs:
        if kwargs['save_fig']:
            save_figure(fig, path)

    return fig, ax

class Patch:
    def __init__(self, x_coord, width, height, color, label_data):
        self.coords = [x_coord, 0]
        self.color = color
        self.width = width
        self.height = height
        self.text = label_data[0]
        self.text_coords = label_data[1]

    def draw(self, ax, font_size):
        ax.add_patch(mpl.patches.Rectangle(self.coords, width=self.width, height=self.height, alpha=0.6))
        ax.patches[-1].set_facecolor(self.color)
        ax.text(self.text_coords[0], self.text_coords[1], self.text, fontsize=font_size)


def draw_labels(ax, x_coords, label_y_coord, height, font_size, layout):
    label_y_coord = label_y_coord
    match layout:
        case 1:
            colors = ["#00ffff", "#aa55ff", "#ff0000", "#00ff00", "#ffff00"]
            width = [9, 9, 6, 3, 2]
            labels = [["Mc12", [3, label_y_coord]], ["Md12", [12, label_y_coord]], ["K", [20.1, label_y_coord]],
                      ["W", [24.5, label_y_coord]], ["dr", [27, label_y_coord]]]
        case 2:
            colors = ["#00ffff", "#aa55ff", "#ff0000", "#00ff00", "#ffff00", "#ff5500"]
            width = [9, 9, 9, 9, 6, 5]
            labels = [["M2", [3, label_y_coord]], ["Mc13", [10.5, label_y_coord]], ["Md13", [19.5, label_y_coord]], ["K", [30, label_y_coord]],
                     ["W", [38, label_y_coord]], ["dr", [43, label_y_coord]]]

        case 3:
            colors = ["#00ffff", "#aa55ff", "#ff0000", "#00ff00", "#ffff00", "#ff5500", "#ff00ff"]
            width = [9, 9, 9, 9, 12, 9, 7]
            labels = [["Mc13", [2, label_y_coord]], ["Mc24", [11.5, label_y_coord]], ["Md13", [20.3, label_y_coord]],
                     ["Md24", [29, label_y_coord]], ["K", [41, label_y_coord]], ["W", [51, label_y_coord]], ["dr", [59, label_y_coord]]]

    for i in range(len(colors)):
        patch = Patch(x_coords[i], width[i], height, colors[i], labels[i])
        patch.draw(ax, font_size)


def plot_kepler_elements(dependent_variables_array, filename, plot_both=True):
    time = dependent_variables_array[:, 0]
    kepler_elements_lst = [dependent_variables_array[:, 30:36], dependent_variables_array[:, 36:42]]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(9, 10))
    fig.suptitle('Kepler elements')

    # Loop over Kepler elements in the following order
    y_labels = ['Semi-major axis [km]',
                'Eccentricity [-]',
                'Inclination [deg]',
                'Argument of Periapsis [deg]',
                'RAAN [deg]',
                'True Anomaly [deg]']
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    if plot_both:
        list_size = 2
    else:
        list_size = 1
    for i in range(list_size):

        # Convert SMA to km
        kepler_elements_lst[i][:, 0] = kepler_elements_lst[i][:, 0] / 1e3
        # Convert radians to degrees
        kepler_elements_lst[i][:, 2:6] = np.rad2deg(kepler_elements_lst[i][:, 2:6])

        # Interpolate to get less dense output
        for element in range(6):
            # time_interp, values_interp = return_sparse_output(time, kepler_elements_lst[i][:, element], 10000)
            # # convert time to hours
            # time_interp_hours = [epoch / 3600 for epoch in time_interp]
            #
            # # Plot
            # current_ax = axes[element]
            # current_ax.plot(time_interp_hours, values_interp)

            time_interp, values_interp = return_sparse_output(time, kepler_elements_lst[i][:, element], 10000)
            # convert time to hours
            time_hrs = [epoch / 3600 for epoch in time]

            # Plot
            current_ax = axes[element]
            current_ax.plot(time_hrs, kepler_elements_lst[i][:, element])

            if i == 1:
                if element >= 4:
                    current_ax.set_xlabel("Time [hours]")
                current_ax.set_ylabel(y_labels[element])
                current_ax.grid()

    fig.tight_layout()
    save_figure(fig, filename)
    plt.close(fig)


def plot_3d(state_history, sc_attitude, sc_position, filename):
    # Convert the elements components to km
    state_history = state_history / 1e3
    sc_position = sc_position / 1e3

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(projection="3d")

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    R = 3000
    xw = R * np.outer(np.cos(u), np.sin(v))
    yw = R * np.outer(np.sin(u), np.sin(v))
    zw = R * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the inertial frame at origin
    x, y, z = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    u, v, w = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.quiver(x, y, z, u, v, w, length=7000, color=['r', 'g', 'b'])

    # Plot Earth
    ax.plot_wireframe(xw, yw, zw)

    # Plot the orbit

    ax.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2], color='orange')

    # Plot the s/c axes system
    axes_direction_x = np.array([sc_attitude[0, 0], sc_attitude[1, 0], sc_attitude[2, 0]])
    axes_direction_y = np.array([sc_attitude[0, 1], sc_attitude[1, 1], sc_attitude[2, 1]])
    axes_direction_z = np.array([sc_attitude[0, 2], sc_attitude[1, 2], sc_attitude[2, 2]])
    ax.quiver(sc_position[0], sc_position[1], sc_position[2],
              axes_direction_x[0], axes_direction_x[1], axes_direction_x[2], length=2000, color='r')
    ax.quiver(sc_position[0], sc_position[1], sc_position[2],
              axes_direction_y[0], axes_direction_y[1], axes_direction_y[2], length=2000, color='g')
    ax.quiver(sc_position[0], sc_position[1], sc_position[2],
              axes_direction_z[0], axes_direction_z[1], axes_direction_z[2], length=2000, color='b')

    # Set axes limits
    ax.set_xlim([-7000, 7000])
    ax.set_ylim([-7000, 7000])
    ax.set_zlim([-7000, 7000])

    # Set an equal aspect ratio
    ax.set_aspect('equal')

    ax.view_init(elev=15, azim=60)

    fig.tight_layout()
    plt.close(fig)
    save_figure(fig, f"Verification/Orbit/Plot3D/{filename}.png")
