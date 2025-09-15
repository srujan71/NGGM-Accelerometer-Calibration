import numpy as np
from Tools.PlotTools import create_plots, save_figure, plt
from scipy.io import loadmat

model_analysis = True
gravity_gradient_analysis = False
if model_analysis:
    number_of_models = 5
    model_labels = ["Nominal", "Earth_D/O_120", "Moon", "Sun", "Jupiter"]

    path = "SimulationOutput/Output/Acceleration_models/"
    nominal_state = np.loadtxt(path + f"model_0/state_history.dat")
    t = (nominal_state[:, 0] - nominal_state[0, 0]) / 3600
    nominal_dependent_variables = np.loadtxt(path + f"model_0/dependent_variable_history.dat")
    nominal_latitude = np.rad2deg(nominal_dependent_variables[:, 42])
    nominal_longitude = np.rad2deg(nominal_dependent_variables[:, 43])
    nominal_rot_IV = nominal_dependent_variables[:, 19:28]
    nominal_altitude = nominal_dependent_variables[:, -4]


    fig, ax = create_plots("Time [hr]", "Position error [m]", False, True, fontsize=16)
    fig1, ax1 = create_plots("Time [hr]", "Latitude error [deg]", False, False, fontsize=16)
    fig2, ax2 = create_plots("Time [hr]", "Longitude error [deg]", False, False, fontsize=16)
    fig3, ax3 = create_plots("Time [hr]", "Latitude [deg]", False, False, fontsize=16)
    fig4, ax4 = create_plots("Time [hr]", "Longitude [deg]", False, False, fontsize=16)
    fig5, ax5 = create_plots("Time [hr]", "Position error [km]", False, False, fontsize=16)
    ax3.plot(t, nominal_latitude, label="Nominal")
    ax4.plot(t, nominal_longitude, label="Nominal")
    for model_index in range(1, number_of_models):
        state = np.loadtxt(path + f"model_{model_index}/state_history.dat")
        dependent_variables = np.loadtxt(path + f"model_{model_index}/dependent_variable_history.dat")
        # Calculate position error
        position_error = np.linalg.norm(state[:, 1:4] - nominal_state[:, 1:4], axis=1)
        # Calculate latitude and longitude error
        latitude_error = np.rad2deg(dependent_variables[:, 51] - nominal_dependent_variables[:, 51])
        longitude_error = np.rad2deg(dependent_variables[:, 52] - nominal_dependent_variables[:, 52])

        longitude_error[longitude_error > 300] -= 360

        ax.plot(t, position_error, label=model_labels[model_index])
        if model_index == 1:
            ax1.plot(t, latitude_error)
            ax2.plot(t, longitude_error)
            ax3.plot(t, np.rad2deg(dependent_variables[:, 51]), label=model_labels[model_index])
            ax4.plot(t, np.rad2deg(dependent_variables[:, 52]), label=model_labels[model_index])

            # kepler elements
            kep = dependent_variables[: 45:51]
            semi_major_axis = dependent_variables[:, 45]
            true_anomaly = dependent_variables[:, 50]


            rot_IV = dependent_variables[:, 19:28]
            nom_pos_V = np.zeros((len(t), 3))
            pos_V = np.zeros((len(t), 3))

            altitude = dependent_variables[:, -4]
            # ax5.plot(t, (altitude - nominal_altitude)/1000, label=model_labels[model_index])
            print()

            pos_diff = (state[:, 7:10] - nominal_state[:, 7:10]) / 1000
            pos_diff_VF = np.zeros((len(t), 3))

            for i in range(len(nominal_rot_IV)):
                R_IV = np.reshape(rot_IV[i, :], (3, 3))
                R_IV_nom = np.reshape(nominal_rot_IV[i, :], (3, 3))

                pos_diff_VF[i, :] = R_IV.T @ pos_diff[i, :]




            ax5.plot(t, pos_diff_VF[:, 0], label='along-track')
            ax5.plot(t, pos_diff_VF[:, 1], label='cross-track')
            ax5.plot(t, pos_diff_VF[:, 2], label='radial')


    ax.legend(fontsize=15)
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    ax3.legend(fontsize=15)
    ax4.legend(fontsize=15)
    ax5.legend(fontsize=15)


    # plt.show()

    save_figure(fig, "Model_setup/acceleration_model_position_error.svg")
    save_figure(fig1, "Model_setup/acceleration_model_latitude_error.svg")
    save_figure(fig2, "Model_setup/acceleration_model_longitude_error.svg")
    save_figure(fig3, "Model_setup/acceleration_model_latitude.svg")
    save_figure(fig4, "Model_setup/acceleration_model_longitude.svg")
    save_figure(fig5, "Model_setup/along_track_error.svg")

if gravity_gradient_analysis:
    # Load the gravity gradient files
    models = 2
    path = "SimulationOutput/Output/Acceleration_models/"

    gravity_gradients_pm = loadmat(path + f"model_{0}/gravity_gradients_120.mat")['gravity_gradients']
    gravity_gradients_120 = loadmat(path + f"model_{1}/gravity_gradients_120.mat")['gravity_gradients']

    Vxx_pm = gravity_gradients_pm[0]['Vxx'][0]
    Vyy_pm = gravity_gradients_pm[0]['Vyy'][0]
    Vzz_pm = gravity_gradients_pm[0]['Vzz'][0]
    Vxy_pm = gravity_gradients_pm[0]['Vxy'][0]
    Vxz_pm = gravity_gradients_pm[0]['Vxz'][0]
    Vyz_pm = gravity_gradients_pm[0]['Vyz'][0]
    V_pm = np.hstack((Vxx_pm, Vyy_pm, Vzz_pm, Vxy_pm, Vxz_pm, Vyz_pm))

    Vxx_120 = gravity_gradients_120[0]['Vxx'][0]
    Vyy_120 = gravity_gradients_120[0]['Vyy'][0]
    Vzz_120 = gravity_gradients_120[0]['Vzz'][0]
    Vxy_120 = gravity_gradients_120[0]['Vxy'][0]
    Vxz_120 = gravity_gradients_120[0]['Vxz'][0]
    Vyz_120 = gravity_gradients_120[0]['Vyz'][0]
    V_120 = np.hstack((Vxx_120, Vyy_120, Vzz_120, Vxy_120, Vxz_120, Vyz_120))

    # Calculate the rms difference
    V_diff = V_120 - V_pm
    rms_diff = np.sqrt(np.mean(V_diff**2, axis=0))
    rms_120 = np.sqrt(np.mean(V_120**2, axis=0))

    print(rms_diff)
    print()
    print(np.max(abs(V_120), axis=0))
    print()
    # print percentage difference
    print(rms_diff / rms_120 * 100)







