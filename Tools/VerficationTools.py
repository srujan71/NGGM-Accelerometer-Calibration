import numpy as np
import os
from Tools.CalibrationTools import calibrate, acc_cal_linearized_function, acc_cal_par_vec_to_mat
from Tools.PlotTools import create_plots, save_figure, create_three_plots, draw_labels
from Tools.SignalProcessingUtilities import get_expected_noise
from matplotlib.ticker import LogLocator, MultipleLocator
import matplotlib.pyplot as plt
import seaborn as sns
import time

def verify_design_matrix(x0, Acc_lst, a_ng_rcst, layout, load_data=False):
    path = f"../SimulationOutput/Output/Verification/DesignMatrix/Configuration_{layout}/"
    if not os.path.exists(path):
        os.makedirs(path)
    factors = [0.05, 0.005]
    match layout:
        case 1:
            if not load_data:
                # 1) Evaluate the design matrix at x0
                ydx, ydy, ydz, yd0x, yd0y, yd0z, Ad0x, Ad0y, Ad0z = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                epochs = len(yd0x)
                y0_org = np.hstack((yd0x, yd0y, yd0z))
                A_org = np.vstack((Ad0x, Ad0y, Ad0z))

                e_rms_x = np.zeros((len(x0), 2))
                e_rms_y = np.zeros((len(x0), 2))
                e_rms_z = np.zeros((len(x0), 2))

                A_rms_x = np.zeros((len(x0), 2))
                A_rms_y = np.zeros((len(x0), 2))
                A_rms_z = np.zeros((len(x0), 2))
                for i in range(len(x0)):
                    for j in range(len(factors)):
                        print(f"Parameter {i} factor{j}")
                        x_new = x0.copy()
                        x_new[i] = x0[i] + factors[j]

                        ydx, ydy, ydz, yd0x, yd0y, yd0z, Adx, Ady, Adz = acc_cal_linearized_function(x_new, Acc_lst, a_ng_rcst, layout)

                        y0_new = np.hstack((yd0x, yd0y, yd0z))

                        error = ((y0_new - y0_org) / (x_new[i] - x0[i])) - A_org[:, i]
                        error = np.reshape(error, (epochs, 3), order='F')

                        edx_rms = np.sqrt(np.mean(error[:, 0] ** 2))
                        edy_rms = np.sqrt(np.mean(error[:, 1] ** 2))
                        edz_rms = np.sqrt(np.mean(error[:, 2] ** 2))

                        Adx_rms = np.sqrt(np.mean(Ad0x[:, i] ** 2))
                        Ady_rms = np.sqrt(np.mean(Ad0y[:, i] ** 2))
                        Adz_rms = np.sqrt(np.mean(Ad0z[:, i] ** 2))

                        e_rms_x[i, j] = edx_rms
                        e_rms_y[i, j] = edy_rms
                        e_rms_z[i, j] = edz_rms

                        A_rms_x[i, j] = Adx_rms
                        A_rms_y[i, j] = Ady_rms
                        A_rms_z[i, j] = Adz_rms

                    np.savetxt(path + f"ex_rms.txt", e_rms_x)
                    np.savetxt(path + f"ey_rms.txt", e_rms_y)
                    np.savetxt(path + f"ez_rms.txt", e_rms_z)
                    np.savetxt(path + f"Ax_rms.txt", A_rms_x)
                    np.savetxt(path + f"Ay_rms.txt", A_rms_y)
                    np.savetxt(path + f"Az_rms.txt", A_rms_z)

            else:
                e_rms_x = np.loadtxt(path + f"ex_rms.txt")
                e_rms_y = np.loadtxt(path + f"ey_rms.txt")
                e_rms_z = np.loadtxt(path + f"ez_rms.txt")
                A_rms_x = np.loadtxt(path + f"Ax_rms.txt")
                A_rms_y = np.loadtxt(path + f"Ay_rms.txt")
                A_rms_z = np.loadtxt(path + f"Az_rms.txt")

            fig, ax = create_three_plots("Parameter[-]", r"RMS [$m/s^{2}$]", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'], False, True, sharey=True)
            ax[0].plot(e_rms_x[:, 0], marker='o', label="Error_rms")
            ax[0].plot(A_rms_x[:, 0], marker='o', label="Design_matrix_rms")
            ax[1].plot(e_rms_y[:, 0], marker='o', label="Error_rms")
            ax[1].plot(A_rms_y[:, 0], marker='o', label="Design_matrix_rms")
            ax[2].plot(e_rms_z[:, 0], marker='o', label="Error_rms")
            ax[2].plot(A_rms_z[:, 0], marker='o', label="Design_matrix_rms")

            fig1, ax1 = create_three_plots("Parameter[-]", r"Factor", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'], False, True)
            ax1[0].plot(e_rms_x[:, 0] / A_rms_x[:, 0], marker='o')
            ax1[1].plot(e_rms_y[:, 0] / A_rms_y[:, 0], marker='o')
            ax1[2].plot(e_rms_z[:, 0] / A_rms_z[:, 0], marker='o')

            fig2, ax2 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'],
                                           False, False, sharey=True)
            ebh_x = e_rms_x[:, 0] / e_rms_x[:, 1]
            ebh_y = e_rms_y[:, 0] / e_rms_y[:, 1]
            ebh_z = e_rms_z[:, 0] / e_rms_z[:, 1]
            ax2[0].plot(ebh_x, marker='o')
            ax2[1].plot(ebh_y, marker='o')
            ax2[2].plot(ebh_z, marker='o')

            height = -1.4
            height = [e_rms_x[:, 0] / e_rms_x[:, 1], ]
            const_x_lines = [-0.5, 8.5, 17.5, 23.5, 26.5, 28.5]
            label_y_coord = -0.35
            fontsize = 13
            for i, axes in enumerate([ax, ax1, ax2]):
                for j in range(3):
                    axes[j].legend(handlelength=3.5, fontsize=13)
                    axes[j].legend(handlelength=3.5, fontsize=13)
                    if not i == 2:
                        axes[j].yaxis.set_minor_locator(LogLocator(subs='all', numticks=20))
                    axes[j].grid(which='minor', axis='y', linestyle=':', linewidth=1)
                    axes[j].xaxis.set_major_locator(MultipleLocator(5))
                    for k in range(len(const_x_lines)):
                        axes[j].axvline(x=const_x_lines[k], color='k', linestyle='--', alpha=0.6)
                    draw_labels(axes[j], const_x_lines, label_y_coord, height, fontsize, layout)

            save_figure(fig, f"Verification/Design_Matrix/Configuration_{layout}/Design_Matrix_rms_diff.svg")
            save_figure(fig1, f"Verification/Design_Matrix/Configuration_{layout}/Factor_diff.svg")
            save_figure(fig2, f"Verification/Design_Matrix/Configuration_{layout}/Error_behaviour_diff.svg")

        case 2:
            if not load_data:
                # 1) Evaluate the design matrix at x0
                ydx, ydy, ydz, ycx, ycy, ycz, yd0x, yd0y, yd0z, yc0x, yc0y, yc0z, \
                    Ad0x, Ad0y, Ad0z, Ac0x, Ac0y, Ac0z = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                epochs = len(yd0x)
                y0_org = np.hstack((yd0x, yd0y, yd0z, yc0x, yc0y, yc0z))
                A_org = np.vstack((Ad0x, Ad0y, Ad0z, Ac0x, Ac0y, Ac0z))

                e_rms_x = np.zeros((len(x0), 4))
                e_rms_y = np.zeros((len(x0), 4))
                e_rms_z = np.zeros((len(x0), 4))

                A_rms_x = np.zeros((len(x0), 4))
                A_rms_y = np.zeros((len(x0), 4))
                A_rms_z = np.zeros((len(x0), 4))
                for i in range(len(x0)):
                    for j in range(len(factors)):
                        print(f"Parameter {i} factor{j}")
                        x_new = x0.copy()
                        x_new[i] = x0[i] + factors[j]

                        ydx, ydy, ydz, ycx, ycy, ycz, yd0x, yd0y, yd0z, yc0x, yc0y, yc0z, \
                            Adx, Ady, Adz, Acx, Acy, Acz = acc_cal_linearized_function(x_new, Acc_lst, a_ng_rcst, layout)

                        y0_new = np.hstack((yd0x, yd0y, yd0z, yc0x, yc0y, yc0z))

                        error = ((y0_new - y0_org) / (x_new[i] - x0[i])) - A_org[:, i]
                        error = np.reshape(error, (epochs, 6), order='F')

                        edx_rms = np.sqrt(np.mean(error[:, 0] ** 2))
                        edy_rms = np.sqrt(np.mean(error[:, 1] ** 2))
                        edz_rms = np.sqrt(np.mean(error[:, 2] ** 2))
                        ecx_rms = np.sqrt(np.mean(error[:, 3] ** 2))
                        ecy_rms = np.sqrt(np.mean(error[:, 4] ** 2))
                        ecz_rms = np.sqrt(np.mean(error[:, 5] ** 2))

                        Adx_rms = np.sqrt(np.mean(Ad0x[:, i] ** 2))
                        Ady_rms = np.sqrt(np.mean(Ad0y[:, i] ** 2))
                        Adz_rms = np.sqrt(np.mean(Ad0z[:, i] ** 2))
                        Acx_rms = np.sqrt(np.mean(Ac0x[:, i] ** 2))
                        Acy_rms = np.sqrt(np.mean(Ac0y[:, i] ** 2))
                        Acz_rms = np.sqrt(np.mean(Ac0z[:, i] ** 2))

                        e_rms_x[i, j*2] = edx_rms
                        e_rms_y[i, j*2] = edy_rms
                        e_rms_z[i, j*2] = edz_rms
                        e_rms_x[i, j*2+1] = ecx_rms
                        e_rms_y[i, j*2+1] = ecy_rms
                        e_rms_z[i, j*2+1] = ecz_rms

                        A_rms_x[i, j*2] = Adx_rms
                        A_rms_y[i, j*2] = Ady_rms
                        A_rms_z[i, j*2] = Adz_rms
                        A_rms_x[i, j*2+1] = Acx_rms
                        A_rms_y[i, j*2+1] = Acy_rms
                        A_rms_z[i, j*2+1] = Acz_rms

                    np.savetxt(path + f"ex_rms.txt", e_rms_x)
                    np.savetxt(path + f"ey_rms.txt", e_rms_y)
                    np.savetxt(path + f"ez_rms.txt", e_rms_z)
                    np.savetxt(path + f"Ax_rms.txt", A_rms_x)
                    np.savetxt(path + f"Ay_rms.txt", A_rms_y)
                    np.savetxt(path + f"Az_rms.txt", A_rms_z)

            else:
                e_rms_x = np.loadtxt(path + f"ex_rms.txt")
                e_rms_y = np.loadtxt(path + f"ey_rms.txt")
                e_rms_z = np.loadtxt(path + f"ez_rms.txt")
                A_rms_x = np.loadtxt(path + f"Ax_rms.txt")
                A_rms_y = np.loadtxt(path + f"Ay_rms.txt")
                A_rms_z = np.loadtxt(path + f"Az_rms.txt")

            fig, ax = create_three_plots("Parameter[-]", r"RMS [$m/s^{2}$]", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'], False, True, sharey=True)
            ax[0].plot(e_rms_x[:, 0], marker='o', label="Error_rms")
            ax[0].plot(A_rms_x[:, 0], marker='o', label="Design_matrix_rms")
            ax[1].plot(e_rms_y[:, 0], marker='o', label="Error_rms")
            ax[1].plot(A_rms_y[:, 0], marker='o', label="Design_matrix_rms")
            ax[2].plot(e_rms_z[:, 0], marker='o', label="Error_rms")
            ax[2].plot(A_rms_z[:, 0], marker='o', label="Design_matrix_rms")


            fig1, ax1 = create_three_plots("Parameter [-]", r"RMS [$m/s^{2}$]", [r'$a_{c13x} - a_{2x}$', r'$a_{c13y} - a_{2y}$', r'$a_{c13z} - a_{2z}$'], False, True, sharey=True)
            ax1[0].plot(e_rms_x[:, 1], marker='o', label="Error_rms")
            ax1[0].plot(A_rms_x[:, 1], marker='o', label="Design_matrix_rms")
            ax1[1].plot(e_rms_y[:, 1], marker='o', label="Error_rms")
            ax1[1].plot(A_rms_y[:, 1], marker='o', label="Design_matrix_rms")
            ax1[2].plot(e_rms_z[:, 1], marker='o', label="Error_rms")
            ax1[2].plot(A_rms_z[:, 1], marker='o', label="Design_matrix_rms")

            fig2, ax2 = create_three_plots("Parameter[-]", r"Factor", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'], False, True)
            ax2[0].plot(e_rms_x[:, 0]/A_rms_x[:, 0], marker='o')
            ax2[1].plot(e_rms_y[:, 0]/A_rms_y[:, 0], marker='o')
            ax2[2].plot(e_rms_z[:, 0]/A_rms_z[:, 0], marker='o')

            fig3, ax3 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{c13x} - a_{2x}$', r'$a_{c13y} - a_{2y}$', r'$a_{c13z} - a_{2z}$'], False, True, sharey=True)
            ax3[0].plot(e_rms_x[:, 1] / A_rms_x[:, 1], marker='o')
            ax3[1].plot(e_rms_y[:, 1] / A_rms_y[:, 1], marker='o')
            ax3[2].plot(e_rms_z[:, 1] / A_rms_z[:, 1], marker='o')

            fig4, ax4 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'],
                                           False, True, sharey=True)
            ebh_dx = e_rms_x[:, 0] / e_rms_x[:, 2]
            ebh_dy = e_rms_y[:, 0] / e_rms_y[:, 2]
            ebh_dz = e_rms_z[:, 0] / e_rms_z[:, 2]
            ax4[0].plot(ebh_dx, marker='o')
            ax4[1].plot(ebh_dy, marker='o')
            ax4[2].plot(ebh_dz, marker='o')

            fig5, ax5 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{c13x} - a_{2x}$', r'$a_{c13y} - a_{2y}$', r'$a_{c13z} - a_{2z}$'],
                                           False, True, sharey=True)
            ebh_cx = e_rms_x[:, 1] / e_rms_x[:, 3]
            ebh_cy = e_rms_y[:, 1] / e_rms_y[:, 3]
            ebh_cz = e_rms_z[:, 1] / e_rms_z[:, 3]
            ax5[0].plot(e_rms_x[:, 1] / e_rms_x[:, 3], marker='o')
            ax5[1].plot(e_rms_y[:, 1] / e_rms_y[:, 3], marker='o')
            ax5[2].plot(e_rms_z[:, 1] / e_rms_z[:, 3], marker='o')

            height = -1.4
            const_x_lines = [-0.5, 8.5, 17.5, 26.5, 35.5, 41.5, 46.6]
            label_y_coord = -0.35
            fontsize = 11
            for i, axes in enumerate([ax, ax1, ax2, ax3, ax4, ax5]):
                for j in range(3):
                    axes[j].legend(handlelength=3.5, fontsize=13)
                    axes[j].legend(handlelength=3.5, fontsize=13)
                    if not (i == 4 or i == 5):
                        axes[j].yaxis.set_minor_locator(LogLocator(subs='all', numticks=20))
                    axes[j].grid(which='minor', axis='y', linestyle=':', linewidth=1)
                    axes[j].xaxis.set_major_locator(MultipleLocator(5))
                    for k in range(len(const_x_lines)):
                        axes[j].axvline(x=const_x_lines[k], color='k', linestyle='--', alpha=0.6)
                    draw_labels(axes[j], const_x_lines, label_y_coord, height, fontsize, layout)

            save_figure(fig, f"Verification/Design_Matrix/Configuration_{layout}/Design_Matrix_rms_diff.svg")
            save_figure(fig1, f"Verification/Design_Matrix/Configuration_{layout}/Design_Matrix_rms_com.svg")
            save_figure(fig2, f"Verification/Design_Matrix/Configuration_{layout}/Factor_diff.svg")
            save_figure(fig3, f"Verification/Design_Matrix/Configuration_{layout}/Factor_com.svg")
            save_figure(fig4, f"Verification/Design_Matrix/Configuration_{layout}/Error_behaviour_diff.svg")
            save_figure(fig5, f"Verification/Design_Matrix/Configuration_{layout}/Error_behaviour_com.svg")

        case 3:
            if not load_data:
                # 1) Evaluate the design matrix at x0
                yd13x, yd13y, yd13z, yd24x, yd24y, yd24z, ycx, ycy, ycz, yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z, \
                    Ad0x13, Ad0y13, Ad013z, Ad024x, Ad024y, Ad024z, Ac0x, Ac0y, Ac0z = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                epochs = len(yd13x)
                y0_org = np.hstack((yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z))
                A_org = np.vstack((Ad0x13, Ad0y13, Ad013z, Ad024x, Ad024y, Ad024z, Ac0x, Ac0y, Ac0z))

                e_rms_x = np.zeros((len(x0), 6))
                e_rms_y = np.zeros((len(x0), 6))
                e_rms_z = np.zeros((len(x0), 6))

                A_rms_x = np.zeros((len(x0), 6))
                A_rms_y = np.zeros((len(x0), 6))
                A_rms_z = np.zeros((len(x0), 6))

                for i in range(len(x0)):
                    for j in range(len(factors)):
                        print(f"Parameter {i} factor{j}")
                        x_new = x0.copy()
                        x_new[i] = x0[i] + factors[j]

                        yd13x, yd13y, yd13z, yd24x, yd24y, yd24z, ycx, ycy, ycz, yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z, \
                            Adx13, Ady13, Adz13, Adx24, Ady24, Adz24, Acx, Acy, Acz = acc_cal_linearized_function(x_new, Acc_lst, a_ng_rcst, layout)

                        y0_new = np.hstack((yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z))

                        error = ((y0_new - y0_org) / (x_new[i] - x0[i])) - A_org[:, i]
                        error = np.reshape(error, (epochs, 9), order='F')

                        ed13x_rms = np.sqrt(np.mean(error[:, 0] ** 2))
                        ed13y_rms = np.sqrt(np.mean(error[:, 1] ** 2))
                        ed13z_rms = np.sqrt(np.mean(error[:, 2] ** 2))
                        ed24x_rms = np.sqrt(np.mean(error[:, 3] ** 2))
                        ed24y_rms = np.sqrt(np.mean(error[:, 4] ** 2))
                        ed24z_rms = np.sqrt(np.mean(error[:, 5] ** 2))
                        ecx_rms = np.sqrt(np.mean(error[:, 6] ** 2))
                        ecy_rms = np.sqrt(np.mean(error[:, 7] ** 2))
                        ecz_rms = np.sqrt(np.mean(error[:, 8] ** 2))

                        Ad13x_rms = np.sqrt(np.mean(Ad0x13[:, i] ** 2))
                        Ad13y_rms = np.sqrt(np.mean(Ad0y13[:, i] ** 2))
                        Ad13z_rms = np.sqrt(np.mean(Ad013z[:, i] ** 2))
                        Ad24x_rms = np.sqrt(np.mean(Ad024x[:, i] ** 2))
                        Ad24y_rms = np.sqrt(np.mean(Ad024y[:, i] ** 2))
                        Ad24z_rms = np.sqrt(np.mean(Ad024z[:, i] ** 2))
                        Acx_rms = np.sqrt(np.mean(Ac0x[:, i] ** 2))
                        Acy_rms = np.sqrt(np.mean(Ac0y[:, i] ** 2))
                        Acz_rms = np.sqrt(np.mean(Ac0z[:, i] ** 2))

                        e_rms_x[i, j*3] = ed13x_rms
                        e_rms_y[i, j*3] = ed13y_rms
                        e_rms_z[i, j*3] = ed13z_rms
                        e_rms_x[i, j*3+1] = ed24x_rms
                        e_rms_y[i, j*3+1] = ed24y_rms
                        e_rms_z[i, j*3+1] = ed24z_rms
                        e_rms_x[i, j*3+2] = ecx_rms
                        e_rms_y[i, j*3+2] = ecy_rms
                        e_rms_z[i, j*3+2] = ecz_rms

                        A_rms_x[i, j*3] = Ad13x_rms
                        A_rms_y[i, j*3] = Ad13y_rms
                        A_rms_z[i, j*3] = Ad13z_rms
                        A_rms_x[i, j*3+1] = Ad24x_rms
                        A_rms_y[i, j*3+1] = Ad24y_rms
                        A_rms_z[i, j*3+1] = Ad24z_rms
                        A_rms_x[i, j*3+2] = Acx_rms
                        A_rms_y[i, j*3+2] = Acy_rms
                        A_rms_z[i, j*3+2] = Acz_rms

                    np.savetxt(path + f"ex_rms.txt", e_rms_x)
                    np.savetxt(path + f"ey_rms.txt", e_rms_y)
                    np.savetxt(path + f"ez_rms.txt", e_rms_z)
                    np.savetxt(path + f"Ax_rms.txt", A_rms_x)
                    np.savetxt(path + f"Ay_rms.txt", A_rms_y)
                    np.savetxt(path + f"Az_rms.txt", A_rms_z)

            else:
                e_rms_x = np.loadtxt(path + f"ex_rms.txt")
                e_rms_y = np.loadtxt(path + f"ey_rms.txt")
                e_rms_z = np.loadtxt(path + f"ez_rms.txt")
                A_rms_x = np.loadtxt(path + f"Ax_rms.txt")
                A_rms_y = np.loadtxt(path + f"Ay_rms.txt")
                A_rms_z = np.loadtxt(path + f"Az_rms.txt")

            fig, ax = create_three_plots("Parameter[-]", r"RMS [$m/s^{2}$]", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'], False, True, sharey=True)
            ax[0].plot(e_rms_x[:, 0], marker='o', label="Error_rms")
            ax[0].plot(A_rms_x[:, 0], marker='o', label="Design_matrix_rms")
            ax[1].plot(e_rms_y[:, 0], marker='o', label="Error_rms")
            ax[1].plot(A_rms_y[:, 0], marker='o', label="Design_matrix_rms")
            ax[2].plot(e_rms_z[:, 0], marker='o', label="Error_rms")
            ax[2].plot(A_rms_z[:, 0], marker='o', label="Design_matrix_rms")

            fig1, ax1 = create_three_plots("Parameter[-]", r"RMS [$m/s^{2}$]", [r'$a_{d24x}$', r'$a_{d24y}$', r'$a_{d24z}$'], False, True, sharey=True)
            ax1[0].plot(e_rms_x[:, 1], marker='o', label="Error_rms")
            ax1[0].plot(A_rms_x[:, 1], marker='o', label="Design_matrix_rms")
            ax1[1].plot(e_rms_y[:, 1], marker='o', label="Error_rms")
            ax1[1].plot(A_rms_y[:, 1], marker='o', label="Design_matrix_rms")
            ax1[2].plot(e_rms_z[:, 1], marker='o', label="Error_rms")
            ax1[2].plot(A_rms_z[:, 1], marker='o', label="Design_matrix_rms")

            fig2, ax2 = create_three_plots("Parameter [-]", r"RMS [$m/s^{2}$]",
                                           [r'$a_{c13x} - a_{c24x}$', r'$a_{c13y} - a_{c24y}$', r'$a_{c13z} - a_{c24z}$'], False, True, sharey=True)
            ax2[0].plot(e_rms_x[:, 2], marker='o', label="Error_rms")
            ax2[0].plot(A_rms_x[:, 2], marker='o', label="Design_matrix_rms")
            ax2[1].plot(e_rms_y[:, 2], marker='o', label="Error_rms")
            ax2[1].plot(A_rms_y[:, 2], marker='o', label="Design_matrix_rms")
            ax2[2].plot(e_rms_z[:, 2], marker='o', label="Error_rms")
            ax2[2].plot(A_rms_z[:, 2], marker='o', label="Design_matrix_rms")

            fig3, ax3 = create_three_plots("Parameter[-]", r"Factor", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'], False, True)
            ax3[0].plot(e_rms_x[:, 0] / A_rms_x[:, 0], marker='o')
            ax3[1].plot(e_rms_y[:, 0] / A_rms_y[:, 0], marker='o')
            ax3[2].plot(e_rms_z[:, 0] / A_rms_z[:, 0], marker='o')

            fig4, ax4 = create_three_plots("Parameter[-]", r"Factor", [r'$a_{d24x}$', r'$a_{d24y}$', r'$a_{d24z}$'], False, True)
            ax4[0].plot(e_rms_x[:, 1] / A_rms_x[:, 1], marker='o')
            ax4[1].plot(e_rms_y[:, 1] / A_rms_y[:, 1], marker='o')
            ax4[2].plot(e_rms_z[:, 1] / A_rms_z[:, 1], marker='o')

            fig5, ax5 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{c13x} - a_{c24x}$', r'$a_{c13y} - a_{c24y}$', r'$a_{c13z} - a_{c24z}$'],
                                           False, True, sharey=True)
            ax5[0].plot(e_rms_x[:, 2] / A_rms_x[:, 2], marker='o')
            ax5[1].plot(e_rms_y[:, 2] / A_rms_y[:, 2], marker='o')
            ax5[2].plot(e_rms_z[:, 2] / A_rms_z[:, 2], marker='o')

            fig6, ax6 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$'],
                                           False, False, sharey=True)
            ax6[0].plot(e_rms_x[:, 0] / e_rms_x[:, 3], marker='o')
            ax6[1].plot(e_rms_y[:, 0] / e_rms_y[:, 3], marker='o')
            ax6[2].plot(e_rms_z[:, 0] / e_rms_z[:, 3], marker='o')

            fig7, ax7 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{d24x}$', r'$a_{d24y}$', r'$a_{d24z}$'],
                                           False, False, sharey=True)
            ax7[0].plot(e_rms_x[:, 1] / e_rms_x[:, 4], marker='o')
            ax7[1].plot(e_rms_y[:, 1] / e_rms_y[:, 4], marker='o')
            ax7[2].plot(e_rms_z[:, 1] / e_rms_z[:, 4], marker='o')

            fig8, ax8 = create_three_plots("Parameter [-]", r"Factor", [r'$a_{c13x} - a_{c24x}$', r'$a_{c13y} - a_{c24y}$', r'$a_{c13z} - a_{c24z}$'],
                                           False, False, sharey=True)
            ax8[0].plot(e_rms_x[:, 2] / e_rms_x[:, 5], marker='o')
            ax8[1].plot(e_rms_y[:, 2] / e_rms_y[:, 5], marker='o')
            ax8[2].plot(e_rms_z[:, 2] / e_rms_z[:, 5], marker='o')

            height = -1.4
            const_x_lines = [-0.5, 8.5, 17.5, 26.5, 35.5, 47.5, 56.5, 63.5]
            label_y_coord = -0.35
            fontsize = 10
            for i, axes in enumerate([ax, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]):
                for j in range(3):
                    axes[j].legend(handlelength=3.5, fontsize=13)
                    axes[j].legend(handlelength=3.5, fontsize=13)
                    if not (i == 6 or i == 7 or i==8):
                        axes[j].yaxis.set_minor_locator(LogLocator(subs='all', numticks=20))
                    axes[j].grid(which='minor', axis='y', linestyle=':', linewidth=1)
                    axes[j].xaxis.set_major_locator(MultipleLocator(5))
                    for k in range(len(const_x_lines)):
                        axes[j].axvline(x=const_x_lines[k], color='k', linestyle='--', alpha=0.6)
                    draw_labels(axes[j], const_x_lines, label_y_coord, height, fontsize, layout)

            save_figure(fig, f"Verification/Design_Matrix/Configuration_{layout}/Design_Matrix_rms_diff_13.svg")
            save_figure(fig1, f"Verification/Design_Matrix/Configuration_{layout}/Design_Matrix_rms_diff_24.svg")
            save_figure(fig2, f"Verification/Design_Matrix/Configuration_{layout}/Design_Matrix_rms_com.svg")
            save_figure(fig3, f"Verification/Design_Matrix/Configuration_{layout}/Factor_diff_13.svg")
            save_figure(fig4, f"Verification/Design_Matrix/Configuration_{layout}/Factor_diff_24.svg")
            save_figure(fig5, f"Verification/Design_Matrix/Configuration_{layout}/Factor_com.svg")
            save_figure(fig6, f"Verification/Design_Matrix/Configuration_{layout}/Error_behaviour_diff_13.svg")
            save_figure(fig7, f"Verification/Design_Matrix/Configuration_{layout}/Error_behaviour_diff_24.svg")
            save_figure(fig8, f"Verification/Design_Matrix/Configuration_{layout}/Error_behaviour_com.svg")


def verify_lstsq(Acc_lst, layout, noise_switch=True, load_data=False, validation=False, sensitivity_analysis=False, **kwargs):
    """
    Verify if the least squares algorithm is implemented correctly by comparing the estimated parameters with the true parameters. Can switch on
    noise to check if the decorrelation filters work as intended.
    :param Acc_lst:
        List of Accelerometer objects
    :param layout:
        Layout of the accelerometers
    :param noise_switch:
        Switch on noise to check if the decorrelation filters work as intended
    :param load_data:
        Load the data from the saved files
    :param validation:
        Switch between validation and verification
    :param sensitivity_analysis:
        Perform sensitivity analysis
    :param kwargs:
        Provide the path for the sensitivity analysis
    :return:
    """
    if validation:
        folder = "Validation"
        save_path = f"{folder}/Least_squares/Configuration_{layout}/"
        if noise_switch:
            path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noisy/"
        else:
            path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noiseless/"

    elif sensitivity_analysis:
        folder = f"Sensitivity_analysis/{kwargs['path']}"
        save_path = f"{folder}/Configuration_{layout}/"
        if noise_switch:
            path = f"../SimulationOutput/Output/{folder}/Configuration_{layout}/Noisy/"
        else:
            path = f"../SimulationOutput/Output/{folder}/Configuration_{layout}/Noiseless/"

    else:
        if 'path' in kwargs:
            path = kwargs['path']
            if 'name' in kwargs:
                save_path = f"Test_figures/{kwargs['name']}/"
            else:
                save_path = f"Test_figures/"
        else:
            folder = "Verification"
            save_path = f"{folder}/Least_squares/Configuration_{layout}/"
            if noise_switch:
                path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noisy/"
            else:
                path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noiseless/"



    if not os.path.exists(path):
        os.makedirs(path)

    if load_data:
        x_err = np.load(path + "x_err.npy")
        x_err_initial = np.loadtxt(path + "x_err_initial.txt")
        residuals = np.load(path + "residuals.npy")
        cov_par = np.load(path + "covariance_matrix.npy")
        x0 = np.loadtxt(path + "x0.txt")
        x_true = np.loadtxt(path + "x_true.txt")
    else:
        NFFT = 86400 // 3
        NFFT = NFFT if NFFT % 2 == 1 else NFFT + 1
        start = time.time()
        x_err_initial, x_err, residuals, cov_par, x0, x_true, sigma, digit_loss = calibrate(Acc_lst, layout, NFFT, noise_switch)
        np.save(path + "x_err.npy", x_err)
        np.savetxt(path + "x_err_initial.txt", x_err_initial)
        np.save(path + "residuals.npy", residuals)
        np.save(path + "covariance_matrix.npy", cov_par)
        np.savetxt(path + "x0.txt", x0)
        np.savetxt(path + "x_true.txt", x_true)
        print(f"Elapsed time: {time.time() - start} s")

    NFFT = 5*5401
    match layout:
        case 1:
            ##########################################################################################################
            # Estimated parameters vs True parameters
            ##########################################################################################################

            # Calculate the standard deviation of the estimated parameters
            std_dev = np.sqrt(np.diag(cov_par))
            std_dev[18:24] = std_dev[18:24] * 1e6

            # Create the plot for errors between estimated and true parameters
            fig_err, ax_err = create_plots("Parameter [-]", "Estimation Error [-]", False, True)
            ax_err.plot(abs(x_err_initial), marker='o', label="Initial guess", color='b')
            ax_err.plot(abs(x_err[-1][-1]), marker='o', label="Estimated Solution", color='r')
            ax_err.plot(abs(std_dev), marker='o', label="Standard deviation", color='g')
            ax_err.grid(alpha=0.5)
            ax_err.set_xticks(np.arange(0, 30, 1), minor=True)
            ax_err.legend()
            const_x_lines = [-0.5, 8.5, 17.5, 23.5, 26.5, 28.5]
            for i in const_x_lines:
                ax_err.axvline(x=i, color='k', linestyle='--', alpha=0.65)

            min_error = min(abs(x_err[-1][-1]))
            height = 1.4 * min_error
            label_y_coord = 0.5 * min_error
            fontsize = 14
            draw_labels(ax_err, const_x_lines, label_y_coord, height, fontsize, layout)

            ##########################################################################################################
            # ASD Plots
            ##########################################################################################################

            ff_dx, P_dx = Acc_lst[0].psd_welch(residuals[0][:, -1], NFFT)
            ff_dy, P_dy = Acc_lst[0].psd_welch(residuals[1][:, -1], NFFT)
            ff_dz, P_dz = Acc_lst[0].psd_welch(residuals[2][:, -1], NFFT)

            noise_diff = get_expected_noise(Acc_lst, layout)[0]
            ff_ndx, Pdx_noise = Acc_lst[0].psd_welch(noise_diff[:, 0], NFFT)
            ff_ndy, Pdy_noise = Acc_lst[0].psd_welch(noise_diff[:, 1], NFFT)
            ff_ndz, Pdz_noise = Acc_lst[0].psd_welch(noise_diff[:, 2], NFFT)

            fig_titles_dx = [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$']
            fig, ax = create_three_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", fig_titles_dx, True, True, sharey=True)
            ax[0].plot(ff_ndx[1:], np.sqrt(Pdx_noise[1:]), label='Expected')
            ax[0].plot(ff_dx[1:], np.sqrt(P_dx[1:]), label='Estimated')
            ax[1].plot(ff_ndy[1:], np.sqrt(Pdy_noise[1:]), label='Expected')
            ax[1].plot(ff_dy[1:], np.sqrt(P_dy[1:]), label='Estimated')
            ax[2].plot(ff_ndz[1:], np.sqrt(Pdz_noise[1:]), label='Expected')
            ax[2].plot(ff_dz[1:], np.sqrt(P_dz[1:]), label='Estimated')

            # if noise_switch:
            #     # Plot the initial residuals
            #     ff_dx0, P_dx0 = Acc_lst[0].psd_welch(residuals[0][:, 0], NFFT)
            #     ff_dy0, P_dy0 = Acc_lst[0].psd_welch(residuals[1][:, 0], NFFT)
            #     ff_dz0, P_dz0 = Acc_lst[0].psd_welch(residuals[2][:, 0], NFFT)
            #     ax[0].plot(ff_dx0[1:], np.sqrt(P_dx0[1:]), label='Initial guess')
            #     ax[1].plot(ff_dy0[1:], np.sqrt(P_dy0[1:]), label='Initial guess')
            #     ax[2].plot(ff_dz0[1:], np.sqrt(P_dz0[1:]), label='Initial guess')

            for j in range(3):
                ax[j].legend(handlelength=3.5, fontsize=13, loc='upper left')
                ax[j].minorticks_on()
                if noise_switch:
                    ax[j].set_ylim(bottom=5e-13, top=None)

            if noise_switch:
                save_figure(fig_err, save_path + f"Estimation_error.svg")
                save_figure(fig, save_path + f"ASD_noise_diff.svg")
            else:
                save_figure(fig_err, save_path + f"Estimation_error_no_noise.svg")
                save_figure(fig, save_path + "ASD_no_noise_diff.svg")

        case 2:
            ##########################################################################################################
            # Estimated parameters vs True parameters
            ##########################################################################################################

            # Calculate the standard deviation of the estimated parameters
            std_dev = np.sqrt(np.diag(cov_par))
            # print(std_dev)
            std_dev[27:36] = std_dev[27:36] * 1e6

            # Create the plot for errors between estimated and true parameters
            fig_err, ax_err = create_plots("Parameter [-]", "Estimation Error [-]", False, True)
            ax_err.plot(abs(x_err_initial), marker='o', label="Initial guess", color='b')
            ax_err.plot(abs(x_err[-1][-1]), marker='o', label="Estimated Solution", color='r')
            # plot first step error
            ax_err.plot(abs(x_err[0][0]), marker='o', label="First step error", color='orange')
            # ax_err.plot(abs(std_dev), marker='o', label="Standard deviation", color='g')
            ax_err.grid(alpha=0.5)
            ax_err.set_xticks(np.arange(0, 47, 1), minor=True)
            ax_err.legend()
            const_x_lines = [-0.5, 8.5, 17.5, 26.5, 35.5, 41.5, 46.6]
            for i in const_x_lines:
                ax_err.axvline(x=i, color='k', linestyle='--', alpha=0.65)


            min_error = min(abs(x_err[-1][-1]))
            height = 1.4 * min_error
            label_y_coord = 0.5 * min_error
            fontsize = 14
            draw_labels(ax_err, const_x_lines, label_y_coord, height, fontsize, layout)

            ##########################################################################################################
            # ASD Plots
            ##########################################################################################################

            ff_dx, P_dx = Acc_lst[0].psd_welch(residuals[0][:, -1], NFFT)
            ff_dy, P_dy = Acc_lst[0].psd_welch(residuals[1][:, -1], NFFT)
            ff_dz, P_dz = Acc_lst[0].psd_welch(residuals[2][:, -1], NFFT)
            ff_cx, P_cx = Acc_lst[0].psd_welch(residuals[3][:, -1], NFFT)
            ff_cy, P_cy = Acc_lst[0].psd_welch(residuals[4][:, -1], NFFT)
            ff_cz, P_cz = Acc_lst[0].psd_welch(residuals[5][:, -1], NFFT)

            # Expected noise
            noise_diff = get_expected_noise(Acc_lst, layout)[0]
            noise_com = get_expected_noise(Acc_lst, layout)[1]

            ff_ndx, Pdx_noise = Acc_lst[0].psd_welch(noise_diff[:, 0], NFFT)
            ff_ndy, Pdy_noise = Acc_lst[0].psd_welch(noise_diff[:, 1], NFFT)
            ff_ndz, Pdz_noise = Acc_lst[0].psd_welch(noise_diff[:, 2], NFFT)
            ff_ncx, Pcx_noise = Acc_lst[0].psd_welch(noise_com[:, 0], NFFT)
            ff_ncy, Pcy_noise = Acc_lst[0].psd_welch(noise_com[:, 1], NFFT)
            ff_ncz, Pcz_noise = Acc_lst[0].psd_welch(noise_com[:, 2], NFFT)

            fig_titles_dx = [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$']
            fig_titles_cx = [r'$a_{c13x} - a_{2x}$', r'$a_{c13y} - a_{2y}$', r'$a_{c13z} - a_{2z}$']
            fig, ax = create_three_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", fig_titles_dx, True, True, sharey=True)
            ax[0].plot(ff_ndx[1:], np.sqrt(Pdx_noise[1:]), label='Expected', color='b')
            ax[0].plot(ff_dx[1:], np.sqrt(P_dx[1:]), label='Estimated', color='r', linestyle='--', alpha=0.7)
            ax[1].plot(ff_ndy[1:], np.sqrt(Pdy_noise[1:]), label='Expected', color='b')
            ax[1].plot(ff_dy[1:], np.sqrt(P_dy[1:]), label='Estimated', color='r', linestyle='--', alpha=0.7)
            ax[2].plot(ff_ndz[1:], np.sqrt(Pdz_noise[1:]), label='Expected', color='b')
            ax[2].plot(ff_dz[1:], np.sqrt(P_dz[1:]), label='Estimated', color='r', linestyle='--', alpha=0.7)

            # if noise_switch:
            #     # Plot the initial residuals
            #     ff_dx0, P_dx0 = Acc_lst[0].psd_welch(residuals[0][:, 0], NFFT)
            #     ff_dy0, P_dy0 = Acc_lst[0].psd_welch(residuals[1][:, 0], NFFT)
            #     ff_dz0, P_dz0 = Acc_lst[0].psd_welch(residuals[2][:, 0], NFFT)
            #     ax[0].plot(ff_dx0[1:], np.sqrt(P_dx0[1:]), label='Initial guess')
            #     ax[1].plot(ff_dy0[1:], np.sqrt(P_dy0[1:]), label='Initial guess')
            #     ax[2].plot(ff_dz0[1:], np.sqrt(P_dz0[1:]), label='Initial guess')

            fig1, ax1 = create_three_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", fig_titles_cx, True, True, sharey=True)
            ax1[0].plot(ff_ncx[1:], np.sqrt(Pcx_noise[1:]), label='Expected', color='b')
            ax1[0].plot(ff_cx[1:], np.sqrt(P_cx[1:]), label='Estimated', color='r', linestyle='--', alpha=0.7)
            ax1[1].plot(ff_ncy[1:], np.sqrt(Pcy_noise[1:]), label='Expected', color='b')
            ax1[1].plot(ff_cy[1:], np.sqrt(P_cy[1:]), label='Estimated', color='r', linestyle='--', alpha=0.7)
            ax1[2].plot(ff_ncz[1:], np.sqrt(Pcz_noise[1:]), label='Expected', color='b')
            ax1[2].plot(ff_cz[1:], np.sqrt(P_cz[1:]), label='Estimated', color='r', linestyle='--', alpha=0.7)

            # if noise_switch:
            #     # Plot the initial residuals
            #     ff_cx0, P_cx0 = Acc_lst[0].psd_welch(residuals[3][:, 0], NFFT)
            #     ff_cy0, P_cy0 = Acc_lst[0].psd_welch(residuals[4][:, 0], NFFT)
            #     ff_cz0, P_cz0 = Acc_lst[0].psd_welch(residuals[5][:, 0], NFFT)
            #     ax1[0].plot(ff_cx0[1:], np.sqrt(P_cx0[1:]), label='Initial guess')
            #     ax1[1].plot(ff_cy0[1:], np.sqrt(P_cy0[1:]), label='Initial guess')
            #     ax1[2].plot(ff_cz0[1:], np.sqrt(P_cz0[1:]), label='Initial guess')

            for j in range(3):
                ax[j].legend(handlelength=3.5, fontsize=13, loc='upper left')
                ax1[j].legend(handlelength=3.5, fontsize=13, loc='upper left')
                if noise_switch:
                    ax[j].set_ylim(bottom=5e-13, top=None)
                    ax1[j].set_ylim(bottom=5e-13, top=None)

            if noise_switch:
                save_figure(fig_err, save_path + f"Estimation_error.svg")
                save_figure(fig, save_path + f"ASD_noise_diff.svg")
                save_figure(fig1, save_path + f"ASD_noise_comm.svg")

            else:
                save_figure(fig_err, save_path + f"Estimation_error_no_noise.svg")
                save_figure(fig, save_path + f"ASD_no_noise_diff.svg")
                save_figure(fig1, save_path + f"ASD_no_noise_comm.svg")

        case 3:
            ##########################################################################################################
            # Estimated parameters vs True parameters
            ##########################################################################################################
            # Print the accelerometers arm lengths
            M1, M2, M3, M4, K1, K2, K3, K4, W1, W2, W3, W4, dr1, dr2, dr3, dr4 = acc_cal_par_vec_to_mat(x0, Acc_lst, layout=3)

            r3 = Acc_lst[2].r + dr3
            r1 = Acc_lst[0].r + dr1
            r2 = Acc_lst[1].r + dr2
            r4 = Acc_lst[3].r + dr4

            r13 = r3 - r1
            r24 = r4 - r2
            print(f"r13: {r13}")
            print(f"r24: {r24}")
            print(f"Length r13: {np.linalg.norm(r13)}")
            print(f"Length r24: {np.linalg.norm(r24)}")

            # Calculate the standard deviation of the estimated parameters
            std_dev = np.sqrt(np.diag(cov_par))
            # # print(std_dev)
            std_dev[36:48] = std_dev[36:48] * 1e6

            x_err_final = x_err[-1][-1]
            # Create the plot for errors between estimated and true parameters
            fig_err, ax_err = create_plots("Parameter [-]", "Estimation Error [-]", False, True)
            ax_err.plot(abs(x_err_initial), marker='o', label="Initial guess", color='b')
            ax_err.plot(abs(x_err_final), marker='o', label="Estimated Solution", color='r')
            ax_err.plot(abs(std_dev), marker='o', label="Standard deviation", color='g')
            ax_err.grid(alpha=0.5)
            ax_err.set_xticks(np.arange(0, 64, 1), minor=True)
            ax_err.legend()

            const_x_lines = [-0.5, 8.5, 17.5, 26.5, 35.5, 47.5, 56.5, 63.5]
            for i in const_x_lines:
                ax_err.axvline(x=i, color='k', linestyle='--', alpha=0.65)

            min_error = min(abs(x_err[-1][-1]))
            height = 1.4 * min_error
            label_y_coord = 0.5 * min_error
            fontsize = 14
            draw_labels(ax_err, const_x_lines, label_y_coord, height, fontsize, layout)

            ##########################################################################################################
            # ASD Plots
            ##########################################################################################################

            ff_d13x, P_d13x = Acc_lst[0].psd_welch(residuals[0][:, -1], NFFT)
            ff_d13y, P_d13y = Acc_lst[0].psd_welch(residuals[1][:, -1], NFFT)
            ff_d13z, P_d13z = Acc_lst[0].psd_welch(residuals[2][:, -1], NFFT)
            ff_d24x, P_d24x = Acc_lst[0].psd_welch(residuals[3][:, -1], NFFT)
            ff_d24y, P_d24y = Acc_lst[0].psd_welch(residuals[4][:, -1], NFFT)
            ff_d24z, P_d24z = Acc_lst[0].psd_welch(residuals[5][:, -1], NFFT)
            ff_cx, P_cx = Acc_lst[0].psd_welch(residuals[6][:, -1], NFFT)
            ff_cy, P_cy = Acc_lst[0].psd_welch(residuals[7][:, -1], NFFT)
            ff_cz, P_cz = Acc_lst[0].psd_welch(residuals[8][:, -1], NFFT)

            # Expected noise
            noise_diff13 = get_expected_noise(Acc_lst, layout)[0]
            noise_diff24 = get_expected_noise(Acc_lst, layout)[1]
            noise_com = get_expected_noise(Acc_lst, layout)[2]

            ff_nd13x, Pd13x_noise = Acc_lst[0].psd_welch(noise_diff13[:, 0], NFFT)
            ff_nd13y, Pd13y_noise = Acc_lst[0].psd_welch(noise_diff13[:, 1], NFFT)
            ff_nd13z, Pd13z_noise = Acc_lst[0].psd_welch(noise_diff13[:, 2], NFFT)
            ff_nd24x, Pd24x_noise = Acc_lst[0].psd_welch(noise_diff24[:, 0], NFFT)
            ff_nd24y, Pd24y_noise = Acc_lst[0].psd_welch(noise_diff24[:, 1], NFFT)
            ff_nd24z, Pd24z_noise = Acc_lst[0].psd_welch(noise_diff24[:, 2], NFFT)
            ff_ncx, Pcx_noise = Acc_lst[0].psd_welch(noise_com[:, 0], NFFT)
            ff_ncy, Pcy_noise = Acc_lst[0].psd_welch(noise_com[:, 1], NFFT)
            ff_ncz, Pcz_noise = Acc_lst[0].psd_welch(noise_com[:, 2], NFFT)

            fig_titles_d13 = [r'$a_{d13x}$', r'$a_{d13y}$', r'$a_{d13z}$']
            fig_titles_d24 = [r'$a_{d24x}$', r'$a_{d24y}$', r'$a_{d24z}$']
            fig_titles_c = [r'$a_{c13x} - a_{2x}$', r'$a_{c13y} - a_{2y}$', r'$a_{c13z} - a_{2z}$']
            fig, ax = create_three_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", fig_titles_d13, True, True, sharey=True)
            ax[0].plot(ff_nd13x[1:], np.sqrt(Pd13x_noise[1:]), label='Expected')
            ax[0].plot(ff_d13x[1:], np.sqrt(P_d13x[1:]), label='Estimated')
            ax[1].plot(ff_nd13y[1:], np.sqrt(Pd13y_noise[1:]), label='Expected')
            ax[1].plot(ff_d13y[1:], np.sqrt(P_d13y[1:]), label='Estimated')
            ax[2].plot(ff_nd13z[1:], np.sqrt(Pd13z_noise[1:]), label='Expected')
            ax[2].plot(ff_d13z[1:], np.sqrt(P_d13z[1:]), label='Estimated')

            # if noise_switch:
            #     # Plot the initial residuals
            #     ff_d13x0, P_d13x0 = Acc_lst[0].psd_welch(residuals[0][:, 0], NFFT)
            #     ff_d13y0, P_d13y0 = Acc_lst[0].psd_welch(residuals[1][:, 0], NFFT)
            #     ff_d13z0, P_d13z0 = Acc_lst[0].psd_welch(residuals[2][:, 0], NFFT)
            #     ax[0].plot(ff_d13x0[1:], np.sqrt(P_d13x0[1:]), label='Initial guess')
            #     ax[1].plot(ff_d13y0[1:], np.sqrt(P_d13y0[1:]), label='Initial guess')
            #     ax[2].plot(ff_d13z0[1:], np.sqrt(P_d13z0[1:]), label='Initial guess')

            fig1, ax1 = create_three_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", fig_titles_d24, True, True, sharey=True)
            ax1[0].plot(ff_nd24x[1:], np.sqrt(Pd24x_noise[1:]), label='Expected')
            ax1[0].plot(ff_d24x[1:], np.sqrt(P_d24x[1:]), label='Estimated')
            ax1[1].plot(ff_nd24y[1:], np.sqrt(Pd24y_noise[1:]), label='Expected')
            ax1[1].plot(ff_d24y[1:], np.sqrt(P_d24y[1:]), label='Estimated')
            ax1[2].plot(ff_nd24z[1:], np.sqrt(Pd24z_noise[1:]), label='Expected')
            ax1[2].plot(ff_d24z[1:], np.sqrt(P_d24z[1:]), label='Estimated')

            # if noise_switch:
            #     # Plot the initial residuals
            #     ff_d24x0, P_d24x0 = Acc_lst[0].psd_welch(residuals[3][:, 0], NFFT)
            #     ff_d24y0, P_d24y0 = Acc_lst[0].psd_welch(residuals[4][:, 0], NFFT)
            #     ff_d24z0, P_d24z0 = Acc_lst[0].psd_welch(residuals[5][:, 0], NFFT)
            #     ax1[0].plot(ff_d24x0[1:], np.sqrt(P_d24x0[1:]), label='Initial guess')
            #     ax1[1].plot(ff_d24y0[1:], np.sqrt(P_d24y0[1:]), label='Initial guess')
            #     ax1[2].plot(ff_d24z0[1:], np.sqrt(P_d24z0[1:]), label='Initial guess')

            fig2, ax2 = create_three_plots("Frequency [Hz]", r"ASD [$m/s^{2}/\sqrt{Hz}$]", fig_titles_c, True, True, sharey=True)
            ax2[0].plot(ff_ncx[1:], np.sqrt(Pcx_noise[1:]), label='Expected')
            ax2[0].plot(ff_cx[1:], np.sqrt(P_cx[1:]), label='Estimated')
            ax2[1].plot(ff_ncy[1:], np.sqrt(Pcy_noise[1:]), label='Expected')
            ax2[1].plot(ff_cy[1:], np.sqrt(P_cy[1:]), label='Estimated')
            ax2[2].plot(ff_ncz[1:], np.sqrt(Pcz_noise[1:]), label='Expected')
            ax2[2].plot(ff_cz[1:], np.sqrt(P_cz[1:]), label='Estimated')

            # if noise_switch:
            #     # Plot the initial residuals
            #     ff_cx0, P_cx0 = Acc_lst[0].psd_welch(residuals[6][:, 0], NFFT)
            #     ff_cy0, P_cy0 = Acc_lst[0].psd_welch(residuals[7][:, 0], NFFT)
            #     ff_cz0, P_cz0 = Acc_lst[0].psd_welch(residuals[8][:, 0], NFFT)
            #     ax2[0].plot(ff_cx0[1:], np.sqrt(P_cx0[1:]), label='Initial guess')
            #     ax2[1].plot(ff_cy0[1:], np.sqrt(P_cy0[1:]), label='Initial guess')
            #     ax2[2].plot(ff_cz0[1:], np.sqrt(P_cz0[1:]), label='Initial guess')

            for j in range(3):
                ax[j].legend(handlelength=3.5, fontsize=13, loc='upper left')
                ax1[j].legend(handlelength=3.5, fontsize=13, loc='upper left')
                ax2[j].legend(handlelength=3.5, fontsize=13, loc='upper left')
                if noise_switch:
                    ax[j].set_ylim(bottom=5e-13, top=None)
                    ax1[j].set_ylim(bottom=5e-13, top=None)
                    ax2[j].set_ylim(bottom=5e-13, top=None)

            if noise_switch:
                save_figure(fig_err, save_path + f"Estimation_error.svg")
                save_figure(fig, save_path + f"ASD_noise_diff_13.svg")
                save_figure(fig1, save_path + f"ASD_noise_diff_24.svg")
                save_figure(fig2, save_path + f"ASD_noise_comm.svg")

            else:
                save_figure(fig_err, save_path + f"Estimation_error_no_noise.svg")
                save_figure(fig, save_path + f"ASD_no_noise_diff_13.svg")
                save_figure(fig1, save_path + f"ASD_no_noise_diff_24.svg")
                save_figure(fig2, save_path + f"ASD_no_noise_comm.svg")


def parameter_estimation(layout, noise_switch=True, validation=False, sensitivity_analysis=False, **kwargs):
    if sensitivity_analysis:
        folder = "Sensitivity_analysis/" + kwargs['path']
        save_path = f"{folder}/Configuration_{layout}/"
        if noise_switch:
            path = f"../SimulationOutput/Output/{folder}/Configuration_{layout}/Noisy/"
        else:
            path = f"../SimulationOutput/Output/{folder}/Configuration_{layout}/Noiseless/"

    else:
        if validation:
            folder = "Validation"
            save_path = f"{folder}/Least_squares/Configuration_{layout}/"
        else:
            folder = "Verification"
            save_path = f"{folder}/Design_matrix/Configuration_{layout}/"
        if noise_switch:
            path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noisy/"
        else:
            path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noiseless/"


    x0 = np.loadtxt(path + "x0.txt")
    x_true = np.loadtxt(path + "x_true.txt")
    cov_x = np.load(path + "covariance_matrix.npy")

    sd_x = np.sqrt(np.diag(cov_x))
    if layout == 1:
        sd_x[18:24] = sd_x[18:24] * 1e6
        par = np.arange(0, 29, 1)
    elif layout == 2:
        sd_x[27:36] = sd_x[27:36] * 1e6
        par = np.arange(0, 47, 1)
    else:
        sd_x[36:48] = sd_x[36:48] * 1e6
        par = np.arange(0, 64, 1)

    k = 3
    ci = k*sd_x
    confidence = 99.7

    z_scores = (x0 - x_true) / sd_x

    if validation:
        if layout == 1:
            z_scores = z_scores[:-2]
            par = par[:-2]
        elif layout == 2:
            z_scores = z_scores[:-5]
            par = par[:-5]
        else:
            z_scores = z_scores[:-7]
            par = par[:-7]

    # Plot the normalized confidence intervals
    fig3, ax3 = create_plots("Parameter", "Value", x_scale_log=False, y_scale_log=False)
    # ax3.errorbar(par, z_scores, yerr=3, fmt='o', label='Estimated Parameter (CI)', capsize=5)
    ax3.scatter(par, z_scores, label='Estimated Parameter', color='b')
    ax3.axhline(k, color='r', linestyle='--', label=f"{confidence}% CI")
    ax3.axhline(-k, color='r', linestyle='--')

    # ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax3.scatter(par, x_true, label='True Value', color='r')
    ax3.legend()

    save_figure(fig3, save_path + f"Parameter_confidence_intervals.svg")



def singular_matrix_analysis(layout, noise_switch=True, sensitivity_analysis=False, **kwargs):
    if sensitivity_analysis:
        folder = "Sensitivity_analysis/" + kwargs['path']
        save_path = f"{folder}/Configuration_{layout}/"
        if noise_switch:
            path = f"../SimulationOutput/Output/{folder}/Configuration_{layout}/Noisy/"
        else:
            path = f"../SimulationOutput/Output/{folder}/Configuration_{layout}/Noiseless/"

    else:
        folder = "Verification"
        save_path = f"{folder}/Design_matrix/Configuration_{layout}/"
        if noise_switch:
            path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noisy/"
        else:
            path = f"../SimulationOutput/Output/{folder}/Least_squares/Configuration_{layout}/Noiseless/"

    # Af = np.load(path + "Af.npy")
    x0 = np.loadtxt(path + "x0.txt")
    x_true = np.loadtxt(path + "x_true.txt")
    cov_x = np.load(path + "covariance_matrix.npy")

    sd_x = np.sqrt(np.diag(cov_x))
    if layout == 1:
        sd_x[18:24] = sd_x[18:24] * 1e6
        cov_x[18:24, 18:24] = cov_x[18:24, 18:24] * 1e12
        par = np.arange(0, 29, 1)
    elif layout == 2:
        sd_x[27:36] = sd_x[27:36] * 1e6
        cov_x[27:36, 27:36] = cov_x[27:36, 27:36] * 1e12
        par = np.arange(0, 47, 1)
    else:
        sd_x[36:48] = sd_x[36:48] * 1e6
        cov_x[36:48, 36:48] = cov_x[36:48, 36:48] * 1e12
        par = np.arange(0, 64, 1)

    # Compute correlation matrix
    rho = cov_x / np.outer(sd_x, sd_x)
    # Singular value decomposition
    # U, S, VT = np.linalg.svd(Af.T @ Af, full_matrices=False)
    # S = np.diag(S)

    # Compute the normalized confidence intervals
    x_normalized = (x0 - x_true) / sd_x
    x_true_normalized = np.zeros_like(x0)
    confidence_level = 3
    lower_bound = x_normalized - confidence_level
    upper_bound = x_normalized + confidence_level

    # Find the largest of the distance between the bounds and the true value
    x_worst = np.zeros(len(x0))
    for i in range(len(x_normalized)):
        if abs(lower_bound[i]) < abs(upper_bound[i]):
            x_worst[i] = x0[i] + (3 * sd_x[i])
        else:
            x_worst[i] = x0[i] - (3 * sd_x[i])

    # fig, ax = create_plots("Value [-]", "Logarithm of Singular value [-]", False, True)
    # ax.plot(np.diag(S), marker='o')
    # ax.minorticks_on()
    # ax.grid(which='minor', axis='y', linestyle=':', linewidth=1)

    # # Create cummulative sum plot
    # fig1, ax1 = create_plots("Value [-]", "Cummulative sum of Singular value [-]", False, False)
    # ax1.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    # ax1.minorticks_on()
    # ax1.grid(which='minor', axis='y', linestyle=':', linewidth=1)
    #
    fig2, ax2 = plt.subplots()
    cax = ax2.matshow(rho, aspect='auto', cmap='seismic')
    fig2.colorbar(cax, ax=ax2)
    # Use the sns plot for enhanced plot with values in the
    # sns.heatmap(rho, cmap='seismic', annot=True, fmt=".1f", cbar_kws={'label': 'Correlation coefficient [-]'})
    # plt.show()

    # Plot the normalized confidence intervals
    fig3, ax3 = create_plots("Parameter", "Value", x_scale_log=False, y_scale_log=False)
    ax3.errorbar(par, x_normalized, yerr=confidence_level, fmt='o', label='Estimated Parameter (CI)', capsize=5)
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax3.scatter(par, x_true_normalized, label='True Value', color='r')
    ax3.legend()
    plt.show()
    # save_figure(fig, save_path + f"Singular_matrix.svg")
    # save_figure(fig1, save_path + f"CumSum.svg")
    # save_figure(fig2, save_path + f"Cov_matrix_color_map.svg")
    # save_figure(fig3, save_path + f"Parameter_confidence_intervals.svg")

    return x_worst, sd_x
