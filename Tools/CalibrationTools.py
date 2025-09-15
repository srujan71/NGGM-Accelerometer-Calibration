import numpy as np
from Tools.SignalProcessingUtilities import bandpass_filter


def acc_cal_par_mat_to_vec(Acc_lst, layout):
    match layout:
        case 1:
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]

            dMc12 = (Acc1.M + Acc2.M) / 2 - np.identity(3)
            dMd12 = (Acc1.M - Acc2.M) / 2
            Wd12 = (Acc1.W - Acc2.W) / 2

            drc12 = (Acc1.dr + Acc2.dr) / 2
            drd12 = (Acc1.dr - Acc2.dr) / 2

            x = np.zeros(29)
            x[0:9] = dMc12.T.flatten()
            x[9:18] = dMd12.T.flatten()
            x[18:21] = np.diag(Acc1.K)
            x[21:24] = np.diag(Acc2.K)
            x[24] = Wd12[1, 0]
            x[25] = Wd12[2, 1]
            x[26] = Wd12[1, 2]

            if Acc1.r[0] != 0:
                # Cannot estimate drd12x if the accelerometers are placed on x-axis
                x[27] = drd12[1]
                x[28] = drd12[2]
            elif Acc1.r[1] != 0:
                # Cannot estimate drd12y if the accelerometers are placed on y-axis
                x[27] = drd12[0]
                x[28] = drd12[2]
            else:
                # Cannot estimate drd12z if the accelerometers are placed on z-axis
                x[27] = drd12[0]
                x[28] = drd12[1]

            return x

        case 2:
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]
            Acc3 = Acc_lst[2]
            dM2 = Acc2.M - np.identity(3)
            dMc13 = (Acc1.M + Acc3.M) / 2 - np.identity(3)
            dMd13 = (Acc1.M - Acc3.M) / 2
            Wd13 = (Acc1.W - Acc3.W) / 2

            # Ensure that W2 = 0
            assert not np.any(Acc2.W)

            Wc = Acc2.W - (Acc1.W + Acc3.W) / 2
            drc13 = (Acc1.dr + Acc3.dr) / 2
            drd13 = (Acc1.dr - Acc3.dr) / 2

            x = np.zeros(47)
            x[0:9] = dM2.T.flatten()
            x[9:18] = dMc13.T.flatten()
            x[18:27] = dMd13.T.flatten()
            x[27:30] = np.diag(Acc1.K)
            x[30:33] = np.diag(Acc2.K)
            x[33:36] = np.diag(Acc3.K)
            x[36] = Wd13[1, 0]
            x[37] = Wd13[2, 1]
            x[38] = Wd13[1, 2]
            x[39] = Wc[1, 0]
            x[40] = Wc[2, 1]
            x[41] = Wc[1, 2]
            x[42:45] = drc13

            if Acc1.r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                x[45] = drd13[1]
                x[46] = drd13[2]
            elif Acc1.r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                x[45] = drd13[0]
                x[46] = drd13[2]
            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                x[45] = drd13[0]
                x[46] = drd13[1]

            return x

        case 3:
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]
            Acc3 = Acc_lst[2]
            Acc4 = Acc_lst[3]

            dMc13 = (Acc1.M + Acc3.M) / 2 - np.identity(3)
            dMc24 = (Acc2.M + Acc4.M) / 2 - np.identity(3)
            dMd13 = (Acc1.M - Acc3.M) / 2
            dMd24 = (Acc2.M - Acc4.M) / 2
            Wd13 = (Acc1.W - Acc3.W) / 2

            # Ensure that W2 = -W4
            assert np.allclose(Acc2.W, -Acc4.W, 1e-6, 1e-9)

            Wd24 = (Acc2.W - Acc4.W) / 2
            Wc = np.zeros((3, 3)) - (Acc1.W + Acc3.W) / 2

            drc13 = (Acc1.dr + Acc3.dr) / 2
            drc24 = (Acc2.dr + Acc4.dr) / 2
            drd13 = (Acc1.dr - Acc3.dr) / 2
            drd24 = (Acc2.dr - Acc4.dr) / 2

            x = np.zeros(64)
            x[0:9] = dMc13.T.flatten()
            x[9:18] = dMc24.T.flatten()
            x[18:27] = dMd13.T.flatten()
            x[27:36] = dMd24.T.flatten()
            x[36:39] = np.diag(Acc1.K)
            x[39:42] = np.diag(Acc2.K)
            x[42:45] = np.diag(Acc3.K)
            x[45:48] = np.diag(Acc4.K)
            x[48] = Wd13[1, 0]
            x[49] = Wd13[2, 1]
            x[50] = Wd13[1, 2]
            x[51] = Wd24[1, 0]
            x[52] = Wd24[2, 1]
            x[53] = Wd24[1, 2]
            x[54] = Wc[1, 0]
            x[55] = Wc[2, 1]
            x[56] = Wc[1, 2]

            x[57:60] = drc13

            if Acc1.r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                x[60] = drd13[1]
                x[61] = drd13[2]
            elif Acc1.r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                x[60] = drd13[0]
                x[61] = drd13[2]
            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                x[60] = drd13[0]
                x[61] = drd13[1]

            if Acc2.r[0] != 0:
                # Cannot estimate drd24x if the accelerometers are placed on x-axis
                x[62] = drd24[1]
                x[63] = drd24[2]
            elif Acc2.r[1] != 0:
                # Cannot estimate drd24y if the accelerometers are placed on y-axis
                x[62] = drd24[0]
                x[63] = drd24[2]
            else:
                # Cannot estimate drd24z if the accelerometers are placed on z-axis
                x[62] = drd24[0]
                x[63] = drd24[1]

            return x


def acc_cal_par_vec_to_mat(x, Acc_lst, layout):
    match layout:
        case 1:
            # Two accelerometers
            dMc12 = np.reshape(x[0:9], (3, 3), order='F')
            dMd12 = np.reshape(x[9:18], (3, 3), order='F')
            K1 = np.diag(x[18:21])
            K2 = np.diag(x[21:24])
            Wd12 = np.zeros((3, 3))

            dM1 = dMc12 + dMd12
            dM2 = dMc12 - dMd12

            M1 = dM1 + np.eye(3)
            M2 = dM2 + np.eye(3)

            Wd12[1, 0] = x[24]
            Wd12[2, 1] = x[25]
            Wd12[1, 2] = x[26]

            # Cannot estimate them independently. Define Wc12 as 0
            W1 = Wd12
            W2 = -Wd12

            drc12 = np.zeros(3)     # Assume drc12 = 0. Cannot estimate it
            drd12 = np.zeros(3)
            if Acc_lst[0].r[0] != 0:
                # Cannot estimate drd12x if the accelerometers are placed on x-axis
                drd12[1] = x[27]
                drd12[2] = x[28]
            elif Acc_lst[0].r[1] != 0:
                # Cannot estimate drd12y if the accelerometers are placed on y-axis
                drd12[0] = x[27]
                drd12[2] = x[28]
            else:
                # Cannot estimate drd12z if the accelerometers are placed on z-axis
                drd12[0] = x[27]
                drd12[1] = x[28]

            dr1 = drc12 + drd12
            dr2 = drc12 - drd12

            return M1, M2, K1, K2, W1, W2, dr1, dr2

        case 2:
            # Three accelerometers
            dM2 = np.reshape(x[0:9], (3, 3), order='F')
            dMc13 = np.reshape(x[9:18], (3, 3), order='F')
            dMd13 = np.reshape(x[18:27], (3, 3), order='F')
            K1 = np.diag(x[27:30])
            K2 = np.diag(x[30:33])
            K3 = np.diag(x[33:36])
            Wd13 = np.zeros((3, 3))
            Wc13 = np.zeros((3, 3))

            dM1 = dMc13 + dMd13
            dM3 = dMc13 - dMd13

            M1 = dM1 + np.eye(3)
            M2 = dM2 + np.eye(3)
            M3 = dM3 + np.eye(3)

            Wd13[1, 0] = x[36]
            Wd13[2, 1] = x[37]
            Wd13[1, 2] = x[38]

            # x[39:42] = W2 - Wc13. Assume W2 = 0
            Wc13[1, 0] = -x[39]
            Wc13[2, 1] = -x[40]
            Wc13[1, 2] = -x[41]

            W1 = Wc13 + Wd13
            W3 = Wc13 - Wd13
            W2 = np.zeros((3, 3))

            drc13 = x[42:45]
            drd13 = np.zeros(3)

            if Acc_lst[0].r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                drd13[1] = x[45]
                drd13[2] = x[46]
            elif Acc_lst[0].r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                drd13[0] = x[45]
                drd13[2] = x[46]
            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                drd13[0] = x[45]
                drd13[1] = x[46]

            dr1 = drc13 + drd13
            dr2 = np.zeros(3)
            dr3 = drc13 - drd13

            return M1, M2, M3, K1, K2, K3, W1, W2, W3, dr1, dr2, dr3

        case 3:
            # Four accelerometers
            dMc13 = np.reshape(x[0:9], (3, 3), order='F')
            dMc24 = np.reshape(x[9:18], (3, 3), order='F')
            dMd13 = np.reshape(x[18:27], (3, 3), order='F')
            dMd24 = np.reshape(x[27:36], (3, 3), order='F')
            K1 = np.diag(x[36:39])
            K2 = np.diag(x[39:42])
            K3 = np.diag(x[42:45])
            K4 = np.diag(x[45:48])
            Wd13 = np.zeros((3, 3))
            Wd24 = np.zeros((3, 3))
            Wc13 = np.zeros((3, 3))

            dM1 = dMc13 + dMd13
            dM2 = dMc24 + dMd24
            dM3 = dMc13 - dMd13
            dM4 = dMc24 - dMd24

            M1 = dM1 + np.eye(3)
            M2 = dM2 + np.eye(3)
            M3 = dM3 + np.eye(3)
            M4 = dM4 + np.eye(3)

            Wd13[1, 0] = x[48]
            Wd13[2, 1] = x[49]
            Wd13[1, 2] = x[50]

            Wd24[1, 0] = x[51]
            Wd24[2, 1] = x[52]
            Wd24[1, 2] = x[53]

            # x[54:57] = Wc24 - Wc13. Assume Wc24 = 0 by asserting the condition that W2 = -W4
            Wc13[1, 0] = -x[54]
            Wc13[2, 1] = -x[55]
            Wc13[1, 2] = -x[56]
            Wc24 = np.zeros((3, 3))

            W1 = Wc13 + Wd13
            W2 = Wc24 + Wd24
            W3 = Wc13 - Wd13
            W4 = Wc24 - Wd24

            drc13 = x[57:60]
            drc24 = np.zeros(3)     # Assume drc24 = 0. Cannot estimate it
            drd13 = np.zeros(3)
            drd24 = np.zeros(3)

            if Acc_lst[0].r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                drd13[1] = x[60]
                drd13[2] = x[61]
            elif Acc_lst[0].r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                drd13[0] = x[60]
                drd13[2] = x[61]
            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                drd13[0] = x[60]
                drd13[1] = x[61]

            if Acc_lst[1].r[0] != 0:
                # Cannot estimate drd24x if the accelerometers are placed on x-axis
                drd24[1] = x[62]
                drd24[2] = x[63]
            elif Acc_lst[1].r[1] != 0:
                # Cannot estimate drd24y if the accelerometers are placed on y-axis
                drd24[0] = x[62]
                drd24[2] = x[63]
            else:
                # Cannot estimate drd24z if the accelerometers are placed on z-axis
                drd24[0] = x[62]
                drd24[1] = x[63]

            dr1 = drc13 + drd13
            dr2 = drc24 + drd24
            dr3 = drc13 - drd13
            dr4 = drc24 - drd24

            return M1, M2, M3, M4, K1, K2, K3, K4, W1, W2, W3, W4, dr1, dr2, dr3, dr4


def acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout):
    # Have 3 cases. First case: two accelerometers, second case: three accelerometers, third case: four accelerometers
    match layout:
        case 1:
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]
            # Measured angular rate and acceleration. Should be same in both accelerometers as it is an external measurement
            assert np.allclose(Acc1.w_meas, Acc2.w_meas, 1e-6, 1e-9)
            assert np.allclose(Acc1.dw_meas, Acc2.dw_meas, 1e-6, 1e-9)

            w_meas = Acc1.w_meas
            dw_meas = Acc1.dw_meas

            M = 29  # Number of parameters
            N = len(w_meas)  # Number of observations

            # Only differential mode
            Ax = np.zeros((N, M))  # Design matrix ydx
            Ay = np.zeros((N, M))  # Design matrix ydy
            Az = np.zeros((N, M))  # Design matrix ydz

            # Since we use x0, we get H matrices instead of M matrices where M = I + dM + ddM = H + ddM
            # x0 reflects dM and acc_cal_par_vec_to_mat makes H = I + dM

            H1, H2, K1, K2, W1, W2, dr1, dr2 = acc_cal_par_vec_to_mat(x0, Acc_lst, layout)

            Hc12 = (H1 + H2) / 2
            Hd12 = (H1 - H2) / 2

            Wd12 = (W1 - W2) / 2

            drc12 = (dr1 + dr2) / 2
            drd12 = (dr1 - dr2) / 2

            G1_meas = Acc1.G1_meas
            G2_meas = Acc1.G2_meas
            G3_meas = Acc1.G3_meas

            # Estimate the nominal position acceleration based on the noisy w and dw and lowest biased estimate of a_ng (a_ng_rcst)
            a1_np = G1_meas * Acc1.r[0] + G2_meas * Acc1.r[1] + G3_meas * Acc1.r[2] + a_ng_rcst
            a2_np = G1_meas * Acc2.r[0] + G2_meas * Acc2.r[1] + G3_meas * Acc2.r[2] + a_ng_rcst

            ac12_np = (a1_np + a2_np) / 2
            ad12_np = (a1_np - a2_np) / 2

            ad12_meas = (Acc1.a_meas - Acc2.a_meas) / 2

            idx_Mc12 = np.reshape(np.arange(9), (3, 3)).T.flatten()
            idx_Md12 = np.reshape(np.arange(9, 18), (3, 3)).T.flatten()

            ###########################################################################################################################
            # Differential Mode #######################################################################################################
            ###########################################################################################################################

            yd = ad12_meas  # Observations
            ydx = yd[:, 0]
            ydy = yd[:, 1]
            ydz = yd[:, 2]
            # Estimate at x0
            yd0 = (ad12_np + G1_meas * drd12[0] + G2_meas * drd12[1] + G3_meas * drd12[2]) @ Hc12.T \
                  + (ac12_np + G1_meas * drc12[0] + G2_meas * drc12[1] + G3_meas * drc12[2]) @ Hd12.T \
                  + 0.5 * (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2 @ K1.T \
                  - 0.5 * (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2 @ K2.T \
                  + dw_meas @ Wd12.T

            yd0x = yd0[:, 0]
            yd0y = yd0[:, 1]
            yd0z = yd0[:, 2]

            # Create the design matrix
            #################################
            # 1) M-parameters: ddMc12, ddMd12
            #################################

            diff_Mc12_kron = np.kron(np.identity(3), ad12_np + G1_meas * drd12[0] + G2_meas * drd12[1] + G3_meas * drd12[2])
            diff_Md12_kron = np.kron(np.identity(3), ac12_np + G1_meas * drc12[0] + G2_meas * drc12[1] + G3_meas * drc12[2])

            # dMc12
            Ax[:, idx_Mc12] = diff_Mc12_kron[0:N, :]
            Ay[:, idx_Mc12] = diff_Mc12_kron[N:2 * N, :]
            Az[:, idx_Mc12] = diff_Mc12_kron[2 * N:3 * N, :]

            # dMd12
            Ax[:, idx_Md12] = diff_Md12_kron[0:N, :]
            Ay[:, idx_Md12] = diff_Md12_kron[N:2 * N, :]
            Az[:, idx_Md12] = diff_Md12_kron[2 * N:3 * N, :]

            #################################
            # 2) K-parameters: K1, K2
            #################################
            diff_K1_kron = 0.5 * np.kron(np.identity(3), (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2)
            diff_K2_kron = -0.5 * np.kron(np.identity(3), (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2)

            Ax[:, 18] = diff_K1_kron[0:N, 0]
            Ay[:, 19] = diff_K1_kron[N:2 * N, 4]
            Az[:, 20] = diff_K1_kron[2 * N:3 * N, 8]

            Ax[:, 21] = diff_K2_kron[0:N, 0]
            Ay[:, 22] = diff_K2_kron[N:2 * N, 4]
            Az[:, 23] = diff_K2_kron[2 * N:3 * N, 8]

            #################################
            # 3) W-parameter: Wd12
            #################################
            diff_Wd12_kron = np.kron(np.identity(3), dw_meas)

            # x component has no parameters (A[0:N, 24:27] = 0)
            # y component
            Ay[:, 24] = diff_Wd12_kron[N:2 * N, 3]
            Ay[:, 26] = diff_Wd12_kron[N:2 * N, 5]
            # z component
            Az[:, 25] = diff_Wd12_kron[2 * N:3 * N, 7]

            #################################
            # 4) dr-parameters: drd12. Cannot estimate drc12. Assume it is 0
            #################################
            # Create local variables to avoid recalculation of common terms
            dfddr1x = (a1_np[:, 0] + G1_meas[:, 0] * dr1[0] + G2_meas[:, 0] * dr1[1] + G3_meas[:, 0] * dr1[2]) * K1[0, 0]
            dfddr1y = (a1_np[:, 1] + G1_meas[:, 1] * dr1[0] + G2_meas[:, 1] * dr1[1] + G3_meas[:, 1] * dr1[2]) * K1[1, 1]
            dfddr1z = (a1_np[:, 2] + G1_meas[:, 2] * dr1[0] + G2_meas[:, 2] * dr1[1] + G3_meas[:, 2] * dr1[2]) * K1[2, 2]

            dfddr2x = (a2_np[:, 0] + G1_meas[:, 0] * dr2[0] + G2_meas[:, 0] * dr2[1] + G3_meas[:, 0] * dr2[2]) * K2[0, 0]
            dfddr2y = (a2_np[:, 1] + G1_meas[:, 1] * dr2[0] + G2_meas[:, 1] * dr2[1] + G3_meas[:, 1] * dr2[2]) * K2[1, 1]
            dfddr2z = (a2_np[:, 2] + G1_meas[:, 2] * dr2[0] + G2_meas[:, 2] * dr2[1] + G3_meas[:, 2] * dr2[2]) * K2[2, 2]

            # # i) drc12 with partials
            # Ax[:, 27] = G1_meas @ Hd12[0, :].T + dfddr1x * G1_meas[:, 0] - dfddr2x * G1_meas[:, 0]
            # Ax[:, 28] = G2_meas @ Hd12[0, :].T + dfddr1x * G2_meas[:, 0] - dfddr2x * G2_meas[:, 0]
            # Ax[:, 29] = G3_meas @ Hd12[0, :].T + dfddr1x * G3_meas[:, 0] - dfddr2x * G3_meas[:, 0]
            #
            # Ay[:, 27] = G1_meas @ Hd12[1, :].T + dfddr1y * G1_meas[:, 1] - dfddr2y * G1_meas[:, 1]
            # Ay[:, 28] = G2_meas @ Hd12[1, :].T + dfddr1y * G2_meas[:, 1] - dfddr2y * G2_meas[:, 1]
            # Ay[:, 29] = G3_meas @ Hd12[1, :].T + dfddr1y * G3_meas[:, 1] - dfddr2y * G3_meas[:, 1]
            #
            # Az[:, 27] = G1_meas @ Hd12[2, :].T + dfddr1z * G1_meas[:, 2] - dfddr2z * G1_meas[:, 2]
            # Az[:, 28] = G2_meas @ Hd12[2, :].T + dfddr1z * G2_meas[:, 2] - dfddr2z * G2_meas[:, 2]
            # Az[:, 29] = G3_meas @ Hd12[2, :].T + dfddr1z * G3_meas[:, 2] - dfddr2z * G3_meas[:, 2]

            # ii) drd12 with partials
            if Acc1.r[0] != 0:
                # Cannot estimate drd12x if the accelerometers are placed on x-axis
                Ax[:, 27] = G2_meas @ Hc12[0, :].T + dfddr1x * G2_meas[:, 0] + dfddr2x * G2_meas[:, 0]
                Ax[:, 28] = G3_meas @ Hc12[0, :].T + dfddr1x * G3_meas[:, 0] + dfddr2x * G3_meas[:, 0]

                Ay[:, 27] = G2_meas @ Hc12[1, :].T + dfddr1y * G2_meas[:, 1] + dfddr2y * G2_meas[:, 1]
                Ay[:, 28] = G3_meas @ Hc12[1, :].T + dfddr1y * G3_meas[:, 1] + dfddr2y * G3_meas[:, 1]

                Az[:, 27] = G2_meas @ Hc12[2, :].T + dfddr1z * G2_meas[:, 2] + dfddr2z * G2_meas[:, 2]
                Az[:, 28] = G3_meas @ Hc12[2, :].T + dfddr1z * G3_meas[:, 2] + dfddr2z * G3_meas[:, 2]

            elif Acc1.r[1] != 0:
                # Cannot estimate drd12y if the accelerometers are placed on y-axis
                Ax[:, 27] = G1_meas @ Hc12[0, :].T + dfddr1x * G1_meas[:, 0] + dfddr2x * G1_meas[:, 0]
                Ax[:, 28] = G3_meas @ Hc12[0, :].T + dfddr1x * G3_meas[:, 0] + dfddr2x * G3_meas[:, 0]

                Ay[:, 27] = G1_meas @ Hc12[1, :].T + dfddr1y * G1_meas[:, 1] + dfddr2y * G1_meas[:, 1]
                Ay[:, 28] = G3_meas @ Hc12[1, :].T + dfddr1y * G3_meas[:, 1] + dfddr2y * G3_meas[:, 1]

                Az[:, 27] = G1_meas @ Hc12[2, :].T + dfddr1z * G1_meas[:, 2] + dfddr2z * G1_meas[:, 2]
                Az[:, 28] = G3_meas @ Hc12[2, :].T + dfddr1z * G3_meas[:, 2] + dfddr2z * G3_meas[:, 2]

            else:
                # Cannot estimate drd12z if the accelerometers are placed on z-axis
                Ax[:, 27] = G1_meas @ Hc12[0, :].T + dfddr1x * G1_meas[:, 0] + dfddr2x * G1_meas[:, 0]
                Ax[:, 28] = G2_meas @ Hc12[0, :].T + dfddr1x * G2_meas[:, 0] + dfddr2x * G2_meas[:, 0]

                Ay[:, 27] = G1_meas @ Hc12[1, :].T + dfddr1y * G1_meas[:, 1] + dfddr2y * G1_meas[:, 1]
                Ay[:, 28] = G2_meas @ Hc12[1, :].T + dfddr1y * G2_meas[:, 1] + dfddr2y * G2_meas[:, 1]

                Az[:, 27] = G1_meas @ Hc12[2, :].T + dfddr1z * G1_meas[:, 2] + dfddr2z * G1_meas[:, 2]
                Az[:, 28] = G2_meas @ Hc12[2, :].T + dfddr1z * G2_meas[:, 2] + dfddr2z * G2_meas[:, 2]

            return ydx, ydy, ydz, yd0x, yd0y, yd0z, Ax, Ay, Az

        case 2:
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]
            Acc3 = Acc_lst[2]

            # Measured angular rate and acceleration. Should be same in all 3 accelerometers as it is an external measurement
            assert np.allclose(Acc1.w_meas, Acc2.w_meas, 1e-6, 1e-9)
            assert np.allclose(Acc1.dw_meas, Acc2.dw_meas, 1e-6, 1e-9)
            assert np.allclose(Acc1.G1_meas, Acc3.G1_meas, 1e-6, 1e-9)

            w_meas = Acc1.w_meas
            dw_meas = Acc1.dw_meas

            M = 47  # Number of parameters
            N = len(w_meas)  # Number of observations

            Ax = np.zeros((N, M))  # Design matrix ydx
            Ay = np.zeros((N, M))  # Design matrix ydy
            Az = np.zeros((N, M))  # Design matrix ydz

            Bx = np.zeros((N, M))  # Design matrix ycx
            By = np.zeros((N, M))  # Design matrix ycy
            Bz = np.zeros((N, M))  # Design matrix ycz

            # Since we use x0, we get H matrices instead of M matrices where M = I + dM + ddM = H + ddM
            # x0 reflects dM and acc_cal_par_vec_to_mat makes H = I + dM

            H1, H2, H3, K1, K2, K3, W1, W2, W3, dr1, dr2, dr3 = acc_cal_par_vec_to_mat(x0, Acc_lst, layout)

            Hc13 = (H1 + H3) / 2
            Hd13 = (H1 - H3) / 2
            Hc13_inv = np.linalg.inv(Hc13)
            H2_inv = np.linalg.inv(H2)

            Wc13 = (W1 + W3) / 2
            Wd13 = (W1 - W3) / 2

            drc13 = (dr1 + dr3) / 2
            drd13 = (dr1 - dr3) / 2

            G1_meas = Acc1.G1_meas
            G2_meas = Acc1.G2_meas
            G3_meas = Acc1.G3_meas

            # Estimate the nominal position acceleration based on the noisy w and dw and lowest estimate of a_ng (a_ng_rcst)
            a1_np = G1_meas * Acc1.r[0] + G2_meas * Acc1.r[1] + G3_meas * Acc1.r[2] + a_ng_rcst
            a2_np = a_ng_rcst
            a3_np = G1_meas * Acc3.r[0] + G2_meas * Acc3.r[1] + G3_meas * Acc3.r[2] + a_ng_rcst

            ac13_np = (a1_np + a3_np) / 2
            ad13_np = (a1_np - a3_np) / 2

            ac13_meas = (Acc1.a_meas + Acc3.a_meas) / 2
            ad13_meas = (Acc1.a_meas - Acc3.a_meas) / 2

            a2_meas = Acc2.a_meas

            idx_M2 = np.reshape(np.arange(0, 9), (3, 3)).T.flatten()
            idx_Mc13 = np.reshape(np.arange(9, 18), (3, 3)).T.flatten()
            idx_Md13 = np.reshape(np.arange(18, 27), (3, 3)).T.flatten()
            ###########################################################################################################################
            # Differential Mode #######################################################################################################
            ###########################################################################################################################

            yd = ad13_meas  # Observations
            ydx = yd[:, 0]
            ydy = yd[:, 1]
            ydz = yd[:, 2]
            # Estimate at x0
            yd0 = (ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2]) @ Hc13.T \
                  + (ac13_np + G1_meas * drc13[0] + G2_meas * drc13[1] + G3_meas * drc13[2]) @ Hd13.T \
                  + 0.5 * (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2 @ K1.T \
                  - 0.5 * (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2 @ K3.T \
                  + dw_meas @ Wd13.T

            yd0x = yd0[:, 0]
            yd0y = yd0[:, 1]
            yd0z = yd0[:, 2]

            # Create the design matrix
            #################################
            # 1) M-parameters: ddMc13, ddMd13
            #################################

            diff_Mc13_kron = np.kron(np.identity(3), ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2])
            diff_Md13_kron = np.kron(np.identity(3), ac13_np + G1_meas * drc13[0] + G2_meas * drc13[1] + G3_meas * drc13[2])

            # dMc13
            Ax[:, idx_Mc13] = diff_Mc13_kron[0:N, :]
            Ay[:, idx_Mc13] = diff_Mc13_kron[N:2 * N, :]
            Az[:, idx_Mc13] = diff_Mc13_kron[2 * N:3 * N, :]

            # dMd13
            Ax[:, idx_Md13] = diff_Md13_kron[0:N, :]
            Ay[:, idx_Md13] = diff_Md13_kron[N:2 * N, :]
            Az[:, idx_Md13] = diff_Md13_kron[2 * N:3 * N, :]

            #################################
            # 2) K-parameters: K1, K3
            #################################
            diff_K1_kron = 0.5 * np.kron(np.identity(3), (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2)
            diff_K3_kron = -0.5 * np.kron(np.identity(3), (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2)

            Ax[:, 27] = diff_K1_kron[0:N, 0]
            Ay[:, 28] = diff_K1_kron[N:2 * N, 4]
            Az[:, 29] = diff_K1_kron[2 * N:3 * N, 8]

            Ax[:, 33] = diff_K3_kron[0:N, 0]
            Ay[:, 34] = diff_K3_kron[N:2 * N, 4]
            Az[:, 35] = diff_K3_kron[2 * N:3 * N, 8]

            #################################
            # 3) W-parameter: Wd13
            #################################
            diff_Wd13_kron = np.kron(np.identity(3), dw_meas)

            # x component has no parameters (A[0:N, 48:51] = 0)
            # y component
            Ay[:, 36] = diff_Wd13_kron[N:2 * N, 3]
            Ay[:, 38] = diff_Wd13_kron[N:2 * N, 5]
            # z component
            Az[:, 37] = diff_Wd13_kron[2 * N:3 * N, 7]

            #################################
            # 4) dr-parameters: drc13, drd13
            #################################
            # Create local variables to avoid recalculation of common terms
            dfddr1x = (a1_np[:, 0] + G1_meas[:, 0] * dr1[0] + G2_meas[:, 0] * dr1[1] + G3_meas[:, 0] * dr1[2]) * K1[0, 0]
            dfddr1y = (a1_np[:, 1] + G1_meas[:, 1] * dr1[0] + G2_meas[:, 1] * dr1[1] + G3_meas[:, 1] * dr1[2]) * K1[1, 1]
            dfddr1z = (a1_np[:, 2] + G1_meas[:, 2] * dr1[0] + G2_meas[:, 2] * dr1[1] + G3_meas[:, 2] * dr1[2]) * K1[2, 2]

            dfddr3x = (a3_np[:, 0] + G1_meas[:, 0] * dr3[0] + G2_meas[:, 0] * dr3[1] + G3_meas[:, 0] * dr3[2]) * K3[0, 0]
            dfddr3y = (a3_np[:, 1] + G1_meas[:, 1] * dr3[0] + G2_meas[:, 1] * dr3[1] + G3_meas[:, 1] * dr3[2]) * K3[1, 1]
            dfddr3z = (a3_np[:, 2] + G1_meas[:, 2] * dr3[0] + G2_meas[:, 2] * dr3[1] + G3_meas[:, 2] * dr3[2]) * K3[2, 2]

            # i) drc13 with partials due to quadratic terms
            Ax[:, 42] = G1_meas @ Hd13[0, :].T + dfddr1x * G1_meas[:, 0] - dfddr3x * G1_meas[:, 0]
            Ax[:, 43] = G2_meas @ Hd13[0, :].T + dfddr1x * G2_meas[:, 0] - dfddr3x * G2_meas[:, 0]
            Ax[:, 44] = G3_meas @ Hd13[0, :].T + dfddr1x * G3_meas[:, 0] - dfddr3x * G3_meas[:, 0]

            Ay[:, 42] = G1_meas @ Hd13[1, :].T + dfddr1y * G1_meas[:, 1] - dfddr3y * G1_meas[:, 1]
            Ay[:, 43] = G2_meas @ Hd13[1, :].T + dfddr1y * G2_meas[:, 1] - dfddr3y * G2_meas[:, 1]
            Ay[:, 44] = G3_meas @ Hd13[1, :].T + dfddr1y * G3_meas[:, 1] - dfddr3y * G3_meas[:, 1]

            Az[:, 42] = G1_meas @ Hd13[2, :].T + dfddr1z * G1_meas[:, 2] - dfddr3z * G1_meas[:, 2]
            Az[:, 43] = G2_meas @ Hd13[2, :].T + dfddr1z * G2_meas[:, 2] - dfddr3z * G2_meas[:, 2]
            Az[:, 44] = G3_meas @ Hd13[2, :].T + dfddr1z * G3_meas[:, 2] - dfddr3z * G3_meas[:, 2]

            # ii) drd13 with partials due to quadratic terms
            if Acc1.r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                Ax[:, 45] = G2_meas @ Hc13[0, :].T + dfddr1x * G2_meas[:, 0] + dfddr3x * G2_meas[:, 0]
                Ax[:, 46] = G3_meas @ Hc13[0, :].T + dfddr1x * G3_meas[:, 0] + dfddr3x * G3_meas[:, 0]

                Ay[:, 45] = G2_meas @ Hc13[1, :].T + dfddr1y * G2_meas[:, 1] + dfddr3y * G2_meas[:, 1]
                Ay[:, 46] = G3_meas @ Hc13[1, :].T + dfddr1y * G3_meas[:, 1] + dfddr3y * G3_meas[:, 1]

                Az[:, 45] = G2_meas @ Hc13[2, :].T + dfddr1z * G2_meas[:, 2] + dfddr3z * G2_meas[:, 2]
                Az[:, 46] = G3_meas @ Hc13[2, :].T + dfddr1z * G3_meas[:, 2] + dfddr3z * G3_meas[:, 2]

            elif Acc1.r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                Ax[:, 45] = G1_meas @ Hc13[0, :].T + dfddr1x * G1_meas[:, 0] + dfddr3x * G1_meas[:, 0]
                Ax[:, 46] = G3_meas @ Hc13[0, :].T + dfddr1x * G3_meas[:, 0] + dfddr3x * G3_meas[:, 0]

                Ay[:, 45] = G1_meas @ Hc13[1, :].T + dfddr1y * G1_meas[:, 1] + dfddr3y * G1_meas[:, 1]
                Ay[:, 46] = G3_meas @ Hc13[1, :].T + dfddr1y * G3_meas[:, 1] + dfddr3y * G3_meas[:, 1]

                Az[:, 45] = G1_meas @ Hc13[2, :].T + dfddr1z * G1_meas[:, 2] + dfddr3z * G1_meas[:, 2]
                Az[:, 46] = G3_meas @ Hc13[2, :].T + dfddr1z * G3_meas[:, 2] + dfddr3z * G3_meas[:, 2]

            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                Ax[:, 45] = G1_meas @ Hc13[0, :].T + dfddr1x * G1_meas[:, 0] + dfddr3x * G1_meas[:, 0]
                Ax[:, 46] = G2_meas @ Hc13[0, :].T + dfddr1x * G2_meas[:, 0] + dfddr3x * G2_meas[:, 0]

                Ay[:, 45] = G1_meas @ Hc13[1, :].T + dfddr1y * G1_meas[:, 1] + dfddr3y * G1_meas[:, 1]
                Ay[:, 46] = G2_meas @ Hc13[1, :].T + dfddr1y * G2_meas[:, 1] + dfddr3y * G2_meas[:, 1]

                Az[:, 45] = G1_meas @ Hc13[2, :].T + dfddr1z * G1_meas[:, 2] + dfddr3z * G1_meas[:, 2]
                Az[:, 46] = G2_meas @ Hc13[2, :].T + dfddr1z * G2_meas[:, 2] + dfddr3z * G2_meas[:, 2]

            ###########################################################################################################################
            # Common Mode #############################################################################################################
            ###########################################################################################################################
            ycx = np.zeros(N)
            ycy = np.zeros(N)
            ycz = np.zeros(N)

            yc0 = (ac13_meas - (ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2]) @ Hd13.T -
                   0.5 * (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2 @ K1.T -
                   0.5 * (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2 @ K3.T - dw_meas @ Wc13.T) @ Hc13_inv.T - \
                  (a2_meas - (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2 @ K2.T - dw_meas @ W2.T) @ H2_inv.T - \
                  (G1_meas * (drc13[0] - dr2[0])) - (G2_meas * (drc13[1] - dr2[1])) - (G3_meas * (drc13[2] - dr2[2]))

            yc0x = yc0[:, 0]
            yc0y = yc0[:, 1]
            yc0z = yc0[:, 2]

            # Create the design matrix
            #######################################
            # 1) M-parameters: ddM2, ddMc13, ddMd13
            #######################################
            # dM2
            com_dM2_kron = np.kron(H2_inv, (a2_meas - (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2 @ K2.T
                                            - dw_meas @ W2.T) @ H2_inv.T)

            Bx[:, idx_M2] = com_dM2_kron[0:N, :]
            By[:, idx_M2] = com_dM2_kron[N:2 * N, :]
            Bz[:, idx_M2] = com_dM2_kron[2 * N:3 * N, :]

            # dMc13
            com_dMc13_kron = -np.kron(Hc13_inv, (ac13_meas - (ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2]) @ Hd13.T -
                                                 0.5 * (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2 @ K1.T -
                                                 0.5 * (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2 @ K3.T -
                                                 dw_meas @ Wc13.T) @ Hc13_inv.T)

            Bx[:, idx_Mc13] = com_dMc13_kron[0:N, :]
            By[:, idx_Mc13] = com_dMc13_kron[N:2 * N, :]
            Bz[:, idx_Mc13] = com_dMc13_kron[2 * N:3 * N, :]

            # dMd13
            com_dMd13_kron = -np.kron(Hc13_inv, ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2])

            Bx[:, idx_Md13] = com_dMd13_kron[0:N, :]
            By[:, idx_Md13] = com_dMd13_kron[N:2 * N, :]
            Bz[:, idx_Md13] = com_dMd13_kron[2 * N:3 * N, :]

            #################################
            # 2) K-parameters: K1, K2, K3
            #################################
            # K1
            com_K1_kron = -0.5 * np.kron(Hc13_inv, (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2)
            Bx[:, 27] = com_K1_kron[0:N, 0]
            Bx[:, 28] = com_K1_kron[0:N, 4]
            Bx[:, 29] = com_K1_kron[0:N, 8]

            By[:, 27] = com_K1_kron[N:2 * N, 0]
            By[:, 28] = com_K1_kron[N:2 * N, 4]
            By[:, 29] = com_K1_kron[N:2 * N, 8]

            Bz[:, 27] = com_K1_kron[2 * N:3 * N, 0]
            Bz[:, 28] = com_K1_kron[2 * N:3 * N, 4]
            Bz[:, 29] = com_K1_kron[2 * N:3 * N, 8]

            # K2
            com_K2_kron = np.kron(H2_inv, (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2)
            Bx[:, 30] = com_K2_kron[0:N, 0]
            Bx[:, 31] = com_K2_kron[0:N, 4]
            Bx[:, 32] = com_K2_kron[0:N, 8]

            By[:, 30] = com_K2_kron[N:2 * N, 0]
            By[:, 31] = com_K2_kron[N:2 * N, 4]
            By[:, 32] = com_K2_kron[N:2 * N, 8]

            Bz[:, 30] = com_K2_kron[2 * N:3 * N, 0]
            Bz[:, 31] = com_K2_kron[2 * N:3 * N, 4]
            Bz[:, 32] = com_K2_kron[2 * N:3 * N, 8]

            # K3
            com_K3_kron = -0.5 * np.kron(Hc13_inv, (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2)

            Bx[:, 33] = com_K3_kron[0:N, 0]
            Bx[:, 34] = com_K3_kron[0:N, 4]
            Bx[:, 35] = com_K3_kron[0:N, 8]

            By[:, 33] = com_K3_kron[N:2 * N, 0]
            By[:, 34] = com_K3_kron[N:2 * N, 4]
            By[:, 35] = com_K3_kron[N:2 * N, 8]

            Bz[:, 33] = com_K3_kron[2 * N:3 * N, 0]
            Bz[:, 34] = com_K3_kron[2 * N:3 * N, 4]
            Bz[:, 35] = com_K3_kron[2 * N:3 * N, 8]

            #################################
            # 3) W-parameter: Wc13, W2
            #################################
            # Since W2 is defined to be zero, we only need to calculate Wc13
            com_Wc13_kron = np.kron(Hc13_inv, dw_meas)

            Bx[:, 39] = com_Wc13_kron[0:N, 3]
            Bx[:, 40] = com_Wc13_kron[0:N, 7]
            Bx[:, 41] = com_Wc13_kron[0:N, 5]

            By[:, 39] = com_Wc13_kron[N:2 * N, 3]
            By[:, 40] = com_Wc13_kron[N:2 * N, 7]
            By[:, 41] = com_Wc13_kron[N:2 * N, 5]

            Bz[:, 39] = com_Wc13_kron[2 * N:3 * N, 3]
            Bz[:, 40] = com_Wc13_kron[2 * N:3 * N, 7]
            Bz[:, 41] = com_Wc13_kron[2 * N:3 * N, 5]

            #################################
            # 4) dr-parameters: drc13, drd13
            #################################
            # i) drc13 with partials due to quadratic terms
            # Create local variables to avoid recalculations of common terms

            dfcdr1_temp = (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) @ K1.T
            dfcdr3_temp = (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) @ K3.T

            dfcdr1x = G1_meas * dfcdr1_temp
            dfcdr1y = G2_meas * dfcdr1_temp
            dfcdr1z = G3_meas * dfcdr1_temp

            dfcdr3x = G1_meas * dfcdr3_temp
            dfcdr3y = G2_meas * dfcdr3_temp
            dfcdr3z = G3_meas * dfcdr3_temp

            # i) drc13 with partials due to quadratic terms
            Bx[:, 42] = -G1_meas[:, 0] - dfcdr1x @ Hc13_inv[0, :].T - dfcdr3x @ Hc13_inv[0, :].T
            Bx[:, 43] = -G2_meas[:, 0] - dfcdr1y @ Hc13_inv[0, :].T - dfcdr3y @ Hc13_inv[0, :].T
            Bx[:, 44] = -G3_meas[:, 0] - dfcdr1z @ Hc13_inv[0, :].T - dfcdr3z @ Hc13_inv[0, :].T

            By[:, 42] = -G1_meas[:, 1] - dfcdr1x @ Hc13_inv[1, :].T - dfcdr3x @ Hc13_inv[1, :].T
            By[:, 43] = -G2_meas[:, 1] - dfcdr1y @ Hc13_inv[1, :].T - dfcdr3y @ Hc13_inv[1, :].T
            By[:, 44] = -G3_meas[:, 1] - dfcdr1z @ Hc13_inv[1, :].T - dfcdr3z @ Hc13_inv[1, :].T

            Bz[:, 42] = -G1_meas[:, 2] - dfcdr1x @ Hc13_inv[2, :].T - dfcdr3x @ Hc13_inv[2, :].T
            Bz[:, 43] = -G2_meas[:, 2] - dfcdr1y @ Hc13_inv[2, :].T - dfcdr3y @ Hc13_inv[2, :].T
            Bz[:, 44] = -G3_meas[:, 2] - dfcdr1z @ Hc13_inv[2, :].T - dfcdr3z @ Hc13_inv[2, :].T

            # ii) drd13 with partials due to quadratic terms
            if Acc1.r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                Bx[:, 45] = -G2_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1y @ Hc13_inv[0, :].T + dfcdr3y @ Hc13_inv[0, :].T
                Bx[:, 46] = -G3_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1z @ Hc13_inv[0, :].T + dfcdr3z @ Hc13_inv[0, :].T

                By[:, 45] = -G2_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1y @ Hc13_inv[1, :].T + dfcdr3y @ Hc13_inv[1, :].T
                By[:, 46] = -G3_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1z @ Hc13_inv[1, :].T + dfcdr3z @ Hc13_inv[1, :].T

                Bz[:, 45] = -G2_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1y @ Hc13_inv[2, :].T + dfcdr3y @ Hc13_inv[2, :].T
                Bz[:, 46] = -G3_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1z @ Hc13_inv[2, :].T + dfcdr3z @ Hc13_inv[2, :].T

            elif Acc1.r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                Bx[:, 45] = -G1_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1x @ Hc13_inv[0, :].T + dfcdr3x @ Hc13_inv[0, :].T
                Bx[:, 46] = -G3_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1z @ Hc13_inv[0, :].T + dfcdr3z @ Hc13_inv[0, :].T

                By[:, 45] = -G1_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1x @ Hc13_inv[1, :].T + dfcdr3x @ Hc13_inv[1, :].T
                By[:, 46] = -G3_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1z @ Hc13_inv[1, :].T + dfcdr3z @ Hc13_inv[1, :].T

                Bz[:, 45] = -G1_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1x @ Hc13_inv[2, :].T + dfcdr3x @ Hc13_inv[2, :].T
                Bz[:, 46] = -G3_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1z @ Hc13_inv[2, :].T + dfcdr3z @ Hc13_inv[2, :].T

            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                Bx[:, 45] = -G1_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1x @ Hc13_inv[0, :].T + dfcdr3x @ Hc13_inv[0, :].T
                Bx[:, 46] = -G2_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1y @ Hc13_inv[0, :].T + dfcdr3y @ Hc13_inv[0, :].T

                By[:, 45] = -G1_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1x @ Hc13_inv[1, :].T + dfcdr3x @ Hc13_inv[1, :].T
                By[:, 46] = -G2_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1y @ Hc13_inv[1, :].T + dfcdr3y @ Hc13_inv[1, :].T

                Bz[:, 45] = -G1_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1x @ Hc13_inv[2, :].T + dfcdr3x @ Hc13_inv[2, :].T
                Bz[:, 46] = -G2_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1y @ Hc13_inv[2, :].T + dfcdr3y @ Hc13_inv[2, :].T

            return ydx, ydy, ydz, ycx, ycy, ycz, yd0x, yd0y, yd0z, yc0x, yc0y, yc0z, Ax, Ay, Az, Bx, By, Bz

        case 3:
            # Four accelerometers
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]
            Acc3 = Acc_lst[2]
            Acc4 = Acc_lst[3]

            # Measured angular rate and acceleration. Should be same in all 4 accelerometers as it is an external measurement
            assert np.allclose(Acc1.dw_meas, Acc2.dw_meas, 1e-6, 1e-9)
            assert np.allclose(Acc1.dw_meas, Acc3.dw_meas, 1e-6, 1e-9)
            assert np.allclose(Acc1.dw_meas, Acc4.dw_meas, 1e-6, 1e-9)

            w_meas = Acc1.dw_meas
            dw_meas = Acc1.dw_meas

            M = 64  # Number of parameters
            N = len(w_meas)  # Number of observations

            # Initialize the design matrix
            Ax = np.zeros((N, M))  # Design matrix for yd13x
            Ay = np.zeros((N, M))  # Design matrix for yd13y
            Az = np.zeros((N, M))  # Design matrix for yd13z

            Bx = np.zeros((N, M))  # Design matrix for yd24x
            By = np.zeros((N, M))  # Design matrix for yd24y
            Bz = np.zeros((N, M))  # Design matrix for yd24z

            Cx = np.zeros((N, M))  # Design matrix for ycx
            Cy = np.zeros((N, M))  # Design matrix for ycy
            Cz = np.zeros((N, M))  # Design matrix for ycz

            # Since we use x0, we get H matrices instead of M matrices where M = I + dM + ddM = H + ddM
            # x0 reflects dM and acc_cal_par_vec_to_mat makes H = I + dM

            H1, H2, H3, H4, K1, K2, K3, K4, W1, W2, W3, W4, dr1, dr2, dr3, dr4 = acc_cal_par_vec_to_mat(x0, Acc_lst, layout)

            Hc13 = (H1 + H3) / 2
            Hc24 = (H2 + H4) / 2
            Hd13 = (H1 - H3) / 2
            Hd24 = (H2 - H4) / 2

            Hc13_inv = np.linalg.inv(Hc13)
            Hc24_inv = np.linalg.inv(Hc24)

            Wc13 = (W1 + W3) / 2
            Wc24 = (W2 + W4) / 2
            Wd13 = (W1 - W3) / 2
            Wd24 = (W2 - W4) / 2

            drc13 = (dr1 + dr3) / 2
            drc24 = (dr2 + dr4) / 2
            drd13 = (dr1 - dr3) / 2
            drd24 = (dr2 - dr4) / 2

            G1_meas = Acc1.G1_meas
            G2_meas = Acc1.G2_meas
            G3_meas = Acc1.G3_meas

            # Estimate the nominal position acceleration based on the noisy w, dw and lowest biased estimate of a_ng (a_ng_rcst)
            a1_np = G1_meas * Acc1.r[0] + G2_meas * Acc1.r[1] + G3_meas * Acc1.r[2] + a_ng_rcst
            a2_np = G1_meas * Acc2.r[0] + G2_meas * Acc2.r[1] + G3_meas * Acc2.r[2] + a_ng_rcst
            a3_np = G1_meas * Acc3.r[0] + G2_meas * Acc3.r[1] + G3_meas * Acc3.r[2] + a_ng_rcst
            a4_np = G1_meas * Acc4.r[0] + G2_meas * Acc4.r[1] + G3_meas * Acc4.r[2] + a_ng_rcst

            ac13_np = (a1_np + a3_np) / 2
            ac24_np = (a2_np + a4_np) / 2
            ad13_np = (a1_np - a3_np) / 2
            ad24_np = (a2_np - a4_np) / 2

            ac13_meas = (Acc1.a_meas + Acc3.a_meas) / 2
            ac24_meas = (Acc2.a_meas + Acc4.a_meas) / 2
            ad13_meas = (Acc1.a_meas - Acc3.a_meas) / 2
            ad24_meas = (Acc2.a_meas - Acc4.a_meas) / 2

            idx_Mc13 = np.reshape(np.arange(0, 9), (3, 3)).T.flatten()
            idx_Mc24 = np.reshape(np.arange(9, 18), (3, 3)).T.flatten()
            idx_Md13 = np.reshape(np.arange(18, 27), (3, 3)).T.flatten()
            idx_Md24 = np.reshape(np.arange(27, 36), (3, 3)).T.flatten()

            ###########################################################################################################################
            # Differential Mode ########################################################################################################
            ###########################################################################################################################

            #######################
            # 1) ad13
            #######################
            yd13 = ad13_meas  # Observations
            yd13x = yd13[:, 0]
            yd13y = yd13[:, 1]
            yd13z = yd13[:, 2]
            # Estimate at x0
            yd013 = (ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2]) @ Hc13.T + \
                    (ac13_np + G1_meas * drc13[0] + G2_meas * drc13[1] + G3_meas * drc13[2]) @ Hd13.T + \
                    0.5 * (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2 @ K1.T - \
                    0.5 * (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2 @ K3.T + \
                    dw_meas @ Wd13.T

            yd013x = yd013[:, 0]
            yd013y = yd013[:, 1]
            yd013z = yd013[:, 2]

            # Create the design matrix
            #################################
            # 1) M-parameters: ddMc13, ddMd13
            #################################

            diff_Mc13_kron = np.kron(np.identity(3), ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2])
            diff_Md13_kron = np.kron(np.identity(3), ac13_np + G1_meas * drc13[0] + G2_meas * drc13[1] + G3_meas * drc13[2])

            # dMc13
            Ax[:, idx_Mc13] = diff_Mc13_kron[0:N, :]
            Ay[:, idx_Mc13] = diff_Mc13_kron[N:2 * N, :]
            Az[:, idx_Mc13] = diff_Mc13_kron[2 * N:3 * N, :]

            # dMd13
            Ax[:, idx_Md13] = diff_Md13_kron[0:N, :]
            Ay[:, idx_Md13] = diff_Md13_kron[N:2 * N, :]
            Az[:, idx_Md13] = diff_Md13_kron[2 * N:3 * N, :]

            #################################
            # 2) K-parameters: K1, K3
            #################################
            diff_K1_kron = 0.5 * np.kron(np.identity(3), (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2)
            diff_K3_kron = -0.5 * np.kron(np.identity(3), (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2)

            Ax[:, 36] = diff_K1_kron[0:N, 0]
            Ay[:, 37] = diff_K1_kron[N:2 * N, 4]
            Az[:, 38] = diff_K1_kron[2 * N:3 * N, 8]

            Ax[:, 42] = diff_K3_kron[0:N, 0]
            Ay[:, 43] = diff_K3_kron[N:2 * N, 4]
            Az[:, 44] = diff_K3_kron[2 * N:3 * N, 8]

            #################################
            # 3) W-parameter: Wd13
            #################################
            diff_Wd13_kron = np.kron(np.identity(3), dw_meas)

            # x component has no parameters (A[0:N, 45:48] = 0)
            # y component
            Ay[:, 48] = diff_Wd13_kron[N:2 * N, 3]
            Ay[:, 50] = diff_Wd13_kron[N:2 * N, 5]
            # z component
            Az[:, 49] = diff_Wd13_kron[2 * N:3 * N, 7]

            #################################
            # 4) dr-parameters: drc13, drd13
            #################################
            # Create local variables to avoid recalculation of common terms
            dfddr1x = (a1_np[:, 0] + G1_meas[:, 0] * dr1[0] + G2_meas[:, 0] * dr1[1] + G3_meas[:, 0] * dr1[2]) * K1[0, 0]
            dfddr1y = (a1_np[:, 1] + G1_meas[:, 1] * dr1[0] + G2_meas[:, 1] * dr1[1] + G3_meas[:, 1] * dr1[2]) * K1[1, 1]
            dfddr1z = (a1_np[:, 2] + G1_meas[:, 2] * dr1[0] + G2_meas[:, 2] * dr1[1] + G3_meas[:, 2] * dr1[2]) * K1[2, 2]

            dfddr3x = (a3_np[:, 0] + G1_meas[:, 0] * dr3[0] + G2_meas[:, 0] * dr3[1] + G3_meas[:, 0] * dr3[2]) * K3[0, 0]
            dfddr3y = (a3_np[:, 1] + G1_meas[:, 1] * dr3[0] + G2_meas[:, 1] * dr3[1] + G3_meas[:, 1] * dr3[2]) * K3[1, 1]
            dfddr3z = (a3_np[:, 2] + G1_meas[:, 2] * dr3[0] + G2_meas[:, 2] * dr3[1] + G3_meas[:, 2] * dr3[2]) * K3[2, 2]

            # i) drc13 with partials due to quadratic terms
            Ax[:, 57] = G1_meas @ Hd13[0, :].T + dfddr1x * G1_meas[:, 0] - dfddr3x * G1_meas[:, 0]
            Ax[:, 58] = G2_meas @ Hd13[0, :].T + dfddr1x * G2_meas[:, 0] - dfddr3x * G2_meas[:, 0]
            Ax[:, 59] = G3_meas @ Hd13[0, :].T + dfddr1x * G3_meas[:, 0] - dfddr3x * G3_meas[:, 0]

            Ay[:, 57] = G1_meas @ Hd13[1, :].T + dfddr1y * G1_meas[:, 1] - dfddr3y * G1_meas[:, 1]
            Ay[:, 58] = G2_meas @ Hd13[1, :].T + dfddr1y * G2_meas[:, 1] - dfddr3y * G2_meas[:, 1]
            Ay[:, 59] = G3_meas @ Hd13[1, :].T + dfddr1y * G3_meas[:, 1] - dfddr3y * G3_meas[:, 1]

            Az[:, 57] = G1_meas @ Hd13[2, :].T + dfddr1z * G1_meas[:, 2] - dfddr3z * G1_meas[:, 2]
            Az[:, 58] = G2_meas @ Hd13[2, :].T + dfddr1z * G2_meas[:, 2] - dfddr3z * G2_meas[:, 2]
            Az[:, 59] = G3_meas @ Hd13[2, :].T + dfddr1z * G3_meas[:, 2] - dfddr3z * G3_meas[:, 2]

            # ii) drd13 with partials due to quadratic terms
            if Acc1.r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                Ax[:, 60] = G2_meas @ Hc13[0, :].T + dfddr1x * G2_meas[:, 0] + dfddr3x * G2_meas[:, 0]
                Ax[:, 61] = G3_meas @ Hc13[0, :].T + dfddr1x * G3_meas[:, 0] + dfddr3x * G3_meas[:, 0]

                Ay[:, 60] = G2_meas @ Hc13[1, :].T + dfddr1y * G2_meas[:, 1] + dfddr3y * G2_meas[:, 1]
                Ay[:, 61] = G3_meas @ Hc13[1, :].T + dfddr1y * G3_meas[:, 1] + dfddr3y * G3_meas[:, 1]

                Az[:, 60] = G2_meas @ Hc13[2, :].T + dfddr1z * G2_meas[:, 2] + dfddr3z * G2_meas[:, 2]
                Az[:, 61] = G3_meas @ Hc13[2, :].T + dfddr1z * G3_meas[:, 2] + dfddr3z * G3_meas[:, 2]

            elif Acc1.r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                Ax[:, 60] = G1_meas @ Hc13[0, :].T + dfddr1x * G1_meas[:, 0] + dfddr3x * G1_meas[:, 0]
                Ax[:, 61] = G3_meas @ Hc13[0, :].T + dfddr1x * G3_meas[:, 0] + dfddr3x * G3_meas[:, 0]

                Ay[:, 60] = G1_meas @ Hc13[1, :].T + dfddr1y * G1_meas[:, 1] + dfddr3y * G1_meas[:, 1]
                Ay[:, 61] = G3_meas @ Hc13[1, :].T + dfddr1y * G3_meas[:, 1] + dfddr3y * G3_meas[:, 1]

                Az[:, 60] = G1_meas @ Hc13[2, :].T + dfddr1z * G1_meas[:, 2] + dfddr3z * G1_meas[:, 2]
                Az[:, 61] = G3_meas @ Hc13[2, :].T + dfddr1z * G3_meas[:, 2] + dfddr3z * G3_meas[:, 2]

            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                Ax[:, 60] = G1_meas @ Hc13[0, :].T + dfddr1x * G1_meas[:, 0] + dfddr3x * G1_meas[:, 0]
                Ax[:, 61] = G2_meas @ Hc13[0, :].T + dfddr1x * G2_meas[:, 0] + dfddr3x * G2_meas[:, 0]

                Ay[:, 60] = G1_meas @ Hc13[1, :].T + dfddr1y * G1_meas[:, 1] + dfddr3y * G1_meas[:, 1]
                Ay[:, 61] = G2_meas @ Hc13[1, :].T + dfddr1y * G2_meas[:, 1] + dfddr3y * G2_meas[:, 1]

                Az[:, 60] = G1_meas @ Hc13[2, :].T + dfddr1z * G1_meas[:, 2] + dfddr3z * G1_meas[:, 2]
                Az[:, 61] = G2_meas @ Hc13[2, :].T + dfddr1z * G2_meas[:, 2] + dfddr3z * G2_meas[:, 2]

            #######################
            # 2) ad24
            #######################
            yd24 = ad24_meas  # Observations
            yd24x = yd24[:, 0]
            yd24y = yd24[:, 1]
            yd24z = yd24[:, 2]
            # Estimate at x0
            yd024 = (ad24_np + G1_meas * drd24[0] + G2_meas * drd24[1] + G3_meas * drd24[2]) @ Hc24.T + \
                                (ac24_np + G1_meas * drc24[0] + G2_meas * drc24[1] + G3_meas * drc24[2]) @ Hd24.T + \
            0.5 * (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2 @ K2.T - \
            0.5 * (a4_np + G1_meas * dr4[0] + G2_meas * dr4[1] + G3_meas * dr4[2]) ** 2 @ K4.T + \
            dw_meas @ Wd24.T

            yd024x = yd024[:, 0]
            yd024y = yd024[:, 1]
            yd024z = yd024[:, 2]

            # Create the design matrix
            #################################
            # 1) M-parameters: ddMc24, ddMd24
            #################################

            diff_Mc24_kron = np.kron(np.identity(3), ad24_np + G1_meas * drd24[0] + G2_meas * drd24[1] + G3_meas * drd24[2])
            diff_Md24_kron = np.kron(np.identity(3), ac24_np + G1_meas * drc24[0] + G2_meas * drc24[1] + G3_meas * drc24[2])

            # dMc24
            Bx[:, idx_Mc24] = diff_Mc24_kron[0:N, :]
            By[:, idx_Mc24] = diff_Mc24_kron[N:2 * N, :]
            Bz[:, idx_Mc24] = diff_Mc24_kron[2 * N:3 * N, :]

            # dMd24
            Bx[:, idx_Md24] = diff_Md24_kron[0:N, :]
            By[:, idx_Md24] = diff_Md24_kron[N:2 * N, :]
            Bz[:, idx_Md24] = diff_Md24_kron[2 * N:3 * N, :]

            #################################
            # 2) K-parameters: K2, K4
            #################################
            diff_K2_kron = 0.5 * np.kron(np.identity(3), (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2)
            diff_K4_kron = -0.5 * np.kron(np.identity(3), (a4_np + G1_meas * dr4[0] + G2_meas * dr4[1] + G3_meas * dr4[2]) ** 2)

            Bx[:, 39] = diff_K2_kron[0:N, 0]
            By[:, 40] = diff_K2_kron[N:2 * N, 4]
            Bz[:, 41] = diff_K2_kron[2 * N:3 * N, 8]

            Bx[:, 45] = diff_K4_kron[0:N, 0]
            By[:, 46] = diff_K4_kron[N:2 * N, 4]
            Bz[:, 47] = diff_K4_kron[2 * N:3 * N, 8]

            #################################
            # 3) W-parameter: Wd24
            #################################
            diff_Wd24_kron = np.kron(np.identity(3), dw_meas)
            # x component has no parameters (B[0:N, 51:54] = 0)
            # y component
            By[:, 51] = diff_Wd24_kron[N:2 * N, 3]
            By[:, 53] = diff_Wd24_kron[N:2 * N, 5]
            # z component
            Bz[:, 52] = diff_Wd24_kron[2 * N:3 * N, 7]

            #################################
            # 4) dr-parameters: drd24. Cannot estimate drc24. Assume it is zero
            #################################
            # Create local variables to avoid recalculation of common terms
            dfddr2x = (a2_np[:, 0] + G1_meas[:, 0] * dr2[0] + G2_meas[:, 0] * dr2[1] + G3_meas[:, 0] * dr2[2]) * K2[0, 0]
            dfddr2y = (a2_np[:, 1] + G1_meas[:, 1] * dr2[0] + G2_meas[:, 1] * dr2[1] + G3_meas[:, 1] * dr2[2]) * K2[1, 1]
            dfddr2z = (a2_np[:, 2] + G1_meas[:, 2] * dr2[0] + G2_meas[:, 2] * dr2[1] + G3_meas[:, 2] * dr2[2]) * K2[2, 2]

            dfddr4x = (a4_np[:, 0] + G1_meas[:, 0] * dr4[0] + G2_meas[:, 0] * dr4[1] + G3_meas[:, 0] * dr4[2]) * K4[0, 0]
            dfddr4y = (a4_np[:, 1] + G1_meas[:, 1] * dr4[0] + G2_meas[:, 1] * dr4[1] + G3_meas[:, 1] * dr4[2]) * K4[1, 1]
            dfddr4z = (a4_np[:, 2] + G1_meas[:, 2] * dr4[0] + G2_meas[:, 2] * dr4[1] + G3_meas[:, 2] * dr4[2]) * K4[2, 2]

            # # i) drc24 with partials due to quadratic terms
            # Bx[:, 60] = G1_meas @ Hd24[0, :].T + dfddr2x * G1_meas[:, 0] - dfddr4x * G1_meas[:, 0]
            # Bx[:, 61] = G2_meas @ Hd24[0, :].T + dfddr2x * G2_meas[:, 0] - dfddr4x * G2_meas[:, 0]
            # Bx[:, 62] = G3_meas @ Hd24[0, :].T + dfddr2x * G3_meas[:, 0] - dfddr4x * G3_meas[:, 0]
            #
            # By[:, 60] = G1_meas @ Hd24[1, :].T + dfddr2y * G1_meas[:, 1] - dfddr4y * G1_meas[:, 1]
            # By[:, 61] = G2_meas @ Hd24[1, :].T + dfddr2y * G2_meas[:, 1] - dfddr4y * G2_meas[:, 1]
            # By[:, 62] = G3_meas @ Hd24[1, :].T + dfddr2y * G3_meas[:, 1] - dfddr4y * G3_meas[:, 1]
            #
            # Bz[:, 60] = G1_meas @ Hd24[2, :].T + dfddr2z * G1_meas[:, 2] - dfddr4z * G1_meas[:, 2]
            # Bz[:, 61] = G2_meas @ Hd24[2, :].T + dfddr2z * G2_meas[:, 2] - dfddr4z * G2_meas[:, 2]
            # Bz[:, 62] = G3_meas @ Hd24[2, :].T + dfddr2z * G3_meas[:, 2] - dfddr4z * G3_meas[:, 2]

            # ii) drd24 with partials due to quadratic terms
            if Acc2.r[0] != 0:
                # Cannot estimate drd24x if the accelerometers are placed on x-axis
                Bx[:, 62] = G2_meas @ Hc24[0, :].T + dfddr2x * G2_meas[:, 0] + dfddr4x * G2_meas[:, 0]
                Bx[:, 63] = G3_meas @ Hc24[0, :].T + dfddr2x * G3_meas[:, 0] + dfddr4x * G3_meas[:, 0]

                By[:, 62] = G2_meas @ Hc24[1, :].T + dfddr2y * G2_meas[:, 1] + dfddr4y * G2_meas[:, 1]
                By[:, 63] = G3_meas @ Hc24[1, :].T + dfddr2y * G3_meas[:, 1] + dfddr4y * G3_meas[:, 1]

                Bz[:, 62] = G2_meas @ Hc24[2, :].T + dfddr2z * G2_meas[:, 2] + dfddr4z * G2_meas[:, 2]
                Bz[:, 63] = G3_meas @ Hc24[2, :].T + dfddr2z * G3_meas[:, 2] + dfddr4z * G3_meas[:, 2]

            elif Acc2.r[1] != 0:
                # Cannot estimate drd24y if the accelerometers are placed on y-axis
                Bx[:, 62] = G1_meas @ Hc24[0, :].T + dfddr2x * G1_meas[:, 0] + dfddr4x * G1_meas[:, 0]
                Bx[:, 63] = G3_meas @ Hc24[0, :].T + dfddr2x * G3_meas[:, 0] + dfddr4x * G3_meas[:, 0]

                By[:, 62] = G1_meas @ Hc24[1, :].T + dfddr2y * G1_meas[:, 1] + dfddr4y * G1_meas[:, 1]
                By[:, 63] = G3_meas @ Hc24[1, :].T + dfddr2y * G3_meas[:, 1] + dfddr4y * G3_meas[:, 1]

                Bz[:, 62] = G1_meas @ Hc24[2, :].T + dfddr2z * G1_meas[:, 2] + dfddr4z * G1_meas[:, 2]
                Bz[:, 63] = G3_meas @ Hc24[2, :].T + dfddr2z * G3_meas[:, 2] + dfddr4z * G3_meas[:, 2]

            else:
                # Cannot estimate drd24z if the accelerometers are placed on z-axis
                Bx[:, 62] = G1_meas @ Hc24[0, :].T + dfddr2x * G1_meas[:, 0] + dfddr4x * G1_meas[:, 0]
                Bx[:, 63] = G2_meas @ Hc24[0, :].T + dfddr2x * G2_meas[:, 0] + dfddr4x * G2_meas[:, 0]

                By[:, 62] = G1_meas @ Hc24[1, :].T + dfddr2y * G1_meas[:, 1] + dfddr4y * G1_meas[:, 1]
                By[:, 63] = G2_meas @ Hc24[1, :].T + dfddr2y * G2_meas[:, 1] + dfddr4y * G2_meas[:, 1]

                Bz[:, 62] = G1_meas @ Hc24[2, :].T + dfddr2z * G1_meas[:, 2] + dfddr4z * G1_meas[:, 2]
                Bz[:, 63] = G2_meas @ Hc24[2, :].T + dfddr2z * G2_meas[:, 2] + dfddr4z * G2_meas[:, 2]

            ###########################################################################################################################
            # Common Mode #############################################################################################################
            ###########################################################################################################################
            ycx = np.zeros(N)
            ycy = np.zeros(N)
            ycz = np.zeros(N)

            yc0 = (ac13_meas - (ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2]) @ Hd13.T -
                   0.5 * (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2 @ K1.T -
                   0.5 * (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2 @ K3.T - dw_meas @ Wc13.T) @ Hc13_inv.T - \
                  (ac24_meas - (ad24_np + G1_meas * drd24[0] + G2_meas * drd24[1] + G3_meas * drd24[2]) @ Hd24.T -
                   0.5 * (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2 @ K2.T -
                   0.5 * (a4_np + G1_meas * dr4[0] + G2_meas * dr4[1] + G3_meas * dr4[2]) ** 2 @ K4.T - dw_meas @ Wc24.T) @ Hc24_inv.T - \
                  (G1_meas * (drc13[0] - drc24[0])) - (G2_meas * (drc13[1] - drc24[1])) - (G3_meas * (drc13[2] - drc24[2]))

            yc0x = yc0[:, 0]
            yc0y = yc0[:, 1]
            yc0z = yc0[:, 2]

            # Create the design matrix
            #################################
            # 1) M-parameters: ddMc13, ddMc24, ddMd13, ddMd24
            #################################
            # dMc13
            com_dMc13_kron = -np.kron(Hc13_inv, (ac13_meas - (ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2]) @ Hd13.T -
                                                 0.5 * (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2 @ K1.T -
                                                 0.5 * (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2 @ K3.T -
                                                 dw_meas @ Wc13.T) @ Hc13_inv.T)

            Cx[:, idx_Mc13] = com_dMc13_kron[0:N, :]
            Cy[:, idx_Mc13] = com_dMc13_kron[N:2 * N, :]
            Cz[:, idx_Mc13] = com_dMc13_kron[2 * N:3 * N, :]

            # dMc24

            com_dMc24_kron = np.kron(Hc24_inv, (ac24_meas - (ad24_np + G1_meas * drd24[0] + G2_meas * drd24[1] + G3_meas * drd24[2]) @ Hd24.T -
                                                0.5 * (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2 @ K2.T -
                                                0.5 * (a4_np + G1_meas * dr4[0] + G2_meas * dr4[1] + G3_meas * dr4[2]) ** 2 @ K4.T -
                                                dw_meas @ Wc24.T) @ Hc24_inv.T)

            Cx[:, idx_Mc24] = com_dMc24_kron[0:N, :]
            Cy[:, idx_Mc24] = com_dMc24_kron[N:2 * N, :]
            Cz[:, idx_Mc24] = com_dMc24_kron[2 * N:3 * N, :]

            # dMd13
            com_dMd13_kron = -np.kron(Hc13_inv, ad13_np + G1_meas * drd13[0] + G2_meas * drd13[1] + G3_meas * drd13[2])

            Cx[:, idx_Md13] = com_dMd13_kron[0:N, :]
            Cy[:, idx_Md13] = com_dMd13_kron[N:2 * N, :]
            Cz[:, idx_Md13] = com_dMd13_kron[2 * N:3 * N, :]

            # dMd24
            com_dMd24_kron = np.kron(Hc24_inv, ad24_np + G1_meas * drd24[0] + G2_meas * drd24[1] + G3_meas * drd24[2])

            Cx[:, idx_Md24] = com_dMd24_kron[0:N, :]
            Cy[:, idx_Md24] = com_dMd24_kron[N:2 * N, :]
            Cz[:, idx_Md24] = com_dMd24_kron[2 * N:3 * N, :]

            #################################
            # 2) K-parameters: K1, K2, K3, K4
            #################################
            # K1
            com_K1_kron = -0.5 * np.kron(Hc13_inv, (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) ** 2)
            Cx[:, 36] = com_K1_kron[0:N, 0]
            Cx[:, 37] = com_K1_kron[0:N, 4]
            Cx[:, 38] = com_K1_kron[0:N, 8]

            Cy[:, 36] = com_K1_kron[N:2 * N, 0]
            Cy[:, 37] = com_K1_kron[N:2 * N, 4]
            Cy[:, 38] = com_K1_kron[N:2 * N, 8]

            Cz[:, 36] = com_K1_kron[2 * N:3 * N, 0]
            Cz[:, 37] = com_K1_kron[2 * N:3 * N, 4]
            Cz[:, 38] = com_K1_kron[2 * N:3 * N, 8]

            # K2
            com_K2_kron = 0.5 * np.kron(Hc24_inv, (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) ** 2)
            Cx[:, 39] = com_K2_kron[0:N, 0]
            Cx[:, 40] = com_K2_kron[0:N, 4]
            Cx[:, 41] = com_K2_kron[0:N, 8]

            Cy[:, 39] = com_K2_kron[N:2 * N, 0]
            Cy[:, 40] = com_K2_kron[N:2 * N, 4]
            Cy[:, 41] = com_K2_kron[N:2 * N, 8]

            Cz[:, 39] = com_K2_kron[2 * N:3 * N, 0]
            Cz[:, 40] = com_K2_kron[2 * N:3 * N, 4]
            Cz[:, 41] = com_K2_kron[2 * N:3 * N, 8]

            # K3
            com_K3_kron = -0.5 * np.kron(Hc13_inv, (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) ** 2)
            Cx[:, 42] = com_K3_kron[0:N, 0]
            Cx[:, 43] = com_K3_kron[0:N, 4]
            Cx[:, 44] = com_K3_kron[0:N, 8]

            Cy[:, 42] = com_K3_kron[N:2 * N, 0]
            Cy[:, 43] = com_K3_kron[N:2 * N, 4]
            Cy[:, 44] = com_K3_kron[N:2 * N, 8]

            Cz[:, 42] = com_K3_kron[2 * N:3 * N, 0]
            Cz[:, 43] = com_K3_kron[2 * N:3 * N, 4]
            Cz[:, 44] = com_K3_kron[2 * N:3 * N, 8]

            # K4
            com_K4_kron = 0.5 * np.kron(Hc24_inv, (a4_np + G1_meas * dr4[0] + G2_meas * dr4[1] + G3_meas * dr4[2]) ** 2)
            Cx[:, 45] = com_K4_kron[0:N, 0]
            Cx[:, 46] = com_K4_kron[0:N, 4]
            Cx[:, 47] = com_K4_kron[0:N, 8]

            Cy[:, 45] = com_K4_kron[N:2 * N, 0]
            Cy[:, 46] = com_K4_kron[N:2 * N, 4]
            Cy[:, 47] = com_K4_kron[N:2 * N, 8]

            Cz[:, 45] = com_K4_kron[2 * N:3 * N, 0]
            Cz[:, 46] = com_K4_kron[2 * N:3 * N, 4]
            Cz[:, 47] = com_K4_kron[2 * N:3 * N, 8]

            #################################
            # 3) W-parameter: Wc13, Wc24
            #################################
            # Since Wc24 is defined to be zero, we only need to consider Wc13
            com_Wc13_kron = np.kron(Hc13_inv, dw_meas)

            Cx[:, 54] = com_Wc13_kron[0:N, 3]
            Cx[:, 55] = com_Wc13_kron[0:N, 7]
            Cx[:, 56] = com_Wc13_kron[0:N, 5]

            Cy[:, 54] = com_Wc13_kron[N:2 * N, 3]
            Cy[:, 55] = com_Wc13_kron[N:2 * N, 7]
            Cy[:, 56] = com_Wc13_kron[N:2 * N, 5]

            Cz[:, 54] = com_Wc13_kron[2 * N:3 * N, 3]
            Cz[:, 55] = com_Wc13_kron[2 * N:3 * N, 7]
            Cz[:, 56] = com_Wc13_kron[2 * N:3 * N, 5]

            #################################
            # 4) dr-parameters: drc13, drc24, drd13, drd24. Cannot estimate drc24. Assume it is zero
            #################################
            # Create local variables to avoid recalculations of common terms

            dfcdr1_temp = (a1_np + G1_meas * dr1[0] + G2_meas * dr1[1] + G3_meas * dr1[2]) @ K1.T
            dfcdr3_temp = (a3_np + G1_meas * dr3[0] + G2_meas * dr3[1] + G3_meas * dr3[2]) @ K3.T

            dfcdr1x = G1_meas * dfcdr1_temp
            dfcdr1y = G2_meas * dfcdr1_temp
            dfcdr1z = G3_meas * dfcdr1_temp

            dfcdr3x = G1_meas * dfcdr3_temp
            dfcdr3y = G2_meas * dfcdr3_temp
            dfcdr3z = G3_meas * dfcdr3_temp

            # i) drc13 with partials due to quadratic terms
            Cx[:, 57] = -G1_meas[:, 0] - dfcdr1x @ Hc13_inv[0, :].T - dfcdr3x @ Hc13_inv[0, :].T
            Cx[:, 58] = -G2_meas[:, 0] - dfcdr1y @ Hc13_inv[0, :].T - dfcdr3y @ Hc13_inv[0, :].T
            Cx[:, 59] = -G3_meas[:, 0] - dfcdr1z @ Hc13_inv[0, :].T - dfcdr3z @ Hc13_inv[0, :].T

            Cy[:, 57] = -G1_meas[:, 1] - dfcdr1x @ Hc13_inv[1, :].T - dfcdr3x @ Hc13_inv[1, :].T
            Cy[:, 58] = -G2_meas[:, 1] - dfcdr1y @ Hc13_inv[1, :].T - dfcdr3y @ Hc13_inv[1, :].T
            Cy[:, 59] = -G3_meas[:, 1] - dfcdr1z @ Hc13_inv[1, :].T - dfcdr3z @ Hc13_inv[1, :].T

            Cz[:, 57] = -G1_meas[:, 2] - dfcdr1x @ Hc13_inv[2, :].T - dfcdr3x @ Hc13_inv[2, :].T
            Cz[:, 58] = -G2_meas[:, 2] - dfcdr1y @ Hc13_inv[2, :].T - dfcdr3y @ Hc13_inv[2, :].T
            Cz[:, 59] = -G3_meas[:, 2] - dfcdr1z @ Hc13_inv[2, :].T - dfcdr3z @ Hc13_inv[2, :].T


            dfcd2_temp = (a2_np + G1_meas * dr2[0] + G2_meas * dr2[1] + G3_meas * dr2[2]) @ K2.T
            dfcd4_temp = (a4_np + G1_meas * dr4[0] + G2_meas * dr4[1] + G3_meas * dr4[2]) @ K4.T

            dfcdr2x = G1_meas * dfcd2_temp
            dfcdr2y = G2_meas * dfcd2_temp
            dfcdr2z = G3_meas * dfcd2_temp

            dfdr4x = G1_meas * dfcd4_temp
            dfdr4y = G2_meas * dfcd4_temp
            dfdr4z = G3_meas * dfcd4_temp

            # Cx[:, 60] = G1_meas[:, 0] + dfcdr2x @ Hc24_inv[0, :].T + dfdr4x @ Hc24_inv[0, :].T
            # Cx[:, 61] = G2_meas[:, 0] + dfcdr2y @ Hc24_inv[0, :].T + dfdr4y @ Hc24_inv[0, :].T
            # Cx[:, 62] = G3_meas[:, 0] + dfcdr2z @ Hc24_inv[0, :].T + dfdr4z @ Hc24_inv[0, :].T
            #
            # Cy[:, 60] = G1_meas[:, 1] + dfcdr2x @ Hc24_inv[1, :].T + dfdr4x @ Hc24_inv[1, :].T
            # Cy[:, 61] = G2_meas[:, 1] + dfcdr2y @ Hc24_inv[1, :].T + dfdr4y @ Hc24_inv[1, :].T
            # Cy[:, 62] = G3_meas[:, 1] + dfcdr2z @ Hc24_inv[1, :].T + dfdr4z @ Hc24_inv[1, :].T
            #
            # Cz[:, 60] = G1_meas[:, 2] + dfcdr2x @ Hc24_inv[2, :].T + dfdr4x @ Hc24_inv[2, :].T
            # Cz[:, 61] = G2_meas[:, 2] + dfcdr2y @ Hc24_inv[2, :].T + dfdr4y @ Hc24_inv[2, :].T
            # Cz[:, 62] = G3_meas[:, 2] + dfcdr2z @ Hc24_inv[2, :].T + dfdr4z @ Hc24_inv[2, :].T

            # iii) drd13 with partials due to quadratic terms
            if Acc1.r[0] != 0:
                # Cannot estimate drd13x if the accelerometers are placed on x-axis
                Cx[:, 60] = -G2_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1y @ Hc13_inv[0, :].T + dfcdr3y @ Hc13_inv[0, :].T
                Cx[:, 61] = -G3_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1z @ Hc13_inv[0, :].T + dfcdr3z @ Hc13_inv[0, :].T

                Cy[:, 60] = -G2_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1y @ Hc13_inv[1, :].T + dfcdr3y @ Hc13_inv[1, :].T
                Cy[:, 61] = -G3_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1z @ Hc13_inv[1, :].T + dfcdr3z @ Hc13_inv[1, :].T

                Cz[:, 60] = -G2_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1y @ Hc13_inv[2, :].T + dfcdr3y @ Hc13_inv[2, :].T
                Cz[:, 61] = -G3_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1z @ Hc13_inv[2, :].T + dfcdr3z @ Hc13_inv[2, :].T

            elif Acc1.r[1] != 0:
                # Cannot estimate drd13y if the accelerometers are placed on y-axis
                Cx[:, 60] = -G1_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1x @ Hc13_inv[0, :].T + dfcdr3x @ Hc13_inv[0, :].T
                Cx[:, 61] = -G3_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1z @ Hc13_inv[0, :].T + dfcdr3z @ Hc13_inv[0, :].T

                Cy[:, 60] = -G1_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1x @ Hc13_inv[1, :].T + dfcdr3x @ Hc13_inv[1, :].T
                Cy[:, 61] = -G3_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1z @ Hc13_inv[1, :].T + dfcdr3z @ Hc13_inv[1, :].T

                Cz[:, 60] = -G1_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1x @ Hc13_inv[2, :].T + dfcdr3x @ Hc13_inv[2, :].T
                Cz[:, 61] = -G3_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1z @ Hc13_inv[2, :].T + dfcdr3z @ Hc13_inv[2, :].T

            else:
                # Cannot estimate drd13z if the accelerometers are placed on z-axis
                Cx[:, 60] = -G1_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1x @ Hc13_inv[0, :].T + dfcdr3x @ Hc13_inv[0, :].T
                Cx[:, 61] = -G2_meas @ Hd13.T @ Hc13_inv[0, :].T - dfcdr1y @ Hc13_inv[0, :].T + dfcdr3y @ Hc13_inv[0, :].T

                Cy[:, 60] = -G1_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1x @ Hc13_inv[1, :].T + dfcdr3x @ Hc13_inv[1, :].T
                Cy[:, 61] = -G2_meas @ Hd13.T @ Hc13_inv[1, :].T - dfcdr1y @ Hc13_inv[1, :].T + dfcdr3y @ Hc13_inv[1, :].T

                Cz[:, 60] = -G1_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1x @ Hc13_inv[2, :].T + dfcdr3x @ Hc13_inv[2, :].T
                Cz[:, 61] = -G2_meas @ Hd13.T @ Hc13_inv[2, :].T - dfcdr1y @ Hc13_inv[2, :].T + dfcdr3y @ Hc13_inv[2, :].T

            # iv) drd24 with partials due to quadratic terms
            if Acc2.r[0] != 0:
                # Cannot estimate drd24x if the accelerometers are placed on x-axis
                Cx[:, 62] = G2_meas @ Hd24.T @ Hc24_inv[0, :].T + dfcdr2y @ Hc24_inv[0, :].T - dfdr4y @ Hc24_inv[0, :].T
                Cx[:, 63] = G3_meas @ Hd24.T @ Hc24_inv[0, :].T + dfcdr2z @ Hc24_inv[0, :].T - dfdr4z @ Hc24_inv[0, :].T

                Cy[:, 62] = G2_meas @ Hd24.T @ Hc24_inv[1, :].T + dfcdr2y @ Hc24_inv[1, :].T - dfdr4y @ Hc24_inv[1, :].T
                Cy[:, 63] = G3_meas @ Hd24.T @ Hc24_inv[1, :].T + dfcdr2z @ Hc24_inv[1, :].T - dfdr4z @ Hc24_inv[1, :].T

                Cz[:, 62] = G2_meas @ Hd24.T @ Hc24_inv[2, :].T + dfcdr2y @ Hc24_inv[2, :].T - dfdr4y @ Hc24_inv[2, :].T
                Cz[:, 63] = G3_meas @ Hd24.T @ Hc24_inv[2, :].T + dfcdr2z @ Hc24_inv[2, :].T - dfdr4z @ Hc24_inv[2, :].T

            elif Acc2.r[1] != 0:
                # Cannot estimate drd24y if the accelerometers are placed on y-axis
                Cx[:, 62] = G1_meas @ Hd24.T @ Hc24_inv[0, :].T + dfcdr2x @ Hc24_inv[0, :].T - dfdr4x @ Hc24_inv[0, :].T
                Cx[:, 63] = G3_meas @ Hd24.T @ Hc24_inv[0, :].T + dfcdr2z @ Hc24_inv[0, :].T - dfdr4z @ Hc24_inv[0, :].T

                Cy[:, 62] = G1_meas @ Hd24.T @ Hc24_inv[1, :].T + dfcdr2x @ Hc24_inv[1, :].T - dfdr4x @ Hc24_inv[1, :].T
                Cy[:, 63] = G3_meas @ Hd24.T @ Hc24_inv[1, :].T + dfcdr2z @ Hc24_inv[1, :].T - dfdr4z @ Hc24_inv[1, :].T

                Cz[:, 62] = G1_meas @ Hd24.T @ Hc24_inv[2, :].T + dfcdr2x @ Hc24_inv[2, :].T - dfdr4x @ Hc24_inv[2, :].T
                Cz[:, 63] = G3_meas @ Hd24.T @ Hc24_inv[2, :].T + dfcdr2z @ Hc24_inv[2, :].T - dfdr4z @ Hc24_inv[2, :].T

            else:
                # Cannot estimate drd24z if the accelerometers are placed on z-axis
                Cx[:, 62] = G1_meas @ Hd24.T @ Hc24_inv[0, :].T + dfcdr2x @ Hc24_inv[0, :].T - dfdr4x @ Hc24_inv[0, :].T
                Cx[:, 63] = G2_meas @ Hd24.T @ Hc24_inv[0, :].T + dfcdr2y @ Hc24_inv[0, :].T - dfdr4y @ Hc24_inv[0, :].T

                Cy[:, 62] = G1_meas @ Hd24.T @ Hc24_inv[1, :].T + dfcdr2x @ Hc24_inv[1, :].T - dfdr4x @ Hc24_inv[1, :].T
                Cy[:, 63] = G2_meas @ Hd24.T @ Hc24_inv[1, :].T + dfcdr2y @ Hc24_inv[1, :].T - dfdr4y @ Hc24_inv[1, :].T

                Cz[:, 62] = G1_meas @ Hd24.T @ Hc24_inv[2, :].T + dfcdr2x @ Hc24_inv[2, :].T - dfdr4x @ Hc24_inv[2, :].T
                Cz[:, 63] = G2_meas @ Hd24.T @ Hc24_inv[2, :].T + dfcdr2y @ Hc24_inv[2, :].T - dfdr4y @ Hc24_inv[2, :].T

            ###########################################################################################################################

            return yd13x, yd13y, yd13z, yd24x, yd24y, yd24z, ycx, ycy, ycz, yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z, Ax, Ay, \
                Az, Bx, By, Bz, Cx, Cy, Cz


# noinspection PyPep8Naming
def calibrate(Acc_lst, layout, NFFT, noise_switch=True):
    # Determine the amount of accelerometers in the layout.
    N = len(Acc_lst)
    # Number of observation equations = 3 * (N - 1)
    n_eq = 3 * (N - 1)

    x_true = acc_cal_par_mat_to_vec(Acc_lst, layout)

    # Control parameters for estimation algorithm. Number of steps = iter_reconstruct * iter_linearize
    iter_reconstruct = 4  # Number of times the stochastic model is adjusted (outer loop)
    iter_linearize = 2  # Number of times the Taylor point is refined (inner loop)
    iter_quadratic = 3  # Number of times the quadratic model is estimated (a_cal)

    match layout:
        case 1:
            # Initialize Taylor point for linearization
            x0 = np.zeros(len(x_true))
            # x0 = np.random.randn(len(x_true))
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]

            # Initialise decorrelation filters

            fil_bp_dx = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dy = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dz = bandpass_filter(NFFT, 1e-4, 0.1)

            # Keep track of error in estimated parameters error analysis from true values
            x_err = []
            x_err_initial = x0 - x_true

            # Keep track of residuals for PSD analysis
            residuals_adx = []
            residuals_ady = []
            residuals_adz = []

            # Reconstruct loop
            for n_reconstruct in range(iter_reconstruct):
                # Estimated calibration parameters
                M1_est, M2_est, K1_est, K2_est, W1_est, W2_est, dr1_est, dr2_est = acc_cal_par_vec_to_mat(x0, Acc_lst, layout)

                if noise_switch:
                    a1_cal = Acc1.a_meas.copy()
                    a2_cal = Acc2.a_meas.copy()

                    # Loop for estimating true quadratic acceleration
                    for n_quad in range(iter_quadratic):
                        a1_cal = (Acc1.a_meas - a1_cal ** 2 @ K1_est.T - Acc1.dw_meas @ W1_est.T) @ np.linalg.inv(M1_est.T)
                        a2_cal = (Acc2.a_meas - a2_cal ** 2 @ K2_est.T - Acc2.dw_meas @ W2_est.T) @ np.linalg.inv(M2_est.T)
                    # end quadratic loop

                    # Reconstruct the acceleration at the nominal position
                    a1_cal_nom = a1_cal - Acc1.G1_meas * dr1_est[0] - Acc1.G2_meas * dr1_est[1] - Acc1.G3_meas * dr1_est[2]
                    a2_cal_nom = a2_cal - Acc2.G1_meas * dr2_est[0] - Acc2.G2_meas * dr2_est[1] - Acc2.G3_meas * dr2_est[2]

                    # Reconstruct biased estimate of non-gravitational acceleration
                    a_ng_rcst = (a1_cal_nom + a2_cal_nom) / 2
                else:
                    a_ng_rcst = Acc1.a_ng

                x_err_linearize = []
                # Loop over linearization
                for n_linearize in range(iter_linearize):
                    ydx, ydy, ydz, yd0x, yd0y, yd0z, Adx, Ady, Adz = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                    yf_dx = Acc1.filter_matrix(fil_bp_dx, ydx)
                    yf_dy = Acc1.filter_matrix(fil_bp_dy, ydy)
                    yf_dz = Acc1.filter_matrix(fil_bp_dz, ydz)

                    yf0_dx = Acc1.filter_matrix(fil_bp_dx, yd0x)
                    yf0_dy = Acc1.filter_matrix(fil_bp_dy, yd0y)
                    yf0_dz = Acc1.filter_matrix(fil_bp_dz, yd0z)

                    Af_dx = Acc1.filter_matrix(fil_bp_dx, Adx)
                    Af_dy = Acc1.filter_matrix(fil_bp_dy, Ady)
                    Af_dz = Acc1.filter_matrix(fil_bp_dz, Adz)

                    A = np.vstack((Adx, Ady, Adz))
                    y = np.hstack((ydx, ydy, ydz))
                    y0 = np.hstack((yd0x, yd0y, yd0z))

                    Af = np.vstack((Af_dx, Af_dy, Af_dz))
                    yf = np.hstack((yf_dx, yf_dy, yf_dz))
                    y0f = np.hstack((yf0_dx, yf0_dy, yf0_dz))

                    dy = y - y0
                    dyf = yf - y0f

                    # Rescale K factors for better conditioning. 1e6  is a good scaling factor
                    # A[:, 18:24] = A[:, 18:24] * 1e6
                    Af[:, 18:24] = Af[:, 18:24] * 1e6

                    # # Check the condition number of the design matrix
                    # cond_A = np.linalg.cond(A.T @ A)
                    cond_Af = np.linalg.cond(Af.T @ Af)
                    # print(f"Condition number of A = {cond_A}")
                    digit_loss = np.log10(cond_Af)
                    print(f"Condition number of AfTAf = {cond_Af}")
                    print(f"Digit loss = {digit_loss}\n")

                    dx = np.linalg.solve(Af.T @ Af, Af.T @ dyf)
                    dx[18:24] = dx[18:24] * 1e6

                    # Update the estimated parameters
                    x0 = x0 + dx
                    x_err_linearize.append(x0 - x_true)

                # end linearization loop
                x_err.append(x_err_linearize)

                # Analytical design matrix
                ydx, ydy, ydz, yd0x, yd0y, yd0z, _, _, _ = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                edx = ydx - yd0x
                edy = ydy - yd0y
                edz = ydz - yd0z

                fil_bp_dx = Acc1.decorrelation_filter(edx, NFFT)
                fil_bp_dy = Acc1.decorrelation_filter(edy, NFFT)
                fil_bp_dz = Acc1.decorrelation_filter(edz, NFFT)

                residuals_adx.append(edx)
                residuals_ady.append(edy)
                residuals_adz.append(edz)
            # end reconstruct loop

            x_err = np.array(x_err)
            residuals_adx = np.array(residuals_adx).T
            residuals_ady = np.array(residuals_ady).T
            residuals_adz = np.array(residuals_adz).T

            # Calculate the covariance matrix of the estimated parameters
            cov_par = np.linalg.inv(Af.T @ Af)

            # Calculate the singular values
            U, S, VT = np.linalg.svd(Af.T @ Af, full_matrices=False)

            return x_err_initial, x_err, [residuals_adx, residuals_ady, residuals_adz], cov_par, x0, x_true, S, digit_loss

        case 2:
            # Initialize Taylor point for linearization
            x0 = np.zeros(len(x_true))
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]
            Acc3 = Acc_lst[2]

            # Initialise decorrelation filters

            fil_bp_dx = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dy = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dz = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_cx = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_cy = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_cz = bandpass_filter(NFFT, 1e-4, 0.1)

            # Keep track of error in estimated parameters error analysis from true values
            x_err = []
            x_err_initial = x0 - x_true

            # Keep track of residuals for PSD analysis
            residuals_adx = []
            residuals_ady = []
            residuals_adz = []
            residuals_acx = []
            residuals_acy = []
            residuals_acz = []

            # Reconstruct loop
            for n_reconstruct in range(iter_reconstruct):
                # Estimated calibration parameters
                M1_est, M2_est, M3_est, K1_est, K2_est, K3_est, W1_est, W2_est, W3_est, dr1_est, dr2_est, dr3_est = acc_cal_par_vec_to_mat(x0,
                                                                                                                                           Acc_lst,
                                                                                                                                           layout)

                if noise_switch:
                    # Initialise calibrated acceleration for first iteration
                    a1_cal = Acc1.a_meas.copy()
                    a2_cal = Acc2.a_meas.copy()
                    a3_cal = Acc3.a_meas.copy()

                    # Loop for estimating true quadratic acceleration
                    for n_quad in range(iter_quadratic):
                        a1_cal = (Acc1.a_meas - a1_cal ** 2 @ K1_est.T - Acc1.dw_meas @ W1_est.T) @ np.linalg.inv(M1_est.T)
                        a2_cal = (Acc2.a_meas - a2_cal ** 2 @ K2_est.T - Acc2.dw_meas @ W2_est.T) @ np.linalg.inv(M2_est.T)
                        a3_cal = (Acc3.a_meas - a3_cal ** 2 @ K3_est.T - Acc3.dw_meas @ W3_est.T) @ np.linalg.inv(M3_est.T)
                    # end quadratic loop

                    # Reconstruct the acceleration at the nominal position
                    a1_cal_nom = a1_cal - Acc1.G1_meas * dr1_est[0] - Acc1.G2_meas * dr1_est[1] - Acc1.G3_meas * dr1_est[2]
                    a2_cal_nom = a2_cal - Acc2.G1_meas * dr2_est[0] - Acc2.G2_meas * dr2_est[1] - Acc2.G3_meas * dr2_est[2]
                    a3_cal_nom = a3_cal - Acc3.G1_meas * dr3_est[0] - Acc3.G2_meas * dr3_est[1] - Acc3.G3_meas * dr3_est[2]

                    # Reconstruct biased estimate of non-gravitational acceleration
                    a_ng_rcst = (a1_cal_nom + a2_cal_nom + a3_cal_nom) / 3
                else:
                    a_ng_rcst = Acc1.a_ng

                x_err_linearize = []
                # Loop over linearization
                for n_linearize in range(iter_linearize):
                    ydx, ydy, ydz, ycx, ycy, ycz, yd0x, yd0y, yd0z, yc0x, yc0y, yc0z, \
                        Adx, Ady, Adz, Acx, Acy, Acz = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                    yf_dx = Acc1.filter_matrix(fil_bp_dx, ydx)
                    yf_dy = Acc1.filter_matrix(fil_bp_dy, ydy)
                    yf_dz = Acc1.filter_matrix(fil_bp_dz, ydz)
                    yf_cx = Acc1.filter_matrix(fil_bp_cx, ycx)
                    yf_cy = Acc1.filter_matrix(fil_bp_cy, ycy)
                    yf_cz = Acc1.filter_matrix(fil_bp_cz, ycz)

                    yf_d0x = Acc1.filter_matrix(fil_bp_dx, yd0x)
                    yf_d0y = Acc1.filter_matrix(fil_bp_dy, yd0y)
                    yf_d0z = Acc1.filter_matrix(fil_bp_dz, yd0z)
                    yf_c0x = Acc1.filter_matrix(fil_bp_cx, yc0x)
                    yf_c0y = Acc1.filter_matrix(fil_bp_cy, yc0y)
                    yf_c0z = Acc1.filter_matrix(fil_bp_cz, yc0z)

                    Af_dx = Acc1.filter_matrix(fil_bp_dx, Adx)
                    Af_dy = Acc1.filter_matrix(fil_bp_dy, Ady)
                    Af_dz = Acc1.filter_matrix(fil_bp_dz, Adz)
                    Af_cx = Acc1.filter_matrix(fil_bp_cx, Acx)
                    Af_cy = Acc1.filter_matrix(fil_bp_cy, Acy)
                    Af_cz = Acc1.filter_matrix(fil_bp_cz, Acz)

                    A = np.vstack((Adx, Ady, Adz, Acx, Acy, Acz))
                    y = np.hstack((ydx, ydy, ydz, ycx, ycy, ycz))
                    y0 = np.hstack((yd0x, yd0y, yd0z, yc0x, yc0y, yc0z))

                    Af = np.vstack((Af_dx, Af_dy, Af_dz, Af_cx, Af_cy, Af_cz))
                    yf = np.hstack((yf_dx, yf_dy, yf_dz, yf_cx, yf_cy, yf_cz))
                    y0f = np.hstack((yf_d0x, yf_d0y, yf_d0z, yf_c0x, yf_c0y, yf_c0z))

                    dy = y - y0
                    dyf = yf - y0f

                    # Rescale K factors for better conditioning. 1e6 is a good scaling factor
                    # A[:, 27:36] = A[:, 27:36] * 1e6
                    Af[:, 27:36] = Af[:, 27:36] * 1e6

                    # Check the condition number of the design matrix
                    cond_A = np.linalg.cond(A.T @ A)
                    cond_Af = np.linalg.cond(Af.T @ Af)
                    print("Condition number of A", cond_A)
                    print("Condition number of Af", cond_Af)
                    digit_loss = np.log10(cond_Af)
                    print(f"Digit loss = {digit_loss}\n")

                    # Solve the normal equations. If noise_switch is False, the noise is not considered in the solution
                    # if noise_switch:
                    #
                    # else:
                    # dx = np.linalg.solve((A.T @ A), (A.T @ dy))
                    dx = np.linalg.solve((Af.T @ Af), (Af.T @ dyf))
                    # Rescale the solution for the K factors
                    dx[27:36] = dx[27:36] * 1e6

                    x0 = x0 + dx

                    x_err_linearize.append(x0 - x_true)

                # end linearize loop
                x_err.append(x_err_linearize)

                # Analytical design matrix
                ydx, ydy, ydz, ycx, ycy, ycz, yd0x, yd0y, yd0z, yc0x, yc0y, yc0z, \
                    _, _, _, _, _, _ = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                edx = ydx - yd0x
                edy = ydy - yd0y
                edz = ydz - yd0z
                ecx = ycx - yc0x
                ecy = ycy - yc0y
                ecz = ycz - yc0z

                fil_bp_dx = Acc1.decorrelation_filter(edx, NFFT)
                fil_bp_dy = Acc1.decorrelation_filter(edy, NFFT)
                fil_bp_dz = Acc1.decorrelation_filter(edz, NFFT)
                fil_bp_cx = Acc1.decorrelation_filter(ecx, NFFT)
                fil_bp_cy = Acc1.decorrelation_filter(ecy, NFFT)
                fil_bp_cz = Acc1.decorrelation_filter(ecz, NFFT)

                residuals_adx.append(edx)
                residuals_ady.append(edy)
                residuals_adz.append(edz)
                residuals_acx.append(ecx)
                residuals_acy.append(ecy)
                residuals_acz.append(ecz)

            # end reconstruct loop
            x_err = np.array(x_err)
            residuals_adx = np.array(residuals_adx).T
            residuals_ady = np.array(residuals_ady).T
            residuals_adz = np.array(residuals_adz).T
            residuals_acx = np.array(residuals_acx).T
            residuals_acy = np.array(residuals_acy).T
            residuals_acz = np.array(residuals_acz).T

            # Calculate the covariance matrix of the parameters
            cov_par = np.linalg.inv(Af.T @ Af)

            # Calculate the singular values
            U, S, VT = np.linalg.svd(Af.T @ Af, full_matrices=False)

            return x_err_initial, x_err, [residuals_adx, residuals_ady, residuals_adz,
                                          residuals_acx, residuals_acy, residuals_acz], cov_par, x0, x_true, S, digit_loss

        case 3:
            # Initialize Taylor point for linearization
            x0 = np.zeros(len(x_true))
            Acc1 = Acc_lst[0]
            Acc2 = Acc_lst[1]
            Acc3 = Acc_lst[2]
            Acc4 = Acc_lst[3]

            # Initialise decorrelation filters
            fil_bp_dx13 = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dy13 = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dz13 = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dx24 = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dy24 = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_dz24 = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_cx = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_cy = bandpass_filter(NFFT, 1e-4, 0.1)
            fil_bp_cz = bandpass_filter(NFFT, 1e-4, 0.1)

            # Keep track of error in estimated parameters error analysis from true values
            x_err = []
            x_err_initial = x0 - x_true

            # Keep track of residuals for PSD analysis
            residuals_adx13 = []
            residuals_ady13 = []
            residuals_adz13 = []
            residuals_adx24 = []
            residuals_ady24 = []
            residuals_adz24 = []
            residuals_acx = []
            residuals_acy = []
            residuals_acz = []

            # Reconstruct loop
            for n_reconstruct in range(iter_reconstruct):
                # Estimated calibration parameters
                M1_est, M2_est, M3_est, M4_est, K1_est, K2_est, K3_est, K4_est, W1_est, W2_est, W3_est, \
                    W4_est, dr1_est, dr2_est, dr3_est, dr4_est = acc_cal_par_vec_to_mat(x0, Acc_lst, layout)

                if noise_switch:
                    # Initialise calibrated acceleration for first iteration
                    a1_cal = Acc1.a_meas.copy()
                    a2_cal = Acc2.a_meas.copy()
                    a3_cal = Acc3.a_meas.copy()
                    a4_cal = Acc4.a_meas.copy()

                    # Loop for estimating true quadratic acceleration
                    for n_quad in range(iter_quadratic):
                        a1_cal = (Acc1.a_meas - a1_cal ** 2 @ K1_est.T - Acc1.dw_meas @ W1_est.T) @ np.linalg.inv(M1_est.T)
                        a2_cal = (Acc2.a_meas - a2_cal ** 2 @ K2_est.T - Acc2.dw_meas @ W2_est.T) @ np.linalg.inv(M2_est.T)
                        a3_cal = (Acc3.a_meas - a3_cal ** 2 @ K3_est.T - Acc3.dw_meas @ W3_est.T) @ np.linalg.inv(M3_est.T)
                        a4_cal = (Acc4.a_meas - a4_cal ** 2 @ K4_est.T - Acc4.dw_meas @ W4_est.T) @ np.linalg.inv(M4_est.T)
                    # end quadratic loop

                    # Reconstruct the acceleration at the nominal position
                    a1_cal_nom = a1_cal - Acc1.G1_meas * dr1_est[0] - Acc1.G2_meas * dr1_est[1] - Acc1.G3_meas * dr1_est[2]
                    a2_cal_nom = a2_cal - Acc2.G1_meas * dr2_est[0] - Acc2.G2_meas * dr2_est[1] - Acc2.G3_meas * dr2_est[2]
                    a3_cal_nom = a3_cal - Acc3.G1_meas * dr3_est[0] - Acc3.G2_meas * dr3_est[1] - Acc3.G3_meas * dr3_est[2]
                    a4_cal_nom = a4_cal - Acc4.G1_meas * dr4_est[0] - Acc4.G2_meas * dr4_est[1] - Acc4.G3_meas * dr4_est[2]

                    # Reconstruct biased estimate of non-gravitational acceleration
                    a_ng_rcst = (a1_cal_nom + a2_cal_nom + a3_cal_nom + a4_cal_nom) / 4
                else:
                    a_ng_rcst = Acc1.a_ng

                x_err_linearize = []
                # Loop over linearization
                for n_linearize in range(iter_linearize):
                    yd13x, yd13y, yd13z, yd24x, yd24y, yd24z, ycx, ycy, ycz, yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z, \
                        Ad13x, Ad13y, Ad13z, Ad24x, Ad24y, Ad24z, Acx, Acy, Acz = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                    yf_dx13 = Acc1.filter_matrix(fil_bp_dx13, yd13x)
                    yf_dy13 = Acc1.filter_matrix(fil_bp_dy13, yd13y)
                    yf_dz13 = Acc1.filter_matrix(fil_bp_dz13, yd13z)
                    yf_dx24 = Acc1.filter_matrix(fil_bp_dx24, yd24x)
                    yf_dy24 = Acc1.filter_matrix(fil_bp_dy24, yd24y)
                    yf_dz24 = Acc1.filter_matrix(fil_bp_dz24, yd24z)
                    yf_cx = Acc1.filter_matrix(fil_bp_cx, ycx)
                    yf_cy = Acc1.filter_matrix(fil_bp_cy, ycy)
                    yf_cz = Acc1.filter_matrix(fil_bp_cz, ycz)

                    yf_d013x = Acc1.filter_matrix(fil_bp_dx13, yd013x)
                    yf_d013y = Acc1.filter_matrix(fil_bp_dy13, yd013y)
                    yf_d013z = Acc1.filter_matrix(fil_bp_dz13, yd013z)
                    yf_d024x = Acc1.filter_matrix(fil_bp_dx24, yd024x)
                    yf_d024y = Acc1.filter_matrix(fil_bp_dy24, yd024y)
                    yf_d024z = Acc1.filter_matrix(fil_bp_dz24, yd024z)
                    yf_c0x = Acc1.filter_matrix(fil_bp_cx, yc0x)
                    yf_c0y = Acc1.filter_matrix(fil_bp_cy, yc0y)
                    yf_c0z = Acc1.filter_matrix(fil_bp_cz, yc0z)

                    Af_dx13 = Acc1.filter_matrix(fil_bp_dx13, Ad13x)
                    Af_dy13 = Acc1.filter_matrix(fil_bp_dy13, Ad13y)
                    Af_dz13 = Acc1.filter_matrix(fil_bp_dz13, Ad13z)
                    Af_dx24 = Acc1.filter_matrix(fil_bp_dx24, Ad24x)
                    Af_dy24 = Acc1.filter_matrix(fil_bp_dy24, Ad24y)
                    Af_dz24 = Acc1.filter_matrix(fil_bp_dz24, Ad24z)
                    Af_cx = Acc1.filter_matrix(fil_bp_cx, Acx)
                    Af_cy = Acc1.filter_matrix(fil_bp_cy, Acy)
                    Af_cz = Acc1.filter_matrix(fil_bp_cz, Acz)

                    A = np.vstack((Ad13x, Ad13y, Ad13z, Ad24x, Ad24y, Ad24z, Acx, Acy, Acz))
                    y = np.hstack((yd13x, yd13y, yd13z, yd24x, yd24y, yd24z, ycx, ycy, ycz))
                    y0 = np.hstack((yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z))

                    Af = np.vstack((Af_dx13, Af_dy13, Af_dz13, Af_dx24, Af_dy24, Af_dz24, Af_cx, Af_cy, Af_cz))
                    yf = np.hstack((yf_dx13, yf_dy13, yf_dz13, yf_dx24, yf_dy24, yf_dz24, yf_cx, yf_cy, yf_cz))
                    y0f = np.hstack((yf_d013x, yf_d013y, yf_d013z, yf_d024x, yf_d024y, yf_d024z, yf_c0x, yf_c0y, yf_c0z))

                    dyf = yf - y0f

                    # Rescale K factors for better conditioning.  is a good scaling factor
                    A[:, 36:48] = A[:, 36:48] * 1e6
                    Af[:, 36:48] = Af[:, 36:48] * 1e6

                    # Check the condition number of the design matrix
                    cond_A = np.linalg.cond(A.T @ A)
                    cond_Af = np.linalg.cond(Af.T @ Af)
                    digit_loss = np.log10(cond_Af)
                    print(f"Condition number of AfTAf = {cond_Af}")
                    print(f"Digit loss = {digit_loss}\n")
                    print(f"Digits loss = {np.log10(cond_Af)}\n")

                    dx = np.linalg.solve((Af.T @ Af), (Af.T @ dyf))
                    dx[36:48] = dx[36:48] * 1e6

                    x0 = x0 + dx

                    x_err_linearize.append(x0 - x_true)

                # end linearize loop

                x_err.append(x_err_linearize)

                # Analytical design matrix
                yd13x, yd13y, yd13z, yd24x, yd24y, yd24z, ycx, ycy, ycz, yd013x, yd013y, yd013z, yd024x, yd024y, yd024z, yc0x, yc0y, yc0z, \
                    _, _, _, _, _, _, _, _, _ = acc_cal_linearized_function(x0, Acc_lst, a_ng_rcst, layout)

                ed13x = yd13x - yd013x
                ed13y = yd13y - yd013y
                ed13z = yd13z - yd013z
                ed24x = yd24x - yd024x
                ed24y = yd24y - yd024y
                ed24z = yd24z - yd024z
                ecx = ycx - yc0x
                ecy = ycy - yc0y
                ecz = ycz - yc0z

                fil_bp_dx13 = Acc1.decorrelation_filter(ed13x, NFFT)
                fil_bp_dy13 = Acc1.decorrelation_filter(ed13y, NFFT)
                fil_bp_dz13 = Acc1.decorrelation_filter(ed13z, NFFT)
                fil_bp_dx24 = Acc1.decorrelation_filter(ed24x, NFFT)
                fil_bp_dy24 = Acc1.decorrelation_filter(ed24y, NFFT)
                fil_bp_dz24 = Acc1.decorrelation_filter(ed24z, NFFT)
                fil_bp_cx = Acc1.decorrelation_filter(ecx, NFFT)
                fil_bp_cy = Acc1.decorrelation_filter(ecy, NFFT)
                fil_bp_cz = Acc1.decorrelation_filter(ecz, NFFT)

                residuals_adx13.append(ed13x)
                residuals_ady13.append(ed13y)
                residuals_adz13.append(ed13z)
                residuals_adx24.append(ed24x)
                residuals_ady24.append(ed24y)
                residuals_adz24.append(ed24z)
                residuals_acx.append(ecx)
                residuals_acy.append(ecy)
                residuals_acz.append(ecz)

            # end reconstruct loop
            x_err = np.array(x_err)
            residuals_adx13 = np.array(residuals_adx13).T
            residuals_ady13 = np.array(residuals_ady13).T
            residuals_adz13 = np.array(residuals_adz13).T
            residuals_adx24 = np.array(residuals_adx24).T
            residuals_ady24 = np.array(residuals_ady24).T
            residuals_adz24 = np.array(residuals_adz24).T
            residuals_acx = np.array(residuals_acx).T
            residuals_acy = np.array(residuals_acy).T
            residuals_acz = np.array(residuals_acz).T

            # Calculate the covariance matrix of the parameters
            cov_par = np.linalg.inv(Af.T @ Af)

            # Calculate the singular values
            U, S, VT = np.linalg.svd(Af.T @ Af, full_matrices=False)

            return x_err_initial, x_err, [residuals_adx13, residuals_ady13, residuals_adz13, residuals_adx24, residuals_ady24, residuals_adz24,
                                          residuals_acx, residuals_acy, residuals_acz], cov_par, x0, x_true, S, digit_loss


def reconstruct_a_ng(x0, acc_lst, layout, **kwargs):
    match layout:
        case 1:
            M1_est, M2_est, K1_est, K2_est, W1_est, W2_est, dr1_est, dr2_est = acc_cal_par_vec_to_mat(x0, acc_lst, layout=layout)

            acc1 = acc_lst[0]
            acc2 = acc_lst[1]

            a1_cal = acc1.a_meas
            a2_cal = acc2.a_meas

            iter_quadratic = 3
            for n_quad in range(iter_quadratic):
                a1_cal = (acc1.a_meas - a1_cal ** 2 @ K1_est.T - acc1.dw_meas @ W1_est.T) @ np.linalg.inv(M1_est.T)
                a2_cal = (acc2.a_meas - a2_cal ** 2 @ K2_est.T - acc2.dw_meas @ W1_est.T) @ np.linalg.inv(M2_est.T)

            if 'gg' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_nom_gg_only * dr1_est[0] - acc1.G2_nom_gg_only * dr1_est[1] - acc1.G3_nom_gg_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_nom_gg_only * dr2_est[0] - acc2.G2_nom_gg_only * dr2_est[1] - acc2.G3_nom_gg_only * dr2_est[2]

            elif 'gg_w_rate' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_meas_gg_w_rate_only * dr1_est[0] - acc1.G2_meas_gg_w_rate_only * dr1_est[1] - acc1.G3_meas_gg_w_rate_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas_gg_w_rate_only * dr2_est[0] - acc2.G2_meas_gg_w_rate_only * dr2_est[1] - acc2.G3_meas_gg_w_rate_only * dr2_est[2]
            else:
                a1_cal_nom = a1_cal - acc1.G1_meas * dr1_est[0] - acc1.G2_meas * dr1_est[1] - acc1.G3_meas * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas * dr2_est[0] - acc2.G2_meas * dr2_est[1] - acc2.G3_meas * dr2_est[2]

            a_ng = (a1_cal_nom + a2_cal_nom) / 2

            return a_ng

        case 2:
            M1_est, M2_est, M3_est, K1_est, K2_est, K3_est, W1_est, W2_est, W3_est, dr1_est, dr2_est, dr3_est = acc_cal_par_vec_to_mat(x0,
                                                                                                                                       acc_lst,
                                                                                                                                       layout=layout)

            acc1 = acc_lst[0]
            acc2 = acc_lst[1]
            acc3 = acc_lst[2]

            a1_cal = acc1.a_meas
            a2_cal = acc2.a_meas
            a3_cal = acc3.a_meas

            iter_quadratic = 3
            for n_quad in range(iter_quadratic):
                a1_cal = (acc1.a_meas - a1_cal ** 2 @ K1_est.T - acc1.dw_meas @ W1_est.T) @ np.linalg.inv(M1_est.T)
                a2_cal = (acc2.a_meas - a2_cal ** 2 @ K2_est.T - acc2.dw_meas @ W2_est.T) @ np.linalg.inv(M2_est.T)
                a3_cal = (acc3.a_meas - a3_cal ** 2 @ K3_est.T - acc3.dw_meas @ W3_est.T) @ np.linalg.inv(M3_est.T)

            if 'gg_only' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_nom_gg_only * dr1_est[0] - acc1.G2_nom_gg_only * dr1_est[1] - acc1.G3_nom_gg_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_nom_gg_only * dr2_est[0] - acc2.G2_nom_gg_only * dr2_est[1] - acc2.G3_nom_gg_only * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_nom_gg_only * dr3_est[0] - acc3.G2_nom_gg_only * dr3_est[1] - acc3.G3_nom_gg_only * dr3_est[2]
            elif 'gg_w_rate_only' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_meas_gg_w_rate_only * dr1_est[0] - acc1.G2_meas_gg_w_rate_only * dr1_est[1] - acc1.G3_meas_gg_w_rate_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas_gg_w_rate_only * dr2_est[0] - acc2.G2_meas_gg_w_rate_only * dr2_est[1] - acc2.G3_meas_gg_w_rate_only * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_meas_gg_w_rate_only * dr3_est[0] - acc3.G2_meas_gg_w_rate_only * dr3_est[1] - acc3.G3_meas_gg_w_rate_only * dr3_est[2]
            elif 'w_rate_only' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_meas_w_rate_only * dr1_est[0] - acc1.G2_meas_w_rate_only * dr1_est[1] - acc1.G3_meas_w_rate_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas_w_rate_only * dr2_est[0] - acc2.G2_meas_w_rate_only * dr2_est[1] - acc2.G3_meas_w_rate_only * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_meas_w_rate_only * dr3_est[0] - acc3.G2_meas_w_rate_only * dr3_est[1] - acc3.G3_meas_w_rate_only * dr3_est[2]
            elif 'dw_only' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_meas_dw_only * dr1_est[0] - acc1.G2_meas_dw_only * dr1_est[1] - acc1.G3_meas_dw_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas_dw_only * dr2_est[0] - acc2.G2_meas_dw_only * dr2_est[1] - acc2.G3_meas_dw_only * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_meas_dw_only * dr3_est[0] - acc3.G2_meas_dw_only * dr3_est[1] - acc3.G3_meas_dw_only * dr3_est[2]
            else:
                a1_cal_nom = a1_cal - acc1.G1_meas * dr1_est[0] - acc1.G2_meas * dr1_est[1] - acc1.G3_meas * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas * dr2_est[0] - acc2.G2_meas * dr2_est[1] - acc2.G3_meas * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_meas * dr3_est[0] - acc3.G2_meas * dr3_est[1] - acc3.G3_meas * dr3_est[2]

            a_ng = (a1_cal_nom + a2_cal_nom + a3_cal_nom) / 3

            return a_ng

        case 3:
            M1_est, M2_est, M3_est, M4_est, K1_est, K2_est, K3_est, K4_est, W1_est, W2_est, W3_est, W4_est, dr1_est, dr2_est,\
                dr3_est, dr4_est = acc_cal_par_vec_to_mat(x0, acc_lst, layout=layout)

            acc1 = acc_lst[0]
            acc2 = acc_lst[1]
            acc3 = acc_lst[2]
            acc4 = acc_lst[3]

            a1_cal = acc1.a_meas
            a2_cal = acc2.a_meas
            a3_cal = acc3.a_meas
            a4_cal = acc4.a_meas

            iter_quadratic = 3
            for n_quad in range(iter_quadratic):
                a1_cal = (acc1.a_meas - a1_cal ** 2 @ K1_est.T - acc1.dw_meas @ W1_est.T) @ np.linalg.inv(M1_est.T)
                a2_cal = (acc2.a_meas - a2_cal ** 2 @ K2_est.T - acc2.dw_meas @ W2_est.T) @ np.linalg.inv(M2_est.T)
                a3_cal = (acc3.a_meas - a3_cal ** 2 @ K3_est.T - acc3.dw_meas @ W3_est.T) @ np.linalg.inv(M3_est.T)
                a4_cal = (acc4.a_meas - a4_cal ** 2 @ K4_est.T - acc4.dw_meas @ W4_est.T) @ np.linalg.inv(M4_est.T)

            if 'gg' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_nom_gg_only * dr1_est[0] - acc1.G2_nom_gg_only * dr1_est[1] - acc1.G3_nom_gg_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_nom_gg_only * dr2_est[0] - acc2.G2_nom_gg_only * dr2_est[1] - acc2.G3_nom_gg_only * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_nom_gg_only * dr3_est[0] - acc3.G2_nom_gg_only * dr3_est[1] - acc3.G3_nom_gg_only * dr3_est[2]
                a4_cal_nom = a4_cal - acc4.G1_nom_gg_only * dr4_est[0] - acc4.G2_nom_gg_only * dr4_est[1] - acc4.G3_nom_gg_only * dr4_est[2]
            elif 'gg_w_rate' in kwargs:
                a1_cal_nom = a1_cal - acc1.G1_meas_gg_w_rate_only * dr1_est[0] - acc1.G2_meas_gg_w_rate_only * dr1_est[1] - acc1.G3_meas_gg_w_rate_only * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas_gg_w_rate_only * dr2_est[0] - acc2.G2_meas_gg_w_rate_only * dr2_est[1] - acc2.G3_meas_gg_w_rate_only * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_meas_gg_w_rate_only * dr3_est[0] - acc3.G2_meas_gg_w_rate_only * dr3_est[1] - acc3.G3_meas_gg_w_rate_only * dr3_est[2]
                a4_cal_nom = a4_cal - acc4.G1_meas_gg_w_rate_only * dr4_est[0] - acc4.G2_meas_gg_w_rate_only * dr4_est[1] - acc4.G3_meas_gg_w_rate_only * dr4_est[2]
            else:
                a1_cal_nom = a1_cal - acc1.G1_meas * dr1_est[0] - acc1.G2_meas * dr1_est[1] - acc1.G3_meas * dr1_est[2]
                a2_cal_nom = a2_cal - acc2.G1_meas * dr2_est[0] - acc2.G2_meas * dr2_est[1] - acc2.G3_meas * dr2_est[2]
                a3_cal_nom = a3_cal - acc3.G1_meas * dr3_est[0] - acc3.G2_meas * dr3_est[1] - acc3.G3_meas * dr3_est[2]
                a4_cal_nom = a4_cal - acc4.G1_meas * dr4_est[0] - acc4.G2_meas * dr4_est[1] - acc4.G3_meas * dr4_est[2]

            a_ng = (a1_cal_nom + a2_cal_nom + a3_cal_nom + a4_cal_nom) / 4

            return a_ng


def reconstruct_L2_modified_a_ng(x0, acc_lst):
        M1_est, M2_est, M3_est, K1_est, K2_est, K3_est, W1_est, W2_est, W3_est, dr1_est, dr2_est, dr3_est = acc_cal_par_vec_to_mat(x0,
                                                                                                                                   acc_lst,
                                                                                                                                   layout=2)
        acc2 = acc_lst[1]
        a2_cal = acc2.a_meas.copy()
        iter_quadratic = 3
        for n_quad in range(iter_quadratic):
            a2_cal = (acc2.a_meas - a2_cal ** 2 @ K2_est.T - acc2.dw_meas @ W2_est.T) @ np.linalg.inv(M2_est.T)
        a2_cal_nom = a2_cal - acc2.G1_meas * dr2_est[0] - acc2.G2_meas * dr2_est[1] - acc2.G3_meas * dr2_est[2]

        return a2_cal_nom

