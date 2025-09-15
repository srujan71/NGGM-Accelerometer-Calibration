import numpy as np
from math import ceil, floor
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import welch, convolve


class Noise(object):
    """
    A class to generate noise characteristics from a given PSD

    Attributes:
    ----------
    psd : np.ndarray
        Desired PSD
    fs : int
        Sampling frequency
    freq : np.ndarray
        Frequency vector
    window: str
        Averaging window. Default = Hann Window
    average_method: str
        Average using mean or median. Default = Median
    scaling: str
        Output as spectral density or spectrum. Default = Spectral density
    is_onesided: Bool
        Select one-sided or two-sided spectrum. Default=True

    fil: np.ndarray
        Finite Impulse Response (FIR) of the Correlation filter
    noise: np.ndarray
        Time series of the noise signal generated from the PSD
    truncate: bool for truncating the signal

    Methods
    -------
    correlation_filter(M):
        Calculates the finite impulse response for a given filter length
    filter_matrix(N):
        Calculates the time series of noise for a given signal length
    psd_welch(x, n)
        Calculates the PSD of the given time series vector
    """

    def __init__(self, psd, fs):
        """

        Constructs the attributes to hold the noise characteristics
        :param psd:
            Desired PSD of the noise
        :param fs:
            Sampling frequency of the noise
        """
        self.desired_psd = psd / 2  # PSD provided as a one-sided spectrum. Convert to two-sided spectrum
        self.fs = fs  # Sampling frequency
        self.frequency_vector = None  # Frequency domain

        # Default settings for Welch's method. Remains same for analysis of all signals
        self.window = "hann"  # Window
        self.average_method = "median"  # Periodogram averaging method
        self.scaling = 'density'
        self.is_onesided = True  # Bool for one or two-sided spectrum

        self.cor_fil = None  # Finite Impulse response of the Correlation filter
        self.noise = None  # Time series of the generated noise signal
        self.truncate = True  # Bool for truncating the signal. Default: True

    def correlation_filter(self, M: int):
        """
            Returns the impulse response of a desired PSD filter by taking the IFFT.
            Filter must be of odd length. An N point time domain running from 0 to N-1 signal generates (N/2 + 1) frequencies running from 0 to N/2.
            Frequency 0 is the DC mean and frequency 0.5 at N/2 is half the sampling rate (Nyquist frequency).

            :param M:
                Desired length of the filter in time domain
            :return:
                Sets the impulse response of the filter centered around sample 0
            :raises:
                ValueError: If the output length is an Even number
            """

        # Check if the filter contains an odd number of points. So that it can be symmetrically centered
        if M % 2 == 0:
            raise ValueError("Filter length must be an odd number!")

        fil = irfft(np.sqrt(self.desired_psd), M)  # Returns the impulse response from sample 0 to N-1

        # Center the impulse response at sample (N-1)/2
        fil = np.hstack((fil[ceil(M / 2):], fil[0:ceil(M / 2)]))

        return fil

    def filter_matrix(self, fil, X):
        """
        :param fil:
            Finite impulse response of the correlation filter
        :param X:
            Input array/matrix to be convolved with the filter
        :return:
            Time series of the noise truncated to the desired output length
        """
        if len(fil) % 2 == 0:
            raise ValueError("Filter length must be an odd number")

        try:
            # Set the length of the convolution output
            length = X.shape[0] + len(fil) - 1

            # Calculate the number of points for the FFT. Convolution of an N point signal with an M point signal results in an N+M-1 output.
            # Round it up to the nearest power of 2 for FFT.

            NFFT = 2 ** ceil(np.log2(length))

            # Generate the FFT of the Correlation filter with an NFFT point FFT
            F = rfft(fil, NFFT)

            # Generate the output time domain signal array
            y = np.zeros((NFFT, X.shape[1]))
            # Perform convolution via FFT of white noise and the filter. Convolution in time domain is multiplication in frequency domain
            for i in range(X.shape[1]):  # Go through the columns of the input Array/Matrix
                if sum(abs(X[:, i])) > 0:  # If the column is 0, skip the convolution to save time
                    y[:, i] = irfft(F * rfft(X[:, i], NFFT))

            if self.truncate:
                # Calculate the offset to truncate the signal to the length of the white noise input
                offset = floor(len(fil) / 2)
                y = y[offset + 1: X.shape[0] + offset + 1, :]

                # Set the end tails to 0 as the convolution is unstable at those points. Basically set the warm-up and cool down to 0
                y[0:offset + 1, :] = 0
                y[len(y) - offset:, :] = 0

                # Truncate the signal to the required output length by trimming away the tail 0s
                y = y[offset + 1: len(y) - offset + 1, :]

        except IndexError or AttributeError:
            # Set the length of the convolution output
            length = len(X) + len(fil) - 1

            # Calculate the number of points for the FFT. Convolution of an N point signal with an M point signal results in an N+M-1 output.
            # Round it up to the nearest power of 2 for FFT.

            NFFT = 2 ** ceil(np.log2(length))
            # Generate the FFT of the Correlation filter with an NFFT point FFT
            F = rfft(fil, NFFT)

            # Perform convolution via FFT of white noise and the filter. Convolution in time domain is multiplication in frequency domain
            y = irfft(F * rfft(X, NFFT))

            if self.truncate:
                # Calculate the offset to truncate the signal to the length of the white noise input
                offset = floor(len(fil) / 2)
                y = y[offset + 1: len(X) + offset + 1]

                # Set the end tails to 0 as the convolution is unstable at those points. Basically set the warm-up and cool down to 0
                y[0:offset + 1] = 0
                y[len(y) - offset:] = 0

                # Truncate the signal to the required output length by trimming away the tail 0s
                y = y[offset + 1: len(y) - offset + 1]

        return y

    def psd_welch(self, x, nperseg):
        """
        :param x: time series vector (1xN)
        :param nperseg: Number of points in the averaging window (1xnperseg)
        :return:
            ff: Frequency vector
            P: The Power spectral density of the time series vector
        """
        # Overlap by default is 50 %
        return welch(x, self.fs, self.window, nperseg, noverlap=None, return_onesided=self.is_onesided, scaling=self.scaling,
                     average=self.average_method)

    def decorrelation_filter(self, x, N):
        # Calculate the PSD of the input signal. This calculates one-sided PSD. But irfft takes two-sided PSD. Divide by 2 to get the two-sided PSD
        f, p = self.psd_welch(x, N)
        p = p / 2
        pf = np.reciprocal(p)
        pf[0] = 0
        fil = irfft(np.sqrt(pf), N)
        fil = np.hstack((fil[ceil(N / 2):], fil[0:ceil(N / 2)]))

        return fil


class Instrument(Noise):
    def __init__(self, psd, fs):
        super().__init__(psd, fs)


class Accelerometer(Noise):
    """
    A class to process all accelerometer related operations. Inherits 'Noise' class to handle noise generation
    Attributes:
    ----------
    id: int
        Identification number of the accelerometer
    Vxx: np.ndarray (N,)
        Gravity x gradient in x direction
    Vyy: np.ndarray (N,)
        Gravity y gradient in y direction
    Vzz: np.ndarray (N,)
        Gravity z gradient in z direction
    Vxy: np.ndarray (N,)
        Gravity x gradient in y direction
    Vxz: np.ndarray (N,)
        Gravity x gradient in z direction
    Vyz: np.ndarray (N,)
        Gravity y gradient in z direction
    w: np.ndarray (N, 3)
        Angular velocity vector
    dw: np.ndarray (N, 3)
        Angular acceleration vector
    a_ng: np.ndarray (N, 3)
        Non-gravitational acceleration vector
    r: np.ndarray (3, )
        Position of the accelerometer in the s/c
    M: np.ndarray (3, 3)
        Calibration matrix ICM
    K: np.ndarray (3, 3)
        Quadratic factor matrix
    W: np.ndarray (3, 3)
        Angular acceleration coupling matrix. Note: Have to assign it manually after class creation or else it will throw an error when invoking
        the methods!
    dr: np.ndarray (3, )
        Position deviation of the accelerometer
    a_np: np.ndarray(N, 3)
        Variable to store acceleration sensed at nominal position
    a_real: np.ndarray(N, 3)
        Variable to store acceleration sensed with accelerometer imperfections
    a_meas: np.ndarray(N, 3)
        Variable to store acceleration sensed with accelerometer imperfections and noise

    Methods
    -------
    get_nominal_acc():
        Calculate the acceleration at nominal position
    get_realistic_acc():
        Calculate the acceleration with accelerometer imperfections
    get_measured_acc():
        Calculate the acceleration with accelerometer imperfections and noise
    """

    def __init__(self, acc_id, V_arr, w_lst, dw_lst, a_ng, position, noise_psd, fs, **kwargs):
        self.id = acc_id
        # Construct the matrices
        self.Vxx = V_arr[:, 0]
        self.Vyy = V_arr[:, 1]
        self.Vzz = V_arr[:, 2]
        self.Vxy = V_arr[:, 3]
        self.Vxz = V_arr[:, 4]
        self.Vyz = V_arr[:, 5]

        self.w_nom = w_lst[0]
        self.dw_nom = dw_lst[0]

        self.a_ng = a_ng

        # Nominal position of the accelerometer
        self.r = position

        # Create the imperfections in accelerometer (without the noise)
        self.M = np.random.randn(3, 3) * 1e-3 + np.eye(3)
        self.K = np.identity(3) * np.random.randn(3) * 10
        self.W = None  # Assign it manually
        self.dr = None  # Assign it manually

        # TODO: assign the correct values to the w_meas variables
        self.w_meas = w_lst[1]
        self.dw_meas = dw_lst[1]
        self.w_noise = self.w_meas - self.w_nom
        self.dw_noise = self.dw_meas - self.dw_nom

        super().__init__(noise_psd, fs)

        # Variable to store the acceleration time series
        self.a_nom = None
        self.a_real = None
        self.a_meas = None
        self.a_gg_only = None
        self.a_gg_w_rate_only = None

        # Generate G = -(V-W^2-dW) column wise
        self.G1_nom = -np.array([self.Vxx + self.w_nom[:, 1] ** 2 + self.w_nom[:, 2] ** 2,
                                 self.Vxy - (self.w_nom[:, 0] * self.w_nom[:, 1]) - self.dw_nom[:, 2],
                                 self.Vxz - (self.w_nom[:, 0] * self.w_nom[:, 2]) + self.dw_nom[:, 1]]).T

        self.G2_nom = -np.array([self.Vxy - (self.w_nom[:, 0] * self.w_nom[:, 1]) + self.dw_nom[:, 2],
                                 self.Vyy + self.w_nom[:, 0] ** 2 + self.w_nom[:, 2] ** 2,
                                 self.Vyz - (self.w_nom[:, 1] * self.w_nom[:, 2]) - self.dw_nom[:, 0]]).T

        self.G3_nom = -np.array([self.Vxz - (self.w_nom[:, 0] * self.w_nom[:, 2]) - self.dw_nom[:, 1],
                                 self.Vyz - (self.w_nom[:, 1] * self.w_nom[:, 2]) + self.dw_nom[:, 0],
                                 self.Vzz + self.w_nom[:, 0] ** 2 + self.w_nom[:, 1] ** 2]).T

        # Generate G from measured angular rate data
        self.G1_meas = -np.array([self.Vxx + self.w_meas[:, 1] ** 2 + self.w_meas[:, 2] ** 2,
                                  self.Vxy - (self.w_meas[:, 0] * self.w_meas[:, 1]) - self.dw_meas[:, 2],
                                  self.Vxz - (self.w_meas[:, 0] * self.w_meas[:, 2]) + self.dw_meas[:, 1]]).T

        self.G2_meas = -np.array([self.Vxy - (self.w_meas[:, 0] * self.w_meas[:, 1]) + self.dw_meas[:, 2],
                                  self.Vyy + self.w_meas[:, 0] ** 2 + self.w_meas[:, 2] ** 2,
                                  self.Vyz - (self.w_meas[:, 1] * self.w_meas[:, 2]) - self.dw_meas[:, 0]]).T

        self.G3_meas = -np.array([self.Vxz - (self.w_meas[:, 0] * self.w_meas[:, 2]) - self.dw_meas[:, 1],
                                  self.Vyz - (self.w_meas[:, 1] * self.w_meas[:, 2]) + self.dw_meas[:, 0],
                                  self.Vzz + self.w_meas[:, 0] ** 2 + self.w_meas[:, 1] ** 2]).T

        self.G1_nom_gg_only = -np.array([self.Vxx,
                                         self.Vxy,
                                         self.Vxz]).T

        self.G2_nom_gg_only = -np.array([self.Vxy,
                                         self.Vyy,
                                         self.Vyz]).T

        self.G3_nom_gg_only = -np.array([self.Vxz,
                                         self.Vyz,
                                         self.Vzz]).T

        self.G1_nom_gg_w_rate_only = -np.array([self.Vxx + self.w_nom[:, 1] ** 2 + self.w_nom[:, 2] ** 2,
                                                self.Vxy - (self.w_nom[:, 0] * self.w_nom[:, 1]),
                                                self.Vxz - (self.w_nom[:, 0] * self.w_nom[:, 2])]).T

        self.G2_nom_gg_w_rate_only = -np.array([self.Vxy - (self.w_nom[:, 0] * self.w_nom[:, 1]),
                                                self.Vyy + self.w_nom[:, 0] ** 2 + self.w_nom[:, 2] ** 2,
                                                self.Vyz - (self.w_nom[:, 1] * self.w_nom[:, 2])]).T

        self.G3_nom_gg_w_rate_only = -np.array([self.Vxz - (self.w_nom[:, 0] * self.w_nom[:, 2]),
                                                self.Vyz - (self.w_nom[:, 1] * self.w_nom[:, 2]),
                                                self.Vzz + self.w_nom[:, 0] ** 2 + self.w_nom[:, 1] ** 2]).T

        self.G1_meas_w_rate_only = -np.array([+ self.w_meas[:, 1] ** 2 + self.w_meas[:, 2] ** 2,
                                              - (self.w_meas[:, 0] * self.w_meas[:, 1]),
                                              - (self.w_meas[:, 0] * self.w_meas[:, 2])]).T

        self.G2_meas_w_rate_only = -np.array([- (self.w_meas[:, 0] * self.w_meas[:, 1]),
                                              + self.w_meas[:, 0] ** 2 + self.w_meas[:, 2] ** 2,
                                              - (self.w_meas[:, 1] * self.w_meas[:, 2])]).T

        self.G3_meas_w_rate_only = -np.array([-(self.w_meas[:, 0] * self.w_meas[:, 2]),
                                              - (self.w_meas[:, 1] * self.w_meas[:, 2]),
                                              + self.w_meas[:, 0] ** 2 + self.w_meas[:, 1] ** 2]).T

        self.G1_meas_dw_only = np.zeros((len(self.Vxx), 3))
        self.G2_meas_dw_only = np.zeros((len(self.Vxx), 3))
        self.G3_meas_dw_only = np.zeros((len(self.Vxx), 3))

        self.G1_meas_dw_only[:, 1] = -self.dw_meas[:, 2]
        self.G1_meas_dw_only[:, 2] = self.dw_meas[:, 1]
        self.G1_meas_dw_only = -self.G1_meas_dw_only

        self.G2_meas_dw_only[:, 0] = self.dw_meas[:, 2]
        self.G2_meas_dw_only[:, 2] = -self.dw_meas[:, 0]
        self.G2_meas_dw_only = -self.G2_meas_dw_only

        self.G3_meas_dw_only[:, 0] = -self.dw_meas[:, 1]
        self.G3_meas_dw_only[:, 1] = self.dw_meas[:, 0]
        self.G3_meas_dw_only = -self.G3_meas_dw_only

    def get_acc_gg_only(self):
        # 1) Apply the position offset (True acceleration at real accelerometer positions
        a = self.G1_nom_gg_only * (self.r[0] + self.dr[0]) + self.G2_nom_gg_only * (self.r[1] + self.dr[1]) + self.G3_nom_gg_only * (
                self.r[2] + self.dr[2])
        # 2) Apply accelerometer imperfections
        a = (a @ self.M.T) + (a ** 2 @ self.K.T) + (self.dw_nom @ self.W.T)

        return a

    def get_acc_gg_w_rate_only(self):
        # 1) Apply the position offset (True acceleration at real accelerometer positions
        a = self.G1_nom_gg_w_rate_only * (self.r[0] + self.dr[0]) + self.G2_nom_gg_w_rate_only * (self.r[1] + self.dr[1]) + self.G3_nom_gg_w_rate_only \
            * (self.r[2] + self.dr[2])
        # 2) Apply accelerometer imperfections
        a = (a @ self.M.T) + (a ** 2 @ self.K.T) + (self.dw_nom @ self.W.T)

        return a

    def get_acc_G_only(self):
        # 1) Apply the position offset (True acceleration at real accelerometer positions
        a = self.G1_nom * (self.r[0] + self.dr[0]) + self.G2_nom * (self.r[1] + self.dr[1]) + self.G3_nom * (self.r[2] + self.dr[2])
        # 2) Apply accelerometer imperfections
        a = (a @ self.M.T) + (a ** 2 @ self.K.T) + (self.dw_nom @ self.W.T)

        return a

    def get_nominal_acc(self):
        """
        :return: Acceleration sensed by a perfect accelerometer at nominal position (N, 3)
        """
        return self.G1_nom * self.r[0] + self.G2_nom * self.r[1] + self.G3_nom * self.r[2] + self.a_ng

    def get_realistic_acc(self):
        """
        :return: Acceleration sensed with accelerometer imperfections (N, 3)
        """
        # Check if the position deviation and angular acceleration coupling matrix is provided!
        assert self.dr is not None
        assert self.W is not None

        # 1) Apply the position offset (True acceleration at real accelerometer positions
        a = self.G1_nom * (self.r[0] + self.dr[0]) + self.G2_nom * (self.r[1] + self.dr[1]) + self.G3_nom * (self.r[2] + self.dr[2]) + self.a_ng
        # 2) Apply accelerometer imperfections
        a = (a @ self.M.T) + (a ** 2 @ self.K.T) + (self.dw_nom @ self.W.T)

        return a

    def get_measured_acc(self):
        """
        :return: Acceleration sensed with accelerometer imperfections and noise (N, 3)
        """
        return self.a_real + self.noise

    def get_rotational_acceleration_nominal(self):
        elem1 = -np.array([+ self.w_nom[:, 1] ** 2 + self.w_nom[:, 2] ** 2,
                           - (self.w_nom[:, 0] * self.w_nom[:, 1]) - self.dw_nom[:, 2],
                           - (self.w_nom[:, 0] * self.w_nom[:, 2]) + self.dw_nom[:, 1]]).T

        elem2 = -np.array([- (self.w_nom[:, 0] * self.w_nom[:, 1]) + self.dw_nom[:, 2],
                           + self.w_nom[:, 0] ** 2 + self.w_nom[:, 2] ** 2,
                           - (self.w_nom[:, 1] * self.w_nom[:, 2]) - self.dw_nom[:, 0]]).T

        elem3 = -np.array([- (self.w_nom[:, 0] * self.w_nom[:, 2]) - self.dw_nom[:, 1],
                           - (self.w_nom[:, 1] * self.w_nom[:, 2]) + self.dw_nom[:, 0],
                           + self.w_nom[:, 0] ** 2 + self.w_nom[:, 1] ** 2]).T
        # a = elem1 * (self.r[0] + self.dr[0]) + elem2 * (self.r[1] + self.dr[1]) + elem3 * (self.r[2] + self.dr[2])
        a = self.G1_nom * (self.r[0]) + self.G2_nom * (self.r[1]) + self.G3_nom * (self.r[2])

        a = (a @ self.M.T) + (a ** 2 @ self.K.T) + (self.dw_nom @ self.W.T)
        return a

    def get_rotational_acceleration_noisy(self):
        elem1 = -np.array([+ self.w_meas[:, 1] ** 2 + self.w_meas[:, 2] ** 2,
                           - (self.w_meas[:, 0] * self.w_meas[:, 1]) - self.dw_meas[:, 2],
                           - (self.w_meas[:, 0] * self.w_meas[:, 2]) + self.dw_meas[:, 1]]).T

        elem2 = -np.array([- (self.w_meas[:, 0] * self.w_meas[:, 1]) + self.dw_meas[:, 2],
                           + self.w_meas[:, 0] ** 2 + self.w_meas[:, 2] ** 2,
                           - (self.w_meas[:, 1] * self.w_meas[:, 2]) - self.dw_meas[:, 0]]).T

        elem3 = -np.array([- (self.w_meas[:, 0] * self.w_meas[:, 2]) - self.dw_meas[:, 1],
                           - (self.w_meas[:, 1] * self.w_meas[:, 2]) + self.dw_meas[:, 0],
                           + self.w_meas[:, 0] ** 2 + self.w_meas[:, 1] ** 2]).T

        # a = elem1 * (self.r[0] + self.dr[0]) + elem2 * (self.r[1] + self.dr[1]) + elem3 * (self.r[2] + self.dr[2])

        a = self.G1_meas * (self.r[0]) + self.G2_meas * (self.r[1]) + self.G3_meas * (self.r[2])
        return a


def hann_window(N):
    """
    Returns the Hann window of length N
    :param N:
        Length of the window
    :return:
        Hann window of length N
    """
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))


def bandpass_filter(NFFT, fc1, fc2):
    """
    Returns the impulse response of a desired bandpass filter by taking the IFFT. Frequencies must be provided as a fraction of sampling rate

    :param NFFT:
        Desired length of the filter in time domain
    :param fc1:
        Beginning of the bandpass filter
    :param fc2:
        End of the bandpass filter
    :return:
        Impulse response of the filter centered around sample 0
    """
    N = NFFT
    if N % 2 == 0:
        raise ValueError("Filter length must be an odd number!")

    f = rfftfreq(N, 1)
    # Define the ideal pass band
    p = np.zeros(len(f))
    idx = np.where((f > fc1) & (f < fc2))[0]
    p[idx] = 1

    hann = hann_window(9)
    p = convolve(p, hann / np.sum(hann), mode='same')
    p[0: ceil(len(hann) / 2)] = 0
    p = p[0:len(f)]

    fil_bp = irfft(np.sqrt(p), N)

    # Center the impulse response at sample (N-1)/2
    fil_bp = np.hstack((fil_bp[ceil(N / 2):], fil_bp[0:ceil(N / 2)]))

    return fil_bp


def integrate_trapezoid(x):
    y = np.zeros(x.shape)
    mx = np.mean(x, axis=0)
    for i in range(x.shape[1]):
        for n in range(1, x.shape[0]):
            y[n, i] = y[n - 1, i] + (x[n, i] + x[n - 1, i]) / 2 - mx[i]

    return y


def create_configuration(V, w, dw, a_ng, pos_dict, noise_psd, fs, filter_length, layout, noise_switch=True):
    """
    Create an accelerometer object with the given parameters
    :param V:
        Gravity gradient vector
    :param w:
        Angular velocity vector
    :param dw:
        Angular acceleration vector
    :param a_ng:
        Non-gravitational acceleration vector
    :param pos_dict:
        Positions of the accelerometers
    :param noise_psd:
        Noise PSD of the accelerometer
    :param fs:
        Sampling frequency
    :param filter_length:
        Length of the correlation filter
    :param layout:
        Layout of the accelerometers
    :param noise_switch:
        Bool to switch on/off the noise
    :return:
        Accelerometer objects
    """
    match layout:
        case 1:
            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            drc12 = np.zeros(3)
            drd12 = np.random.randn(3) * 1e-3

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = pos_dict["pos_acc2"]

            # Check on which axis the accelerometers are placed
            if pos_acc1[0] != 0:
                drd12[0] = 0
            elif pos_acc1[1] != 0:
                drd12[1] = 0
            else:
                drd12[2] = 0

            W_idx = [(1, 0), (2, 1), (1, 2)]  # W2 = -W1
            # Acc1
            acc1 = Accelerometer(1, V, w, dw, a_ng, pos_acc1, noise_psd, fs)
            acc1.W = np.zeros((3, 3))
            acc1.W[tuple(np.transpose(W_idx))] = 1e-4 * np.random.randn(3)
            acc1.dr = drc12 + drd12
            acc1.a_np = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            # Acc2
            acc2 = Accelerometer(2, V, w, dw, a_ng, pos_acc2, noise_psd, fs)
            acc2.W = -acc1.W.copy()
            acc2.dr = drc12 - drd12
            acc2.a_np = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            if noise_switch:
                acc1.a_meas = acc1.a_real + acc1.noise
                acc2.a_meas = acc2.a_real + acc2.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real

            return [acc1, acc2]

        case 2:
            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc3 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            drc13 = np.random.randn(3) * 1e-3
            drd13 = np.random.randn(3) * 1e-3

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = np.zeros(3)
            pos_acc3 = pos_dict["pos_acc3"]

            # Check on which axis the accelerometers are placed
            if pos_acc1[0] != 0:
                drd13[0] = 0
            elif pos_acc1[1] != 0:
                drd13[1] = 0
            else:
                drd13[2] = 0

            W_idx = [(1, 0), (2, 1), (1, 2)]  # W2 = 0
            # Acc1
            acc1 = Accelerometer(1, V, w, dw, a_ng, pos_acc1, noise_psd, fs)
            acc1.W = np.zeros((3, 3))
            acc1.W[tuple(np.transpose(W_idx))] = 1e-4 * np.random.randn(3)
            acc1.dr = drc13 + drd13
            acc1.a_np = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            # Acc2
            acc2 = Accelerometer(2, V, w, dw, a_ng, pos_acc2, noise_psd, fs)
            acc2.W = np.zeros((3, 3))
            acc2.dr = np.zeros(3)
            acc2.a_np = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            # Acc3
            acc3 = Accelerometer(3, V, w, dw, a_ng, pos_acc3, noise_psd, fs)
            acc3.W = np.zeros((3, 3))
            acc3.W[tuple(np.transpose(W_idx))] = 1e-4 * np.random.randn(3)
            acc3.dr = drc13 - drd13
            acc3.a_np = acc3.get_nominal_acc()
            acc3.a_real = acc3.get_realistic_acc()
            acc3.cor_fil = acc3.correlation_filter(filter_length)
            acc3.noise = acc3.filter_matrix(acc3.cor_fil, white_noise_acc3)

            if noise_switch:
                acc1.a_meas = acc1.a_real + acc1.noise
                acc2.a_meas = acc2.a_real + acc2.noise
                acc3.a_meas = acc3.a_real + acc3.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real
                acc3.a_meas = acc3.a_real

            return [acc1, acc2, acc3]

        case 3:
            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc3 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc4 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            drc13 = np.random.randn(3) * 1e-3
            drd13 = np.random.randn(3) * 1e-3
            drc24 = np.zeros(3)
            drd24 = np.random.randn(3) * 1e-3

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = pos_dict["pos_acc2"]
            pos_acc3 = pos_dict["pos_acc3"]
            pos_acc4 = pos_dict["pos_acc4"]

            # Check on which axis the accelerometers are placed
            if pos_acc1[0] != 0:
                drd13[0] = 0
            elif pos_acc1[1] != 0:
                drd13[1] = 0
            else:
                drd13[2] = 0

            if pos_acc2[0] != 0:
                drd24[0] = 0
            elif pos_acc2[1] != 0:
                drd24[1] = 0
            else:
                drd24[2] = 0

            W_idx = [(1, 0), (2, 1), (1, 2)]  # W4 = -W2
            # Acc1
            acc1 = Accelerometer(1, V, w, dw, a_ng, pos_acc1, noise_psd, fs)
            acc1.W = np.zeros((3, 3))

            acc1.W[tuple(np.transpose(W_idx))] = 1e-4 * np.random.randn(3)
            acc1.dr = drc13 + drd13
            acc1.a_np = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            # Acc3
            acc3 = Accelerometer(3, V, w, dw, a_ng, pos_acc3, noise_psd, fs)
            acc3.W = np.zeros((3, 3))
            acc3.W[tuple(np.transpose(W_idx))] = 1e-4 * np.random.randn(3)
            acc3.dr = drc13 - drd13
            acc3.a_np = acc3.get_nominal_acc()
            acc3.a_real = acc3.get_realistic_acc()
            acc3.cor_fil = acc3.correlation_filter(filter_length)
            acc3.noise = acc3.filter_matrix(acc3.cor_fil, white_noise_acc3)

            # Acc2
            acc2 = Accelerometer(2, V, w, dw, a_ng, pos_acc2, noise_psd, fs)
            acc2.W = np.zeros((3, 3))
            acc2.W[tuple(np.transpose(W_idx))] = 1e-4 * np.random.randn(3)
            acc2.dr = drc24 + drd24
            acc2.a_np = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            # Acc4
            acc4 = Accelerometer(4, V, w, dw, a_ng, pos_acc4, noise_psd, fs)
            acc4.W = -acc2.W.copy()
            acc4.dr = drc24 - drd24
            acc4.a_np = acc4.get_nominal_acc()
            acc4.a_real = acc4.get_realistic_acc()
            acc4.cor_fil = acc4.correlation_filter(filter_length)
            acc4.noise = acc4.filter_matrix(acc4.cor_fil, white_noise_acc4)

            if noise_switch:
                acc1.a_meas = acc1.a_real + acc1.noise
                acc2.a_meas = acc2.a_real + acc2.noise
                acc3.a_meas = acc3.a_real + acc3.noise
                acc4.a_meas = acc4.a_real + acc4.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real
                acc3.a_meas = acc3.a_real
                acc4.a_meas = acc4.a_real

            return [acc1, acc2, acc3, acc4]


def get_expected_noise(Acc_lst, layout):
    """
    Get the expected noise from the accelerometers
    :param Acc_lst:
        List of accelerometer objects
    :param layout:
        Layout of the accelerometers
    :return:
        Expected noise from the accelerometers
    """
    match layout:
        case 1:
            acc1 = Acc_lst[0]
            acc2 = Acc_lst[1]
            noise_diff = (acc1.noise - acc2.noise) / 2 + (acc1.G1_meas - acc1.G1_nom) * (acc1.r[0] - acc2.r[0]) / 2 + \
                         (acc1.G2_meas - acc1.G2_nom) * (acc1.r[1] - acc2.r[1]) / 2 + (acc1.G3_meas - acc1.G3_nom) * (acc1.r[2] - acc2.r[2]) / 2 + \
                         0.5 * (acc1.noise + (acc1.G1_meas - acc1.G1_nom) * acc1.dr[0] + (acc1.G2_meas - acc1.G2_nom) * acc1.dr[1] +
                                (acc1.G3_meas - acc1.G3_nom) * acc1.dr[2]) ** 2 + \
                         0.5 * (acc2.noise + (acc2.G1_meas - acc2.G1_nom) * acc2.dr[0] + (acc2.G2_meas - acc2.G2_nom) * acc2.dr[1] +
                                (acc2.G3_meas - acc2.G3_nom) * acc2.dr[2]) ** 2 + \
                         acc1.dw_noise @ (acc1.W - acc2.W).T / 2

            return [noise_diff]
        case 2:
            acc1 = Acc_lst[0]
            acc2 = Acc_lst[1]
            acc3 = Acc_lst[2]

            noise_diff = (acc1.noise - acc3.noise) / 2 + (acc1.G1_meas - acc1.G1_nom) * (acc1.r[0] + acc1.dr[0] - acc3.r[0] - acc3.dr[0]) / 2 + \
                         (acc1.G2_meas - acc1.G2_nom) * (acc1.r[1] + acc1.dr[1] - acc3.r[1] - acc3.dr[1]) / 2 + (acc1.G3_meas - acc1.G3_nom) * (
                                 acc1.r[2] + acc1.dr[2] - acc3.r[2] - acc3.dr[2]) / 2 + \
                         0.5 * (acc1.noise + (acc1.G1_meas - acc1.G1_nom) * (acc1.r[0] + acc1.dr[0]) + (acc1.G2_meas - acc1.G2_nom) * (
                    acc1.r[1] + acc1.dr[1]) +
                                (acc1.G3_meas - acc1.G3_nom) * (acc1.r[2] + acc1.dr[2])) ** 2 + \
                         0.5 * (acc3.noise + (acc3.G1_meas - acc3.G1_nom) * acc3.dr[0] + (acc3.G2_meas - acc3.G2_nom) * acc3.dr[1] +
                                (acc3.G3_meas - acc3.G3_nom) * acc3.dr[2]) ** 2 + \
                         acc1.dw_noise @ (acc1.W - acc3.W).T / 2

            noise_com = acc2.noise - (acc1.noise + acc3.noise) / 2 - \
                        (acc1.G1_meas - acc1.G1_nom) * (acc1.dr[0] + acc3.dr[0]) / 2 - \
                        (acc1.G2_meas - acc1.G2_nom) * (acc1.dr[1] + acc3.dr[1]) / 2 - \
                        (acc1.G3_meas - acc1.G3_nom) * (acc1.dr[2] + acc3.dr[2]) / 2 - \
                        0.5 * (acc1.noise + (acc1.G1_meas - acc1.G1_nom) * (acc1.r[0] + acc1.dr[0]) + (acc1.G2_meas - acc1.G2_nom) * (
                    acc1.r[1] + acc1.dr[1]) + (
                                       acc1.G3_meas - acc1.G3_nom) * (acc1.r[2] + acc1.dr[2])) ** 2 - \
                        0.5 * (acc3.noise + (acc3.G1_meas - acc3.G1_nom) * (acc3.r[0] + acc3.dr[0]) + (acc3.G2_meas - acc3.G2_nom) * (
                    acc3.r[1] + acc3.dr[1]) + (
                                       acc3.G3_meas - acc3.G3_nom) * (acc3.r[2] + acc3.dr[2])) ** 2 - \
                        acc1.dw_noise @ (acc1.W + acc3.W).T / 2 + acc2.dw_noise @ acc2.W.T

            return [noise_diff, noise_com]

        case 3:
            acc1 = Acc_lst[0]
            acc2 = Acc_lst[1]
            acc3 = Acc_lst[2]
            acc4 = Acc_lst[3]

            noise_diff13 = (acc1.noise - acc3.noise) / 2 + (acc1.G1_meas - acc1.G1_nom) * (acc1.r[0] - acc3.r[0]) / 2 + \
                           (acc1.G2_meas - acc1.G2_nom) * (acc1.r[1] - acc3.r[1]) / 2 + (acc1.G3_meas - acc1.G3_nom) * (acc1.r[2] - acc3.r[2]) / 2 + \
                           0.5 * (acc1.noise + (acc1.G1_meas - acc1.G1_nom) * acc1.dr[0] + (acc1.G2_meas - acc1.G2_nom) * acc1.dr[1] +
                                  (acc1.G3_meas - acc1.G3_nom) * acc1.dr[2]) ** 2 + \
                           0.5 * (acc3.noise + (acc3.G1_meas - acc3.G1_nom) * acc3.dr[0] + (acc3.G2_meas - acc3.G2_nom) * acc3.dr[1] +
                                  (acc3.G3_meas - acc3.G3_nom) * acc3.dr[2]) ** 2 + \
                           acc1.dw_noise @ (acc1.W - acc3.W).T / 2

            noise_diff24 = (acc2.noise - acc4.noise) / 2 + (acc2.G1_meas - acc2.G1_nom) * (acc2.r[0] - acc4.r[0]) / 2 + \
                           (acc2.G2_meas - acc2.G2_nom) * (acc2.r[1] - acc4.r[1]) / 2 + (acc2.G3_meas - acc2.G3_nom) * (acc2.r[2] - acc4.r[2]) / 2 + \
                           0.5 * (acc2.noise + (acc2.G1_meas - acc2.G1_nom) * acc2.dr[0] + (acc2.G2_meas - acc2.G2_nom) * acc2.dr[1] +
                                  (acc2.G3_meas - acc2.G3_nom) * acc2.dr[2]) ** 2 + \
                           0.5 * (acc4.noise + (acc4.G1_meas - acc4.G1_nom) * acc4.dr[0] + (acc4.G2_meas - acc4.G2_nom) * acc4.dr[1] +
                                  (acc4.G3_meas - acc4.G3_nom) * acc4.dr[2]) ** 2 + \
                           acc2.dw_noise @ (acc2.W - acc4.W).T / 2

            noise_com_13 = (acc1.noise + acc3.noise) / 2 - \
                           (acc1.G1_meas - acc1.G1_nom) * (acc1.dr[0] + acc3.dr[0]) / 2 - \
                           (acc1.G2_meas - acc1.G2_nom) * (acc1.dr[1] + acc3.dr[1]) / 2 - \
                           (acc1.G3_meas - acc1.G3_nom) * (acc1.dr[2] + acc3.dr[2]) / 2 - \
                           0.5 * (acc1.noise + (acc1.G1_meas - acc1.G1_nom) * acc1.dr[0] + (acc1.G2_meas - acc1.G2_nom) * acc1.dr[1] + (
                    acc1.G3_meas - acc1.G3_nom) * acc1.dr[2]) ** 2 - \
                           0.5 * (acc3.noise + (acc3.G1_meas - acc3.G1_nom) * acc3.dr[0] + (acc3.G2_meas - acc3.G2_nom) * acc3.dr[1] + (
                    acc3.G3_meas - acc3.G3_nom) * acc3.dr[2]) ** 2 - acc1.dw_noise @ (acc1.W + acc3.W).T / 2

            noise_com_24 = (acc2.noise + acc4.noise) / 2 - \
                           (acc2.G1_meas - acc2.G1_nom) * (acc2.dr[0] + acc4.dr[0]) / 2 - \
                           (acc2.G2_meas - acc2.G2_nom) * (acc2.dr[1] + acc4.dr[1]) / 2 - \
                           (acc2.G3_meas - acc2.G3_nom) * (acc2.dr[2] + acc4.dr[2]) / 2 - \
                           0.5 * (acc2.noise + (acc2.G1_meas - acc2.G1_nom) * acc2.dr[0] + (acc2.G2_meas - acc2.G2_nom) * acc2.dr[1] + (
                    acc2.G3_meas - acc2.G3_nom) * acc2.dr[2]) ** 2 - \
                           0.5 * (acc4.noise + (acc4.G1_meas - acc4.G1_nom) * acc4.dr[0] + (acc4.G2_meas - acc4.G2_nom) * acc4.dr[1] + (
                    acc4.G3_meas - acc4.G3_nom) * acc4.dr[2]) ** 2 - acc2.dw_noise @ (acc2.W + acc4.W).T / 2

            noise_com = noise_com_13 - noise_com_24

            return [noise_diff13, noise_diff24, noise_com]


def create_attitude_sensor_noise(angular_acceleration_psd, time_vector_length, filter_length):
    """
    Create the noise for the attitude sensor
    :param angular_acceleration_psd: Power spectral density of the angular acceleration noise
    :param time_vector_length: Length of the time vector
    :param filter_length: Length of the correlation filter
    :return:
    """
    dw_noise = Noise(angular_acceleration_psd, fs=1)
    dw_noise.cor_fil = dw_noise.correlation_filter(filter_length)

    dw_noise.noise = dw_noise.filter_matrix(dw_noise.cor_fil, np.random.randn(time_vector_length + filter_length - 1, 3))

    w_noise = integrate_trapezoid(dw_noise.noise)

    return w_noise, dw_noise.noise


def create_thrust_noise(noise_psd, time_vector, filter_length):
    """
    Create the noise for the thruster
    :param noise_psd: Power spectral density of the thruster noise
    :param time_vector: Length of the time vector
    :param filter_length: Correlation filter length
    :return:
    """

    white_noise = np.random.randn(len(time_vector) + filter_length - 1, 3)

    # Create thruster noise.
    # Smoothen the filter with a Hann window
    hann = hann_window(9)

    noise_psd = np.convolve(noise_psd, hann / np.sum(hann), mode='same')

    thruster_noise = Noise(noise_psd, fs=1)
    thruster_noise.cor_fil = thruster_noise.correlation_filter(filter_length)
    thruster_noise.noise = thruster_noise.filter_matrix(thruster_noise.cor_fil, white_noise)
    # # lowpass filter the thruster noise
    # filter_kernel = np.zeros(M+1)
    # for i in range(M):
    #     if i - M/2 == 0:
    #         filter_kernel[i] = 2*np.pi*fc
    #     else:
    #         filter_kernel[i] = np.sin(2*np.pi*fc*(i - M/2)) / (i - M/2)
    #         # Apply a blackman window
    #         filter_kernel[i] = filter_kernel[i] * (0.42 - 0.5 * np.cos(2*np.pi*i/M) + 0.08 * np.cos(4*np.pi*i/M))
    #
    # fig, ax = plt.subplots()
    # ax.plot(filter_kernel)
    # # Normalize the filter kernel
    # filter_kernel = filter_kernel / np.sum(filter_kernel)
    # # Apply the filter
    # thruster_noise.noise = thruster_noise.filter_matrix(filter_kernel, thruster_noise.noise)

    return thruster_noise


def create_thrust_series(linear_shaking_dict, angular_shaking_dict, frequency, filter_length, time_vector_length):
    """
    Create the thrust series for the linear and angular shaking
    :param linear_shaking_dict: Parameters for the linear shaking. Dictionary with keys 'x', 'y', 'z' and values [bool, amplitude, start frequency, end frequency]
    :param angular_shaking_dict: Parameters for the angular shaking. Dictionary with keys 'x', 'y', 'z' and values [bool, amplitude, start frequency, end frequency]
    :param frequency: Frequency vector to generate the PSD
    :param filter_length: Length of the correlation filter
    :param time_vector_length: Length of the time vector
    :return: Linear and angular acceleration time series
    """
    linear_shaking_psd = np.zeros((len(frequency), 3))
    angular_shaking_psd = np.zeros((len(frequency), 3))

    for i, (key, value) in enumerate(linear_shaking_dict.items()):
        condition_1 = (value[2] < frequency) & (frequency < value[3])
        condition_2 = frequency < value[2]
        condition_3 = frequency > value[3]
        linear_shaking_psd[condition_1, i] = value[1] ** 2
        linear_shaking_psd[condition_2, i] = (value[1] / 10) ** 2
        linear_shaking_psd[condition_3, i] = np.interp(frequency[condition_3], [value[3], 0.5], [value[1] / 10, 1e-10]) ** 2
        # Set the DC component to zero
        linear_shaking_psd[0] = 0

    for i, (key, value) in enumerate(angular_shaking_dict.items()):
        condition_1 = (value[2] < frequency) & (frequency < value[3])
        condition_2 = frequency < value[2]
        condition_3 = frequency > value[3]
        angular_shaking_psd[condition_1, i] = value[1] ** 2
        angular_shaking_psd[condition_2, i] = (value[1] / 10) ** 2
        angular_shaking_psd[condition_3, i] = np.interp(frequency[condition_3], [value[3], 0.5], [value[1] / 10, 1e-10]) ** 2
        # Set the DC component to zero
        angular_shaking_psd[0] = 0

    # Smoothen the filter with a Hann window
    hann = hann_window(9)
    for i in range(1):
        for j in range(3):
            linear_shaking_psd[:, j] = np.convolve(linear_shaking_psd[:, j], hann / np.sum(hann), mode='same')
            angular_shaking_psd[:, j] = np.convolve(angular_shaking_psd[:, j], hann / np.sum(hann), mode='same')

    linear_shaking_x = Noise(linear_shaking_psd[:, 0], fs=1)
    linear_shaking_y = Noise(linear_shaking_psd[:, 1], fs=1)
    linear_shaking_z = Noise(linear_shaking_psd[:, 2], fs=1)

    angular_shaking_x = Noise(angular_shaking_psd[:, 0], fs=1)
    angular_shaking_y = Noise(angular_shaking_psd[:, 1], fs=1)
    angular_shaking_z = Noise(angular_shaking_psd[:, 2], fs=1)

    linear_shaking_x.cor_fil = linear_shaking_x.correlation_filter(filter_length)
    linear_shaking_y.cor_fil = linear_shaking_y.correlation_filter(filter_length)
    linear_shaking_z.cor_fil = linear_shaking_z.correlation_filter(filter_length)
    angular_shaking_x.cor_fil = angular_shaking_x.correlation_filter(filter_length)
    angular_shaking_y.cor_fil = angular_shaking_y.correlation_filter(filter_length)
    angular_shaking_z.cor_fil = angular_shaking_z.correlation_filter(filter_length)

    if linear_shaking_dict['x'][0]:
        linear_shaking_x.noise = linear_shaking_x.filter_matrix(linear_shaking_x.cor_fil, np.random.randn(time_vector_length + filter_length - 1, 1))
    else:
        linear_shaking_x.noise = np.zeros((time_vector_length, 1))

    if linear_shaking_dict['y'][0]:
        linear_shaking_y.noise = linear_shaking_y.filter_matrix(linear_shaking_y.cor_fil, np.random.randn(time_vector_length + filter_length - 1, 1))
    else:
        linear_shaking_y.noise = np.zeros((time_vector_length, 1))

    if linear_shaking_dict['z'][0]:
        linear_shaking_z.noise = linear_shaking_z.filter_matrix(linear_shaking_z.cor_fil, np.random.randn(time_vector_length + filter_length - 1, 1))
    else:
        linear_shaking_z.noise = np.zeros((time_vector_length, 1))

    if angular_shaking_dict['x'][0]:
        angular_shaking_x.noise = angular_shaking_x.filter_matrix(angular_shaking_x.cor_fil,
                                                                  np.random.randn(time_vector_length + filter_length - 1, 1))
    else:
        angular_shaking_x.noise = np.zeros((time_vector_length, 1))

    if angular_shaking_dict['y'][0]:
        angular_shaking_y.noise = angular_shaking_y.filter_matrix(angular_shaking_y.cor_fil,
                                                                  np.random.randn(time_vector_length + filter_length - 1, 1))
    else:
        angular_shaking_y.noise = np.zeros((time_vector_length, 1))

    if angular_shaking_dict['z'][0]:
        angular_shaking_z.noise = angular_shaking_z.filter_matrix(angular_shaking_z.cor_fil,
                                                                  np.random.randn(time_vector_length + filter_length - 1, 1))
    else:
        angular_shaking_z.noise = np.zeros((time_vector_length, 1))

    linear_acceleration_shaking = np.hstack((linear_shaking_x.noise, linear_shaking_y.noise, linear_shaking_z.noise))
    angular_acceleration_shaking = np.hstack((angular_shaking_x.noise, angular_shaking_y.noise, angular_shaking_z.noise))

    angular_rates_shaking = integrate_trapezoid(angular_acceleration_shaking)

    return linear_acceleration_shaking, angular_acceleration_shaking, angular_rates_shaking


def create_validation_accelerometer_configuration(CAL, V, w_lst, dw_lst, a_ng, pos_dict, noise_psd, fs, filter_length, layout, noise_switch=True):
    """
    Create the accelerometer configuration for the validation using GOCE calibrated parameters
    :param CAL: Array with the calibration parameters
    :param V: Gravity gradient tensor
    :param w_lst: List with the angular rates. w_lst[0] = w_true, w_lst[1] = w_meas
    :param dw_lst: List with the angular accelerations. dw_lst[0] = dw_true, dw_lst[1] = dw_meas. With shaking if applicable
    :param a_ng: Non-gravitational accelerations with shaking if applicable
    :param pos_dict: Position dictionary with the accelerometer positions.
    :param noise_psd: Noise power spectral density of the accelerometer
    :param fs: Sampling frequency of the accelerometer
    :param filter_length: Correlation filter length
    :param layout: Configuration layout of the accelerometers. layout 1= 2 accelerometers, layout 2= 3 accelerometers, layout 3= 4 accelerometers
    :param noise_switch: Boolean to switch on or off the noise
    :return:
    """
    # Load the calibration parameters
    M14 = CAL[0]['M14'][0]
    M25 = CAL[0]['M25'][0]
    M36 = CAL[0]['M36'][0]
    K14 = CAL[0]['K14'][0]
    K25 = CAL[0]['K25'][0]
    K36 = CAL[0]['K36'][0]
    Wc14 = CAL[0]['WC14'][0]
    Wd14 = CAL[0]['WD14'][0]
    Wc25 = CAL[0]['WC25'][0]
    Wd25 = CAL[0]['WD25'][0]
    Wc36 = CAL[0]['WC36'][0]
    Wd36 = CAL[0]['WD36'][0]
    Lx = 0.5140135
    Ly = 0.49989
    Lz = 0.500201

    match layout:
        case 1:
            Mc12 = M14[0:3, 0:3]
            Md12 = M14[3:6, 0:3]
            K1 = K14[0:3, 0:3]
            K2 = K14[3:6, 3:6]
            W1 = Wc14 + Wd14

            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            drc12 = np.zeros(3)
            drd12 = np.zeros(3)

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = pos_dict["pos_acc2"]

            # Check on which axis the accelerometers are placed
            if pos_acc1[0] != 0:
                drd12[0] = 0
                Mc12 = M14[0:3, 0:3]
                Md12 = M14[3:6, 0:3]
                K1 = K14[0:3, 0:3]
                K2 = K14[3:6, 3:6]
                W1 = Wc14 + Wd14
            elif pos_acc1[1] != 0:
                drd12[1] = 0
                Mc12 = M25[0:3, 0:3]
                Md12 = M25[3:6, 0:3]
                K1 = K25[0:3, 0:3]
                K2 = K25[3:6, 3:6]
                W1 = Wc25 + Wd25
                W1 = np.array([[0, 0, 0], [W1[2, 0], 0, W1[0, 2]], [0, W1[2, 1], 0]])
            else:
                drd12[2] = 0
                Mc12 = M36[0:3, 0:3]
                Md12 = M36[3:6, 0:3]
                K1 = K36[0:3, 0:3]
                K2 = K36[3:6, 3:6]
                W1 = Wc36 + Wd36
                W1 = np.array([[0, 0, 0], [W1[1, 0], 0, W1[1, 2]], [0, W1[0, 1], 0]])

            W_idx = [(1, 0), (2, 1), (1, 2)]  # W2 = -W1
            # Acc1
            acc1 = Accelerometer(1, V, w_lst, dw_lst, a_ng, pos_acc1, noise_psd, fs)
            acc1.M = Mc12 + Md12
            acc1.K = K1
            acc1.W = W1
            acc1.dr = drc12 + drd12
            acc1.a_nom = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            # Acc2
            acc2 = Accelerometer(2, V, w_lst, dw_lst, a_ng, pos_acc2, noise_psd, fs)
            acc2.M = Mc12 - Md12
            acc2.K = K2
            acc2.W = -acc1.W.copy()
            acc2.dr = drc12 - drd12
            acc2.a_nom = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            if noise_switch:
                acc1.a_meas = acc1.a_real + acc1.noise
                acc2.a_meas = acc2.a_real + acc2.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real

            return [acc1, acc2]

        case 2:
            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc3 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            drc13 = np.zeros(3)
            drd13 = np.zeros(3)

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = np.zeros(3)
            pos_acc3 = pos_dict["pos_acc3"]

            # Check on which axis the accelerometers are placed
            if pos_acc1[0] != 0:
                drd13[0] = 0
                Mc13 = M14[0:3, 0:3]
                Md13 = M14[3:6, 0:3]
                K1 = K14[0:3, 0:3]
                K3 = K14[3:6, 3:6]
                W1 = Wc14 + Wd14
                W3 = Wc14 - Wd14
            elif pos_acc1[1] != 0:
                drd13[1] = 0
                Mc13 = M25[0:3, 0:3]
                Md13 = M25[3:6, 0:3]
                K1 = K25[0:3, 0:3]
                K3 = K25[3:6, 3:6]
                W1 = Wc25 + Wd25
                W3 = Wc25 - Wd25
                W1 = np.array([[0, 0, 0], [W1[2, 0], 0, W1[0, 2]], [0, W1[2, 1], 0]])
                W3 = np.array([[0, 0, 0], [W3[2, 0], 0, W3[0, 2]], [0, W3[2, 1], 0]])
            else:
                drd13[2] = 0
                Mc13 = M36[0:3, 0:3]
                Md13 = M36[3:6, 0:3]
                K1 = K36[0:3, 0:3]
                K3 = K36[3:6, 3:6]
                W1 = Wc36 + Wd36
                W3 = Wc36 - Wd36
                W1 = np.array([[0, 0, 0], [W1[1, 0], 0, W1[1, 2]], [0, W1[0, 1], 0]])
                W3 = np.array([[0, 0, 0], [W3[1, 0], 0, W3[1, 2]], [0, W3[0, 1], 0]])

            W_idx = [(1, 0), (2, 1), (1, 2)]  # W2 = 0
            # Acc1
            acc1 = Accelerometer(1, V, w_lst, dw_lst, a_ng, pos_acc1, noise_psd, fs)
            acc1.M = Mc13 + Md13
            acc1.K = K1
            acc1.W = W1
            acc1.dr = drc13 + drd13
            acc1.a_nom = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            # Acc2
            acc2 = Accelerometer(2, V, w_lst, dw_lst, a_ng, pos_acc2, noise_psd, fs)
            acc2.W = np.zeros((3, 3))
            acc2.dr = np.zeros(3)
            acc2.a_nom = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            # Acc3
            acc3 = Accelerometer(3, V, w_lst, dw_lst, a_ng, pos_acc3, noise_psd, fs)
            acc3.M = Mc13 - Md13
            acc3.K = K3
            acc3.W = W3
            acc3.dr = drc13 - drd13
            acc3.a_nom = acc3.get_nominal_acc()
            acc3.a_real = acc3.get_realistic_acc()
            acc3.cor_fil = acc3.correlation_filter(filter_length)
            acc3.noise = acc3.filter_matrix(acc3.cor_fil, white_noise_acc3)

            if noise_switch:
                acc1.a_meas = acc1.a_real + acc1.noise
                acc2.a_meas = acc2.a_real + acc2.noise
                acc3.a_meas = acc3.a_real + acc3.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real
                acc3.a_meas = acc3.a_real

            return [acc1, acc2, acc3]

        case 3:
            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc3 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc4 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = pos_dict["pos_acc2"]
            pos_acc3 = pos_dict["pos_acc3"]
            pos_acc4 = pos_dict["pos_acc4"]

            drc13 = np.zeros(3)
            drd13 = np.zeros(3)
            drc24 = np.zeros(3)
            drd24 = np.zeros(3)

            # Check on which axis the accelerometers are placed
            if pos_acc1[0] != 0:
                drd13[0] = 0
                Mc13 = M14[0:3, 0:3]
                Md13 = M14[3:6, 0:3]
                K1 = K14[0:3, 0:3]
                K3 = K14[3:6, 3:6]
                W1 = Wc14 + Wd14
                W3 = Wc14 - Wd14

            elif pos_acc1[1] != 0:
                drd13[1] = 0
                Mc13 = M25[0:3, 0:3]
                Md13 = M25[3:6, 0:3]
                K1 = K25[0:3, 0:3]
                K3 = K25[3:6, 3:6]
                W1 = Wc25 + Wd25
                W3 = Wc25 - Wd25
                W1 = np.array([[0, 0, 0], [W1[2, 0], 0, W1[0, 2]], [0, W1[2, 1], 0]])
                W3 = np.array([[0, 0, 0], [W3[2, 0], 0, W3[0, 2]], [0, W3[2, 1], 0]])

            else:
                drd13[2] = 0
                Mc13 = M36[0:3, 0:3]
                Md13 = M36[3:6, 0:3]
                K1 = K36[0:3, 0:3]
                K3 = K36[3:6, 3:6]
                W1 = Wc36 + Wd36
                W3 = Wc36 - Wd36
                W1 = np.array([[0, 0, 0], [W1[1, 0], 0, W1[1, 2]], [0, W1[0, 1], 0]])
                W3 = np.array([[0, 0, 0], [W3[1, 0], 0, W3[1, 2]], [0, W3[0, 1], 0]])

            if pos_acc2[0] != 0:
                drd24[0] = 0
                Mc24 = M14[0:3, 0:3]
                Md24 = M14[3:6, 0:3]
                K2 = K14[0:3, 0:3]
                K4 = K14[3:6, 3:6]
                W2 = Wc14 + Wd14
            elif pos_acc2[1] != 0:
                drd24[1] = 0
                Mc24 = M25[0:3, 0:3]
                Md24 = M25[3:6, 0:3]
                K2 = K25[0:3, 0:3]
                K4 = K25[3:6, 3:6]
                W2 = Wc25 + Wd25
                W2 = np.array([[0, 0, 0], [W2[2, 0], 0, W2[0, 2]], [0, W2[2, 1], 0]])
            else:
                drd24[2] = 0
                Mc24 = M36[0:3, 0:3]
                Md24 = M36[3:6, 0:3]
                K2 = K36[0:3, 0:3]
                K4 = K36[3:6, 3:6]
                W2 = Wc36 + Wd36
                W2 = np.array([[0, 0, 0], [W2[1, 0], 0, W2[1, 2]], [0, W2[0, 1], 0]])

            W_idx = [(1, 0), (2, 1), (1, 2)]  # W4 = -W2
            # Acc1
            acc1 = Accelerometer(1, V, w_lst, dw_lst, a_ng, pos_acc1, noise_psd, fs)
            acc1.M = Mc13 + Md13
            acc1.K = K1
            acc1.W = W1
            acc1.dr = drc13 + drd13
            acc1.a_nom = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            # Acc3
            acc3 = Accelerometer(3, V, w_lst, dw_lst, a_ng, pos_acc3, noise_psd, fs)
            acc3.M = Mc13 - Md13
            acc3.K = K3
            acc3.W = W3
            acc3.dr = drc13 - drd13
            acc3.a_nom = acc3.get_nominal_acc()
            acc3.a_real = acc3.get_realistic_acc()
            acc3.cor_fil = acc3.correlation_filter(filter_length)
            acc3.noise = acc3.filter_matrix(acc3.cor_fil, white_noise_acc3)

            # Acc2
            acc2 = Accelerometer(2, V, w_lst, dw_lst, a_ng, pos_acc2, noise_psd, fs)
            acc2.M = Mc24 + Md24
            acc2.K = K2
            acc2.W = W2
            acc2.dr = drc24 + drd24
            acc2.a_nom = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            # Acc4
            acc4 = Accelerometer(4, V, w_lst, dw_lst, a_ng, pos_acc4, noise_psd, fs)
            acc4.M = Mc24 - Md24
            acc4.K = K4
            acc4.W = -acc2.W.copy()
            acc4.dr = drc24 - drd24
            acc4.a_nom = acc4.get_nominal_acc()
            acc4.a_real = acc4.get_realistic_acc()
            acc4.cor_fil = acc4.correlation_filter(filter_length)
            acc4.noise = acc4.filter_matrix(acc4.cor_fil, white_noise_acc4)

            if noise_switch:
                acc1.a_meas = acc1.a_real + acc1.noise
                acc2.a_meas = acc2.a_real + acc2.noise
                acc3.a_meas = acc3.a_real + acc3.noise
                acc4.a_meas = acc4.a_real + acc4.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real
                acc3.a_meas = acc3.a_real
                acc4.a_meas = acc4.a_real

            return [acc1, acc2, acc3, acc4]


def load_configuration(PAR, V, w, dw, a_ng, pos_dict, noise_psd, fs, filter_length, layout, noise_switch=True, **kwargs):
    """
    Create an accelerometer object with the given parameters
    :param PAR:
        Dictionary with the true parameters
    :param V:
        Gravity gradient vector
    :param w:
        Angular velocity vector
    :param dw:
        Angular acceleration vector
    :param a_ng:
        Non-gravitational acceleration vector
    :param pos_dict:
        Positions of the accelerometers
    :param noise_psd:
        Noise PSD of the accelerometer
    :param fs:
        Sampling frequency
    :param filter_length:
        Length of the correlation filter
    :param layout:
        Layout of the accelerometers
    :param noise_switch:
        Bool to switch on/off the noise
    :return:
        Accelerometer objects
    """
    match layout:
        case 1:
            M1 = PAR["M1"]
            M2 = PAR["M2"]
            K1 = PAR["K1"]
            K2 = PAR["K2"]
            W1 = PAR["W1"]
            W2 = PAR["W2"]
            dr1 = PAR["dr1"]
            dr2 = PAR["dr2"]

            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = pos_dict["pos_acc2"]

            # Acc1
            acc1 = Accelerometer(1, V, w, dw, a_ng, pos_acc1, noise_psd, fs)
            acc1.M = M1
            acc1.K = K1
            acc1.W = W1
            acc1.dr = dr1
            acc1.a_np = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            acc1.a_real_gg_only = acc1.get_acc_gg_only()
            acc1.a_real_gg_w_rate_only = acc1.get_acc_gg_w_rate_only()

            # Acc2
            acc2 = Accelerometer(2, V, w, dw, a_ng, pos_acc2, noise_psd, fs)
            acc2.M = M2
            acc2.K = K2
            acc2.W = W2
            acc2.dr = dr2
            acc2.a_np = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            acc2.a_real_gg_only = acc2.get_acc_gg_only()
            acc2.a_real_gg_w_rate_only = acc2.get_acc_gg_w_rate_only()

            if noise_switch:
                if 'gg' in kwargs:
                    acc1.a_meas = acc1.a_real_gg_only + acc1.noise
                    acc2.a_meas = acc2.a_real_gg_only + acc2.noise
                elif 'gg_w_rate' in kwargs:
                    acc1.a_meas = acc1.a_real_gg_w_rate_only + acc1.noise
                    acc2.a_meas = acc2.a_real_gg_w_rate_only + acc2.noise
                else:
                    acc1.a_meas = acc1.a_real + acc1.noise
                    acc2.a_meas = acc2.a_real + acc2.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real

            return [acc1, acc2]

        case 2:
            M1 = PAR["M1"]
            M2 = PAR["M2"]
            M3 = PAR["M3"]
            K1 = PAR["K1"]
            K2 = PAR["K2"]
            K3 = PAR["K3"]
            W1 = PAR["W1"]
            W2 = PAR["W2"]
            W3 = PAR["W3"]
            dr1 = PAR["dr1"]
            dr2 = PAR["dr2"]
            dr3 = PAR["dr3"]

            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc3 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = np.zeros(3)
            pos_acc3 = pos_dict["pos_acc3"]

            # Acc1
            acc1 = Accelerometer(1, V, w, dw, a_ng, pos_acc1, noise_psd, fs)
            acc1.M = M1
            acc1.K = K1
            acc1.W = W1
            acc1.dr = dr1
            acc1.a_np = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            acc1.a_real_gg_only = acc1.get_acc_gg_only()
            acc1.a_real_gg_w_rate_only = acc1.get_acc_gg_w_rate_only()

            # Acc2
            acc2 = Accelerometer(2, V, w, dw, a_ng, pos_acc2, noise_psd, fs)
            acc2.M = M2
            acc2.K = K2
            acc2.W = W2
            acc2.dr = dr2
            acc2.a_np = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            acc2.a_real_gg_only = acc2.get_acc_gg_only()
            acc2.a_real_gg_w_rate_only = acc2.get_acc_gg_w_rate_only()

            # Acc3
            acc3 = Accelerometer(3, V, w, dw, a_ng, pos_acc3, noise_psd, fs)
            acc3.M = M3
            acc3.K = K3
            acc3.W = W3
            acc3.dr = dr3
            acc3.a_np = acc3.get_nominal_acc()
            acc3.a_real = acc3.get_realistic_acc()
            acc3.cor_fil = acc3.correlation_filter(filter_length)
            acc3.noise = acc3.filter_matrix(acc3.cor_fil, white_noise_acc3)

            acc3.a_real_gg_only = acc3.get_acc_gg_only()
            acc3.a_real_gg_w_rate_only = acc3.get_acc_gg_w_rate_only()

            if noise_switch:
                if 'gg' in kwargs:
                    acc1.a_meas = acc1.a_real_gg_only + acc1.noise
                    acc2.a_meas = acc2.a_real_gg_only + acc2.noise
                    acc3.a_meas = acc3.a_real_gg_only + acc3.noise
                elif 'gg_w_rate' in kwargs:
                    acc1.a_meas = acc1.a_real_gg_w_rate_only + acc1.noise
                    acc2.a_meas = acc2.a_real_gg_w_rate_only + acc2.noise
                    acc3.a_meas = acc3.a_real_gg_w_rate_only + acc3.noise
                else:
                    acc1.a_meas = acc1.a_real + acc1.noise
                    acc2.a_meas = acc2.a_real + acc2.noise
                    acc3.a_meas = acc3.a_real + acc3.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real
                acc3.a_meas = acc3.a_real

            return [acc1, acc2, acc3]

        case 3:
            M1 = PAR["M1"]
            M2 = PAR["M2"]
            M3 = PAR["M3"]
            M4 = PAR["M4"]
            K1 = PAR["K1"]
            K2 = PAR["K2"]
            K3 = PAR["K3"]
            K4 = PAR["K4"]
            W1 = PAR["W1"]
            W2 = PAR["W2"]
            W3 = PAR["W3"]
            W4 = PAR["W4"]
            dr1 = PAR["dr1"]
            dr2 = PAR["dr2"]
            dr3 = PAR["dr3"]
            dr4 = PAR["dr4"]

            white_noise_acc1 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc2 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc3 = np.random.randn(len(a_ng) + filter_length - 1, 3)
            white_noise_acc4 = np.random.randn(len(a_ng) + filter_length - 1, 3)

            pos_acc1 = pos_dict["pos_acc1"]
            pos_acc2 = pos_dict["pos_acc2"]
            pos_acc3 = pos_dict["pos_acc3"]
            pos_acc4 = pos_dict["pos_acc4"]

            # Acc1
            acc1 = Accelerometer(1, V, w, dw, a_ng, pos_acc1, noise_psd, fs)
            acc1.M = M1
            acc1.K = K1
            acc1.W = W1
            acc1.dr = dr1
            acc1.a_np = acc1.get_nominal_acc()
            acc1.a_real = acc1.get_realistic_acc()
            acc1.cor_fil = acc1.correlation_filter(filter_length)
            acc1.noise = acc1.filter_matrix(acc1.cor_fil, white_noise_acc1)

            acc1.a_real_gg_only = acc1.get_acc_gg_only()
            acc1.a_real_gg_w_rate_only = acc1.get_acc_gg_w_rate_only()

            # Acc3
            acc3 = Accelerometer(3, V, w, dw, a_ng, pos_acc3, noise_psd, fs)
            acc3.M = M3
            acc3.K = K3
            acc3.W = W3
            acc3.dr = dr3
            acc3.a_np = acc3.get_nominal_acc()
            acc3.a_real = acc3.get_realistic_acc()
            acc3.cor_fil = acc3.correlation_filter(filter_length)
            acc3.noise = acc3.filter_matrix(acc3.cor_fil, white_noise_acc3)

            acc3.a_real_gg_only = acc3.get_acc_gg_only()
            acc3.a_real_gg_w_rate_only = acc3.get_acc_gg_w_rate_only()

            # Acc2
            acc2 = Accelerometer(2, V, w, dw, a_ng, pos_acc2, noise_psd, fs)
            acc2.M = M2
            acc2.K = K2
            acc2.W = W2
            acc2.dr = dr2
            acc2.a_np = acc2.get_nominal_acc()
            acc2.a_real = acc2.get_realistic_acc()
            acc2.cor_fil = acc2.correlation_filter(filter_length)
            acc2.noise = acc2.filter_matrix(acc2.cor_fil, white_noise_acc2)

            acc2.a_real_gg_only = acc2.get_acc_gg_only()
            acc2.a_real_gg_w_rate_only = acc2.get_acc_gg_w_rate_only()

            # Acc4
            acc4 = Accelerometer(4, V, w, dw, a_ng, pos_acc4, noise_psd, fs)
            acc4.M = M4
            acc4.K = K4
            acc4.W = W4
            acc4.dr = dr4
            acc4.a_np = acc4.get_nominal_acc()
            acc4.a_real = acc4.get_realistic_acc()
            acc4.cor_fil = acc4.correlation_filter(filter_length)
            acc4.noise = acc4.filter_matrix(acc4.cor_fil, white_noise_acc4)

            acc4.a_real_gg_only = acc4.get_acc_gg_only()
            acc4.a_real_gg_w_rate_only = acc4.get_acc_gg_w_rate_only()

            if noise_switch:
                if 'gg' in kwargs:
                    acc1.a_meas = acc1.a_real_gg_only + acc1.noise
                    acc2.a_meas = acc2.a_real_gg_only + acc2.noise
                    acc3.a_meas = acc3.a_real_gg_only + acc3.noise
                    acc4.a_meas = acc4.a_real_gg_only + acc4.noise
                elif 'gg_w_rate' in kwargs:
                    acc1.a_meas = acc1.a_real_gg_w_rate_only + acc1.noise
                    acc2.a_meas = acc2.a_real_gg_w_rate_only + acc2.noise
                    acc3.a_meas = acc3.a_real_gg_w_rate_only + acc3.noise
                    acc4.a_meas = acc4.a_real_gg_w_rate_only + acc4.noise
                else:
                    acc1.a_meas = acc1.a_real + acc1.noise
                    acc2.a_meas = acc2.a_real + acc2.noise
                    acc3.a_meas = acc3.a_real + acc3.noise
                    acc4.a_meas = acc4.a_real + acc4.noise
            else:
                acc1.a_meas = acc1.a_real
                acc2.a_meas = acc2.a_real
                acc3.a_meas = acc3.a_real
                acc4.a_meas = acc4.a_real

            return [acc1, acc2, acc3, acc4]
