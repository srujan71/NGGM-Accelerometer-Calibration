import numpy as np
from Tools.PlotTools import plot_asd, plt, create_plots, save_figure
import os
from scipy.fft import rfftfreq

current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Freq_0.1_30_seeds/y-axis_old/Thrust_1e-06/Seed_1/Nominal/Run_0/"

NFFT = 27001
freq_acc = rfftfreq(NFFT, 1)
a_ng_req = (5e-12 * np.sqrt(1 + np.where(freq_acc != 0, 0.001 / freq_acc, np.inf) ** 2 + (100 * freq_acc ** 2) ** 2))
a_ng_req[0] = 0
lri = 220e3*1e-13 * np.sqrt(1 + (0.01/freq_acc)**2) * np.sqrt(1 + (0.001/freq_acc)**2)
lri[0] = 0

lri_acc = lri * (2*np.pi*freq_acc)**2
combined = np.sqrt(a_ng_req ** 2 + lri_acc ** 2)


a_ng_error = np.loadtxt(path + "a_ng_error.txt")

fig_asd, ax_asd = plot_asd(a_ng_error, 27001, [r"Relative $a_{ng}$ error after calibration"], "", plot_all_axes=False, colormap='Wistia_r', fontsize=16, figsize=(9, 7))

# fig_asd, ax_asd = create_plots('Frequency [Hz]', "Relative acceleration\n" + r"measurement error ASD [$m/s^{2}/\sqrt{Hz}$]", fontsize=16, figsize=(8, 6))
ax_asd.plot(freq_acc[1:], a_ng_req[1:], label=r"Relative $a_{ng}$ error requirement", color='black', linestyle='--')
ax_asd.plot(freq_acc[1:], lri_acc[1:], label="Double time derivative of inter-satellite\n distance variation requirement", color='blue', linestyle=':')
ax_asd.plot(freq_acc[1:], combined[1:], label="Combined requirement", color='red', linestyle='-')
# ax_asd.annotate('Orbital harmonic\npeaks', xy=(1.85e-4, 5e-10), xytext=(2e-4, 1e-9), arrowprops=dict(facecolor='black', arrowstyle='->'), ha='left')
# ax_asd.annotate('', xy=(3.7e-4, 9.5e-11), xytext=(3.1e-4, 0.9e-9), arrowprops=dict(facecolor='black', arrowstyle='->'), ha='left')
# ax_asd.annotate('', xy=(5.55e-4, 3.1e-11), xytext=(3.1e-4, 0.9e-9), arrowprops=dict(facecolor='black', arrowstyle='->'), ha='left')
ax_asd.vlines(1e-3, 5e-13, 1e-7, color='green', linestyle='-.')
ax_asd.vlines(1e-4, 5e-13, 1e-7, color='green', linestyle='-.')
ax_asd.legend(fontsize=14)
ax_asd.set_xlabel('Frequency [Hz]')
ax_asd.set_ylabel("Relative acceleration\n" + r"measurement error ASD [$m/s^{2}/\sqrt{Hz}$]")
ax_asd.set_ylim([6e-13, 1e-7])
ax_asd.set_yscale('log')
ax_asd.set_xscale('log')
ax_asd.grid()
fig_asd.tight_layout()
save_figure(fig_asd, "Sensitivity_analysis/metric_asd.svg")

plt.show()
