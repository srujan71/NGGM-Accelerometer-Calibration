import numpy as np
from Tools.PlotTools import create_plots, save_figure, plt
import os
from scipy import stats

current_dir = os.path.dirname(__file__)
path = current_dir + "/../SimulationOutput/Output/Sensitivity_analysis/One_at_a_time/Frequency_analysis/avg_removed/"
fig_save_path = "Sensitivity_analysis/One_at_a_time/Frequency_analysis/avg_removed/"  # The first half of the path is defined already in save figure function

freqs_sh = [0.01, 0.1]
seeds = list(range(1, 201))

ratio_freqs = []
cum_mean_freqs = []
cum_std_freqs = []
for f_sh in freqs_sh:
    ratio_lst = []
    for seed in seeds:
        # if f_sh != 0.1:
        #     folder_path = path + "wo_factor/" + f"f_{f_sh}/Seed_{seed}/"
        # else:
        folder_path = path + f"f_{f_sh}/Seed_{seed}/"
        ratio = np.loadtxt(folder_path + "PSD_ratio.txt")
        # if ratio > 100:
        #     continue
        ratio_lst.append(ratio)

    # Filter the ratios using IQR method
    # ratio_lst_arr = np.array(ratio_lst)
    # Q1 = np.percentile(ratio_lst_arr, 25)
    # Q3 = np.percentile(ratio_lst_arr, 75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # ratio_lst = ratio_lst_arr[(ratio_lst_arr > lower_bound) & (ratio_lst_arr < upper_bound)]

    # Calculate cumulative mean and standard deviation
    cum_mean = []
    cum_std = []
    for i in range(len(ratio_lst)):
        ratio_cum = ratio_lst[:i+1]
        cum_mean.append(np.mean(ratio_cum))
        cum_std.append(np.std(ratio_cum, ddof=1))

    # convert to numpy arrays
    cum_mean = np.array(cum_mean)
    cum_std = np.array(cum_std)

    # Append to the lists
    ratio_freqs.append(ratio_lst)
    cum_mean_freqs.append(cum_mean)
    cum_std_freqs.append(cum_std)


bins = [100, 500]
for i, f_sh in enumerate(freqs_sh):
    x = np.linspace(min(ratio_freqs[i]), max(ratio_freqs[i]), 5000)
    # Gamma dstribution
    params_gamma = stats.gamma.fit(ratio_freqs[i])
    pdf_fitted_gamma = stats.gamma.pdf(x, *params_gamma)

    # pareto distribution
    params_pareto = stats.pareto.fit(ratio_freqs[i])
    pdf_fitted_pareto = stats.pareto.pdf(x, *params_pareto)

    # F-distribution
    params_f = stats.f.fit(ratio_freqs[i])
    pdf_fitted_f_dist = stats.f.pdf(x, *params_f)

    # exponential distribution
    params_exp = stats.expon.fit(ratio_freqs[i])
    pdf_fitted_exp = stats.expon.pdf(x, *params_exp)

    # Chi-squared distribution
    params_chi2 = stats.chi2.fit(ratio_freqs[i])
    pdf_fitted_chi2 = stats.chi2.pdf(x, *params_chi2)

    ks_stat_f, p_value_f = stats.kstest(ratio_freqs[i], "f", args=params_f)
    ks_stat_chi2, p_value_chi2 = stats.kstest(ratio_freqs[i], "chi2", args=params_chi2)

    ks_stat_gamma, p_value_gamma = stats.kstest(ratio_freqs[i], "gamma", args=params_gamma)
    ks_stat_pareto, p_value_pareto = stats.kstest(ratio_freqs[i], "pareto", args=params_pareto)
    ks_stat_exp, p_value_exp = stats.kstest(ratio_freqs[i], "expon", args=params_exp)

    print("--------------------")
    print(f"Frequency: {f_sh}")
    print("--------------------")
    print(f"KS test for f-distribution: {ks_stat_f}, {p_value_f}")
    print(f"KS test for chi-squared distribution: {ks_stat_chi2}, {p_value_chi2}")
    print(f"KS test for gamma distribution: {ks_stat_gamma}, {p_value_gamma}")
    print(f"KS test for pareto distribution: {ks_stat_pareto}, {p_value_pareto}")
    print(f"KS test for exponential distribution: {ks_stat_exp}, {p_value_exp}")
    print("\n")



    # Create a histogram of the ratios
    fig, ax = create_plots("Ratio", "Frequency", False, False)
    ax.hist(ratio_freqs[i], bins=bins[i], density=True, histtype='barstacked', edgecolor='black', label=f"f = {f_sh} Hz", color='gray')
    if i == 6:
        # ax.plot(x, pdf_fitted_gamma, label='Gamma distribution', color='red')
        ax.plot(x, pdf_fitted_pareto, label='Pareto distribution', color='green')
        ax.plot(x, pdf_fitted_f_dist, label='F-distribution', color='blue')
        ax.plot(x, pdf_fitted_exp, label='Exponential distribution', color='orange')
        # ax.plot(x, pdf_fitted_chi2, label='Chi-squared distribution', color='purple')
    elif i == 0:
        # ax.plot(x, pdf_fitted_f_dist, label='F-distribution', color='blue')
        ax.plot(x, pdf_fitted_chi2, label='Chi-squared distribution', color='purple')
        ax.plot(x, pdf_fitted_pareto, label='Pareto distribution', color='green')
    else:
        # ax.plot(x, pdf_fitted_f_dist, label='F-distribution', color='blue')
        ax.plot(x, pdf_fitted_pareto, label='Pareto distribution', color='green')

    ax.set_xscale('log') if i == 1 else None
    ax.legend()

    # 95% Confidence interval for pareto distribution
    # lower_bound = stats.pareto.ppf(0.025, *params_pareto)
    # upper_bound = stats.pareto.ppf(0.975, *params_pareto)
    alpha = params_pareto[0]
    xm = params_pareto[1] + params_pareto[2]
    confidence = 0.95



plt.show()
# Create a histogram with all frequencies stacked
fig_all, ax_all = create_plots("Ratio", "Frequency", False, False)
ax_all.hist(ratio_freqs, bins=1000, density=True, histtype='barstacked', edgecolor='black', label=[f"f = {f_sh} Hz" for f_sh in freqs_sh])
ax_all.legend()
ax_all.set_xscale('log')

# Create a cumulative mean plot
fig_cum_mean, ax_cum_mean = create_plots("Number of samples", "Cumulative mean", False, False)
for i, f_sh in enumerate(freqs_sh):
    ax_cum_mean.plot(cum_mean_freqs[i], label=f"f = {f_sh} Hz")
ax_cum_mean.legend()
ax_cum_mean.set_title("Cumulative mean")

# Create a cumulative standard deviation plot
fig_cum_std, ax_cum_std = create_plots("Number of samples", "Cumulative standard deviation", False, False)
for i, f_sh in enumerate(freqs_sh):
    ax_cum_std.plot(cum_std_freqs[i], label=f"f = {f_sh} Hz")
ax_cum_std.legend()
ax_cum_std.set_title("Cumulative standard deviation")

# high frequency bootstrap method for confidence intervals
# data = ratio_freqs[2]
# params_pareto = stats.pareto.fit(data)
# n_bootstraps = 1000
# bootstrap_samples = np.random.pareto(params_pareto[0], size=(n_bootstraps, len(data))) * (params_pareto[1] + params_pareto[2])
#
# bootstrap_means = np.mean(bootstrap_samples, axis=1)
#
# # 99% confidence interval
# lower_bound, upper_bound = np.percentile(bootstrap_means, [0.5, 99.5])
#
# print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
#
# # plot the bootstrap samples
# fig_bootstrap, ax_bootstrap = create_plots("Ratio", "Frequency", False, False)
# ax_bootstrap.hist(bootstrap_means, bins=100, density=True, histtype='barstacked', edgecolor='black', color='gray')
# ax_bootstrap.axvline(lower_bound, color='red', label="Lower bound")
# ax_bootstrap.axvline(upper_bound, color='red', label="Upper bound")
# ax_bootstrap.legend()
#
# data = ratio_freqs[2]  # your data for the current frequency band
# params_f = stats.f.fit(data)  # Fit F-distribution
#
# # Parameters for the F-distribution
# dfn, dfd = params_f[0], params_f[1]
#
# # Generate a range of x values for plotting the F-distribution PDF
# x = np.linspace(0, np.max(data), 1000)
#
# # Calculate the F-distribution PDF using the fitted parameters
# pdf_fitted_f_dist = stats.f.pdf(x, dfn, dfd)
#
# # Calculate 95% confidence interval for the F-distribution using the percentiles
# ci_lower = stats.f.ppf(0.025, dfn, dfd)
# ci_upper = stats.f.ppf(0.975, dfn, dfd)
#
# print(f"95% Confidence Interval: ({ci_lower:.2f}, {ci_upper:.2f})")
#
# # Plot the F-distribution PDF
# plt.plot(x, pdf_fitted_f_dist, label='F-distribution (fitted)', color='b')
#
# # Shade the area within the confidence interval
# plt.fill_between(x, pdf_fitted_f_dist, where=(x >= ci_lower) & (x <= ci_upper), color='gray', alpha=0.5, label='95% Confidence Interval')
#
# # Add vertical lines for the confidence interval bounds
# plt.axvline(ci_lower, color='r', linestyle='--', label=f'CI Lower Bound: {ci_lower:.2f}')
# plt.axvline(ci_upper, color='r', linestyle='--', label=f'CI Upper Bound: {ci_upper:.2f}')
#
# # Labeling the plot
# plt.title('F-distribution with 95% Confidence Interval')
# plt.xlabel('Value')
# plt.ylabel('Density')
# plt.legend()
#
#




# save_figure(fig, fig_save_path + "Histogram_ratios.png")
# save_figure(fig_cum_mean, fig_save_path + "Cumulative_mean.png")
# save_figure(fig_cum_std, fig_save_path + "Cumulative_std.png")


