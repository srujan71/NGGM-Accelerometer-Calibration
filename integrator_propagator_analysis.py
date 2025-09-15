import numpy as np
from Tools.PlotTools import create_plots, save_figure, plt

# benchmark_step_sizes = [30, 15, 7, 3, 1, 0.5]
# max_position_error = np.empty(len(benchmark_step_sizes))
# for i in range(len(benchmark_step_sizes)):
#     state_diff = np.loadtxt(f"SimulationOutput/Output/benchmarks/benchmarks_state_difference_t={str(benchmark_step_sizes[i])}.dat")
#     dep_vars_diff = np.loadtxt(f"SimulationOutput/Output/benchmarks/benchmarks_dependent_variable_difference_t={str(benchmark_step_sizes[i])}.dat")
#     t = state_diff[6:-6, 0]
#     position_error = np.linalg.norm(state_diff[6:-6, 1:4], axis=1)
#     max_position_error[i] = np.max(position_error)
#
# fig, ax = create_plots(r'$\Delta t$ [s]', 'Maximum position error [m]', True, True)
# ax.plot(benchmark_step_sizes, max_position_error, label='Position error', marker='o')
# ax.legend()
# save_figure(fig, "Model_setup/benchmark_position_error.svg")

number_of_integrators = 6
number_of_step_sizes = 3
number_of_propagators = 3

# Plot position error as a function of time
fig_eu_rk, ax_eu_rk = create_plots("Time [hr]", r"Position error $\epsilon_{r}$ [m]", False, True, fontsize=16)
integrator_label = ["RK4", "Euler"]
for integrator_index in range(number_of_integrators):
    if integrator_index < 4:
        continue
    fig1, ax1 = create_plots("Time [hr]", r"Position error $\epsilon_{r}$ [m]", False, True, fontsize=16)

    for step_size_index in range(number_of_step_sizes):
        state_diff = np.loadtxt(f"SimulationOutput/Output/integrator_propagator/prop_0/int_{integrator_index}/"
                                f"step_size_{step_size_index}/state_difference_wrt_benchmark.dat")
        t = (state_diff[:, 0] - state_diff[0, 0]) / 3600
        position_error = np.linalg.norm(state_diff[:, 1:4], axis=1)
        fixed_step_size = 2 ** step_size_index
        ax1.plot(t, position_error, label=fr"$\Delta t$ = {fixed_step_size}")

        if step_size_index == 0:
            ax_eu_rk.plot(t, position_error, label=f"Integrator {integrator_label[integrator_index - 4]}")

    ax_eu_rk.legend(fontsize=15)
    ax1.legend(fontsize=15)
    save_figure(fig1, f"Model_setup/propagator_0_integrator_{integrator_label[integrator_index - 4]}_position_error.svg")
    save_figure(fig_eu_rk, f"Model_setup/propagator_0_integrator_comparison.svg")

# Plot the position error for all the propagators
integrator_index = 4
step_size_index = 0
propagator_labels = ["Cowell", "Encke", "MEE"]
fig2, ax2 = create_plots("Time [hr]", r"Position error $\epsilon_{r}$ [m]", False, True, fontsize=16)
for propagator_index in range(number_of_propagators):
    for step_size_index in range(number_of_step_sizes):
        unprocessed_history = np.loadtxt(f"SimulationOutput/Output/integrator_propagator/prop_{propagator_index}/int_{integrator_index}/"
                                         f"step_size_{step_size_index}/unprocessed_state_history.dat")
        unprocessed_time = (unprocessed_history[:, 0] - unprocessed_history[0, 0]) / 3600
        state_diff = np.loadtxt(f"SimulationOutput/Output/integrator_propagator/prop_{propagator_index}/int_{integrator_index}/"
                                f"step_size_{step_size_index}/state_difference_wrt_benchmark.dat")
        t = (state_diff[:, 0] - state_diff[0, 0]) / 3600
        position_error = np.linalg.norm(state_diff[:, 1:4], axis=1)
        ax2.plot(t, position_error, label=fr"{propagator_labels[propagator_index]} $\Delta t$ = {2 ** step_size_index}")

ax2.legend(fontsize=15)
# plt.show()
save_figure(fig2, f"Model_setup/Propagators_position_error.svg")

# Plot the position error between dt = 4s and dt = 1s for cowell propagator
# interpolator_settings = interpolators.lagrange_interpolation(8, boundary_interpolation=interpolators.extrapolate_at_boundary)
# state_dt_1s = np.loadtxt("SimulationOutput/Output/integrator_propagator/prop_0/int_4/step_size_0/state_history.dat")
# dep_vars_dt_1s = np.loadtxt("SimulationOutput/Output/integrator_propagator/prop_0/int_4/step_size_0/dependent_variable_history.dat")
#
# state_dt_1s_history = {state_dt_1s[i, 0]: state_dt_1s[i, 1:] for i in range(len(state_dt_1s))}
# deps_vars_dt_1s_history = {dep_vars_dt_1s[i, 0]: dep_vars_dt_1s[i, 1:] for i in range(len(dep_vars_dt_1s))}
#
# state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
#     state_dt_1s_history,
#     interpolator_settings
# )
#
# dep_vars_interpolator = interpolators.create_one_dimensional_vector_interpolator(
#     deps_vars_dt_1s_history,
#     interpolator_settings
# )
#
# last_epoch = state_dt_1s[-1, 0]
#
# state_dt_4s = np.loadtxt("SimulationOutput/Output/integrator_propagator/prop_0/int_4/step_size_2/state_history.dat")
# dep_vars_dt_4s = np.loadtxt("SimulationOutput/Output/integrator_propagator/prop_0/int_4/step_size_2/dependent_variable_history.dat")
# state_dt_4s_history = {state_dt_4s[i, 0]: state_dt_4s[i, 1:] for i in range(len(state_dt_4s))}
# deps_vars_dt_4s_history = {dep_vars_dt_4s[i, 0]: dep_vars_dt_4s[i, 1:] for i in range(len(dep_vars_dt_4s))}
#
# state_difference = dict()
# dependent_difference = dict()
#
# for epoch in state_dt_4s_history.keys():
#     if epoch < 6 ** 1 or epoch >= last_epoch - 6 ** 1:
#         continue
#     else:
#         state_difference[epoch] = state_dt_4s_history[epoch] - state_interpolator.interpolate(epoch)
#         dependent_difference[epoch] = deps_vars_dt_4s_history[epoch] - dep_vars_interpolator.interpolate(epoch)
#
# output_path = "SimulationOutput/Output/integrator_propagator/"
# save2txt(state_difference, f"state_difference_wrt_dt1s.dat", output_path)
# save2txt(dependent_difference, f"dependent_variable_difference_wrt_dt1s.dat", output_path)


