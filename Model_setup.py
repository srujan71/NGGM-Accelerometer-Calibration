"""

The entries of the vector 'linear_thrust' contains the following:
- Entry 0: Constant thrust magnitude per sqrt of frequency
- Entry 1: Lower bound of frequency
- Entry 2: upper bound of frequency
- Entry 3: Duration for which this thrust has to last.
- Entry 4: Point in time at which the thrust has to be applied. Is a fraction of the total duration.

"""
###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import time

import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.data import save2txt
# Problem-specific imports
from Tools.Dynamics import SpacecraftDynamics, get_initial_states, save_gravity_gradients_angular_rates, \
    generate_benchmarks, compare_benchmarks

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################
start_time = time.time()
# Load spice kernels
spice_interface.load_standard_kernels()

# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Set simulation start epoch
simulation_start_epoch = 0  # s
maximum_duration = constants.JULIAN_DAY  # s

sc_properties = {"Mass": 1000,
                 "Length": 3.1225,
                 "Width": 1.944,
                 "Height": 0.775,
                 "Aerodynamic_coefficients": np.array([2.3, 0, 0]),
                 "Reference_area": 0.955,
                 "Specular_reflection_coefficient": 0.4,
                 "Diffuse_reflection_coefficient": 0.26}

###########################################################################
# CREATE SIMULATION INSTANCE ##############################################
###########################################################################

sim = SpacecraftDynamics(simulation_start_epoch, maximum_duration, sc_properties)

initial_states = get_initial_states()
# Define the integrator, and propagator settings.
bodies = sim.define_environment(rings=3)
sim.bodies = bodies
dependent_variables = sim.define_dependent_variables()
termination_settings = sim.define_termination_settings()
propagation_settings = sim.define_propagation(bodies, initial_states, dependent_variables, termination_settings)

###########################################################################
# Create a benchmark ######################################################
###########################################################################
are_dependent_variables_to_save = False if not dependent_variables else True
benchmark_interpolator_settings = interpolators.lagrange_interpolation(8, boundary_interpolation=interpolators.extrapolate_at_boundary)

benchmark_output_path = "SimulationOutput/Output/benchmarks/"

use_best_benchmark = False
benchmark_found = False
run_integrator_analysis = False
run_model_analysis = True
gravity_gradients_calculation = False

if use_best_benchmark:
    if benchmark_found:
        chosen_benchmark_step_size = 0.25
        # benchmark = np.loadtxt(f"SimulationOutput/Output/benchmarks/benchmark_1_states_t={chosen_benchmark_step_size}.dat")
        # benchmark_diff = np.loadtxt(f"SimulationOutput/Output/benchmarks/benchmarks_state_difference_t={chosen_benchmark_step_size}.dat")
        # benchmark_dep_vars = np.loadtxt(f"SimulationOutput/Output/benchmarks/benchmark_1_dependent_variables_t={chosen_benchmark_step_size}.dat")

        benchmark = np.loadtxt(f"SimulationOutput/Output/benchmarks/benchmark_2_states.dat")
        benchmark_dep_vars = np.loadtxt(f"SimulationOutput/Output/benchmarks/benchmark_2_dependent_variables.dat")

        benchmark_state_history = {benchmark[i, 0]: benchmark[i, 1:] for i in range(len(benchmark))}
        benchmark_dep_vars_history = {benchmark_dep_vars[j, 0]: benchmark_dep_vars[j, 1:] for j in range(len(benchmark_dep_vars))}
        benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            benchmark_state_history,
            benchmark_interpolator_settings)

        benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            benchmark_dep_vars_history,
            benchmark_interpolator_settings)

        last_epoch = benchmark[-1, 0]
    else:
        benchmark_step_sizes = [0.5]
        for i in range(len(benchmark_step_sizes)):
            benchmark_list = generate_benchmarks(benchmark_step_sizes[i],
                                                 bodies,
                                                 propagation_settings,
                                                 are_dependent_variables_to_save,
                                                 benchmark_output_path)

            first_benchmark_state_history = benchmark_list[0]
            second_benchmark_state_history = benchmark_list[1]

            # Compare the benchmarks, returning interpolator of the first benchmark and writing the results to a file
            benchmark_state_difference = compare_benchmarks(first_benchmark_state_history,
                                                            second_benchmark_state_history,
                                                            benchmark_output_path,
                                                            f"benchmarks_state_difference_t={str(benchmark_step_sizes[i])}.dat")

            if are_dependent_variables_to_save:
                first_benchmark_dependent_variable_history = benchmark_list[2]
                second_benchmark_dependent_variable_history = benchmark_list[3]

                # Compare benchmark dependent variables, returning interpolator of the first benchmark, and writing difference
                # to file if write_results_to_file is set to True
                benchmark_dependent_difference = compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                    second_benchmark_dependent_variable_history,
                                                                    benchmark_output_path,
                                                                    f'benchmarks_dependent_variable_difference_t={str(benchmark_step_sizes[i])}.dat')

    ###########################################################################
    # Integrator analysis ####################################################
    ###########################################################################
    if run_integrator_analysis:
        available_propagators = [propagation_setup.propagator.cowell,
                                 propagation_setup.propagator.encke,
                                 propagation_setup.propagator.gauss_modified_equinoctial]

        number_of_propagators = len(available_propagators)
        number_of_integrators = 6

        for propagator_index in range(number_of_propagators):
            if propagator_index != 0:
                continue
            current_propagator = available_propagators[propagator_index]

            # Define the propagator settings
            current_propagator_settings = sim.define_propagation(bodies,
                                                                 initial_states,
                                                                 dependent_variables,
                                                                 termination_settings,
                                                                 current_propagator)

            for integrator_index in range(number_of_integrators):
                if integrator_index != 4:
                    continue
                else:
                    number_of_integrator_step_size_settings = 3

                for step_size_index in range(number_of_integrator_step_size_settings):
                    to_print = f"Current run: \n Propagator_index = {str(propagator_index)} \n Integrator_index = {str(integrator_index)} \n" \
                               f" Step_size_index = {str(step_size_index)} \n"

                    print(to_print)

                    output_path = f"SimulationOutput/Output/integrator_propagator/prop_{propagator_index}/int_{integrator_index}/step_size_{step_size_index}/"

                    current_integrator_settings = sim.define_integrator(integrator_index, step_size_index)
                    current_propagator_settings.integrator_settings = current_integrator_settings

                    state_history, unprocessed_state_history, dependent_variable_history, dict_to_write = \
                        sim.run_sim(bodies, current_propagator_settings, output_path, return_files=True)

                    #######################
                    # Benchmark comparison#
                    #######################
                    state_difference = dict()
                    dependent_difference = dict()
                    for epoch in state_history.keys():
                        if epoch < 6 ** chosen_benchmark_step_size or epoch >= last_epoch - 6 ** chosen_benchmark_step_size:
                            continue
                        else:
                            state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)
                            dependent_difference[epoch] = dependent_variable_history[epoch] - benchmark_dependent_variable_interpolator.interpolate(epoch)

                    save2txt(state_difference, f"state_difference_wrt_benchmark.dat", output_path)
                    save2txt(dependent_difference, f"dependent_variable_difference_wrt_benchmark.dat", output_path)

if run_model_analysis:
    models = 5
    model_labels = ["Nominal", "Earth_D/O_120", "Moon", "Sun", "Jupiter"]
    current_integrator_settings = sim.define_integrator(4, 0)
    for model_index in range(models):
        print(model_index)
        output_path = f"SimulationOutput/Output/Acceleration_models/model_{model_index}/"

        current_propagator_settings = sim.define_propagation(bodies, initial_states, dependent_variables, termination_settings,
                                                             run_model_analysis=True, model=model_index)
        current_propagator_settings.integrator_settings = current_integrator_settings

        state_history, unprocessed_state_history, dependent_variable_history, dict_to_write = \
            sim.run_sim(bodies, current_propagator_settings, output_path, return_files=True)
        dict_to_write['Model'] = model_labels[model_index]
        save2txt(dict_to_write, "ancillary_simulation_info.txt", output_path)

if gravity_gradients_calculation:
    model_labels = ["Nominal", "Earth_D/O_120", "Moon", "Sun", "Jupiter"]
    for model_index in range(2):
        dependent_variables = np.loadtxt(f"SimulationOutput/Output/Acceleration_models/model_{model_index}/dependent_variable_history.dat")
        pos_ecrf = dependent_variables[:, -3:]
        R_BV = dependent_variables[:, 10:19]
        np.savetxt(f"SimulationOutput/Output/Acceleration_models/model_{model_index}/pos_ecrf.txt", pos_ecrf)
        np.savetxt(f"SimulationOutput/Output/Acceleration_models/model_{model_index}/R_BV.txt", R_BV)
        print(f"Gravity gradients calculation Model {model_labels[model_index]}")
        ecrf_path = f"SimulationOutput/Output/Acceleration_models/model_{model_index}/pos_ecrf.txt"
        rbv_path = f"SimulationOutput/Output/Acceleration_models/model_{model_index}/R_BV.txt"
        save_path = f"SimulationOutput/Output/Acceleration_models/model_{model_index}/"

        save_gravity_gradients_angular_rates(120, ecrf_path, rbv_path, save_path)
