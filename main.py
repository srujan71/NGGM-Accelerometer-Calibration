###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import os
import time
import datetime
import numpy as np

from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.astro import time_conversion

# Problem-specific imports
from Tools.Dynamics import SpacecraftDynamics, get_initial_states, attitude_correction_matrices, save_gravity_gradients_angular_rates

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
calendar_date = datetime.datetime(2034, 10, 1, 13, 0, 0)
julian_date = time_conversion.calendar_date_to_julian_day_since_epoch(calendar_date)
simulation_start_epoch = julian_date * 24 * 3600  # s
maximum_duration = constants.JULIAN_DAY * 2  # s

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
integrator_settings = sim.define_integrator(4, 0)
propagation_settings = sim.define_propagation(bodies, initial_states, dependent_variables, termination_settings)
propagation_settings.integrator_settings = integrator_settings

output_path = current_dir + "/SimulationOutput/Output/Orbit_data/"
sim.run_sim(bodies, propagation_settings, output_path)
print("Simulation completed in: ", time.time() - start_time, " seconds")

# Correct the attitude and save the quaternions and other rotation matrices
state = np.loadtxt(output_path + "state_history.dat")
deps_var = np.loadtxt(output_path + "dependent_variable_history.dat")
attitude_correction_matrices(state, deps_var, output_path)


# Store the gravity gradients
ecrf_path = output_path + f"pos_ecrf.txt"
rbv_path = output_path + f"R_BV.txt"
save_path = output_path

save_gravity_gradients_angular_rates(130, ecrf_path, rbv_path, output_path)
