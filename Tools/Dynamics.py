###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################
import os.path

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import environment
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel import constants
import Tools.Constants as Cs
from tudatpy.data import save2txt
import time
import matlab.engine


def get_initial_states() -> np.ndarray:
    """
        Converts the initial state to inertial coordinates.

        The initial state is expressed in Earth-centered Kepler elements.
        These are first converted into Earth-centered cartesian coordinates,
        then they are finally converted in the global (inertial) coordinate
        system.

        Parameters
        ----------

        Returns
        -------
        initial_state_inertial_coordinates : np.ndarray
            The initial state of the vehicles expressed in inertial coordinates.
        """

    # Set initial keplerian elements
    semi_major_axis = 0.63781363000E+07 + Cs.altitude   # Radius taken from GOCO05
    eccentricity = Cs.eccentricity
    inclination = np.deg2rad(Cs.inclination)
    argument_of_periapsis = np.deg2rad(Cs.argument_of_periapsis)
    longitude_of_ascending_node = np.deg2rad(Cs.longitude_of_ascending_node)
    true_anomaly_nggm1 = np.deg2rad(Cs.true_anomaly_nggm1)
    true_anomaly_nggm2 = true_anomaly_nggm1 - Cs.angular_separation
    gravitational_parameter = 0.39860044150e+15         # GM taken from GOCO05

    initial_cartesian_state_inertial_nggm1 = element_conversion.keplerian_to_cartesian_elementwise(
        semi_major_axis, eccentricity, inclination, argument_of_periapsis, longitude_of_ascending_node,
        true_anomaly_nggm1, gravitational_parameter)

    initial_cartesian_state_inertial_nggm2 = element_conversion.keplerian_to_cartesian_elementwise(
        semi_major_axis, eccentricity, inclination, argument_of_periapsis, longitude_of_ascending_node,
        true_anomaly_nggm2, gravitational_parameter)

    return np.concatenate((initial_cartesian_state_inertial_nggm1, initial_cartesian_state_inertial_nggm2))


class SpacecraftDynamics:
    def __init__(self, simulation_start_epoch: float, maximum_duration: float, vehicle_properties: dict):
        """
        Parameters
        ----------
        simulation_start_epoch: float
            Start of the simulation [s] with t=0 at J2000.
        maximum_duration: float
            Duration [s] the simulation will run for.
        """

        self.simulation_start_epoch = simulation_start_epoch
        self.maximum_duration = maximum_duration

        # Vehicle properties
        self.vehicle_mass = vehicle_properties["Mass"]  # kg
        self.vehicle_dry_mass = self.vehicle_mass - 100 # kg. Fuel Mass of 100 kg
        self.specific_impulse = 2000  # s
        self.length = vehicle_properties["Length"]  # m
        self.width = vehicle_properties["Width"]  # m
        self.height = vehicle_properties["Height"]  # m
        self.reference_area = vehicle_properties["Reference_area"]  # m^2
        self.aero_coefficients = vehicle_properties["Aerodynamic_coefficients"]
        self.solar_array_area = 0  # m^2
        self.box_specular_reflectivity = vehicle_properties["Specular_reflection_coefficient"]
        self.box_diffuse_reflectivity = vehicle_properties["Diffuse_reflection_coefficient"]
        self.solar_array_specular_reflectivity = vehicle_properties["Specular_reflection_coefficient"]
        self.solar_array_diffuse_reflectivity = vehicle_properties["Diffuse_reflection_coefficient"]

        # Define the global frame origin and orientation as Earth and J2000
        self.global_frame_origin = "Earth"
        self.global_frame_orientation = 'J2000'

        # Define bodies to create and propagate
        self.bodies_to_create = ["Earth", "Sun", "Moon", "Jupiter"]
        self.bodies_to_propagate = ["NGGM1", "NGGM2"]
        self.central_bodies = ["Earth", "Earth"]

        # Define bodies object
        self.bodies = None

    def get_body_fixed_angular_velocity(self):
        ang_vel = self.bodies.get("NGGM2").inertial_angular_velocity
        return ang_vel

    def define_dependent_variables(self) -> list:
        """
            Retrieves the dependent variables to save.

            Currently, the dependent variables saved include:
            - Keplerian state of NGGMs
            - Rotation matrix from body to inertial frame of reference of NGGM2

            Parameters
            ----------

            Returns
            -------
            dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
                List of dependent variables to save.
            """

        dependent_variable_to_save = [propagation_setup.dependent_variable.inertial_to_body_fixed_rotation_frame("NGGM2"),
                                      propagation_setup.dependent_variable.intermediate_aerodynamic_rotation_matrix_variable("NGGM2",
                                                                                                                             environment.vertical_frame,
                                                                                                                             environment.body_frame,
                                                                                                                             "Earth"),
                                      propagation_setup.dependent_variable.intermediate_aerodynamic_rotation_matrix_variable("NGGM2",
                                                                                                                             environment.vertical_frame,
                                                                                                                             environment.inertial_frame,
                                                                                                                             "Earth"),

                                      propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.aerodynamic_type,
                                                                                               "NGGM2", "Earth"),

                                      propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type,
                                                                                               "NGGM2", "Sun"),
                                      propagation_setup.dependent_variable.single_acceleration(propagation_setup.acceleration.radiation_pressure_type,
                                                                                               "NGGM2", "Earth"),

                                      propagation_setup.dependent_variable.received_irradiance("NGGM2", "Sun"),
                                      propagation_setup.dependent_variable.received_irradiance("NGGM2", "Earth"),

                                      propagation_setup.dependent_variable.keplerian_state("NGGM1", "Earth"),
                                      propagation_setup.dependent_variable.keplerian_state("NGGM2", "Earth"),
                                      propagation_setup.dependent_variable.latitude("NGGM2", "Earth"),
                                      propagation_setup.dependent_variable.longitude("NGGM2", "Earth"),
                                      propagation_setup.dependent_variable.relative_position("Sun", "Earth"),
                                      propagation_setup.dependent_variable.relative_distance("NGGM1", "NGGM2"),
                                      propagation_setup.dependent_variable.altitude("NGGM2", "Earth"),
                                      propagation_setup.dependent_variable.central_body_fixed_cartesian_position("NGGM2", "Earth")]
        return dependent_variable_to_save

    def x_axis_rotation(self, current_time: float) -> float:
        # get the angle between z-axis and direction vector
        r2 = self.bodies.get("NGGM2").state[0:3]
        # print(self.bodies.get("NGGM2").state)
        r2_dir = r2 / np.linalg.norm(r2)

        r1 = self.bodies.get("NGGM1").state[0:3]
        r1_dir = r1 / np.linalg.norm(r1)

        r21_dir = r1_dir - r2_dir

        x_body_frame = np.array([1, 0, 0])
        z_body_frame = np.array([0, 0, 1])
        if current_time == 0:
            return 0

        else:
            R = self.bodies.get("NGGM2").body_fixed_to_inertial_frame
            x_inertial = R.T @ x_body_frame
            z_inertial = R @ z_body_frame
            angle = np.arccos(np.dot(z_inertial, r2_dir))
            self.z_angle.append(-np.rad2deg(angle))
            return -angle

    def define_environment(self, rings: int):
        """
             Defines the environment models by using the default body settings and add the two spacecrafts as empty bodies for propagation.

             Returns
             -------
             bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
             """
        body_settings = environment_setup.get_default_body_settings(self.bodies_to_create, self.global_frame_origin, self.global_frame_orientation)

        # Add the atmosphere model to Earth. Add the NRLMSISE00 model
        body_settings.get("Earth").atmosphere_settings = environment_setup.atmosphere.nrlmsise00()
        body_settings.get("Earth").shape_settings = environment_setup.shape.spherical(0.63781363000E+07)

        # Solar radiation source
        solar_luminosity_settings = environment_setup.radiation_pressure.irradiance_based_constant_luminosity(1361, constants.ASTRONOMICAL_UNIT)
        body_settings.get("Sun").radiation_source_settings = environment_setup.radiation_pressure.isotropic_radiation_source(
            solar_luminosity_settings)

        # Add the albedo and thermal radiation of Earth to the body settings as extended source
        earth_surface_radiosity_models = [
            environment_setup.radiation_pressure.variable_albedo_surface_radiosity(
                environment_setup.radiation_pressure.predefined_knocke_type_surface_property_distribution(
                    environment_setup.radiation_pressure.albedo_knocke), "Sun"),
            environment_setup.radiation_pressure.thermal_emission_blackbody_variable_emissivity(
                emissivity_distribution_model=environment_setup.radiation_pressure.predefined_knocke_type_surface_property_distribution(
                    environment_setup.radiation_pressure.emissivity_knocke),
                original_source_name="Sun")]

        rings_lst = []
        for i in range(rings):
            rings_lst.append(6 * (i+1))
        body_settings.get("Earth").radiation_source_settings = environment_setup.radiation_pressure.panelled_extended_radiation_source(
            earth_surface_radiosity_models, rings_lst)

        # Create an empty body_settings for the spacecrafts
        body_settings.add_empty_settings("NGGM1")
        body_settings.add_empty_settings("NGGM2")
        nggm1_settings = body_settings.get("NGGM1")
        nggm2_settings = body_settings.get("NGGM2")

        # Set mass
        nggm1_settings.constant_mass = self.vehicle_mass
        nggm2_settings.constant_mass = self.vehicle_mass

        # Create aerodynamic coefficient settings
        # CD from crude estimation. Montebruck and Gill (2005). Also DOI: 10.1016/j.asr.2018.10.025
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(reference_area=self.reference_area,
                                                                                        constant_force_coefficient=self.aero_coefficients,
                                                                                        are_coefficients_in_aerodynamic_frame=True,
                                                                                        are_coefficients_in_negative_axis_direction=True
                                                                                        )

        nggm1_settings.aerodynamic_coefficient_settings = aero_coefficient_settings
        nggm2_settings.aerodynamic_coefficient_settings = aero_coefficient_settings

        # Create a dummy rotation model settings to intialise the radiation pressure model
        nggm1_settings.rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(lambda t: np.array([1, 0, 0]),
                                                                                                                    self.global_frame_orientation,
                                                                                                                    "NGGM1_fixed")
        nggm2_settings.rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(lambda t: np.array([1, 0, 0]),
                                                                                                                    self.global_frame_orientation,
                                                                                                                    "NGGM2_fixed")

        ###############################################
        # Add box-wing model for radiation modelling
        ###############################################
        occulting_bodies = {"Sun": ["Earth"]}
        panelled_body_settings = environment_setup.vehicle_systems.box_wing_panelled_body_settings(
            length=self.length,
            width=self.width,
            height=self.height,
            solar_array_area=0,
            box_specular_reflectivity=self.box_specular_reflectivity,
            box_diffuse_reflectivity=self.box_diffuse_reflectivity,
            solar_array_specular_reflectivity=self.solar_array_specular_reflectivity,
            solar_array_diffuse_reflectivity=self.solar_array_diffuse_reflectivity,
        )

        nggm1_settings.vehicle_shape_settings = panelled_body_settings
        nggm2_settings.vehicle_shape_settings = panelled_body_settings
        radiation_pressure_settings = environment_setup.radiation_pressure.panelled_radiation_target(occulting_bodies)

        nggm1_settings.radiation_pressure_target_settings = radiation_pressure_settings
        nggm2_settings.radiation_pressure_target_settings = radiation_pressure_settings

        # Create the System of bodies
        bodies = environment_setup.create_system_of_bodies(body_settings)

        # Add rotation model based on custom inertial direction based
        nggm_attitude_law = AttitudeGuidance(bodies)
        nggm1_rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(nggm_attitude_law.get_nggm2_attitude,
                                                                                                         self.global_frame_orientation,
                                                                                                         "NGGM1_fixed")
        nggm2_rotation_model_settings = environment_setup.rotation_model.custom_inertial_direction_based(nggm_attitude_law.get_nggm2_attitude,
                                                                                                         self.global_frame_orientation,
                                                                                                         "NGGM2_fixed")
        environment_setup.add_rotation_model(bodies, "NGGM1", nggm1_rotation_model_settings)
        environment_setup.add_rotation_model(bodies, "NGGM2", nggm2_rotation_model_settings)

        print("Environment created successfully")
        return bodies

    def define_termination_settings(self):
        """
        Get termination settings for the simulation. The termination is based on time, and mass constraints.

        Termination settings currently include:
        - Simulation time

        Returns
        -------
        termination_settings :  tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
            Propagation termination settings object.
        """
        # Create single PropagationTerminationSettings object(s)
        # Time
        time_termination_settings = propagation_setup.propagator.time_termination(
            self.simulation_start_epoch + self.maximum_duration,
            terminate_exactly_on_final_condition=False
        )

        print("Termination settings created successfully")
        return time_termination_settings

    def define_integrator(self, integrator_index: int, settings_index: float):
        """
            Retrieves the integrator settings.

            It selects a combination of integrator to be used (first argument) and
            the related setting (tolerance for variable step size integrators
            or step size for fixed step size integrators). The code, as provided, runs the following:
            - if j=0,1,2,3: a variable-step-size, multi-stage integrator is used (see multiStageTypes list for specific type),
                             with tolerances 10^(-10+*k)
            - if j=4      : a fixed-step-size RK4 integrator is used, with step-size k
            - if j=5      : a fixed-step-size Euler integrator is used, with step-size k

            Parameters
            ----------
            integrator_index : int
                Index that selects the integrator type as follows:
                    0 -> RK4(5)
                    1 -> RK5(6)
                    2 -> RK7(8)
                    3 -> RKDP7(8)
                    4 -> RK4
                    5 -> Euler
            settings_index : int
                Index that selects the tolerance or the step size (depending on the integrator type).

            Returns
            -------
            integrator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings
                Integrator settings to be provided to the dynamics simulator.
            """

        # Define a list of multi-stage integrators
        multi_stage_integrators = [propagation_setup.integrator.CoefficientSets.rkf_45,
                                   propagation_setup.integrator.CoefficientSets.rkf_56,
                                   propagation_setup.integrator.CoefficientSets.rkf_78,
                                   propagation_setup.integrator.CoefficientSets.rkdp_87]

        fixed_step_size_integrators = [propagation_setup.integrator.CoefficientSets.rk_4,
                                       propagation_setup.integrator.CoefficientSets.euler_forward]

        # Define the step size
        fixed_step_size = 2 ** (settings_index)
        # Define the tolerance
        current_tolerance = 10 ** (-14.0 + settings_index)

        if integrator_index < 4:
            # Select variable step-size integrator
            current_coefficient_set = multi_stage_integrators[integrator_index]
            # Create integrator settings
            integrator = propagation_setup.integrator
            # Here (epsilon, inf) are set as respectively min and max step sizes
            # also note that the relative and absolute tolerances are the same value
            integrator_settings = integrator.runge_kutta_variable_step_size(
                self.simulation_start_epoch,
                1.0,
                current_coefficient_set,
                np.finfo(float).eps,
                np.inf,
                current_tolerance,
                current_tolerance)

        else:
            # Create integrator settings
            integrator = propagation_setup.integrator
            integrator_settings = integrator.runge_kutta_fixed_step_size(
                fixed_step_size, fixed_step_size_integrators[integrator_index - 4])

        print("Integrator settings created successfully")
        return integrator_settings

    def define_propagation(self, bodies, initial_state, dependent_variables, termination_settings,
                           current_propagator=propagation_setup.propagator.cowell, run_model_analysis=False, **kwargs):
        """
            Creates the propagator settings.

            This function creates the propagator settings for translational motion and mass, for the given simulation settings.
            The propagator settings that are returned as output of this function are not yet usable: they do not
            contain any integrator settings, which should be set at a later point by the user

            Parameters
            ----------
            bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
                Collection of the bodies and their settings in the simulation
            initial_state: np.ndarray
                Initial cartesian states both the NGGMs spacecrafts
            dependent_variables: list
                Dependent variables to be given in addition to the states
            termination_settings: tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
                Propagation terminations settings object
            current_propagator : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.TranslationalPropagatorType
                Type of propagator to be used for translational dynamics

            Returns
            -------
            propagator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.MultiTypePropagatorSettings
                Propagator settings to be provided to the dynamics simulator.
            """

        if run_model_analysis:
            # Define accelerations acting on vehicle
            for key, value in kwargs.items():
                if key == "model":
                    if value == 0:
                        # Define accelerations acting on vehicle
                        acceleration_settings_on_nggm1 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()]
                        }
                        acceleration_settings_on_nggm2 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()]
                        }
                    elif value == 1:
                        # Add the 120 D/O gravity field of the Earth
                        acceleration_settings_on_nggm1 = {
                            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(120, 120),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()]
                        }
                        acceleration_settings_on_nggm2 = {
                            'Earth': [propagation_setup.acceleration.spherical_harmonic_gravity(120, 120),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()]
                        }
                    elif value == 2:
                        # Add the gravity of Moon
                        acceleration_settings_on_nggm1 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()],
                            'Moon': [propagation_setup.acceleration.point_mass_gravity()]
                        }
                        acceleration_settings_on_nggm2 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()],
                            'Moon': [propagation_setup.acceleration.point_mass_gravity()]
                        }

                    elif value == 3:
                        # Add the gravity of Sun
                        acceleration_settings_on_nggm1 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure(),
                                    propagation_setup.acceleration.point_mass_gravity()]
                        }
                        acceleration_settings_on_nggm2 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure(),
                                    propagation_setup.acceleration.point_mass_gravity()]
                        }
                    elif value == 4:
                        # Add gravity of Jupiter
                        acceleration_settings_on_nggm1 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()],
                            'Jupiter': [propagation_setup.acceleration.point_mass_gravity()]
                        }
                        acceleration_settings_on_nggm2 = {
                            'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                                      propagation_setup.acceleration.aerodynamic(),
                                      propagation_setup.acceleration.radiation_pressure()
                                      ],
                            'Sun': [propagation_setup.acceleration.radiation_pressure()],
                            'Jupiter': [propagation_setup.acceleration.point_mass_gravity()]
                        }


        else:
            # Define accelerations acting on vehicle
            acceleration_settings_on_nggm1 = {
                'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                          propagation_setup.acceleration.aerodynamic(),
                          propagation_setup.acceleration.radiation_pressure()
                          ],
                'Sun': [propagation_setup.acceleration.radiation_pressure()]
            }
            acceleration_settings_on_nggm2 = {
                'Earth': [propagation_setup.acceleration.point_mass_gravity(),
                          propagation_setup.acceleration.aerodynamic(),
                          propagation_setup.acceleration.radiation_pressure()
                          ],
                'Sun': [propagation_setup.acceleration.radiation_pressure()]
            }

        # Create acceleration models
        acceleration_settings = {"NGGM1": acceleration_settings_on_nggm1,
                                 "NGGM2": acceleration_settings_on_nggm2}
        acceleration_models = propagation_setup.create_acceleration_models(bodies,
                                                                           acceleration_settings,
                                                                           self.bodies_to_propagate,
                                                                           self.central_bodies)

        # dependent_variables.append(propagation_setup.dependent_variable.custom_dependent_variable(self.get_body_fixed_angular_velocity, 3))
        # Create propagation settings for the translational dynamics.
        translational_propagator_settings = propagation_setup.propagator.translational(self.central_bodies,
                                                                                       acceleration_models,
                                                                                       self.bodies_to_propagate,
                                                                                       initial_state,
                                                                                       self.simulation_start_epoch,
                                                                                       None,
                                                                                       termination_settings,
                                                                                       current_propagator,
                                                                                       output_variables=dependent_variables)
        translational_propagator_settings.print_settings.print_dependent_variable_indices = True
        print("Propagator settings created successfully")
        return translational_propagator_settings

    def run_sim(self, bodies, propagation_settings, output_path, return_files=False):
        start_time = time.time()
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(bodies, propagation_settings)
        end_time = time.time()

        state_history = dynamics_simulator.state_history
        unprocessed_state_history = dynamics_simulator.unprocessed_state_history
        dependent_variable_history = dynamics_simulator.dependent_variable_history
        function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
        number_of_function_evaluations = list(function_evaluation_dict.values())[-1]

        dict_to_write = {'Number of function evaluations': number_of_function_evaluations,
                         'Propagation outcome': dynamics_simulator.integration_completed_successfully,
                         'Runtime': end_time - start_time}

        save2txt(state_history, "state_history.dat", output_path)
        save2txt(unprocessed_state_history, "unprocessed_state_history.dat", output_path)
        save2txt(dependent_variable_history, "dependent_variable_history.dat", output_path)
        save2txt(dict_to_write, "ancillary_simulation_info.txt", output_path)

        if return_files:
            return state_history, unprocessed_state_history, dependent_variable_history, dict_to_write


class AttitudeGuidance:
    """
    Class for defining Attitude guidance of the spacecraft

    Attributes
    ----------
    nggm1
    nggm2
    central_body

    Methods
    ------
    get_nggm2_attitude()

    """

    def __init__(self, bodies: environment.SystemOfBodies):
        # Extract the spacecraft and Earth
        self.nggm1 = bodies.get("NGGM1")
        self.nggm2 = bodies.get("NGGM2")
        self.central_body = bodies.get("Earth")

    def get_nggm2_attitude(self, current_time: float) -> np.ndarray:
        if (current_time == current_time):
            nggm1_position = self.nggm1.state[0:3]
            nggm2_position = self.nggm2.state[0:3]

            # LoS vector, directed from NGGM2 to NGGM1
            r21 = nggm1_position - nggm2_position
            r21_dir = r21 / np.linalg.norm(r21)

            # xb_inertial_frame_nggm2 = self.nggm2.state[3:6] / np.linalg.norm(self.nggm2.state[3:6])

            xb_inertial_frame_nggm2 = r21_dir

            return xb_inertial_frame_nggm2


def attitude_correction_matrices(state, deps_var, output_path):
    """
    :param state: state history
    :param deps_var: Dependent variables of the simulation
    :return:
    """
    time = state[:, 0]
    state_nggm2 = state[:, 7:13]

    a_ng = deps_var[:, 28:31] + deps_var[:, 31:34] + deps_var[:, 34:37]

    a_ng_aero = deps_var[:, 28:31]
    a_ng_sun = deps_var[:, 31:34]
    a_ng_earth = deps_var[:, 34:37]

    quaternion_history_RBI = np.zeros((len(time), 4))
    quaternion_history_RIB = np.zeros((len(time), 4))
    dcm_history_RBI = np.zeros((len(time), 9))
    dcm_history_RIB = np.zeros((len(time), 9))
    a_ng_body = np.zeros((len(time), 3))
    a_ng_body_aero = np.zeros((len(time), 3))
    a_ng_body_sun = np.zeros((len(time), 3))
    a_ng_body_earth = np.zeros((len(time), 3))

    R_BV_vector = np.zeros((len(time), 9))
    for i in range(len(time)):
        BI_rot_mat_vector = deps_var[i, 1:10]
        R_BI = np.reshape(BI_rot_mat_vector, (3, 3))
        R_IB = R_BI.T

        x_body = np.array([1, 0, 0])
        y_body = np.array([0, 1, 0])
        z_body = np.array([0, 0, 1])

        ##################################################
        # Calculate the correction angle for the roll axis
        ##################################################
        # 1) Convert the body frame to inertial frame
        # x_inertial = R_IB @ x_body
        # y_inertial = R_IB @ y_body
        z_inertial = R_IB @ z_body

        # 2) Get the angle between the position vector in inertial frame and the z-axis of the body frame
        r = state_nggm2[i, 0:3]
        r_dir = r / np.linalg.norm(r)
        theta_zr = np.arccos(np.dot(r_dir, z_inertial))

        # 3) Calculate the correction angle
        z_corr = (-np.pi - theta_zr)
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(z_corr), np.sin(z_corr)],
                        [0, -np.sin(z_corr), np.cos(z_corr)]])

        R_IB_new = R_IB @ R_x

        # x_inertial_new = R_IB_new @ x_body
        # y_inertial_new = R_IB_new @ y_body
        # z_inertial_new = R_IB_new @ z_body

        ##################################################
        # Convert the rotation matrix to quaternion
        ##################################################
        # Store the rotation from inertial to body frame
        quaternion_history_RIB[i, :] = element_conversion.rotation_matrix_to_quaternion_entries(R_IB_new)
        dcm_history_RIB[i, :] = R_IB_new.flatten()
        # Store the rotation from inertial to body frame
        R_BI_new = R_IB_new.T
        quaternion_history_RBI[i, :] = element_conversion.rotation_matrix_to_quaternion_entries(R_BI_new)
        dcm_history_RBI[i, :] = R_BI_new.flatten()

        # Convert a_ng to body frame
        a_ng_body[i, :] = R_BI_new @ a_ng[i, :]
        a_ng_body_aero[i, :] = R_BI_new @ a_ng_aero[i, :]
        a_ng_body_sun[i, :] = R_BI_new @ a_ng_sun[i, :]
        a_ng_body_earth[i, :] = R_BI_new @ a_ng_earth[i, :]

        # Get the corrected rotation vector from vertical to body frame
        R_IV = np.reshape(deps_var[i, 19:28], (3, 3))
        R_BV = R_BI_new @ R_IV

        R_BV_vector[i, :] = R_BV.flatten()

    # Save the quaternion history in the matlab functions folder
    np.savetxt(output_path + "quaternions_history_RBI.txt", quaternion_history_RBI)
    np.savetxt(output_path + "quaternions_history_RIB.txt", quaternion_history_RIB)
    np.savetxt(output_path + "dcm_history_RBI.txt", dcm_history_RBI)
    np.savetxt(output_path + "dcm_history_RIB.txt", dcm_history_RIB)
    np.savetxt(output_path + "a_ng_body.txt", a_ng_body)
    np.savetxt(output_path + "a_ng_body_aero.txt", a_ng_body_aero)
    np.savetxt(output_path + "a_ng_body_sun.txt", a_ng_body_sun)
    np.savetxt(output_path + "a_ng_body_earth.txt", a_ng_body_earth)

    # Save ecrf position
    pos_ecrf = deps_var[:, -3:]
    np.savetxt(output_path + "pos_ecrf.txt", pos_ecrf)

    # Save R_BV
    np.savetxt(output_path + "R_BV.txt", R_BV_vector)

    print("Attitude correction matrices saved successfully")


def save_gravity_gradients_angular_rates(l_max, ecrf_path, RBV_path, save_path):
    """
    :param l_max: Maximum degree of the gravity field
    :return: Saves the gravity gradients and angular rates in the SimulationOutput/Output folder
    """
    eng = matlab.engine.start_matlab()
    script_folder = r"C:\Users\sruja\Desktop\Astro_class_code\nggm\Matlab_functions\gravity"
    eng.addpath(script_folder, nargout=0)
    eng.calculate_gravity_gradients(matlab.double(l_max), ecrf_path, RBV_path, save_path, nargout=0)

    # Add path of the angular_rates
    script_folder_new = r"C:\Users\sruja\Desktop\Astro_class_code\nggm\Matlab_functions\angular_rates"
    eng.addpath(script_folder_new, nargout=0)
    eng.calculate_angular_rates(nargout=0)
    eng.quit()
    print("Gravity gradients saved as 'gravity_gradients.mat'")
    print("Angular rates saved as 'angular_rates.mat'")
    print("Angular acceleration saved as 'angular_accelerations.mat'")


def generate_benchmarks(benchmark_step_size: float,
                        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings:
                        tudatpy.kernel.numerical_simulation.propagation_setup.propagator.MultiTypePropagatorSettings,
                        are_dependent_variables_present: bool,
                        output_path: str = None):
    """
    Function to generate to accurate benchmarks.

    This function runs two propagations with two different integrator settings that serve as benchmarks for
    the nominal runs. The state and dependent variable history for both benchmarks are returned and, if desired, 
    they are also written to files (to the directory ./SimulationOutput/benchmarks/) in the following way:
    * benchmark_1_states.dat, benchmark_2_states.dat
        The numerically propagated states from the two benchmarks.
    * benchmark_1_dependent_variables.dat, benchmark_2_dependent_variables.dat
        The dependent variables from the two benchmarks.

    Parameters
    ----------
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
    thrust_parameters
        List that represents the thrust parameters for the spacecraft.
    are_dependent_variables_present : bool
        If there are dependent variables to save.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    return_list : list
        List of state and dependent variable history in this order: state_1, state_2, dependent_1_ dependent_2.
    """
    ### CREATION OF THE TWO BENCHMARKS ###
    # Define benchmarks' step sizes
    first_benchmark_step_size = benchmark_step_size  # s
    second_benchmark_step_size = first_benchmark_step_size / 2

    # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
    # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        first_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rk_4)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = True

    print('Running first benchmark...')
    first_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings)

    # Create integrator settings for the second benchmark in the same way
    benchmark_propagator_settings.integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
        second_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rk_4)
    benchmark_propagator_settings.print_settings.print_dependent_variable_indices = False

    print('Running second benchmark...')
    second_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings)

    ### WRITE BENCHMARK RESULTS TO FILE ###
    # Retrieve state history
    first_benchmark_states = first_dynamics_simulator.state_history
    second_benchmark_states = second_dynamics_simulator.state_history
    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_states, f'benchmark_1_states_t={str(first_benchmark_step_size)}.dat', output_path)
        save2txt(second_benchmark_states, 'benchmark_2_states.dat', output_path)
    # Add items to be returned
    return_list = [first_benchmark_states,
                   second_benchmark_states]

    ### DO THE SAME FOR DEPENDENT VARIABLES ###
    if are_dependent_variables_present:
        # Retrieve dependent variable history
        first_benchmark_dependent_variable = first_dynamics_simulator.dependent_variable_history
        second_benchmark_dependent_variable = second_dynamics_simulator.dependent_variable_history
        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable, f'benchmark_1_dependent_variables_t={str(first_benchmark_step_size)}.dat',  output_path)
            save2txt(second_benchmark_dependent_variable,  'benchmark_2_dependent_variables.dat',  output_path)
        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable)
        return_list.append(second_benchmark_dependent_variable)

    return return_list

def compare_benchmarks(first_benchmark: dict,
                       second_benchmark: dict,
                       output_path: str,
                       filename: str) -> dict:
    """
    It compares the results of two benchmark runs.

    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.

    Parameters
    ----------
    first_benchmark : dict
        State (or dependent variable history) from the first benchmark.
    second_benchmark : dict
        State (or dependent variable history) from the second benchmark.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.

    Returns
    -------
    benchmark_difference : dict
        Interpolated difference between the two benchmarks' state (or dependent variable) history.
    """
    # Create 8th-order Lagrange interpolator for first benchmark
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark,  interpolators.lagrange_interpolation(8))
    # Calculate the difference between the benchmarks
    print('Calculating benchmark differences...')
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                             second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference

