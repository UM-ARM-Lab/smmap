import sys
import subprocess


def run_trial(experiment,
              logging_enabled = None,
              test_id = None,
              planning_horizion = None,
              multi_model = None,
              deformability_override = None,
              translational_deformability = None,
              rotational_deformability = None,
              kalman_parameters_override = None,
              process_noise_factor = None,
              observation_noise_factor = None,
              feedback_covariance = None):
    # Constant values that we need
    roslaunch_command = ["roslaunch", "smmap"]

    # Use the correct launch file
    roslaunch_command.append(experiment + ".launch")

    # Setup logging parameters
    if logging_enabled is not None:
        roslaunch_command.append('logging_enabled:=' + logging_enabled)
    if test_id is not None:
        roslaunch_command.append('test_id:=' + test_id)

    # Setup planner parameters
    if planning_horizion is not None:
        roslaunch_command.append('planning_horizion:=' + str(planning_horizion))

    # Setup model parameters
    if deformability_override is not None:
        assert(translational_deformability is not None and rotational_deformability is not None)
        roslaunch_command.append('deformability_override:=' + str(deformability_override))
        roslaunch_command.append('translational_deformability:=' + str(translational_deformability))
        roslaunch_command.append('rotational_deformability:=' + str(rotational_deformability))

    # Setup multi-model parameters
    if multi_model is not None:
        roslaunch_command.append('multi_model:=' + multi_model)
    if kalman_parameters_override is not None:
        assert(process_noise_factor is not None and observation_noise_factor is not None)
        roslaunch_command.append('kalman_parameters_override:=' + kalman_parameters_override)
        roslaunch_command.append('process_noise_factor:=' + str(process_noise_factor))
        roslaunch_command.append('observation_noise_factor:=' + str(observation_noise_factor))

    # Setup simulator parameters
    if feedback_covariance is not None:
        roslaunch_command.append('feedback_covariance:=' + str(feedback_covariance))

    # Add any extra parameters that have been added
    roslaunch_command += sys.argv[1:]
    print " ".join(roslaunch_command), "\n"

    subprocess.call(roslaunch_command)