import sys
import subprocess


def run_trial(experiment,
              logging_enabled = None,
              test_id = None,
              planning_horizon = None,
              optimization_enabled = None,
              bandit_algorithm = None,
              multi_model = None,
              deformability_override = None,
              translational_deformability = None,
              rotational_deformability = None,
              use_adaptive_model = None,
              adaptive_model_learning_rate = None,
              kalman_parameters_override = None,
              process_noise_factor = None,
              observation_noise_factor = None,
              feedback_covariance = None,
              start_bullet_viewer = 'true',
              calculate_regret = 'false',
              screenshots_enabled = 'true',
              use_random_seed = None):
    # Constant values that we need
    roslaunch_command = ["roslaunch", "smmap"]

    # Use the correct launch file
    roslaunch_command.append("generic_experiment.launch task_type:=" + experiment)

    # Setup logging parameters
    if logging_enabled is not None:
        roslaunch_command.append('logging_enabled:=' + logging_enabled)
    if test_id is not None:
        roslaunch_command.append('test_id:=' + test_id)

    # Setup planner parameters
    if planning_horizon is not None:
        roslaunch_command.append('planning_horizion:=' + str(planning_horizon))

    if optimization_enabled is not None:
        roslaunch_command.append('optimization_enabled:=' + str(optimization_enabled))

    # Setup model parameters
    if deformability_override is not None:
        assert(translational_deformability is not None and rotational_deformability is not None)
        roslaunch_command.append('deformability_override:=' + str(deformability_override))
        roslaunch_command.append('translational_deformability:=' + str(translational_deformability))
        roslaunch_command.append('rotational_deformability:=' + str(rotational_deformability))

    if use_adaptive_model is not None:
        roslaunch_command.append('use_adaptive_model:=' + str(use_adaptive_model))

    if adaptive_model_learning_rate is not None:
        roslaunch_command.append('adaptive_model_learning_rate:=' + str(adaptive_model_learning_rate))

    # Setup multi-model parameters
    if bandit_algorithm is not None:
        roslaunch_command.append('bandit_algorithm:=' + bandit_algorithm)

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

    if start_bullet_viewer is not None:
        roslaunch_command.append('start_bullet_viewer:=' + str(start_bullet_viewer))

    if use_random_seed is not None:
        roslaunch_command.append('use_random_seed:=' + str(use_random_seed))

    roslaunch_command.append('calculate_regret:=' + str(calculate_regret))
    roslaunch_command.append('screenshots_enabled:=' + str(screenshots_enabled))

    # Add any extra parameters that have been added
    roslaunch_command += sys.argv[1:]
    print " ".join(roslaunch_command), "\n"

    subprocess.call(args=" ".join(roslaunch_command), shell=True)
