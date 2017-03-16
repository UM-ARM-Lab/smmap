#include "smmap/planner.h"

#include <assert.h>
#include <numeric>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/log.hpp>

//#include "smmap/optimization.hpp"
#include "smmap/timing.hpp"

using namespace smmap;
using namespace EigenHelpersConversions;

////////////////////////////////////////////////////////////////////////////////
// Constructor and model list builder
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::Planner
 * @param error_fn
 * @param execute_trajectory_fn
 * @param vis
 * @param dt
 */
Planner::Planner(
        RobotInterface& robot,
        Visualizer& vis,
        const std::shared_ptr<TaskSpecification>& task_specification,
        const LoggingFunctionType& logging_fn)
    : nh_("")
    , ph_("~")
    , logging_fn_(logging_fn)
    , robot_(robot)
    , vis_(vis)
    , task_specification_(task_specification)
    , calculate_regret_(GetCalculateRegret(ph_))
    , reward_std_dev_scale_factor_(1.0)
    , process_noise_factor_(GetProcessNoiseFactor(ph_))
    , observation_noise_factor_(GetObservationNoiseFactor(ph_))
    , correlation_strength_factor_(GetCorrelationStrengthFactor(ph_))
    , seed_(GetPlannerSeed(ph_))
    , generator_(seed_)
{
    ROS_INFO_STREAM_NAMED("planner", "Using seed " << std::hex << seed_ );

    if (GetLoggingEnabled(nh_))
    {
        const std::string log_folder = GetLogFolder(nh_);
        Log::Log seed_log(log_folder + "seed.txt", false);
        LOG_STREAM(seed_log, std::hex << seed_);
    }
}

void Planner::addModel(DeformableModel::Ptr model)
{
    model_list_.push_back(model);
}

void Planner::createBandits()
{
    num_models_ = (ssize_t)model_list_.size();
    ROS_INFO_STREAM_NAMED("planner", "Generating bandits for " << num_models_ << " bandits");

#ifdef UCB_BANDIT
    model_utility_bandit_ = UCB1Normal<std::mt19937_64>(num_models_);
#endif
#ifdef KFMANB_BANDIT
    model_utility_bandit_ = KalmanFilterMANB<std::mt19937_64>(
                Eigen::VectorXd::Zero(num_models_),
                Eigen::VectorXd::Ones(num_models_) * 1e6);
#endif
#ifdef KFMANDB_BANDIT
    model_utility_bandit_ = KalmanFilterMANDB<std::mt19937_64>(
                Eigen::VectorXd::Zero(num_models_),
                Eigen::MatrixXd::Identity(num_models_, num_models_) * 1e6);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// The one function that gets invoked externally
////////////////////////////////////////////////////////////////////////////////

void Planner::detectFutureConstraintViolations(const WorldState &current_world_state)
{
    ROS_INFO_NAMED("planner", "------------------------------------------------------------------------------------");
    if (task_specification_->is_dijkstras_type_task_)
    {
        ROS_INFO_NAMED("planner", "Starting future constraint violation detection");
        std::shared_ptr<DijkstrasCoverageTask> dijkstras_task = std::dynamic_pointer_cast<DijkstrasCoverageTask>(task_specification_);

        const std::vector<EigenHelpers::VectorVector3d> projected_paths = dijkstras_task->findPathFromObjectToTarget(current_world_state.object_configuration_, dijkstras_task->getErrorThreshold());

        for (size_t path_ind = 0; path_ind < projected_paths.size(); ++path_ind)
        {
            if (projected_paths[path_ind].size() > 1)
            {
                vis_.visualizePoints("projected_point_path", projected_paths[path_ind], Visualizer::Magenta(), (int32_t)path_ind);
            }
            else
            {
                vis_.visualizePoints("projected_point_path", EigenHelpers::VectorVector3d(), Visualizer::Magenta(), (int32_t)path_ind);
            }
        }
    }
    else
    {
        ROS_ERROR_NAMED("planner", "Unable to do future constraint violation detection");
    }
}

/**
 * @brief Planner::getNextTrajectory
 * @param world_current_state
 * @param planning_horizion
 * @param dt
 * @param max_gripper_velocity
 * @param obstacle_avoidance_scale
 * @return
 */
WorldState Planner::sendNextCommand(const WorldState& current_world_state)
{
    ROS_INFO_NAMED("planner", "------------------------------------------------------------------------------------");
    const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return task_specification_->calculateDesiredDirection(world_state);
    };
    const ObjectDeltaAndWeight task_desired_motion = task_desired_direction_fn(current_world_state);
//    visualizeDesiredMotion(current_world_state, task_desired_motion);

    DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, current_world_state, robot_.dt_);

    // Pick an arm to use
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);
    const bool get_action_for_all_models = model_utility_bandit_.generateAllModelActions();
    ROS_INFO_STREAM_COND_NAMED(num_models_ > 1, "planner", "Using model index " << model_to_use);

    // Querry each model for it's best gripper delta
    stopwatch(RESET);
    std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>> suggested_robot_commands(num_models_);
    #pragma omp parallel for
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        if (calculate_regret_ || get_action_for_all_models || (ssize_t)model_ind == model_to_use)
        {
            suggested_robot_commands[model_ind] =
                model_list_[model_ind]->getSuggestedGrippersCommand(
                        model_input_data,
                        robot_.max_gripper_velocity_,
                        task_specification_->collisionScalingFactor());
        }
    }
    // Measure the time it took to pick a model
    ROS_INFO_STREAM_NAMED("planner", "Calculated model suggestions and picked one in " << stopwatch(READ) << " seconds");

    //
    std::vector<double> individual_rewards(num_models_, std::numeric_limits<double>::infinity());
    if (calculate_regret_ && num_models_ > 1)
    {
        stopwatch(RESET);
        const double prev_error = task_specification_->calculateError(current_world_state.object_configuration_);
        const auto test_feedback_fn = [&] (const size_t model_ind, const WorldState& world_state)
        {
            individual_rewards[model_ind] = prev_error - task_specification_->calculateError(world_state.object_configuration_);
            return;
        };

        std::vector<AllGrippersSinglePose> poses_to_test(num_models_);
        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            poses_to_test[model_ind] = kinematics::applyTwist(current_world_state.all_grippers_single_pose_, suggested_robot_commands[model_ind].first);
        }
        robot_.testGrippersPoses(poses_to_test, test_feedback_fn);

        ROS_INFO_STREAM_NAMED("planner", "Collected data to calculate regret in " << stopwatch(READ) << " seconds");
    }


    // Execute the command
    const AllGrippersSinglePoseDelta& selected_command = suggested_robot_commands[(size_t)model_to_use].first;
    ObjectPointSet predicted_object_delta = model_list_[(size_t)model_to_use]->getObjectDelta(model_input_data, selected_command);
    const Eigen::Map<Eigen::VectorXd> predicted_object_delta_as_vector(predicted_object_delta.data(), predicted_object_delta.size());
    ROS_INFO_STREAM_NAMED("planner", "Sending command to robot, action norm:  " << MultipleGrippersVelocity6dNorm(selected_command));
    ROS_INFO_STREAM_NAMED("planner", "Task desired deformable movement norm:  " << EigenHelpers::WeightedNorm(task_desired_motion.delta, task_desired_motion.weight));
    ROS_INFO_STREAM_NAMED("planner", "Task predicted deformable movment norm: " << EigenHelpers::WeightedNorm(predicted_object_delta_as_vector, task_desired_motion.weight));
    WorldState world_feedback = robot_.sendGrippersPoses(kinematics::applyTwist(current_world_state.all_grippers_single_pose_, selected_command));


    ROS_INFO_NAMED("planner", "Updating models and logging data");
    ROS_INFO_STREAM_NAMED("planner", "Correlation strength factor: " << correlation_strength_factor_);
    updateModels(current_world_state, task_desired_motion, suggested_robot_commands, model_to_use, world_feedback);

#ifdef UCB_BANDIT
    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getUCB(), model_to_use, individual_rewards, correlation_strength_factor_);
#endif
#ifdef KFMANB_BANDIT
    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getVariance(), model_to_use, individual_rewards, correlation_strength_factor_);
#endif
#ifdef KFMANDB_BANDIT
    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getCovariance(), model_to_use, individual_rewards, correlation_strength_factor_);
#endif

    return world_feedback;
}

void Planner::visualizeDesiredMotion(const WorldState& current_world_state, const ObjectDeltaAndWeight& desired_motion)
{
    ssize_t num_nodes = current_world_state.object_configuration_.cols();
    std::vector<std_msgs::ColorRGBA> colors((size_t)num_nodes);
    for (size_t node_ind = 0; node_ind < (size_t)num_nodes; node_ind++)
    {
        colors[node_ind].r = (float)desired_motion.weight((ssize_t)node_ind * 3);
        colors[node_ind].g = 0.0f;
        colors[node_ind].b = 0.0f;
        colors[node_ind].a = desired_motion.weight((ssize_t)node_ind * 3) > 0 ? 1.0f : 0.0f;
    }
    task_specification_->visualizeDeformableObject(
            vis_,
            "desired_position",
            AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
            colors);

    if (task_specification_->deformable_type_ == DeformableType::CLOTH)
    {
        vis_.visualizeObjectDelta(
                    "desired_position",
                    current_world_state.object_configuration_,
                    AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
                    Visualizer::Green());
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal helpers for the getNextTrajectory() function
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::updateModels
 * @param suggested_trajectories
 * @param model_used
 * @param world_feedback
 */
void Planner::updateModels(const WorldState& starting_world_state,
        const ObjectDeltaAndWeight& task_desired_motion,
        const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands,
        const ssize_t model_used,
        const WorldState& world_feedback)
{
    // First we update the bandit algorithm
    const double starting_error = task_specification_->calculateError(starting_world_state.object_configuration_);
    const double true_error_reduction = starting_error - task_specification_->calculateError(world_feedback.object_configuration_);
    reward_std_dev_scale_factor_ = std::max(1e-10, 0.9 * reward_std_dev_scale_factor_ + 0.1 * std::abs(true_error_reduction));
    const double process_noise_scaling_factor = process_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);
    const double observation_noise_scaling_factor = observation_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);

#ifdef UCB_BANDIT
    (void)task_desired_motion;
    (void)suggested_commands;
    (void)process_noise_scaling_factor;
    (void)observation_noise_scaling_factor;
    model_utility_bandit_.updateArms(model_used, true_error_reduction);
#endif
#ifdef KFMANB_BANDIT
    (void)task_desired_motion;
    (void)suggested_commands;
    model_utility_bandit_.updateArms(process_noise_scaling_factor * Eigen::VectorXd::Ones(num_models_), model_used, true_error_reduction, observation_noise_scaling_factor * 1.0);
#endif
#ifdef KFMANDB_BANDIT
    (void)task_desired_motion;

    const Eigen::MatrixXd process_noise = calculateProcessNoise(suggested_commands);
//    const Eigen::VectorXd observed_reward = calculateObservedReward(starting_world_state, task_desired_motion, model_used, world_feedback);
//    const Eigen::MatrixXd observation_matrix = Eigen::MatrixXd::Identity(num_models_, num_models_);
//    const Eigen::MatrixXd observation_noise = calculateObservationNoise(process_noise, model_used);

    Eigen::MatrixXd observation_matrix = Eigen::RowVectorXd::Zero(num_models_);
    observation_matrix(0, model_used) = 1.0;
    const Eigen::VectorXd observed_reward = Eigen::VectorXd::Ones(1) * true_error_reduction;
    const Eigen::MatrixXd observation_noise = Eigen::MatrixXd::Ones(1, 1);

    model_utility_bandit_.updateArms(
                process_noise_scaling_factor * process_noise,
                observation_matrix,
                observed_reward,
                observation_noise_scaling_factor * observation_noise);
#endif

    // Then we allow each model to update itself based on the new data
    #pragma omp parallel for
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        model_list_[model_ind]->updateModel(starting_world_state, world_feedback);
    }
}

Eigen::MatrixXd Planner::calculateProcessNoise(const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands)
{
    std::vector<double> grippers_velocity_norms((size_t)num_models_);

    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        grippers_velocity_norms[model_ind] = MultipleGrippersVelocity6dNorm(suggested_commands[model_ind].first);
    }

    Eigen::MatrixXd process_noise = Eigen::MatrixXd::Identity(num_models_, num_models_);
    for (ssize_t i = 0; i < num_models_; i++)
    {
        for (ssize_t j = i+1; j < num_models_; j++)
        {
            if (grippers_velocity_norms[(size_t)i] != 0 &&
                grippers_velocity_norms[(size_t)j] != 0)
            {
                process_noise(i, j) =
                        MultipleGrippersVelocityDotProduct(
                            suggested_commands[(size_t)i].first,
                            suggested_commands[(size_t)j].first)
                        / (grippers_velocity_norms[(size_t)i] * grippers_velocity_norms[(size_t)j]);
            }
            else if (grippers_velocity_norms[(size_t)i] == 0 &&
                     grippers_velocity_norms[(size_t)j] == 0)
            {
                process_noise(i, j) = 1;
            }
            else
            {
                process_noise(i, j) = 0;
            }

            process_noise(j, i) = process_noise(i, j);
        }
    }

    return correlation_strength_factor_ * process_noise + (1.0 - correlation_strength_factor_) * Eigen::MatrixXd::Identity(num_models_, num_models_);
}

/*
Eigen::VectorXd Planner::calculateObservedReward(
        const WorldState& starting_world_state,
        const ObjectDeltaAndWeight& task_desired_motion,
        const ssize_t model_used,
        const WorldState& world_feedback)
{
    const double starting_error = task_specification_->calculateError(starting_world_state.object_configuration_);
    const double true_error_reduction = starting_error - task_specification_->calculateError(world_feedback.object_configuration_);
    const Eigen::VectorXd true_object_diff = CalculateObjectDeltaAsVector(starting_world_state.object_configuration_, world_feedback.object_configuration_);
    const double true_object_delta_norm = EigenHelpers::WeightedNorm(true_object_diff, task_desired_motion.weight);

    // TODO: remove this auto
    const AllGrippersSinglePoseDelta grippers_pose_delta = CalculateGrippersPoseDelta(starting_world_state.all_grippers_single_pose_, world_feedback.all_grippers_single_pose_);

    Eigen::VectorXd angle_between_true_and_predicted = Eigen::VectorXd::Zero(num_models_);
    for (ssize_t model_ind = 0; model_ind < num_models_; model_ind++)
    {
        const ObjectPointSet predicted_object_delta = model_list_[(size_t)model_ind]->getPredictedObjectDelta(
                    starting_world_state,
                    grippers_pose_delta,
                    robot_.dt_);

//        const Eigen::VectorXd predicted_object_diff = CalculateObjectDeltaAsVector(starting_world_state.object_configuration_, predicted_motion_under_true_gripper_movement);
        const double predicted_object_delta_norm = EigenHelpers::WeightedNorm(predicted_object_delta, task_desired_motion.weight);

        // Deal with the cloth not moving potentially (i.e. in fake data land)
        if (predicted_object_delta_norm > 1e-10 && predicted_object_delta_norm > 1e-10)
        {
            angle_between_true_and_predicted(model_ind) = EigenHelpers::WeightedAngleBetweenVectors(true_object_diff, predicted_object_delta, task_desired_motion.weight);
        }
        else if (true_object_delta_norm <= 1e-10 && predicted_object_delta_norm <= 1e-10)
        {
            angle_between_true_and_predicted(model_ind) = 0;
        }
        else
        {
            angle_between_true_and_predicted(model_ind) = M_PI/2.0;
        }
    }

    const Eigen::IOFormat CommaSpaceFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", "");
    const Eigen::IOFormat CommaSpaceSpaceFmt(Eigen::StreamPrecision, 0, ",  ", "\n", "", "", "", "");

    Eigen::VectorXd observed_reward = Eigen::VectorXd::Zero(num_models_);
    const double angle_to_model_chosen = angle_between_true_and_predicted(model_used);
    for (ssize_t model_ind = 0; model_ind < num_models_; model_ind++)
    {
        const double angle_delta = angle_to_model_chosen - angle_between_true_and_predicted(model_ind);
        observed_reward(model_ind) = true_error_reduction + 1.0 * angle_delta * std::abs(true_error_reduction);
    }

    return observed_reward;
}

Eigen::MatrixXd Planner::calculateObservationNoise(
        const Eigen::MatrixXd& process_noise,
        const ssize_t model_used)
{
    // Observation noise
    Eigen::MatrixXd observation_noise;
    observation_noise.resize(num_models_, num_models_);
    for (ssize_t i = 0; i < num_models_; i++)
    {
        // reuse dot products between true gripper movement and suggested gripper movement - as the model_used gripper movement is the same as the true gipper movement (right now)
        #pragma message "This term needs to change when we work with a non-perfectly tracking robot"
        observation_noise(i, i) = std::exp(-process_noise(i, model_used));
    }

    for (ssize_t i = 0; i < num_models_; i++)
    {
        for (ssize_t j = i + 1; j < num_models_; j++)
        {
            // reuse dot products between true gripper movement and suggested gripper movement
            observation_noise(i, j) = process_noise(i, j) * std::sqrt(observation_noise(i, i)) * std::sqrt(observation_noise(j, j));
            observation_noise(j, i) = observation_noise(i, j);
        }
    }

    observation_noise += 0.1 * Eigen::MatrixXd::Identity(num_models_, num_models_);

    return observation_noise;
}
*/
