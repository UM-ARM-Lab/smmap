#include "smmap/planner.h"

#include <assert.h>
#include <numeric>
#include <arc_utilities/pretty_print.hpp>

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
    , seed_(GetPlannerSeed(ph_))
    , generator_(seed_)
{
    std::cout << seed_ << std::endl;
    std::cout << seed_ << std::endl;
    std::cout << seed_ << std::endl;
    std::cout << seed_ << std::endl;
}

void Planner::addModel(DeformableModel::Ptr model)
{
    model_list_.push_back(model);
}

void Planner::createBandits()
{
    num_models_ = (ssize_t)model_list_.size();
    ROS_INFO_STREAM_NAMED("planner", "Generating bandits for " << num_models_ << " bandits");

#ifdef KFRDB_BANDIT
    model_utility_bandit_ = KalmanFilterRDB<std::mt19937_64>(
                Eigen::VectorXd::Zero(num_models_),
                Eigen::MatrixXd::Identity(num_models_, num_models_) * 1e6);
#endif
#ifdef KFMANB_BANDIT
    model_utility_bandit_ = KalmanFilterMANB<std::mt19937_64>(
                Eigen::VectorXd::Zero(num_models_),
                Eigen::VectorXd::Ones(num_models_) * 1e6);
#endif
#ifdef UCB_BANDIT
            model_utility_bandit_ = UCB1Normal(num_models_);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// The one function that gets invoked externally
////////////////////////////////////////////////////////////////////////////////

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
    // Pick an arm to use
#ifdef KFRDB_BANDIT
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);
#endif
#ifdef KFMANB_BANDIT
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);
#endif
#ifdef UCB_BANDIT
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull();
#endif
    ROS_INFO_STREAM_COND_NAMED(num_models_ > 1, "planner", "Using model index " << model_to_use);

    const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return task_specification_->calculateDesiredDirection(world_state);
    };

    // Querry each model for it's best trajectory
    stopwatch(RESET);
    std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>> suggested_robot_commands(num_models_);
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        suggested_robot_commands[model_ind] =
            model_list_[model_ind]->getSuggestedGrippersCommand(
                    task_desired_direction_fn,
                    current_world_state,
                    robot_.dt_,
                    robot_.max_gripper_velocity_,
                    task_specification_->collisionScalingFactor());
    }
    // Measure the time it took to pick a model
    ROS_INFO_STREAM_NAMED("planner", "Calculated model suggestions and picked one in " << stopwatch(READ) << " seconds");


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


    AllGrippersSinglePoseDelta selected_command = suggested_robot_commands[(size_t)model_to_use].first;
    // Execute the command
    ROS_INFO_NAMED("planner", "Sending command to robot");
    WorldState world_feedback = robot_.sendGripperCommand(kinematics::applyTwist(current_world_state.all_grippers_single_pose_, selected_command));

    ROS_INFO_NAMED("planner", "Updating models and logging data");
    const ObjectDeltaAndWeight task_desired_motion = task_desired_direction_fn(current_world_state);
    updateModels(current_world_state, task_desired_motion, suggested_robot_commands, model_to_use, world_feedback, individual_rewards);

#ifdef KFRDB_BANDIT
    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getCovariance(), model_to_use, individual_rewards);
#endif
#ifdef KFMANB_BANDIT
    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getVariance(), model_to_use, individual_rewards);
#endif
#ifdef UCB_BANDIT
    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getUCB(), model_to_use, individual_rewards);
#endif

    return world_feedback;
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
        const WorldState& world_feedback,
        const std::vector<double>& individual_rewards)
{
    const double starting_error = task_specification_->calculateError(starting_world_state.object_configuration_);
    const double true_error_reduction = starting_error - task_specification_->calculateError(world_feedback.object_configuration_);
    reward_std_dev_scale_factor_ = std::max(1e-10, 0.9 * reward_std_dev_scale_factor_ + 0.1 * std::abs(true_error_reduction));
    const double process_noise_scaling_factor = process_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);
    const double observation_noise_scaling_factor = observation_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);

#ifdef KFRDB_BANDIT
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
#ifdef KFMANB_BANDIT
    (void)task_desired_motion;
    (void)suggested_commands;
    model_utility_bandit_.updateArms(process_noise_scaling_factor * Eigen::VectorXd::Ones(num_models_), model_used, true_error_reduction, observation_noise_scaling_factor * 1);
#endif
#ifdef UCB_BANDIT
    (void)task_desired_motion;
    (void)suggested_commands;
    (void)process_noise_scaling_factor;
    (void)observation_noise_scaling_factor;
    model_utility_bandit_.updateArms(model_used, true_error_reduction);
#endif

    // Then we allow the model to update itself based on the new data
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

    process_noise += 0.1 * Eigen::MatrixXd::Identity(num_models_, num_models_);

    return process_noise;
}

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
