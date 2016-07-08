#include "smmap/planner.h"

#include <assert.h>
#include <chrono>
#include <numeric>

#include <arc_utilities/pretty_print.hpp>

#include "smmap/optimization.hpp"
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
        const ErrorFunctionType& error_fn,
        const TaskExecuteGripperTrajectoryFunctionType& execute_trajectory_fn,
        const LoggingFunctionType& logging_fn,
        Visualizer& vis,
        const double dt)
    : error_fn_(error_fn)
    , execute_trajectory_fn_(execute_trajectory_fn)
    , ph_("~")
    , logging_fn_(logging_fn)
    , vis_(vis)
    , dt_(dt)
    , reward_std_dev_scale_factor_(1.0)
    , process_noise_factor_(GetProcessNoiseFactor(ph_))
    , observation_noise_factor_(GetObservationNoiseFactor(ph_))
    , generator_(0xa8710913d2b5df6c) // a30cd67f3860ddb3) // MD5 sum of "Dale McConachie"
//    , generator_(std::chrono::system_clock::now().time_since_epoch().count())
{}

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
std::vector<WorldState> Planner::sendNextTrajectory(
        const WorldState& current_world_state,
        const TaskDesiredObjectDeltaFunctionType& task_desired_object_delta_fn,
        const int planning_horizion,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{
    // Querry each model for it's best trajectory
    ROS_INFO_STREAM_COND_NAMED(planning_horizion != 1, "planner", "Getting trajectory suggestions for each model of length " << planning_horizion);
    stopwatch(RESET);
    std::vector<std::pair<AllGrippersPoseTrajectory, ObjectTrajectory>> suggested_trajectories(num_models_);
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        suggested_trajectories[model_ind] =
            model_list_[model_ind]->getSuggestedGrippersTrajectory(
                current_world_state,
                planning_horizion,
                dt_,
                max_gripper_velocity,
                obstacle_avoidance_scale);

//        ObjectFinalConfigurationPredictionFunctionType prediction_fn = std::bind(
//                    &DeformableModel::getFinalConfiguration,
//                    model_list_[model_ind],
//                    std::placeholders::_1,
//                    std::placeholders::_2,
//                    std::placeholders::_3,
//                    std::placeholders::_4);

//        ErrorFunctionDerivitiveType derivitive_fn = std::bind(
//                    &ErrorFunctionNumericalDerivitive,
//                    std::placeholders::_1,
//                    std::placeholders::_2,
//                    std::placeholders::_3,
//                    error_fn_,
//                    prediction_fn,
//                    std::placeholders::_4);

//        AllGrippersPoseTrajectory optimized_grippers_pose_traj =
//                OptimizeTrajectoryDirectShooting(
//                    current_world_state,
//                    suggested_trajectories[model_ind].first,
//                    error_fn_,
//                    derivitive_fn,
//                    prediction_fn,
//                    max_gripper_velocity * dt_,
//                    dt_ );

//        suggested_trajectories[model_ind].first = optimized_grippers_pose_traj;
//        suggested_trajectories[model_ind].second = model_list_[model_ind]->getPrediction(
//                    current_world_state,
//                    optimized_grippers_pose_traj,
//                    CalculateGrippersPoseDeltas(optimized_grippers_pose_traj),
//                    dt_);
    }
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

    // Measure the time it took to pick a model
    ROS_INFO_STREAM_NAMED("planner", "Calculated model suggestions and picked one in " << stopwatch(READ) << " seconds");

    ROS_INFO_STREAM_COND_NAMED(num_models_ > 1, "planner", "Using model index " << model_to_use);

    AllGrippersPoseTrajectory best_trajectory = suggested_trajectories[(size_t)model_to_use].first;
    best_trajectory.erase(best_trajectory.begin());
    // Execute the trajectory
    ROS_INFO_NAMED("planner", "Sending trajectory to robot");
    std::vector<WorldState> world_feedback = execute_trajectory_fn_(best_trajectory);
    // Get feedback
    world_feedback.emplace(world_feedback.begin(), current_world_state);

    const ObjectDeltaAndWeight task_desired_motion = task_desired_object_delta_fn(current_world_state);

    ROS_INFO_NAMED("planner", "Updating models and logging data");
    updateModels(current_world_state, task_desired_motion, suggested_trajectories, model_to_use, world_feedback);
#ifdef KFRDB_BANDIT
    logging_fn_(world_feedback.back(), model_utility_bandit_.getMean(), model_utility_bandit_.getCovariance(), model_to_use);
#endif
#ifdef KFMANB_BANDIT
    logging_fn_(world_feedback.back(), model_utility_bandit_.getMean(), model_utility_bandit_.getVariance(), model_to_use);
#endif
#ifdef UCB_BANDIT
    logging_fn_(world_feedback.back(), model_utility_bandit_.getMean(), model_utility_bandit_.getUCB(), model_to_use);
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
void Planner::updateModels(
        const WorldState& starting_world_state,
        const ObjectDeltaAndWeight& task_desired_motion,
        const std::vector<std::pair<AllGrippersPoseTrajectory, ObjectTrajectory>>& suggested_trajectories,
        const ssize_t model_used,
        const std::vector<WorldState>& world_feedback)
{
    const double starting_error = error_fn_(starting_world_state.object_configuration_);
    const double true_error_reduction = starting_error - error_fn_(world_feedback.back().object_configuration_);
    reward_std_dev_scale_factor_ = std::max(1e-10, 0.9 * reward_std_dev_scale_factor_ + 0.1 * std::abs(true_error_reduction));
    const double process_noise_scaling_factor = process_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);
    const double observation_noise_scaling_factor = observation_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);

#ifdef KFRDB_BANDIT
    (void)task_desired_motion;

    const Eigen::MatrixXd process_noise = calculateProcessNoise(suggested_trajectories);
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
    (void)suggested_trajectories;
    model_utility_bandit_.updateArms(process_noise_scaling_factor * Eigen::VectorXd::Ones(num_models_), model_used, true_error_reduction, observation_noise_scaling_factor * 1);
#endif
#ifdef UCB_BANDIT
    (void)task_desired_motion;
    (void)suggested_trajectories;
    (void)process_noise_scaling_factor;
    (void)observation_noise_scaling_factor;
    model_utility_bandit_.updateArms(model_used, true_error_reduction);
#endif

    // Then we allow the model to update itself based on the new data
    #pragma omp parallel for
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        model_list_[model_ind]->updateModel(world_feedback);
    }
}

Eigen::MatrixXd Planner::calculateProcessNoise(
        const std::vector<std::pair<AllGrippersPoseTrajectory, ObjectTrajectory>>& suggested_trajectories)
{
    std::vector<double> grippers_velocity_norms((size_t)num_models_);
    std::vector<AllGrippersPoseDeltaTrajectory> grippers_suggested_pose_deltas((size_t)num_models_);
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        grippers_suggested_pose_deltas[model_ind] = CalculateGrippersPoseDeltas(suggested_trajectories[model_ind].first);
        grippers_velocity_norms[model_ind] = MultipleGrippersVelocityTrajectory6dNorm(grippers_suggested_pose_deltas[model_ind]);
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
                        MultipleGrippersVelocityTrajectoryDotProduct(
                            grippers_suggested_pose_deltas[(size_t)i],
                            grippers_suggested_pose_deltas[(size_t)j])
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
        const std::vector<WorldState>& world_feedback)
{
    const double starting_error = error_fn_(starting_world_state.object_configuration_);
    const double true_error_reduction = starting_error - error_fn_(world_feedback.back().object_configuration_);
    const Eigen::VectorXd true_object_diff = CalculateObjectDeltaAsVector(starting_world_state.object_configuration_, world_feedback.back().object_configuration_);
    const double true_object_diff_norm = EigenHelpers::WeightedNorm(true_object_diff, task_desired_motion.weight);

    // TODO: remove this auto
    const auto grippers_trajectory = GetGripperTrajectories(world_feedback);
    const auto grippers_pose_deltas = CalculateGrippersPoseDeltas(grippers_trajectory);

    Eigen::VectorXd angle_between_true_and_predicted = Eigen::VectorXd::Zero(num_models_);
    for (ssize_t model_ind = 0; model_ind < num_models_; model_ind++)
    {
        const ObjectPointSet predicted_motion_under_true_gripper_movement = model_list_[(size_t)model_ind]->getFinalConfiguration(
                    starting_world_state,
                    grippers_trajectory,
                    grippers_pose_deltas,
                    dt_);

        const Eigen::VectorXd predicted_object_diff = CalculateObjectDeltaAsVector(starting_world_state.object_configuration_, predicted_motion_under_true_gripper_movement);
        const double predicted_object_diff_norm = EigenHelpers::WeightedNorm(predicted_object_diff, task_desired_motion.weight);

        // Deal with the cloth not moving potentially (i.e. in fake data land)
        if (true_object_diff_norm > 1e-10 && predicted_object_diff_norm > 1e-10)
        {
            angle_between_true_and_predicted(model_ind) = EigenHelpers::WeightedAngleBetweenVectors(true_object_diff, predicted_object_diff, task_desired_motion.weight);
        }
        else if (true_object_diff_norm <= 1e-10 && predicted_object_diff_norm <= 1e-10)
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

/*
ObjectTrajectory Planner::combineModelPredictions(
        const VectorObjectTrajectory& model_predictions) const
{
    assert(model_predictions.size() > 0);
    assert(model_predictions[0].size() > 0);

    const std::vector<double>& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate(model_confidences.begin(), model_confidences.end(), 0.);

    ObjectTrajectory weighted_average_trajectory(model_predictions[0].size(),
            ObjectPointSet::Zero(3, model_predictions[0][0].cols()));

    // Itterate through each model prediction
    for (size_t model_ind = 0; model_ind < model_predictions.size(); model_ind++)
    {
        // For each model, itterate through time, weighing by the model confidence
        for (size_t time_ind = 0; time_ind < model_predictions[model_ind].size(); time_ind++)
        {
            weighted_average_trajectory[time_ind] +=
                    model_predictions[model_ind][time_ind] * model_confidences[model_ind] / total_weight;
        }
    }

    return weighted_average_trajectory;
}

ObjectPointSet Planner::combineModelPredictionsLastTimestep(
        const VectorObjectTrajectory& model_predictions) const
{
    assert(model_predictions.size() > 0);
    size_t traj_length = model_predictions[0].size();
    assert(traj_length > 0);

    const std::vector<double>& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate(model_confidences.begin(), model_confidences.end(), 0.);

    ObjectPointSet weighted_average_configuration = ObjectPointSet::Zero(3, model_predictions[0][0].cols());

    // Itterate through each model prediction
    for (size_t model_ind = 0; model_ind < model_predictions.size(); model_ind++)
    {
        assert(model_predictions[model_ind].size() == traj_length);

        weighted_average_configuration +=
                model_predictions[model_ind][traj_length-1] * model_confidences[model_ind] / total_weight;
    }

    return weighted_average_configuration;
}

Eigen::VectorXd Planner::combineModelDerivitives(
        const std::vector<Eigen::VectorXd>& model_derivitives) const
{
    assert(model_derivitives.size() > 0);

    const std::vector<double>& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate(model_confidences.begin(), model_confidences.end(), 0.);

    Eigen::VectorXd weighted_average_derivitive = Eigen::VectorXd::Zero(model_derivitives[0].size());

    // Itterate through each model derivitive
    for (size_t model_ind = 0; model_ind < model_derivitives.size(); model_ind++)
    {
        weighted_average_derivitive +=
                model_derivitives[model_ind] * model_confidences[model_ind] / total_weight;
    }

    return weighted_average_derivitive;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> Planner::combineModelDerivitives(
        const std::vector<std::pair<Eigen::VectorXd, Eigen::MatrixXd>>& model_derivitives) const
{
    assert(model_derivitives.size() > 0);

    const std::vector<double>& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate(model_confidences.begin(), model_confidences.end(), 0.);

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> weighted_average_derivitive(
            Eigen::VectorXd::Zero(model_derivitives[0].first.size()),
            Eigen::MatrixXd::Zero(model_derivitives[0].second.rows(), model_derivitives[0].second.cols()));

    // Itterate through each model derivitive
    for (size_t model_ind = 0; model_ind < model_derivitives.size(); model_ind++)
    {
        weighted_average_derivitive.first +=
                model_derivitives[model_ind].first * model_confidences[model_ind] / total_weight;

        weighted_average_derivitive.second +=
                model_derivitives[model_ind].second * model_confidences[model_ind] / total_weight;
    }

    return weighted_average_derivitive;
}
*/
