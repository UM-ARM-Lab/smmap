#include "smmap/planner.h"

#include <assert.h>
#include <chrono>
#include <numeric>

#include <arc_utilities/pretty_print.hpp>

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
Planner::Planner(const ErrorFunctionType& error_fn,
                  const TaskExecuteGripperTrajectoryFunctionType& execute_trajectory_fn,
                  const LoggingFunctionType& logging_fn,
                  Visualizer& vis,
                  const double dt)
    : error_fn_(error_fn)
    , execute_trajectory_fn_(execute_trajectory_fn)
    , dt_(dt)
    , generator_(0xa8710913d2b5df6c) // a30cd67f3860ddb3) // MD5 sum of "Dale McConachie"
//    , generator_(std::chrono::system_clock::now().time_since_epoch().count())
    , logging_fn_(logging_fn)
    , vis_(vis)
{}

void Planner::addModel(DeformableModel::Ptr model)
{
    model_list_.push_back(model);
}

void Planner::createBandits()
{
    ROS_INFO_STREAM_NAMED("planner", "Generating bandits for " << model_list_.size() << " bandits");
    model_utility_bandit_ = KalmanFilterMultiarmBandit< std::mt19937_64 >(
                Eigen::VectorXd::Zero((ssize_t)model_list_.size()),
                Eigen::MatrixXd::Identity((ssize_t)model_list_.size(), (ssize_t)model_list_.size()));
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
std::vector< WorldState > Planner::sendNextTrajectory(
        const WorldState& current_world_state,
        const TaskDesiredObjectDeltaFunctionType& task_desired_object_delta_fn,
        const int planning_horizion,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{
    // Querry each model for it's best trajectory
    ROS_INFO_STREAM_NAMED("planner", "Getting trajectory suggestions for each model  of length " << planning_horizion);

    std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > > suggested_trajectories(model_list_.size());
    for (size_t model_ind = 0; model_ind < model_list_.size(); model_ind++)
    {
        suggested_trajectories[model_ind] =
            model_list_[model_ind]->getSuggestedGrippersTrajectory(
                current_world_state,
                planning_horizion,
                dt_,
                max_gripper_velocity,
                obstacle_avoidance_scale);
    }

    // Pick an arm to use
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);
    AllGrippersPoseTrajectory best_trajectory = suggested_trajectories[(size_t)model_to_use].first;
    best_trajectory.erase(best_trajectory.begin());
    // Execute the trajectory
    std::vector< WorldState > world_feedback = execute_trajectory_fn_(best_trajectory);
    // Get feedback
    world_feedback.emplace(world_feedback.begin(), current_world_state);

    ObjectDeltaAndWeight task_desired_motion = task_desired_object_delta_fn(current_world_state);

    updateModels(current_world_state, task_desired_motion, suggested_trajectories, model_to_use, world_feedback);
    logging_fn_(world_feedback.back(), model_utility_bandit_.getMean(), model_utility_bandit_.getCovariance(), model_to_use);

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
        ObjectDeltaAndWeight task_desired_motion,
        const std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > >& suggested_trajectories,
        ssize_t model_used,
        const std::vector< WorldState >& world_feedback)
{
    const Eigen::MatrixXd process_noise = calculateProcessNoise(suggested_trajectories);
    const Eigen::VectorXd observed_reward = calculateObservedReward(
                starting_world_state,
                task_desired_motion,
                model_used,
                world_feedback);

    const Eigen::MatrixXd observation_matrix = Eigen::MatrixXd::Identity((ssize_t)model_list_.size(), (ssize_t)model_list_.size());
    const Eigen::MatrixXd observation_noise = calculateObservationNoise(process_noise, model_used);

    // Perform the Kalman update
    #warning "Bandit variance magic numbers here"
    ros::NodeHandle ph("~");
//    const double current_reward_scale_factor = std::sqrt(std::abs(observed_reward(model_used))) + 1e-10;

    // TODO: Make this a low pass filter on abs reward?
    const double current_reward_scale_factor = std::pow(std::abs(observed_reward(model_used)), 1.0) + 1e-10;
    const double process_noise_scaling_factor = ROSHelpers::GetParam(ph, "process_noise_factor", 0.001) * current_reward_scale_factor;
    const double observation_noise_scaling_factor = ROSHelpers::GetParam(ph, "observation_noise_factor", 0.001) * current_reward_scale_factor;
    model_utility_bandit_.updateArms(
                process_noise_scaling_factor * process_noise,
                observation_matrix,
                observed_reward,
                observation_noise_scaling_factor * observation_noise);

//    std::cerr << model_utility_bandit_.getMean().transpose() << std::endl;

    // Then we allow the model to update itself based on the new data
    #pragma omp parallel for
    for (size_t model_ind = 0; model_ind < model_list_.size(); model_ind++)
    {
        model_list_[model_ind]->updateModel(world_feedback);
    }
}

Eigen::MatrixXd Planner::calculateProcessNoise(
        const std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > >& suggested_trajectories)
{
    const ssize_t num_models = (ssize_t)model_list_.size();

    std::vector< double > grippers_velocity_norms((size_t)num_models);
    std::vector< AllGrippersPoseDeltaTrajectory > grippers_suggested_pose_deltas((size_t)num_models);
    for (size_t model_ind = 0; model_ind < (size_t)num_models; model_ind++)
    {
        grippers_suggested_pose_deltas[model_ind] = CalculateGrippersPoseDeltas(suggested_trajectories[model_ind].first);
        grippers_velocity_norms[model_ind] = MultipleGrippersVelocityTrajectory6dNorm(grippers_suggested_pose_deltas[model_ind]);
    }

    Eigen::MatrixXd process_noise = Eigen::MatrixXd::Identity(num_models, num_models);
    for (ssize_t i = 0; i < num_models; i++)
    {
        for (ssize_t j = i+1; j < num_models; j++)
        {
            process_noise(i, j) =
                    MultipleGrippersVelocityTrajectoryDotProduct(
                        grippers_suggested_pose_deltas[(size_t)i],
                        grippers_suggested_pose_deltas[(size_t)j])
                    / (grippers_velocity_norms[(size_t)i] * grippers_velocity_norms[(size_t)j]);

            process_noise(j, i) = process_noise(i, j);
        }
    }

    process_noise += Eigen::MatrixXd::Identity(num_models, num_models);

    return process_noise;
}

Eigen::VectorXd Planner::calculateObservedReward(
        const WorldState& starting_world_state,
        ObjectDeltaAndWeight task_desired_motion,
        ssize_t model_used,
        const std::vector< WorldState >& world_feedback)
{
    const ssize_t num_models = (ssize_t)model_list_.size();

    const double starting_error = error_fn_(starting_world_state.object_configuration_);
    const double true_error_reduction = starting_error - error_fn_(world_feedback.back().object_configuration_);
    const Eigen::VectorXd true_object_diff = CalculateObjectDeltaAsVector(starting_world_state.object_configuration_, world_feedback.back().object_configuration_);

    // TODO: remove this auto
    const auto grippers_trajectory = GetGripperTrajectories(world_feedback);
    const auto grippers_pose_deltas = CalculateGrippersPoseDeltas(grippers_trajectory);


    Eigen::VectorXd angle_between_true_and_predicted = Eigen::VectorXd::Zero(num_models);
    for (ssize_t model_ind = 0; model_ind < num_models; model_ind++)
    {
        const ObjectPointSet predicted_motion_under_true_gripper_movement = model_list_[(size_t)model_ind]->getFinalConfiguration(
                    starting_world_state,
                    grippers_trajectory,
                    grippers_pose_deltas,
                    dt_);

        const Eigen::VectorXd predicted_object_diff = CalculateObjectDeltaAsVector(starting_world_state.object_configuration_, predicted_motion_under_true_gripper_movement);

        // Deal with the cloth not moving potentially (i.e. in fake data land)
        if (true_object_diff.squaredNorm() > 1e-10 && predicted_object_diff.squaredNorm() > 1e-10)
        {
            angle_between_true_and_predicted(model_ind) = EigenHelpers::WeightedAngleBetweenVectors(true_object_diff, predicted_object_diff, task_desired_motion.weight);
        }
        else
        {
            angle_between_true_and_predicted(model_ind) = 0;
        }
    }

    const Eigen::IOFormat CommaSpaceFmt(Eigen::StreamPrecision, 0, ", ", "\n", "", "", "", "");
    const Eigen::IOFormat CommaSpaceSpaceFmt(Eigen::StreamPrecision, 0, ",  ", "\n", "", "", "", "");

//    std::cerr << std::fixed << std::setprecision(6);
//    std::cerr << "Model used: " << model_used << std::endl;
//    std::cerr << "Angles:         " << angle_between_true_and_predicted.transpose().format(CommaSpaceSpaceFmt) << std::endl;
//    std::cerr << "Angle deltas:  ";

    Eigen::VectorXd observed_reward = Eigen::VectorXd::Zero(num_models);
    const double angle_to_model_chosen = angle_between_true_and_predicted(model_used);
    for (ssize_t model_ind = 0; model_ind < num_models; model_ind++)
    {
        const double angle_delta = angle_to_model_chosen - angle_between_true_and_predicted(model_ind);
//        observed_reward(model_ind) = (1.0 + std::cos(angle_delta)) / 2.0;
        observed_reward(model_ind) = true_error_reduction + 1.0 * angle_delta * std::abs(true_error_reduction);

//        if (angle_delta >= 0)
//            std::cerr << " ";
//        std::cerr << angle_delta << ", ";
    }

//    std::cerr << std::endl;

//    if ((observed_reward.array() < 0.0).any())
//    {
//        std::cerr << "Rewards:       " << observed_reward.transpose().format(CommaSpaceFmt) << std::endl;
//    }
//    else
//    {
//        std::cerr << "Rewards:        " << observed_reward.transpose().format(CommaSpaceSpaceFmt) << std::endl;
//    }

    return observed_reward;
}

Eigen::MatrixXd Planner::calculateObservationNoise(
        const Eigen::MatrixXd& process_noise,
        ssize_t model_used)
{
    const ssize_t num_models = (ssize_t)model_list_.size();

    // Observation noise
    Eigen::MatrixXd observation_noise;
    observation_noise.resize(num_models, num_models);
    for (ssize_t i = 0; i < num_models; i++)
    {
        // reuse dot products between true gripper movement and suggested gripper movement - as the model_used gripper movement is the same as the true gipper movement (right now)
        #warning "This term needs to change when we work with a non-perfectly tracking robot"
        observation_noise(i, i) = std::exp(-process_noise(i, model_used));
    }

    for (ssize_t i = 0; i < num_models; i++)
    {
        for (ssize_t j = i + 1; j < num_models; j++)
        {
            // reuse dot products between true gripper movement and suggested gripper movement
            observation_noise(i, j) = process_noise(i, j) * std::sqrt(observation_noise(i, i)) * std::sqrt(observation_noise(j, j));
            observation_noise(j, i) = observation_noise(i, j);
        }
    }

    observation_noise += Eigen::MatrixXd::Identity(num_models, num_models);

    return observation_noise;
}

/*
ObjectTrajectory Planner::combineModelPredictions(
        const VectorObjectTrajectory& model_predictions) const
{
    assert(model_predictions.size() > 0);
    assert(model_predictions[0].size() > 0);

    const std::vector< double >& model_confidences = get_model_utility_fn_();
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

    const std::vector< double >& model_confidences = get_model_utility_fn_();
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
        const std::vector< Eigen::VectorXd >& model_derivitives) const
{
    assert(model_derivitives.size() > 0);

    const std::vector< double >& model_confidences = get_model_utility_fn_();
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

std::pair< Eigen::VectorXd, Eigen::MatrixXd > Planner::combineModelDerivitives(
        const std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >& model_derivitives) const
{
    assert(model_derivitives.size() > 0);

    const std::vector< double >& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate(model_confidences.begin(), model_confidences.end(), 0.);

    std::pair< Eigen::VectorXd, Eigen::MatrixXd > weighted_average_derivitive(
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

std::vector< AllGrippersSinglePose > Planner::optimizeTrajectoryDirectShooting(
        const WorldFeedback& current_world_configuration,
        std::vector< AllGrippersSinglePose > grippers_trajectory,
        double dt) const
{
    ROS_INFO_NAMED("planner" , "Using direct shooting to optimize the trajectory");

    // TODO: move these magic numbers elsewhere
    #warning "Magic numbers here need to be moved elsewhere"
    const int MAX_ITTR = 1000;
    const double LEARNING_RATE = 0.1;
    const double TOLERANCE = 1e-6;

    double objective_delta = std::numeric_limits< double >::infinity();

    std::vector< AllGrippersSingleVelocity > grippers_velocities =
            CalculateGrippersVelocities(grippers_trajectory, dt);

    std::function< double(const ObjectPointSet&) > objective_function =
            std::bind(&Task::calculateError, task_.get(), std::placeholders::_1 );

    double objective_value = objective_function(
                combineModelPredictionsLastTimestep(
                     model_set_->makePredictions(
                         current_world_configuration,
                         grippers_trajectory,
                         grippers_velocities,
                         dt)));

    int ittr = 0;
    do
    {
        ROS_INFO_STREAM_NAMED("planner" , "  Direct shooting itteration " << ittr << ". Current objective value " << objective_value);

        // Find the first derivitive of the objective function with
        // respect to the gripper velocities
        Eigen::VectorXd derivitive =
                combineModelDerivitives(
                    model_set_->getObjectiveFunction1stDerivitive(
                        current_world_configuration,
                        grippers_trajectory,
                        grippers_velocities,
                        dt,
                        objective_function));

        Eigen::VectorXd velocity_update = -derivitive;

//        auto derivitives =
//                model_set_->getObjectiveFunction2ndDerivitive(
//                    current_world_configuration,
//                    grippers_trajectory,
//                    grippers_velocities,
//                    dt,
//                    objective_function)[0];

//        // Update the gripper velocities based on a Newton style gradient descent
//        Eigen::VectorXd velocity_update = derivitives.second.colPivHouseholderQr().solve(-derivitives.first);

        // create a new velocity and trajectory to test
        std::vector< AllGrippersSingleVelocity > test_grippers_velocities = grippers_velocities;
        for (size_t time_ind = 0; time_ind < test_grippers_velocities.size(); time_ind++)
        {
            for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
            {
                test_grippers_velocities[time_ind][gripper_ind] += LEARNING_RATE *
                        velocity_update.segment< 6 >(
                            (long)(time_ind * grippers_data_.size() * 6 +
                                    gripper_ind * 6));

                if (GripperVelocity6dNorm(test_grippers_velocities[time_ind][gripper_ind]) > MAX_GRIPPER_VELOCITY)
                {
                    test_grippers_velocities[time_ind][gripper_ind] *=
                            MAX_GRIPPER_VELOCITY / GripperVelocity6dNorm(test_grippers_velocities[time_ind][gripper_ind]);
                }
            }
        }

        // Update the trajectory of the grippers based on the new velocities
        std::vector< AllGrippersSinglePose > test_grippers_trajectory = CalculateGrippersTrajectory(
                    grippers_trajectory[0],
                    test_grippers_velocities,
                    dt);

        // Calculate the new value of the objective function at the updated velocity
        // locations
        double new_objective_value = objective_function(
                    combineModelPredictionsLastTimestep(
                        model_set_->makePredictions(
                            current_world_configuration,
                            test_grippers_trajectory,
                            test_grippers_velocities,
                            dt)));

        objective_delta = new_objective_value - objective_value;
        objective_value = new_objective_value;

        // TODO: clean up this code to be more efficient
        //       only need to update the result traj after the last step
        if (objective_delta < TOLERANCE)
        {
            grippers_velocities = test_grippers_velocities;
            grippers_trajectory = test_grippers_trajectory;
        }

        ittr++;
    }
    while (ittr < MAX_ITTR  && objective_delta < TOLERANCE && std::abs(objective_delta) > objective_value * TOLERANCE);

    ROS_INFO_STREAM_NAMED("planner" , "  Direct shooting final objective value " << objective_value);

    return grippers_trajectory;
}
*/
