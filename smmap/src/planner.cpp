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
 * @param model_prediction_fn
 * @param model_suggested_traj_fn
 * @param model_utility_fn
 * @param vis
 */
Planner::Planner( const ErrorFunctionType& error_fn,
                  Visualizer& vis,
                  const double dt )
    : error_fn_( error_fn )
    , dt_( dt )
    , last_model_used_( -1 )
    , generator_( 0xa8710913d2b5df6c ) // a30cd67f3860ddb3 ) // MD5 sum of "Dale McConachie"
//    , generator_( std::chrono::system_clock::now().time_since_epoch().count() )
    , vis_( vis )
{}

void Planner::addModel( DeformableModel::Ptr model )
{
    model_list_.push_back( model );
}

void Planner::createBandits()
{
    model_utility_bandit_ = KalmanFilterMultiarmBandit< std::mt19937_64 >(
                Eigen::VectorXd::Zero( model_list_.size() ),
                Eigen::MatrixXd::Identity( model_list_.size(), model_list_.size() ) );
}

size_t Planner::getLastModelUsed()
{
    return last_model_used_;
}

////////////////////////////////////////////////////////////////////////////////
// The two functions that gets invoked externally
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
AllGrippersPoseTrajectory Planner::getNextTrajectory(
        const WorldState& world_current_state,
        const int planning_horizion,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale )
{
    last_model_used_ = model_utility_bandit_.selectArmToPull( generator_ );

    // Querry each model for it's best trajectory
    ROS_INFO_STREAM_NAMED( "planner", "Getting trajectory suggestion for model "
                           << last_model_used_ << " of length " << planning_horizion );

    const std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > suggested_trajectory =
            model_list_[last_model_used_]->getSuggestedGrippersTrajectory(
                world_current_state,
                planning_horizion,
                dt_,
                max_gripper_velocity,
                obstacle_avoidance_scale );

    AllGrippersPoseTrajectory best_trajectory =
            suggested_trajectory.first;
//            optimizeTrajectoryDirectShooting(
//                world_feedback.back(),
//                suggested_trajectories.first,
//                dt_ );

    return best_trajectory;
}

/**
 * @brief Planner::updateModels
 * @param world_feedback
 * @param weights
 */
void Planner::updateModels(
        const std::vector< WorldState >& world_feedback,
        const Eigen::VectorXd& weights )
{
    if ( last_model_used_ < model_list_.size() )
    {
        const double error_reduction =
                error_fn_( world_feedback.front().object_configuration_ )
                - error_fn_( world_feedback.back().object_configuration_ );

        #warning "Bandit variance prior magic numbers here"
        model_utility_bandit_.updateArms(
                    0.1 * std::abs( error_reduction ) * Eigen::MatrixXd::Identity( model_list_.size(), model_list_.size() ),
                    last_model_used_,
                    error_reduction,
                    0.0 * std::abs( error_reduction ) );
    }

    // Then we allow the model to update itself based on the new data
    #pragma omp parallel for
    for ( size_t model_ind = 0; model_ind < model_list_.size(); model_ind++ )
    {
        model_list_[model_ind]->updateModel( world_feedback );
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal helpers for the getNextTrajectory() function
////////////////////////////////////////////////////////////////////////////////

/*
ObjectTrajectory Planner::combineModelPredictions(
        const VectorObjectTrajectory& model_predictions ) const
{
    assert( model_predictions.size() > 0 );
    assert( model_predictions[0].size() > 0 );

    const std::vector< double >& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate( model_confidences.begin(), model_confidences.end(), 0. );

    ObjectTrajectory weighted_average_trajectory( model_predictions[0].size(),
            ObjectPointSet::Zero( 3, model_predictions[0][0].cols() ) );

    // Itterate through each model prediction
    for ( size_t model_ind = 0; model_ind < model_predictions.size(); model_ind++ )
    {
        // For each model, itterate through time, weighing by the model confidence
        for ( size_t time_ind = 0; time_ind < model_predictions[model_ind].size(); time_ind++ )
        {
            weighted_average_trajectory[time_ind] +=
                    model_predictions[model_ind][time_ind] * model_confidences[model_ind] / total_weight;
        }
    }

    return weighted_average_trajectory;
}

ObjectPointSet Planner::combineModelPredictionsLastTimestep(
        const VectorObjectTrajectory& model_predictions ) const
{
    assert( model_predictions.size() > 0 );
    size_t traj_length = model_predictions[0].size();
    assert( traj_length > 0 );

    const std::vector< double >& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate( model_confidences.begin(), model_confidences.end(), 0. );

    ObjectPointSet weighted_average_configuration = ObjectPointSet::Zero( 3, model_predictions[0][0].cols() );

    // Itterate through each model prediction
    for ( size_t model_ind = 0; model_ind < model_predictions.size(); model_ind++ )
    {
        assert( model_predictions[model_ind].size() == traj_length );

        weighted_average_configuration +=
                model_predictions[model_ind][traj_length-1] * model_confidences[model_ind] / total_weight;
    }

    return weighted_average_configuration;
}

Eigen::VectorXd Planner::combineModelDerivitives(
        const std::vector< Eigen::VectorXd >& model_derivitives ) const
{
    assert( model_derivitives.size() > 0 );

    const std::vector< double >& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate( model_confidences.begin(), model_confidences.end(), 0. );

    Eigen::VectorXd weighted_average_derivitive = Eigen::VectorXd::Zero( model_derivitives[0].size() );

    // Itterate through each model derivitive
    for ( size_t model_ind = 0; model_ind < model_derivitives.size(); model_ind++ )
    {
        weighted_average_derivitive +=
                model_derivitives[model_ind] * model_confidences[model_ind] / total_weight;
    }

    return weighted_average_derivitive;
}

std::pair< Eigen::VectorXd, Eigen::MatrixXd > Planner::combineModelDerivitives(
        const std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >& model_derivitives ) const
{
    assert( model_derivitives.size() > 0 );

    const std::vector< double >& model_confidences = get_model_utility_fn_();
    double total_weight = std::accumulate( model_confidences.begin(), model_confidences.end(), 0. );

    std::pair< Eigen::VectorXd, Eigen::MatrixXd > weighted_average_derivitive(
            Eigen::VectorXd::Zero( model_derivitives[0].first.size() ),
            Eigen::MatrixXd::Zero( model_derivitives[0].second.rows(), model_derivitives[0].second.cols() ) );

    // Itterate through each model derivitive
    for ( size_t model_ind = 0; model_ind < model_derivitives.size(); model_ind++ )
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
        double dt ) const
{
    ROS_INFO_NAMED( "planner" , "Using direct shooting to optimize the trajectory" );

    // TODO: move these magic numbers elsewhere
    #warning "Magic numbers here need to be moved elsewhere"
    const int MAX_ITTR = 1000;
    const double LEARNING_RATE = 0.1;
    const double TOLERANCE = 1e-6;

    double objective_delta = std::numeric_limits< double >::infinity();

    std::vector< AllGrippersSingleVelocity > grippers_velocities =
            CalculateGrippersVelocities( grippers_trajectory, dt );

    std::function< double( const ObjectPointSet& ) > objective_function =
            std::bind( &Task::calculateError, task_.get(), std::placeholders::_1  );

    double objective_value = objective_function(
                combineModelPredictionsLastTimestep(
                     model_set_->makePredictions(
                         current_world_configuration,
                         grippers_trajectory,
                         grippers_velocities,
                         dt ) ) );

    int ittr = 0;
    do
    {
        ROS_INFO_STREAM_NAMED( "planner" , "  Direct shooting itteration " << ittr << ". Current objective value " << objective_value );

        // Find the first derivitive of the objective function with
        // respect to the gripper velocities
        Eigen::VectorXd derivitive =
                combineModelDerivitives(
                    model_set_->getObjectiveFunction1stDerivitive(
                        current_world_configuration,
                        grippers_trajectory,
                        grippers_velocities,
                        dt,
                        objective_function ) );

        Eigen::VectorXd velocity_update = -derivitive;

//        auto derivitives =
//                model_set_->getObjectiveFunction2ndDerivitive(
//                    current_world_configuration,
//                    grippers_trajectory,
//                    grippers_velocities,
//                    dt,
//                    objective_function )[0];

//        // Update the gripper velocities based on a Newton style gradient descent
//        Eigen::VectorXd velocity_update = derivitives.second.colPivHouseholderQr().solve( -derivitives.first );

        // create a new velocity and trajectory to test
        std::vector< AllGrippersSingleVelocity > test_grippers_velocities = grippers_velocities;
        for ( size_t time_ind = 0; time_ind < test_grippers_velocities.size(); time_ind++ )
        {
            for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
            {
                test_grippers_velocities[time_ind][gripper_ind] += LEARNING_RATE *
                        velocity_update.segment< 6 >(
                            (long)( time_ind * grippers_data_.size() * 6 +
                                    gripper_ind * 6 ) );

                if ( GripperVelocity6dNorm( test_grippers_velocities[time_ind][gripper_ind] ) > MAX_GRIPPER_VELOCITY )
                {
                    test_grippers_velocities[time_ind][gripper_ind] *=
                            MAX_GRIPPER_VELOCITY / GripperVelocity6dNorm( test_grippers_velocities[time_ind][gripper_ind] );
                }
            }
        }

        // Update the trajectory of the grippers based on the new velocities
        std::vector< AllGrippersSinglePose > test_grippers_trajectory = CalculateGrippersTrajectory(
                    grippers_trajectory[0],
                    test_grippers_velocities,
                    dt );

        // Calculate the new value of the objective function at the updated velocity
        // locations
        double new_objective_value = objective_function(
                    combineModelPredictionsLastTimestep(
                        model_set_->makePredictions(
                            current_world_configuration,
                            test_grippers_trajectory,
                            test_grippers_velocities,
                            dt ) ) );

        objective_delta = new_objective_value - objective_value;
        objective_value = new_objective_value;

        // TODO: clean up this code to be more efficient
        //       only need to update the result traj after the last step
        if ( objective_delta < TOLERANCE )
        {
            grippers_velocities = test_grippers_velocities;
            grippers_trajectory = test_grippers_trajectory;
        }

        ittr++;
    }
    while ( ittr < MAX_ITTR  && objective_delta < TOLERANCE && std::abs( objective_delta ) > objective_value * TOLERANCE );

    ROS_INFO_STREAM_NAMED( "planner" , "  Direct shooting final objective value " << objective_value );

    return grippers_trajectory;
}
*/
