#include "smmap/planner.h"

#include <numeric>
#include <assert.h>

#include <arc_utilities/pretty_print.hpp>

using namespace smmap;
using namespace EigenHelpersConversions;

////////////////////////////////////////////////////////////////////////////////
// Constructor
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
                  const ModelPredictionFunctionType& model_prediction_fn,
                  const ModelSuggestedGrippersTrajFunctionType& model_suggested_grippers_traj_fn,
                  const ModelGetUtilityFunctionType& model_get_utility_fn,
                  Visualizer& vis )
    : error_fn_( error_fn )
    , model_prediction_fn_( model_prediction_fn )
    , model_suggested_grippers_traj_fn_( model_suggested_grippers_traj_fn )
    , model_get_utility_fn_( model_get_utility_fn )
    , vis_( vis )
{}

////////////////////////////////////////////////////////////////////////////////
// The one function that gets invoked externally
////////////////////////////////////////////////////////////////////////////////

AllGrippersPoseTrajectory Planner::getNextTrajectory(
        const WorldState& world_current_state,
        const int planning_horizion,
        const double dt,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale ) const
{
    // Querry each model for it's best trajectory
    ROS_INFO_STREAM_NAMED( "planner", "Getting trajectory suggestions of length " << planning_horizion );
    const std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > > suggested_trajectories =
            model_suggested_grippers_traj_fn_(
                world_current_state,
                planning_horizion,
                max_gripper_velocity,
                dt,
                obstacle_avoidance_scale );

    // Get the utility of each model
    const std::vector< double >& model_utility = model_get_utility_fn_();

    // Confirm that our data sizes match what they are supposed to
    assert( model_utility.size() == suggested_trajectories.size() );

    size_t min_weighted_cost_ind = 0;
    double min_weighted_cost = std::numeric_limits< double >::infinity();
    ROS_INFO_NAMED( "planner" , "Finding 'best' trajectory suggestion for a seed" );
    for ( size_t model_ind = 0; model_ind < suggested_trajectories.size(); model_ind++ )
    {
        double weighted_cost = model_utility[model_ind];
//                * error_fn_( suggested_trajectories[model_ind].second.back() );
        if ( weighted_cost < min_weighted_cost )
        {
            min_weighted_cost = weighted_cost;
            min_weighted_cost_ind = model_ind;
        }
    }

    std::vector< AllGrippersSinglePose > best_trajectory =
            suggested_trajectories[min_weighted_cost_ind].first;
//            optimizeTrajectoryDirectShooting(
//                world_feedback.back(),
//                suggested_trajectories[min_weighted_cost_ind].first,
//                dt );

    // TODO: deal with multiple predictions, which one is the best?
//    VectorObjectTrajectory model_predictions = model_set_->makePredictions(
//                world_feedback.back(),
//                best_trajectory,
//                CalculateGrippersVelocities( best_trajectory, dt ),
//                dt );


//    for ( int time_ind = 0; time_ind < planning_horizion; time_ind++ )
//    {
//        LOG_COND( loggers.at( "model_chosen"), logging_enabled_,
//                  min_weighted_cost_ind );

//        LOG_COND( loggers.at( "object_current_configuration" ), logging_enabled_,
//                  world_feedback[fbk_ind].object_configuration_.format( eigen_io_one_line_ ) );

//        LOG_COND( loggers.at( "suggested_grippers_delta"), logging_enabled_,
//                  PrintDeltaOneLine( suggested_trajectories[min_weighted_cost_ind].first ) );

//        LOG_COND( loggers.at( "object_predicted_configuration" ), logging_enabled_,
//                  model_predictions[min_weighted_cost_ind][fbk_ind].format( eigen_io_one_line_ ) );
//    }

    return best_trajectory;
}

////////////////////////////////////////////////////////////////////////////////
// Internal helpers for the getNextTrajectory() function
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::combineModelPredictions
 * @param model_predictions
 * @return
 */
ObjectTrajectory Planner::combineModelPredictions(
        const VectorObjectTrajectory& model_predictions ) const
{
    assert( model_predictions.size() > 0 );
    assert( model_predictions[0].size() > 0 );

    const std::vector< double >& model_confidences = model_get_utility_fn_();
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

/**
 * @brief Planner::combineModelPredictionsLastTimestep
 * @param model_predictions
 * @return
 */
ObjectPointSet Planner::combineModelPredictionsLastTimestep(
        const VectorObjectTrajectory& model_predictions ) const
{
    assert( model_predictions.size() > 0 );
    size_t traj_length = model_predictions[0].size();
    assert( traj_length > 0 );

    const std::vector< double >& model_confidences = model_get_utility_fn_();
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

/**
 * @brief Planner::combineModelDerivitives
 * @param model_derivitives
 * @return
 */
Eigen::VectorXd Planner::combineModelDerivitives(
        const std::vector< Eigen::VectorXd >& model_derivitives ) const
{
    assert( model_derivitives.size() > 0 );

    const std::vector< double >& model_confidences = model_get_utility_fn_();
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

/**
 * @brief Planner::combineModelDerivitives
 * @param model_derivitives
 * @return
 */
std::pair< Eigen::VectorXd, Eigen::MatrixXd > Planner::combineModelDerivitives(
        const std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >& model_derivitives ) const
{
    assert( model_derivitives.size() > 0 );

    const std::vector< double >& model_confidences = model_get_utility_fn_();
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

/*
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
