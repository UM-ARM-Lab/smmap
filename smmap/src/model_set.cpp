#include "smmap/model_set.h"

#include <assert.h>

using namespace smmap;

ModelSet::ModelSet( const UpdateModelUtilityFunctionType& update_model_utility_fn )
    : update_model_utility_fn_( update_model_utility_fn )
{}

void ModelSet::addModel( DeformableModel::Ptr model )
{
    assert( model_list_.size() == model_utility_.size() );

    model_list_.push_back( model );
    model_utility_.push_back( 0 );
}

// TODO: this is currently being invoked from Task, should this be invoked from the planner?
void ModelSet::updateModels( const std::vector< WorldState >& feedback,
                             const Eigen::VectorXd& weights )
{
    // TODO: avoid doing all this recalculation
    const WorldState& starting_world_state = feedback.front();
    const AllGrippersPoseTrajectory& grippers_pose_trajectory = GetGripperTrajectories( feedback );
    const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory = CalculateGrippersPoseDeltas( grippers_pose_trajectory );
    const double dt = 0.01;

    // Allow each model to update itself based on the new data
    #pragma omp parallel for
    for ( size_t model_ind = 0; model_ind < model_list_.size(); model_ind++ )
    {
        const ObjectPointSet prediction = model_list_[model_ind]->getFinalConfiguration(
                    starting_world_state,
                    grippers_pose_trajectory,
                    grippers_pose_delta_trajectory,
                    dt );

        model_utility_[model_ind] = update_model_utility_fn_(
                    model_utility_[model_ind],
                    starting_world_state,
                    prediction,
                    weights );

        model_list_[model_ind]->updateModel( feedback );
    }
}

VectorObjectTrajectory ModelSet::getPredictions(
        const WorldState& starting_world_state,
        const AllGrippersPoseTrajectory& grippers_pose_trajectory,
        const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory,
        const double dt ) const
{
    VectorObjectTrajectory predictions( model_list_.size() );

    #pragma omp parallel for
    for ( size_t model_ind = 0; model_ind < model_list_.size(); model_ind++ )
    {
        predictions[model_ind] = model_list_[model_ind]->getPrediction(
                    starting_world_state,
                    grippers_pose_trajectory,
                    grippers_pose_delta_trajectory,
                    dt );
    }

    return predictions;
}

std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > >
ModelSet::getSuggestedGrippersTrajectories(
        const WorldState& world_current_state,
        const int planning_horizion,
        const double dt,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale,
        const double stretching_correction_threshold ) const
{
    std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > > trajectories( model_list_.size() );

    #pragma omp parallel for
    for ( size_t model_ind = 0; model_ind < model_list_.size(); model_ind++ )
    {
        trajectories[model_ind] = model_list_[model_ind]->getSuggestedGrippersTrajectory(
                    world_current_state,
                    planning_horizion,
                    dt,
                    max_gripper_velocity,
                    obstacle_avoidance_scale,
                    stretching_correction_threshold );
    }

    return trajectories;
}

/*
std::vector< Eigen::VectorXd > ModelSet::getObjectiveFunction1stDerivitive(
        const WorldFeedback& current_world_configuration,
        const std::vector< AllGrippersSinglePose >& grippers_trajectory,
        const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
        double dt,
        std::function< double( const ObjectPointSet& ) > objective_function ) const
{
    std::vector< Eigen::VectorXd > derivitives( model_list_.size() );

    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        derivitives[ind] = model_list_[ind]->getObjectiveFunction1stDerivitive(
                    current_world_configuration,
                    grippers_trajectory,
                    grippers_velocities,
                    dt,
                    objective_function );
    }

    return derivitives;
}

std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > > ModelSet::getObjectiveFunction2ndDerivitive(
        const WorldFeedback& current_world_configuration,
        const std::vector< AllGrippersSinglePose >& grippers_trajectory,
        const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
        double dt,
        std::function< double( const ObjectPointSet& ) > objective_function ) const
{
    std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > > derivitives( model_list_.size() );

    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        derivitives[ind] = model_list_[ind]->getObjectiveFunction2ndDerivitive(
                    current_world_configuration,
                    grippers_trajectory,
                    grippers_velocities,
                    dt,
                    objective_function );
    }

    return derivitives;
}
*/

const std::vector< double >& ModelSet::getModelUtility() const
{
    return model_utility_;
}
