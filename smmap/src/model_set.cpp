#include "smmap/model_set.h"

#include <cmath>
#include <chrono>
#include <stdexcept>
#include <assert.h>

#include <kinematics_toolbox/kinematics.h>

#include "smmap/diminishing_rigidity_model.h"

using namespace smmap;

ModelSet::ModelSet( const std::vector< GripperData >& grippers_data,
        const ObjectPointSet& object_initial_configuration,
        const Task& task, double deformability_override )
    : rnd_generator_( (unsigned long)std::chrono::system_clock::now().time_since_epoch().count() )
{
    // Initialize model types with their needed data
    DeformableModel::UpdateGrippersData( grippers_data );
    DiminishingRigidityModel::SetInitialObjectConfiguration( object_initial_configuration );

    // Initialze a diminishing rigidity model
    double deformability;

    if ( deformability_override == -1 )
    {
        deformability = task.getDeformability();
    }
    else if ( deformability_override >= 0 )
    {
        deformability = deformability_override;
    }
    else
    {
        throw new std::invalid_argument( "deformability_override must be -1 or >= 0" );
    }

//    for ( double deformability = 0; deformability <= 15; deformability += 0.5 )
    {
        addModel( DeformableModel::Ptr( new DiminishingRigidityModel(
                        deformability, deformability, task.getUseRotation(),
//                        task.getDeformability(), task.getDeformability()*1.5, task.getUseRotation(),
//                        deformability, deformability*1.5, task.getUseRotation(),
                        task.getCollisionScalingFactor(), task.getStretchingScalingThreshold() ) ) );
    }
}

ModelSet::~ModelSet()
{}

/*
VectorObjectTrajectory ModelSet::makePredictions(
        const WorldFeedback& current_world_configuration,
        const std::vector<AllGrippersSinglePose>& grippers_trajectory ) const
{
    VectorObjectTrajectory predictions( model_list_.size() );
    std::vector< kinematics::VectorVector6d > gripper_velocities =
            calculateGrippersVelocities( grippers_trajectory );

    #pragma omp parallel for
    for ( size_t model_ind = 0; model_ind < model_list_.size(); model_ind++ )
    {
        predictions[model_ind] = model_list_[model_ind]->getPrediction(
                    current_world_configuration, grippers_trajectory, gripper_velocities );
    }

    return predictions;
}
*/

void ModelSet::updateModels( const std::vector< WorldFeedback >& feedback )
{
    // Evaluate our confidence in each model
    evaluateConfidence( feedback );

    // Allow each model to update itself based on the new data
    #pragma omp parallel for
    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        model_list_[ind]->updateModel( feedback );
    }
}

std::vector< std::pair< std::vector< AllGrippersSinglePose >, double > > ModelSet::getDesiredGrippersTrajectories(
        const WorldFeedback& world_feedback,
        const ObjectPointSet& object_desired_configuration,
        double max_step_size, size_t num_steps )
{
    std::vector< std::pair< std::vector< AllGrippersSinglePose >, double > > grippers_trajectories( model_list_.size() );

    #pragma omp parallel for
    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        std::vector< AllGrippersSinglePose > grippers_trajectory = model_list_[ind]->getDesiredGrippersTrajectory(
                    world_feedback,
                    object_desired_configuration,
                    max_step_size,
                    num_steps );

        grippers_trajectories[ind] = std::pair< std::vector< AllGrippersSinglePose >, double >( grippers_trajectory, model_confidence_[ind] );
    }

    return grippers_trajectories;
}

const std::vector< double >& ModelSet::getModelConfidence() const
{
    return model_confidence_;
}

////////////////////////////////////////////////////////////////////////////////
// Private helpers
////////////////////////////////////////////////////////////////////////////////

void ModelSet::evaluateConfidence( const std::vector< WorldFeedback >& feedback )
{
    const std::vector< AllGrippersSinglePose > grippers_trajectory =
            getGripperTrajectories( feedback );

    const std::vector< kinematics::VectorVector6d > grippers_velocities =
            getGripperVelocities( feedback );

    // TODO: deal with the object/gripers not moving at all
    #pragma omp parallel for
    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        const ObjectTrajectory model_prediction = model_list_[ind]->getPrediction(
                feedback.front(), grippers_trajectory, grippers_velocities );

        const double dist = distanceRMS( feedback, model_prediction );

        model_confidence_[ind] = 1.0 / ( 1 + dist );
    }
}

void ModelSet::addModel( DeformableModel::Ptr model )
{
    assert( model_list_.size() == model_confidence_.size() );

    model_list_.push_back( model );
    model_confidence_.push_back( 0 );
}
