#include "smmap/model_set.h"

#include <cmath>
#include <chrono>
#include <assert.h>

#include <kinematics_toolbox/kinematics.h>

#include "smmap/diminishing_rigidity_model.h"

using namespace smmap;

ModelSet::ModelSet( const VectorGrippersData& grippers_data,
        const ObjectPointSet& object_initial_configuration,
        const Task& task )
    : task_( task )
    , object_initial_configuration_( object_initial_configuration )
    , rnd_generator_( (unsigned long)std::chrono::system_clock::now().time_since_epoch().count() )
{
//    for ( double rigidity = 0; rigidity <= 20; rigidity += 0.5 )
    {
        addModel( DeformableModel::Ptr( new DiminishingRigidityModel(
                        grippers_data, object_initial_configuration_,
                        task_.getRigidity(), task_.getUseRotation(),
                        task_.getCollisionScalingFactor(), task_.getStretchingScalingThreshold() ) ) );
    }
}

ModelSet::~ModelSet()
{}

VectorObjectTrajectory ModelSet::makePredictions(
        const AllGrippersTrajectory& grippers_trajectory,
        const ObjectPointSet& object_configuration ) const
{
    VectorObjectTrajectory predictions( model_list_.size() );
    std::vector< kinematics::VectorVector6d > gripper_velocities = calculateGrippersVelocities( grippers_trajectory );

    for ( size_t model_ind = 0; model_ind < model_list_.size(); model_ind++ )
    {
        predictions[model_ind] = model_list_[model_ind]->getPrediction( object_configuration, grippers_trajectory, gripper_velocities );
    }

    return predictions;
}

void ModelSet::updateModels(
        const VectorGrippersData& grippers_data,
        const AllGrippersTrajectory& grippers_trajectory,
        const ObjectTrajectory& object_trajectory )
{
    assert( grippers_trajectory.size() > 0 );
    assert( object_trajectory.size() == grippers_trajectory[0].size() );

    // Do some math to calculate velocities for the grippers and the object
    std::vector< kinematics::VectorVector6d > grippers_velocities =
        calculateGrippersVelocities( grippers_trajectory );

    kinematics::VectorMatrix3Xd object_velocities =
            calculateObjectVelocities( object_trajectory );

    // Evaluate our confidence in each model
    evaluateConfidence( grippers_trajectory, grippers_velocities, object_trajectory );

    // Allow each model to update itself based on the new data
    for ( auto& model: model_list_ )
    {
        model->updateModel( grippers_data,
                grippers_trajectory,
                grippers_velocities,
                object_trajectory,
                object_velocities );
    }
}

std::vector< std::pair< AllGrippersTrajectory, double > > ModelSet::getDesiredGrippersTrajectories(
        const ObjectPointSet& object_current_configuration,
        const ObjectPointSet& object_desired_configuration,
        const VectorGrippersData& grippers_data,
        double max_step_size, size_t num_steps )
{
    std::vector< std::pair< AllGrippersTrajectory, double > > grippers_trajectories;

    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        AllGrippersTrajectory grippers_trajectory = model_list_[ind]->getDesiredGrippersTrajectory(
                    object_current_configuration,
                    object_desired_configuration,
                    grippers_data,
                    max_step_size,
                    num_steps );

        grippers_trajectories.push_back( std::pair< AllGrippersTrajectory, double >( grippers_trajectory, model_confidence_[ind] ) );
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

void ModelSet::evaluateConfidence(
        const AllGrippersTrajectory& grippers_trajectory,
        const std::vector< kinematics::VectorVector6d >& grippers_velocities,
        const ObjectTrajectory& object_trajectory )
{
    // TODO: deal with the object/gripers not moving at all
    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        const ObjectTrajectory model_prediction = model_list_[ind]->getPrediction(
                object_trajectory[0], grippers_trajectory, grippers_velocities );

        const double dist = distanceRMS(  object_trajectory, model_prediction );

        model_confidence_[ind] = 1.0 / ( 1 + dist );
    }
}

std::vector< kinematics::VectorVector6d > ModelSet::calculateGrippersVelocities(
        const AllGrippersTrajectory& grippers_trajectory ) const
{
    std::vector< kinematics::VectorVector6d > grippers_velocities( grippers_trajectory.size() );
    for ( size_t ind = 0; ind < grippers_trajectory.size(); ind++ )
    {
        grippers_velocities[ind] = kinematics::calculateVelocities( grippers_trajectory[ind] );

//        std::cout << "GripperTraj:\n" << PrettyPrint::PrettyPrint( grippers_trajectory[ind], true, "\n\n" ) << std::endl;
//        std::cout << "GripperVel:\n" << PrettyPrint::PrettyPrint( grippers_velocities[ind], true , "\n\n" ) << std::endl;
    }
    return grippers_velocities;
}

kinematics::VectorMatrix3Xd ModelSet::calculateObjectVelocities(
        const ObjectTrajectory& object_trajectory ) const
{
    kinematics::VectorMatrix3Xd object_velocities( object_trajectory.size() - 1, object_trajectory[0] );

    for ( size_t ind = 0; ind < object_trajectory.size() - 1; ind ++ )
    {
        object_velocities[ind] = object_trajectory[ind + 1] - object_trajectory[ind];
    }

    return object_velocities;
}

void ModelSet::addModel( DeformableModel::Ptr model )
{
    assert( model_list_.size() == model_confidence_.size() );

    model_list_.push_back( model );
    model_confidence_.push_back( 0 );
}
