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
                    const Task& task )
    : ModelSet( grippers_data,
                object_initial_configuration,
                task,
                task.getDeformability(),
                task.getDeformability() )
{}

ModelSet::ModelSet( const std::vector< GripperData >& grippers_data,
                    const ObjectPointSet& object_initial_configuration,
                    const Task& task,
                    const size_t num_models_per_parameter )
    : rnd_generator_( (unsigned long)std::chrono::system_clock::now().time_since_epoch().count() )
{
    assert( num_models_per_parameter == 20 );

    // Initialize model types with their needed data
    DeformableModel::UpdateGrippersData( grippers_data );
    DiminishingRigidityModel::SetInitialObjectConfiguration( object_initial_configuration );

    const double deform_step = 0.5;
    //const double deform_min = std::max( 0., task.getDeformability() - 5 );
    const double deform_min = 10;
    const double deform_max = deform_min + (double)num_models_per_parameter * deform_step;

    for ( double trans_deform = deform_min; trans_deform < deform_max; trans_deform += deform_step )
    {
        for ( double rot_deform = deform_min; rot_deform < deform_max; rot_deform += deform_step )
        {
            addModel( DeformableModel::Ptr( new DiminishingRigidityModel(
                            task,
                            trans_deform,
                            rot_deform,
                            task.getUseRotation(),
                            task.getCollisionScalingFactor(),
                            task.getStretchingScalingThreshold() ) ) );
        }
    }
}

ModelSet::ModelSet( const std::vector< GripperData >& grippers_data,
                    const ObjectPointSet& object_initial_configuration,
                    const Task& task,
                    double translational_deformability,
                    double rotational_deformability )
    : rnd_generator_( (unsigned long)std::chrono::system_clock::now().time_since_epoch().count() )
{
    // Initialize model types with their needed data
    DeformableModel::UpdateGrippersData( grippers_data );
    DiminishingRigidityModel::SetInitialObjectConfiguration( object_initial_configuration );

    addModel( DeformableModel::Ptr( new DiminishingRigidityModel(
                    task,
                    translational_deformability,
                    rotational_deformability,
                    task.getUseRotation(),
                    task.getCollisionScalingFactor(),
                    task.getStretchingScalingThreshold() ) ) );
}

ModelSet::~ModelSet()
{}


VectorObjectTrajectory ModelSet::makePredictions(
        const WorldFeedback& current_world_configuration,
        const std::vector<AllGrippersSinglePose>& grippers_trajectory,
        const std::vector< kinematics::VectorVector6d >& grippers_velocities,
        double dt ) const
{
    VectorObjectTrajectory predictions( model_list_.size() );

    #pragma omp parallel for
    for ( size_t model_ind = 0; model_ind < model_list_.size(); model_ind++ )
    {
        predictions[model_ind] = model_list_[model_ind]->getPrediction(
                    current_world_configuration, grippers_trajectory, grippers_velocities, dt );
    }

    return predictions;
}


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
        const double max_step_size,
        const size_t num_steps )
{
    std::vector< std::pair< std::vector< AllGrippersSinglePose >, double > > grippers_trajectories( model_list_.size() );

    #pragma omp parallel for
    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        std::vector< AllGrippersSinglePose > grippers_trajectory = model_list_[ind]->getDesiredGrippersTrajectory(
                    world_feedback,
                    max_step_size,
                    num_steps );

        grippers_trajectories[ind] = std::pair< std::vector< AllGrippersSinglePose >, double >( grippers_trajectory, model_confidence_[ind] );
    }

    return grippers_trajectories;
}


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
            GetGripperTrajectories( feedback );

    const std::vector< kinematics::VectorVector6d > grippers_velocities =
            GetGripperVelocities( feedback );

    // TODO: deal with the object/gripers not moving at all
    #pragma omp parallel for
    for ( size_t ind = 0; ind < model_list_.size(); ind++ )
    {
        // TODO: remove this terribad constant 'dt' value
        #warning "Constant value here that should be replaced"
        const ObjectTrajectory model_prediction = model_list_[ind]->getPrediction(
                feedback.front(), grippers_trajectory, grippers_velocities, 0.01 );

        const double dist = distanceRMS( feedback, model_prediction );

        model_confidence_[ind] = 1.0 / ( 1.0 + dist );
    }
}

void ModelSet::addModel( DeformableModel::Ptr model )
{
    assert( model_list_.size() == model_confidence_.size() );

    model_list_.push_back( model );
    model_confidence_.push_back( 0 );
}
