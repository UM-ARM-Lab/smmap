#include "smmap/task.h"
#include "smmap/ros_params.hpp"

using namespace smmap;

Task::Task( ros::NodeHandle& nh )
    : nh_( nh )
    , ph_( "~" )
    , vis_( nh )
    , task_type_( GetTaskType( nh_ ) )
    , deformable_type_( GetDeformableType( nh_ ) )
    , error_fn_( createErrorFunction() )
    , model_prediction_fn_( createModelPredictionFunction() )
    , model_suggested_grippers_traj_fn_( createModelSuggestedGrippersTrajFunction() )
    , model_get_utility_fn_( createModelGetUtilityFunction() )
    , model_utility_update_fn_( createModelUtilityUpdateFunction() )
    , gripper_collision_check_fn_( createGripperCollisionCheckFunction() )
    , task_desired_object_delta_fn_( createTaskDesiredObjectDeltaFunction() )
    , model_set_( model_utility_update_fn_ )

{}

void Task::execute()
{

}

////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////

ErrorFunctionType Task::createErrorFunction()
{
    if ( deformable_type_ == DeformableType::ROPE && task_type_ == TaskType::COVERAGE )
    {
        return std::bind( &Task::calculateRopeCoverageError, this, std::placeholders::_1 );
    }
    else if ( deformable_type_ == DeformableType::CLOTH && task_type_ == TaskType::COVERAGE )
    {
        return std::bind( &Task::calculateClothCoverageError, this, std::placeholders::_1 );
    }
    else if ( deformable_type_ == DeformableType::CLOTH && task_type_ == TaskType::COLAB_FOLDING )
    {
        return std::bind( &Task::calculateClothColabFoldingError, this, std::placeholders::_1 );
    }
    else
    {
        throw new std::invalid_argument( "Invalid task and deformable pair in createErrorFunction(), this should not be possible" );
    }
}

ModelPredictionFunctionType Task::createModelPredictionFunction()
{
    return std::bind( &ModelSet::getPredictions, &model_set_,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4 );
}

ModelSuggestedGrippersTrajFunctionType Task::createModelSuggestedGrippersTrajFunction()
{
    return std::bind( &ModelSet::getSuggestedGrippersTrajectories, &model_set_,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4,
                      std::placeholders::_5 );
}

ModelGetUtilityFunctionType Task::createModelGetUtilityFunction()
{
    return std::bind( &ModelSet::getModelUtility, &model_set_ );
}

ModelUpdateUtilityFunctionType Task::createModelUtilityUpdateFunction()
{

}

GripperCollisionCheckFunctionType Task::createGripperCollisionCheckFunction()
{

}

TaskDesiredObjectDeltaFunctionType Task::createTaskDesiredObjectDeltaFunction()
{
    if ( deformable_type_ == DeformableType::ROPE && task_type_ == TaskType::COVERAGE )
    {
        return std::bind( &Task::calculateRopeCoverageDesiredDelta, this, std::placeholders::_1 );
    }
    else if ( deformable_type_ == DeformableType::CLOTH && task_type_ == TaskType::COVERAGE )
    {
        return std::bind( &Task::calculateClothCoverageDesiredDelta, this, std::placeholders::_1 );
    }
    else if ( deformable_type_ == DeformableType::CLOTH && task_type_ == TaskType::COLAB_FOLDING )
    {
        return std::bind( &Task::calculateClothColabFoldingDesiredDelta, this, std::placeholders::_1 );
    }
    else
    {
        throw new std::invalid_argument( "Invalid task and deformable pair in createTaskDesiredObjectDeltaFunction(), this should not be possible" );
    }
}
