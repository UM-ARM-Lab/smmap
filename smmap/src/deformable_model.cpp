#include "smmap/deformable_model.h"

#include <arc_utilities/arc_exceptions.hpp>

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Constructor
////////////////////////////////////////////////////////////////////////////////

DeformableModel::DeformableModel()
{
    if (!grippers_data_initialized_.load())
    {
        throw new arc_exceptions::runtime_error("You must call SetGrippersData before constructing a DeformableObjectModel", __FILE__, __LINE__);
    }

    if (!function_pointers_initialized_.load())
    {
        throw new arc_exceptions::runtime_error("You must call SetCallbackFunctions before constructing a DeformableObjectModel", __FILE__, __LINE__);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Static member initialization
////////////////////////////////////////////////////////////////////////////////

std::atomic_bool DeformableModel::grippers_data_initialized_(false);
std::vector<GripperData> DeformableModel::grippers_data_;

void DeformableModel::SetGrippersData(
        const std::vector<GripperData>& grippers_data)
{
    grippers_data_ = grippers_data;

    grippers_data_initialized_.store(true);
}

std::atomic_bool DeformableModel::function_pointers_initialized_(false);
ErrorFunctionType DeformableModel::error_fn_;
GripperCollisionCheckFunctionType DeformableModel::gripper_collision_check_fn_;
TaskDesiredObjectDeltaFunctionType DeformableModel::task_desired_object_delta_fn_;
TaskObjectDeltaProjectionFunctionType DeformableModel::task_object_delta_projection_fn_;

void DeformableModel::SetCallbackFunctions(
        const ErrorFunctionType& error_fn,
        const GripperCollisionCheckFunctionType& gripper_collision_check_fn,
        const TaskDesiredObjectDeltaFunctionType& task_desired_object_delta_fn,
        const TaskObjectDeltaProjectionFunctionType& task_object_delta_projection_fn)
{
    error_fn_                           = error_fn;
    gripper_collision_check_fn_         = gripper_collision_check_fn;
    task_desired_object_delta_fn_       = task_desired_object_delta_fn;
    task_object_delta_projection_fn_    = task_object_delta_projection_fn;

    function_pointers_initialized_.store(true);
}
