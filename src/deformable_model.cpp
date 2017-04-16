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
        throw_arc_exception(std::runtime_error, "You must call SetGrippersData before constructing a DeformableObjectModel");
    }

    if (!function_pointers_initialized_.load())
    {
        throw_arc_exception(std::runtime_error, "You must call SetCallbackFunctions before constructing a DeformableObjectModel");
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
GripperCollisionCheckFunctionType DeformableModel::gripper_collision_check_fn_;

void DeformableModel::SetCallbackFunctions(
        const GripperCollisionCheckFunctionType& gripper_collision_check_fn)
{
    gripper_collision_check_fn_ = gripper_collision_check_fn;
    function_pointers_initialized_.store(true);
}

////////////////////////////////////////////////////////////////////////////////
// Interface wrappers
////////////////////////////////////////////////////////////////////////////////

void DeformableModel::updateModel(
        const WorldState& previous,
        const WorldState& next)
{
    return updateModel_impl(previous, next);
}

ObjectPointSet DeformableModel::getObjectDelta(
        const DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& grippers_pose_delta)
{
    return getObjectDelta_impl(input_data, grippers_pose_delta);
}

std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> DeformableModel::getSuggestedGrippersCommand(
        const DeformableModelInputData& input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{
    return getSuggestedGrippersCommand_impl(input_data, max_gripper_velocity, obstacle_avoidance_scale);
}

// Mengyao defined: Get ObjectDelta as a stacked vector, the above is the resized one
ObjectPointSet DeformableModel::getProjectedObjectDelta(
        const DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& grippers_pose_delta,
        const ObjectPointSet& current_configuration)
{
    return getProjectedObjectDelta_impl(
                input_data,grippers_pose_delta,current_configuration);
}

