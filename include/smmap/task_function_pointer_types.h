#ifndef TASK_FUNCTION_POINTER_TYPES_H
#define TASK_FUNCTION_POINTER_TYPES_H

#include <functional>

#include "smmap/trajectory.hpp"
#include "smmap/visualization_tools.h"

namespace smmap
{
    typedef std::function<std::vector<CollisionData>(
            const AllGrippersSinglePose& gripper_poses
            )>
    GripperCollisionCheckFunctionType;

    typedef std::function<ObjectDeltaAndWeight(
            const WorldState& world_state
            )>
    TaskDesiredObjectDeltaFunctionType;

    typedef std::function<void(
            const size_t test_id,
            const WorldState& resulting_world_state
            )>
    TestGrippersPosesFeedbackCallbackFunctionType;

    // function type for nomad solver
    typedef std::function<double(
        const AllGrippersSinglePoseDelta& test_gripper_motion
        )>
    CostOfTestGrippersMotionFunctionType;

    typedef std::function<double(
        const AllGrippersSinglePoseDelta& test_gripper_motion
        )>
    AllGrippersCollisionConstraintFunctionType;

    typedef std::function<double(
        const AllGrippersSinglePoseDelta& test_gripper_motion
        )>
    AllGrippersStretchingConstraintFunctionType;

    typedef std::function<double(
        const AllGrippersSinglePoseDelta& test_gripper_motion
        )>
    AllGrippersMotionSizeConstraintFunctionType;

}

#endif // TASK_FUNCTION_POINTER_TYPES_H
