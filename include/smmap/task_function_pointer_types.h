#ifndef TASK_FUNCTION_POINTER_TYPES_H
#define TASK_FUNCTION_POINTER_TYPES_H

#include <functional>

#include "smmap/trajectory.hpp"

namespace smmap
{
    typedef std::function<std::vector<CollisionData>(
            const AllGrippersSinglePose& gripper_poses
            )>
    GripperCollisionCheckFunction;

    typedef std::function<void(
            const size_t test_id,
            const WorldState& resulting_world_state
            )>
    TestRobotMotionFeedbackCallback;

    typedef std::function<void(
            const size_t test_id,
            const deformable_manipulation_msgs::TransitionTestResult& test_result
            )>
    GenerateTransitionDataFeedbackCallback;
}

#endif // TASK_FUNCTION_POINTER_TYPES_H
