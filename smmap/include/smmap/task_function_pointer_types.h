#ifndef TASK_FUNCTION_POINTER_TYPES_H
#define TASK_FUNCTION_POINTER_TYPES_H

#include <functional>

#include "smmap/trajectory.hpp"
#include "smmap/visualization_tools.h"

namespace smmap
{
    typedef std::function<double(
            const ObjectPointSet& object_current_state
            )>
    ErrorFunctionType;

    /* Derivitive of error with respect to grippers pose deltas */
    typedef std::function<Eigen::VectorXd(
            const WorldState& world_initial_state,
            const AllGrippersPoseTrajectory& gripper_pose_trajectory,
            const AllGrippersPoseDeltaTrajectory& gripper_pose_delta_trajectory,
            const double dt
            )>
    ErrorFunctionDerivitiveType;

    typedef std::function<std::vector<CollisionData>(
            const AllGrippersSinglePose&            /* Gripper poses to test for collision */
            )>
    GripperCollisionCheckFunctionType;

    typedef std::function<ObjectDeltaAndWeight(
            const WorldState& world_current_state
            )>
    TaskDesiredObjectDeltaFunctionType;

    typedef std::function<ObjectPointSet(
            const WorldState& world_initial_state,
            const AllGrippersPoseTrajectory& gripper_pose_trajectory,
            const AllGrippersPoseDeltaTrajectory& gripper_pose_delta_trajectory,
            const double dt
            )>
    ObjectFinalConfigurationPredictionFunctionType;

    typedef std::function<Eigen::MatrixXd(
            const ObjectPointSet& object_current_state,
            Eigen::VectorXd object_delta
            )>
    TaskObjectDeltaProjectionFunctionType;

    typedef std::function<std::vector<WorldState>(
            const AllGrippersPoseTrajectory& desired_robot_trajectory
            )>
    TaskExecuteGripperTrajectoryFunctionType;

    typedef std::function<void(
            const size_t test_id,
            const WorldState& resulting_world_state
            )>
    TestGrippersPosesFeedbackCallbackFunctionType;

    typedef std::function<bool(
            const std::vector<AllGrippersSinglePose>& pose_sets_to_test,
            const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback
            )>
    TestGrippersPosesFunctionType;

    typedef std::function<void(
            const WorldState& current_world_state,
            const Eigen::VectorXd& model_utility_mean,
            const Eigen::MatrixXd& model_utility_covariance,
            const ssize_t model_used
            )>
    LoggingFunctionType;
}

#endif // TASK_FUNCTION_POINTER_TYPES_H
