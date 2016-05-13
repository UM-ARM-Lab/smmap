#ifndef TASK_FUNCTION_POINTER_TYPES_H
#define TASK_FUNCTION_POINTER_TYPES_H

#include <functional>

#include "smmap/trajectory.hpp"
#include "smmap/visualization_tools.h"

namespace smmap
{
    typedef std::function< double(
            const ObjectPointSet&                   /* Object configuration */
            ) >
    ErrorFunctionType;

    typedef std::function< std::vector< CollisionData >(
            const AllGrippersSinglePose&            /* Gripper poses to test for collision */
            ) >
    GripperCollisionCheckFunctionType;

    typedef std::function< ObjectDeltaAndWeight(
            const WorldState&                       /* current world state */
            ) >
    TaskDesiredObjectDeltaFunctionType;

    typedef std::function< Eigen::MatrixXd(
            const ObjectPointSet&,                  /* current object state */
            Eigen::VectorXd                         /* object delta */
    ) >
    TaskObjectDeltaProjectionFunctionType;

    typedef std::function< std::vector< WorldState >(
            const AllGrippersPoseTrajectory&        /* Desired robot trajectory */
            ) >
    TaskExecuteGripperTrajectoryFunctionType;

    typedef std::function< void(
            const WorldState& current_world_state,
            const Eigen::VectorXd& model_utility_mean,
            const Eigen::MatrixXd& model_utility_covariance,
            const ssize_t model_used
            ) >
    LoggingFunctionType;
}

#endif // TASK_FUNCTION_POINTER_TYPES_H
