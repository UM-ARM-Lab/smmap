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

    typedef std::function< std::pair< Eigen::VectorXd, Eigen::VectorXd >(
            const WorldState&                       /* current world state */
            ) >
    TaskDesiredObjectDeltaFunctionType;
}

#endif // TASK_FUNCTION_POINTER_TYPES_H
