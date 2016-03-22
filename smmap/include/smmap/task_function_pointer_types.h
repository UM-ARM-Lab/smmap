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

    typedef std::function< std::vector< ObjectTrajectory >(
            const WorldState&,                      /* current state of the world at the start of prediction */
            const AllGrippersPoseTrajectory&,       /* Gripper pose at each timestep */
            const AllGrippersPoseDeltaTrajectory&,  /* Gripper pose delta between timesteps */
            const double dt                         /* time delta between timesteps */
        ) >
    ModelPredictionFunctionType;

    typedef std::function< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory >(
            const size_t,                           /* model index */
            const WorldState&,                      /* current state of the world at the start of planning */
            const int,                              /* planning horizion */
            const double,                           /* dt */
            const double,                           /* max gripper velocity */
            const double                            /* obstacle avoidance scale */
            ) >
    ModelSuggestedGrippersTrajFunctionType;

    typedef std::function< const std::vector< double >&(
            void
            ) >
    GetModelUtilityFunctionType;

    // TODO: this works on a single model, but the get version works on all models
    // Should this be changed?
    typedef std::function< double(
            const size_t,                           /* model index */
            const double,                           /* old utility of the given model*/
            const WorldState&,                      /* world feedback since the last update */
            const ObjectPointSet&,                  /* the prediction of the given model */
            const Eigen::VectorXd&                  /* diagonal weighting matrix */
            ) >
    UpdateModelUtilityFunctionType;

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
