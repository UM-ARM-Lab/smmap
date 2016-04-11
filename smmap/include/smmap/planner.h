#ifndef SMMAP_PLANNER_H
#define SMMAP_PLANNER_H

//#include <arc_utilities/log.hpp>

#include "smmap/deformable_model.h"
#include "smmap/task_function_pointer_types.h"
#include "smmap/visualization_tools.h"
#include "smmap/kalman_filter_multiarm_bandit.hpp"

namespace smmap
{
    class Planner
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructor and model list builder
            ////////////////////////////////////////////////////////////////////

            Planner( const ErrorFunctionType& error_fn,
                     const TaskExecuteGripperTrajectoryFunctionType& execute_trajectory_fn,
                     const LoggingFunctionType& logging_fn,
                     Visualizer& vis,
                     const double dt );

            void addModel( DeformableModel::Ptr model );
            void createBandits();
            size_t getLastModelUsed();

            ////////////////////////////////////////////////////////////////////
            // The two functions that gets invoked repeatedly
            ////////////////////////////////////////////////////////////////////

            // TODO: move/replace this default for obstacle_avoidance_scale
            std::vector< WorldState > sendNextTrajectory(
                    const WorldState& current_world_state,
                    const TaskDesiredObjectDeltaFunctionType& task_desired_object_delta_fn,
                    const int planning_horizion = 1,
                    const double max_gripper_velocity = 0.05/20.0/0.01,
                    const double obstacle_avoidance_scale = 100.0*20.0 );

        private:
            const ErrorFunctionType error_fn_;
            const TaskExecuteGripperTrajectoryFunctionType execute_trajectory_fn_;

            ////////////////////////////////////////////////////////////////////
            // Model list management
            ////////////////////////////////////////////////////////////////////

            // TODO: this is the wrong spot to store this (mentally)
            const double dt_;
            std::vector< DeformableModel::Ptr > model_list_;
            KalmanFilterMultiarmBandit< std::mt19937_64 > model_utility_bandit_;
            std::mt19937_64 generator_;

            void updateModels(
                    const WorldState& starting_world_state,
                    std::pair< Eigen::VectorXd, Eigen::VectorXd > task_desired_motion,
                    const std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory> >& suggested_trajectories,
                    ssize_t model_used,
                    const std::vector< WorldState >& world_feedback );

            VectorObjectTrajectory getPredictions(
                    const WorldState& starting_world_state,
                    const AllGrippersPoseTrajectory& grippers_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory,
                    const double dt ) const;

            ////////////////////////////////////////////////////////////////////
            // Logging and visualization functionality
            ////////////////////////////////////////////////////////////////////

            const LoggingFunctionType logging_fn_;
            Visualizer& vis_;

            ////////////////////////////////////////////////////////////////////
            // Internal helpers for the getNextTrajectory() function
            ////////////////////////////////////////////////////////////////////
/*
            ObjectTrajectory combineModelPredictions(
                    const VectorObjectTrajectory& model_predictions ) const;

            ObjectPointSet combineModelPredictionsLastTimestep(
                    const VectorObjectTrajectory& model_predictions ) const;

            Eigen::VectorXd combineModelDerivitives(
                    const std::vector< Eigen::VectorXd >& model_derivitives ) const;

            std::pair< Eigen::VectorXd, Eigen::MatrixXd > combineModelDerivitives(
                    const std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >& model_derivitives ) const;

            std::vector< AllGrippersSinglePose > optimizeTrajectoryDirectShooting(
                    const WorldFeedback& current_world_configuration,
                    std::vector<AllGrippersSinglePose> grippers_trajectory,
                    double dt,
                    const double max_gripper_velocity ) const;
*/
    };
}

#endif // SMMAP_PLANNER_H
