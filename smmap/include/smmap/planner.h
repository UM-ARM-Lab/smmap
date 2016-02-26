#ifndef SMMAP_PLANNER_H
#define SMMAP_PLANNER_H

//#include <arc_utilities/log.hpp>

#include "smmap/task_function_pointer_types.h"
#include "smmap/visualization_tools.h"

namespace smmap
{
    class Planner
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructor
            ////////////////////////////////////////////////////////////////////

            Planner( const ErrorFunctionType& error_fn,
                     const ModelPredictionFunctionType& model_prediction_fn,
                     const ModelSuggestedGrippersTrajFunctionType& model_suggested_grippers_traj_fn,
                     const GetModelUtilityFunctionType& get_model_utility_fn,
                     Visualizer& vis );

            ////////////////////////////////////////////////////////////////////
            // The one function that gets invoked repeatedly
            ////////////////////////////////////////////////////////////////////

            // TODO: move/replace this default for obstacle_avoidance_scale
            AllGrippersPoseTrajectory getNextTrajectory(
                    const WorldState& world_current_state,
                    const int planning_horizion = 1,
                    const double dt = 0.01,
                    const double max_gripper_velocity = 0.05/20.0/0.01,
                    const double obstacle_avoidance_scale = 100.0*20.0,
                    const double stretching_corection_scale = 0.1/20.0 ) const;

            static double UpdateUtility( const double old_utility,
                                         const WorldState& world_state,
                                         const ObjectPointSet& prediction,
                                         const Eigen::VectorXd& weights );

        private:
            const ErrorFunctionType error_fn_;
            const ModelPredictionFunctionType model_prediction_fn_;
            const ModelSuggestedGrippersTrajFunctionType model_suggested_grippers_traj_fn_;
            const GetModelUtilityFunctionType get_model_utility_fn_;

            ////////////////////////////////////////////////////////////////////
            // Logging and visualization functionality
            ////////////////////////////////////////////////////////////////////

//            const bool logging_enabled_;
//            std::map< std::string, Log::Log > loggers;
            Visualizer& vis_;

            ////////////////////////////////////////////////////////////////////
            // Internal helpers for the getNextTrajectory() function
            ////////////////////////////////////////////////////////////////////

            ObjectTrajectory combineModelPredictions(
                    const VectorObjectTrajectory& model_predictions ) const;

            ObjectPointSet combineModelPredictionsLastTimestep(
                    const VectorObjectTrajectory& model_predictions ) const;

            Eigen::VectorXd combineModelDerivitives(
                    const std::vector< Eigen::VectorXd >& model_derivitives ) const;

            std::pair< Eigen::VectorXd, Eigen::MatrixXd > combineModelDerivitives(
                    const std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >& model_derivitives ) const;

/*
            std::vector< AllGrippersSinglePose > optimizeTrajectoryDirectShooting(
                    const WorldFeedback& current_world_configuration,
                    std::vector<AllGrippersSinglePose> grippers_trajectory,
                    double dt,
                    const double max_gripper_velocity ) const;
*/
    };
}

#endif // SMMAP_PLANNER_H
