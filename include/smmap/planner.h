#ifndef SMMAP_PLANNER_H
#define SMMAP_PLANNER_H

//#include <arc_utilities/log.hpp>

#include "smmap/task_function_pointer_types.h"
#include "smmap/task_specification.h"
#include "smmap/visualization_tools.h"
#include "smmap/robot_interface.hpp"
#include "smmap/deformable_model.h"
#include "smmap/kalman_filter_multiarm_bandit.hpp"
#include "smmap/ucb_multiarm_bandit.hpp"
#include "smmap/virtual_rubber_band.h"

namespace smmap
{
    class Planner
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructor and model list builder
            ////////////////////////////////////////////////////////////////////

            Planner(RobotInterface& robot,
                    Visualizer& vis,
                    const std::shared_ptr<TaskSpecification>& task_specification,
                    const LoggingFunctionType& logging_fn);

            void addModel(DeformableModel::Ptr model);
            void createBandits();

            ////////////////////////////////////////////////////////////////////
            // The two functions that gets invoked repeatedly
            ////////////////////////////////////////////////////////////////////

            WorldState sendNextCommand(
                    const WorldState& current_world_state);

            void visualizeDesiredMotion(
                    const WorldState& current_world_state,
                    const ObjectDeltaAndWeight& desired_motion,
                    const bool visualization_enabled = true);

        private:
            ////////////////////////////////////////////////////////////////////
            // Logging and visualization functionality
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            const LoggingFunctionType logging_fn_;

            RobotInterface& robot_;
            Visualizer& vis_;
            std::shared_ptr<TaskSpecification> task_specification_;
            std::shared_ptr<DijkstrasCoverageTask> dijkstras_task_;

            ////////////////////////////////////////////////////////////////////
            // Model list management
            ////////////////////////////////////////////////////////////////////

            const bool calculate_regret_;
            ssize_t num_models_;
            std::vector<DeformableModel::Ptr> model_list_;
#ifdef UCB_BANDIT
            UCB1Normal<std::mt19937_64> model_utility_bandit_;
#endif
#ifdef KFMANB_BANDIT
            KalmanFilterMANB<std::mt19937_64> model_utility_bandit_;
#endif
#ifdef KFMANDB_BANDIT
            KalmanFilterMANDB<std::mt19937_64> model_utility_bandit_;
#endif
            double reward_std_dev_scale_factor_;
            const double process_noise_factor_;
            const double observation_noise_factor_;
            const double correlation_strength_factor_;
            const unsigned long seed_;
            std::mt19937_64 generator_;


            ////////////////////////////////////////////////////////////////////
            // Constraint violation detection
            ////////////////////////////////////////////////////////////////////

            const size_t num_lookahead_steps_;
            std::shared_ptr<VirtualRubberBand> virtual_rubber_band_between_grippers_;

            void visualizeProjectedPaths(
                    const std::vector<EigenHelpers::VectorVector3d>& projected_paths,
                    const bool visualization_enabled = true);

            bool checkForClothStretchingViolations(
                    const std::vector<EigenHelpers::VectorVector3d>& projected_paths,
                    const bool visualization_enabled = true);

            std::pair<std::vector<EigenHelpers::VectorVector3d>, std::vector<VirtualRubberBand>> detectFutureConstraintViolations(
                    const WorldState& current_world_state,
                    const bool visualization_enabled = true);

            ////////////////////////////////////////////////////////////////////
            // Global gripper planner functions and data
            ////////////////////////////////////////////////////////////////////

            bool executing_global_gripper_trajectory_;
            size_t global_plan_current_timestep_;
            AllGrippersPoseTrajectory global_plan_gripper_trajectory_;

            EigenHelpers::VectorVector3d findPathBetweenPositions(
                    const Eigen::Vector3d& start,
                    const Eigen::Vector3d& goal) const;

            AllGrippersSinglePose getGripperTargets(
                    const WorldState& current_world_state,
                    const std::vector<EigenHelpers::VectorVector3d>& projected_deformable_point_paths) const;

            void planGlobalGripperTrajectory(
                    const WorldState& current_world_state,
                    const std::vector<EigenHelpers::VectorVector3d>& projected_deformable_point_paths,
                    const std::vector<VirtualRubberBand>& projected_virtual_rubber_bands);


            ////////////////////////////////////////////////////////////////////
            // Sending gripper commands
            ////////////////////////////////////////////////////////////////////

            WorldState sendNextCommandUsingLocalController(
                    const WorldState& current_world_state);

            WorldState sendNextCommandUsingGlobalGripperPlannerResults(
                    const WorldState& current_world_state);

            ////////////////////////////////////////////////////////////////////
            // Model utility functions
            ////////////////////////////////////////////////////////////////////

            void updateModels(
                    const WorldState& starting_world_state,
                    const ObjectDeltaAndWeight& task_desired_motion,
                    const std::vector<std::pair<AllGrippersSinglePoseDelta,
                    ObjectPointSet>>& suggested_commands,
                    const ssize_t model_used,
                    const WorldState& world_feedback);

            Eigen::MatrixXd calculateProcessNoise(
                    const std::vector<std::pair<AllGrippersSinglePoseDelta,
                    ObjectPointSet>>& suggested_commands);
    };
}

#endif // SMMAP_PLANNER_H
