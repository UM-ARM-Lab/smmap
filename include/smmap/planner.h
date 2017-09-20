#ifndef SMMAP_PLANNER_H
#define SMMAP_PLANNER_H

#include <arc_utilities/log.hpp>

#include "smmap/task_function_pointer_types.h"
#include "smmap/task_specification.h"
#include "smmap/visualization_tools.h"
#include "smmap/robot_interface.hpp"
#include "smmap/deformable_model.h"
#include "smmap/deformable_controller.hpp"
#include "smmap/kalman_filter_multiarm_bandit.hpp"
#include "smmap/ucb_multiarm_bandit.hpp"
#include "smmap/virtual_rubber_band.h"
#include "smmap/rrt_helper.h"

namespace smmap
{
    class Planner
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructor and the one function that gets called externally
            ////////////////////////////////////////////////////////////////////

            Planner(RobotInterface& robot,
                    Visualizer& vis,
                    const std::shared_ptr<TaskSpecification>& task_specification);

            void execute();

        private:            
            ////////////////////////////////////////////////////////////////////
            // Multipurpose
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            const unsigned long seed_;
            std::mt19937_64 generator_;

            RobotInterface& robot_;
            std::shared_ptr<TaskSpecification> task_specification_;
            std::shared_ptr<DijkstrasCoverageTask> dijkstras_task_;

            ////////////////////////////////////////////////////////////////////
            // Sending gripper commands
            ////////////////////////////////////////////////////////////////////

            WorldState sendNextCommand(
                    const WorldState& current_world_state);

            WorldState sendNextCommandUsingLocalController(
                    const WorldState& current_world_state);

            WorldState sendNextCommandUsingGlobalGripperPlannerResults(
                    const WorldState& current_world_state);

            // Helper function to force some task type use only local controller
            // --- Added by Mengyao
            bool canUseGlobalPlanner();


            ////////////////////////////////////////////////////////////////////
            // Constraint violation detection
            ////////////////////////////////////////////////////////////////////

            void visualizeProjectedPaths(
                    const std::vector<EigenHelpers::VectorVector3d>& projected_paths,
                    const bool visualization_enabled = true);

            bool checkForClothStretchingViolations(
                    const std::vector<EigenHelpers::VectorVector3d>& projected_paths,
                    const bool visualization_enabled = true);

            std::pair<std::vector<EigenHelpers::VectorVector3d>, std::vector<VirtualRubberBand>> detectFutureConstraintViolations(
                    const WorldState& current_world_state,
                    const bool visualization_enabled = true);

            bool globalPlannerNeededDueToOverstretch(
                    const WorldState& current_world_state);

            bool globalPlannerNeededDueToLackOfProgress();

            ////////////////////////////////////////////////////////////////////
            // Global gripper planner functions
            ////////////////////////////////////////////////////////////////////

            AllGrippersPoseTrajectory convertRRTResultIntoGripperTrajectory(
                    const AllGrippersSinglePose& starting_poses,
                    const std::vector<RRTConfig, RRTAllocator>& rrt_result) const;

            AllGrippersSinglePose getGripperTargets(
                    const WorldState& world_state);

            void planGlobalGripperTrajectory(
                    const WorldState& world_state);

            ////////////////////////////////////////////////////////////////////
            // Model list management
            ////////////////////////////////////////////////////////////////////

            void initializeModelAndControllerSet(const WorldState& initial_world_state);
            void addModel(DeformableModel::Ptr model);
            void createBandits();

            // Added by Mengyao. --- Initialize Max Grippers Distance, for 2 grippers only
            void initializeGrippersMaxDistance();

            const bool calculate_regret_;
            ssize_t num_models_;
            std::vector<DeformableModel::Ptr> model_list_;
            std::vector<DeformableController::Ptr> controller_list_;

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

            ////////////////////////////////////////////////////////////////////
            // Model utility functions
            ////////////////////////////////////////////////////////////////////

            void updateModels(
                    const WorldState& starting_world_state,
                    const ObjectDeltaAndWeight& task_desired_motion,
                    const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands,
                    const ssize_t model_used,
                    const WorldState& world_feedback);

            Eigen::MatrixXd calculateProcessNoise(
                    const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands) const;

            ////////////////////////////////////////////////////////////////////
            // Constraint violation and global planner data
            ////////////////////////////////////////////////////////////////////

            const bool enable_stuck_detection_;
            std::shared_ptr<VirtualRubberBand> virtual_rubber_band_between_grippers_;
            std::vector<ssize_t> path_between_grippers_through_object_;
            const size_t max_lookahead_steps_;
            const size_t max_grippers_pose_history_length_;
            AllGrippersPoseTrajectory grippers_pose_history_;
            std::vector<double> error_history_;

            bool executing_global_gripper_trajectory_;
            size_t global_plan_current_timestep_;
            AllGrippersPoseTrajectory global_plan_gripper_trajectory_;
            std::unique_ptr<RRTHelper> rrt_helper_;

            const Eigen::MatrixXd object_initial_node_distance_;
            double max_grippers_distance_;
            int controller_count_;

            ////////////////////////////////////////////////////////////////////
            // Logging and visualization functionality
            ////////////////////////////////////////////////////////////////////

            void visualizeDesiredMotion(
                    const WorldState& current_world_state,
                    const ObjectDeltaAndWeight& desired_motion,
                    const bool visualization_enabled = true) const;

            // Visulize Force on Gripper  --- Added by Mengyao
            void visualizeTotalForceOnGripper(
                    const WorldState& current_world_state,
                    const bool visualization_enabled = true) const;

            void visualize_gripper_motion(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta& gripper_motion,
                    const ssize_t model_ind);

            void initializeLogging();

            void initializeControllerLogging();

            void logData(
                    const WorldState& current_world_state,
                    const Eigen::VectorXd& model_utility_mean,
                    const Eigen::MatrixXd& model_utility_covariance,
                    const ssize_t model_used,
                    const std::vector<double>& rewards_for_all_models);

            // Contoller logger.  --- Added by Mengyao
            void controllerLogData(const WorldState& current_world_state,
                    const std::vector<double>& ave_contol_error,
                    const std::vector<double> current_stretching_factor,
                    const std::vector<double> num_stretching_violation);

            const bool logging_enabled_;
            const bool controller_logging_enabled_;
            std::unordered_map<std::string, Log::Log> loggers_;
            std::unordered_map<std::string, Log::Log> controller_loggers_;

            Visualizer& vis_;
            const bool visualize_desired_motion_;
            const bool visualize_predicted_motion_;

        public:
            const static std::string DESIRED_DELTA_NS;
            const static std::string PREDICTED_DELTA_NS;
            const static std::string PROJECTED_GRIPPER_NS;
            const static std::string PROJECTED_BAND_NS;
            const static std::string PROJECTED_POINT_PATH_NS;
            const static std::string PROJECTED_POINT_PATH_LINES_NS;

            const static std::string CONSTRAINT_VIOLATION_VERSION1_NS;

            const static std::string CLUSTERING_TARGETS_NS;
            const static std::string CLUSTERING_RESULTS_PRE_PROJECT_NS;
            const static std::string CLUSTERING_RESULTS_POST_PROJECT_NS;
            const static std::string CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS;
    };
}

#endif // SMMAP_PLANNER_H
