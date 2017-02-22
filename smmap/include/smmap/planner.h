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

            std::pair<EigenHelpers::VectorVector3d, std::vector<double>> createVirtualRubberBand(
                    const WorldState &current_world_state,
                    std::shared_ptr<DijkstrasCoverageTask> dijkstras_task);

            bool checkForClothStretchingViolations(const std::vector<EigenHelpers::VectorVector3d>& projected_paths);
            void detectFutureConstraintViolations(const WorldState& current_world_state);

//            std::pair<EigenHelpers::VectorVector3d, std::vector<double>> forwardSimulateVirtualRubberBand(
//                    std::shared_ptr<DijkstrasCoverageTask> task,
//                    const EigenHelpers::VectorVector3d& starting_band,
//                    const std::vector<double>& starting_dists,
//                    const AllGrippersSinglePose& starting_grippers_single_pose,
//                    const AllGrippersSinglePose& ending_grippers_single_pose) const;



            WorldState sendNextCommand(const WorldState& current_world_state);
            void visualizeDesiredMotion(const WorldState& current_world_state,
                                        const ObjectDeltaAndWeight& desired_motion);

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

            void updateModels(const WorldState& starting_world_state,
                    const ObjectDeltaAndWeight& task_desired_motion,
                    const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands,
                    const ssize_t model_used,
                    const WorldState& world_feedback);

            Eigen::MatrixXd calculateProcessNoise(const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands);

//            Eigen::VectorXd calculateObservedReward(
//                    const WorldState& starting_world_state,
//                    const ObjectDeltaAndWeight& task_desired_motion,
//                    const ssize_t model_used,
//                    const WorldState& world_feedback);

//            Eigen::MatrixXd calculateObservationNoise(
//                    const Eigen::MatrixXd& process_noise,
//                    const ssize_t model_used);


        ////////////////////////////////////
        // Random stuff to be properly managed later
        ////////////////////////////////////

        private:
            std::pair<EigenHelpers::VectorVector3d, std::vector<double>> virtual_rubber_band_between_grippers_version2a_;
            std::pair<EigenHelpers::VectorVector3d, std::vector<double>> virtual_rubber_band_between_grippers_version2b_;
            double max_gripper_distance_;
    };
}

#endif // SMMAP_PLANNER_H
