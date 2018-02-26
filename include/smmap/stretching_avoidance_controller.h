#ifndef STRETCHINGAVOIDANCECONTROLLER_H
#define STRETCHINGAVOIDANCECONTROLLER_H

#include <sdf_tools/sdf.hpp>
#include <smmap_utilities/visualization_tools.h>

#include "smmap/deformable_controller.h"
#include "smmap/grippers.hpp"
#include "smmap/robot_interface.hpp"

namespace smmap
{
    class StretchingAvoidanceController : public DeformableController
    {
        public:
            StretchingAvoidanceController(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const DeformableModel::Ptr& deformable_model,
                    const RobotInterface::Ptr& robot,
                    const sdf_tools::SignedDistanceField& sdf,
                    std::mt19937_64& generator,
                    const smmap_utilities::Visualizer::Ptr& vis,
                    const StretchingAvoidanceControllerSolverType gripper_controller_type,
                    const int max_count);

        private:
            /////////////////////////////////////////////////////////////////////////////////////////
            // Optimization functions
            /////////////////////////////////////////////////////////////////////////////////////////

            OutputData getGripperMotion_impl(const InputData& input_data);

            OutputData solvedByRandomSampling(const InputData& input_data);

            OutputData solvedByNomad(const InputData& input_data);

            /////////////////////////////////////////////////////////////////////////////////////////
            // Helper functions
            /////////////////////////////////////////////////////////////////////////////////////////

            kinematics::Vector6d singleGripperPoseDeltaSampler(const double max_delta);

            AllGrippersSinglePoseDelta allGripperPoseDeltaSampler(const ssize_t num_grippers, const double max_delta);

            double errorOfControlByPrediction(const ObjectPointSet predicted_object_p_dot,
                                              const Eigen::VectorXd &desired_object_p_dot,
                                              const Eigen::VectorXd &desired_p_dot_weight) const;

            void visualize_stretching_vector(const ObjectPointSet& object_configuration);

            void visualize_rope_stretching_vector(const ObjectPointSet& object_configuration);

            void visualize_cloth_stretching_vector(const ObjectPointSet& object_configuration);

            void visualize_gripper_motion(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta& gripper_motion);

            /////////////////////////////////////////////////////////////////////////////////////////
            // Collision constraint related function
            /////////////////////////////////////////////////////////////////////////////////////////

            double gripperCollisionCheckHelper(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta& test_gripper_motion) const;

            bool gripperCollisionCheckResult(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta &test_gripper_motion) const;

            /////////////////////////////////////////////////////////////////////////////////////////
            // Stretching constraint related function
            /////////////////////////////////////////////////////////////////////////////////////////

            bool stretchingDetection(
                    const InputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            double ropeTwoGripperStretchingHelper(
                    const InputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            bool ropeTwoGrippersStretchingDetection(
                    const InputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            double clothTwoGripperStretchingHelper(
                    const InputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            bool clothTwoGrippersStretchingDetection(
                    const InputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

        private:	    
            GripperCollisionChecker gripper_collision_checker_;
            const double robot_min_distance_to_obstacles_;

            const std::vector<GripperData> grippers_data_;

            const sdf_tools::SignedDistanceField environment_sdf_;
            std::mt19937_64& generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;

            smmap_utilities::Visualizer::Ptr vis_;

            StretchingAvoidanceControllerSolverType gripper_controller_type_;
            const DeformableType deformable_type_;
            const TaskType task_type_;
            const DeformableModel::Ptr model_;

            const Eigen::MatrixXd max_node_distance_;
            const Eigen::MatrixXd max_node_squared_distance_;
            const double distance_to_obstacle_threshold_;
            double stretching_cosine_threshold_;

            const int max_count_;
            int sample_count_;

            bool fix_step_;
            bool over_stretch_;
            const std::string log_file_path_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_RANDOM_SAMPLING_H
