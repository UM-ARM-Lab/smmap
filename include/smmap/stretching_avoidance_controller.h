#ifndef STRETCHINGAVOIDANCECONTROLLER_H
#define STRETCHINGAVOIDANCECONTROLLER_H

#include <sdf_tools/sdf.hpp>

#include "smmap/deformable_controller.hpp"
#include "smmap/grippers.hpp"
#include "smmap/robot_interface.hpp"
#include "smmap/nomad_solvers.h"

namespace smmap
{
    class StretchingAvoidanceController : public DeformableController
    {
        public:
            StretchingAvoidanceController(ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    RobotInterface& robot,
                    const sdf_tools::SignedDistanceField& sdf,
                    std::mt19937_64& generator,
                    Visualizer& vis,
                    GripperControllerType gripper_controller_type,
                    const DeformableModel::Ptr& deformable_model,
                    const int max_count);

            //////////////////////////////////////////////////////////////////////////////////////
            // Called from outside to find the optimal gripper command
            //////////////////////////////////////////////////////////////////////////////////////

        private:
            /////////////////////////////////////////////////////////////////////////////////////////
            // Optimization function
            /////////////////////////////////////////////////////////////////////////////////////////

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion_impl(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity);

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByRandomSampling(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity);

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByNomad(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity);



            /////////////////////////////////////////////////////////////////////////////////////////
            // Helper function
            /////////////////////////////////////////////////////////////////////////////////////////

            kinematics::Vector6d singleGripperPoseDeltaSampler(const double max_delta);

            AllGrippersSinglePoseDelta allGripperPoseDeltaSampler(const ssize_t num_grippers, const double max_delta);

            AllGrippersSinglePoseDelta setAllGripperPoseDeltaZero(const ssize_t num_grippers);

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
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            double ropeTwoGripperStretchingHelper(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            bool ropeTwoGrippersStretchingDetection(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            double clothTwoGripperStretchingHelper(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            bool clothTwoGrippersStretchingDetection(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

        private:	    
            const Eigen::MatrixXd object_initial_node_distance_;
            GripperCollisionChecker gripper_collision_checker_;

            const std::vector<GripperData> grippers_data_;

            const sdf_tools::SignedDistanceField enviroment_sdf_;
            std::mt19937_64& generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;

            Visualizer& vis_;

            GripperControllerType gripper_controller_type_;
            const DeformableType deformable_type_;
            const TaskType task_type_;
            const DeformableModel::Ptr model_;

            const double distance_to_obstacle_threshold_;
            double max_stretch_factor_;
            double stretching_cosine_threshold_;

            const int max_count_;
            int sample_count_;

            bool fix_step_;
            bool over_stretch_;
            const std::string log_file_path_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_RANDOM_SAMPLING_H
