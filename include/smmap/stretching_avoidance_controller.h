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

            OutputData solvedByGradientDescentProjectionViaGurobiAndManualMethod(const InputData& input_data);
            OutputData solvedByGradientDescentProjectionViaGurobiMethod(const InputData& input_data);
            OutputData solvedByGradientDescentBarrierMethod(const InputData& input_data);
            OutputData solvedByGradientDescentOld(const InputData& input_data);

            /////////////////////////////////////////////////////////////////////////////////////////
            // Helper functions
            /////////////////////////////////////////////////////////////////////////////////////////

            kinematics::Vector6d singleGripperPoseDeltaSampler(const double max_delta);

            AllGrippersSinglePoseDelta allGripperPoseDeltaSampler(const ssize_t num_grippers, const double max_delta);

            kinematics::Vector6d getConstraintAwareGripperDeltaSample(
                    const Eigen::Isometry3d& gripper_pose,
                    const CollisionData& collision_data,
                    const double max_delta,
                    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& stretching_correction_data);

            kinematics::Vector6d getFeasibleGripperDeltaGurobi(
                    const Eigen::Isometry3d& gripper_pose,
                    const CollisionData& collision_data,
                    const double max_delta,
                    const std::pair<Eigen::Vector3d, Eigen::Vector3d>& stretching_correction_data) const;


            /*
            template <typename T1, typename T2>
            inline kinematics::Vector6d getConstraintAwareGripperDeltaSample(
                    const T1& collision_constraint_fn,
                    const T2& stretching_constraint_fn,
                    const double max_step_size)
            {
                kinematics::Vector6d sample = kinematics::Vector6d::Zero();
                bool collision_satisfied = (collision_constraint_fn(sample) < 0.0);
                bool stretching_satisified = (stretching_constraint_fn(sample) < 0.0);
                bool valid_sample =  collision_satisfied && stretching_satisified;
//                bool valid_sample = (collision_constraint_fn(sample) < 0.0) && (stretching_constraint_fn(sample) < 0.0);

//                std::cout << "collision_satisfied:  " << collision_constraint_fn(sample) << std::endl;
//                std::cout << "stretching_satisfied: " << stretching_constraint_fn(sample) << std::endl;

                while (!valid_sample)
                {
                    sample = singleGripperPoseDeltaSampler(max_step_size);
                    collision_satisfied = (collision_constraint_fn(sample) < 0.0);
                    stretching_satisified = (stretching_constraint_fn(sample) < 0.0);
                    valid_sample = collision_satisfied && stretching_satisified;
//                    valid_sample = (collision_constraint_fn(sample) < 0.0) && (stretching_constraint_fn(sample) < 0.0);

//                    std::cout << "collision_satisfied:  " << collision_constraint_fn(sample) << std::endl;
//                    std::cout << "stretching_satisfied: " << stretching_constraint_fn(sample) << std::endl;
                }

//                if (collision_constraint_fn(sample) >= 0.0)
//                {
//                    std::cout << "Collision constraint violated on sampling: " << collision_constraint_fn(sample) << std::endl;
//                }

//                if (stretching_constraint_fn(sample) >= 0.0)
//                {
//                    std::cout << "Stretching constraint violated on sampling: " << stretching_constraint_fn(sample) << std::endl;
//                }

//                assert(collision_constraint_fn(sample) < 0.0);
//                assert(stretching_constraint_fn(sample) < 0.0);

                return sample;
            }
            */

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

            double stretchingFunctionEvaluation(
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

            double evaluateStretchingConstraint(const std::pair<Eigen::Vector3d, Eigen::Vector3d>& stretching_constraint_data, const kinematics::Vector6d& gripper_delta) const;
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> stretchingCorrectionVectorsAndPoints(const InputData& input_data) const;
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> ropeTwoGrippersStretchingCorrectionVectorsAndPoints(const InputData& input_data) const;
            std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> clothTwoGrippersStretchingCorrectionVectorsAndPoints(const InputData& input_data) const;







            void visualizeCone(const Eigen::Vector3d& cone_direction, const double min_normalized_dot_product, const Eigen::Isometry3d& pose, const int marker_id);





        private:	    
            GripperCollisionChecker gripper_collision_checker_;

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
