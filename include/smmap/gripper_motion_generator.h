#ifndef GRIPPER_MOTION_GENERATOR_H
#define GRIPPER_MOTION_GENERATOR_H

#include <sdf_tools/sdf.hpp>

#include "smmap/deformable_model.h"
#include "smmap/grippers.hpp"
#include "smmap/robot_interface.hpp"

namespace smmap
{
    class GripperMotionGenerator
    {
        public:
            GripperMotionGenerator(
                    ros::NodeHandle& nh,
                    const sdf_tools::SignedDistanceField& environment_sdf,
                    RobotInterface& robot,
                    std::mt19937_64& generator,
                    Visualizer vis,
                    GripperControllerType gripper_controller_type,
                    const int64_t max_count,
                    const double distance_to_obstacle_threshold);

            //////////////////////////////////////////////////////////////////////////////////////
            // Called from outside to find the optimal gripper command
            //////////////////////////////////////////////////////////////////////////////////////

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> findOptimalGripperMotion(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const DeformableModel::Ptr deformable_model,
                    const double max_gripper_velocity);

            void SetGripperControllerType(GripperControllerType gripper_controller_type);


        private:



            /////////////////////////////////////////////////////////////////////////////////////////
            // Optimization function
            /////////////////////////////////////////////////////////////////////////////////////////

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByRandomSampling(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const DeformableModel::Ptr deformable_model,
                    const double max_gripper_velocity);

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByUniformSampling(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const DeformableModel::Ptr deformable_model,
                    const double max_gripper_velocity);



            //////////////////////////////////////////////////////////////////////////////////
            // Helper function
            //////////////////////////////////////////////////////////////////////////////////

            kinematics::Vector6d singleGripperPoseDeltaSampler(const double max_delta);

            AllGrippersSinglePoseDelta allGripperPoseDeltaSampler(const ssize_t num_grippers, const double max_delta);

            AllGrippersSinglePoseDelta setAllGripperPoseDeltaZero(const ssize_t num_grippers);


            double errorOfControlByPrediction(const ObjectPointSet predicted_object_p_dot,
                                              const Eigen::VectorXd &desired_object_p_dot);

            void visualize_stretching_vector(const ObjectPointSet& object_configuration);

            void visualize_gripper_motion(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta& gripper_motion);

            //////////////////////////////////////////////////////////////////////////////////
            // Collision constraint related function
            //////////////////////////////////////////////////////////////////////////////////

            bool gripperCollisionCheckResult(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta &test_gripper_motion);

            //////////////////////////////////////////////////////////////////////////////////
            // Stretching constraint related function
            //////////////////////////////////////////////////////////////////////////////////

            bool stretchingDetection(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            bool ropeTwoGrippersStretchingDetection(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);




        private:

//            ros::NodeHandle nh_;

//            DeformableModel::Ptr deformable_model_;
            const Eigen::MatrixXd object_initial_node_distance_;
            GripperCollisionChecker gripper_collision_checker_;

            const std::vector<GripperData> grippers_data_;

            const sdf_tools::SignedDistanceField enviroment_sdf_;
//            RobotInterface& robot_;
            std::mt19937_64& generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;

            Visualizer vis_;

//            AllGrippersSinglePoseDelta grippers_pose_delta_;
//            std::vector<GripperData> grippers_data_;

            GripperControllerType gripper_controller_type_;
            DeformableType deformable_type_;
            TaskType task_type_;

            const double distance_to_obstacle_threshold_;
            double stretching_factor_threshold_;
            double stretching_cosine_threshold_;

            const int64_t max_count_;
            int sample_count_;
            bool over_stretch_;


//            Visualizer& vis_;
//            const GripperMotionLoggingFunctionType logging_fn_;





    };


}

#endif // GRIPPER_MOTION_GENERATOR_H
