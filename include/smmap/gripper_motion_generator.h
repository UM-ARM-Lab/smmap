#ifndef GRIPPER_MOTION_GENERATOR_H
#define GRIPPER_MOTION_GENERATOR_H

#include "smmap/deformable_model.h"
#include "smmap/grippers.hpp"
#include "smmap/robot_interface.hpp"

#include "sdf_tools/sdf.hpp"

namespace smmap {

    class GripperMotionGenerator
    {
        public:
            GripperMotionGenerator(ros::NodeHandle& nh,
                                   const sdf_tools::SignedDistanceField& environment_sdf,
                           //        RobotInterface& robot,
                                   std::mt19937_64& generator,
                                   GripperControllerType gripper_controller_type,
                                   const double max_gripper_translation_step,
                                   const double max_gripper_rotation_step,
                                   const int64_t max_count,
                                   const double distance_to_obstacle_threshold);

            //////////////////////////////////////////////////////////////////////////////////////
            // Called from outside to find the optimal gripper command
            //////////////////////////////////////////////////////////////////////////////////////

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> findOptimalGripperMotion(
                    const DeformableModel::Ptr deformable_model,
                    const DeformableModel::DeformableModelInputData &input_data,
                    const ObjectPointSet& object_configuration,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale);

            void SetGripperControllerType(GripperControllerType gripper_controller_type);


        private:



            /////////////////////////////////////////////////////////////////////////////////////////
            // Optimization function
            /////////////////////////////////////////////////////////////////////////////////////////

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByRandomSampling(
                    const DeformableModel::Ptr deformable_model,
                    const DeformableModel::DeformableModelInputData &input_data,
                    const ObjectPointSet& object_configuration,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale);

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByUniformSampling(
                    const DeformableModel::Ptr deformable_model,
                    const DeformableModel::DeformableModelInputData &input_data,
                    const ObjectPointSet& object_configuration,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale);



            //////////////////////////////////////////////////////////////////////////////////
            // Helper function
            //////////////////////////////////////////////////////////////////////////////////

            kinematics::Vector6d singleGripperPoseDeltaSampler();

            double errorOfControlByPrediction(
                    const ObjectPointSet& predicted_object_p_dot,
                    const Eigen::VectorXd &desired_object_p_dot) const;

            AllGrippersSinglePoseDelta allGripperPoseDeltaSampler(const ssize_t num_grippers);

            std::pair<bool, std::vector<CollisionData>> gripperCollisionCheckResult(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta &test_gripper_motion);

            bool stretchingDetection(const DeformableModel::DeformableModelInputData &input_data,
                                     const AllGrippersSinglePose &current_gripper_pose,
                                     const AllGrippersSinglePoseDelta &test_gripper_motion,
                                     const ObjectPointSet& object_configuration);


        private:

//            ros::NodeHandle nh_;

//            DeformableModel::Ptr deformable_model_;
            GripperCollisionChecker gripper_collision_checker_;
            const sdf_tools::SignedDistanceField enviroment_sdf_;
//            RobotInterface& robot_;
            std::mt19937_64& generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;
//            AllGrippersSinglePoseDelta grippers_pose_delta_;
//            std::vector<GripperData> grippers_data_;

            GripperControllerType gripper_controller_type_;

            const double translation_lower_bound_;
            const double translation_upper_bound_;
            const double rotation_lower_bound_;
            const double rotation_upper_bound_;

            const double distance_to_obstacle_threshold_;

            const int64_t max_count_;


//            Visualizer& vis_;
//            const GripperMotionLoggingFunctionType logging_fn_;





    };


}

#endif // GRIPPER_MOTION_GENERATOR_H
