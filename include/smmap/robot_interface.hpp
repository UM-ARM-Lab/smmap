#ifndef ROBOT_INTERFACE_HPP
#define ROBOT_INTERFACE_HPP

#include <thread>

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <deformable_manipulation_msgs/messages.h>

#include "smmap/grippers.hpp"
#include "smmap/task_function_pointer_types.h"


namespace smmap
{
    class RobotInterface
    {
        public:
            RobotInterface(ros::NodeHandle& nh);
            ~RobotInterface();

            WorldState start();

            bool ok() const;
            void shutdown();

            const std::vector<GripperData>& getGrippersData() const;

            const AllGrippersSinglePose getGrippersPose();

            // This function assumes only 2 grippers, and it is called before the grippers are moved by sendGrippersPoses
            double getGrippersInitialDistance();

            WorldState commandRobotMotion(
                    const AllGrippersSinglePose& target_grippers_poses,
                    const Eigen::VectorXd& target_robot_configuration,
                    const bool robot_configuration_valid);

            bool testRobotMotion(
                    const std::vector<AllGrippersSinglePose>& test_grippers_poses,
                    const std::vector<Eigen::VectorXd>& test_robot_configurations,
                    const bool robot_configuration_valid,
                    const TestRobotMotionFeedbackCallbackFunctionType& feedback_callback);

            std::vector<CollisionData> checkGripperCollision(const AllGrippersSinglePose& grippers_pose);

            Eigen::MatrixXd getGrippersJacobian(const Eigen::VectorXd& robot_configuration);

        private:
            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            const std::string world_frame_name_; // Frame that all incomming data must be in
            const std::vector<GripperData> grippers_data_;
            GripperCollisionChecker gripper_collision_checker_;
            ros::ServiceClient execute_gripper_movement_client_;
            actionlib::SimpleActionClient<deformable_manipulation_msgs::TestRobotMotionAction> test_grippers_poses_client_;

        // TODO: comments, and placement, and stuff
        public:
            const double dt_;
            const double max_gripper_velocity_;

        private:
            std::thread spin_thread_;

            WorldState commandRobotMotion_impl(
                    const deformable_manipulation_msgs::ExecuteRobotMotionRequest& movement);

            deformable_manipulation_msgs::ExecuteRobotMotionRequest noOpGripperMovement();
            deformable_manipulation_msgs::ExecuteRobotMotionRequest toRosMovementRequest(
                    const AllGrippersSinglePose& grippers_poses,
                    const Eigen::VectorXd& robot_configuration,
                    const bool robot_configuration_valid) const;

            ////////////////////////////////////////////////////////////////////
            // Testing specific gripper movements
            ////////////////////////////////////////////////////////////////////

            volatile size_t feedback_counter_;
            std::vector<bool> feedback_recieved_;
            void internalTestPoseFeedbackCallback(
                    const deformable_manipulation_msgs::TestRobotMotionActionFeedbackConstPtr& feedback,
                    const TestRobotMotionFeedbackCallbackFunctionType& feedback_callback);

            bool testRobotMotion_impl(
                    const deformable_manipulation_msgs::TestRobotMotionGoal& goal,
                    const TestRobotMotionFeedbackCallbackFunctionType& feedback_callback);

            deformable_manipulation_msgs::TestRobotMotionGoal toRosTestPosesGoal(
                    const std::vector<AllGrippersSinglePose>& grippers_poses,
                    const std::vector<Eigen::VectorXd>& robot_configurations,
                    const bool robot_configurations_valid) const;
    };
}

#endif // ROBOT_INTERFACE_HPP
