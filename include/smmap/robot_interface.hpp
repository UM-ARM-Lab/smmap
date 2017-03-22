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

            WorldState sendGrippersPoses(const AllGrippersSinglePose& grippers_pose);

            bool testGrippersPoses(const std::vector<AllGrippersSinglePose>& grippers_pose,
                                   const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback);

            std::vector<CollisionData> checkGripperCollision(const AllGrippersSinglePose& grippers_pose);

        private:
            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            std::vector<GripperData> grippers_data_;
            GripperCollisionChecker gripper_collision_checker_;
            ros::ServiceClient execute_gripper_movement_client_;
            actionlib::SimpleActionClient<deformable_manipulation_msgs::TestGrippersPosesAction> test_grippers_poses_client_;

        // TODO: comments, and placement, and stuff
        public:
            const double dt_;
            const double max_gripper_velocity_;

        private:
            std::thread spin_thread_;

            WorldState sendGrippersPoses_impl(const deformable_manipulation_msgs::ExecuteGripperMovementRequest& movement);

            deformable_manipulation_msgs::ExecuteGripperMovementRequest noOpGripperMovement();
            deformable_manipulation_msgs::ExecuteGripperMovementRequest toRosGrippersPoses(const AllGrippersSinglePose& grippers_pose) const;

            ////////////////////////////////////////////////////////////////////
            // Testing specific gripper movements
            ////////////////////////////////////////////////////////////////////

            size_t feedback_counter_;
            std::vector<bool> feedback_recieved_;
            void internalTestPoseFeedbackCallback(
                    const deformable_manipulation_msgs::TestGrippersPosesActionFeedbackConstPtr& feedback,
                    const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback);

            bool testGrippersPoses_impl(
                    const deformable_manipulation_msgs::TestGrippersPosesGoal& goal,
                    const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback);

            deformable_manipulation_msgs::TestGrippersPosesGoal toRosTestPosesGoal(
                    const std::vector<AllGrippersSinglePose>& grippers_poses) const;
    };
}

#endif // ROBOT_INTERFACE_HPP