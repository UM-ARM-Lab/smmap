#ifndef ROBOT_INTERFACE_HPP
#define ROBOT_INTERFACE_HPP

#include <thread>

#include <ros/ros.h>
//#include <ros/callback_queue.h>
#include <actionlib/client/simple_action_client.h>
//#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <smmap_msgs/messages.h>

//#include "smmap/ros_communication_helpers.hpp"
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

            WorldState sendGripperCommand(const AllGrippersSinglePose& grippers_poses);

            bool testGrippersPoses(const std::vector<AllGrippersSinglePose>& grippers_poses,
                                   const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback);

            std::vector<CollisionData> checkGripperCollision(const AllGrippersSinglePose& grippers_poses);

        private:
            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            std::vector<GripperData> grippers_data_;
            GripperCollisionChecker gripper_collision_checker_;
            actionlib::SimpleActionClient<smmap_msgs::CmdGrippersTrajectoryAction> cmd_grippers_traj_client_;
            actionlib::SimpleActionClient<smmap_msgs::TestGrippersPosesAction> test_grippers_poses_client_;

        // TODO: comments, and placement, and stuff
        public:
            const double dt_;
            const double max_gripper_velocity_;

        private:
            // Our internal version of ros::spin()
            std::thread spin_thread_;
            static void spin(double loop_rate);




            std::vector<WorldState> sendGripperTrajectory_impl(
                    const smmap_msgs::CmdGrippersTrajectoryGoal& goal);

            smmap_msgs::CmdGrippersTrajectoryGoal noOpTrajectoryGoal(size_t num_no_op);

            smmap_msgs::CmdGrippersTrajectoryGoal toRosTrajectoryGoal(
                    const AllGrippersPoseTrajectory& trajectory) const;





            size_t feedback_counter_;
            std::vector<bool> feedback_recieved_;
            void internalTestPoseFeedbackCallback(
                    const smmap_msgs::TestGrippersPosesActionFeedbackConstPtr& feedback,
                    const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback);

            bool testGrippersPoses_impl(
                    const smmap_msgs::TestGrippersPosesGoal& goal,
                    const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback);

            smmap_msgs::TestGrippersPosesGoal toRosTestPosesGoal(
                    const std::vector<AllGrippersSinglePose>& grippers_poses) const;
    };
}

#endif // ROBOT_INTERFACE_HPP
