#ifndef ROBOT_INTERFACE_HPP
#define ROBOT_INTERFACE_HPP

#include <chrono>
#include <thread>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <actionlib/client/simple_action_client.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <smmap_msgs/messages.h>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/grippers.hpp"
#include "smmap/task_function_pointer_types.h"


namespace smmap
{
    class RobotInterface
    {
        public:
            RobotInterface(ros::NodeHandle& nh)
                : nh_(nh)
                , grippers_data_(GetGrippersData(nh_))
                , gripper_collision_checker_(nh_)
                , cmd_grippers_traj_client_(nh_, GetCommandGripperTrajTopic(nh_), false)
                , test_grippers_poses_client_(nh_, GetTestGrippersPosesTopic(nh_), false)
                // TODO: remove this hardcoded spin rate
                , spin_thread_(spin, 1000)
            {
                execute_gripper_movement_and_update_sim_client_ =
                        nh.serviceClient<smmap_msgs::ExecuteGripperMovement>(GetExecuteGrippersMovementAndUpdateSimTopic(nh_));
            }

            ~RobotInterface()
            {
                ROS_INFO_NAMED("task", "Terminating");
                spin_thread_.join();
            }

            std::vector<WorldState> start()
            {
                ROS_INFO_NAMED("robot_bridge", "Waiting for the robot gripper action server to be available");
//                cmd_grippers_traj_client_.waitForServer();
                test_grippers_poses_client_.waitForServer();

                ROS_INFO_NAMED("robot_bridge", "Kickstarting the planner with a no-op");
                return sendGripperTrajectory_impl(noOpTrajectoryGoal(1));
            }

            bool ok() const
            {
                return ros::ok();
            }

            void shutdown()
            {
                ros::shutdown();
            }

            const std::vector<GripperData>& getGrippersData() const
            {
                return grippers_data_;
            }

            const AllGrippersSinglePose getGrippersPose()
            {
                AllGrippersSinglePose grippers_pose(grippers_data_.size());

                for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
                {
                    ros::ServiceClient gripper_pose_client =
                        nh_.serviceClient<smmap_msgs::GetGripperPose>(GetGripperPoseTopic(nh_));
                    gripper_pose_client.waitForExistence();

                    smmap_msgs::GetGripperPose pose_srv_data;
                    pose_srv_data.request.name = grippers_data_[gripper_ind].name;
                    if (!gripper_pose_client.call(pose_srv_data))
                    {
                        ROS_FATAL_STREAM_NAMED("task", "Unabled to retrieve gripper pose: "
                                                << grippers_data_[gripper_ind].name);
                    }

                    grippers_pose[gripper_ind] =
                            EigenHelpersConversions::GeometryPoseToEigenAffine3d(pose_srv_data.response.pose);
                }

                return grippers_pose;
            }

            std::vector<WorldState> sendGripperTrajectory(
                    const AllGrippersPoseTrajectory& trajectory)
            {
                return sendGripperTrajectory_impl(toRosTrajectoryGoal(trajectory));
            }

            bool testGrippersPoses(const std::vector<AllGrippersSinglePose>& grippers_poses,
                                   const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback)
            {
                return testGrippersPoses_impl(toRosTestPosesGoal(grippers_poses), feedback_callback);
            }

            std::vector<CollisionData> checkGripperCollision(
                    const AllGrippersSinglePose& gripper_poses)
            {
                return gripper_collision_checker_.gripperCollisionCheck(gripper_poses);
            }

            static constexpr double MAX_GRIPPER_VELOCITY = 0.2;

        private:
            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            std::vector<GripperData> grippers_data_;
            GripperCollisionChecker gripper_collision_checker_;
            actionlib::SimpleActionClient<smmap_msgs::CmdGrippersTrajectoryAction> cmd_grippers_traj_client_;
            actionlib::SimpleActionClient<smmap_msgs::TestGrippersPosesAction> test_grippers_poses_client_;
            ros::ServiceClient execute_gripper_movement_and_update_sim_client_;

            // Our internal version of ros::spin()
            std::thread spin_thread_;
            static void spin(double loop_rate)
            {
                ROS_INFO_NAMED("task" , "Starting feedback spinner");
                while (ros::ok())
                {
                    ros::getGlobalCallbackQueue()->callAvailable(ros::WallDuration(loop_rate));
                }
            }

            smmap_msgs::CmdGrippersTrajectoryGoal noOpTrajectoryGoal(size_t num_no_op)
            {
                smmap_msgs::CmdGrippersTrajectoryGoal goal;
                goal.gripper_names = GetGripperNames(grippers_data_);

                smmap_msgs::VectorPose grippers_pose;
                grippers_pose.pose.resize(grippers_data_.size());
                for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
                {
                    ros::ServiceClient gripper_pose_client =
                        nh_.serviceClient<smmap_msgs::GetGripperPose>(GetGripperPoseTopic(nh_));
                    gripper_pose_client.waitForExistence();

                    smmap_msgs::GetGripperPose pose_srv_data;
                    pose_srv_data.request.name = grippers_data_[gripper_ind].name;
                    if (!gripper_pose_client.call(pose_srv_data))
                    {
                        ROS_FATAL_STREAM_NAMED("task", "Unabled to retrieve gripper pose: "
                                                << grippers_data_[gripper_ind].name);
                    }

                    grippers_pose.pose[gripper_ind] = pose_srv_data.response.pose;
                }

                goal.trajectory.resize(num_no_op, grippers_pose);

                return goal;
            }

            smmap_msgs::CmdGrippersTrajectoryGoal toRosTrajectoryGoal(
                    const AllGrippersPoseTrajectory& trajectory) const
            {
                smmap_msgs::CmdGrippersTrajectoryGoal goal;
                goal.gripper_names = GetGripperNames(grippers_data_);

                goal.trajectory.resize(trajectory.size());
                for (size_t time_ind = 0; time_ind < trajectory.size(); time_ind++)
                {
                    goal.trajectory[time_ind].pose =
                            EigenHelpersConversions::VectorAffine3dToVectorGeometryPose(
                                trajectory[time_ind]);
                }

                return goal;
            }

            smmap_msgs::TestGrippersPosesGoal toRosTestPosesGoal(
                    const std::vector<AllGrippersSinglePose>& grippers_poses) const
            {
                smmap_msgs::TestGrippersPosesGoal goal;
                goal.gripper_names = GetGripperNames(grippers_data_);

                goal.poses_to_test.resize(grippers_poses.size());
                for (size_t pose_ind = 0; pose_ind < grippers_poses.size(); pose_ind++)
                {
                    goal.poses_to_test[pose_ind].pose =
                            EigenHelpersConversions::VectorAffine3dToVectorGeometryPose(
                                grippers_poses[pose_ind]);
                }

                return goal;
            }

            std::vector<WorldState> sendGripperTrajectory_impl(
                    const smmap_msgs::CmdGrippersTrajectoryGoal& goal)
            {
                std::vector<WorldState> feedback(1);

                smmap_msgs::ExecuteGripperMovement gripper_execution;
                gripper_execution.request.grippers_names = goal.gripper_names;
                gripper_execution.request.grippers_poses = goal.trajectory.back();

                if (!execute_gripper_movement_and_update_sim_client_.call(gripper_execution))
                {
                    ROS_FATAL("VERY BAD STUFF");
                    assert(false);
                }

                feedback[0] = ConvertToEigenFeedback(gripper_execution.response.sim_state);

//                cmd_grippers_traj_client_.sendGoalAndWait(goal);
//                if (cmd_grippers_traj_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
//                {
//                    feedback = ParseGripperActionResult(cmd_grippers_traj_client_.getResult());
//                }
//                else
//                {
//                    ROS_FATAL_NAMED("planner", "Sending a goal to the robot failed");
//                }

                return feedback;
            }

            bool testGrippersPoses_impl(
                    const smmap_msgs::TestGrippersPosesGoal& goal,
                    const TestGrippersPosesFeedbackCallbackFunctionType& feedback_callback)
            {
                size_t feedback_counter = goal.poses_to_test.size();
                const auto internal_feedback_fn = [&feedback_callback, &feedback_counter] (const smmap_msgs::TestGrippersPosesFeedbackConstPtr& feedback)
                {
                    ROS_INFO_STREAM_NAMED("robot_interface", "Got feedback for test number " << feedback->test_id);
                    feedback_callback(feedback->test_id, ConvertToEigenFeedback(feedback->sim_state));
                    feedback_counter--;
                };

                test_grippers_poses_client_.sendGoal(
                            goal,
                            actionlib::SimpleActionClient<smmap_msgs::TestGrippersPosesAction>::SimpleDoneCallback(),
                            actionlib::SimpleActionClient<smmap_msgs::TestGrippersPosesAction>::SimpleActiveCallback(),
                            internal_feedback_fn);

                const bool result = test_grippers_poses_client_.waitForResult();

                while(feedback_counter > 0)
                {
                    std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
                }

                return result;
            }
    };
}

#endif // ROBOT_INTERFACE_HPP
