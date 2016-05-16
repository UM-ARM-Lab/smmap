#ifndef ROBOT_INTERFACE_HPP
#define ROBOT_INTERFACE_HPP

#include <thread>
#include <arc_utilities/eigen_helpers_conversions.hpp>

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <actionlib/client/simple_action_client.h>
#include <smmap_msgs/messages.h>

#include "smmap/gripper_collision_checker.hpp"
#include "smmap/gripper_helpers.hpp"
#include "smmap/ros_communication_helpers.hpp"


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
                // TODO: remove this hardcoded spin rate
                , spin_thread_(spin, 1000)
            {}

            ~RobotInterface()
            {
                ROS_INFO_NAMED("task", "Terminating");
                spin_thread_.join();
            }

            std::vector< WorldState > start()
            {
                ROS_INFO_NAMED("robot_bridge", "Waiting for the robot gripper action server to be available");
                cmd_grippers_traj_client_.waitForServer();

                ROS_INFO_NAMED("robot_bridge", "Kickstarting the planner with a no-op");
                return sendGripperTrajectory_impl(noOpTrajectoryGoal(1));
            }

            bool ok() const
            {
                return ros::ok();
            }

            // TODO: rename this to "burn in a fire and die"
            void shutdown()
            {
                ros::shutdown();
            }

            const std::vector< GripperData >& getGrippersData() const
            {
                return grippers_data_;
            }

            const AllGrippersSinglePose getGrippersPose()
            {
                AllGrippersSinglePose grippers_pose(grippers_data_.size());

                for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
                {
                    ros::ServiceClient gripper_pose_client =
                        nh_.serviceClient< smmap_msgs::GetGripperPose >(GetGripperPoseTopic(nh_));
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

            std::vector< WorldState > sendGripperTrajectory(
                    const AllGrippersPoseTrajectory& trajectory)
            {
                return sendGripperTrajectory_impl(toRosGoal(trajectory));
            }

            std::vector< CollisionData > checkGripperCollision(
                    const AllGrippersSinglePose& gripper_poses)
            {
                return gripper_collision_checker_.gripperCollisionCheck(gripper_poses);
            }

            static constexpr double MAX_GRIPPER_VELOCITY = 0.05/20.0/0.01;

        private:
            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            std::vector< GripperData > grippers_data_;
            GripperCollisionChecker gripper_collision_checker_;
            actionlib::SimpleActionClient< smmap_msgs::CmdGrippersTrajectoryAction > cmd_grippers_traj_client_;

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
                        nh_.serviceClient< smmap_msgs::GetGripperPose >(GetGripperPoseTopic(nh_));
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

            smmap_msgs::CmdGrippersTrajectoryGoal toRosGoal(
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

            std::vector< WorldState > sendGripperTrajectory_impl(
                    const smmap_msgs::CmdGrippersTrajectoryGoal& goal)
            {
                std::vector< WorldState > feedback;

                cmd_grippers_traj_client_.sendGoalAndWait(goal);
                if (cmd_grippers_traj_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
                {
                    feedback = ParseGripperActionResult(cmd_grippers_traj_client_.getResult());
                }
                else
                {
                    ROS_FATAL_NAMED("planner", "Sending a goal to the robot failed");
                }

                return feedback;
            }

    };
}

#endif // ROBOT_INTERFACE_HPP
