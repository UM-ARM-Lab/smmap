#ifndef ROBOT_INTERFACE_HPP
#define ROBOT_INTERFACE_HPP

#include <thread>
#include <functional>

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
            typedef std::shared_ptr<RobotInterface> Ptr;

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


            // This a Jacobian between the movement of the grippers (in the gripper body frame)
            // and the movement of the robot's DOF
            Eigen::MatrixXd getGrippersJacobian(const Eigen::VectorXd& robot_configuration);

            // This looks up the points of interest as reporeted by the external robot (i.e. OpenRAVE)
            // then querrys Bullet for the data needed to do collision avoidance, and querrys OpenRAVE for the Jacobian
            // of the movement of the point relative to the robot DOF movement.
            //
            // This includes the grippers.
            std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> getPointsOfInterestCollisionData(
                    const Eigen::VectorXd& configuration);

            void setCallbackFunctions(
                    std::function<Eigen::MatrixXd(const Eigen::VectorXd& configuration)> get_grippers_jacobian_fn,
                    std::function<std::vector<Eigen::Vector3d>(const Eigen::VectorXd& configuration)> get_collision_points_of_interest_fn,
                    std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd& configuration)> get_collision_points_of_interest_jacobians_fn);

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
            const double max_gripper_velocity_norm_;
            const double max_dof_velocity_norm_;
            const double min_controller_distance_to_obstacles_;
            const Eigen::VectorXd joint_lower_limits_;
            const Eigen::VectorXd joint_upper_limits_;

        private:
            std::thread spin_thread_;

            // Function pointers that allow for generic(ish) external robots, without explicit inheritance
            std::function<Eigen::MatrixXd(const Eigen::VectorXd& configuration)> get_grippers_jacobian_fn_;
            std::function<std::vector<Eigen::Vector3d>(const Eigen::VectorXd& configuration)> get_collision_points_of_interest_fn_;
            std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd& configuration)> get_collision_points_of_interest_jacobians_fn_;

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
