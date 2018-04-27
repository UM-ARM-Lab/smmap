#ifndef ROBOT_INTERFACE_HPP
#define ROBOT_INTERFACE_HPP

#include <thread>
#include <functional>

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <deformable_manipulation_msgs/messages.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include "smmap/grippers.hpp"
#include "smmap/task_function_pointer_types.h"


namespace smmap
{
    class RobotInterface
    {
        public:
            typedef std::shared_ptr<RobotInterface> Ptr;

            RobotInterface(ros::NodeHandle& nh, ros::NodeHandle& ph);
            ~RobotInterface();

            WorldState start();

            bool ok() const;
            void shutdown();

            const std::vector<GripperData>& getGrippersData() const;

            AllGrippersSinglePose getGrippersPoses();

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



            AllGrippersSinglePose getGrippersPoses(const Eigen::VectorXd& robot_configuration) const;

            AllGrippersSinglePose getGrippersPoses(const std::pair<Eigen::VectorXd, Eigen::VectorXd>& robot_configuration) const;

            // This a Jacobian between the movement of the grippers (in the gripper body frame)
            // and the movement of the robot's DOF
            Eigen::MatrixXd getGrippersJacobian(const Eigen::VectorXd& robot_configuration) const;

            // This looks up the points of interest as reporeted by the external robot (i.e. OpenRAVE)
            // then querrys Bullet for the data needed to do collision avoidance, and querrys OpenRAVE for the Jacobian
            // of the movement of the point relative to the robot DOF movement.
            //
            // This includes the grippers.
            std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> getPointsOfInterestCollisionData(
                    const Eigen::VectorXd& configuration);


            Eigen::VectorXd mapGripperMotionToRobotMotion(
                    const Eigen::VectorXd& robot_configuration,
                    const AllGrippersSinglePoseDelta& grippers_delta) const;

            // Only intended for use by 2 manipulators
            std::pair<Eigen::VectorXd, Eigen::VectorXd> mapGripperMotionToRobotMotion(
                    const std::pair<Eigen::VectorXd, Eigen::VectorXd>& robot_configuration,
                    const AllGrippersSinglePoseDelta& grippers_delta) const;


            bool checkRobotCollision(const Eigen::VectorXd& robot_configuration) const;

            // Only intended for use by 2 manipulators
            bool checkRobotCollision(const std::pair<Eigen::VectorXd, Eigen::VectorXd>& robot_configuration) const;

            std::vector<Eigen::VectorXd> getCloseIkSolutions(const std::string& gripper, const Eigen::Isometry3d& target_pose) const;

            std::vector<std::vector<Eigen::VectorXd>> getCloseIkSolutions(const AllGrippersSinglePose& target_poses) const;

            std::pair<bool, Eigen::VectorXd> getGeneralIkSolution(
                    const Eigen::VectorXd& starting_config,
                    const std::vector<std::string>& gripper_names,
                    const AllGrippersSinglePose& target_poses) const;

            std::pair<bool, std::pair<Eigen::VectorXd, Eigen::VectorXd>> getGeneralIkSolution(
                    const std::pair<Eigen::VectorXd, Eigen::VectorXd>& robot_configuration,
                    const AllGrippersSinglePose& target_poses) const;

            bool testPathForCollision(const std::vector<Eigen::VectorXd>& path) const;

            void setCallbackFunctions(
                    const std::function<AllGrippersSinglePose(const Eigen::VectorXd& configuration)>& get_ee_poses_fn,
                    const std::function<Eigen::MatrixXd(const Eigen::VectorXd& configuration)>& get_grippers_jacobian_fn,
                    const std::function<std::vector<Eigen::Vector3d>(const Eigen::VectorXd& configuration)>& get_collision_points_of_interest_fn,
                    const std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd& configuration)>& get_collision_points_of_interest_jacobians_fn,
                    const std::function<bool(const Eigen::VectorXd& configuration)>& full_robot_collision_check_fn,
                    const std::function<std::vector<Eigen::VectorXd>(const std::string& gripper, const Eigen::Isometry3d& target_pose)>& close_ik_solutions_fn,
                    const std::function<std::pair<bool, Eigen::VectorXd>(const Eigen::VectorXd& starting_config, const std::vector<std::string>& gripper_names, const AllGrippersSinglePose& target_poses)> general_ik_solution_fn,
                    const std::function<bool(const std::vector<Eigen::VectorXd>& path)> test_path_for_collision_fn);

            // Defaults the timespace to "latest available", indicted by ros::Time(0)
            Eigen::Vector3d transformToFrame(
                    const Eigen::Vector3d& point,
                    const std::string& source_frame,
                    const std::string& target_frame,
                    const ros::Time& time = ros::Time(0)) const;

            const Eigen::Isometry3d& getWorldToTaskFrameTf() const;

        private:
            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;

        public:
            const std::string bullet_frame_name_;
            const std::string world_frame_name_; // Frame that all incomming data must be in
        private:
            tf2_ros::Buffer tf_buffer_;
            const tf2_ros::TransformListener tf_listener_;
            Eigen::Isometry3d world_to_bullet_tf_;

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
            const Eigen::VectorXd dof_weights_;

        private:
            std::thread spin_thread_;

            // Function pointers that allow for generic(ish) external robots, without explicit inheritance
            std::function<AllGrippersSinglePose(const Eigen::VectorXd& configuration)> get_ee_poses_fn_;
            std::function<Eigen::MatrixXd(const Eigen::VectorXd& configuration)> get_grippers_jacobian_fn_;
            std::function<std::vector<Eigen::Vector3d>(const Eigen::VectorXd& configuration)> get_collision_points_of_interest_fn_;
            std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd& configuration)> get_collision_points_of_interest_jacobians_fn_;
            std::function<bool(const Eigen::VectorXd& configuration)> full_robot_collision_check_fn_;
            std::function<std::vector<Eigen::VectorXd>(const std::string& gripper, const Eigen::Isometry3d& target_pose)> close_ik_solutions_fn_;
            std::function<std::pair<bool, Eigen::VectorXd>(const Eigen::VectorXd& starting_config, const std::vector<std::string>& gripper_names, const AllGrippersSinglePose& target_poses)> general_ik_solution_fn_;
            std::function<bool(const std::vector<Eigen::VectorXd>& path)> test_path_for_collision_fn_;

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
