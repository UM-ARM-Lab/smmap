#include "smmap/robot_interface.hpp"

#include <std_srvs/Empty.h>
#include <ros/callback_queue.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <tf2_eigen/tf2_eigen.h>
#include "smmap/ros_communication_helpers.hpp"


namespace smmap
{
    RobotInterface::RobotInterface(ros::NodeHandle& nh, ros::NodeHandle& ph)
        : nh_(nh)
        , ph_(ph)

        , bullet_frame_name_(smmap::GetBulletFrameName())
        , world_frame_name_(smmap::GetWorldFrameName())
        , tf_buffer_()
        , tf_listener_(tf_buffer_)

        , grippers_data_(GetGrippersData(nh_))
        , gripper_collision_checker_(nh_)
        , execute_gripper_movement_client_(nh_.serviceClient<deformable_manipulation_msgs::ExecuteRobotMotion>(GetExecuteRobotMotionTopic(nh_), true))
        , test_grippers_poses_client_(nh_, GetTestRobotMotionTopic(nh_), false)
        , dt_(GetRobotControlPeriod(nh_))
        , max_gripper_velocity_norm_(GetMaxGripperVelocityNorm(nh_))
        , max_dof_velocity_norm_(GetMaxDOFVelocityNorm(nh_))
        , min_controller_distance_to_obstacles_(GetControllerMinDistanceToObstacles(ph_))

        // TODO: remove this hardcoded spin period
        , spin_thread_(ROSHelpers::Spin, 0.01)

        , get_ee_poses_fn_(nullptr)
        , get_grippers_jacobian_fn_(nullptr)
        , get_collision_points_of_interest_fn_(nullptr)
        , get_collision_points_of_interest_jacobians_fn_(nullptr)
        , full_robot_collision_check_fn_(nullptr)
        , close_ik_solutions_fn_(nullptr)
        , general_ik_solution_fn_(nullptr)
        , test_path_for_collision_fn_(nullptr)
    {}

    RobotInterface::~RobotInterface()
    {
        ROS_INFO_NAMED("robot_interface", "Terminating");
        spin_thread_.join();
    }

    WorldState RobotInterface::start()
    {
    //    assert(get_grippers_jacobian_fn_ != nullptr && "Function pointers must be initialized");
    //    assert(get_collision_points_of_interest_fn_ != nullptr && "Function pointers must be initialized");
    //    assert(get_collision_points_of_interest_jacobians_fn_ != nullptr && "Function pointers must be initialized");

        const double timeout = 5.0;
        ROS_INFO_STREAM("Waiting for tf from world to bullet frame for at most " << timeout << " seconds");
        try
        {
            world_to_bullet_tf_ = EigenHelpersConversions::GeometryTransformToEigenIsometry3d(
                    tf_buffer_.lookupTransform(world_frame_name_, bullet_frame_name_, ros::Time(0), ros::Duration(timeout)).transform);
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN("%s", ex.what());
            ROS_WARN("Assuming this means that no transform has been broadcast from world to bullet, so assuming identity, but NOT broadcasting");
            world_to_bullet_tf_.setIdentity();
        }

        ROS_INFO_NAMED("robot_interface", "Waiting for the robot gripper movement service to be available");
        execute_gripper_movement_client_.waitForExistence();
        // TODO: Parameterize this ability to be enabled or not
        ROS_INFO_NAMED("robot_interface", "Waiting for the robot gripper test grippers poses to be available");
        test_grippers_poses_client_.waitForServer();

        ROS_INFO_NAMED("robot_interface", "Kickstarting the planner with a no-op");
        return commandRobotMotion_impl(noOpGripperMovement());
    }

    bool RobotInterface::ok() const
    {
        return ros::ok();
    }

    void RobotInterface::shutdown()
    {
        ros::ServiceClient shutdown_sim_client_ = nh_.serviceClient<std_srvs::Empty>(GetTerminateSimulationTopic(nh_));
        std_srvs::Empty empty;
        shutdown_sim_client_.call(empty);

        ros::shutdown();
    }

    const std::vector<GripperData>& RobotInterface::getGrippersData() const
    {
        return grippers_data_;
    }

    AllGrippersSinglePose RobotInterface::getGrippersPoses()
    {
        AllGrippersSinglePose grippers_pose(grippers_data_.size());

        for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
        {
            ros::ServiceClient gripper_pose_client =
                nh_.serviceClient<deformable_manipulation_msgs::GetGripperPose>(GetGripperPoseTopic(nh_));
            gripper_pose_client.waitForExistence();

            deformable_manipulation_msgs::GetGripperPose pose_srv_data;
            pose_srv_data.request.name = grippers_data_[gripper_ind].name_;
            if (!gripper_pose_client.call(pose_srv_data))
            {
                ROS_FATAL_STREAM_NAMED("robot_interface", "Unabled to retrieve gripper pose: " << grippers_data_[gripper_ind].name_);
            }
            CHECK_FRAME_NAME("robot_interface", world_frame_name_, pose_srv_data.response.header.frame_id);

            grippers_pose[gripper_ind] =
                    EigenHelpersConversions::GeometryPoseToEigenIsometry3d(pose_srv_data.response.pose);
        }

        return grippers_pose;
    }

    double RobotInterface::getGrippersInitialDistance()
    {
        const AllGrippersSinglePose poses = getGrippersPoses();
        if (poses.size() == 2)
        {
            const auto gripper0_translation = poses[0].translation();
            const auto gripper1_translation = poses[1].translation();
            return (gripper0_translation - gripper1_translation).norm();
        }
        else
        {
            return 0.0;
        }
    }

    WorldState RobotInterface::commandRobotMotion(
            const AllGrippersSinglePose& target_grippers_poses,
            const Eigen::VectorXd& target_robot_configuration,
            const bool robot_configuration_valid)
    {
        return commandRobotMotion_impl(
                    toRosMovementRequest(target_grippers_poses, target_robot_configuration, robot_configuration_valid));
    }

    bool RobotInterface::testRobotMotion(
            const std::vector<AllGrippersSinglePose>& test_grippers_poses,
            const std::vector<Eigen::VectorXd>& test_robot_configurations,
            const bool robot_configuration_valid,
            const TestRobotMotionFeedbackCallbackFunctionType& feedback_callback)
    {
        return testRobotMotion_impl(
                    toRosTestPosesGoal(test_grippers_poses,
                                       test_robot_configurations,
                                       robot_configuration_valid), feedback_callback);
    }

    std::vector<CollisionData> RobotInterface::checkGripperCollision(
            const AllGrippersSinglePose& grippers_poses)
    {
        return gripper_collision_checker_.gripperCollisionCheck(grippers_poses);
    }


    void RobotInterface::resetRandomSeeds(const unsigned long seed, const unsigned long num_discards)
    {
        if (reset_random_seeds_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked to reset random seeds, but function pointer is null");
            return;
        }
        reset_random_seeds_fn_(seed, num_discards);
    }

    void RobotInterface::lockEnvironment() const
    {
        if (lock_env_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked to lock environment, but function pointer is null");
            return;
        }
        return lock_env_fn_();
    }

    void RobotInterface::unlockEnvironment() const
    {
        if (unlock_env_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked to unlock environment, but function pointer is null");
            return;
        }
        return unlock_env_fn_();
    }

    const Eigen::VectorXd& RobotInterface::getJointLowerLimits() const
    {
        return joint_lower_limits_;
    }

    const Eigen::VectorXd& RobotInterface::getJointUpperLimits() const
    {
        return joint_upper_limits_;
    }

    const Eigen::VectorXd& RobotInterface::getJointWeights() const
    {
        return joint_weights_;
    }

    void RobotInterface::setActiveDOFValues(const Eigen::VectorXd& robot_configuration) const
    {
        if (set_active_dof_values_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked to set active DOF values, but function pointer is null");
            return;
        }
        set_active_dof_values_fn_(robot_configuration);
    }

    AllGrippersSinglePose RobotInterface::getGrippersPosesFunctionPointer() const
    {
        if (get_ee_poses_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked for gripper poses (function pointer input), but function pointer is null");
            return AllGrippersSinglePose();
        }
        return get_ee_poses_fn_();
    }

    // This a Jacobian between the movement of the grippers (in the gripper body frame)
    // and the movement of the robot's DOF
    Eigen::MatrixXd RobotInterface::getGrippersJacobian() const
    {
        if (get_grippers_jacobian_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked for robot jacobian, but function pointer is null");
            return Eigen::MatrixXd();
        }
        return get_grippers_jacobian_fn_();
    }

    // This looks up the points of interest as reporeted by the external robot (i.e. OpenRAVE)
    // then querrys Bullet for the data needed to do collision avoidance, and querrys OpenRAVE
    // for the Jacobian of the movement of the point relative to the robot DOF movement.
    //
    // This includes the grippers.
    std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> RobotInterface::getPointsOfInterestCollisionData()
    {
        if (get_collision_points_of_interest_fn_ == nullptr || get_collision_points_of_interest_jacobians_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked for POI collision data, but function pointer is null");
            return std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>>();
        }

        const std::vector<Eigen::Vector3d> poi = get_collision_points_of_interest_fn_();
        const std::vector<Eigen::MatrixXd> poi_jacobians = get_collision_points_of_interest_jacobians_fn_();
        assert(poi.size() == poi_jacobians.size());

        AllGrippersSinglePose poses_to_test(poi.size(), Eigen::Isometry3d::Identity());
        for (size_t ind = 0; ind < poi.size(); ++ind)
        {
            poses_to_test[ind].translation() = poi[ind];
        }
        // TODO: address the fact that we're using checkGripperCollision for everything, even non-grippers
        const std::vector<CollisionData> poi_collision_data = checkGripperCollision(poses_to_test);

    //    for (size_t idx = 0; idx < poi_collision_data.size(); ++idx)
    //    {
    //        const CollisionData& data = poi_collision_data[idx];
    //        std::cout << "Poi: " << poi[idx].transpose() << std::endl;
    //        std::cout << "Dist: " << data.distance_to_obstacle_
    //                  << " Nearest: " << data.nearest_point_to_obstacle_.transpose()
    //                  << " Normal: " << data.obstacle_surface_normal_.transpose() << std::endl;
    //        std::cout << "Jacobian:\n" << poi_jacobians[idx] << std::endl;
    //    }


        std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> results;
        results.reserve(poi.size());

        for (size_t idx = 0; idx < poi.size(); ++idx)
        {
            results.push_back({poi_collision_data[idx], poi_jacobians[idx]});
        }

        return results;
    }

    Eigen::VectorXd RobotInterface::mapGripperMotionToRobotMotion(
            const AllGrippersSinglePoseDelta& grippers_delta) const
    {
        const auto stacked_gripper_delta = EigenHelpers::VectorEigenVectorToEigenVectorX(grippers_delta);
        const auto jacobian = get_grippers_jacobian_fn_();

    //    std::cout << "Stacked delta size: " << stacked_gripper_delta.rows() << " x " << stacked_gripper_delta.cols() << std::endl;
    //    std::cout << "Jacobian size     : " << jacobian.rows() << " x " << jacobian.cols() << std::endl;

    //    std::cout << "Invoking QR solver" << std::endl;
    //    const Eigen::VectorXd result = jacobian.colPivHouseholderQr().solve(stacked_gripper_delta);
    //    std::cout << "result: " << result.transpose() << std::endl;

    //    const auto result = EigenHelpers::Pinv(jacobian, EigenHelpers::SuggestedRcond()) * stacked_gripper_delta;
        const auto result = EigenHelpers::UnderdeterminedSolver(jacobian, stacked_gripper_delta, EigenHelpers::SuggestedRcond(), EigenHelpers::SuggestedRcond());

        return result;
    }

    bool RobotInterface::checkRobotCollision() const
    {
        if (full_robot_collision_check_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked for robot collision check, but function pointer is null");
            return true;
        }
        return full_robot_collision_check_fn_();
    }

    std::vector<Eigen::VectorXd> RobotInterface::getCloseIkSolutions(
            const AllGrippersSinglePose& target_poses,
            const double max_gripper_distance) const
    {
        assert(target_poses.size() == grippers_data_.size() &&
               "This function assumes that the target poses are in order, and there is one for each gripper");
        std::vector<std::string> gripper_names(grippers_data_.size());

        for (size_t idx = 0; idx < grippers_data_.size(); ++idx)
        {
            gripper_names[idx] = grippers_data_[idx].name_;
        }

        return getCloseIkSolutions(gripper_names, target_poses, max_gripper_distance);
    }

    std::vector<Eigen::VectorXd> RobotInterface::getCloseIkSolutions(
            const std::vector<std::string>& gripper_names,
            const AllGrippersSinglePose& target_poses,
            const double max_gripper_distance) const
    {
        assert(target_poses.size() == gripper_names.size() && "Must request one pose per gripper");

        if (close_ik_solutions_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked for ik solutions, but function pointer is null");
            return std::vector<Eigen::VectorXd>(0);
        }

        return close_ik_solutions_fn_(gripper_names, target_poses, max_gripper_distance);
    }

    std::pair<bool, Eigen::VectorXd> RobotInterface::getGeneralIkSolution(
            const Eigen::VectorXd& starting_config,
            const AllGrippersSinglePose& target_poses) const
    {
        assert(target_poses.size() == grippers_data_.size() &&
               "This function assumes that the target poses are in order, and there is one for each gripper");
        std::vector<std::string> gripper_names(grippers_data_.size());

        for (size_t idx = 0; idx < grippers_data_.size(); ++idx)
        {
            gripper_names[idx] = grippers_data_[idx].name_;
        }

        return getGeneralIkSolution(starting_config, gripper_names, target_poses);
    }

    std::pair<bool, Eigen::VectorXd> RobotInterface::getGeneralIkSolution(
            const Eigen::VectorXd& starting_config,
            const std::vector<std::string>& gripper_names,
            const AllGrippersSinglePose& target_poses) const
    {
        if (general_ik_solution_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked for generalik solution, but function pointer is null");
            return {false, Eigen::VectorXd(0)};
        }
        return general_ik_solution_fn_(starting_config, gripper_names, target_poses);
    }

    bool RobotInterface::testPathForCollision(const std::vector<Eigen::VectorXd>& path) const
    {
        if (test_path_for_collision_fn_ == nullptr)
        {
            ROS_ERROR_ONCE_NAMED("robot_interface", "Asked for test_path_for_collision_fn_, but function pointer is null");
            return true;
        }
        return test_path_for_collision_fn_(path);
    }

    void RobotInterface::setCallbackFunctions(
            const std::function<void(const size_t, const size_t)>& reset_random_seeds_fn,
            const std::function<void()>& lock_env_fn,
            const std::function<void()>& unlock_env_fn,
            const std::function<std::vector<Eigen::VectorXd>     ()>& get_robot_joint_info_fn,
            const std::function<void                             (const Eigen::VectorXd& configuration)> set_active_dof_values_fn,
            const std::function<AllGrippersSinglePose            ()>& get_ee_poses_fn,
            const std::function<Eigen::MatrixXd                  ()>& get_grippers_jacobian_fn,
            const std::function<std::vector<Eigen::Vector3d>     ()>& get_collision_points_of_interest_fn,
            const std::function<std::vector<Eigen::MatrixXd>     ()>& get_collision_points_of_interest_jacobians_fn,
            const std::function<bool                             ()>& full_robot_collision_check_fn,
            const std::function<std::vector<Eigen::VectorXd>     (const std::vector<std::string>& gripper_names, const AllGrippersSinglePose& target_poses, const double max_gripper_distance)>& close_ik_solutions_fn,
            const std::function<std::pair<bool, Eigen::VectorXd> (const Eigen::VectorXd& starting_config, const std::vector<std::string>& gripper_names, const AllGrippersSinglePose& target_poses)>& general_ik_solution_fn,
            const std::function<bool                             (const std::vector<Eigen::VectorXd>& path)>& test_path_for_collision_fn)
    {
        reset_random_seeds_fn_ = reset_random_seeds_fn;
        lock_env_fn_ = lock_env_fn;
        unlock_env_fn_ = unlock_env_fn;
        set_active_dof_values_fn_ = set_active_dof_values_fn;
        get_ee_poses_fn_ = get_ee_poses_fn;
        get_grippers_jacobian_fn_ = get_grippers_jacobian_fn;
        get_collision_points_of_interest_fn_ = get_collision_points_of_interest_fn;
        get_collision_points_of_interest_jacobians_fn_ = get_collision_points_of_interest_jacobians_fn;
        full_robot_collision_check_fn_ = full_robot_collision_check_fn;
        close_ik_solutions_fn_ = close_ik_solutions_fn;
        general_ik_solution_fn_ = general_ik_solution_fn;
        test_path_for_collision_fn_ = test_path_for_collision_fn;

        // Verify the joint info makes sense - this is a guard against bad data flowwing downhill in the RRT
        assert(get_robot_joint_info_fn != nullptr);

        const std::vector<Eigen::VectorXd> joint_info = get_robot_joint_info_fn();
        assert(joint_info.size() == 3 && "Joint info is assumed to be in the order [lower_limits, upper_limits, joint_weights]");
        joint_lower_limits_ = joint_info[0];
        joint_upper_limits_ = joint_info[1];
        joint_weights_ = joint_info[2];

        assert(joint_weights_.size() > 0);
        assert(joint_weights_.size() == joint_lower_limits_.size());
        assert(joint_weights_.size() == joint_upper_limits_.size());
    }


    Eigen::Vector3d RobotInterface::transformToFrame(
            const Eigen::Vector3d& point,
            const std::string& source_frame,
            const std::string& target_frame,
            const ros::Time& time) const
    {
        tf2::Stamped<Eigen::Vector3d> point_stamped(point, time, source_frame);
        const tf2::Stamped<Eigen::Vector3d> transformed = tf_buffer_.transform(point_stamped, target_frame);
        const Eigen::Vector3d tmp(transformed.x(), transformed.y(), transformed.z());
        return tmp;
    }

    const Eigen::Isometry3d& RobotInterface::getWorldToTaskFrameTf() const
    {
        return world_to_bullet_tf_;
    }

    ////////////////////////////////////////////////////////////////////
    // ROS objects and helpers
    ////////////////////////////////////////////////////////////////////

    deformable_manipulation_msgs::ExecuteRobotMotionRequest RobotInterface::noOpGripperMovement()
    {
        deformable_manipulation_msgs::ExecuteRobotMotionRequest movement_request;
        movement_request.grippers_names = GetGripperNames(grippers_data_);
        movement_request.gripper_poses.resize(grippers_data_.size());

        // TODO: resolve code duplication between here, getGrippersPose(), and toRosTestPosesGoal() etc.
        ros::ServiceClient gripper_pose_client =
                nh_.serviceClient<deformable_manipulation_msgs::GetGripperPose>(GetGripperPoseTopic(nh_));
        gripper_pose_client.waitForExistence();
        for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
        {
            deformable_manipulation_msgs::GetGripperPose pose_srv_data;
            pose_srv_data.request.name = grippers_data_[gripper_ind].name_;
            if (!gripper_pose_client.call(pose_srv_data))
            {
                ROS_FATAL_STREAM_NAMED("robot_interface", "Unabled to retrieve gripper pose: " << grippers_data_[gripper_ind].name_);
            }
            CHECK_FRAME_NAME("robot_interface", world_frame_name_, pose_srv_data.response.header.frame_id);

            movement_request.gripper_poses[gripper_ind] = pose_srv_data.response.pose;
        }

        ros::ServiceClient robot_configuration_client =
                nh_.serviceClient<deformable_manipulation_msgs::GetRobotConfiguration>(GetRobotConfigurationTopic(nh_));
        robot_configuration_client.waitForExistence();
        deformable_manipulation_msgs::GetRobotConfiguration robot_config_srv_data;
        robot_configuration_client.call(robot_config_srv_data);
        movement_request.robot_configuration = robot_config_srv_data.response.configuration;
        movement_request.robot_configuration_valid = robot_config_srv_data.response.valid;

        movement_request.header.frame_id = world_frame_name_;
        movement_request.header.stamp = ros::Time::now();
        return movement_request;
    }

    deformable_manipulation_msgs::ExecuteRobotMotionRequest RobotInterface::toRosMovementRequest(
            const AllGrippersSinglePose& grippers_poses,
            const Eigen::VectorXd& robot_configuration,
            const bool robot_configuration_valid) const
    {
        deformable_manipulation_msgs::ExecuteRobotMotionRequest movement_request;
        movement_request.grippers_names = GetGripperNames(grippers_data_);
        movement_request.gripper_poses = EigenHelpersConversions::VectorIsometry3dToVectorGeometryPose(grippers_poses);
        movement_request.robot_configuration = EigenHelpers::EigenVectorXToStdVector(robot_configuration);
        movement_request.robot_configuration_valid = robot_configuration_valid;
        movement_request.header.frame_id = world_frame_name_;
        movement_request.header.stamp = ros::Time::now();
        return movement_request;
    }

    deformable_manipulation_msgs::TestRobotMotionGoal RobotInterface::toRosTestPosesGoal(
            const std::vector<AllGrippersSinglePose>& grippers_poses,
            const std::vector<Eigen::VectorXd>& robot_configurations,
            const bool robot_configurations_valid) const
    {
        assert(!robot_configurations_valid ||
               robot_configurations.size() == grippers_poses.size());

        deformable_manipulation_msgs::TestRobotMotionGoal goal;
        goal.gripper_names = GetGripperNames(grippers_data_);

        goal.poses_to_test.resize(grippers_poses.size());
        for (size_t pose_ind = 0; pose_ind < grippers_poses.size(); ++pose_ind)
        {
            goal.poses_to_test[pose_ind].poses =
                    EigenHelpersConversions::VectorIsometry3dToVectorGeometryPose(grippers_poses[pose_ind]);
        }

        goal.configurations_to_test.resize(robot_configurations.size());
        for (size_t config_ind = 0; config_ind < robot_configurations.size(); ++config_ind)
        {
            goal.configurations_to_test[config_ind].configuration =
                    EigenHelpers::EigenVectorXToStdVector(robot_configurations[config_ind]);
        }
        goal.robot_configurations_valid = robot_configurations_valid;

        goal.header.frame_id = world_frame_name_;
        goal.header.stamp = ros::Time::now();
        return goal;
    }

    WorldState RobotInterface::commandRobotMotion_impl(
            const deformable_manipulation_msgs::ExecuteRobotMotionRequest& movement)
    {
        deformable_manipulation_msgs::ExecuteRobotMotionResponse result;
        while (!execute_gripper_movement_client_.call(movement, result))
        {
            ROS_WARN_THROTTLE_NAMED(1.0, "robot_interface", "Sending a gripper movement to the robot failed, reconnecting");
            execute_gripper_movement_client_ =
                    nh_.serviceClient<deformable_manipulation_msgs::ExecuteRobotMotion>(GetExecuteRobotMotionTopic(nh_), true);
            execute_gripper_movement_client_.waitForExistence();
        }
        CHECK_FRAME_NAME("robot_interface", world_frame_name_, result.world_state.header.frame_id);
        return ConvertToEigenFeedback(result.world_state);
    }



    void RobotInterface::internalTestPoseFeedbackCallback(
            const deformable_manipulation_msgs::TestRobotMotionActionFeedbackConstPtr& feedback,
            const TestRobotMotionFeedbackCallbackFunctionType& feedback_callback)
    {
        ROS_INFO_STREAM_NAMED("robot_interface", "Got feedback for test number " << feedback->feedback.test_id);
        CHECK_FRAME_NAME("robot_interface", world_frame_name_, feedback->feedback.world_state.header.frame_id);
        feedback_callback(feedback->feedback.test_id, ConvertToEigenFeedback(feedback->feedback.world_state));
        if (feedback_recieved_[feedback->feedback.test_id] == false)
        {
            feedback_recieved_[feedback->feedback.test_id] = true;
            feedback_counter_--;
        }
    }


    bool RobotInterface::testRobotMotion_impl(
            const deformable_manipulation_msgs::TestRobotMotionGoal& goal,
            const TestRobotMotionFeedbackCallbackFunctionType& feedback_callback)
    {

        feedback_counter_ = goal.poses_to_test.size();
        feedback_recieved_.clear();
        feedback_recieved_.resize(goal.poses_to_test.size(), false);

        ros::Subscriber internal_feedback_sub = nh_.subscribe<deformable_manipulation_msgs::TestRobotMotionActionFeedback>(
                    GetTestRobotMotionTopic(nh_) + "/feedback",
                    1000,
                    boost::bind(&RobotInterface::internalTestPoseFeedbackCallback, this, _1, feedback_callback));

        test_grippers_poses_client_.sendGoal(goal);

        // TODO: Why am I waitingForResult and checking the feedback counter?
        // One possible reason is because messages can arrive out of order
        const bool result = test_grippers_poses_client_.waitForResult();
        while (feedback_counter_ > 0)
        {
            std::this_thread::sleep_for(std::chrono::duration<double>(0.0001));
        }

        return result;
    }
}
