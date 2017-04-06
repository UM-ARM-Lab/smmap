#include "smmap/planner.h"

#include <assert.h>
#include <numeric>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/log.hpp>

//#include "smmap/optimization.hpp"
#include "smmap/timing.hpp"

using namespace smmap;
using namespace EigenHelpersConversions;

////////////////////////////////////////////////////////////////////////////////
// Constructor and model list builder
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::Planner
 * @param error_fn
 * @param execute_trajectory_fn
 * @param vis
 * @param dt
 */
Planner::Planner(
        RobotInterface& robot,
        Visualizer& vis,
        const std::shared_ptr<TaskSpecification>& task_specification,
        const LoggingFunctionType& logging_fn)
    : nh_("")
    , ph_("~")
    , logging_fn_(logging_fn)
    , robot_(robot)
    , vis_(vis)
    , task_specification_(task_specification)
    , calculate_regret_(GetCalculateRegret(ph_))
    , reward_std_dev_scale_factor_(1.0)
    , process_noise_factor_(GetProcessNoiseFactor(ph_))
    , observation_noise_factor_(GetObservationNoiseFactor(ph_))
    , correlation_strength_factor_(GetCorrelationStrengthFactor(ph_))
    , seed_(GetPlannerSeed(ph_))
    , generator_(seed_)
{
    ROS_INFO_STREAM_NAMED("planner", "Using seed " << std::hex << seed_ );

    if (GetLoggingEnabled(nh_))
    {
        const std::string log_folder = GetLogFolder(nh_);
        Log::Log seed_log(log_folder + "seed.txt", false);
        LOG_STREAM(seed_log, std::hex << seed_);
    }
}

void Planner::addModel(DeformableModel::Ptr model)
{
    model_list_.push_back(model);
}

void Planner::createBandits()
{
    num_models_ = (ssize_t)model_list_.size();
    ROS_INFO_STREAM_NAMED("planner", "Generating bandits for " << num_models_ << " bandits");

#ifdef UCB_BANDIT
    model_utility_bandit_ = UCB1Normal<std::mt19937_64>(num_models_);
#endif
#ifdef KFMANB_BANDIT
    model_utility_bandit_ = KalmanFilterMANB<std::mt19937_64>(
                Eigen::VectorXd::Zero(num_models_),
                Eigen::VectorXd::Ones(num_models_) * 1e6);
#endif
#ifdef KFMANDB_BANDIT
    model_utility_bandit_ = KalmanFilterMANDB<std::mt19937_64>(
                Eigen::VectorXd::Zero(num_models_),
                Eigen::MatrixXd::Identity(num_models_, num_models_) * 1e6);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::checkForClothStretchingViolations
 * @param projected_paths
 * @return
 */
bool Planner::checkForClothStretchingViolations(
        const std::vector<EigenHelpers::VectorVector3d>& projected_paths)
{
    bool violations_exist = false;

    const EigenHelpers::VectorVector3d empty_path;
    EigenHelpers::VectorVector3d vis_start_points;
    EigenHelpers::VectorVector3d vis_end_points;

    // For each node, check it's projected path against it's neighbours
    for (ssize_t node_ind = 0; node_ind < (ssize_t)projected_paths.size(); ++node_ind)
    {
        if (projected_paths[node_ind].size() > 1)
        {
            vis_.visualizePoints("projected_point_path", projected_paths[node_ind], Visualizer::Magenta(), (int32_t)node_ind);
            vis_.visualizeXYZTrajectory("projected_point_path_lines", projected_paths[node_ind], Visualizer::Magenta(), (int32_t)node_ind);

            // For each neighbour, check for violations
            for (ssize_t neighbour_ind : task_specification_->getNodeNeighbours(node_ind))
            {
                // Only check a neighbour if we have not checked this pair before
                if (neighbour_ind > node_ind)
                {
                    const size_t max_time = std::min(projected_paths[node_ind].size(), projected_paths[neighbour_ind].size());

                    // At every future timestep, check for violations
                    for (size_t t = 1; t < max_time; ++t)
                    {
                        if (task_specification_->stretchingConstraintViolated(node_ind, projected_paths[node_ind][t], neighbour_ind, projected_paths[neighbour_ind][t]))
                        {
                            violations_exist = true;
                            vis_start_points.push_back(projected_paths[node_ind][t]);
                            vis_end_points.push_back(projected_paths[neighbour_ind][t]);
                        }
                    }
                }
            }

        }
        else
        {
            vis_.visualizePoints("projected_point_path", empty_path,Visualizer::Magenta(), (int32_t)node_ind);
            vis_.visualizeXYZTrajectory("projected_point_path_lines", empty_path, Visualizer::Magenta(), (int32_t)node_ind);
        }
    }

    vis_.visualizeLines("constraint_violation_detection_lines_version_1", vis_start_points, vis_end_points, Visualizer::Blue());

    return violations_exist;
}

void Planner::detectFutureConstraintViolations(const WorldState &current_world_state)
{
    ROS_INFO_NAMED("planner", "------------------------------------------------------------------------------------");
    if (task_specification_->is_dijkstras_type_task_)
    {
        std::shared_ptr<DijkstrasCoverageTask> dijkstras_task = std::dynamic_pointer_cast<DijkstrasCoverageTask>(task_specification_);

        #warning "Fix this magic number"
        const size_t NUM_SIMSTEPS = 50;

        const std_msgs::ColorRGBA gripper_color = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 0.0f, 0.6f, 1.0f);
        const std_msgs::ColorRGBA gripper_rubber_band_safe_color = Visualizer::Black();
        const std_msgs::ColorRGBA gripper_rubber_band_violation_color = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 1.0f, 1.0f, 1.0f);
        const bool verbose = false;

        //////////////////////////////////////////////////////////////////////////////////////////
        // Constraint violation Version 1 - Purely cloth overstretch
        //////////////////////////////////////////////////////////////////////////////////////////
        stopwatch(RESET);
        const auto projected_deformable_point_paths = dijkstras_task->findPathFromObjectToTarget(current_world_state.object_configuration_, dijkstras_task->getErrorThreshold(), NUM_SIMSTEPS);
        const bool stretching_violations_exist = checkForClothStretchingViolations(projected_deformable_point_paths.first);
        ROS_INFO_STREAM_NAMED("planner", "Calculated future constraint violation detection - Version 1 - in " << stopwatch(READ) << " seconds");

        //////////////////////////////////////////////////////////////////////////////////////////
        // Constraint violation Version 2a - Vector field forward "simulation" - rubber band
        /////////////////////////////////////////////////////////////////////////////////////////
        if (true)
        {
            ROS_INFO_STREAM_NAMED("planner", "Starting future constraint violation detection - Version 2a - Total steps is " << NUM_SIMSTEPS);
            assert(num_models_ == 1 && current_world_state.all_grippers_single_pose_.size() == 2);
            const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
            {
                return dijkstras_task->getErrorCorrectionVectorsAndWeights(world_state.object_configuration_, projected_deformable_point_paths.second);
            };

            // Create the initial rubber band if needed
            if (unlikely(virtual_rubber_band_between_grippers_version2a_ == nullptr))
            {
                virtual_rubber_band_between_grippers_version2a_ = std::make_shared<VirtualRubberBand>(current_world_state, dijkstras_task, vis_);
            }

            // Visualize the initial rubber band
            const std_msgs::ColorRGBA& gripper_visualization_color = virtual_rubber_band_between_grippers_version2a_->isOverstreched() ? gripper_rubber_band_violation_color: gripper_rubber_band_safe_color;
            virtual_rubber_band_between_grippers_version2a_->visualize("gripper_rubber_band_version2a", gripper_visualization_color, 0);

            //////////////////////////////////////////////////////////////////////////////////////////
            WorldState world_state_copy = current_world_state;
            VirtualRubberBand virtual_rubber_band_between_grippers_copy = *virtual_rubber_band_between_grippers_version2a_.get();
            for (size_t t = 0; t < NUM_SIMSTEPS; ++t)
            {
                // Make a copy so that we can reference this original state when we forward simulate the rubber band
                const AllGrippersSinglePose starting_grippers_single_pose = world_state_copy.all_grippers_single_pose_;

                // Move the grippers and cloth
                DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, current_world_state, robot_.dt_);
                const std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> robot_command = model_list_[0]->getSuggestedGrippersCommand(model_input_data, dijkstras_task->work_space_grid_.minStepDimension() / robot_.dt_, task_specification_->collisionScalingFactor());
                world_state_copy.all_grippers_single_pose_ = kinematics::applyTwist(world_state_copy.all_grippers_single_pose_, robot_command.first);
                world_state_copy.gripper_collision_data_ = robot_.checkGripperCollision(world_state_copy.all_grippers_single_pose_);
                world_state_copy.sim_time_ += robot_.dt_;
                for (ssize_t node_ind = 0; node_ind < world_state_copy.object_configuration_.cols(); ++node_ind)
                {
                    if (projected_deformable_point_paths.first[node_ind].size() > t + 1)
                    {
                        world_state_copy.object_configuration_.col(node_ind) = projected_deformable_point_paths.first[node_ind][t + 1];
                    }
                }

                // Move the virtual rubber band to follow the grippers, projecting out of collision as needed
                virtual_rubber_band_between_grippers_copy.forwardSimulateVirtualRubberBand(
                            world_state_copy.all_grippers_single_pose_[0].translation() - starting_grippers_single_pose[0].translation(),
                            world_state_copy.all_grippers_single_pose_[1].translation() - starting_grippers_single_pose[1].translation(),
                            verbose);

                // Visualize
                const std_msgs::ColorRGBA& gripper_visualization_color =  virtual_rubber_band_between_grippers_copy.isOverstreched() ? gripper_rubber_band_violation_color : gripper_rubber_band_safe_color;
                virtual_rubber_band_between_grippers_copy.visualize("gripper_rubber_band_version2a", gripper_visualization_color, (int32_t)t+1);
                vis_.visualizeGrippers("forward_simulated_grippers_version2a", world_state_copy.all_grippers_single_pose_, gripper_color);
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////
        // Constraint violation Version 2b - Jacobian forward simulation - rubber band
        //////////////////////////////////////////////////////////////////////////////////////////
        if (false)
        {
            ROS_INFO_STREAM_NAMED("planner", "Starting future constraint violation detection - Version 2b - Total steps is " << NUM_SIMSTEPS);
            assert(num_models_ == 1 && current_world_state.all_grippers_single_pose_.size() == 2);
            const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
            {
//                return task_specification_->calculateDesiredDirection(world_state);
                return task_specification_->calculateObjectErrorCorrectionDelta(world_state);
            };

            // Create the initial rubber band if needed
            if (unlikely(virtual_rubber_band_between_grippers_version2b_ == nullptr))
            {
                virtual_rubber_band_between_grippers_version2b_ = std::make_shared<VirtualRubberBand>(current_world_state, dijkstras_task, vis_);
            }

            // Visualize the initial rubber band
            const std_msgs::ColorRGBA& gripper_visualization_color = virtual_rubber_band_between_grippers_version2b_->isOverstreched() ? gripper_rubber_band_violation_color: gripper_rubber_band_safe_color;
            virtual_rubber_band_between_grippers_version2b_->visualize("gripper_rubber_band_version2b", gripper_visualization_color, 0);

            //////////////////////////////////////////////////////////////////////////////////////////
            WorldState world_state_copy = current_world_state;
            VirtualRubberBand virtual_rubber_band_between_grippers_copy = *virtual_rubber_band_between_grippers_version2b_.get();
            for (size_t t = 0; t < NUM_SIMSTEPS; ++t)
            {
                // Make a copy so that we can reference this original state when we forward simulate the rubber band
                const AllGrippersSinglePose starting_grippers_single_pose = world_state_copy.all_grippers_single_pose_;

                // Move the grippers and cloth
                DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, current_world_state, robot_.dt_);
                const std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> robot_command = model_list_[0]->getSuggestedGrippersCommand(model_input_data, robot_.max_gripper_velocity_, task_specification_->collisionScalingFactor());
                world_state_copy.all_grippers_single_pose_ = kinematics::applyTwist(world_state_copy.all_grippers_single_pose_, robot_command.first);
                world_state_copy.gripper_collision_data_ = robot_.checkGripperCollision(world_state_copy.all_grippers_single_pose_);
                world_state_copy.sim_time_ += robot_.dt_;
                world_state_copy.object_configuration_ += robot_command.second;

                // Move the virtual rubber band to follow the grippers, projecting out of collision as needed
                virtual_rubber_band_between_grippers_copy.forwardSimulateVirtualRubberBand(
                            world_state_copy.all_grippers_single_pose_[0].translation() - starting_grippers_single_pose[0].translation(),
                            world_state_copy.all_grippers_single_pose_[1].translation() - starting_grippers_single_pose[1].translation(),
                            verbose);

                // Visualize
                const std_msgs::ColorRGBA& gripper_visualization_color =  virtual_rubber_band_between_grippers_copy.isOverstreched() ? gripper_rubber_band_violation_color : gripper_rubber_band_safe_color;
                virtual_rubber_band_between_grippers_copy.visualize("gripper_rubber_band_version2b", gripper_visualization_color, (int32_t)t+1);
                vis_.visualizeGrippers("forward_simulated_grippers_version2b", world_state_copy.all_grippers_single_pose_, gripper_color);
            }
        }
    }
    else
    {
        ROS_ERROR_NAMED("planner", "Unable to do future constraint violation detection");
    }
}

////////////////////////////////////////////////////////////////////////////////
// The one function that gets invoked externally
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::sendNextCommand
 * @param current_world_state
 * @return
 */
WorldState Planner::sendNextCommand(const WorldState& current_world_state)
{
    ROS_INFO_NAMED("planner", "------------------------------------------------------------------------------------");
    const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return task_specification_->calculateDesiredDirection(world_state);
    };
    const ObjectDeltaAndWeight task_desired_motion = task_desired_direction_fn(current_world_state);
//    visualizeDesiredMotion(current_world_state, task_desired_motion);

    DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, current_world_state, robot_.dt_);

    // Pick an arm to use
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);
    const bool get_action_for_all_models = model_utility_bandit_.generateAllModelActions();
    ROS_INFO_STREAM_COND_NAMED(num_models_ > 1, "planner", "Using model index " << model_to_use);

    // Querry each model for it's best gripper delta
    stopwatch(RESET);
    std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>> suggested_robot_commands(num_models_);
    #pragma omp parallel for
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        if (calculate_regret_ || get_action_for_all_models || (ssize_t)model_ind == model_to_use)
        {
            suggested_robot_commands[model_ind] =
                model_list_[model_ind]->getSuggestedGrippersCommand(
                        model_input_data,
                        robot_.max_gripper_velocity_,
                        task_specification_->collisionScalingFactor());
        }
    }
    // Measure the time it took to pick a model
    ROS_INFO_STREAM_NAMED("planner", "Calculated model suggestions and picked one in " << stopwatch(READ) << " seconds");

    //
    std::vector<double> individual_rewards(num_models_, std::numeric_limits<double>::infinity());
    if (calculate_regret_ && num_models_ > 1)
    {
        stopwatch(RESET);
        const double prev_error = task_specification_->calculateError(current_world_state.object_configuration_);
        const auto test_feedback_fn = [&] (const size_t model_ind, const WorldState& world_state)
        {
            individual_rewards[model_ind] = prev_error - task_specification_->calculateError(world_state.object_configuration_);
            return;
        };

        std::vector<AllGrippersSinglePose> poses_to_test(num_models_);
        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            poses_to_test[model_ind] = kinematics::applyTwist(current_world_state.all_grippers_single_pose_, suggested_robot_commands[model_ind].first);
        }
        robot_.testGrippersPoses(poses_to_test, test_feedback_fn);

        ROS_INFO_STREAM_NAMED("planner", "Collected data to calculate regret in " << stopwatch(READ) << " seconds");
    }


    // Execute the command
    const AllGrippersSinglePoseDelta& selected_command = suggested_robot_commands[(size_t)model_to_use].first;
    ObjectPointSet predicted_object_delta = model_list_[(size_t)model_to_use]->getObjectDelta(model_input_data, selected_command);
    const Eigen::Map<Eigen::VectorXd> predicted_object_delta_as_vector(predicted_object_delta.data(), predicted_object_delta.size());
    ROS_INFO_STREAM_NAMED("planner", "Sending command to robot, action norm:  " << MultipleGrippersVelocity6dNorm(selected_command));
    ROS_INFO_STREAM_NAMED("planner", "Task desired deformable movement norm:  " << EigenHelpers::WeightedNorm(task_desired_motion.delta, task_desired_motion.weight));
    ROS_INFO_STREAM_NAMED("planner", "Task predicted deformable movment norm: " << EigenHelpers::WeightedNorm(predicted_object_delta_as_vector, task_desired_motion.weight));
    WorldState world_feedback = robot_.sendGrippersPoses(kinematics::applyTwist(current_world_state.all_grippers_single_pose_, selected_command));

    const bool verbose = false;
    if (virtual_rubber_band_between_grippers_version2a_ != nullptr)
    {
        virtual_rubber_band_between_grippers_version2a_->forwardSimulateVirtualRubberBand(
                    world_feedback.all_grippers_single_pose_[0].translation() - current_world_state.all_grippers_single_pose_[0].translation(),
                    world_feedback.all_grippers_single_pose_[1].translation() - current_world_state.all_grippers_single_pose_[1].translation(),
                    verbose);
    }

    if (virtual_rubber_band_between_grippers_version2b_ != nullptr)
    {
        virtual_rubber_band_between_grippers_version2b_->forwardSimulateVirtualRubberBand(
                    world_feedback.all_grippers_single_pose_[0].translation() - current_world_state.all_grippers_single_pose_[0].translation(),
                    world_feedback.all_grippers_single_pose_[1].translation() - current_world_state.all_grippers_single_pose_[1].translation(),
                    verbose);
    }

    ROS_INFO_NAMED("planner", "Updating models and logging data");
    ROS_INFO_STREAM_NAMED("planner", "Correlation strength factor: " << correlation_strength_factor_);
    updateModels(current_world_state, task_desired_motion, suggested_robot_commands, model_to_use, world_feedback);

    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), model_to_use, individual_rewards, correlation_strength_factor_);

    return world_feedback;
}

void Planner::visualizeDesiredMotion(const WorldState& current_world_state, const ObjectDeltaAndWeight& desired_motion)
{
    ssize_t num_nodes = current_world_state.object_configuration_.cols();
    std::vector<std_msgs::ColorRGBA> colors((size_t)num_nodes);
    for (size_t node_ind = 0; node_ind < (size_t)num_nodes; node_ind++)
    {
        colors[node_ind].r = (float)desired_motion.weight((ssize_t)node_ind * 3);
        colors[node_ind].g = 0.0f;
        colors[node_ind].b = 0.0f;
        colors[node_ind].a = desired_motion.weight((ssize_t)node_ind * 3) > 0 ? 1.0f : 0.0f;
    }
    task_specification_->visualizeDeformableObject(
            vis_,
            "desired_position",
            AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
            colors);

    if (task_specification_->deformable_type_ == DeformableType::CLOTH)
    {
        vis_.visualizeObjectDelta(
                    "desired_position",
                    current_world_state.object_configuration_,
                    AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
                    Visualizer::Green());
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal helpers for the getNextTrajectory() function
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::updateModels
 * @param suggested_trajectories
 * @param model_used
 * @param world_feedback
 */
void Planner::updateModels(const WorldState& starting_world_state,
        const ObjectDeltaAndWeight& task_desired_motion,
        const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands,
        const ssize_t model_used,
        const WorldState& world_feedback)
{
    // First we update the bandit algorithm
    const double starting_error = task_specification_->calculateError(starting_world_state.object_configuration_);
    const double true_error_reduction = starting_error - task_specification_->calculateError(world_feedback.object_configuration_);
    reward_std_dev_scale_factor_ = std::max(1e-10, 0.9 * reward_std_dev_scale_factor_ + 0.1 * std::abs(true_error_reduction));
    const double process_noise_scaling_factor = process_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);
    const double observation_noise_scaling_factor = observation_noise_factor_ * std::pow(reward_std_dev_scale_factor_, 2);

#ifdef UCB_BANDIT
    (void)task_desired_motion;
    (void)suggested_commands;
    (void)process_noise_scaling_factor;
    (void)observation_noise_scaling_factor;
    model_utility_bandit_.updateArms(model_used, true_error_reduction);
#endif
#ifdef KFMANB_BANDIT
    (void)task_desired_motion;
    (void)suggested_commands;
    model_utility_bandit_.updateArms(process_noise_scaling_factor * Eigen::VectorXd::Ones(num_models_), model_used, true_error_reduction, observation_noise_scaling_factor * 1.0);
#endif
#ifdef KFMANDB_BANDIT
    (void)task_desired_motion;

    const Eigen::MatrixXd process_noise = calculateProcessNoise(suggested_commands);
//    const Eigen::VectorXd observed_reward = calculateObservedReward(starting_world_state, task_desired_motion, model_used, world_feedback);
//    const Eigen::MatrixXd observation_matrix = Eigen::MatrixXd::Identity(num_models_, num_models_);
//    const Eigen::MatrixXd observation_noise = calculateObservationNoise(process_noise, model_used);

    Eigen::MatrixXd observation_matrix = Eigen::RowVectorXd::Zero(num_models_);
    observation_matrix(0, model_used) = 1.0;
    const Eigen::VectorXd observed_reward = Eigen::VectorXd::Ones(1) * true_error_reduction;
    const Eigen::MatrixXd observation_noise = Eigen::MatrixXd::Ones(1, 1);

    model_utility_bandit_.updateArms(
                process_noise_scaling_factor * process_noise,
                observation_matrix,
                observed_reward,
                observation_noise_scaling_factor * observation_noise);
#endif

    // Then we allow each model to update itself based on the new data
    #pragma omp parallel for
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        model_list_[model_ind]->updateModel(starting_world_state, world_feedback);
    }
}

/**
 * @brief Planner::calculateProcessNoise
 * @param suggested_commands
 * @return
 */
Eigen::MatrixXd Planner::calculateProcessNoise(const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands)
{
    std::vector<double> grippers_velocity_norms((size_t)num_models_);

    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        grippers_velocity_norms[model_ind] = MultipleGrippersVelocity6dNorm(suggested_commands[model_ind].first);
    }

    Eigen::MatrixXd process_noise = Eigen::MatrixXd::Identity(num_models_, num_models_);
    for (ssize_t i = 0; i < num_models_; i++)
    {
        for (ssize_t j = i+1; j < num_models_; j++)
        {
            if (grippers_velocity_norms[(size_t)i] != 0 &&
                grippers_velocity_norms[(size_t)j] != 0)
            {
                process_noise(i, j) =
                        MultipleGrippersVelocityDotProduct(
                            suggested_commands[(size_t)i].first,
                            suggested_commands[(size_t)j].first)
                        / (grippers_velocity_norms[(size_t)i] * grippers_velocity_norms[(size_t)j]);
            }
            else if (grippers_velocity_norms[(size_t)i] == 0 &&
                     grippers_velocity_norms[(size_t)j] == 0)
            {
                process_noise(i, j) = 1;
            }
            else
            {
                process_noise(i, j) = 0;
            }

            process_noise(j, i) = process_noise(i, j);
        }
    }

    return correlation_strength_factor_ * process_noise + (1.0 - correlation_strength_factor_) * Eigen::MatrixXd::Identity(num_models_, num_models_);
}
