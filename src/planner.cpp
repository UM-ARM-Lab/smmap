#include "smmap/planner.h"

#include <future>
#include <assert.h>
#include <numeric>
#include <memory>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/log.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/simple_kmeans_clustering.hpp>
#include <arc_utilities/simple_astar_planner.hpp>
#include <arc_utilities/get_neighbours.hpp>
#include <arc_utilities/shortcut_smoothing.hpp>

#include "smmap/timing.hpp"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;
using namespace EigenHelpersConversions;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor and model list builder
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    , seed_(GetPlannerSeed(ph_))
    , generator_(seed_)
    , logging_fn_(logging_fn)
    , robot_(robot)
    , vis_(vis)
    , task_specification_(task_specification)
    , dijkstras_task_(nullptr)
    , calculate_regret_(GetCalculateRegret(ph_))
    , reward_std_dev_scale_factor_(1.0)
    , process_noise_factor_(GetProcessNoiseFactor(ph_))
    , observation_noise_factor_(GetObservationNoiseFactor(ph_))
    , correlation_strength_factor_(GetCorrelationStrengthFactor(ph_))
    , max_lookahead_steps_(GetNumLookaheadSteps(ph_))
    , max_grippers_pose_history_length_(GetMaxGrippersPoseHistoryLength(ph_))
    , executing_global_gripper_trajectory_(false)
    , global_plan_current_timestep_(-1)
    , global_plan_gripper_trajectory_(0)
    , rrt_helper_(nullptr)
{
    // Pass in all the config values that the RRT needs; for example goal bias, step size, etc.
    if (task_specification_->is_dijkstras_type_task_)
    {
        dijkstras_task_ = std::dynamic_pointer_cast<DijkstrasCoverageTask>(task_specification_);

        rrt_helper_ = std::unique_ptr<RRTHelper>(
                    new RRTHelper(
                        dijkstras_task_->environment_sdf_,
                        vis_,
                        generator_,
                        dijkstras_task_->work_space_grid_.getXMin(),
                        dijkstras_task_->work_space_grid_.getXMax(),
                        dijkstras_task_->work_space_grid_.getYMin(),
                        dijkstras_task_->work_space_grid_.getYMax(),
                        dijkstras_task_->work_space_grid_.getZMin(),
                        dijkstras_task_->work_space_grid_.getZMax(),
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        GetGripperRadius(),
                        GetRRTHomotopyDistancePenalty(),
                        GetRRTMaxShortcutIndexDistance(),
                        GetRRTMaxSmoothingIterations(),
                        GetRRTMaxFailedSmoothingIterations()));
    }

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
                VectorXd::Zero(num_models_),
                VectorXd::Ones(num_models_) * 1e6);
#endif
#ifdef KFMANDB_BANDIT
    model_utility_bandit_ = KalmanFilterMANDB<std::mt19937_64>(
                VectorXd::Zero(num_models_),
                MatrixXd::Identity(num_models_, num_models_) * 1e6);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The two functions that gets invoked externally (repeatedly)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::sendNextCommand
 * @param current_world_state
 * @return
 */
WorldState Planner::sendNextCommand(
        const WorldState& world_state)
{
    ROS_INFO_NAMED("planner", "---------------------------- Start of Loop -----------------------------------------");
    const double current_error = task_specification_->calculateError(world_state);
    ROS_INFO_STREAM_NAMED("planner", "Planner/Task sim time " << world_state.sim_time_ << "\t Error: " << current_error);

    if (task_specification_->is_dijkstras_type_task_ && world_state.all_grippers_single_pose_.size() == 2)
    {
        if (executing_global_gripper_trajectory_)
        {
            return sendNextCommandUsingGlobalGripperPlannerResults(world_state);
        }
        else
        {
            const auto detection_results = detectFutureConstraintViolations(world_state);
            const std::vector<VectorVector3d>& projected_deformable_point_paths = detection_results.first;
            const std::vector<VirtualRubberBand>& projected_rubber_bands = detection_results.second;

            const bool global_planner_needed_due_to_overstretch =
                    globalPlannerNeededDueToOverstretch(projected_rubber_bands);

            const bool global_planner_needed_due_to_collision =
                    globalPlannerNeededDueToCollision(world_state);

            ROS_INFO_NAMED("planner", "----------------------------------------------------------------------------");

            if (global_planner_needed_due_to_overstretch || global_planner_needed_due_to_collision)
            {
                vis_.deleteObjects("desired_position", 1, 100);

                ROS_WARN_COND_NAMED(global_planner_needed_due_to_overstretch, "planner", "Invoking global planner due to overstretch");
                ROS_WARN_COND_NAMED(global_planner_needed_due_to_collision, "planner", "Invoking global planner due to collision");

                vis_.deleteObjects("projected_gripper_rubber_band", 1, (int32_t)max_lookahead_steps_ + 10);
                vis_.deleteObjects("projected_grippers",            1, (int32_t)max_lookahead_steps_ + 10);
                vis_.deleteObjects("projected_point_path",          1, (int32_t)projected_deformable_point_paths.size() + 10);
                vis_.deleteObjects("projected_point_path_lines",    1, (int32_t)projected_deformable_point_paths.size() + 10);

                rrt_helper_->addBandToBlacklist(virtual_rubber_band_between_grippers_->getVectorRepresentation());
                planGlobalGripperTrajectory(world_state);

                ROS_INFO_NAMED("planner", "----------------------------------------------------------------------------");
                return sendNextCommandUsingGlobalGripperPlannerResults(world_state);
            }
            else
            {
                return sendNextCommandUsingLocalController(world_state);
            }
        }
    }
    else
    {
        ROS_WARN_NAMED("planner", "Unable to do future constraint violation detection");
        return sendNextCommandUsingLocalController(world_state);
    }
}

void Planner::visualizeDesiredMotion(
        const WorldState& current_world_state,
        const ObjectDeltaAndWeight& desired_motion,
        const bool visualization_enabled) const
{
    if (visualization_enabled)
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gripper movement functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

WorldState Planner::sendNextCommandUsingLocalController(
        const WorldState& current_world_state)
{
    const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return task_specification_->calculateDesiredDirection(world_state);
    };
    const ObjectDeltaAndWeight task_desired_motion = task_desired_direction_fn(current_world_state);
    const DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, current_world_state, robot_.dt_);

//    visualizeDesiredMotion(current_world_state, task_desired_motion);

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
    ROS_INFO_STREAM_NAMED("planner", "Calculated model suggestions and picked one in  " << stopwatch(READ) << " seconds");

    // Calculate regret if we need to
    std::vector<double> individual_rewards(num_models_, std::numeric_limits<double>::infinity());
    if (calculate_regret_ && num_models_ > 1)
    {
        stopwatch(RESET);
        const double prev_error = task_specification_->calculateError(current_world_state);
        const auto test_feedback_fn = [&] (const size_t model_ind, const WorldState& world_state)
        {
            individual_rewards[model_ind] = prev_error - task_specification_->calculateError(world_state);
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
    ROS_INFO_STREAM_NAMED("planner", "Sending command to robot, action norm:  " << MultipleGrippersVelocity6dNorm(selected_command));
    const WorldState world_feedback = robot_.sendGrippersPoses(kinematics::applyTwist(current_world_state.all_grippers_single_pose_, selected_command));

    if (virtual_rubber_band_between_grippers_ != nullptr)
    {
        const bool verbose = false;
        virtual_rubber_band_between_grippers_->forwardSimulateVirtualRubberBandToEndpointTargets(
                    world_feedback.all_grippers_single_pose_[0].translation(),
                    world_feedback.all_grippers_single_pose_[1].translation(),
                    verbose);

        // Keep the last N grippers positions recorded to detect if the grippers are stuck
        grippers_pose_history_.push_back(world_feedback.all_grippers_single_pose_);
        if (grippers_pose_history_.size() > max_grippers_pose_history_length_)
        {
            grippers_pose_history_.erase(grippers_pose_history_.begin());
        }
    }

    ROS_INFO_NAMED("planner", "Updating models and logging data");
    updateModels(current_world_state, task_desired_motion, suggested_robot_commands, model_to_use, world_feedback);

    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), model_to_use, individual_rewards);

    return world_feedback;
}

WorldState Planner::sendNextCommandUsingGlobalGripperPlannerResults(
        const WorldState& current_world_state)
{
    (void)current_world_state;
    assert(executing_global_gripper_trajectory_);
    assert(global_plan_gripper_trajectory_.size() < 5000);
    assert(global_plan_current_timestep_ < 5000);
    assert(global_plan_current_timestep_ < global_plan_gripper_trajectory_.size());

    const WorldState world_feedback = robot_.sendGrippersPoses(global_plan_gripper_trajectory_[global_plan_current_timestep_]);
    const bool verbose = false;
    virtual_rubber_band_between_grippers_->forwardSimulateVirtualRubberBandToEndpointTargets(
                world_feedback.all_grippers_single_pose_[0].translation(),
                world_feedback.all_grippers_single_pose_[1].translation(),
                verbose);

    ++global_plan_current_timestep_;
    if (global_plan_current_timestep_ == global_plan_gripper_trajectory_.size())
    {
        std::cerr << "Global plan finished, resetting grippers pose history\n";

        executing_global_gripper_trajectory_ = false;
        grippers_pose_history_.clear();

        vis_.deleteObjects(RRTHelper::RRT_SOLUTION_GRIPPER_A_NS, 1, 200);
        vis_.deleteObjects(RRTHelper::RRT_SOLUTION_GRIPPER_B_NS, 1, 200);
        vis_.deleteObjects(RRTHelper::RRT_SOLUTION_RUBBER_BAND_NS, 1, 200);
        vis_.deleteObjects(RRTHelper::RRT_SHORTCUT_REMAINDER_NS, 1, 200);
    }

    const std::vector<double> fake_rewards(model_list_.size(), NAN);
    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), -1, fake_rewards);

    return world_feedback;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constraint violation detection
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Planner::visualizeProjectedPaths(
        const std::vector<VectorVector3d>& projected_paths,
        const bool visualization_enabled)
{
    if (visualization_enabled)
    {
        for (ssize_t node_ind = 0; node_ind < (ssize_t)projected_paths.size(); ++node_ind)
        {
            if (projected_paths[node_ind].size() > 1)
            {
                vis_.visualizePoints("projected_point_path", projected_paths[node_ind], Visualizer::Magenta(), (int32_t)node_ind + 1);
                vis_.visualizeXYZTrajectory("projected_point_path_lines", projected_paths[node_ind], Visualizer::Magenta(), (int32_t)node_ind + 1);
            }
            else
            {
                const VectorVector3d empty_path;
                vis_.visualizePoints("projected_point_path", empty_path,Visualizer::Magenta(), (int32_t)node_ind + 1);
                vis_.visualizeXYZTrajectory("projected_point_path_lines", empty_path, Visualizer::Magenta(), (int32_t)node_ind + 1);
            }
        }
    }
}

/**
 * @brief Planner::checkForClothStretchingViolations
 * @param projected_paths
 * @return
 */
bool Planner::checkForClothStretchingViolations(
        const std::vector<VectorVector3d>& projected_paths,
        const bool visualization_enabled)
{
    bool violations_exist = false;

    VectorVector3d vis_start_points;
    VectorVector3d vis_end_points;

    // For each node, check it's projected path against it's neighbours
    for (ssize_t node_ind = 0; node_ind < (ssize_t)projected_paths.size(); ++node_ind)
    {
        if (projected_paths[node_ind].size() > 1)
        {
            // For each neighbour, check for violations
            for (ssize_t neighbour_ind : dijkstras_task_->getNodeNeighbours(node_ind))
            {
                // Only check a neighbour if we have not checked this pair before
                if (neighbour_ind > node_ind)
                {
                    const size_t max_time = std::min(projected_paths[node_ind].size(), projected_paths[neighbour_ind].size());

                    // At every future timestep, check for violations
                    for (size_t t = 1; t < max_time; ++t)
                    {
                        if (dijkstras_task_->stretchingConstraintViolated(node_ind, projected_paths[node_ind][t], neighbour_ind, projected_paths[neighbour_ind][t]))
                        {
                            violations_exist = true;
                            if (visualization_enabled)
                            {
                                vis_start_points.push_back(projected_paths[node_ind][t]);
                                vis_end_points.push_back(projected_paths[neighbour_ind][t]);
                            }
                        }
                    }
                }
            }

        }
    }

    if (visualization_enabled)
    {
        vis_.visualizeLines("constraint_violation_detection_lines_version_1", vis_start_points, vis_end_points, Visualizer::Blue());
    }

    return violations_exist;
}

size_t sizeOfLargestVector(const std::vector<VectorVector3d>& vectors)
{
    size_t largest_vector = 0;

    for (size_t idx = 0; idx < vectors.size(); ++idx)
    {
        largest_vector = std::max(largest_vector, vectors[idx].size());
    }

    return largest_vector;
}

std::pair<std::vector<VectorVector3d>, std::vector<VirtualRubberBand>> Planner::detectFutureConstraintViolations(
        const WorldState& current_world_state,
        const bool visualization_enabled)
{
    assert(task_specification_->is_dijkstras_type_task_ && current_world_state.all_grippers_single_pose_.size() == 2);
    std::pair<std::vector<VectorVector3d>, std::vector<VirtualRubberBand>> projected_deformable_point_paths_and_projected_virtual_rubber_bands;

    const static std_msgs::ColorRGBA gripper_color = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 0.0f, 0.6f, 1.0f);
    const static std_msgs::ColorRGBA rubber_band_safe_color = Visualizer::Black();
    const static std_msgs::ColorRGBA rubber_band_violation_color = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 1.0f, 1.0f, 1.0f);
    const bool verbose = false;

    vis_.deleteObjects("projected_gripper_rubber_band", 1, (int32_t)max_lookahead_steps_ + 10);
    vis_.deleteObjects("projected_grippers", 1, (int32_t)max_lookahead_steps_ + 10);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constraint violation Version 1 - Purely cloth overstretch
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    stopwatch(RESET);
    const std::vector<VectorVector3d> projected_deformable_point_paths = dijkstras_task_->findPathFromObjectToTarget(current_world_state, max_lookahead_steps_);

    const size_t actual_lookahead_steps = sizeOfLargestVector(projected_deformable_point_paths) - 1;
    // sizeOfLargest(...) should be at least 2, so this assert should always be true
    assert(actual_lookahead_steps <= max_lookahead_steps_);

    ROS_INFO_STREAM_NAMED("planner", "Calculated projected cloth paths                 - Version 1 - in " << stopwatch(READ) << " seconds");
    visualizeProjectedPaths(projected_deformable_point_paths, visualization_enabled);
    projected_deformable_point_paths_and_projected_virtual_rubber_bands.first = projected_deformable_point_paths;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constraint violation Version 2a - Vector field forward "simulation" - rubber band
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ROS_INFO_STREAM_NAMED("planner", "Starting future constraint violation detection   - Version 2a - Total steps taken " << actual_lookahead_steps);
    assert(num_models_ == 1 && current_world_state.all_grippers_single_pose_.size() == 2);
    const auto& correspondences = dijkstras_task_->getCoverPointCorrespondences(current_world_state);
    const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return dijkstras_task_->calculateErrorCorrectionDeltaFixedCorrespondences(world_state, correspondences.correspondences_);
    };

    // Create the initial rubber band if needed
    if (unlikely(virtual_rubber_band_between_grippers_ == nullptr))
    {
        virtual_rubber_band_between_grippers_ = std::make_shared<VirtualRubberBand>(
                    current_world_state.all_grippers_single_pose_[0].translation(),
                    current_world_state.all_grippers_single_pose_[1].translation(),
                    dijkstras_task_, vis_);
    }
    virtual_rubber_band_between_grippers_->visualize("projected_gripper_rubber_band", rubber_band_safe_color, rubber_band_violation_color, 1, visualization_enabled);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    WorldState world_state_copy = current_world_state;
    VirtualRubberBand virtual_rubber_band_between_grippers_copy = *virtual_rubber_band_between_grippers_.get();
    const DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, world_state_copy, robot_.dt_);

    projected_deformable_point_paths_and_projected_virtual_rubber_bands.second.reserve(actual_lookahead_steps);
    for (size_t t = 0; t < actual_lookahead_steps; ++t)
    {
        // Determine what direction to move the grippers
        const std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> robot_command =
                model_list_[0]->getSuggestedGrippersCommand(
                    model_input_data,
                    dijkstras_task_->work_space_grid_.minStepDimension() / robot_.dt_,
                    dijkstras_task_->collisionScalingFactor());

        // Move the grippers forward
        world_state_copy.all_grippers_single_pose_ = kinematics::applyTwist(world_state_copy.all_grippers_single_pose_, robot_command.first);
        world_state_copy.all_grippers_single_pose_[0].translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollision3d(world_state_copy.all_grippers_single_pose_[0].translation());
        world_state_copy.all_grippers_single_pose_[1].translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollision3d(world_state_copy.all_grippers_single_pose_[1].translation());

        auto collision_check_future = std::async(std::launch::async, &RobotInterface::checkGripperCollision, &robot_, world_state_copy.all_grippers_single_pose_);

        // Move the cloth forward - copy the projected state of the cloth into the world_state_copy
        world_state_copy.sim_time_ += robot_.dt_;
        for (ssize_t node_ind = 0; node_ind < world_state_copy.object_configuration_.cols(); ++node_ind)
        {
            if (projected_deformable_point_paths[node_ind].size() > t + 1)
            {
                world_state_copy.object_configuration_.col(node_ind) = projected_deformable_point_paths[node_ind][t + 1];
            }
        }

        // Move the virtual rubber band to follow the grippers, projecting out of collision as needed
        virtual_rubber_band_between_grippers_copy.forwardSimulateVirtualRubberBandToEndpointTargets(
                    world_state_copy.all_grippers_single_pose_[0].translation(),
                    world_state_copy.all_grippers_single_pose_[1].translation(),
                    verbose);
        projected_deformable_point_paths_and_projected_virtual_rubber_bands.second.push_back(virtual_rubber_band_between_grippers_copy);

        // Visualize
        virtual_rubber_band_between_grippers_copy.visualize("projected_gripper_rubber_band", rubber_band_safe_color, rubber_band_violation_color, (int32_t)t + 2, visualization_enabled);
        vis_.visualizeGrippers("projected_grippers", world_state_copy.all_grippers_single_pose_, gripper_color);

        // Finish collecting the gripper collision data
        world_state_copy.gripper_collision_data_ = collision_check_future.get();
    }
    ROS_INFO_STREAM_NAMED("planner", "Calculated future constraint violation detection - Version 2a - in " << stopwatch(READ) << " seconds");

    return projected_deformable_point_paths_and_projected_virtual_rubber_bands;
}

bool Planner::globalPlannerNeededDueToOverstretch(
        const std::vector<VirtualRubberBand>& projected_rubber_bands) const
{
    size_t num_violations = 0;

    for (size_t t = 0; t < projected_rubber_bands.size(); ++t)
    {
        const VirtualRubberBand& band = projected_rubber_bands[t];
        const double band_length = band.totalLength();
        const std::pair<Eigen::Vector3d, Eigen::Vector3d> endpoints = band.getEndpoints();
        const double distance_between_endpoints = (endpoints.first - endpoints.second).norm();

        if (band.isOverstretched() && !CloseEnough(band_length, distance_between_endpoints, 1e-6))
        {
            ++num_violations;
        }
    }

    return num_violations > (projected_rubber_bands.size() / 2);
}

bool Planner::globalPlannerNeededDueToCollision(
        const WorldState& current_world_state) const
{
    (void)current_world_state;

    // If we have not yet collected enough data, then assume we are not stuck
    if (grippers_pose_history_.size() < max_grippers_pose_history_length_)
    {
        return false;
    }

    assert(grippers_pose_history_.size() == max_grippers_pose_history_length_);

    // Calculate distances from the first gripper config to the last
    const AllGrippersSinglePose& start_config = grippers_pose_history_[0];
    std::vector<double> distances(max_grippers_pose_history_length_ - 1);
    for (size_t time_idx = 1; time_idx < max_grippers_pose_history_length_; ++time_idx)
    {
        const AllGrippersSinglePoseDelta grippers_delta = CalculateGrippersPoseDelta(start_config, grippers_pose_history_[time_idx]);
        const double distance = MultipleGrippersVelocity6dNorm(grippers_delta);
        distances[time_idx - 1] = distance;
    }

    // Determine if there is a general positive slope on the distances
    // - we should be moving away from the start config if we are not stuck
    std::cerr //<< "Gripper distances for collision stuck detection:\n"
              << PrettyPrint::PrettyPrint(distances, true, " ") << std::endl;

    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global gripper planner functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AllGrippersPoseTrajectory Planner::convertRRTResultIntoGripperTrajectory(
        const AllGrippersSinglePose& starting_poses,
        const std::vector<RRTConfig, RRTAllocator>& rrt_result) const
{
    assert(starting_poses.size() == 2);

    AllGrippersPoseTrajectory traj;
    traj.reserve(rrt_result.size());

    for (size_t ind = 0; ind < rrt_result.size(); ++ind)
    {
        AllGrippersSinglePose grippers_poses(starting_poses);
        grippers_poses[0].translation() = rrt_result[ind].getGrippers().first;
        grippers_poses[1].translation() = rrt_result[ind].getGrippers().second;

        traj.push_back(grippers_poses);
    }

    const auto distance_fn = [] (const AllGrippersSinglePose& a, const AllGrippersSinglePose& b)
    {
        const double gripper0_dist_sq = (a[0].translation() - b[0].translation()).squaredNorm();
        const double gripper1_dist_sq = (a[1].translation() - b[1].translation()).squaredNorm();

        return std::sqrt(gripper0_dist_sq + gripper1_dist_sq);
    };

    const auto interpolation_fn = [] (const AllGrippersSinglePose& a, const AllGrippersSinglePose& b, const double ratio)
    {
        AllGrippersSinglePose result = a;
        result[0].translation() = Interpolate(a[0].translation(), b[0].translation(), ratio);
        result[1].translation() = Interpolate(a[1].translation(), b[1].translation(), ratio);
        return result;
    };

    const auto resampled_traj = shortcut_smoothing::ResamplePath<AllGrippersSinglePose>(traj, robot_.max_gripper_velocity_ * robot_.dt_, distance_fn, interpolation_fn);
    return resampled_traj;
}

AllGrippersSinglePose Planner::getGripperTargets(const WorldState& world_state)
{
    const auto correspondences = dijkstras_task_->getCoverPointCorrespondences(world_state);
    const auto& cover_point_indices_= correspondences.uncovered_target_points_idxs_;

    VectorVector3d cluster_targets;
    cluster_targets.reserve(cover_point_indices_.size());
    for (size_t idx = 0; idx < cover_point_indices_.size(); ++idx)
    {
        const ssize_t cover_idx = cover_point_indices_[idx];
        cluster_targets.push_back(dijkstras_task_->cover_points_.col(cover_idx));
    }

    vis_.visualizePoints("points_to_be_clustered", cluster_targets, Visualizer::Blue(), 1);

    const Matrix3Xd cluster_targets_as_matrix = VectorEigenVector3dToEigenMatrix3Xd(cluster_targets);
    const MatrixXd distance_matrix = CalculateSquaredDistanceMatrix(cluster_targets_as_matrix);

    // Get the 2 most disparate points to initialize the clustering
    ssize_t row, col;
    distance_matrix.maxCoeff(&row, &col);
    assert(row != col);
    const VectorVector3d starting_cluster_centers = {cluster_targets[row], cluster_targets[col]};

    // Cluster the target points using K-means, then extract the cluster centers
    const std::function<double(const Vector3d&, const Vector3d&)> distance_fn = [] (const Vector3d& v1, const Vector3d& v2)
    {
        return (v1 - v2).norm();
    };
    const std::function<Vector3d(const VectorVector3d&)> average_fn = [] (const VectorVector3d& data)
    {
        return AverageEigenVector3d(data);
    };
    const auto cluster_results = simple_kmeans_clustering::SimpleKMeansClustering::Cluster(cluster_targets, distance_fn, average_fn, starting_cluster_centers);
    VectorVector3d cluster_centers = cluster_results.second;

    vis_.visualizeCubes("cluster_centers_pre_project", cluster_centers, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Red(), 1);

    // TODO: check if this can leave the grippers too close to collision for the RRT to actually reach the target (due to size of grippers)
    // Project the cluster centers to be out of collision
    cluster_centers[0] = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(cluster_centers[0], GetGripperRadius());
    cluster_centers[1] = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(cluster_centers[1], GetGripperRadius());
    assert(dijkstras_task_->environment_sdf_.Get3d(cluster_centers[0]) > GetGripperRadius());
    assert(dijkstras_task_->environment_sdf_.Get3d(cluster_centers[1]) > GetGripperRadius());
    vis_.visualizeCubes("cluster_centers_post_project", cluster_centers, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Green(), 1);


    // Assign the gripper poses based on which one is nearest to each cluster center
    AllGrippersSinglePose target_gripper_poses = world_state.all_grippers_single_pose_;
    {
        const double dist_sq_gripper0_to_cluster0 = (target_gripper_poses[0].translation() - cluster_centers[0]).squaredNorm();
        const double dist_sq_gripper0_to_cluster1 = (target_gripper_poses[0].translation() - cluster_centers[1]).squaredNorm();
        const double dist_sq_gripper1_to_cluster0 = (target_gripper_poses[1].translation() - cluster_centers[0]).squaredNorm();
        const double dist_sq_gripper1_to_cluster1 = (target_gripper_poses[1].translation() - cluster_centers[1]).squaredNorm();

        const bool gripper0_is_closest_to_cluster_0 = dist_sq_gripper0_to_cluster0 < dist_sq_gripper1_to_cluster0;
        const bool gripper0_is_closest_to_cluster_1 = dist_sq_gripper0_to_cluster1 < dist_sq_gripper1_to_cluster1;

        // If gripper0 is closest to both, set the target for gripper0 to be whichever one it is closer to
        if (gripper0_is_closest_to_cluster_0 && gripper0_is_closest_to_cluster_1)
        {
            if (dist_sq_gripper0_to_cluster0 < dist_sq_gripper0_to_cluster1)
            {
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
            else
            {
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
        }
        // If gripper1 is closest to both, set the target for gripper1 to be whichever one it is closer to
        else if (!gripper0_is_closest_to_cluster_0 && !gripper0_is_closest_to_cluster_1)
        {
            if (dist_sq_gripper1_to_cluster0 < dist_sq_gripper1_to_cluster1)
            {
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
            else
            {
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
        }
        // Otherwise, each one is closer to just one cluster center, so match them accordingly
        else
        {
            if (gripper0_is_closest_to_cluster_0)
            {
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
            else
            {
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
        }
    }

    return target_gripper_poses;
}

void Planner::planGlobalGripperTrajectory(const WorldState& world_state)
{
    // Deserialization
//    {
//        try
//        {
//            ROS_INFO_NAMED("rrt_planner_results", "Checking if RRT solution already exists");
//            const std::string rrt_file_path = GetDijkstrasStorageLocation(nh_) + ".rrt";
//            stopwatch(RESET);
//            std::ifstream prev_rrt_result(rrt_file_path, std::ios::binary | std::ios::in | std::ios::ate);
//            if (!prev_rrt_result.is_open())
//            {
//                throw_arc_exception(std::runtime_error, "Couldn't open file");
//            }

//            ROS_INFO_NAMED("rrt_planner_results", "Reading contents of file");
//            std::streamsize size = prev_rrt_result.tellg();
//            prev_rrt_result.seekg(0, std::ios::beg);
//            std::vector<uint8_t> file_buffer((size_t)size);
//            if (!(prev_rrt_result.read(reinterpret_cast<char*>(file_buffer.data()), size)))
//            {
//                throw_arc_exception(std::runtime_error, "Unable to read entire contents of file");
//            }
//            const std::vector<uint8_t> decompressed_rrt_results = ZlibHelpers::DecompressBytes(file_buffer);

//            const auto deserialized_results = DeserializeAllGrippersPoseTrajectory(decompressed_rrt_results, 0);
//            const auto deserialized_bytes_read = deserialized_results.second;
//            assert(deserialized_bytes_read == decompressed_rrt_results.size());

//            global_plan_current_timestep_ = 0;
//            executing_global_gripper_trajectory_ = true;
//            global_plan_gripper_trajectory_ = deserialized_results.first;

//            ROS_INFO_STREAM_NAMED("rrt_planner_results", "Read RRT solutions in " << stopwatch(READ) << " seconds");
//            return;
//        }
//        catch (...)
//        {
//            ROS_ERROR_NAMED("rrt_planner_results", "Loading RRT results from file failed");
//        }
//    }

    // Planning
    {
        RRTConfig start_config(
                    std::pair<Vector3d, Vector3d>(
                                world_state.all_grippers_single_pose_[0].translation(),
                                world_state.all_grippers_single_pose_[1].translation()),
                    *virtual_rubber_band_between_grippers_,
                    true);

        // Note that the rubber band part of the target is ignored at the present time
        const AllGrippersSinglePose target_grippers_pose = getGripperTargets(world_state);
        const RRTGrippersRepresentation rrt_grippers_goal(
            target_grippers_pose[0].translation(),
            target_grippers_pose[1].translation());

        const std::chrono::duration<double> time_limit(GetRRTTimeout(ph_));
        const auto rrt_results = rrt_helper_->rrtPlan(start_config, rrt_grippers_goal, time_limit);

        rrt_helper_->visualize(rrt_results);

        global_plan_current_timestep_ = 0;
        executing_global_gripper_trajectory_ = true;
        global_plan_gripper_trajectory_ = convertRRTResultIntoGripperTrajectory(world_state.all_grippers_single_pose_, rrt_results);
    }

    // Serialization
    {
        try
        {
            // First serialize
            std::vector<uint8_t> buffer;
            const uint64_t bytes_used = SerializeAllGrippersPoseTrajectory(global_plan_gripper_trajectory_, buffer);
            const auto deserialized_results = DeserializeAllGrippersPoseTrajectory(buffer, 0);
            const auto& deserialized_traj = deserialized_results.first;
            const auto deserialized_bytes_read = deserialized_results.second;

            // Verify no mistakes made in serialization
            assert(deserialized_bytes_read == bytes_used);
            assert(global_plan_gripper_trajectory_.size() == deserialized_traj.size());
            for (size_t time_idx = 0; time_idx < deserialized_traj.size(); ++time_idx)
            {
                const auto& planned_poses = global_plan_gripper_trajectory_[time_idx];
                const auto& deserialized_poses = deserialized_traj[time_idx];

                assert(planned_poses.size() == deserialized_poses.size());
                for (size_t gripper_idx = 0; gripper_idx < deserialized_poses.size(); ++gripper_idx)
                {
                    assert(planned_poses[gripper_idx].matrix() == deserialized_poses[gripper_idx].matrix());
                }
            }

            // Compress and save to file
            ROS_INFO_NAMED("rrt_planner_results", "Compressing for storage");
            const std::vector<uint8_t> compressed_serialized_data = ZlibHelpers::CompressBytes(buffer);
            ROS_INFO_NAMED("rrt_planner_results", "Saving RRT results to file");
            const std::string rrt_file_path = GetDijkstrasStorageLocation(nh_) + ".rrt";
            std::ofstream output_file(rrt_file_path, std::ios::out | std::ios::binary);
            uint64_t serialized_size = compressed_serialized_data.size();
            output_file.write(reinterpret_cast<const char*>(compressed_serialized_data.data()), (std::streamsize)serialized_size);
            output_file.close();
        }
        catch (...)
        {
            ROS_ERROR_NAMED("coverage_task", "Saving RRT results to file failed");
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Model utility functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    const double starting_error = task_specification_->calculateError(starting_world_state);
    const double true_error_reduction = starting_error - task_specification_->calculateError(world_feedback);
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
    model_utility_bandit_.updateArms(process_noise_scaling_factor * VectorXd::Ones(num_models_), model_used, true_error_reduction, observation_noise_scaling_factor * 1.0);
#endif
#ifdef KFMANDB_BANDIT
    (void)task_desired_motion;

    const MatrixXd process_noise = calculateProcessNoise(suggested_commands);
    MatrixXd observation_matrix = RowVectorXd::Zero(num_models_);
    observation_matrix(0, model_used) = 1.0;
    const VectorXd observed_reward = VectorXd::Ones(1) * true_error_reduction;
    const MatrixXd observation_noise = MatrixXd::Ones(1, 1);

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
MatrixXd Planner::calculateProcessNoise(
        const std::vector<std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>>& suggested_commands) const
{
    std::vector<double> grippers_velocity_norms((size_t)num_models_);

    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        grippers_velocity_norms[model_ind] = MultipleGrippersVelocity6dNorm(suggested_commands[model_ind].first);
    }

    MatrixXd process_noise = MatrixXd::Identity(num_models_, num_models_);
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

    return correlation_strength_factor_ * process_noise + (1.0 - correlation_strength_factor_) * MatrixXd::Identity(num_models_, num_models_);
}
