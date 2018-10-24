#include "smmap/task_framework.h"

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
#include <arc_utilities/path_utils.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/zlib_helpers.hpp>

#include "smmap/diminishing_rigidity_model.h"
#include "smmap/adaptive_jacobian_model.h"
#include "smmap/least_squares_jacobian_model.h"
#include "smmap/constraint_jacobian_model.h"

#include "smmap/least_squares_controller_with_object_avoidance.h"
#include "smmap/least_squares_stretching_constraint_controller.h"
#include "smmap/stretching_avoidance_controller.h"
#include "smmap/straight_line_controller.h"

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;
using namespace Eigen;
using namespace EigenHelpers;
using namespace EigenHelpersConversions;
using ColorBuilder = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>;

const static std_msgs::ColorRGBA PREDICTION_GRIPPER_COLOR = ColorBuilder::MakeFromFloatColors(0.0f, 0.0f, 0.6f, 1.0f);
const static std_msgs::ColorRGBA PREDICTION_RUBBER_BAND_SAFE_COLOR = ColorBuilder::MakeFromFloatColors(0.0f, 0.0f, 0.0f, 1.0f);
const static std_msgs::ColorRGBA PREDICTION_RUBBER_BAND_VIOLATION_COLOR = ColorBuilder::MakeFromFloatColors(0.0f, 1.0f, 1.0f, 1.0f);

#pragma message "Magic number - reward scaling factor starting value"
#define REWARD_STANDARD_DEV_SCALING_FACTOR_START (1.0)

//#define ENABLE_SEND_NEXT_COMMAND_LOAD_SAVE 1
#define ENABLE_SEND_NEXT_COMMAND_LOAD_SAVE 0

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Internal helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief GetShortestPathBetweenGrippersThroughObject
 * @param grippers_data
 * @param object
 * @param neighour_fn
 * @return The index of the nodes between the grippers, following the shortest path through the object
 */
static std::vector<ssize_t> getShortestPathBetweenGrippersThroughObject(
        const std::vector<GripperData>& grippers_data,
        const ObjectPointSet& object,
        const std::function<std::vector<ssize_t>(const ssize_t& node)> neighbour_fn)
{
    assert(grippers_data.size() == 2);
    assert(grippers_data[0].node_indices_.size() > 0);
    assert(grippers_data[1].node_indices_.size() > 0);

    const auto start = grippers_data[0].node_indices_[0];
    const auto goal = grippers_data[1].node_indices_[0];
    const auto distance_fn = [&] (const ssize_t& first_node, const ssize_t& second_node)
    {
        return (object.col(first_node) - object.col(second_node)).norm();
    };
    const auto heuristic_fn = [&] (const ssize_t& node)
    {
        return distance_fn(node, goal);
    };
    const auto goal_reached_fn = [&] (const ssize_t& test_node)
    {
        return test_node == goal;
    };
    const auto astar_results = simple_astar_planner::SimpleAStarPlanner<ssize_t>::Plan(
                start, neighbour_fn, distance_fn, heuristic_fn, goal_reached_fn);

    const auto plan = astar_results.first;
    assert(plan.size() > 0);
    return plan;
}

static EigenHelpers::VectorVector3d getPathBetweenGrippersThroughObject(
        const WorldState& world_state,
        const std::vector<ssize_t>& object_node_idxs_between_grippers)
{
    assert(world_state.all_grippers_single_pose_.size() == 2);
    EigenHelpers::VectorVector3d nodes;
    nodes.reserve(object_node_idxs_between_grippers.size() + 2);

    nodes.push_back(world_state.all_grippers_single_pose_[0].translation());
    for (size_t path_idx = 0; path_idx < object_node_idxs_between_grippers.size(); ++path_idx)
    {
        const ssize_t node_idx = object_node_idxs_between_grippers[path_idx];
        nodes.push_back(world_state.object_configuration_.col(node_idx));
    }
    nodes.push_back(world_state.all_grippers_single_pose_[1].translation());

    return nodes;
}

template <typename T, typename Alloc = std::allocator<T>>
static size_t sizeOfLargestVector(const std::vector<T, Alloc>& vectors)
{
    size_t largest_vector = 0;

    for (size_t idx = 0; idx < vectors.size(); ++idx)
    {
        largest_vector = std::max(largest_vector, vectors[idx].size());
    }

    return largest_vector;
}

std::vector<uint32_t> numberOfPointsInEachCluster(
        const std::vector<uint32_t>& cluster_labels,
        const uint32_t num_clusters,
        const std::vector<long>& grapsed_points,
        const DijkstrasCoverageTask::Correspondences& correspondences)
{
    std::vector<uint32_t> counts(num_clusters, 0);
    const auto& uncovered_target_point_idxs = correspondences.uncovered_target_points_idxs_;

    for (size_t grasped_point_idx = 0; grasped_point_idx < grapsed_points.size(); ++grasped_point_idx)
    {
        const long deform_idx = grapsed_points[grasped_point_idx];

        const std::vector<ssize_t>& correspondences_for_current_deform_idx            = correspondences.correspondences_[deform_idx];
        const std::vector<bool>&    correspondences_is_covered_for_current_deform_idx = correspondences.correspondences_is_covered_[deform_idx];

        for (size_t correspondence_idx = 0; correspondence_idx < correspondences_for_current_deform_idx.size(); ++correspondence_idx)
        {
            const ssize_t cover_idx = correspondences_for_current_deform_idx[correspondence_idx];
            const bool is_covered   = correspondences_is_covered_for_current_deform_idx[correspondence_idx];

            // If the current correspondece is not covered, lookup its position in cluster_labels
            if (!is_covered)
            {
                const auto found_itr = std::find(uncovered_target_point_idxs.begin(), uncovered_target_point_idxs.end(), cover_idx);
                assert(found_itr != uncovered_target_point_idxs.end()); // The point is not covered, so it should exist in the vector
                const ssize_t found_idx = std::distance(uncovered_target_point_idxs.begin(), found_itr);
                assert(found_idx >= 0);
                assert(found_idx < (ssize_t)cluster_labels.size());
                const uint32_t cluster_label = cluster_labels[found_idx];
                counts[cluster_label]++;
            }
        }
    }

    return counts;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor and model list builder
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::Planner
 * @param robot
 * @param vis
 * @param task_specification
 */
TaskFramework::TaskFramework(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const RobotInterface::Ptr& robot,
        Visualizer::Ptr vis,
        const std::shared_ptr<TaskSpecification>& task_specification)
    // Robot and task parameters
    : nh_(nh)
    , ph_(ph)
    , seed_(GetPlannerSeed(ph_))
    , generator_(std::make_shared<std::mt19937_64>(seed_))
    , robot_(robot)
    , task_specification_(task_specification)
    , dijkstras_task_(std::dynamic_pointer_cast<DijkstrasCoverageTask>(task_specification_)) // If possible, this will be done, if not, it will be NULL (nullptr?)
    // Multi-model and regret based model selection parameters
    , collect_results_for_all_models_(GetCollectResultsForAllModels(ph_))
    , reward_std_dev_scale_factor_(REWARD_STANDARD_DEV_SCALING_FACTOR_START)
    , process_noise_factor_(GetProcessNoiseFactor(ph_))
    , observation_noise_factor_(GetObservationNoiseFactor(ph_))
    , correlation_strength_factor_(GetCorrelationStrengthFactor(ph_))
    // 'Stuck' detection and RRT prameters
    , enable_stuck_detection_(GetEnableStuckDetection(ph_))
    , max_lookahead_steps_(GetNumLookaheadSteps(ph_))
    , max_grippers_pose_history_length_(GetMaxGrippersPoseHistoryLength(ph_))
    , executing_global_trajectory_(false)
    , global_plan_next_timestep_(-1)
    , rrt_helper_(nullptr)
    // Used to generate some log data by some controllers
    , object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh_)))
    , initial_grippers_distance_(robot_->getGrippersInitialDistance())
    // Logging and visualization parameters
    , bandits_logging_enabled_(GetBanditsLoggingEnabled(ph_))
    , controller_logging_enabled_(GetControllerLoggingEnabled(ph_))
    , vis_(vis)
    , visualize_desired_motion_(!GetDisableAllVisualizations(ph_) && GetVisualizeObjectDesiredMotion(ph_))
    , visualize_gripper_motion_(!GetDisableAllVisualizations(ph_) && GetVisualizerGripperMotion(ph_))
    , visualize_predicted_motion_(!GetDisableAllVisualizations(ph_) && GetVisualizeObjectPredictedMotion(ph_))
{
    ROS_INFO_STREAM_NAMED("task_framework", "Using seed " << std::hex << seed_ );
    initializeBanditsLogging();
    initializeControllerLogging();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The one function that gets invoked externally
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TaskFramework::execute()
{
    WorldState world_feedback = robot_->start();
    const double start_time = world_feedback.sim_time_;
    initializeModelAndControllerSet(world_feedback);

    if (enable_stuck_detection_)
    {
        assert(dijkstras_task_ != nullptr);

        // TODO: Assumptions in the implementation that need be addressed later
        assert(robot_->getGrippersData().size() == 2);
        assert(model_list_.size() == 1);

        // Extract the maximum distance between the grippers
        // This assumes that the starting position of the grippers is at the maximum "unstretched" distance
        const auto& grippers_starting_poses = world_feedback.all_grippers_single_pose_;
        const double max_calced_band_distance =
                (grippers_starting_poses[0].translation() - grippers_starting_poses[1].translation()).norm()
                * dijkstras_task_->maxStretchFactor();
        ROS_ERROR_STREAM_COND_NAMED(!CloseEnough(max_calced_band_distance, dijkstras_task_->maxBandLength(), 1e-3),
                                    "task_framework",
                                    "Calc'd max band distance is: " << max_calced_band_distance <<
                                    " but the ros param saved distance is " << dijkstras_task_->maxBandLength() <<
                                    ". Double check the stored value in the roslaunch file.");

        // Find the shortest path through the object, between the grippers, while follow nodes of the object.
        // Used to determine the starting position of the rubber band at each timestep
        const auto neighbour_fn = [&] (const ssize_t& node)
        {
            return dijkstras_task_->getNodeNeighbours(node);
        };
        path_between_grippers_through_object_ = getShortestPathBetweenGrippersThroughObject(
                    robot_->getGrippersData(), GetObjectInitialConfiguration(nh_), neighbour_fn);

        // Create the initial rubber band
        const auto starting_band_points = getPathBetweenGrippersThroughObject(
                    world_feedback, path_between_grippers_through_object_);
        rubber_band_between_grippers_ = std::make_shared<RubberBand>(
                    starting_band_points,
                    dijkstras_task_->maxBandLength(),
                    dijkstras_task_,
                    vis_,
                    generator_);

#if ENABLE_SEND_NEXT_COMMAND_LOAD_SAVE
        if (useStoredWorldState())
        {
            const auto world_state_and_band = loadStoredWorldState();
            world_feedback = world_state_and_band.first;
            vis_->visualizeCloth("controller_input_deformable_object", world_feedback.object_configuration_, Visualizer::Green(0.5), 1);

            const auto starting_band_points = getPathBetweenGrippersThroughObject(
                        world_feedback, path_between_grippers_through_object_);
            rubber_band_between_grippers_ = std::make_shared<RubberBand>(
                        starting_band_points,
                        dijkstras_task_->maxBandLength(),
                        dijkstras_task_,
                        vis_,
                        generator_);
        }
#endif

        // Algorithm parameters
        const auto use_cbirrt_style_projection = GetUseCBiRRTStyleProjection(ph_);
        const auto forward_tree_extend_iterations = GetRRTForwardTreeExtendIterations(ph_);
        const auto backward_tree_extend_iterations = GetRRTBackwardTreeExtendIterations(ph_);
        const auto kd_tree_grow_threshold = GetKdTreeGrowThreshold(ph_);
        const auto use_brute_force_nn = GetUseBruteForceNN(ph_);
        const auto goal_bias = GetRRTGoalBias(ph_);
        const auto best_near_radius = GetRRTBestNearRadius(ph_);

        // Smoothing parameters
        const auto max_shortcut_index_distance = GetRRTMaxShortcutIndexDistance(ph_);
        const auto max_smoothing_iterations = GetRRTMaxSmoothingIterations(ph_);
        const auto max_failed_smoothing_iterations = GetRRTMaxFailedSmoothingIterations(ph_);

        // Task defined parameters
        const auto task_aligned_frame = robot_->getWorldToTaskFrameTf();
        const auto task_frame_lower_limits = Vector3d(
                    GetRRTPlanningXMinBulletFrame(ph_),
                    GetRRTPlanningYMinBulletFrame(ph_),
                    GetRRTPlanningZMinBulletFrame(ph_));
        const auto task_frame_upper_limits = Vector3d(
                    GetRRTPlanningXMaxBulletFrame(ph_),
                    GetRRTPlanningYMaxBulletFrame(ph_),
                    GetRRTPlanningZMaxBulletFrame(ph_));
        const auto max_gripper_step_size = dijkstras_task_->work_space_grid_.minStepDimension();
        const auto max_robot_step_size = GetRRTMaxRobotDOFStepSize(ph_);
        const auto min_robot_step_size = GetRRTMinRobotDOFStepSize(ph_);
        const auto max_gripper_rotation = GetRRTMaxGripperRotation(ph_); // only matters for real robot
        const auto goal_reached_radius = dijkstras_task_->work_space_grid_.minStepDimension();
//        const auto homotopy_distance_penalty = GetRRTHomotopyDistancePenalty();
        const auto min_gripper_distance_to_obstacles = GetRRTMinGripperDistanceToObstacles(ph_); // only matters for simulation
        const auto band_distance2_scaling_factor = GetRRTBandDistance2ScalingFactor(ph_);
        const auto band_max_points = GetRRTBandMaxPoints(ph_);

        // Visualization
        const auto enable_rrt_visualizations = GetVisualizeRRT(ph_);

        // Pass in all the config values that the RRT needs; for example goal bias, step size, etc.
        rrt_helper_ = std::make_shared<RRTHelper>(
                    // Robot/environment related parameters
                    nh_,
                    ph_,
                    robot_,
                    world_feedback.robot_configuration_valid_,
                    dijkstras_task_->sdf_,
                    dijkstras_task_->work_space_grid_,
                    generator_,
                    // Planning algorithm parameters
                    use_cbirrt_style_projection,
                    forward_tree_extend_iterations,
                    backward_tree_extend_iterations,
                    kd_tree_grow_threshold,
                    use_brute_force_nn,
                    goal_bias,
                    best_near_radius,
                    // Smoothing parameters
                    max_shortcut_index_distance,
                    max_smoothing_iterations,
                    max_failed_smoothing_iterations,
                    // Task defined parameters
                    task_aligned_frame,
                    task_frame_lower_limits,
                    task_frame_upper_limits,
                    max_gripper_step_size,
                    max_robot_step_size,
                    min_robot_step_size,
                    max_gripper_rotation,
                    goal_reached_radius,
                    min_gripper_distance_to_obstacles,
                    // Dual stage NN checking variables
                    band_distance2_scaling_factor,
                    band_max_points,
                    // Visualization
                    vis_,
                    enable_rrt_visualizations);
    }

    while (robot_->ok())
    {
        // TODO: Can I remove this extraneous world_state object? All it does is cache the value of world_feedback for
        // a single function call.
        const WorldState world_state = world_feedback;
        world_feedback = sendNextCommand(world_state);

         if (unlikely(world_feedback.sim_time_ - start_time >= task_specification_->maxTime()
                     || task_specification_->taskDone(world_feedback)))
        {
            ROS_INFO_NAMED("task_framework", "------------------------------- End of Task -------------------------------------------");
            const double current_error = task_specification_->calculateError(world_feedback);
            ROS_INFO_STREAM_NAMED("task_framework", "   Planner/Task sim time " << world_feedback.sim_time_ << "\t Error: " << current_error);


            vis_->purgeMarkerList();
            visualization_msgs::Marker marker;
            marker.action = visualization_msgs::Marker::DELETEALL;
            vis_->publish(marker);
            vis_->forcePublishNow();
            vis_->purgeMarkerList();

            if (world_feedback.sim_time_ - start_time >= task_specification_->maxTime())
            {
                ROS_INFO("Terminating task as time has run out");
            }
            if (task_specification_->taskDone(world_feedback))
            {
                ROS_INFO("Terminating task as the task has been completed");
            }
            robot_->shutdown();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Gripper movement functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::sendNextCommand
 * @param current_world_state
 * @return
 */
WorldState TaskFramework::sendNextCommand(
        WorldState world_state)
{
#if ENABLE_SEND_NEXT_COMMAND_LOAD_SAVE
    if (useStoredWorldState())
    {
        const auto world_state_and_band = loadStoredWorldState();
        world_state = world_state_and_band.first;
        rubber_band_between_grippers_ = world_state_and_band.second;
        vis_->visualizeCloth("controller_input_deformable_object", world_state.object_configuration_, Visualizer::Green(0.5), 1);
    }
    else
    {
        storeWorldState(world_state, rubber_band_between_grippers_);
    }
#endif

    ROS_INFO_NAMED("task_framework", "---------------------------- Start of Loop -----------------------------------------");
    const double current_error = task_specification_->calculateError(world_state);
    ROS_INFO_STREAM_NAMED("task_framework", "Planner/Task sim time " << world_state.sim_time_ << "\t Error: " << current_error);

    if (enable_stuck_detection_)
    {
        #if ENABLE_SEND_NEXT_COMMAND_LOAD_SAVE
        // Update the band with the new position of the deformable object - added here to help with debugging and visualization
        if (useStoredWorldState())
        {
            vis_->purgeMarkerList();
            vis_->visualizeCloth("controller_input_deformable_object", world_state.object_configuration_, Visualizer::Green(0.5), 1);
//            const auto band_points = getPathBetweenGrippersThroughObject(world_state, path_between_grippers_through_object_);
//            rubber_band_between_grippers_->setPointsAndSmooth(band_points);
        }
        #endif



        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // First, check if we need to (re)plan
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        bool planning_needed = false;

        // Check if the global plan has 'hooked' the deformable object on something
        if (executing_global_trajectory_)
        {
            Stopwatch stopwatch;
            const bool global_plan_will_overstretch = predictStuckForGlobalPlannerResults();
            ROS_INFO_STREAM_NAMED("task_framework", "Determined if global planner needed in          " << stopwatch(READ) << " seconds");

            if (global_plan_will_overstretch)
            {
                planning_needed = true;

                ROS_WARN_NAMED("task_framework", "Invoking global planner as the current plan will overstretch the deformable object");
                ROS_INFO_NAMED("task_framework", "----------------------------------------------------------------------------");
            }
        }
        // Check if the local controller will be stuck
        else
        {
            Stopwatch stopwatch;
            arc_helpers::DoNotOptimize(world_state);
            const bool global_planner_needed_due_to_overstretch = globalPlannerNeededDueToOverstretch(world_state);
            const bool global_planner_needed_due_to_lack_of_progress = globalPlannerNeededDueToLackOfProgress();
            arc_helpers::DoNotOptimize(global_planner_needed_due_to_lack_of_progress);
            ROS_INFO_STREAM_NAMED("task_framework", "Determined if global planner needed in " << stopwatch(READ) << " seconds");

            if (global_planner_needed_due_to_overstretch || global_planner_needed_due_to_lack_of_progress)
            {
                planning_needed = true;

                ROS_WARN_COND_NAMED(global_planner_needed_due_to_overstretch, "task_framework", "Invoking global planner due to overstretch");
                ROS_WARN_COND_NAMED(global_planner_needed_due_to_lack_of_progress, "task_framework", "Invoking global planner due to collision");
                ROS_INFO_NAMED("task_framework", "----------------------------------------------------------------------------");

                if (!GetDisableAllVisualizations(ph_))
                {
                    vis_->forcePublishNow(2.0);
                }
            }
        }

        // If we need to (re)plan due to the local controller getting stuck, or the gobal plan failing, then do so
        if (planning_needed)
        {
//            std::cout << "Waiting for keystroke before planning" << std::endl;
//            std::getchar();

            vis_->purgeMarkerList();
            visualization_msgs::Marker marker;
            marker.action = visualization_msgs::Marker::DELETEALL;
            marker.header.frame_id = "world_origin";
            marker.header.stamp = ros::Time::now();
            vis_->publish(marker);
            vis_->forcePublishNow(0.5);
            vis_->purgeMarkerList();

            planGlobalGripperTrajectory(world_state);

//            std::cout << "Waiting for keystroke after planning" << std::endl;
//            std::getchar();
        }

        // Execute a single step in the global plan, or use the local controller if we have no plan to follow
        WorldState world_feedback;
        if (executing_global_trajectory_)
        {
            world_feedback = sendNextCommandUsingGlobalPlannerResults(world_state);

            // Band is updated internally in sendNextCommandUsingGlobalPlannerResults
        }
        else
        {
            world_feedback = sendNextCommandUsingLocalController(world_state);

            // Update the band with the new position of the deformable object
            const auto band_points = getPathBetweenGrippersThroughObject(world_feedback, path_between_grippers_through_object_);
            rubber_band_between_grippers_->setPointsAndSmooth(band_points);
        }

        // Keep the last N grippers positions recorded to detect if the grippers are stuck
        grippers_pose_history_.push_back(world_feedback.all_grippers_single_pose_);
        error_history_.push_back(dijkstras_task_->calculateError(world_feedback));
        assert(grippers_pose_history_.size() == error_history_.size());
        if (grippers_pose_history_.size() > max_grippers_pose_history_length_)
        {
            grippers_pose_history_.erase(grippers_pose_history_.begin());
            error_history_.erase(error_history_.begin());
        }

        return world_feedback;
    }
    else
    {
        ROS_WARN_ONCE_NAMED("task_framework", "Unable to do future constraint violation detection");
        return sendNextCommandUsingLocalController(world_state);
    }
}

/**
 * @brief Planner::sendNextCommandUsingLocalController
 * @param world_state
 * @return
 */
WorldState TaskFramework::sendNextCommandUsingLocalController(
        const WorldState& current_world_state)
{
    Stopwatch stopwatch;
    Stopwatch function_wide_stopwatch;

    // Temporaries needed here bercause model_input_data takes things by reference
    const DesiredDirection desired_object_manipulation_direction = task_specification_->calculateDesiredDirection(current_world_state);
    // It is assumed that the robot's internal state matches that that is passed to us, so we do not need to set active dof values
    const MatrixXd robot_dof_to_grippers_poses_jacobian = robot_->getGrippersJacobian();
    // Build the constraints for the gippers and other points of interest on the robot - includes the grippers
    const std::vector<std::pair<CollisionData, Matrix3Xd>> poi_collision_data_ = robot_->getPointsOfInterestCollisionData();

    const DeformableController::InputData model_input_data(
                current_world_state,
                desired_object_manipulation_direction,
                robot_,
                robot_dof_to_grippers_poses_jacobian,
                current_world_state.robot_configuration_valid_,
                poi_collision_data_,
                robot_->max_gripper_velocity_norm_ * robot_->dt_,
                robot_->max_dof_velocity_norm_ * robot_->dt_);

    if (visualize_desired_motion_)
    {
        visualizeDesiredMotion(current_world_state, model_input_data.desired_object_motion_.error_correction_);
//        std::this_thread::sleep_for(std::chrono::duration<double>(2.0));
    }

    // Pick an arm to use
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(*generator_);

    const bool get_action_for_all_models = model_utility_bandit_.generateAllModelActions();
    ROS_INFO_STREAM_COND_NAMED(num_models_ > 1, "task_framework", "Using model index " << model_to_use);

    // Querry each model for it's best gripper delta
    ROS_INFO_STREAM_NAMED("task_framework", "Generating model suggestions");
    stopwatch(RESET);
    std::vector<DeformableController::OutputData> suggested_robot_commands(num_models_);
    std::vector<double> controller_computation_time(num_models_, 0.0);
    #pragma omp parallel for
    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        if (collect_results_for_all_models_ || get_action_for_all_models || (ssize_t)model_ind == model_to_use)
        {
            Stopwatch individual_controller_stopwatch;

            suggested_robot_commands[model_ind] =
                controller_list_[model_ind]->getGripperMotion(
                        model_input_data);

            controller_computation_time[model_ind] = individual_controller_stopwatch(READ);

            // Measure the time it took to pick a model
            ROS_DEBUG_STREAM_NAMED("task_framework", model_ind << "th Controller get suggested motion in          " << controller_computation_time[model_ind] << " seconds");
        }
    }
    // Measure the time it took to pick a model
    const DeformableController::OutputData& selected_command = suggested_robot_commands[(size_t)model_to_use];
    ROS_INFO_STREAM_NAMED("task_framework", "Calculated model suggestions and picked one in  " << stopwatch(READ) << " seconds");
    if (current_world_state.robot_configuration_valid_)
    {
        ROS_INFO_STREAM_NAMED("task_framework", "Robot DOF motion: " << selected_command.robot_dof_motion_.transpose());
    }
    for (size_t ind = 0; ind < selected_command.grippers_motion_.size(); ++ind)
    {
        ROS_INFO_STREAM_NAMED("task_framework", "Gripper " << ind << " motion: " << selected_command.grippers_motion_[ind].transpose());
    }
    ROS_INFO_STREAM_NAMED("task_framework", "Selected command gripper action norm:  " << MultipleGrippersVelocity6dNorm(selected_command.grippers_motion_));

    // Collect feedback data for logging purposes
    std::vector<WorldState> individual_model_results(num_models_);
    if (collect_results_for_all_models_)
    {
        stopwatch(RESET);
        // Build a feedback function to log data for each model that we are testing
        const auto test_feedback_fn = [&] (const size_t model_ind, const WorldState& resulting_world_state)
        {
            individual_model_results[model_ind] = resulting_world_state;
        };
        std::vector<AllGrippersSinglePose> poses_to_test(num_models_);
        std::vector<VectorXd> configurations_to_test(num_models_);
        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            poses_to_test[model_ind] = kinematics::applyTwist(
                        current_world_state.all_grippers_single_pose_, suggested_robot_commands[model_ind].grippers_motion_);

            configurations_to_test[model_ind] =
                    current_world_state.robot_configuration_ + suggested_robot_commands[model_ind].robot_dof_motion_;
        }
        robot_->testRobotMotion(poses_to_test, configurations_to_test, current_world_state.robot_configuration_valid_, test_feedback_fn);

        ROS_INFO_STREAM_NAMED("task_framework", "Collected data to calculate regret in " << stopwatch(READ) << " seconds");
    }

    if (visualize_gripper_motion_)
    {
        ROS_WARN_THROTTLE_NAMED(1.0, "task_framework", "Asked to visualize grippper motion but this is disabled. Manually enable the type of visualization you want.");

//        for (ssize_t model_ind = 0; model_ind < num_models_; ++model_ind)
//        {
//            ssize_t model_ind = 0;
//            visualizeGripperMotion(world_state.all_grippers_single_pose_,
//                                   suggested_robot_commands[(size_t)model_ind].grippers_motion_,
//                                   model_ind);
//        }
//        for (size_t gripper_idx = 0; gripper_idx < all_grippers_single_pose.size(); ++gripper_idx)
//        {
//            vis_->visualizeGripper("target_gripper_positions", all_grippers_single_pose[gripper_idx], Visualizer::Yellow(), (int)gripper_idx + 1);
//        }

//        const size_t num_grippers = world_feedback.all_grippers_single_pose_.size();
//        for (size_t gripper_idx = 0; gripper_idx < num_grippers; ++gripper_idx)
//        {
//            std::cerr << "Desired delta: " << selected_command.grippers_motion_[gripper_idx].head<3>().transpose() << std::endl;
//            std::cerr << "Actual delta:  " << kinematics::calculateError(world_state.all_grippers_single_pose_[gripper_idx], world_feedback.all_grippers_single_pose_[gripper_idx]).head<3>().transpose() << std::endl;
//        }
    }

    // Execute the command
    ROS_INFO_STREAM_NAMED("task_framework", "Sending command to robot");
    const auto all_grippers_single_pose = kinematics::applyTwist(current_world_state.all_grippers_single_pose_, selected_command.grippers_motion_);
    const auto robot_configuration = current_world_state.robot_configuration_ + selected_command.robot_dof_motion_;
    // Measure execution time
    stopwatch(RESET);
    arc_helpers::DoNotOptimize(all_grippers_single_pose);
    const WorldState world_feedback = robot_->commandRobotMotion(
                all_grippers_single_pose,
                robot_configuration,
                current_world_state.robot_configuration_valid_);
    arc_helpers::DoNotOptimize(world_feedback);
    const double robot_execution_time = stopwatch(READ);

    const double predicted_delta_scale_factor = 25.0;
    if (visualize_predicted_motion_)
    {
//        ROS_WARN_THROTTLE_NAMED(1.0, "task_framework", "Asked to visualize predicted motion but this is disabled. Manually enable the type of visualization you want.");

        const ObjectPointSet true_object_delta = world_feedback.object_configuration_ - current_world_state.object_configuration_;
        vis_->visualizeObjectDelta(
                    "true_object_delta",
                    current_world_state.object_configuration_,
                    current_world_state.object_configuration_ + predicted_delta_scale_factor * true_object_delta,
                    Visualizer::Green());

//        task_specification_->visualizeDeformableObject(
//                PREDICTED_DELTA_NS,
//                world_state.object_configuration_ + object_delta,
//                Visualizer::Blue());
    }

    std::vector<double> model_prediction_errors_weighted(model_list_.size(), 0.0);
    std::vector<double> model_prediction_errors_unweighted(model_list_.size(), 0.0);
    if (collect_results_for_all_models_)
    {
        ROS_INFO_NAMED("task_framework", "Calculating model predictions based on real motion taken");

        const ObjectPointSet true_object_delta = world_feedback.object_configuration_ - current_world_state.object_configuration_;
        const AllGrippersSinglePoseDelta true_robot_delta = CalculateGrippersPoseDelta(current_world_state.all_grippers_single_pose_, world_feedback.all_grippers_single_pose_);

        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            const ObjectPointSet predicted_delta = model_list_[model_ind]->getObjectDelta(current_world_state, true_robot_delta);
            const ObjectPointSet prediction_error_sq = (predicted_delta - true_object_delta).cwiseAbs2();

            const Map<const VectorXd> error_sq_as_vector(prediction_error_sq.data(), prediction_error_sq.size());
            model_prediction_errors_weighted[model_ind] = error_sq_as_vector.dot(desired_object_manipulation_direction.error_correction_.weight);
            model_prediction_errors_unweighted[model_ind] = prediction_error_sq.sum();

            if (visualize_predicted_motion_)
            {
                if (task_specification_->task_type_ != CLOTH_PLACEMAT_LINEAR_MOTION)
                {
                    ROS_WARN_NAMED("task_framework", "this visualization is only desgined for one task");
                }
                if (model_ind == 0)
                {
                    vis_->visualizeObjectDelta(
                                "constraint_model_prediction",
                                current_world_state.object_configuration_,
                                current_world_state.object_configuration_ + predicted_delta_scale_factor * predicted_delta,
                                Visualizer::Cyan());
                }

                else if (model_ind == 1)
                {
                    vis_->visualizeObjectDelta(
                                "diminishing_model_prediction",
                                current_world_state.object_configuration_,
                                current_world_state.object_configuration_ + predicted_delta_scale_factor * predicted_delta,
                                Visualizer::Red(0.3f));
                }
            }
        }
    }

    ROS_INFO_NAMED("task_framework", "Updating models");
    updateModels(current_world_state, model_input_data.desired_object_motion_.error_correction_, suggested_robot_commands, model_to_use, world_feedback);

    const double controller_time = function_wide_stopwatch(READ) - robot_execution_time;
    ROS_INFO_STREAM_NAMED("task_framework", "Total local controller time                     " << controller_time << " seconds");

    ROS_INFO_NAMED("task_framework", "Logging data");
    logBanditsData(current_world_state, world_feedback, individual_model_results, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), model_to_use);
    controllerLogData(current_world_state, world_feedback, individual_model_results, model_input_data, controller_computation_time, model_prediction_errors_weighted, model_prediction_errors_unweighted);

    return world_feedback;
}

/**
 * @brief Planner::sendNextCommandUsingGlobalGripperPlannerResults
 * @param current_world_state
 * @return
 */
WorldState TaskFramework::sendNextCommandUsingGlobalPlannerResults(
        const WorldState& current_world_state)
{
    assert(executing_global_trajectory_);
    assert(global_plan_next_timestep_ < rrt_planned_path_.size());

    // Check if we need to interpolate the command - we only advance the traj waypoint
    // pointer if we started within one step of the current target
    bool next_waypoint_targetted = true;
    Eigen::VectorXd next_dof_target(0);
    AllGrippersSinglePose next_grippers_target(0);
    if (current_world_state.robot_configuration_valid_)
    {
        next_dof_target = rrt_planned_path_[global_plan_next_timestep_].robotConfiguration();
        const auto& grippers_poses_as_pair = rrt_planned_path_[global_plan_next_timestep_].grippers();
        next_grippers_target = {grippers_poses_as_pair.first, grippers_poses_as_pair.second};

        const double max_dof_delta = robot_->max_dof_velocity_norm_ * robot_->dt_;
        const double dist_to_waypoint =
                RRTDistance::Distance(current_world_state.robot_configuration_, next_dof_target);

        if (dist_to_waypoint > max_dof_delta)
        {
            const double ratio = max_dof_delta / dist_to_waypoint;
            next_dof_target = Interpolate(current_world_state.robot_configuration_, next_dof_target, ratio);
            robot_->lockEnvironment();
            robot_->setActiveDOFValues(next_dof_target);
            next_grippers_target = robot_->getGrippersPosesFunctionPointer();
            robot_->unlockEnvironment();
            next_waypoint_targetted = false;
        }
    }
    else
    {
        const auto& grippers_poses_as_pair = rrt_planned_path_[global_plan_next_timestep_].grippers();
        next_grippers_target = {grippers_poses_as_pair.first, grippers_poses_as_pair.second};

        const double max_grippers_delta = robot_->max_gripper_velocity_norm_ * robot_->dt_;
        const double dist_to_waypoint = RRTDistance::Distance(
                    {current_world_state.all_grippers_single_pose_[0], current_world_state.all_grippers_single_pose_[1]},
                    grippers_poses_as_pair);

        if (dist_to_waypoint > max_grippers_delta)
        {
            const double ratio = max_grippers_delta / dist_to_waypoint;
            next_grippers_target[0] = Interpolate(
                        current_world_state.all_grippers_single_pose_[0],
                        next_grippers_target[0], ratio);
            next_grippers_target[1] = Interpolate(
                        current_world_state.all_grippers_single_pose_[1],
                        next_grippers_target[1], ratio);
            next_waypoint_targetted = false;
        }
    }

    const WorldState world_feedback =
            robot_->commandRobotMotion(next_grippers_target,
                                       next_dof_target,
                                       current_world_state.robot_configuration_valid_);
    // Update the band with the new position of the deformable object
    const auto band_points = getPathBetweenGrippersThroughObject(world_feedback, path_between_grippers_through_object_);
    rubber_band_between_grippers_->setPointsAndSmooth(band_points);

    // If we are directly targetting the waypoint itself (i.e., no interpolation)
    // then update the waypoint index, and record the resulting configuration in
    // the "path actually taken" index
    if (next_waypoint_targetted)
    {
        ++global_plan_next_timestep_;
        rrt_executed_path_.push_back(
                    RRTNode(
                        {world_feedback.all_grippers_single_pose_[0], world_feedback.all_grippers_single_pose_[1]},
                        world_feedback.robot_configuration_,
                        std::make_shared<RubberBand>(*rubber_band_between_grippers_)));
    }

    if (global_plan_next_timestep_ == rrt_planned_path_.size())
    {
        ROS_INFO_NAMED("task_framework", "Global plan finished, resetting grippers pose history and error history");

        executing_global_trajectory_ = false;
        grippers_pose_history_.clear();
        error_history_.clear();

        vis_->purgeMarkerList();
        visualization_msgs::Marker marker;
        marker.action = visualization_msgs::Marker::DELETEALL;
        marker.header.frame_id = "world_origin";
        marker.header.stamp = ros::Time::now();
        vis_->publish(marker);
        vis_->forcePublishNow();
        vis_->purgeMarkerList();
    }

    const std::vector<WorldState> fake_all_models_results(num_models_, world_feedback);
    logBanditsData(current_world_state, world_feedback, fake_all_models_results, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), -1);

    return world_feedback;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constraint violation detection
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TaskFramework::visualizeProjectedPaths(
        const std::vector<VectorVector3d>& projected_paths,
        const bool visualization_enabled)
{
    if (visualization_enabled)
    {
        EigenHelpers::VectorVector3d points;
        EigenHelpers::VectorVector3d lines_start_points;
        EigenHelpers::VectorVector3d lines_end_points;

        for (ssize_t node_ind = 0; node_ind < (ssize_t)projected_paths.size(); ++node_ind)
        {
            const auto& current_points = projected_paths[node_ind];
            if (current_points.size() > 1)
            {
                points.insert(points.end(), current_points.begin(), current_points.end());
                for (size_t point_idx = 1; point_idx < current_points.size(); ++point_idx)
                {
                    lines_start_points.push_back(current_points[point_idx - 1]);
                    lines_end_points.push_back(current_points[point_idx]);
                }
            }
        }
        vis_->visualizePoints(PROJECTED_POINT_PATH_NS, points, Visualizer::Magenta(), 1);
        vis_->visualizeLines(PROJECTED_POINT_PATH_LINES_NS, lines_start_points, lines_end_points, Visualizer::Magenta(), 1);
    }
}

std::pair<std::vector<VectorVector3d>, std::vector<RubberBand>> TaskFramework::projectFutureSystemState(
        const WorldState& starting_world_state,
        const bool visualization_enabled)
{
    Stopwatch stopwatch;
    Stopwatch function_wide_stopwatch;

    assert(task_specification_->is_dijkstras_type_task_ && starting_world_state.all_grippers_single_pose_.size() == 2);
    std::pair<std::vector<VectorVector3d>, std::vector<RubberBand>> projected_deformable_point_paths_and_projected_virtual_rubber_bands;

    const bool band_verbose = false;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constraint violation Version 1 - Purely cloth overstretch
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    stopwatch(RESET);
    const std::vector<VectorVector3d> projected_deformable_point_paths = dijkstras_task_->findPathFromObjectToTarget(starting_world_state, max_lookahead_steps_);

    const size_t actual_lookahead_steps = sizeOfLargestVector(projected_deformable_point_paths) - 1;
    // sizeOfLargest(...) should be at least 2, so this assert should always be true
    assert(actual_lookahead_steps <= max_lookahead_steps_);

    ROS_INFO_STREAM_NAMED("task_framework", "Calculated projected cloth paths                 - Version 1 - in " << stopwatch(READ) << " seconds");
    ROS_INFO_STREAM_NAMED("task_framework", "Max lookahead steps: " << max_lookahead_steps_ << " Actual steps: " << actual_lookahead_steps);

    visualizeProjectedPaths(projected_deformable_point_paths, visualization_enabled);
    projected_deformable_point_paths_and_projected_virtual_rubber_bands.first = projected_deformable_point_paths;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constraint violation Version 2a - Vector field forward "simulation" - rubber band
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ROS_INFO_STREAM_NAMED("task_framework", "Starting future constraint violation detection   - Version 2a - Total steps taken " << actual_lookahead_steps);
    assert(num_models_ == 1 && starting_world_state.all_grippers_single_pose_.size() == 2);
    const auto& correspondences = dijkstras_task_->getCoverPointCorrespondences(starting_world_state);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    WorldState world_state_copy = starting_world_state;
    RubberBand rubber_band_between_grippers_copy = *rubber_band_between_grippers_.get();
    rubber_band_between_grippers_copy.visualize(PROJECTED_BAND_NS, PREDICTION_RUBBER_BAND_SAFE_COLOR, PREDICTION_RUBBER_BAND_VIOLATION_COLOR, 1, visualization_enabled);

    projected_deformable_point_paths_and_projected_virtual_rubber_bands.second.reserve(actual_lookahead_steps);
    for (size_t t = 0; t < actual_lookahead_steps; ++t)
    {
        // We only want the error correction part of the movement
        DesiredDirection desired_object_manipulation_direction;
        desired_object_manipulation_direction.error_correction_ =
                dijkstras_task_->calculateErrorCorrectionDeltaFixedCorrespondences(world_state_copy, correspondences.correspondences_);
        desired_object_manipulation_direction.stretching_correction_ = ObjectDeltaAndWeight(world_state_copy.object_configuration_.size());
        desired_object_manipulation_direction.combined_correction_ = desired_object_manipulation_direction.error_correction_;

        // It is assumed that the robot's internal state matches that that is passed to us, so we do not need to set active dof values
        const MatrixXd robot_dof_to_grippers_poses_jacobian = robot_->getGrippersJacobian();
        const std::vector<std::pair<CollisionData, Matrix3Xd>> poi_collision_data_ = robot_->getPointsOfInterestCollisionData();

        const double normal_motion_grippers_max_step = robot_->max_gripper_velocity_norm_ * robot_->dt_;
        const double forward_prediction_grippers_max_step = dijkstras_task_->work_space_grid_.minStepDimension() * 1.1;
        const double velocity_scale_factor = forward_prediction_grippers_max_step / normal_motion_grippers_max_step;
        const double normal_motion_robot_dof_max_step = robot_->max_dof_velocity_norm_ * robot_->dt_;
        const double forward_prediction_robot_dof_max_step = velocity_scale_factor * normal_motion_robot_dof_max_step * 2.0;

        const DeformableController::InputData input_data(
                    world_state_copy,
                    desired_object_manipulation_direction,
                    robot_,
                    robot_dof_to_grippers_poses_jacobian,
                    world_state_copy.robot_configuration_valid_,
                    poi_collision_data_,
                    forward_prediction_grippers_max_step,
                    forward_prediction_robot_dof_max_step);

        const DeformableController::OutputData robot_command = controller_list_[0]->getGripperMotion(input_data);

        /*
           Things to be updated in world_state_copy after "executing" a robot commad
                ObjectPointSet object_configuration_;
                AllGrippersSinglePose all_grippers_single_pose_;
                VectorXd robot_configuration_;
                bool robot_configuration_valid_;
                std::vector<CollisionData> gripper_collision_data_;
                double sim_time_;
        */


        // Move the grippers forward
        world_state_copy.all_grippers_single_pose_
                = kinematics::applyTwist(world_state_copy.all_grippers_single_pose_, robot_command.grippers_motion_);
        for (auto& pose : world_state_copy.all_grippers_single_pose_)
        {
            pose.translation() = dijkstras_task_->sdf_->ProjectOutOfCollisionToMinimumDistance3d(pose.translation(), GetRobotGripperRadius());
        }

        // Update the gripper collision data
        auto collision_check_future = std::async(std::launch::async, &RobotInterface::checkGripperCollision, robot_, world_state_copy.all_grippers_single_pose_);

        // Move the robot DOF forward
        world_state_copy.robot_configuration_ += robot_command.robot_dof_motion_;
        robot_->setActiveDOFValues(world_state_copy.robot_configuration_);

        // Move the cloth forward - copy the "projected" state of the cloth into the world_state_copy
        world_state_copy.sim_time_ += robot_->dt_;
        for (ssize_t node_ind = 0; node_ind < world_state_copy.object_configuration_.cols(); ++node_ind)
        {
            if (projected_deformable_point_paths[node_ind].size() > t + 1)
            {
                world_state_copy.object_configuration_.col(node_ind) = projected_deformable_point_paths[node_ind][t + 1];
            }
        }

        // Move the virtual rubber band to follow the grippers, projecting out of collision as needed
        rubber_band_between_grippers_copy.forwardPropagateRubberBandToEndpointTargets(
                    world_state_copy.all_grippers_single_pose_[0].translation(),
                    world_state_copy.all_grippers_single_pose_[1].translation(),
                    band_verbose);
        projected_deformable_point_paths_and_projected_virtual_rubber_bands.second.push_back(rubber_band_between_grippers_copy);

        // Visualize
        if (visualization_enabled)
        {
            rubber_band_between_grippers_copy.visualize(PROJECTED_BAND_NS, PREDICTION_RUBBER_BAND_SAFE_COLOR, PREDICTION_RUBBER_BAND_VIOLATION_COLOR, (int32_t)t + 2, visualization_enabled);
            vis_->visualizeGrippers(PROJECTED_GRIPPER_NS, world_state_copy.all_grippers_single_pose_, PREDICTION_GRIPPER_COLOR, (int32_t)(2 * t) + 2);
        }

        // Finish collecting the gripper collision data
        world_state_copy.gripper_collision_data_ = collision_check_future.get();
    }
    ROS_INFO_STREAM_NAMED("task_framework", "Calculated future constraint violation detection - Version 2a - in " << function_wide_stopwatch(READ) << " seconds");

    // Add duplicates of the last state to clear out any old visualizations
    if (visualization_enabled)
    {
        for (size_t t = actual_lookahead_steps; t < max_lookahead_steps_; ++t)
        {
            rubber_band_between_grippers_copy.visualize(PROJECTED_BAND_NS, PREDICTION_RUBBER_BAND_SAFE_COLOR, PREDICTION_RUBBER_BAND_VIOLATION_COLOR, (int32_t)t + 2, visualization_enabled);
            vis_->visualizeGrippers(PROJECTED_GRIPPER_NS, world_state_copy.all_grippers_single_pose_, PREDICTION_GRIPPER_COLOR, (int32_t)(2 * t) + 2);
        }
        vis_->forcePublishNow();
    }

    // Revert the robot state back to what it was before this was called
    robot_->setActiveDOFValues(starting_world_state.robot_configuration_);

    return projected_deformable_point_paths_and_projected_virtual_rubber_bands;
}

bool TaskFramework::globalPlannerNeededDueToOverstretch(
        const WorldState& current_world_state)
{
    static double annealing_factor = GetRubberBandOverstretchPredictionAnnealingFactor(ph_);

    const bool visualization_enabled = true;
    const auto projection_results = projectFutureSystemState(current_world_state, visualization_enabled);
    const auto& projected_rubber_bands = projection_results.second;

    if (projected_rubber_bands.size() == 0)
    {
        return false;
    }

    double filtered_band_length = projected_rubber_bands.front().totalLength();

    for (size_t t = 0; t < projected_rubber_bands.size(); ++t)
    {
        const RubberBand& band = projected_rubber_bands[t];
        const double band_length = band.totalLength();
        const std::pair<Vector3d, Vector3d> endpoints = band.getEndpoints();
        const double distance_between_endpoints = (endpoints.first - endpoints.second).norm();

        // Apply a low pass filter to the band length to try and remove "blips" in the estimate
        filtered_band_length = annealing_factor * filtered_band_length + (1.0 - annealing_factor) * band_length;

//        std::cout << "Band length:          " << band_length << std::endl;
//        std::cout << "Filtered band length: " << filtered_band_length << std::endl;
//        std::cout << "Max band length:      " << band.maxSafeLength() << std::endl;
//        std::cout << "distance between endpoints: " << distance_between_endpoints << std::endl;

        // If the band is currently overstretched, and not in free space, then predict future problems
        if (filtered_band_length > band.maxSafeLength() && !CloseEnough(band_length, distance_between_endpoints, 1e-3))
        {
            return true;
        }
    }

    return false;
}

bool TaskFramework::globalPlannerNeededDueToLackOfProgress()
{
    static double error_delta_threshold_for_progress = GetErrorDeltaThresholdForProgress(ph_);
    static double grippers_distance_delta_threshold_for_progress = GetGrippersDistanceDeltaThresholdForProgress(ph_);

    // If we have not yet collected enough data, then assume we are not stuck
    if (grippers_pose_history_.size() < max_grippers_pose_history_length_)
    {
        return false;
    }

    assert(grippers_pose_history_.size() == max_grippers_pose_history_length_);

    // Calculate distances from the first gripper config to the last
    const AllGrippersSinglePose& start_config = grippers_pose_history_[0];
    const double start_error = error_history_[0];
    std::vector<double> grippers_distance_deltas(max_grippers_pose_history_length_ - 1);
    std::vector<double> error_deltas(max_grippers_pose_history_length_ - 1);
    for (size_t time_idx = 1; time_idx < max_grippers_pose_history_length_; ++time_idx)
    {
        const AllGrippersSinglePoseDelta grippers_delta = CalculateGrippersPoseDelta(start_config, grippers_pose_history_[time_idx]);
        const double distance = MultipleGrippersVelocity6dNorm(grippers_delta);
        grippers_distance_deltas[time_idx - 1] = distance;
        error_deltas[time_idx - 1] = error_history_[time_idx] - start_error;
    }

    if (bandits_logging_enabled_)
    {
        // Determine if there is a general positive slope on the distances
        // - we should be moving away from the start config if we are not stuck
        LOG(loggers_.at("grippers_distance_delta_history"), PrettyPrint::PrettyPrint(grippers_distance_deltas, false, ", "));

        LOG(loggers_.at("error_delta_history"), PrettyPrint::PrettyPrint(error_deltas, false, ", "));
    }

    // If error has not decreased sufficiently, then we may not be making progress
    const double error_improvemnt = start_error - error_history_.back();
    if (error_improvemnt < error_delta_threshold_for_progress)
    {
        // If error has not decreased sufficiently, and the grippers have not moved much, then we are not making progress
        const double grippers_distance_delta = grippers_distance_deltas.back();
        if (grippers_distance_delta < grippers_distance_delta_threshold_for_progress)
        {
            return true;
        }
    }

    return false;
}

bool TaskFramework::predictStuckForGlobalPlannerResults(const bool visualization_enabled)
{
    static double annealing_factor = GetRubberBandOverstretchPredictionAnnealingFactor(ph_);

    //#warning "!!!!!!! Global plan overstretch check disabled!!!!!"
    //return false;

    assert(global_plan_next_timestep_ < rrt_planned_path_.size());

    constexpr bool band_verbose = false;

    RubberBand band = *rubber_band_between_grippers_;

    bool overstretch_predicted = false;
    double filtered_band_length = band.totalLength();

    for (size_t t = 0; t < max_lookahead_steps_; ++t)
    {
        // Always predict the full number of steps, duplicating the last point in the path as needed
        const size_t next_idx = std::min(rrt_planned_path_.size() - 1, global_plan_next_timestep_+ t);

        // Forward project the band and check for overstretch
        const auto& grippers_pose = rrt_planned_path_[next_idx].grippers();
        band.forwardPropagateRubberBandToEndpointTargets(
                    grippers_pose.first.translation(),
                    grippers_pose.second.translation(),
                    band_verbose);
        const double band_length = band.totalLength();
        const std::pair<Vector3d, Vector3d> endpoints = band.getEndpoints();
        const double distance_between_endpoints = (endpoints.first - endpoints.second).norm();

        filtered_band_length = annealing_factor * filtered_band_length + (1.0 - annealing_factor) * band_length;
        if (filtered_band_length > band.maxSafeLength() && !CloseEnough(band_length, distance_between_endpoints, 1e-3))
        {
            overstretch_predicted = true;
        }

        // Visualize
        band.visualize(PROJECTED_BAND_NS, PREDICTION_RUBBER_BAND_SAFE_COLOR, PREDICTION_RUBBER_BAND_VIOLATION_COLOR, (int32_t)t + 2, visualization_enabled);
        vis_->visualizeGrippers(PROJECTED_GRIPPER_NS, {grippers_pose.first, grippers_pose.second}, PREDICTION_GRIPPER_COLOR, (int32_t)(2 * t) + 2);
    }

    vis_->forcePublishNow();

    return overstretch_predicted;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global gripper planner functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AllGrippersSinglePose TaskFramework::getGripperTargets(const WorldState& world_state)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////// Determine the cluster centers /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const auto& correspondences = dijkstras_task_->getCoverPointCorrespondences(world_state);
    const auto& cover_point_indices_= correspondences.uncovered_target_points_idxs_;

    // Only cluster the points that are not covered
    VectorVector3d cluster_targets;
    cluster_targets.reserve(cover_point_indices_.size());
    for (size_t idx = 0; idx < cover_point_indices_.size(); ++idx)
    {
        const ssize_t cover_idx = cover_point_indices_[idx];
        cluster_targets.push_back(dijkstras_task_->cover_points_.col(cover_idx));
    }

//    vis_->visualizePoints(CLUSTERING_TARGETS_NS, cluster_targets, Visualizer::Blue(), 1);

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
    const std::vector<uint32_t>& cluster_labels = cluster_results.first;
    const VectorVector3d cluster_centers = cluster_results.second;
    const uint32_t num_clusters = (uint32_t)cluster_centers.size();
    assert(num_clusters == 2);

//    vis_->visualizeCubes(CLUSTERING_RESULTS_PRE_PROJECT_NS, cluster_centers, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Red(), 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////// Determine which gripper gets assigned which cluster center ////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Get the orientations for each gripper based on their starting orientation
    AllGrippersSinglePose target_gripper_poses = world_state.all_grippers_single_pose_;

    // Decide which gripper goes to which cluster
    {
        const auto& gripper0_grapsed_points = dijkstras_task_->getGripperAttachedNodesIndices(0);
        const auto& gripper1_grapsed_points = dijkstras_task_->getGripperAttachedNodesIndices(1);

        const auto gripper0_cluster_counts = numberOfPointsInEachCluster(cluster_labels, num_clusters, gripper0_grapsed_points, correspondences);
        const auto gripper1_cluster_counts = numberOfPointsInEachCluster(cluster_labels, num_clusters, gripper1_grapsed_points, correspondences);

        // Set some values so that the logic used in the if-else chain makes sense to read
        const bool gripper0_no_match_to_cluster0    = gripper0_cluster_counts[0] == 0;
        const bool gripper0_no_match_to_cluster1    = gripper0_cluster_counts[1] == 0;
        const bool gripper0_no_match_to_any_cluster = gripper0_no_match_to_cluster0 && gripper0_no_match_to_cluster1;
        const bool gripper0_best_match_to_cluster0  = gripper0_cluster_counts[0] > gripper1_cluster_counts[0]; // Note that this requires at least 1 correspondence for gripper0 to cluster0
        const bool gripper0_best_match_to_cluster1  = gripper0_cluster_counts[1] > gripper1_cluster_counts[1]; // Note that this requires at least 1 correspondence for gripper0 to cluster1

        const bool gripper1_no_match_to_cluster0    = gripper1_cluster_counts[0] == 0;
        const bool gripper1_no_match_to_cluster1    = gripper1_cluster_counts[1] == 0;
        const bool gripper1_no_match_to_any_cluster = gripper1_no_match_to_cluster0 && gripper1_no_match_to_cluster1;
        const bool gripper1_best_match_to_cluster0  = gripper1_cluster_counts[0] > gripper0_cluster_counts[0]; // Note that this requires at least 1 correspondence for gripper1 to cluster0
        const bool gripper1_best_match_to_cluster1  = gripper1_cluster_counts[1] > gripper0_cluster_counts[1]; // Note that this requires at least 1 correspondence for gripper1 to cluster1

        const bool equal_match_to_cluster0 = (!gripper0_no_match_to_cluster0) && (!gripper1_no_match_to_cluster0) && (gripper0_cluster_counts[0] == gripper1_cluster_counts[0]);
        const bool equal_match_to_cluster1 = (!gripper0_no_match_to_cluster1) && (!gripper1_no_match_to_cluster1) && (gripper0_cluster_counts[1] == gripper1_cluster_counts[1]);

        const bool gripper0_best_match_to_both = gripper0_best_match_to_cluster0 && gripper0_best_match_to_cluster1;
        const bool gripper1_best_match_to_both = gripper1_best_match_to_cluster0 && gripper1_best_match_to_cluster1;


        // If each gripper has a unique best pull direction, use it
        if (gripper0_best_match_to_cluster0 && gripper1_best_match_to_cluster1)
        {
            target_gripper_poses[0].translation() = cluster_centers[0];
            target_gripper_poses[1].translation() = cluster_centers[1];
        }
        else if (gripper0_best_match_to_cluster1 && gripper1_best_match_to_cluster0)
        {
            target_gripper_poses[0].translation() = cluster_centers[1];
            target_gripper_poses[1].translation() = cluster_centers[0];
        }
        // If a single gripper has the best pull to both, then that gripper dominates the choice
        else if (gripper0_best_match_to_both)
        {
            if (gripper0_cluster_counts[0] > gripper0_cluster_counts[1])
            {
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
            else if (gripper0_cluster_counts[0] < gripper0_cluster_counts[1])
            {
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
            // If gripper0 has no unique best target, then allow gripper1 to make the choice
            else
            {
                if (gripper1_cluster_counts[0] > gripper1_cluster_counts[1])
                {
                    target_gripper_poses[0].translation() = cluster_centers[1];
                    target_gripper_poses[1].translation() = cluster_centers[0];
                }
                else if (gripper1_cluster_counts[0] < gripper1_cluster_counts[1])
                {
                    target_gripper_poses[0].translation() = cluster_centers[0];
                    target_gripper_poses[1].translation() = cluster_centers[1];
                }
                // If everything is all tied up, decide what to do later
                else
                {
                    assert(false && "Setting gripper targets needs more logic");
                }
            }
        }
        else if (gripper1_best_match_to_both)
        {
            if (gripper1_cluster_counts[0] > gripper1_cluster_counts[1])
            {
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
            else if (gripper1_cluster_counts[0] < gripper1_cluster_counts[1])
            {
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
            // If gripper1 has no unique best target, then allow gripper0 to make the choice
            else
            {
                if (gripper0_cluster_counts[0] > gripper0_cluster_counts[1])
                {
                    target_gripper_poses[0].translation() = cluster_centers[0];
                    target_gripper_poses[1].translation() = cluster_centers[1];
                }
                else if (gripper0_cluster_counts[0] < gripper0_cluster_counts[1])
                {
                    target_gripper_poses[0].translation() = cluster_centers[1];
                    target_gripper_poses[1].translation() = cluster_centers[0];
                }
                // If everything is all tied up, decide what to do later
                else
                {
                    assert(false && "Setting gripper targets needs more logic");
                }
            }
        }
        // If there is only a pull on a single gripper, then that gripper dominates the choice
        else if (!gripper0_no_match_to_any_cluster &&  gripper1_no_match_to_any_cluster)
        {
            // Double check the logic that got us here; lets me simplify the resulting logic
            // Gripper1 has no pulls on it, and gripper0 has pulls from only 1 cluster, otherwise
            // one of the other caes would have triggered before this one
            assert(!gripper0_best_match_to_both);
            if (gripper0_best_match_to_cluster0)
            {
                assert(gripper0_no_match_to_cluster1);
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
            else if (gripper0_best_match_to_cluster1)
            {
                assert(gripper0_no_match_to_cluster0);
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
            else
            {
                assert(false && "Logic error in set gripper targets");
            }

        }
        else if ( gripper0_no_match_to_any_cluster && !gripper1_no_match_to_any_cluster)
        {
            // Double check the logic that got us here; lets me simplify the resulting logic
            // Gripper0 has no pulls on it, and gripper1 has pulls from only 1 cluster, otherwise
            // one of the other caes would have triggered before this one
            assert(!gripper1_best_match_to_both);
            if (gripper1_best_match_to_cluster0)
            {
                assert(gripper1_no_match_to_cluster1);
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
            else if (gripper1_best_match_to_cluster1)
            {
                assert(gripper1_no_match_to_cluster0);
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
            else
            {
                assert(false && "Logic error in set gripper targets");
            }
        }
        // If neither gripper has a pull on it, or both grippers have equal pull, then use some other metric
        else if ((gripper0_no_match_to_any_cluster  && gripper1_no_match_to_any_cluster) ||
                 (equal_match_to_cluster0           && equal_match_to_cluster1))
        {
            const std::vector<double> gripper0_distances_to_clusters =
                    dijkstras_task_->averageDijkstrasDistanceBetweenGrippersAndClusters(world_state.all_grippers_single_pose_[0], correspondences.uncovered_target_points_idxs_, cluster_labels, num_clusters);
            const std::vector<double> gripper1_distances_to_clusters =
                    dijkstras_task_->averageDijkstrasDistanceBetweenGrippersAndClusters(world_state.all_grippers_single_pose_[1], correspondences.uncovered_target_points_idxs_, cluster_labels, num_clusters);

            const bool gripper0_is_closest_to_cluster0 = gripper0_distances_to_clusters[0] <= gripper1_distances_to_clusters[0];
            const bool gripper0_is_closest_to_cluster1 = gripper0_distances_to_clusters[1] <= gripper1_distances_to_clusters[1];

            // If there is a unique best match, then use it
            if (gripper0_is_closest_to_cluster0 && !gripper0_is_closest_to_cluster1)
            {
                target_gripper_poses[0].translation() = cluster_centers[0];
                target_gripper_poses[1].translation() = cluster_centers[1];
            }
            else if (!gripper0_is_closest_to_cluster0 && gripper0_is_closest_to_cluster1)
            {
                target_gripper_poses[0].translation() = cluster_centers[1];
                target_gripper_poses[1].translation() = cluster_centers[0];
            }
            // Otherwise, pick the combination that minimizes the total distance
            else
            {
                const double dist_version0 = gripper0_distances_to_clusters[0] + gripper1_distances_to_clusters[1];
                const double dist_version1 = gripper0_distances_to_clusters[1] + gripper1_distances_to_clusters[0];

                if (dist_version0 <= dist_version1)
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
        // If none of the above are true, than there is a logic error
        else
        {
            assert(false && "Unhandled edge case in get gripper targets");
        }

//        vis_->visualizeCubes(CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS, {world_state.all_grippers_single_pose_[0].translation(), target_gripper_poses[0].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Magenta(), 1);
//        vis_->visualizeCubes(CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS, {world_state.all_grippers_single_pose_[1].translation(), target_gripper_poses[1].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Cyan(), 5);
    }

    // Project the targets out of collision
    const double min_dist_to_obstacles = std::max(GetControllerMinDistanceToObstacles(ph_), GetRRTMinGripperDistanceToObstacles(ph_)) * GetRRTTargetMinDistanceScaleFactor(ph_);
    const Vector3d gripper0_position_pre_project = target_gripper_poses[0].translation();
    const Vector3d gripper1_position_pre_project = target_gripper_poses[1].translation();
    target_gripper_poses[0].translation() = dijkstras_task_->sdf_->ProjectOutOfCollisionToMinimumDistance3d(gripper0_position_pre_project, min_dist_to_obstacles);
    target_gripper_poses[1].translation() = dijkstras_task_->sdf_->ProjectOutOfCollisionToMinimumDistance3d(gripper1_position_pre_project, min_dist_to_obstacles);

    // Visualization
    {
        vis_->visualizeCubes(CLUSTERING_RESULTS_POST_PROJECT_NS, {target_gripper_poses[0].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Magenta(), 1);
        vis_->visualizeCubes(CLUSTERING_RESULTS_POST_PROJECT_NS, {target_gripper_poses[1].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Red(), 5);

        std::vector<std_msgs::ColorRGBA> colors;
        for (size_t idx = 0; idx < cluster_targets.size(); ++idx)
        {
            colors.push_back(arc_helpers::GenerateUniqueColor<std_msgs::ColorRGBA>(cluster_labels[idx] + 2, 0.5));
        }
        const std::vector<double> radiuses(cluster_targets.size(), dijkstras_task_->work_space_grid_.minStepDimension());
        vis_->visualizeSpheres(CLUSTERING_TARGETS_NS, cluster_targets, colors, 10, radiuses);
        vis_->forcePublishNow();
    }


//    std::cout << "cover_points = [\n" << dijkstras_task_->cover_points_ << "];\n";
//    std::cout << "cluster_targets = [\n" << PrettyPrint::PrettyPrint(cluster_targets, false, "\n") << "];\n";
//    std::cout << "cluster_labels = [" << PrettyPrint::PrettyPrint(cluster_labels, false, " ") << "];\n";



    return target_gripper_poses;
}

void TaskFramework::planGlobalGripperTrajectory(const WorldState& world_state)
{
    static int num_times_invoked = 0;
    num_times_invoked++;
    ROS_INFO_STREAM("!!!!!!!!!!!!!!!!!! Planner Invoked " << num_times_invoked << " times!!!!!!!!!!!");

    // Resample the band for the purposes of first order vis checking
    const auto distance_fn = [] (const Eigen::Vector3d& v1, const Eigen::Vector3d& v2)
    {
        return (v1 - v2).norm();
    };
    const auto interpolation_fn = [] (const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const double ratio)
    {
        return EigenHelpers::Interpolate(v1, v2, ratio);
    };
    rrt_helper_->addBandToBlacklist(
                path_utils::ResamplePath(
                    rubber_band_between_grippers_->getVectorRepresentation(),
                    dijkstras_task_->work_space_grid_.minStepDimension() / 2.0,
                    distance_fn,
                    interpolation_fn));

    vis_->purgeMarkerList();
    visualization_msgs::Marker marker;
    marker.action = visualization_msgs::Marker::DELETEALL;
    marker.header.frame_id = "world_origin";
    marker.header.stamp = ros::Time::now();
    vis_->publish(marker);
    vis_->forcePublishNow();
    vis_->purgeMarkerList();

    rrt_planned_path_.clear();
    if (GetRRTReuseOldResults(ph_))
    {
        // Deserialization
        ROS_INFO_NAMED("rrt_planner_results", "Checking if RRT solution already exists");
        const std::string file_path =
                GetLogFolder(nh_) +
                "rrt_cache_step." +
                PrettyPrint::PrettyPrint(num_times_invoked);
        rrt_planned_path_ = rrt_helper_->loadStoredPath(file_path);

        if (world_state.robot_configuration_valid_)
        {
            std::cerr << "!!!!!!!!!!!!! About to execute a stored plan on a physical robot\n";
            std::cerr << "Are you sure about this? ";
            std::string input;
            std::cin >> input;
            if (input != "yes")
            {
                std::cerr << "Ignoring path loaded from file, replanning.\n";
                rrt_planned_path_.clear();
            }
        }
    }

    // Planning if we did not load a plan from file
    if (rrt_planned_path_.size() == 0)
    {
        const RRTGrippersRepresentation gripper_config(
                    world_state.all_grippers_single_pose_[0],
                    world_state.all_grippers_single_pose_[1]);

        RRTRobotRepresentation robot_config;
        if (world_state.robot_configuration_valid_)
        {
            robot_config = world_state.robot_configuration_;
        }
        else
        {
            robot_config.resize(6);
            robot_config.head<3>() = gripper_config.first.translation();
            robot_config.tail<3>() = gripper_config.second.translation();
        }

        const RRTNode start_config(
                    gripper_config,
                    robot_config,
                    rubber_band_between_grippers_);

        const AllGrippersSinglePose target_grippers_poses_vec = getGripperTargets(world_state);
        const RRTGrippersRepresentation target_grippers_poses(
                    target_grippers_poses_vec[0],
                    target_grippers_poses_vec[1]);

        const std::chrono::duration<double> time_limit(GetRRTTimeout(ph_));
        const size_t num_trials = GetRRTNumTrials(ph_);
        for (size_t trial_idx = 0; trial_idx < num_trials; ++trial_idx)
        {
            // Only use the seed resetting if we are performing more than 1 trial
            if (num_trials > 1)
            {
                robot_->resetRandomSeeds(seed_, trial_idx * 0xFFFF);
                flann::seed_random((unsigned int)seed_);
                generator_->seed(seed_);
                generator_->discard(trial_idx * 0xFFFF);
                for (size_t discard_idx = 0; discard_idx < trial_idx * 0xFFFF; ++discard_idx)
                {
                    std::rand();
                }

                std::cout << "Trial idx: " << trial_idx << std::endl;
            }

            rrt_planned_path_.clear();
            while (rrt_planned_path_.size() == 0)
            {
                rrt_planned_path_ = rrt_helper_->plan(
                            start_config,
                            target_grippers_poses,
                            time_limit);
            }

            if (!GetDisableAllVisualizations(ph_))
            {
                vis_->deleteObjects(RRTHelper::RRT_BLACKLISTED_GOAL_BANDS_NS, 1, 2);
                rrt_helper_->visualizePath(rrt_planned_path_);
                vis_->forcePublishNow(0.5);
            }

            // Serialization
            if (GetRRTStoreNewResults(ph_))
            {
                ROS_INFO_NAMED("rrt_planner_results", "Compressing and saving RRT results to file for storage");
                const std::string file_path =
                        GetLogFolder(nh_) +
                        "rrt_cache_step." +
                        PrettyPrint::PrettyPrint(num_times_invoked);
                rrt_helper_->storePath(rrt_planned_path_, file_path);
            }
        }
    }

    global_plan_next_timestep_ = 0;
    executing_global_trajectory_ = true;

//    assert(false && "Terminating as this is just a planning test");
//    std::cout << "Waiting on keystroke before executing trajectory" << std::endl;
//    std::getchar();
}

/*
void TaskFramework::convertRRTResultIntoGripperTrajectory(
        const std::vector<RRTNode, RRTAllocator>& rrt_result)
{
    AllGrippersPoseTrajectory traj;
    traj.reserve(rrt_result.size());

    for (size_t ind = 0; ind < rrt_result.size(); ++ind)
    {
        const AllGrippersSinglePose grippers_poses = {
            rrt_result[ind].grippers().first,
            rrt_result[ind].grippers().second};
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
        result[0].translation() = Interpolate(Vector3d(a[0].translation()), Vector3d(b[0].translation()), ratio);
        result[1].translation() = Interpolate(Vector3d(a[1].translation()), Vector3d(b[1].translation()), ratio);
        return result;
    };

    global_plan_gripper_trajectory_ = path_utils::ResamplePath<AllGrippersSinglePose>(traj, robot_->max_gripper_velocity_norm_ * robot_->dt_, distance_fn, interpolation_fn);
    global_plan_full_robot_trajectory_ = std::vector<VectorXd>(global_plan_gripper_trajectory_.size(), VectorXd(0));
}

void TaskFramework::convertRRTResultIntoFullRobotTrajectory(
        const std::vector<RRTNode, RRTAllocator>& rrt_result)
{
    global_plan_gripper_trajectory_.clear();
    global_plan_gripper_trajectory_.reserve(rrt_result.size());

    global_plan_full_robot_trajectory_.clear();
    global_plan_full_robot_trajectory_.reserve(rrt_result.size());

    for (size_t ind = 0; ind < rrt_result.size(); ++ind)
    {
        const AllGrippersSinglePose grippers_poses = {
            rrt_result[ind].grippers().first,
            rrt_result[ind].grippers().second};
        global_plan_gripper_trajectory_.push_back(grippers_poses);
        global_plan_full_robot_trajectory_.push_back(rrt_result[ind].robotConfiguration());
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Model list management
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TaskFramework::initializeModelAndControllerSet(const WorldState& initial_world_state)
{
    // Initialze each model type with the shared data
    DeformableModel::SetGrippersData(robot_->getGrippersData());
    DeformableModel::SetCallbackFunctions(std::bind(&RobotInterface::checkGripperCollision, robot_, std::placeholders::_1));
    DiminishingRigidityModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));
    ConstraintJacobianModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));

    const bool optimization_enabled = GetJacobianControllerOptimizationEnabled(ph_);
    const TrialType trial_type = GetTrialType(ph_);

    switch (trial_type)
    {
        case DIMINISHING_RIGIDITY_SINGLE_MODEL_LEAST_SQUARES_STRETCHING_AVOIDANCE_CONTROLLER:
        {
            double translational_deformability, rotational_deformability;
            if (ph_.getParam("translational_deformability", translational_deformability) &&
                     ph_.getParam("rotational_deformability", rotational_deformability))
            {
                ROS_INFO_STREAM_NAMED("task_framework", "Overriding deformability values to "
                                       << translational_deformability << " "
                                       << rotational_deformability);
            }
            else
            {
                translational_deformability = task_specification_->defaultDeformability();
                rotational_deformability = task_specification_->defaultDeformability();
                ROS_INFO_STREAM_NAMED("task_framework", "Using default deformability value of "
                                       << task_specification_->defaultDeformability());
            }

            model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                      translational_deformability,
                                      rotational_deformability));

            controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                           nh_,
                                           ph_,
                                           robot_,
                                           vis_,
                                           model_list_.back(),
                                           task_specification_->collisionScalingFactor(),
                                           optimization_enabled));
            break;
        }
        case ADAPTIVE_JACOBIAN_SINGLE_MODEL_LEAST_SQUARES_STRETCHING_AVOIDANCE_CONTROLLER:
        {
            model_list_.push_back(std::make_shared<AdaptiveJacobianModel>(
                                      DiminishingRigidityModel(task_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(initial_world_state),
                                      GetAdaptiveModelLearningRate(ph_)));

            controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                           nh_,
                                           ph_,
                                           robot_,
                                           vis_,
                                           model_list_.back(),
                                           task_specification_->collisionScalingFactor(),
                                           optimization_enabled));
            break;
        }
        case DIMINISHING_RIGIDITY_SINGLE_MODEL_LEAST_SQUARES_STRETCHING_CONSTRAINT_CONTROLLER:
        {
            double translational_deformability, rotational_deformability;
            if (ph_.getParam("translational_deformability", translational_deformability) &&
                     ph_.getParam("rotational_deformability", rotational_deformability))
            {
                ROS_INFO_STREAM_NAMED("task_framework", "Overriding deformability values to "
                                       << translational_deformability << " "
                                       << rotational_deformability);
            }
            else
            {
                translational_deformability = task_specification_->defaultDeformability();
                rotational_deformability = task_specification_->defaultDeformability();
                ROS_INFO_STREAM_NAMED("task_framework", "Using default deformability value of "
                                       << task_specification_->defaultDeformability());
            }

            model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                      translational_deformability,
                                      rotational_deformability));

            controller_list_.push_back(std::make_shared<LeastSquaresControllerWithStretchingConstraint>(
                                           nh_,
                                           ph_,
                                           robot_,
                                           vis_,
                                           model_list_.back()));
            break;
        }
        case CONSTRAINT_SINGLE_MODEL_CONSTRAINT_CONTROLLER:
        {
            ROS_INFO_NAMED("task_framework", "Using constraint model and random sampling controller");

            const double translation_dir_deformability = GetConstraintTranslationalDir(ph_);
            const double translation_dis_deformability = GetConstraintTranslationalDis(ph_);
            const double rotation_deformability = GetConstraintRotational(ph_);
//            const double translational_deformability = GetConstraintTranslationalOldVersion(ph_);

            const sdf_tools::SignedDistanceField::ConstPtr environment_sdf(GetEnvironmentSDF(nh_));

            model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                  translation_dir_deformability,
                                  translation_dis_deformability,
                                  rotation_deformability,
//                                  translational_deformability,
                                  environment_sdf));

            controller_list_.push_back(std::make_shared<StretchingAvoidanceController>(
                                           nh_,
                                           ph_,
                                           robot_,
                                           vis_,
                                           model_list_.back(),
                                           environment_sdf,
                                           generator_,
                                           GetStretchingAvoidanceControllerSolverType(ph_),
                                           GetMaxSamplingCounts(ph_)));
            break;
        }
        case DIMINISHING_RIGIDITY_SINGLE_MODEL_CONSTRAINT_CONTROLLER:
        {
            ROS_INFO_NAMED("task_framework", "Using dminishing model and random sampling controller");

            double translational_deformability, rotational_deformability;
            const sdf_tools::SignedDistanceField::ConstPtr environment_sdf(GetEnvironmentSDF(nh_));

            if (ph_.getParam("translational_deformability", translational_deformability) &&
                     ph_.getParam("rotational_deformability", rotational_deformability))
            {
                ROS_INFO_STREAM_NAMED("task_framework", "Overriding deformability values to "
                                       << translational_deformability << " "
                                       << rotational_deformability);
            }
            else
            {
                translational_deformability = task_specification_->defaultDeformability();
                rotational_deformability = task_specification_->defaultDeformability();
                ROS_INFO_STREAM_NAMED("task_framework", "Using default deformability value of "
                                       << task_specification_->defaultDeformability());
            }

            model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                      translational_deformability,
                                      rotational_deformability));

            controller_list_.push_back(std::make_shared<StretchingAvoidanceController>(
                                           nh_,
                                           ph_,
                                           robot_,
                                           vis_,
                                           model_list_.back(),
                                           environment_sdf,
                                           generator_,
                                           GetStretchingAvoidanceControllerSolverType(ph_),
                                           GetMaxSamplingCounts(ph_)));
            break;
        }
        case MULTI_MODEL_BANDIT_TEST:
        {
            ////////////////////////////////////////////////////////////////////////
            // Diminishing rigidity models
            ////////////////////////////////////////////////////////////////////////

            #pragma message "Magic numbers: Multi-model deform range"
            const double deform_min = 0.0;
            const double deform_max = 25.0;
            const double deform_step = 4.0;

            for (double trans_deform = deform_min; trans_deform < deform_max; trans_deform += deform_step)
            {
                for (double rot_deform = deform_min; rot_deform < deform_max; rot_deform += deform_step)
                {
                    model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                              trans_deform,
                                              rot_deform));

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerWithStretchingConstraint>(
                                                   nh_,
                                                   ph_,
                                                   robot_,
                                                   vis_,
                                                   model_list_.back()));
                }
            }
            ROS_INFO_STREAM_NAMED("task_framework", "Num diminishing rigidity models: "
                                   << std::floor((deform_max - deform_min) / deform_step));

            ////////////////////////////////////////////////////////////////////////
            // Adaptive jacobian models
            ////////////////////////////////////////////////////////////////////////

            #pragma message "Magic numbers: Multi-model adaptive range"
            const double learning_rate_min = 1e-10;
            const double learning_rate_max = 1.1e0;
            const double learning_rate_step = 10.0;

            for (double learning_rate = learning_rate_min; learning_rate < learning_rate_max; learning_rate *= learning_rate_step)
            {
                    model_list_.push_back(std::make_shared<AdaptiveJacobianModel>(
                                              DiminishingRigidityModel(task_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(initial_world_state),
                                              learning_rate));

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerWithStretchingConstraint>(
                                                   nh_,
                                                   ph_,
                                                   robot_,
                                                   vis_,
                                                   model_list_.back()));
            }
            ROS_INFO_STREAM_NAMED("task_framework", "Num adaptive Jacobian models: "
                                   << std::floor(std::log(learning_rate_max / learning_rate_min) / std::log(learning_rate_step)));
            break;
        }
        case MULTI_MODEL_CONTROLLER_TEST:
        {
            ROS_INFO_NAMED("task_framework", "Using multiple model-controller sets");

            // Constraint Model with New Controller. (MM)
            {
                const sdf_tools::SignedDistanceField::ConstPtr environment_sdf(GetEnvironmentSDF(nh_));

                const double translation_dir_deformability = GetConstraintTranslationalDir(ph_);
                const double translation_dis_deformability = GetConstraintTranslationalDis(ph_);
                const double rotation_deformability = GetConstraintRotational(ph_);
//                const double translational_deformability = GetConstraintTranslationalOldVersion(ph_);

                model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                      translation_dir_deformability,
                                      translation_dis_deformability,
                                      rotation_deformability,
//                                      translational_deformability,
                                      environment_sdf));

                controller_list_.push_back(std::make_shared<StretchingAvoidanceController>(
                                               nh_,
                                               ph_,
                                               robot_,
                                               vis_,
                                               model_list_.back(),
                                               environment_sdf,
                                               generator_,
                                               GetStretchingAvoidanceControllerSolverType(ph_),
                                               GetMaxSamplingCounts(ph_)));
            }

            // Dminishing Model with New Controller. (DM)
            /*
            {
                double translational_deformability, rotational_deformability;
                const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

                if (ph_.getParam("translational_deformability", translational_deformability) &&
                         ph_.getParam("rotational_deformability", rotational_deformability))
                {
                    ROS_INFO_STREAM_NAMED("task_framework", "Overriding deformability values to "
                                           << translational_deformability << " "
                                           << rotational_deformability);
                }
                else
                {
                    translational_deformability = task_specification_->defaultDeformability();
                    rotational_deformability = task_specification_->defaultDeformability();
                    ROS_INFO_STREAM_NAMED("task_framework", "Using default deformability value of "
                                           << task_specification_->defaultDeformability());
                }

                model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                          translational_deformability,
                                          rotational_deformability));

                controller_list_.push_back(std::make_shared<StretchingAvoidanceController>(
                                               nh_,
                                               ph_,
                                               robot_,
                                               vis_,
                                               model_list_.back(),
                                               environment_sdf,
                                               generator_,
                                               GetStretchingAvoidanceControllerSolverType(ph_),
                                               GetMaxSamplingCounts(ph_)));
            }
            */

            // Dminishing Model with Old Controller. (DD)
            {
                double translational_deformability, rotational_deformability;
                if (ph_.getParam("translational_deformability", translational_deformability) &&
                         ph_.getParam("rotational_deformability", rotational_deformability))
                {
                    ROS_INFO_STREAM_NAMED("task_framework", "Overriding deformability values to "
                                           << translational_deformability << " "
                                           << rotational_deformability);
                }
                else
                {
                    translational_deformability = task_specification_->defaultDeformability();
                    rotational_deformability = task_specification_->defaultDeformability();
                    ROS_INFO_STREAM_NAMED("task_framework", "Using default deformability value of "
                                           << task_specification_->defaultDeformability());
                }

                model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                          translational_deformability,
                                          rotational_deformability));

                controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                               nh_,
                                               ph_,
                                               robot_,
                                               vis_,
                                               model_list_.back(),
                                               task_specification_->collisionScalingFactor(),
                                               optimization_enabled));
            }
            break;
        }
        case MULTI_MODEL_ACCURACY_TEST:
        {
            // Constraint Model
            {
                const sdf_tools::SignedDistanceField::ConstPtr environment_sdf(GetEnvironmentSDF(nh_));

                const double translation_dir_deformability = GetConstraintTranslationalDir(ph_);
                const double translation_dis_deformability = GetConstraintTranslationalDis(ph_);
                const double rotation_deformability = GetConstraintRotational(ph_);
//                const double translational_deformability = GetConstraintTranslationalOldVersion(ph_);

                model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                      translation_dir_deformability,
                                      translation_dis_deformability,
                                      rotation_deformability,
//                                      translational_deformability,
                                      environment_sdf));

                controller_list_.push_back(std::make_shared<StraightLineController>(
                                               nh_,
                                               ph_,
                                               robot_,
                                               vis_,
                                               model_list_.back()));
            }
            // Dminishing Rigidity Model
            {
                double translational_deformability, rotational_deformability;
                if (ph_.getParam("translational_deformability", translational_deformability) &&
                         ph_.getParam("rotational_deformability", rotational_deformability))
                {
                    ROS_INFO_STREAM_NAMED("task_framework", "Overriding deformability values to "
                                           << translational_deformability << " "
                                           << rotational_deformability);
                }
                else
                {
                    translational_deformability = task_specification_->defaultDeformability();
                    rotational_deformability = task_specification_->defaultDeformability();
                    ROS_INFO_STREAM_NAMED("task_framework", "Using default deformability value of "
                                           << task_specification_->defaultDeformability());
                }

                model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                          translational_deformability,
                                          rotational_deformability));

                controller_list_.push_back(std::make_shared<StraightLineController>(
                                               nh_,
                                               ph_,
                                               robot_,
                                               vis_,
                                               model_list_.back()));
            }
            break;
        }
        default:
        {
            ROS_FATAL_NAMED("task_framework", "Invalid trial type, this should not be possible.");
            assert(false && "Invalid trial type, this should not be possible.");
        }
    }

    assert(controller_list_.size() == model_list_.size());

    createBandits();
}

void TaskFramework::createBandits()
{
    num_models_ = (ssize_t)model_list_.size();
    ROS_INFO_STREAM_NAMED("task_framework", "Generating bandits for " << num_models_ << " bandits");

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
// Model utility functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::updateModels
 * @param suggested_trajectories
 * @param model_used
 * @param world_feedback
 */
void TaskFramework::updateModels(
        const WorldState& starting_world_state,
        const ObjectDeltaAndWeight& task_desired_motion,
        const std::vector<DeformableController::OutputData>& suggested_commands,
        const ssize_t model_used,
        const WorldState& world_feedback)
{
    const static double reward_annealing_factor = GetRewardScaleAnnealingFactor(ph_);

    // First we update the bandit algorithm
    const double starting_error = task_specification_->calculateError(starting_world_state);
    const double true_error_reduction = starting_error - task_specification_->calculateError(world_feedback);
    reward_std_dev_scale_factor_ = std::max(1e-10, reward_annealing_factor * reward_std_dev_scale_factor_ + (1.0 - reward_annealing_factor) * std::abs(true_error_reduction));
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
MatrixXd TaskFramework::calculateProcessNoise(
        const std::vector<DeformableController::OutputData>& suggested_commands) const
{
    std::vector<double> grippers_velocity_norms((size_t)num_models_);

    for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
    {
        grippers_velocity_norms[model_ind] = MultipleGrippersVelocity6dNorm(suggested_commands[model_ind].grippers_motion_);
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
                            suggested_commands[(size_t)i].grippers_motion_,
                            suggested_commands[(size_t)j].grippers_motion_)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Logging and visualization functionality
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TaskFramework::visualizeDesiredMotion(
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
                DESIRED_DELTA_NS,
                AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
                colors);

//        if (task_specification_->deformable_type_ == DeformableType::CLOTH)
        {
            vis_->visualizeObjectDelta(
                        DESIRED_DELTA_NS,
                        current_world_state.object_configuration_,
                        AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
                        Visualizer::Green(),
                        10);
        }
    }
}

void TaskFramework::visualizeGripperMotion(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& gripper_motion,
        const ssize_t model_ind) const
{
    const auto grippers_test_poses = kinematics::applyTwist(current_gripper_pose, gripper_motion);
    EigenHelpers::VectorVector3d line_starts;
    EigenHelpers::VectorVector3d line_ends;

    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
    {
        line_starts.push_back(current_gripper_pose[gripper_ind].translation());
        line_ends.push_back(current_gripper_pose[gripper_ind].translation() + 100.0 * (grippers_test_poses[gripper_ind].translation() - current_gripper_pose[gripper_ind].translation()));
    }

    switch (model_ind)
    {
        case 0:
        {
            vis_->visualizeLines("MM_grippers_motion", line_starts, line_ends, Visualizer::Silver());
            vis_->visualizeLines("MM_grippers_motion", line_starts, line_ends, Visualizer::Silver());
            vis_->visualizeLines("MM_grippers_motion", line_starts, line_ends, Visualizer::Silver());
            vis_->visualizeLines("MM_grippers_motion", line_starts, line_ends, Visualizer::Silver());
            break;
        }
        case 1:
        {
            vis_->visualizeLines("DD_grippers_motion", line_starts, line_ends, Visualizer::Yellow());
            vis_->visualizeLines("DD_grippers_motion", line_starts, line_ends, Visualizer::Yellow());
            vis_->visualizeLines("DD_grippers_motion", line_starts, line_ends, Visualizer::Yellow());
            vis_->visualizeLines("DD_grippers_motion", line_starts, line_ends, Visualizer::Yellow());
            break;
        }
        default:
        {
            assert(false && "grippers_motion color not assigned for this index");
            break;
        }
    }
}

void TaskFramework::initializeBanditsLogging()
{
    if (bandits_logging_enabled_)
    {
        const std::string log_folder = GetLogFolder(nh_);
        ROS_INFO_STREAM_NAMED("task_framework", "Logging to " << log_folder);

        Log::Log seed_log(log_folder + "seed.txt", false);
        LOG_STREAM(seed_log, std::hex << seed_);

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "time",
                            Log::Log(log_folder + "time.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "error",
                            Log::Log(log_folder + "error.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "utility_mean",
                            Log::Log(log_folder + "utility_mean.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "utility_covariance",
                            Log::Log(log_folder + "utility_covariance.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "model_chosen",
                            Log::Log(log_folder + "model_chosen.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "rewards_for_all_models",
                            Log::Log(log_folder + "rewards_for_all_models.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "grippers_distance_delta_history",
                            Log::Log(log_folder + "grippers_distance_delta_history.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "error_delta_history",
                            Log::Log(log_folder + "error_delta_history.txt", false)));
    }
}

void TaskFramework::initializeControllerLogging()
{
    if(controller_logging_enabled_)
    {
        const std::string log_folder = GetLogFolder(nh_);
        ROS_INFO_STREAM_NAMED("task_framework", "Logging to " << log_folder);

        Log::Log seed_log(log_folder + "seed.txt", false);
        LOG_STREAM(seed_log, std::hex << seed_);

        controller_loggers_.insert(std::make_pair<std::string, Log::Log>(
                                       "control_time",
                                       Log::Log(log_folder + "control_time.txt", false)));

        controller_loggers_.insert(std::make_pair<std::string, Log::Log>(
                                       "control_error_realtime",
                                       Log::Log(log_folder + "control_error_realtime.txt", false)));

        controller_loggers_.insert(std::make_pair<std::string, Log::Log>(
                                       "realtime_stretching_factor",
                                       Log::Log(log_folder + "realtime_stretching_factor.txt", false)));

        controller_loggers_.insert(std::make_pair<std::string, Log::Log>(
                                       "individual_computation_times",
                                       Log::Log(log_folder + "individual_computation_times.txt", false)));

        controller_loggers_.insert(std::make_pair<std::string, Log::Log>(
                                       "model_prediction_error_weighted",
                                       Log::Log(log_folder + "model_prediction_error_weighted.txt", false)));

        controller_loggers_.insert(std::make_pair<std::string, Log::Log>(
                                       "model_prediction_error_unweighted",
                                       Log::Log(log_folder + "model_prediction_error_unweighted.txt", false)));
    }
}

// Note that resulting_world_state may not be exactly indentical to individual_model_rewards[model_used]
// because of the way forking works (and doesn't) in Bullet. They should be very close however.
void TaskFramework::logBanditsData(
        const WorldState& initial_world_state,
        const WorldState& resulting_world_state,
        const std::vector<WorldState>& individual_model_results,
        const VectorXd& model_utility_mean,
        const MatrixXd& model_utility_covariance,
        const ssize_t model_used)
{
    if (bandits_logging_enabled_)
    {
        std::vector<double> rewards_for_all_models(num_models_, std::numeric_limits<double>::quiet_NaN());
        if (collect_results_for_all_models_)
        {
            const double prev_error = task_specification_->calculateError(initial_world_state);
            for (ssize_t model_ind = 0; model_ind < num_models_; ++model_ind)
            {
                const double current_error = task_specification_->calculateError(individual_model_results[(size_t)model_ind]);
                rewards_for_all_models[(size_t)model_ind] = prev_error - current_error;
            }
        }

        const static IOFormat single_line(
                    StreamPrecision,
                    DontAlignCols,
                    " ", " ", "", "");

        LOG(loggers_.at("time"),
             resulting_world_state.sim_time_);

        LOG(loggers_.at("error"),
             task_specification_->calculateError(resulting_world_state));

        LOG(loggers_.at("utility_mean"),
             model_utility_mean.format(single_line));

        LOG(loggers_.at("utility_covariance"),
             model_utility_covariance.format(single_line));

        LOG(loggers_.at("model_chosen"),
             model_used);

        LOG(loggers_.at("rewards_for_all_models"),
            PrettyPrint::PrettyPrint(rewards_for_all_models, false, " "));
    }
}

// Note that resulting_world_state may not be exactly indentical to individual_model_rewards[model_used]
// because of the way forking works (and doesn't) in Bullet. They should be very close however.
void TaskFramework::controllerLogData(
        const WorldState& initial_world_state,
        const WorldState& resulting_world_state,
        const std::vector<WorldState>& individual_model_results,
        const DeformableController::InputData& input_data,
        const std::vector<double>& individual_computation_times,
        const std::vector<double>& model_prediction_errors_weighted,
        const std::vector<double>& model_prediction_errors_unweighted)
{
    if (controller_logging_enabled_)
    {
        // This function only works properly if we've collected all the data for each model,
        // so make sure the code crashes at a known point if that's the case
        assert(collect_results_for_all_models_);

        // Split out data used for computation for each model
        const ObjectDeltaAndWeight& task_desired_error_correction = input_data.desired_object_motion_.error_correction_;
        const VectorXd& desired_p_dot = task_desired_error_correction.delta;
        const VectorXd& desired_p_dot_weight = task_desired_error_correction.weight;
        const ssize_t num_grippers = initial_world_state.all_grippers_single_pose_.size();
        const ssize_t num_nodes = initial_world_state.object_configuration_.cols();

        // Data used by the function, per model
        std::vector<double> avg_control_error(num_models_, 0.0);
        std::vector<double> current_stretching_factor(num_models_, 0.0);

        for (size_t model_ind = 0; (ssize_t)model_ind < num_models_; ++model_ind)
        {
            // Get control errors for different model-controller sets.
            const ObjectPointSet real_p_dot =
                    individual_model_results[model_ind].object_configuration_
                    - initial_world_state.object_configuration_;

            int point_count = 0;
            double max_stretching = 0.0;
            double desired_p_dot_avg_norm = 0.0;
            double desired_p_dot_max = 0.0;

            // Calculate stretching factor
            const MatrixXd node_distance =
                    CalculateDistanceMatrix(individual_model_results[model_ind].object_configuration_);

            for (ssize_t node_ind = 0; node_ind < num_nodes; node_ind++)
            {
                const double point_weight = desired_p_dot_weight(node_ind * 3);
                if (point_weight > 0.0)
                {
                    //  Calculate p_dot error
                    const Vector3d& point_real_p_dot = real_p_dot.col(node_ind);
                    const Vector3d& point_desired_p_dot = desired_p_dot.segment<3>(node_ind * 3);
                    avg_control_error[model_ind] += (point_real_p_dot - point_desired_p_dot).norm();

                    desired_p_dot_avg_norm += point_desired_p_dot.norm();
                    if (point_desired_p_dot.norm() > desired_p_dot_max)
                    {
                        desired_p_dot_max = point_desired_p_dot.norm();
                    }

                    point_count++;
                }

                const ssize_t first_node = node_ind;
                for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
                {
                    const double this_stretching_factor = node_distance(first_node, second_node)
                            / object_initial_node_distance_(first_node, second_node);
                    max_stretching = std::max(max_stretching, this_stretching_factor);
                }
            }

            if (point_count > 0)
            {
                avg_control_error[model_ind] = avg_control_error[model_ind] / point_count;
                desired_p_dot_avg_norm /= point_count;
            }

            // Catch cases where the grippers and the nodes don't align, this should still be flagged as large stretch
            if (num_grippers == 2)
            {
                const auto gripper_delta =
                        individual_model_results[model_ind].all_grippers_single_pose_.at(0).translation()
                        - individual_model_results[model_ind].all_grippers_single_pose_.at(1).translation();
                const double this_stretching_factor = gripper_delta.norm() / initial_grippers_distance_;
                max_stretching = std::max(max_stretching, this_stretching_factor);
            }
            current_stretching_factor[model_ind] = max_stretching;
            ROS_INFO_STREAM_NAMED("task_framework", "average desired p dot is       " << desired_p_dot_avg_norm);
            ROS_INFO_STREAM_NAMED("task_framework", "max pointwise desired p dot is " << desired_p_dot_max);
        }

        // Do the actual logging itself
        const static IOFormat single_line(
                    StreamPrecision,
                    DontAlignCols,
                    " ", " ", "", "");

        LOG(controller_loggers_.at("control_time"),
            resulting_world_state.sim_time_);

        LOG(controller_loggers_.at("control_error_realtime"),
            PrettyPrint::PrettyPrint(avg_control_error, false, " "));

        LOG(controller_loggers_.at("realtime_stretching_factor"),
            PrettyPrint::PrettyPrint(current_stretching_factor, false, " "));

        LOG(controller_loggers_.at("individual_computation_times"),
            PrettyPrint::PrettyPrint(individual_computation_times, false, " "));

        LOG(controller_loggers_.at("model_prediction_error_weighted"),
            PrettyPrint::PrettyPrint(model_prediction_errors_weighted, false, " "));

        LOG(controller_loggers_.at("model_prediction_error_unweighted"),
            PrettyPrint::PrettyPrint(model_prediction_errors_unweighted, false, " "));
    }
}




void TaskFramework::storeWorldState(const WorldState& world_state, const RubberBand::Ptr band)
{
    try
    {
        const auto log_folder = GetLogFolder(nh_);
        arc_utilities::CreateDirectory(log_folder);
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "world_state_file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load world_state_file_name_prefix from parameter server");
        }

        const std::string file_name_suffix = arc_helpers::GetCurrentTimeAsStringWithMilliseconds(); //arc_helpers::GetCurrentTimeAsString();
        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix + ".compressed";
        const std::string full_path = log_folder + file_name;
        ROS_DEBUG_STREAM("Saving world_state to " << full_path);

        std::vector<uint8_t> buffer;
        world_state.serialize(buffer);
        band->serialize(buffer);
        ZlibHelpers::CompressAndWriteToFile(buffer, full_path);

        const auto deserialized_results = WorldState::Deserialize(buffer, 0);
        assert(deserialized_results.first == world_state);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to store world_state: "  <<  e.what());
    }
}

std::pair<WorldState, RubberBand::Ptr> TaskFramework::loadStoredWorldState()
{
    std::pair<WorldState, RubberBand::Ptr> deserialized_result;

    try
    {
        const auto log_folder = GetLogFolder(nh_);
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "world_state_file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load band_file_name_prefix from parameter server");
        }
        const auto file_name_suffix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "world_state_file_name_suffix_to_load", __func__);
        if (!file_name_suffix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load world_state_file_name_suffix_to_load from parameter server");
        }

        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix.GetImmutable() + ".compressed";
        const std::string full_path = log_folder + file_name;
        ROS_INFO_STREAM("Loading world state from " << full_path);

        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(full_path);
        const auto deserialized_world_state = WorldState::Deserialize(buffer, 0);
        deserialized_result.first = deserialized_world_state.first;

        deserialized_result.second = std::make_shared<RubberBand>(*rubber_band_between_grippers_);
        deserialized_result.second->deserializeIntoSelf(buffer, deserialized_world_state.second);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to load stored world_state: "  <<  e.what());
    }

    return deserialized_result;
}

bool TaskFramework::useStoredWorldState() const
{
    return ROSHelpers::GetParamDebugLog<bool>(ph_, "use_stored_world_state", false);
}
