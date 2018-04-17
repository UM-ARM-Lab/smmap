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
#include <arc_utilities/timing.hpp>
#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/zlib_helpers.hpp>

#include "smmap/diminishing_rigidity_model.h"
#include "smmap/adaptive_jacobian_model.h"
#include "smmap/least_squares_jacobian_model.h"
#include "smmap/constraint_jacobian_model.h"

#include "smmap/least_squares_controller_with_object_avoidance.h"
#include "smmap/stretching_avoidance_controller.h"
#include "smmap/straight_line_controller.h"

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;
using namespace Eigen;
using namespace EigenHelpers;
using namespace EigenHelpersConversions;
using ColorBuilder = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>;

#pragma message "Magic number - reward scaling factor starting value"
#define REWARD_STANDARD_DEV_SCALING_FACTOR_START (1.0)

//#define ENABLE_LOCAL_CONTROLLER_LOAD_SAVE 1
#define ENABLE_LOCAL_CONTROLLER_LOAD_SAVE 0

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
Planner::Planner(ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const RobotInterface::Ptr& robot,
        Visualizer::Ptr vis,
        const std::shared_ptr<TaskSpecification>& task_specification)
    // Robot and task parameters
    : nh_(nh)
    , ph_(ph)
    , seed_(GetPlannerSeed(ph_))
    , generator_(seed_)
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
    , executing_global_gripper_trajectory_(false)
    , global_plan_current_timestep_(-1)
    , global_plan_gripper_trajectory_(0)
    , rrt_helper_(nullptr)
    , prm_helper_(nullptr)
    // Used to generate some log data by some controllers
    , object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh_)))
    , initial_grippers_distance_(robot_->getGrippersInitialDistance())
    // Logging and visualization parameters
    , planner_logging_enabled_(GetPlannerLoggingEnabled(ph_))
    , controller_logging_enabled_(GetControllerLoggingEnabled(ph_))
    , vis_(vis)
    , visualize_desired_motion_(!GetDisableAllVisualizations(ph) && GetVisualizeObjectDesiredMotion(ph_))
    , visualize_gripper_motion_(!GetDisableAllVisualizations(ph) && GetVisualizerGripperMotion(ph_))
    , visualize_predicted_motion_(!GetDisableAllVisualizations(ph) && GetVisualizeObjectPredictedMotion(ph_))
    , visualize_free_space_graph_(!GetDisableAllVisualizations(ph) && GetVisualizeFreeSpaceGraph(ph_))
{
    ROS_INFO_STREAM_NAMED("planner", "Using seed " << std::hex << seed_ );
    initializePlannerLogging();
    initializeControllerLogging();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The one function that gets invoked externally
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Planner::execute()
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
        const double max_band_distance =
                (grippers_starting_poses[0].translation() - grippers_starting_poses[1].translation()).norm()
                * dijkstras_task_->maxStretchFactor();

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
                    max_band_distance,
                    dijkstras_task_,
                    vis_,
                    generator_);

        const Vector3d task_frame_lower_limits = Vector3d(
                    GetRRTPlanningXMinBulletFrame(ph_),
                    GetRRTPlanningYMinBulletFrame(ph_),
                    GetRRTPlanningZMinBulletFrame(ph_));

        const Vector3d task_frame_upper_limits = Vector3d(
                    GetRRTPlanningXMaxBulletFrame(ph_),
                    GetRRTPlanningYMaxBulletFrame(ph_),
                    GetRRTPlanningZMaxBulletFrame(ph_));

        prm_helper_ = std::make_shared<PRMHelper>(
                    dijkstras_task_->environment_sdf_,
                    vis_,
                    generator_,
                    robot_->getWorldToTaskFrameTf(),
                    task_frame_lower_limits,
                    task_frame_upper_limits,
                    !GetDisableAllVisualizations(ph_),
                    GetPRMNumNearest(ph_),
                    GetPRMNumSamples(ph_),
                    dijkstras_task_->work_space_grid_.minStepDimension());
        prm_helper_->initializeRoadmap();
        prm_helper_->visualize(GetVisualizePRM(ph_));

        // Pass in all the config values that the RRT needs; for example goal bias, step size, etc.
        rrt_helper_ = std::unique_ptr<RRTHelper>(
                    new RRTHelper(
                        robot_,
                        dijkstras_task_->environment_sdf_,
                        vis_,
                        generator_,
                        prm_helper_,
                        robot_->getWorldToTaskFrameTf(),
                        task_frame_lower_limits,
                        task_frame_upper_limits,
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        GetRRTMaxRobotDOFStepSize(ph_),
                        GetRRTMinRobotDOFStepSize(ph_),
                        GetRRTMaxGripperRotation(ph_),
                        GetRRTGoalBias(ph_),
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        GetRRTMinGripperDistanceToObstacles(ph_),
                        GetRRTHomotopyDistancePenalty(),
                        GetRRTMaxShortcutIndexDistance(ph_),
                        GetRRTMaxSmoothingIterations(ph_),
                        GetRRTMaxFailedSmoothingIterations(ph_),
                        !GetDisableAllVisualizations(ph_)));
    }

    if (visualize_free_space_graph_ && dijkstras_task_ != nullptr)
    {
        dijkstras_task_->visualizeFreeSpaceGraph();
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
            ROS_INFO_NAMED("planner", "------------------------------- End of Task -------------------------------------------");
            const double current_error = task_specification_->calculateError(world_feedback);
            ROS_INFO_STREAM_NAMED("planner", "   Planner/Task sim time " << world_feedback.sim_time_ << "\t Error: " << current_error);

            vis_->deleteObjects(PROJECTED_BAND_NS, 1, 30);
            vis_->clearVisualizationsBullet();
            std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
            vis_->clearVisualizationsBullet();
            std::this_thread::sleep_for(std::chrono::duration<double>(5.0));

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
WorldState Planner::sendNextCommand(
        const WorldState& world_state)
{
    ROS_INFO_NAMED("planner", "---------------------------- Start of Loop -----------------------------------------");
    const double current_error = task_specification_->calculateError(world_state);
    ROS_INFO_STREAM_NAMED("planner", "Planner/Task sim time " << world_state.sim_time_ << "\t Error: " << current_error);

    if (enable_stuck_detection_)
    {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // First, check if we need to (re)plan
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        bool planning_needed = false;

        // Check if the global plan has 'hooked' the deformable object on something
        if (executing_global_gripper_trajectory_)
        {
            Stopwatch stopwatch;
            const bool global_plan_will_overstretch = predictStuckForGlobalPlannerResults();
            ROS_INFO_STREAM_NAMED("planner", "Determined if global planner needed in          " << stopwatch(READ) << " seconds");

            if (global_plan_will_overstretch)
            {
                planning_needed = true;

                vis_->deleteObjects(PROJECTED_GRIPPER_NS,            1, (int32_t)(4 * max_lookahead_steps_) + 10);
                vis_->deleteObjects(PROJECTED_BAND_NS,               1, (int32_t)max_lookahead_steps_ + 10);

                ROS_WARN_NAMED("planner", "Invoking global planner as the current plan will overstretch the deformable object");
                ROS_INFO_NAMED("planner", "----------------------------------------------------------------------------");
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
            ROS_INFO_STREAM_NAMED("planner", "Determined if global planner needed in " << stopwatch(READ) << " seconds");

            if (global_planner_needed_due_to_overstretch || global_planner_needed_due_to_lack_of_progress)
            {
                planning_needed = true;

                std::this_thread::sleep_for(std::chrono::duration<double>(5.0));

//                vis_->deleteObjects(DESIRED_DELTA_NS, 1, 100);
//                vis_->deleteObjects(PROJECTED_GRIPPER_NS,            1, (int32_t)(4 * max_lookahead_steps_) + 10);
//                vis_->deleteObjects(PROJECTED_BAND_NS,               1, (int32_t)max_lookahead_steps_ + 10);
//                vis_->deleteObjects(PROJECTED_POINT_PATH_NS,         1, 2);
//                vis_->deleteObjects(PROJECTED_POINT_PATH_LINES_NS,   1, 2);

                vis_->clearVisualizationsBullet();
                std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
                vis_->clearVisualizationsBullet();

                ROS_WARN_COND_NAMED(global_planner_needed_due_to_overstretch, "planner", "Invoking global planner due to overstretch");
                ROS_WARN_COND_NAMED(global_planner_needed_due_to_lack_of_progress, "planner", "Invoking global planner due to collision");
                ROS_INFO_NAMED("planner", "----------------------------------------------------------------------------");
            }
        }

        // If we need to (re)plan due to the local controller getting stuck, or the gobal plan failing, then do so
        if (planning_needed)
        {
            rrt_helper_->addBandToBlacklist(rubber_band_between_grippers_->getVectorRepresentation());
            planGlobalGripperTrajectory(world_state);
        }

        // Execute a single step in the global plan, or use the local controller if we have no plan to follow
        WorldState world_feedback;
        if (executing_global_gripper_trajectory_)
        {
            world_feedback = sendNextCommandUsingGlobalGripperPlannerResults(world_state);
        }
        else
        {
            world_feedback = sendNextCommandUsingLocalController(world_state);
        }

        // Update the band with the new position of the deformable object
        const auto band_points = getPathBetweenGrippersThroughObject(world_feedback, path_between_grippers_through_object_);
        rubber_band_between_grippers_->setPointsAndSmooth(band_points);

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
        ROS_WARN_NAMED("planner", "Unable to do future constraint violation detection");
        return sendNextCommandUsingLocalController(world_state);
    }
}

/**
 * @brief Planner::sendNextCommandUsingLocalController
 * @param world_state
 * @return
 */
WorldState Planner::sendNextCommandUsingLocalController(
        WorldState world_state)
{
#if ENABLE_LOCAL_CONTROLLER_LOAD_SAVE
    if (useStoredWorldState())
    {
        loadStoredWorldState(world_state);
    }
    else
    {
        storeWorldState(world_state);
    }
#endif

    Stopwatch stopwatch;
    Stopwatch function_wide_stopwatch;

    vis_->visualizeCloth("input_deformable_object", world_state.object_configuration_, Visualizer::Green(0.5), 1);
    vis_->visualizeCloth("input_deformable_object", world_state.object_configuration_, Visualizer::Green(0.5), 1);
    vis_->visualizeCloth("input_deformable_object", world_state.object_configuration_, Visualizer::Green(0.5), 1);
    vis_->visualizeCloth("input_deformable_object", world_state.object_configuration_, Visualizer::Green(0.5), 1);
    vis_->visualizeCloth("input_deformable_object", world_state.object_configuration_, Visualizer::Green(0.5), 1);

    // Temporaries needed here bercause model_input_data takes things by reference
    const DesiredDirection desired_object_manipulation_direction = task_specification_->calculateDesiredDirection(world_state);
    const MatrixXd robot_dof_to_grippers_poses_jacobian = robot_->getGrippersJacobian(world_state.robot_configuration_);
    // Build the constraints for the gippers and other points of interest on the robot - includes the grippers
    const std::vector<std::pair<CollisionData, Matrix3Xd>> poi_collision_data_ =
            robot_->getPointsOfInterestCollisionData(world_state.robot_configuration_);
    const DeformableController::InputData model_input_data(
                world_state,
                desired_object_manipulation_direction,
                robot_,
                robot_dof_to_grippers_poses_jacobian,
                world_state.robot_configuration_valid_,
                poi_collision_data_);

    if (visualize_desired_motion_)
    {
        visualizeDesiredMotion(world_state, model_input_data.desired_object_motion_.error_correction_);
//        std::this_thread::sleep_for(std::chrono::duration<double>(2.0));
    }

    // Pick an arm to use





    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);






    const bool get_action_for_all_models = model_utility_bandit_.generateAllModelActions();
    ROS_INFO_STREAM_COND_NAMED(num_models_ > 1, "planner", "Using model index " << model_to_use);

    // Querry each model for it's best gripper delta
    ROS_INFO_STREAM_NAMED("planner", "Generating model suggestions");
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
            ROS_INFO_STREAM_NAMED("planner", model_ind << "th Controller get suggested motion in          " << controller_computation_time[model_ind] << " seconds");
        }
    }
    // Measure the time it took to pick a model
    const DeformableController::OutputData& selected_command = suggested_robot_commands[(size_t)model_to_use];
    ROS_INFO_STREAM_NAMED("planner", "Calculated model suggestions and picked one in  " << stopwatch(READ) << " seconds");
    if (world_state.robot_configuration_valid_)
    {
        ROS_INFO_STREAM_NAMED("planner", "Robot DOF motion: " << selected_command.robot_dof_motion_.transpose());
    }
    for (size_t ind = 0; ind < selected_command.grippers_motion_.size(); ++ind)
    {
        ROS_INFO_STREAM_NAMED("planner", "Gripper " << ind << " motion: " << selected_command.grippers_motion_[ind].transpose());
    }
    ROS_INFO_STREAM_NAMED("planner", "Selected command gripper action norm:  " << MultipleGrippersVelocity6dNorm(selected_command.grippers_motion_));

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
        std::vector<Eigen::VectorXd> configurations_to_test(num_models_);
        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            poses_to_test[model_ind] = kinematics::applyTwist(
                        world_state.all_grippers_single_pose_, suggested_robot_commands[model_ind].grippers_motion_);

            configurations_to_test[model_ind] =
                    world_state.robot_configuration_ + suggested_robot_commands[model_ind].robot_dof_motion_;
        }
        robot_->testRobotMotion(poses_to_test, configurations_to_test, world_state.robot_configuration_valid_, test_feedback_fn);

        ROS_INFO_STREAM_NAMED("planner", "Collected data to calculate regret in " << stopwatch(READ) << " seconds");
    }





    if (visualize_gripper_motion_)
    {
        ROS_WARN_THROTTLE_NAMED(1.0, "planner", "Asked to visualize grippper motion but this is disabled");

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
    ROS_INFO_STREAM_NAMED("planner", "Sending command to robot");
    const auto all_grippers_single_pose = kinematics::applyTwist(world_state.all_grippers_single_pose_, selected_command.grippers_motion_);
    const auto robot_configuration = world_state.robot_configuration_ + selected_command.robot_dof_motion_;
    // Measure execution time
    stopwatch(RESET);
    arc_helpers::DoNotOptimize(all_grippers_single_pose);
    const WorldState world_feedback = robot_->commandRobotMotion(
                all_grippers_single_pose,
                robot_configuration,
                world_state.robot_configuration_valid_);
    arc_helpers::DoNotOptimize(world_feedback);
    const double robot_execution_time = stopwatch(READ);



    if (visualize_predicted_motion_)
    {
        const ObjectPointSet& object_delta_constrained = suggested_robot_commands[0].object_motion_;
        const ObjectPointSet& object_delta_diminishing = suggested_robot_commands[1].object_motion_;


        vis_->visualizeObjectDelta(
                    "constraint_model_prediction",
                    world_state.object_configuration_,
                    world_state.object_configuration_ + 50.0 * object_delta_constrained,
                    Visualizer::Cyan());

        vis_->visualizeObjectDelta(
                    "diminishing_model_prediction",
                    world_state.object_configuration_,
                    world_state.object_configuration_ + 50.0 * object_delta_diminishing,
                    Visualizer::Red(0.3f));

        const ObjectPointSet true_object_delta = world_feedback.object_configuration_ - world_state.object_configuration_;

        vis_->visualizeObjectDelta(
                    "true_object_delta",
                    world_state.object_configuration_,
                    world_state.object_configuration_ + 50.0 * true_object_delta,
                    Visualizer::Green());

//        task_specification_->visualizeDeformableObject(
//                PREDICTED_DELTA_NS,
//                world_state.object_configuration_ + object_delta,
//                Visualizer::Blue());
    }


    #warning "!!!!!!!!!!!!!!! This data collection is only valid if the robot took the actual action requested!!!!!!!!!!!!!!!!!"
    std::vector<double> model_prediction_errors_weighted(model_list_.size(), 0.0);
    std::vector<double> model_prediction_errors_unweighted(model_list_.size(), 0.0);
    if (collect_results_for_all_models_)
    {
        ROS_INFO_NAMED("planner", "Calculating model predictions based on real motion taken");

        const ObjectPointSet true_object_delta = world_feedback.object_configuration_ - world_state.object_configuration_;

        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            const ObjectPointSet predicted_delta = model_list_[model_ind]->getObjectDelta(world_state, selected_command.grippers_motion_);
            const ObjectPointSet prediction_error_sq = (predicted_delta - true_object_delta).cwiseAbs2();

            const Map<const VectorXd> error_sq_as_vector(prediction_error_sq.data(), prediction_error_sq.size());
            model_prediction_errors_weighted[model_ind] = error_sq_as_vector.dot(desired_object_manipulation_direction.error_correction_.weight);

            model_prediction_errors_unweighted[model_ind] = prediction_error_sq.sum();
        }
    }


    ROS_INFO_NAMED("planner", "Updating models");
    updateModels(world_state, model_input_data.desired_object_motion_.error_correction_, suggested_robot_commands, model_to_use, world_feedback);

    const double controller_time = function_wide_stopwatch(READ) - robot_execution_time;
    ROS_INFO_STREAM_NAMED("planner", "Total local controller time                     " << controller_time << " seconds");

    ROS_INFO_NAMED("planner", "Logging data");
    logPlannerData(world_state, world_feedback, individual_model_results, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), model_to_use);
    controllerLogData(world_state, world_feedback, individual_model_results, model_input_data, controller_computation_time, model_prediction_errors_weighted, model_prediction_errors_unweighted);

    return world_feedback;
}

/**
 * @brief Planner::sendNextCommandUsingGlobalGripperPlannerResults
 * @param current_world_state
 * @return
 */
WorldState Planner::sendNextCommandUsingGlobalGripperPlannerResults(
        const WorldState& current_world_state)
{
    assert(executing_global_gripper_trajectory_);
    assert(global_plan_current_timestep_ < global_plan_gripper_trajectory_.size());

    assert(false && "Not implemented yet - needs refactor for generic robot");
    const auto world_feedback = current_world_state;
//    const WorldState world_feedback = robot_->commandRobotMotion(global_plan_gripper_trajectory_[global_plan_current_timestep_]);

    ++global_plan_current_timestep_;
    if (global_plan_current_timestep_ == global_plan_gripper_trajectory_.size())
    {
        ROS_INFO_NAMED("planner", "Global plan finished, resetting grippers pose history and error history");

        executing_global_gripper_trajectory_ = false;
        grippers_pose_history_.clear();
        error_history_.clear();

        std::this_thread::sleep_for(std::chrono::duration<double>(5.0));
        vis_->clearVisualizationsBullet();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
        vis_->clearVisualizationsBullet();

//        vis_->deleteObjects(RRTHelper::RRT_SOLUTION_GRIPPER_A_NS,   1, 2);
//        vis_->deleteObjects(RRTHelper::RRT_SOLUTION_GRIPPER_B_NS,   1, 2);
//        vis_->deleteObjects(RRTHelper::RRT_SOLUTION_RUBBER_BAND_NS, 1, 2);
    }

    const std::vector<WorldState> fake_all_models_results(num_models_, world_feedback);
    logPlannerData(current_world_state, world_feedback, fake_all_models_results, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), -1);

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

/**
 * @brief Planner::checkForClothStretchingViolations
 * @param projected_paths
 * @return
 */
bool Planner::checkForClothStretchingViolations(
        const std::vector<VectorVector3d>& projected_paths,
        const bool visualization_enabled)
{
    assert(false && "This function has not been checked/updated in a long time, and may no longer be valid/useful");

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
        vis_->visualizeLines(CONSTRAINT_VIOLATION_VERSION1_NS, vis_start_points, vis_end_points, Visualizer::Blue());
    }

    return violations_exist;
}

std::pair<std::vector<VectorVector3d>, std::vector<RubberBand>> Planner::detectFutureConstraintViolations(
        const WorldState& starting_world_state,
        const bool visualization_enabled)
{
    Stopwatch stopwatch;
    Stopwatch function_wide_stopwatch;

    assert(task_specification_->is_dijkstras_type_task_ && starting_world_state.all_grippers_single_pose_.size() == 2);
    std::pair<std::vector<VectorVector3d>, std::vector<RubberBand>> projected_deformable_point_paths_and_projected_virtual_rubber_bands;

    // TODO: Move to class wide location, currently in 2 locations in this file
    const static std_msgs::ColorRGBA gripper_color = ColorBuilder::MakeFromFloatColors(0.0f, 0.0f, 0.6f, 1.0f);
    const static std_msgs::ColorRGBA rubber_band_safe_color = Visualizer::Black();
    const static std_msgs::ColorRGBA rubber_band_violation_color = Visualizer::Cyan();
    const bool band_verbose = false;

    vis_->clearVisualizationsBullet();

    vis_->deleteObjects(PROJECTED_BAND_NS, 1, (int32_t)max_lookahead_steps_ + 10);
    vis_->deleteObjects(PROJECTED_POINT_PATH_NS, 1, 2);
    vis_->deleteObjects(PROJECTED_POINT_PATH_LINES_NS, 1, 2);
    vis_->deleteObjects(PROJECTED_GRIPPER_NS, 1, (int32_t)(4 * max_lookahead_steps_) + 10);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constraint violation Version 1 - Purely cloth overstretch
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    stopwatch(RESET);
    const std::vector<VectorVector3d> projected_deformable_point_paths = dijkstras_task_->findPathFromObjectToTarget(starting_world_state, max_lookahead_steps_);

    const size_t actual_lookahead_steps = sizeOfLargestVector(projected_deformable_point_paths) - 1;
    // sizeOfLargest(...) should be at least 2, so this assert should always be true
    assert(actual_lookahead_steps <= max_lookahead_steps_);

    ROS_INFO_STREAM_NAMED("planner", "Calculated projected cloth paths                 - Version 1 - in " << stopwatch(READ) << " seconds");
    ROS_INFO_STREAM_NAMED("planner", "Max lookahead steps: " << max_lookahead_steps_ << " Actual steps: " << actual_lookahead_steps);
    visualizeProjectedPaths(projected_deformable_point_paths, visualization_enabled);
    projected_deformable_point_paths_and_projected_virtual_rubber_bands.first = projected_deformable_point_paths;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constraint violation Version 2a - Vector field forward "simulation" - rubber band
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ROS_INFO_STREAM_NAMED("planner", "Starting future constraint violation detection   - Version 2a - Total steps taken " << actual_lookahead_steps);
    assert(num_models_ == 1 && starting_world_state.all_grippers_single_pose_.size() == 2);
    const auto& correspondences = dijkstras_task_->getCoverPointCorrespondences(starting_world_state);


    vis_->clearVisualizationsBullet();

    rubber_band_between_grippers_->visualize(PROJECTED_BAND_NS, rubber_band_safe_color, rubber_band_violation_color, 1, visualization_enabled);

    vis_->deleteObjects(PROJECTED_POINT_PATH_NS, 1, 2);
    vis_->deleteObjects(PROJECTED_POINT_PATH_LINES_NS, 1, 2);
    vis_->deleteObjects(PROJECTED_GRIPPER_NS, 1, (int32_t)(4 * max_lookahead_steps_) + 10);




    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    WorldState world_state_copy = starting_world_state;
    RubberBand rubber_band_between_grippers_copy = *rubber_band_between_grippers_.get();

    projected_deformable_point_paths_and_projected_virtual_rubber_bands.second.reserve(actual_lookahead_steps);
    for (size_t t = 0; t < actual_lookahead_steps; ++t)
    {
        // We only want the error correction part of the movement
        DesiredDirection desired_object_manipulation_direction;
        desired_object_manipulation_direction.error_correction_ =
                dijkstras_task_->calculateErrorCorrectionDeltaFixedCorrespondences(world_state_copy, correspondences.correspondences_);
        desired_object_manipulation_direction.stretching_correction_ = ObjectDeltaAndWeight(world_state_copy.object_configuration_.size());
        desired_object_manipulation_direction.combined_correction_ = desired_object_manipulation_direction.error_correction_;

        const MatrixXd robot_dof_to_grippers_poses_jacobian = robot_->getGrippersJacobian(world_state_copy.robot_configuration_);
        const std::vector<std::pair<CollisionData, Matrix3Xd>> poi_collision_data_ =
                robot_->getPointsOfInterestCollisionData(world_state_copy.robot_configuration_);
        const DeformableController::InputData model_input_data(
                    world_state_copy,
                    desired_object_manipulation_direction,
                    robot_,
                    robot_dof_to_grippers_poses_jacobian,
                    world_state_copy.robot_configuration_valid_,
                    poi_collision_data_);

        const DeformableController::OutputData robot_command = controller_list_[0]->getGripperMotion(model_input_data);

        /*
           Things to be updated in world_state_copy after "executing" a robot commad
                ObjectPointSet object_configuration_;
                AllGrippersSinglePose all_grippers_single_pose_;
                Eigen::VectorXd robot_configuration_;
                bool robot_configuration_valid_;
                std::vector<CollisionData> gripper_collision_data_;
                double sim_time_;
        */


        // Move the grippers forward
        world_state_copy.all_grippers_single_pose_
                = kinematics::applyTwist(world_state_copy.all_grippers_single_pose_, robot_command.grippers_motion_);
        for (auto& pose : world_state_copy.all_grippers_single_pose_)
        {
            pose.translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(pose.translation(), GetRobotGripperRadius());
        }

        // Update the gripper collision data
        auto collision_check_future = std::async(std::launch::async, &RobotInterface::checkGripperCollision, robot_, world_state_copy.all_grippers_single_pose_);

        // Move the robot DOF forward
        world_state_copy.robot_configuration_ += robot_command.robot_dof_motion_;

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
        rubber_band_between_grippers_copy.visualize(PROJECTED_BAND_NS, rubber_band_safe_color, rubber_band_violation_color, (int32_t)t + 2, visualization_enabled);
        vis_->visualizeGrippers(PROJECTED_GRIPPER_NS, world_state_copy.all_grippers_single_pose_, gripper_color, (int32_t)(4 * t) + 1);

        // Finish collecting the gripper collision data
        world_state_copy.gripper_collision_data_ = collision_check_future.get();
    }
    ROS_INFO_STREAM_NAMED("planner", "Calculated future constraint violation detection - Version 2a - in " << function_wide_stopwatch(READ) << " seconds");

    return projected_deformable_point_paths_and_projected_virtual_rubber_bands;
}

bool Planner::globalPlannerNeededDueToOverstretch(
        const WorldState& current_world_state)
{
    static double annealing_factor = GetRubberBandOverstretchPredictionAnnealingFactor(ph_);

    const auto detection_results = detectFutureConstraintViolations(current_world_state);
    const auto& projected_rubber_bands = detection_results.second;

    if (projected_rubber_bands.size() == 0)
    {
        return false;
    }

    double filtered_band_length = projected_rubber_bands.front().totalLength();

    for (size_t t = 0; t < projected_rubber_bands.size(); ++t)
    {
        const RubberBand& band = projected_rubber_bands[t];
        const double band_length = band.totalLength();
        const std::pair<Eigen::Vector3d, Eigen::Vector3d> endpoints = band.getEndpoints();
        const double distance_between_endpoints = (endpoints.first - endpoints.second).norm();

        // Apply a low pass filter to the band length to try and remove "blips" in the estimate
        filtered_band_length = annealing_factor * filtered_band_length + (1.0 - annealing_factor) * band_length;
        // If the band is currently overstretched, and not in free space, then predict future problems
        if (filtered_band_length > band.maxSafeLength() && !CloseEnough(band_length, distance_between_endpoints, 1e-6))
        {
            return true;
        }
    }

    return false;
}

bool Planner::globalPlannerNeededDueToLackOfProgress()
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

    if (planner_logging_enabled_)
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

bool Planner::predictStuckForGlobalPlannerResults(const bool visualization_enabled)
{
    assert(global_plan_current_timestep_ < global_plan_gripper_trajectory_.size());

    // TODO: Move to class wide location, currently in 2 positions in this file
    const static std_msgs::ColorRGBA gripper_color = ColorBuilder::MakeFromFloatColors(0.0f, 0.0f, 0.6f, 1.0f);
    const static std_msgs::ColorRGBA rubber_band_safe_color = Visualizer::Black();
    const static std_msgs::ColorRGBA rubber_band_violation_color = Visualizer::Cyan();
    constexpr bool band_verbose = false;

    RubberBand rubber_band_between_grippers_copy = *rubber_band_between_grippers_;

    bool overstretch_predicted = false;
    const size_t traj_waypoints_per_large_step = (size_t)std::floor(dijkstras_task_->work_space_grid_.minStepDimension() / robot_->dt_ / robot_->max_gripper_velocity_norm_);
    for (size_t t = 0; (t < max_lookahead_steps_) &&
                       (global_plan_current_timestep_ + t * traj_waypoints_per_large_step < global_plan_gripper_trajectory_.size()); ++t)
    {
        // Forward project the band and check for overstretch
        const auto& grippers_pose = global_plan_gripper_trajectory_[global_plan_current_timestep_ + t * traj_waypoints_per_large_step];
        rubber_band_between_grippers_copy.forwardPropagateRubberBandToEndpointTargets(
                    grippers_pose[0].translation(),
                    grippers_pose[1].translation(),
                    band_verbose);
        overstretch_predicted |= rubber_band_between_grippers_copy.isOverstretched();

        // Visualize
        rubber_band_between_grippers_copy.visualize(PROJECTED_BAND_NS, rubber_band_safe_color, rubber_band_violation_color, (int32_t)t + 2, visualization_enabled);
        vis_->visualizeGrippers(PROJECTED_GRIPPER_NS, grippers_pose, gripper_color, (int32_t)(4 * t) + 1);
    }

    return overstretch_predicted;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global gripper planner functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AllGrippersSinglePose Planner::getGripperTargets(const WorldState& world_state)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////// Determine the cluster centers /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const auto& correspondences = dijkstras_task_->getCoverPointCorrespondences(world_state);
    const auto& cover_point_indices_= correspondences.uncovered_target_points_idxs_;

    VectorVector3d cluster_targets;
    cluster_targets.reserve(cover_point_indices_.size());
    for (size_t idx = 0; idx < cover_point_indices_.size(); ++idx)
    {
        const ssize_t cover_idx = cover_point_indices_[idx];
        cluster_targets.push_back(dijkstras_task_->cover_points_.col(cover_idx));
    }

    vis_->visualizePoints(CLUSTERING_TARGETS_NS, cluster_targets, Visualizer::Blue(), 1);

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

//    vis_->visualizeCubes(CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS, {world_state.all_grippers_single_pose_[0].translation(), target_gripper_poses[0].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Magenta(), 1);
//    vis_->visualizeCubes(CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS, {world_state.all_grippers_single_pose_[1].translation(), target_gripper_poses[1].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Cyan(), 5);

    // Project the targets out of collision
    const double min_dist_to_obstacles = GetRRTMinGripperDistanceToObstacles(ph_) * GetRRTTargetMinDistanceScaleFactor(ph_);
    const Eigen::Vector3d gripper0_position_pre_project = target_gripper_poses[0].translation();
    const Eigen::Vector3d gripper1_position_pre_project = target_gripper_poses[1].translation();
    target_gripper_poses[0].translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(gripper0_position_pre_project, min_dist_to_obstacles);
    target_gripper_poses[1].translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(gripper1_position_pre_project, min_dist_to_obstacles);

    vis_->visualizeCubes(CLUSTERING_RESULTS_POST_PROJECT_NS, {target_gripper_poses[0].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Magenta(0.5), 1);
    vis_->visualizeCubes(CLUSTERING_RESULTS_POST_PROJECT_NS, {target_gripper_poses[1].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Orange(0.5), 5);

    return target_gripper_poses;
}

void Planner::planGlobalGripperTrajectory(const WorldState& world_state)
{
    static int num_times_invoked = 0;
    num_times_invoked++;

    if (GetRRTReuseOldResults(ph_))
    {
        // Deserialization
        try
        {
            Stopwatch stopwatch;
            ROS_INFO_NAMED("rrt_planner_results", "Checking if RRT solution already exists");
            const std::string rrt_file_path = GetLogFolder(nh_) + "rrt_cache_step." + PrettyPrint::PrettyPrint(num_times_invoked);
            std::ifstream prev_rrt_result(rrt_file_path, std::ios::binary | std::ios::in | std::ios::ate);
            if (!prev_rrt_result.is_open())
            {
                throw_arc_exception(std::runtime_error, "Couldn't open file");
            }

            ROS_INFO_NAMED("rrt_planner_results", "Reading contents of file");
            std::streamsize size = prev_rrt_result.tellg();
            prev_rrt_result.seekg(0, std::ios::beg);
            std::vector<uint8_t> file_buffer((size_t)size);
            if (!(prev_rrt_result.read(reinterpret_cast<char*>(file_buffer.data()), size)))
            {
                throw_arc_exception(std::runtime_error, "Unable to read entire contents of file");
            }
            const std::vector<uint8_t> decompressed_rrt_results = ZlibHelpers::DecompressBytes(file_buffer);

            const auto deserialized_results = DeserializeAllGrippersPoseTrajectory(decompressed_rrt_results, 0);
            const auto deserialized_bytes_read = deserialized_results.second;
            assert(deserialized_bytes_read == decompressed_rrt_results.size());

            global_plan_current_timestep_ = 0;
            executing_global_gripper_trajectory_ = true;
            global_plan_gripper_trajectory_ = deserialized_results.first;

            ROS_INFO_STREAM_NAMED("rrt_planner_results", "Read RRT solutions in " << stopwatch(READ) << " seconds");
            return;
        }
        catch (...)
        {
            ROS_ERROR_NAMED("rrt_planner_results", "Loading RRT results from file failed");
        }
    }

    // Planning if we did not load a plan from file
    {
        vis_->clearVisualizationsBullet();

        RRTGrippersRepresentation gripper_config(
                    world_state.all_grippers_single_pose_[0].translation(),
                    world_state.all_grippers_single_pose_[1].translation());

        RRTRobotRepresentation robot_config;
        if (world_state.robot_configuration_valid_)
        {
            robot_config.first = world_state.robot_configuration_.head<7>();
            robot_config.second = world_state.robot_configuration_.tail<7>();
        }

        RRTConfig start_config(
                    gripper_config,
                    robot_config,
                    0,
                    *rubber_band_between_grippers_,
                    true);

        // Note that the rubber band part of the target is ignored at the present time
        const AllGrippersSinglePose target_grippers_pose = getGripperTargets(world_state);
        const RRTGrippersRepresentation rrt_grippers_goal(
            target_grippers_pose[0].translation(),
            target_grippers_pose[1].translation());

        const std::chrono::duration<double> time_limit(GetRRTTimeout(ph_));
        const auto rrt_results = rrt_helper_->rrtPlan(start_config, rrt_grippers_goal, time_limit);

//        vis_->deleteObjects(CLUSTERING_TARGETS_NS, 1, 2);
        vis_->clearVisualizationsBullet();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
        vis_->clearVisualizationsBullet();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
        rrt_helper_->visualizePath(rrt_results);
        std::this_thread::sleep_for(std::chrono::duration<double>(5.0));

        global_plan_current_timestep_ = 0;
        executing_global_gripper_trajectory_ = true;
        global_plan_gripper_trajectory_ = convertRRTResultIntoGripperTrajectory(world_state.all_grippers_single_pose_, rrt_results);

        // Serialization
        if (GetRRTStoreNewResults(ph_))
        {
            try
            {
                // First serialize
                std::vector<uint8_t> buffer;

                const uint64_t bytes_used = SerializeAllGrippersPoseTrajectory(global_plan_gripper_trajectory_, buffer);

                // Verify no mistakes made in serialization
                {
                    const auto deserialized_results = DeserializeAllGrippersPoseTrajectory(buffer, 0);
                    const auto& deserialized_traj = deserialized_results.first;
                    const auto deserialized_bytes_read = deserialized_results.second;
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
                }

                // Compress and save to file
                ROS_INFO_NAMED("rrt_planner_results", "Compressing for storage");
                const std::vector<uint8_t> compressed_serialized_data = ZlibHelpers::CompressBytes(buffer);
                ROS_INFO_NAMED("rrt_planner_results", "Saving RRT results to file");
                const std::string rrt_file_path = GetLogFolder(nh_) + "rrt_cache_step." + PrettyPrint::PrettyPrint(num_times_invoked);
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
}

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

    const auto resampled_traj = shortcut_smoothing::ResamplePath<AllGrippersSinglePose>(traj, robot_->max_gripper_velocity_norm_ * robot_->dt_, distance_fn, interpolation_fn);
    return resampled_traj;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Model list management
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Planner::initializeModelAndControllerSet(const WorldState& initial_world_state)
{
    // Initialze each model type with the shared data
    DeformableModel::SetGrippersData(robot_->getGrippersData());
    DeformableModel::SetCallbackFunctions(std::bind(&RobotInterface::checkGripperCollision, robot_, std::placeholders::_1));
    DiminishingRigidityModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));
    ConstraintJacobianModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));

    const bool optimization_enabled = GetJacobianControllerOptimizationEnabled(ph_);
    const PlannerTrialType planner_trial_type = GetPlannerTrialType(ph_);

    switch (planner_trial_type)
    {
        case DIMINISHING_RIGIDITY_SINGLE_MODEL_LEAST_SQUARES_CONTROLLER:
        {
            double translational_deformability, rotational_deformability;
            if (ph_.getParam("translational_deformability", translational_deformability) &&
                     ph_.getParam("rotational_deformability", rotational_deformability))
            {
                ROS_INFO_STREAM_NAMED("planner", "Overriding deformability values to "
                                       << translational_deformability << " "
                                       << rotational_deformability);
            }
            else
            {
                translational_deformability = task_specification_->defaultDeformability();
                rotational_deformability = task_specification_->defaultDeformability();
                ROS_INFO_STREAM_NAMED("planner", "Using default deformability value of "
                                       << task_specification_->defaultDeformability());
            }

            model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                      translational_deformability,
                                      rotational_deformability));

            controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                           model_list_.back(),
                                           robot_,
                                           task_specification_->collisionScalingFactor(),
                                           optimization_enabled));
            break;
        }
        case ADAPTIVE_JACOBIAN_SINGLE_MODEL_LEAST_SQUARES_CONTROLLER:
        {
            model_list_.push_back(std::make_shared<AdaptiveJacobianModel>(
                                      DiminishingRigidityModel(task_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(initial_world_state),
                                      GetAdaptiveModelLearningRate(ph_)));

            controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                           model_list_.back(),
                                           robot_,
                                           task_specification_->collisionScalingFactor(),
                                           optimization_enabled));
            break;
        }
        case CONSTRAINT_SINGLE_MODEL_CONSTRAINT_CONTROLLER:
        {
            ROS_INFO_NAMED("planner", "Using constraint model and random sampling controller");

            const double translation_dir_deformability = GetConstraintTranslationalDir(ph_);
            const double translation_dis_deformability = GetConstraintTranslationalDis(ph_);
            const double rotation_deformability = GetConstraintRotational(ph_);
//            const double translational_deformability = GetConstraintTranslationalOldVersion(ph_);

            const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

            model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                  translation_dir_deformability,
                                  translation_dis_deformability,
                                  rotation_deformability,
//                                  translational_deformability,
                                  environment_sdf));

            controller_list_.push_back(std::make_shared<StretchingAvoidanceController>(
                                           nh_,
                                           ph_,
                                           model_list_.back(),
                                           robot_,
                                           environment_sdf,
                                           generator_,
                                           vis_,
                                           GetStretchingAvoidanceControllerSolverType(ph_),
                                           GetMaxSamplingCounts(ph_)));
            break;
        }
        case DIMINISHING_RIGIDITY_SINGLE_MODEL_CONSTRAINT_CONTROLLER:
        {
            ROS_INFO_NAMED("planner", "Using dminishing model and random sampling controller");

            double translational_deformability, rotational_deformability;
            const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

            if (ph_.getParam("translational_deformability", translational_deformability) &&
                     ph_.getParam("rotational_deformability", rotational_deformability))
            {
                ROS_INFO_STREAM_NAMED("planner", "Overriding deformability values to "
                                       << translational_deformability << " "
                                       << rotational_deformability);
            }
            else
            {
                translational_deformability = task_specification_->defaultDeformability();
                rotational_deformability = task_specification_->defaultDeformability();
                ROS_INFO_STREAM_NAMED("planner", "Using default deformability value of "
                                       << task_specification_->defaultDeformability());
            }

            model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                      translational_deformability,
                                      rotational_deformability));

            controller_list_.push_back(std::make_shared<StretchingAvoidanceController>(
                                           nh_,
                                           ph_,
                                           model_list_.back(),
                                           robot_,
                                           environment_sdf,
                                           generator_,
                                           vis_,
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

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                                   model_list_.back(),
                                                   robot_,
                                                   task_specification_->collisionScalingFactor(),
                                                   optimization_enabled));
                }
            }
            ROS_INFO_STREAM_NAMED("planner", "Num diminishing rigidity models: "
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

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                                   model_list_.back(),
                                                   robot_,
                                                   task_specification_->collisionScalingFactor(),
                                                   optimization_enabled));
            }
            ROS_INFO_STREAM_NAMED("planner", "Num adaptive Jacobian models: "
                                   << std::floor(std::log(learning_rate_max / learning_rate_min) / std::log(learning_rate_step)));
            break;
        }
        case MULTI_MODEL_CONTROLLER_TEST:
        {
            ROS_INFO_NAMED("planner", "Using multiple model-controller sets");

            // Constraint Model with New Controller. (MM)
            {
                const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

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
                                               model_list_.back(),
                                               robot_,
                                               environment_sdf,
                                               generator_,
                                               vis_,
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
                    ROS_INFO_STREAM_NAMED("planner", "Overriding deformability values to "
                                           << translational_deformability << " "
                                           << rotational_deformability);
                }
                else
                {
                    translational_deformability = task_specification_->defaultDeformability();
                    rotational_deformability = task_specification_->defaultDeformability();
                    ROS_INFO_STREAM_NAMED("planner", "Using default deformability value of "
                                           << task_specification_->defaultDeformability());
                }

                model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                          translational_deformability,
                                          rotational_deformability));

                controller_list_.push_back(std::make_shared<StretchingAvoidanceController>(
                                               nh_,
                                               ph_,
                                               model_list_.back(),
                                               robot_,
                                               environment_sdf,
                                               generator_,
                                               vis_,
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
                    ROS_INFO_STREAM_NAMED("planner", "Overriding deformability values to "
                                           << translational_deformability << " "
                                           << rotational_deformability);
                }
                else
                {
                    translational_deformability = task_specification_->defaultDeformability();
                    rotational_deformability = task_specification_->defaultDeformability();
                    ROS_INFO_STREAM_NAMED("planner", "Using default deformability value of "
                                           << task_specification_->defaultDeformability());
                }

                model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                          translational_deformability,
                                          rotational_deformability));

                controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                               model_list_.back(),
                                               robot_,
                                               task_specification_->collisionScalingFactor(),
                                               optimization_enabled));
            }
            break;
        }
        case MULTI_MODEL_ACCURACY_TEST:
        {
            // Constraint Model
            {
                const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

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
                                               ph_,
                                               model_list_.back(),
                                               robot_));
            }
            // Dminishing Rigidity Model
            {
                double translational_deformability, rotational_deformability;
                if (ph_.getParam("translational_deformability", translational_deformability) &&
                         ph_.getParam("rotational_deformability", rotational_deformability))
                {
                    ROS_INFO_STREAM_NAMED("planner", "Overriding deformability values to "
                                           << translational_deformability << " "
                                           << rotational_deformability);
                }
                else
                {
                    translational_deformability = task_specification_->defaultDeformability();
                    rotational_deformability = task_specification_->defaultDeformability();
                    ROS_INFO_STREAM_NAMED("planner", "Using default deformability value of "
                                           << task_specification_->defaultDeformability());
                }

                model_list_.push_back(std::make_shared<DiminishingRigidityModel>(
                                          translational_deformability,
                                          rotational_deformability));

                controller_list_.push_back(std::make_shared<StraightLineController>(
                                               ph_,
                                               model_list_.back(),
                                               robot_));
            }
            break;
        }
        default:
        {
            ROS_FATAL_NAMED("planner", "Invalid trial type, this should not be possible.");
            assert(false && "Invalid trial type, this should not be possible.");
        }
    }

    assert(controller_list_.size() == model_list_.size());

    createBandits();
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
// Model utility functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::updateModels
 * @param suggested_trajectories
 * @param model_used
 * @param world_feedback
 */
void Planner::updateModels(
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
MatrixXd Planner::calculateProcessNoise(
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

void Planner::visualizeGripperMotion(
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

void Planner::initializePlannerLogging()
{
    if (planner_logging_enabled_)
    {
        const std::string log_folder = GetLogFolder(nh_);
        ROS_INFO_STREAM_NAMED("planner", "Logging to " << log_folder);

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

void Planner::initializeControllerLogging()
{
    if(controller_logging_enabled_)
    {
        const std::string log_folder = GetLogFolder(nh_);
        ROS_INFO_STREAM_NAMED("planner", "Logging to " << log_folder);

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
// because of the way forking words (and doesn't) in Bullet. They should be very close however.
void Planner::logPlannerData(
        const WorldState& initial_world_state,
        const WorldState& resulting_world_state,
        const std::vector<WorldState>& individual_model_results,
        const Eigen::VectorXd& model_utility_mean,
        const Eigen::MatrixXd& model_utility_covariance,
        const ssize_t model_used)
{
    if (planner_logging_enabled_)
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

        const static Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
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
// because of the way forking words (and doesn't) in Bullet. They should be very close however.
void Planner::controllerLogData(
        const WorldState& initial_world_state,
        const WorldState& resulting_world_state,
        const std::vector<WorldState>& individual_model_results,
        const DeformableController::InputData& controller_input_data,
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
        const ObjectDeltaAndWeight& task_desired_error_correction = controller_input_data.desired_object_motion_.error_correction_;
        const Eigen::VectorXd& desired_p_dot = task_desired_error_correction.delta;
        const Eigen::VectorXd& desired_p_dot_weight = task_desired_error_correction.weight;
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
            const Eigen::MatrixXd node_distance =
                    CalculateDistanceMatrix(individual_model_results[model_ind].object_configuration_);

            for (ssize_t node_ind = 0; node_ind < num_nodes; node_ind++)
            {
                const double point_weight = desired_p_dot_weight(node_ind * 3);
                if (point_weight > 0.0)
                {
                    //  Calculate p_dot error
                    const Eigen::Vector3d& point_real_p_dot = real_p_dot.col(node_ind);
                    const Eigen::Vector3d& point_desired_p_dot = desired_p_dot.segment<3>(node_ind * 3);
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
            ROS_INFO_STREAM_NAMED("planner", "average desired p dot is       " << desired_p_dot_avg_norm);
            ROS_INFO_STREAM_NAMED("planner", "max pointwise desired p dot is " << desired_p_dot_max);
        }

        // Do the actual logging itself
        const static Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
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




void Planner::storeWorldState(const WorldState& world_state)
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
        ZlibHelpers::CompressAndWriteToFile(buffer, full_path);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to store world_state: "  <<  e.what());
    }
}

void Planner::loadStoredWorldState(WorldState& world_state)
{
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
        const auto deserialized_results = WorldState::Deserialize(buffer, 0);
        world_state = deserialized_results.first;

    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to load stored world_state: "  <<  e.what());
    }
}

bool Planner::useStoredWorldState() const
{
    return ROSHelpers::GetParamDebugLog<bool>(ph_, "use_stored_world_state", false);
}
