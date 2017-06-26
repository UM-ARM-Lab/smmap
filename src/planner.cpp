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

#pragma message "Magic number - reward scaling factor starting value"
#define REWARD_STANDARD_DEV_SCALING_FACTOR_START (1.0)

const std::string Planner::DESIRED_DELTA_NS                         = "desired delta";

const std::string Planner::PROJECTED_GRIPPER_NS                     = "projected_grippers";
const std::string Planner::PROJECTED_BAND_NS                        = "projected_band";
const std::string Planner::PROJECTED_POINT_PATH_NS                  = "projected_point_paths";
const std::string Planner::PROJECTED_POINT_PATH_LINES_NS            = "projected_point_path_lines";

const std::string Planner::CONSTRAINT_VIOLATION_VERSION1_NS         = "constraint_violation_version1";

const std::string Planner::CLUSTERING_TARGETS_NS                    = "clustering_targets";
const std::string Planner::CLUSTERING_RESULTS_PRE_PROJECT_NS        = "clustering_pre_project";
const std::string Planner::CLUSTERING_RESULTS_POST_PROJECT_NS       = "clustering_post_project";
const std::string Planner::CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS   = "clustering_assigned_centers";


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Internal helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static size_t SizeOfLargestVector(const std::vector<VectorVector3d>& vectors)
{
    size_t largest_vector = 0;

    for (size_t idx = 0; idx < vectors.size(); ++idx)
    {
        largest_vector = std::max(largest_vector, vectors[idx].size());
    }

    return largest_vector;
}

std::vector<uint32_t> NumberOfPointsInEachCluster(
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
    , reward_std_dev_scale_factor_(REWARD_STANDARD_DEV_SCALING_FACTOR_START)
    , process_noise_factor_(GetProcessNoiseFactor(ph_))
    , observation_noise_factor_(GetObservationNoiseFactor(ph_))
    , correlation_strength_factor_(GetCorrelationStrengthFactor(ph_))
    , max_lookahead_steps_(GetNumLookaheadSteps(ph_))
    , max_grippers_pose_history_length_(GetMaxGrippersPoseHistoryLength(ph_))
    , executing_global_gripper_trajectory_(false)
    , global_plan_current_timestep_(-1)
    , global_plan_gripper_trajectory_(0)
    , rrt_helper_(nullptr)
    , logging_enabled_(GetLoggingEnabled(nh_))
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
//                        0.0,
                        dijkstras_task_->work_space_grid_.getXMax(),
                        dijkstras_task_->work_space_grid_.getYMin(),
//                        -0.4,
                        dijkstras_task_->work_space_grid_.getYMax(),
                        dijkstras_task_->work_space_grid_.getZMin(),
//                        0.5,
                        dijkstras_task_->work_space_grid_.getZMax(),
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        GetRRTGoalBias(ph_),
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        GetRRTMinGripperDistanceToObstacles(ph_),
                        GetRRTHomotopyDistancePenalty(),
                        GetRRTMaxShortcutIndexDistance(ph_),
                        GetRRTMaxSmoothingIterations(ph_),
                        GetRRTMaxFailedSmoothingIterations(ph_),
                        !GetDisableAllVisualizations(ph_)));
    }

    ROS_INFO_STREAM_NAMED("planner", "Using seed " << std::hex << seed_ );

    if (logging_enabled_)
    {
        const std::string log_folder = GetLogFolder(nh_);
        Log::Log seed_log(log_folder + "seed.txt", false);
        LOG_STREAM(seed_log, std::hex << seed_);

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "grippers_distance_delta_history",
                            Log::Log(log_folder + "grippers_distance_delta_history.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "error_delta_history",
                            Log::Log(log_folder + "error_delta_history.txt", false)));
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
            Stopwatch stopwatch;
            arc_helpers::DoNotOptimize(world_state);
            const bool global_planner_needed_due_to_overstretch =
                    globalPlannerNeededDueToOverstretch(world_state);

            const bool global_planner_needed_due_to_lack_of_progress =
                    globalPlannerNeededDueToLackOfProgress(world_state);
            arc_helpers::DoNotOptimize(global_planner_needed_due_to_lack_of_progress);
            ROS_INFO_STREAM_NAMED("planner", "Determined if global planner needed in " << stopwatch(READ) << " seconds");


            ROS_INFO_NAMED("planner", "----------------------------------------------------------------------------");

            if (global_planner_needed_due_to_overstretch || global_planner_needed_due_to_lack_of_progress)
            {
                vis_.deleteObjects(DESIRED_DELTA_NS, 1, 100);

                ROS_WARN_COND_NAMED(global_planner_needed_due_to_overstretch, "planner", "Invoking global planner due to overstretch");
                ROS_WARN_COND_NAMED(global_planner_needed_due_to_lack_of_progress, "planner", "Invoking global planner due to collision");

                vis_.deleteObjects(PROJECTED_GRIPPER_NS,            1, (int32_t)(4 * max_lookahead_steps_) + 10);
                vis_.deleteObjects(PROJECTED_BAND_NS,               1, (int32_t)max_lookahead_steps_ + 10);
                vis_.deleteObjects(PROJECTED_POINT_PATH_NS,         1, 2);
                vis_.deleteObjects(PROJECTED_POINT_PATH_LINES_NS,   1, 2);

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
                DESIRED_DELTA_NS,
                AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
                colors);

        if (task_specification_->deformable_type_ == DeformableType::CLOTH)
        {
            vis_.visualizeObjectDelta(
                        DESIRED_DELTA_NS,
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
        const WorldState& world_state)
{
    Stopwatch stopwatch;
    Stopwatch function_wide_stopwatch;

    const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return task_specification_->calculateDesiredDirection(world_state);
    };

    const ObjectDeltaAndWeight task_desired_motion = task_desired_direction_fn(world_state);
    const DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, world_state, robot_.dt_);

//    visualizeDesiredMotion(current_world_state, task_desired_motion);

    // Pick an arm to use
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);
    assert(model_to_use == 0);


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
            // grippers motion generated from pseudo-invese
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
        const double prev_error = task_specification_->calculateError(world_state);
        const auto test_feedback_fn = [&] (const size_t model_ind, const WorldState& world_state)
        {
            individual_rewards[model_ind] = prev_error - task_specification_->calculateError(world_state);
        };

        std::vector<AllGrippersSinglePose> poses_to_test(num_models_);
        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            poses_to_test[model_ind] = kinematics::applyTwist(world_state.all_grippers_single_pose_, suggested_robot_commands[model_ind].first);
        }
        robot_.testGrippersPoses(poses_to_test, test_feedback_fn);

        ROS_INFO_STREAM_NAMED("planner", "Collected data to calculate regret in " << stopwatch(READ) << " seconds");
    }


    // Execute the command
    const AllGrippersSinglePoseDelta& selected_command = suggested_robot_commands[(size_t)model_to_use].first;
    ROS_INFO_STREAM_NAMED("planner", "Sending command to robot, action norm:  " << MultipleGrippersVelocity6dNorm(selected_command));
    const auto all_grippers_single_pose = kinematics::applyTwist(world_state.all_grippers_single_pose_, selected_command);
    // Measure execution time
    stopwatch(RESET);
    arc_helpers::DoNotOptimize(all_grippers_single_pose);
    const WorldState world_feedback = robot_.sendGrippersPoses(all_grippers_single_pose);
    arc_helpers::DoNotOptimize(world_feedback);
    const double robot_execution_time = stopwatch(READ);

    if (virtual_rubber_band_between_grippers_ != nullptr)
    {
        const bool verbose = false;
        virtual_rubber_band_between_grippers_->forwardSimulateVirtualRubberBandToEndpointTargets(
                    world_feedback.all_grippers_single_pose_[0].translation(),
                    world_feedback.all_grippers_single_pose_[1].translation(),
                    verbose);

        // Keep the last N grippers positions recorded to detect if the grippers are stuck
        grippers_pose_history_.push_back(world_feedback.all_grippers_single_pose_);
        error_history_.push_back(dijkstras_task_->calculateError(world_state));
        assert(grippers_pose_history_.size() == error_history_.size());
        if (grippers_pose_history_.size() > max_grippers_pose_history_length_)
        {
            grippers_pose_history_.erase(grippers_pose_history_.begin());
            error_history_.erase(error_history_.begin());
        }
    }

    visualizeDesiredMotion(world_state, task_desired_motion);

    const ObjectPointSet& predicted_object_delta = suggested_robot_commands[(size_t)model_to_use].second;
    vis_.visualizeObjectDelta(
                "Model back_generated position",
                world_state.object_configuration_,
                world_state.object_configuration_ + 300.0 * predicted_object_delta,
                Visualizer::Blue());
//                Visualizer::Black());

    ROS_INFO_NAMED("planner", "Updating models and logging data");
    updateModels(world_state, task_desired_motion, suggested_robot_commands, model_to_use, world_feedback);

    logging_fn_(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), model_to_use, individual_rewards);

    const double controller_time = function_wide_stopwatch(READ) - robot_execution_time;
    ROS_INFO_STREAM_NAMED("planner", "Total local controller time                     " << controller_time << " seconds");

    return world_feedback;
}

WorldState Planner::sendNextCommandUsingGlobalGripperPlannerResults(
        const WorldState& current_world_state)
{
    (void)current_world_state;
    assert(executing_global_gripper_trajectory_);
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
        std::cerr << "Global plan finished, resetting error history\n";

        executing_global_gripper_trajectory_ = false;
        grippers_pose_history_.clear();
        error_history_.clear();

        vis_.deleteObjects(RRTHelper::RRT_SOLUTION_GRIPPER_A_NS,   1, 2);
        vis_.deleteObjects(RRTHelper::RRT_SOLUTION_GRIPPER_B_NS,   1, 2);
        vis_.deleteObjects(RRTHelper::RRT_SOLUTION_RUBBER_BAND_NS, 1, 2);
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
        vis_.visualizePoints(PROJECTED_POINT_PATH_NS, points, Visualizer::Magenta(), 1);
        vis_.visualizeLines(PROJECTED_POINT_PATH_LINES_NS, lines_start_points, lines_end_points, Visualizer::Magenta(), 1);
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
        vis_.visualizeLines(CONSTRAINT_VIOLATION_VERSION1_NS, vis_start_points, vis_end_points, Visualizer::Blue());
    }

    return violations_exist;
}

std::pair<std::vector<VectorVector3d>, std::vector<VirtualRubberBand>> Planner::detectFutureConstraintViolations(
        const WorldState& current_world_state,
        const bool visualization_enabled)
{
    Stopwatch stopwatch;
    Stopwatch function_wide_stopwatch;

    assert(task_specification_->is_dijkstras_type_task_ && current_world_state.all_grippers_single_pose_.size() == 2);
    std::pair<std::vector<VectorVector3d>, std::vector<VirtualRubberBand>> projected_deformable_point_paths_and_projected_virtual_rubber_bands;

    const static std_msgs::ColorRGBA gripper_color = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 0.0f, 0.6f, 1.0f);
    const static std_msgs::ColorRGBA rubber_band_safe_color = Visualizer::Black();
    const static std_msgs::ColorRGBA rubber_band_violation_color = Visualizer::Cyan();
    const bool verbose = false;


    vis_.deleteObjects(PROJECTED_BAND_NS, 1, (int32_t)max_lookahead_steps_ + 10);
    vis_.deleteObjects(PROJECTED_POINT_PATH_NS, 1, 2);
    vis_.deleteObjects(PROJECTED_POINT_PATH_LINES_NS, 1, 2);
    vis_.deleteObjects(PROJECTED_GRIPPER_NS, 1, (int32_t)(4 * max_lookahead_steps_) + 10);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Constraint violation Version 1 - Purely cloth overstretch
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    stopwatch(RESET);
    const std::vector<VectorVector3d> projected_deformable_point_paths = dijkstras_task_->findPathFromObjectToTarget(current_world_state, max_lookahead_steps_);

    const size_t actual_lookahead_steps = SizeOfLargestVector(projected_deformable_point_paths) - 1;
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
                    dijkstras_task_, vis_, generator_);
    }
    virtual_rubber_band_between_grippers_->visualize(PROJECTED_BAND_NS, rubber_band_safe_color, rubber_band_violation_color, 1, visualization_enabled);




//    vis_.deleteObjects(PROJECTED_POINT_PATH_NS, 1, 2);
//    vis_.deleteObjects(PROJECTED_POINT_PATH_LINES_NS, 1, 2);
//    vis_.deleteObjects(PROJECTED_GRIPPER_NS, 1, (int32_t)(4 * max_lookahead_steps_) + 10);




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
        #warning "This projection shouold be projecting to distance"
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
        virtual_rubber_band_between_grippers_copy.visualize(PROJECTED_BAND_NS, rubber_band_safe_color, rubber_band_violation_color, (int32_t)t + 2, visualization_enabled);
        vis_.visualizeGrippers(PROJECTED_GRIPPER_NS, world_state_copy.all_grippers_single_pose_, gripper_color, (int32_t)(4 * t) + 1);

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

//    std::cerr << "Max band length: " << virtual_rubber_band_between_grippers_->maxSafeLength() << std::endl;

    for (size_t t = 0; t < projected_rubber_bands.size(); ++t)
    {
        const VirtualRubberBand& band = projected_rubber_bands[t];
        const double band_length = band.totalLength();
        const std::pair<Eigen::Vector3d, Eigen::Vector3d> endpoints = band.getEndpoints();
        const double distance_between_endpoints = (endpoints.first - endpoints.second).norm();

        // Apply a low pass filter to the band length to try and remove "blips" in the estimate
        filtered_band_length = annealing_factor * filtered_band_length + (1.0 - annealing_factor) * band_length;
//        std::cerr << "Current band length: " << band_length << " Filtered band length: " << filtered_band_length << std::endl;
        // If the band is currently overstretched, and not in free space, then predict future problems
        if (filtered_band_length > band.maxSafeLength() && !CloseEnough(band_length, distance_between_endpoints, 1e-6))
        {
            return true;
        }
    }

    return false;
}

bool Planner::globalPlannerNeededDueToLackOfProgress(
        const WorldState& current_world_state)
{
    static double error_delta_threshold_for_progress = GetErrorDeltaThresholdForProgress(ph_);
    static double grippers_distance_delta_threshold_for_progress = GetGrippersDistanceDeltaThresholdForProgress(ph_);
    (void)current_world_state;

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

    if (logging_enabled_)
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Global gripper planner functions
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

AllGrippersSinglePose Planner::getGripperTargets(const WorldState& world_state)
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////// Determine the cluster centers /////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const auto correspondences = dijkstras_task_->getCoverPointCorrespondences(world_state);
    const auto& cover_point_indices_= correspondences.uncovered_target_points_idxs_;

    VectorVector3d cluster_targets;
    cluster_targets.reserve(cover_point_indices_.size());
    for (size_t idx = 0; idx < cover_point_indices_.size(); ++idx)
    {
        const ssize_t cover_idx = cover_point_indices_[idx];
        cluster_targets.push_back(dijkstras_task_->cover_points_.col(cover_idx));
    }

    vis_.visualizePoints(CLUSTERING_TARGETS_NS, cluster_targets, Visualizer::Blue(), 1);

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

//    vis_.visualizeCubes(CLUSTERING_RESULTS_PRE_PROJECT_NS, cluster_centers, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Red(), 1);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //////////////////////////// Determine which gripper gets assigned which cluster center ////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Get the orientations for each gripper based on their starting orientation
    AllGrippersSinglePose target_gripper_poses = world_state.all_grippers_single_pose_;

    const auto& gripper0_grapsed_points = dijkstras_task_->getGripperAttachedNodesIndices(0);
    const auto& gripper1_grapsed_points = dijkstras_task_->getGripperAttachedNodesIndices(1);

    const auto gripper0_cluster_counts = NumberOfPointsInEachCluster(cluster_labels, num_clusters, gripper0_grapsed_points, correspondences);
    const auto gripper1_cluster_counts = NumberOfPointsInEachCluster(cluster_labels, num_clusters, gripper1_grapsed_points, correspondences);

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

//    vis_.visualizeCubes(CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS, {world_state.all_grippers_single_pose_[0].translation(), target_gripper_poses[0].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Magenta(), 1);
//    vis_.visualizeCubes(CLUSTERING_RESULTS_ASSIGNED_CENTERS_NS, {world_state.all_grippers_single_pose_[1].translation(), target_gripper_poses[1].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Cyan(), 5);

    // Project the targets out of collision
    const double min_dist_to_obstacles = GetRRTMinGripperDistanceToObstacles(ph_) * GetRRTTargetMinDistanceScaleFactor(ph_);
    const Eigen::Vector3d gripper0_position_pre_project = target_gripper_poses[0].translation();
    const Eigen::Vector3d gripper1_position_pre_project = target_gripper_poses[1].translation();
    target_gripper_poses[0].translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(gripper0_position_pre_project, min_dist_to_obstacles);
    target_gripper_poses[1].translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(gripper1_position_pre_project, min_dist_to_obstacles);

//    vis_.visualizeCubes(CLUSTERING_RESULTS_POST_PROJECT_NS, {target_gripper_poses[0].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Magenta(), 1);
//    vis_.visualizeCubes(CLUSTERING_RESULTS_POST_PROJECT_NS, {target_gripper_poses[1].translation()}, Vector3d::Ones() * dijkstras_task_->work_space_grid_.minStepDimension(), Visualizer::Cyan(), 5);

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

        vis_.deleteObjects(CLUSTERING_TARGETS_NS, 1, 2);
        rrt_helper_->visualizePath(rrt_results);

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

    const auto resampled_traj = shortcut_smoothing::ResamplePath<AllGrippersSinglePose>(traj, robot_.max_gripper_velocity_ * robot_.dt_, distance_fn, interpolation_fn);
    return resampled_traj;
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
