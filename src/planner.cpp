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
#include "smmap/diminishing_rigidity_model.h"
#include "smmap/adaptive_jacobian_model.h"
#include "smmap/least_squares_jacobian_model.h"
#include "smmap/constraint_jacobian_model.h"

#include "smmap/least_squares_controller_with_object_avoidance.h"
#include "smmap/least_squares_controller_random_sampling.h"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;
using namespace EigenHelpersConversions;

#pragma message "Magic number - reward scaling factor starting value"
#define REWARD_STANDARD_DEV_SCALING_FACTOR_START (1.0)

const std::string Planner::DESIRED_DELTA_NS                         = "desired delta";
const std::string Planner::PREDICTED_DELTA_NS                       = "predicted_delta";

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

/**
 * @brief GetShortestPathBetweenGrippersThroughObject
 * @param grippers_data
 * @param object
 * @param neighour_fn
 * @return The index of the nodes between the grippers, following the shortest path through the object
 */
static std::vector<ssize_t> GetShortestPathBetweenGrippersThroughObject(
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

static EigenHelpers::VectorVector3d GetPathBetweenGrippersThroughObject(
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
 * @param robot
 * @param vis
 * @param task_specification
 */
Planner::Planner(
        RobotInterface& robot,
        Visualizer& vis,
        const std::shared_ptr<TaskSpecification>& task_specification)
    : nh_("")
    , ph_("~")
    , seed_(GetPlannerSeed(ph_))
    , generator_(seed_)
    , robot_(robot)
    , task_specification_(task_specification)
    , dijkstras_task_(nullptr)

    , calculate_regret_(GetCalculateRegret(ph_))
    , reward_std_dev_scale_factor_(REWARD_STANDARD_DEV_SCALING_FACTOR_START)
    , process_noise_factor_(GetProcessNoiseFactor(ph_))
    , observation_noise_factor_(GetObservationNoiseFactor(ph_))
    , correlation_strength_factor_(GetCorrelationStrengthFactor(ph_))

    , enable_stuck_detection_(GetEnableStuckDetection(ph_))
    , max_lookahead_steps_(GetNumLookaheadSteps(ph_))
    , max_grippers_pose_history_length_(GetMaxGrippersPoseHistoryLength(ph_))
    , executing_global_gripper_trajectory_(false)
    , global_plan_current_timestep_(-1)
    , global_plan_gripper_trajectory_(0)
    , rrt_helper_(nullptr)
    , object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh_)))
    , controller_count_(0)

    , logging_enabled_(GetLoggingEnabled(nh_))
    , controller_logging_enabled_(true)
    , vis_(vis)
    , visualize_desired_motion_(GetVisualizeObjectDesiredMotion(ph_))
    , visualize_predicted_motion_(GetVisualizeObjectPredictedMotion(ph_))
{
    ROS_INFO_STREAM_NAMED("planner", "Using seed " << std::hex << seed_ );
    initializeLogging();
    initializeControllerLogging();
    initializeGrippersMaxDistance();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The one function that gets invoked externally
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Planner::execute()
{
    WorldState world_feedback = robot_.start();
    const double start_time = world_feedback.sim_time_;
    initializeModelAndControllerSet(world_feedback);

    if (enable_stuck_detection_)
    {
        assert(robot_.getGrippersData().size() == 2);

        dijkstras_task_ = std::dynamic_pointer_cast<DijkstrasCoverageTask>(task_specification_);
        assert(dijkstras_task_ != nullptr);

        // Pass in all the config values that the RRT needs; for example goal bias, step size, etc.
        rrt_helper_ = std::unique_ptr<RRTHelper>(
                    new RRTHelper(
                        dijkstras_task_->environment_sdf_,
                        vis_,
                        generator_,
                        GetRRTPlanningXMin(ph_),
                        GetRRTPlanningXMax(ph_),
                        GetRRTPlanningYMin(ph_),
                        GetRRTPlanningYMax(ph_),
                        GetRRTPlanningZMin(ph_),
                        GetRRTPlanningZMax(ph_),
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        GetRRTGoalBias(ph_),
                        dijkstras_task_->work_space_grid_.minStepDimension(),
                        GetRRTMinGripperDistanceToObstacles(ph_),
                        GetRRTHomotopyDistancePenalty(),
                        GetRRTMaxShortcutIndexDistance(ph_),
                        GetRRTMaxSmoothingIterations(ph_),
                        GetRRTMaxFailedSmoothingIterations(ph_),
                        !GetDisableAllVisualizations(ph_)));

        // Create the initial rubber band
        const auto neighbour_fn = [&] (const ssize_t& node)
        {
            return dijkstras_task_->getNodeNeighbours(node);
        };
        path_between_grippers_through_object_ = GetShortestPathBetweenGrippersThroughObject(robot_.getGrippersData(), GetObjectInitialConfiguration(nh_), neighbour_fn);
        const auto starting_band_points = GetPathBetweenGrippersThroughObject(world_feedback, path_between_grippers_through_object_);

        const double max_band_distance = (world_feedback.all_grippers_single_pose_[0].translation() - world_feedback.all_grippers_single_pose_[1].translation()).norm() * dijkstras_task_->maxStretchFactor();
        virtual_rubber_band_between_grippers_ = std::make_shared<VirtualRubberBand>(
                    starting_band_points,
                    max_band_distance,
                    dijkstras_task_,
                    vis_,
                    generator_);
    }

    while (robot_.ok())
    {
        const WorldState world_state = world_feedback;
        world_feedback = sendNextCommand(world_state);

        if (unlikely(world_feedback.sim_time_ - start_time >= task_specification_->maxTime()
                     || task_specification_->taskDone(world_feedback)))
        {
            ROS_INFO_NAMED("planner", "------------------------------- End of Task -------------------------------------------");
            const double current_error = task_specification_->calculateError(world_state);
            ROS_INFO_STREAM_NAMED("planner", "   Planner/Task sim time " << world_state.sim_time_ << "\t Error: " << current_error);

            vis_.deleteObjects(Planner::PROJECTED_GRIPPER_NS,            1, (int32_t)(4 * GetNumLookaheadSteps(ph_)) + 10);
            vis_.deleteObjects(Planner::PROJECTED_BAND_NS,               1, (int32_t)GetNumLookaheadSteps(ph_) + 10);
            vis_.deleteObjects(Planner::PROJECTED_POINT_PATH_NS,         1, 2);
            vis_.deleteObjects(Planner::PROJECTED_POINT_PATH_LINES_NS,   1, 2);

            if (world_feedback.sim_time_ - start_time >= task_specification_->maxTime())
            {
                ROS_INFO("Terminating task as time has run out");
            }
            if (task_specification_->taskDone(world_feedback))
            {
                ROS_INFO("Terminating task as the task has been completed");
            }
            robot_.shutdown();
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

        // This bool variable here forces some tasks to utilize only local controller --- Added by Mengyao
        // const bool can_use_global_planner = canUseGlobalPlanner();

        // Check if the global plan has 'hooked' the deformable object on something
        if (executing_global_gripper_trajectory_)
        {

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

                vis_.deleteObjects(DESIRED_DELTA_NS, 1, 100);
                vis_.deleteObjects(PROJECTED_GRIPPER_NS,            1, (int32_t)(4 * max_lookahead_steps_) + 10);
                vis_.deleteObjects(PROJECTED_BAND_NS,               1, (int32_t)max_lookahead_steps_ + 10);
                vis_.deleteObjects(PROJECTED_POINT_PATH_NS,         1, 2);
                vis_.deleteObjects(PROJECTED_POINT_PATH_LINES_NS,   1, 2);

                ROS_WARN_COND_NAMED(global_planner_needed_due_to_overstretch, "planner", "Invoking global planner due to overstretch");
                ROS_WARN_COND_NAMED(global_planner_needed_due_to_lack_of_progress, "planner", "Invoking global planner due to collision");

                ROS_INFO_NAMED("planner", "----------------------------------------------------------------------------");
            }
        }

        // If we need to (re)plan due to the local controller getting stuck, or the gobal plan failing, then do so
        if (planning_needed)
        {
            rrt_helper_->addBandToBlacklist(virtual_rubber_band_between_grippers_->getVectorRepresentation());
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
        const auto band_points = GetPathBetweenGrippersThroughObject(world_feedback, path_between_grippers_through_object_);
        virtual_rubber_band_between_grippers_->setPointsAndSmooth(band_points);

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
        const WorldState& world_state)
{
    Stopwatch stopwatch;
    Stopwatch function_wide_stopwatch;
    Stopwatch controller_stopwatch;

    const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return task_specification_->calculateDesiredDirection(world_state);
    };
    DeformableModel::DeformableModelInputData model_input_data(
                task_desired_direction_fn,
                world_state,
                robot_.dt_);
  //  ObjectDeltaAndWeight& task_desired_motion = model_input_data.desired_object_motion_;
    const ObjectPointSet current_object_configuration = world_state.object_configuration_;

    if (visualize_desired_motion_)
    {
        visualizeDesiredMotion(world_state, model_input_data.desired_object_motion_);
    }

    // Use a model-controller to get a valid object motion from testGrippersPoses first. --- Added by Mengyao
    /*
    if (calculate_regret_ && num_models_ > 1)
    {
        controller_count_ = controller_count_ - controller_count_ / num_models_ * num_models_;
        std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> helper_robot_command;
        helper_robot_command =
            controller_list_[controller_count_]->getGripperMotion(
                    model_input_data,
                    robot_.max_gripper_velocity_);

        controller_count_ ++;

        // Helper Lambda to get a valid p_dot
        const auto valid_motion_feedback_fn = [&] (const size_t model_ind, const WorldState& world_state)
        {
            ObjectPointSet desired_delta = world_state.object_configuration_ - current_object_configuration;
            for (ssize_t node_ind = 0; node_ind < desired_delta.cols(); node_ind++)
            {
                model_input_data.desired_object_motion_.delta.segment(node_ind * 3, 3) = desired_delta.col(node_ind);
            //    model_input_data.desired_object_motion_.weight.segment(node_ind * 3, 3) = Eigen::MatrixXd::Ones(3,1);
            }
        };

        std::vector<AllGrippersSinglePose> helper_grippers_poses(1);
        helper_grippers_poses[0] = kinematics::applyTwist(world_state.all_grippers_single_pose_, helper_robot_command.first);
        robot_.testGrippersPoses(helper_grippers_poses, valid_motion_feedback_fn);
    }
    */

    /*
    const auto valid_desired_direction_fn = [&] (const WorldState& world_state)
    {
        return task_desired_motion;
    };
    const DeformableModel::DeformableModelInputData model_input_data(task_desired_direction_fn, world_state, robot_.dt_);
    */

    // Pick an arm to use
    const ssize_t model_to_use = model_utility_bandit_.selectArmToPull(generator_);
    #pragma message "Foce model_to_use = 0"
    //const ssize_t model_to_use = 0;
    //assert(model_to_use == 0);


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
            controller_stopwatch(RESET);

            suggested_robot_commands[model_ind] =
                controller_list_[model_ind]->getGripperMotion(
                        model_input_data,
                        robot_.max_gripper_velocity_);


            visualize_gripper_motion( world_state.all_grippers_single_pose_,
                                      suggested_robot_commands[model_ind].first,
                                      model_ind);

            // Measure the time it took to pick a model
            ROS_INFO_STREAM_NAMED("planner", model_ind << "th Controller get suggested motion in" << controller_stopwatch(READ) << " seconds");
        }
    }

    // Measure the time it took to pick a model
    ROS_INFO_STREAM_NAMED("planner", "Calculated model suggestions and picked one in  " << stopwatch(READ) << " seconds");

    // Calculate regret if we need to
    std::vector<double> individual_rewards(num_models_, std::numeric_limits<double>::infinity());

    // Calculate control error, stretching severity    --- Added by Mengyao
    std::vector<double> ave_control_error(num_models_, 0.0);
    std::vector<long> stretching_count(num_models_, 0);
    std::vector<double> current_stretching_factor(num_models_, 0.0);

    ObjectDeltaAndWeight& task_desired_motion = model_input_data.desired_object_motion_;
    const Eigen::VectorXd desired_p_dot = task_desired_motion.delta;
    const Eigen::VectorXd desired_p_dot_weight = task_desired_motion.weight;
    const int num_grippers = GetGrippersData(nh_).size();

    if (calculate_regret_ && num_models_ > 1)
    {
        stopwatch(RESET);

        const double prev_error = task_specification_->calculateError(world_state);
        const auto test_feedback_fn = [&] (const size_t model_ind, const WorldState& world_state)
        {
            individual_rewards[model_ind] = prev_error - task_specification_->calculateError(world_state);

            // TODO: Double check with Dale the implementation here
            // Get control errors for different model-controller sets. --- Added by Mengyao
            ObjectPointSet real_p_dot = world_state.object_configuration_ - current_object_configuration;
            ssize_t num_nodes = real_p_dot.cols();

            int point_count = 0;
            bool over_stretch = false;
            const double max_stretch_factor = GetMaxStretchFactor(ph_);
            double max_stretching = 0.0;
            double desired_p_dot_ave_norm = 0.0;
            double desired_p_dot_max = 0.0;

            for (ssize_t node_ind = 0; node_ind < num_nodes; node_ind++)
            {
                //  Calculate p_dot error
                const Eigen::Vector3d& point_real_p_dot = real_p_dot.col(node_ind);
                const Eigen::Vector3d& point_desired_p_dot = desired_p_dot.segment<3>(node_ind * 3);
                const double point_weight = desired_p_dot_weight(node_ind * 3);

                if (point_weight > 0)
                {
                    point_count ++;
                    ave_control_error[model_ind] = ave_control_error[model_ind] + (point_real_p_dot - point_desired_p_dot).norm();

                    desired_p_dot_ave_norm += point_desired_p_dot.norm();
                    if (point_desired_p_dot.norm() > desired_p_dot_max)
                    {
                        desired_p_dot_max = point_desired_p_dot.norm();
                    }
                }

                // Calculate stretching factor
                const Eigen::MatrixXd node_squared_distance =
                        CalculateSquaredDistanceMatrix(world_state.object_configuration_);

                ssize_t first_node = node_ind;

                for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
                {
                    double this_stretching_factor = std::sqrt(node_squared_distance(first_node, second_node))
                            / object_initial_node_distance_(first_node, second_node);
                    if (this_stretching_factor > max_stretching)
                    {
                        max_stretching = this_stretching_factor;
                    }

                    const double max_distance = max_stretch_factor * object_initial_node_distance_(first_node, second_node);
                    if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
                    {
                        over_stretch = true;
                    }
                }
            }
            if(point_count > 0)
            {
                ave_control_error[model_ind] = ave_control_error[model_ind] / point_count;
                desired_p_dot_ave_norm /= point_count;
            }
            if (num_grippers == 2)
            {
                double this_stretching_factor = (world_state.all_grippers_single_pose_.at(0).translation()
                        - world_state.all_grippers_single_pose_.at(1).translation()).norm()
                        / max_grippers_distance_;
                if (this_stretching_factor > max_stretching)
                {
                    max_stretching = this_stretching_factor;
                }
            }
            current_stretching_factor[model_ind] = max_stretching;
            ROS_INFO_STREAM_NAMED("planner", "average desired p dot is" << desired_p_dot_ave_norm);
            ROS_INFO_STREAM_NAMED("planner", "max pointwise desired p dot is" << desired_p_dot_max);
        };

        // const auto control_error_fn = [&] (const size_t model_ind, const WorldState& feed_back_world_state)
        // { };

        std::vector<AllGrippersSinglePose> poses_to_test(num_models_);
        for (size_t model_ind = 0; model_ind < (size_t)num_models_; model_ind++)
        {
            poses_to_test[model_ind] = kinematics::applyTwist(world_state.all_grippers_single_pose_, suggested_robot_commands[model_ind].first);

        //    stretching_count[model_ind] = controller_list_[model_to_use]->getStretchingViolationCount();
        //    current_stretching_factor[model_ind] = controller_list_[model_to_use]->getCurrentStretchingFactor();

        }
        robot_.testGrippersPoses(poses_to_test, test_feedback_fn);

        ROS_INFO_STREAM_NAMED("planner", "Collected data to calculate regret in " << stopwatch(READ) << " seconds");
    }
    else if (num_models_ == 1)
    {
        ssize_t model_ind = 0;
        ObjectPointSet real_p_dot = world_state.object_configuration_ - current_object_configuration;
        ssize_t num_nodes = real_p_dot.cols();

        int point_count = 0;
        bool over_stretch = false;
        const double max_stretch_factor = GetMaxStretchFactor(ph_);
        double max_stretching = 0.0;
        double desired_p_dot_ave_norm = 0.0;
        double desired_p_dot_max = 0.0;

        for (ssize_t node_ind = 0; node_ind < num_nodes; node_ind++)
        {
            //  Calculate p_dot error
            const Eigen::Vector3d& point_real_p_dot = real_p_dot.col(node_ind);
            const Eigen::Vector3d& point_desired_p_dot = desired_p_dot.segment<3>(node_ind * 3);
            const double point_weight = desired_p_dot_weight(node_ind * 3);

            if (point_weight > 0)
            {
                point_count ++;
                ave_control_error[model_ind] = ave_control_error[model_ind] + (point_real_p_dot - point_desired_p_dot).norm();

                desired_p_dot_ave_norm += point_desired_p_dot.norm();
                if (point_desired_p_dot.norm() > desired_p_dot_max)
                {
                    desired_p_dot_max = point_desired_p_dot.norm();
                }
            }

            // Calculate stretching factor
            const Eigen::MatrixXd node_squared_distance =
                    CalculateSquaredDistanceMatrix(world_state.object_configuration_);

            ssize_t first_node = node_ind;

            for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
            {
                double this_stretching_factor = std::sqrt(node_squared_distance(first_node, second_node))
                        / object_initial_node_distance_(first_node, second_node);
                if (this_stretching_factor > max_stretching)
                {
                    max_stretching = this_stretching_factor;
                }

                const double max_distance = max_stretch_factor * object_initial_node_distance_(first_node, second_node);
                if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
                {
                    over_stretch = true;
                }
            }
        }
        if(point_count > 0)
        {
            ave_control_error[model_ind] = ave_control_error[model_ind] / point_count;
            desired_p_dot_ave_norm /= point_count;
        }
        if (num_grippers == 2)
        {
            double this_stretching_factor = (world_state.all_grippers_single_pose_.at(0).translation()
                    - world_state.all_grippers_single_pose_.at(1).translation()).norm()
                    / max_grippers_distance_;
            if (this_stretching_factor > max_stretching)
            {
                max_stretching = this_stretching_factor;
            }
        }
        current_stretching_factor[model_ind] = max_stretching;

        ROS_INFO_STREAM_NAMED("planner", "average desired p dot is  " << desired_p_dot_ave_norm);
        ROS_INFO_STREAM_NAMED("planner", "max pointwise desired p dot is  " << desired_p_dot_max);
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

    // Visualize Force on object, should add new ros function for new flag. --- Added by Mengyao

    /*
    if (visualize_desired_motion_)
    {
    //    visualizeTotalForceOnGripper(world_state);

        double force_scale = 0.1;
        switch (GetDeformableType(nh_))
        {
            case ROPE:
            {
                const ObjectWrench& object_wrench = world_state.object_wrench_;
                vis_.visualizeObjectForce(
                            "ForceOnObject",
                            world_state.object_configuration_,
                            object_wrench.MagnifiedForce(force_scale),
                            Visualizer::Cyan());

                std::cout << "head force norm: "
                          << object_wrench.GetRopeEndsForce().first.norm()
                          << std::endl;
                std::cout << "head force norm: "
                          << object_wrench.GetRopeEndsForce().second.norm()
                          << std::endl;
                break;
            }
            case CLOTH:
            {
                const ObjectPointSet& object_configuration = world_state.object_configuration_;
                const ObjectWrench& object_wrench = world_state.object_wrench_;
                const std::vector<GripperData>& grippers_data = model_list_[model_to_use]->GetGrippersData();

                // Assume knowing it is 2
                const int num_grippers = grippers_data.size();
                int num_total_attached_nodes = 0;
                for (int gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
                {
                    num_total_attached_nodes += grippers_data.at(gripper_ind).node_indices_.size();
                }

                ObjectPointSet nodes_attached(3, num_total_attached_nodes);
                std::vector<Eigen::Vector3d> forces_attached;

                size_t node_ind = 0;

                std::vector<double> total_force_per_gripper(2, 0.0);

                for(int gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
                {
                    for (int node_gripper_ind = 0;
                         node_gripper_ind < grippers_data.at(gripper_ind).node_indices_.size();
                         node_gripper_ind++)
                    {
                        nodes_attached.col(node_ind)
                                = object_configuration.col(
                                    grippers_data.at(gripper_ind).node_indices_.at(node_gripper_ind));

                        forces_attached.push_back(
                                force_scale
                                * object_wrench.object_force[grippers_data.at(gripper_ind).node_indices_.at(node_gripper_ind)]);

                        total_force_per_gripper.at(gripper_ind) += forces_attached.at(node_ind).norm();

                        node_ind++;
                    }
                    std::cout << "total force on " << gripper_ind << " is "
                              << total_force_per_gripper.at(gripper_ind) << std::endl;
                }

                vis_.visualizeObjectForce(
                            "ForceOnObjectGraspedNodes",
                            nodes_attached,
                            forces_attached,
                            Visualizer::Cyan());
                break;
            }
            default:
                assert(false && "deformable type is neither rope nor cloth -- planner.cpp");
                break;
        }
    }
    */


    if (visualize_predicted_motion_)
    {
        const ObjectPointSet& object_delta = suggested_robot_commands[(size_t)model_to_use].second;
        vis_.visualizeObjectDelta(
                    PREDICTED_DELTA_NS,
                    world_state.object_configuration_,
                    world_state.object_configuration_ + 80.0 * object_delta,
                    Visualizer::Blue());

    }

    // desired_p_dot.resizeLike(real_p_dot);
    // desired_p_dot_weight.resizeLike(real_p_dot);

    ROS_INFO_NAMED("planner", "Updating models and logging data");
    updateModels(world_state, task_desired_motion, suggested_robot_commands, model_to_use, world_feedback);

    logData(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), model_to_use, individual_rewards);

    controllerLogData(world_feedback, ave_control_error, current_stretching_factor, stretching_count);

    const double controller_time = function_wide_stopwatch(READ) - robot_execution_time;
    ROS_INFO_STREAM_NAMED("planner", "Total local controller time                     " << controller_time << " seconds");

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
    logData(world_feedback, model_utility_bandit_.getMean(), model_utility_bandit_.getSecondStat(), -1, fake_rewards);

    return world_feedback;
}

// Helper function to force some task type use only local controller
// --- Added by Mengyao
/*
bool Planner::canUseGlobalPlanner()
{
    switch (GetDeformableType(nh_))
    {
        case ROPE:
        {
            return false;
            break;
        }
        case CLOTH:
        {
            if (GetTaskType(nh_) == CLOTH_WAFR)
            {
                return false;
            }
            return true;
        }
        default:
        {
            assert(false && "deformabletype is neither cloth nor rope");
            return true;
        }
    }
}
*/

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
                controller_list_[0]->getGripperMotion(
                    model_input_data,
                    dijkstras_task_->work_space_grid_.minStepDimension() / robot_.dt_);

        // Move the grippers forward
        auto& current_grippers_pose = world_state_copy.all_grippers_single_pose_;
        current_grippers_pose = kinematics::applyTwist(world_state_copy.all_grippers_single_pose_, robot_command.first);
        for (auto& pose : current_grippers_pose)
        {
            pose.translation() = dijkstras_task_->environment_sdf_.ProjectOutOfCollisionToMinimumDistance3d(pose.translation(), GetRobotGripperRadius());
        }

        auto collision_check_future = std::async(std::launch::async, &RobotInterface::checkGripperCollision, &robot_, current_grippers_pose);

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
                    current_grippers_pose[0].translation(),
                    current_grippers_pose[1].translation(),
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
// Model list management
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Planner::initializeModelAndControllerSet(const WorldState& initial_world_state)
{
    // Initialze each model type with the shared data
    DeformableModel::SetGrippersData(robot_.getGrippersData());
    DeformableModel::SetCallbackFunctions(std::bind(&RobotInterface::checkGripperCollision, &robot_, std::placeholders::_1));
    DiminishingRigidityModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));
    ConstraintJacobianModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));

    const bool optimization_enabled = GetJacobianControllerOptimizationEnabled(ph_);

    // Create some models and add them to the model set
    if (GetUseDiminishingRigidityModel(ph_))
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
                                       nh_,
                                       ph_,
                                       model_list_.back(),
                                       task_specification_->collisionScalingFactor(),
                                       optimization_enabled));
    }
    else if (GetUseMultiModel(ph_))
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
                                               nh_,
                                               ph_,
                                               model_list_.back(),
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

        const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
        {
            return task_specification_->calculateDesiredDirection(world_state);
        };

        const DeformableModel::DeformableModelInputData input_data(task_desired_direction_fn, initial_world_state, robot_.dt_);
        for (double learning_rate = learning_rate_min; learning_rate < learning_rate_max; learning_rate *= learning_rate_step)
        {
                model_list_.push_back(std::make_shared<AdaptiveJacobianModel>(
                                          DiminishingRigidityModel(task_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(input_data),
                                          learning_rate));

                controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                               nh_,
                                               ph_,
                                               model_list_.back(),
                                               task_specification_->collisionScalingFactor(),
                                               optimization_enabled));
        }
        ROS_INFO_STREAM_NAMED("planner", "Num adaptive Jacobian models: "
                               << std::floor(std::log(learning_rate_max / learning_rate_min) / std::log(learning_rate_step)));
    }
    else if (GetUseAdaptiveModel(ph_))
    {
        const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
        {
            return task_specification_->calculateDesiredDirection(world_state);
        };

        const DeformableModel::DeformableModelInputData input_data(task_desired_direction_fn, initial_world_state, robot_.dt_);
        model_list_.push_back(std::make_shared<AdaptiveJacobianModel>(
                                  DiminishingRigidityModel(task_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(input_data),
                                  GetAdaptiveModelLearningRate(ph_)));

        controller_list_.push_back(std::make_shared<LeastSquaresControllerWithObjectAvoidance>(
                                       nh_,
                                       ph_,
                                       model_list_.back(),
                                       task_specification_->collisionScalingFactor(),
                                       optimization_enabled));

    }
    // Test of multiple model controller sets at the same time.   --- Added by Mengyao
    else if (GetUseMultiModelController(ph_))
    {
        ROS_INFO_NAMED("planner", "Using multiple model-controller sets");

        // Constraint Model with New Controller. (MM)
        if (false) // using a range of params
        {
            const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

            /*
            const double translation_dir_deformability = GetConstraintTranslationalDir(ph_);
            const double translation_dis_deformability = GetConstraintTranslationalDis(ph_);
            const double rotation_deformability = GetConstraintRotational(ph_);
            */

            const double rotation_deformability = GetConstraintRotational(ph_);

            for (double translation_dis_deformability = 10; translation_dis_deformability < 26; translation_dis_deformability += 5)
            {
                double translation_dir_deformability = 20;

                // ind 0, 4, 8, 12:  trans_dir: 20
                {
                    model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                          translation_dir_deformability,
                                          translation_dis_deformability,
                                          rotation_deformability,
                                          environment_sdf));

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                                   nh_,
                                                   ph_,
                                                   robot_,
                                                   environment_sdf,
                                                   generator_,
                                                   vis_,
                                                   GetGripperControllerType(ph_),
                                                   model_list_.back(),
                                                   GetMaxSamplingCounts(ph_),
                                                   GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
                }

                translation_dir_deformability = 30;
                // ind 1, 5, 9, 13:  trans_dir: 30
                {
                    model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                          translation_dir_deformability,
                                          translation_dis_deformability,
                                          rotation_deformability,
                                          environment_sdf));

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                                   nh_,
                                                   ph_,
                                                   robot_,
                                                   environment_sdf,
                                                   generator_,
                                                   vis_,
                                                   GetGripperControllerType(ph_),
                                                   model_list_.back(),
                                                   GetMaxSamplingCounts(ph_),
                                                   GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
                }

                translation_dir_deformability = 60;
                // ind 2, 6, 10, 14:  trans_dir: 60
                {
                    model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                          translation_dir_deformability,
                                          translation_dis_deformability,
                                          rotation_deformability,
                                          environment_sdf));

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                                   nh_,
                                                   ph_,
                                                   robot_,
                                                   environment_sdf,
                                                   generator_,
                                                   vis_,
                                                   GetGripperControllerType(ph_),
                                                   model_list_.back(),
                                                   GetMaxSamplingCounts(ph_),
                                                   GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
                }

                translation_dir_deformability = 200;
                // ind 3, 7, 11, 15:  trans_dir: 100
                {
                    model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                          translation_dir_deformability,
                                          translation_dis_deformability,
                                          rotation_deformability,
                                          environment_sdf));

                    controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                                   nh_,
                                                   ph_,
                                                   robot_,
                                                   environment_sdf,
                                                   generator_,
                                                   vis_,
                                                   GetGripperControllerType(ph_),
                                                   model_list_.back(),
                                                   GetMaxSamplingCounts(ph_),
                                                   GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
                }

            }

        }
        else
        {
            const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

            const double translation_dir_deformability = GetConstraintTranslationalDir(ph_);
            const double translation_dis_deformability = GetConstraintTranslationalDis(ph_);
            const double rotation_deformability = GetConstraintRotational(ph_);

            model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                                  translation_dir_deformability,
                                  translation_dis_deformability,
                                  rotation_deformability,
                                  environment_sdf));

            controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                           nh_,
                                           ph_,
                                           robot_,
                                           environment_sdf,
                                           generator_,
                                           vis_,
                                           GetGripperControllerType(ph_),
                                           model_list_.back(),
                                           GetMaxSamplingCounts(ph_),
                                           GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
        }

        // Dminishing Model with New Controller. (DM)
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

            controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                           nh_,
                                           ph_,
                                           robot_,
                                           environment_sdf,
                                           generator_,
                                           vis_,
                                           GetGripperControllerType(ph_),
                                           model_list_.back(),
                                           GetMaxSamplingCounts(ph_),
                                           GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
        }

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
                                           nh_,
                                           ph_,
                                           model_list_.back(),
                                           task_specification_->collisionScalingFactor(),
                                           optimization_enabled));
        }

    }
    else if (GetUseConstraintModel(ph_))
    {
        ROS_INFO_NAMED("planner", "Using constraint model and random sampling controller");

        const double translation_dir_deformability = GetConstraintTranslationalDir(ph_);
        const double translation_dis_deformability = GetConstraintTranslationalDis(ph_);
        const double rotation_deformability = GetConstraintRotational(ph_);

        const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));

        model_list_.push_back(std::make_shared<ConstraintJacobianModel>(
                              translation_dir_deformability,
                              translation_dis_deformability,
                              rotation_deformability,
                              environment_sdf));

        controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                       nh_,
                                       ph_,
                                       robot_,
                                       environment_sdf,
                                       generator_,
                                       vis_,
                                       GetGripperControllerType(ph_),
                                       model_list_.back(),
                                       GetMaxSamplingCounts(ph_),
                                       GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
    }
    else if (GetUseDiminishingModelWithSamplingController(ph_))
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

        controller_list_.push_back(std::make_shared<LeastSquaresControllerRandomSampling>(
                                       nh_,
                                       ph_,
                                       robot_,
                                       environment_sdf,
                                       generator_,
                                       vis_,
                                       GetGripperControllerType(ph_),
                                       model_list_.back(),
                                       GetMaxSamplingCounts(ph_),
                                       GetRobotGripperRadius() + GetRobotMinGripperDistanceToObstacles()));
    }
    else
    {
        ROS_FATAL_NAMED("planner", "No model specified, terminating");
        assert(false && "You must specify at a model type");
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

// Initialize max grippers distance  --- Added by Mengyao
void Planner::initializeGrippersMaxDistance()
{
    if (GetGrippersData(nh_).size())
    {
        if (GetDeformableType(nh_) == CLOTH)
        {
            max_grippers_distance_ = GetClothYSize(nh_) - 0.03;
        }
        else if (GetDeformableType(nh_) == ROPE)
        {
            max_grippers_distance_ = GetRopeSegmentLength(nh_) * GetRopeNumLinks(nh_);
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
        /*
        task_specification_->visualizeDeformableObject(
                vis_,
                DESIRED_DELTA_NS,
                AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta),
                colors);
        */

      //  if (task_specification_->deformable_type_ == DeformableType::CLOTH)
      //  {
            vis_.visualizeObjectDelta(
                        DESIRED_DELTA_NS,
                        current_world_state.object_configuration_,
                        AddObjectDelta(current_world_state.object_configuration_, desired_motion.delta * 100),
                        Visualizer::Green());
      //  }
    }
}

// Visulize Force on Gripper  --- Added by Mengyao
void Planner::visualizeTotalForceOnGripper(
        const WorldState &current_world_state,
        const bool visualization_enabled) const
{
    if(visualization_enabled)
    {
        const AllGrippersSinglePose gripper_poses = current_world_state.all_grippers_single_pose_;
        assert(false && "Not yet implemented");
        /*
        const AllGrippersWrench gripper_wrenchs = current_world_state.gripper_wrench_;
        {
            int gripper_ind = 0;
            std::cout << "Friction data on :" << gripper_ind << "th gripper :" << std::endl;
            vis_.visualizeTranslation(
                        "total_force_on_gripper_top_0",
                        gripper_poses.at(gripper_ind).translation(),
                        gripper_poses.at(gripper_ind).translation()
                        + 0.1 * gripper_wrenchs.at(gripper_ind).top_clamp.force,
                        Visualizer::Silver());
            std::cout << "Force magnitude on the top clamp is " << gripper_wrenchs.at(gripper_ind).top_clamp.force.norm() << std::endl;
            vis_.visualizeTranslation(
                        "total_force_on_gripper_bottom_0",
                        gripper_poses.at(gripper_ind).translation(),
                        gripper_poses.at(gripper_ind).translation()
                        + 0.1 * gripper_wrenchs.at(gripper_ind).bottom_clamp.force,
                        Visualizer::Yellow());
            std::cout << "Force magnitude on the bottom clamp is " << gripper_wrenchs.at(gripper_ind).bottom_clamp.force.norm() << std::endl;

            gripper_ind = 1;
            std::cout << "Friction data on :" << gripper_ind << "th gripper :" << std::endl;
            vis_.visualizeTranslation(
                        "total_force_on_gripper_top_1",
                        gripper_poses.at(gripper_ind).translation(),
                        gripper_poses.at(gripper_ind).translation()
                        + 0.1 * gripper_wrenchs.at(gripper_ind).top_clamp.force,
                        Visualizer::Silver());
            std::cout << "Force magnitude on the top clamp is " << gripper_wrenchs.at(gripper_ind).top_clamp.force.norm() << std::endl;
            vis_.visualizeTranslation(
                        "total_force_on_gripper_bottom_1",
                        gripper_poses.at(gripper_ind).translation(),
                        gripper_poses.at(gripper_ind).translation()
                        + 0.1 * gripper_wrenchs.at(gripper_ind).bottom_clamp.force,
                        Visualizer::Yellow());
            std::cout << "Force magnitude on the bottom clamp is " << gripper_wrenchs.at(gripper_ind).bottom_clamp.force.norm() << std::endl;
        }
        */
    }
}


void Planner::visualize_gripper_motion(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& gripper_motion,
        const ssize_t model_ind)
{
    const auto grippers_test_poses = kinematics::applyTwist(current_gripper_pose, gripper_motion);
    EigenHelpers::VectorVector3d line_starts;
    EigenHelpers::VectorVector3d line_ends;

    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
    {
        line_starts.push_back(current_gripper_pose[gripper_ind].translation());
        line_ends.push_back(current_gripper_pose[gripper_ind].translation() + 100 * (grippers_test_poses[gripper_ind].translation() - current_gripper_pose[gripper_ind].translation()));
    }

    switch (model_ind)
    {
        case 0:
    {
        vis_.visualizeLines("MM grippers motion",
                            line_starts,
                            line_ends,
                            Visualizer::Black());
        std::cout << "0 first gripper motion norm: "
                  << gripper_motion.at(0).norm()
                  << std::endl;
        break;
    }
        case 1:
    {
        vis_.visualizeLines("DM grippers motion",
                            line_starts,
                            line_ends,
                            Visualizer::Silver());
        std::cout << "1 first gripper motion norm: "
                  << gripper_motion.at(0).norm()
                  << std::endl;
        break;
    }
        case 2:
    {
//        break;
        vis_.visualizeLines("DD grippers motion",
                            line_starts,
                            line_ends,
                            Visualizer::Yellow());
        std::cout << "2 first gripper motion norm: "
                  << gripper_motion.at(0).norm()
                  << std::endl;
        break;
    }
        default:
        {
            assert(false && "grippers_motion color not assigned for this index");
            break;
        }
    }
}


void Planner::initializeLogging()
{
    if (logging_enabled_)
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

        // Loggers for controller performance.  --- Added by Mengyao
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
                                       "count_stretching_violation",
                                       Log::Log(log_folder + "count_stretching_violation.txt", false)));
    }
}

void Planner::logData(
        const WorldState& current_world_state,
        const Eigen::VectorXd& model_utility_mean,
        const Eigen::MatrixXd& model_utility_covariance,
        const ssize_t model_used,
        const std::vector<double>& rewards_for_all_models)
{
    if (logging_enabled_)
    {
        const static Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
                    " ", " ", "", "");

        LOG(loggers_.at("time"),
             current_world_state.sim_time_);

        LOG(loggers_.at("error"),
             task_specification_->calculateError(current_world_state));

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

// Contoller logger.  --- Added by Mengyao
void Planner::controllerLogData(
        const WorldState& current_world_state,
        const std::vector<double> &ave_contol_error,
        const std::vector<double> current_stretching_factor,
        const std::vector<long> num_stretching_violation)
{
    if(controller_logging_enabled_)
    {
        const static Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
                    " ", " ", "", "");

        LOG(controller_loggers_.at("control_time"),
            current_world_state.sim_time_);

        LOG(controller_loggers_.at("control_error_realtime"),
            PrettyPrint::PrettyPrint(ave_contol_error, false, " "));

        LOG(controller_loggers_.at("realtime_stretching_factor"),
            PrettyPrint::PrettyPrint(current_stretching_factor, false, " "));

        LOG(controller_loggers_.at("count_stretching_violation"),
            PrettyPrint::PrettyPrint(num_stretching_violation, false, " "));
    }
}



