#include "smmap/rrt_helper.h"

#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>

using namespace smmap;

RRTHelper::RRTHelper(
        const sdf_tools::SignedDistanceField& environment_sdf,
        const Visualizer& vis,
        std::mt19937_64& generator,
        const double step_size,
        const double x_limits_lower,
        const double x_limits_upper,
        const double y_limits_lower,
        const double y_limits_upper,
        const double z_limits_lower,
        const double z_limits_upper,
        const double goal_reach_radius,
        const int64_t max_shortcut_index_distance,
        const uint32_t max_smoothing_iterations,
        const uint32_t max_failed_smoothing_iterations)
    : x_limits_(x_limits_lower,x_limits_upper)
    , y_limits_(y_limits_lower,y_limits_upper)
    , z_limits_(z_limits_lower,z_limits_upper)
    , goal_reach_radius_(goal_reach_radius)
    , step_size_(step_size)
    , max_shortcut_index_distance_(max_shortcut_index_distance)
    , max_smoothing_iterations_(max_smoothing_iterations)
    , max_failed_smoothing_iterations_(max_failed_smoothing_iterations)
    , uniform_unit_distribution_(0.0, 1.0)
    , environment_sdf_(environment_sdf)
    , vis_(vis)
    , generator_(generator)
    , marker_id_(1)
    , rubber_band_safe_color_(Visualizer::Black())
    , rubber_band_overstretched_color_(Visualizer::Cyan())
{
    assert(max_shortcut_index_distance > 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Helper function for original rrt planning
///////////////////////////////////////////////////////////////////////////////////////////////

// returned distance is the Euclidian distance of two grippers pos
double RRTHelper::distance(
        const RRTConfig& a_node,
        const RRTConfig& b_node) const
{
    const Eigen::Vector3d& transA_first = a_node.first.first;
    const Eigen::Vector3d& transB_first = b_node.first.first;
    const Eigen::Vector3d& transA_second = a_node.first.second;
    const Eigen::Vector3d& transB_second = b_node.first.second;
    return std::sqrt((transA_first - transB_first).squaredNorm() + (transA_second - transB_second).squaredNorm());
}

int64_t RRTHelper::nearestNeighbour(
        const std::vector<simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, Allocator>>& nodes,
        const RRTConfig& config) const
{
    const std::function<double(const simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, Allocator>&, const RRTConfig&)> distance_fn = [&] (
            const simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, Allocator>& rrt_state,
            const RRTConfig& rrt_config)
    {
        return distance(rrt_state.GetValueImmutable(), rrt_config);
    };

    const size_t K = 1;
    return arc_helpers::GetKNearestNeighbors(nodes, config, distance_fn, K)[0].first;
}

// const std::function<T(void)>& sampling_fn,
inline std::pair<Eigen::Vector3d, Eigen::Vector3d> RRTHelper::posPairSampling()
{
    std::pair<Eigen::Vector3d, Eigen::Vector3d> rand_sample;
    const bool sample_goal = uniform_unit_distribution_(generator_) < 0.1;

    if (sample_goal)
    {
        rand_sample = gripper_goal_positions_;
    }
    else
    {
        const double x1 = EigenHelpers::Interpolate(x_limits_.first, x_limits_.second, uniform_unit_distribution_(generator_));
        const double y1 = EigenHelpers::Interpolate(y_limits_.first, y_limits_.second, uniform_unit_distribution_(generator_));
        const double z1 = EigenHelpers::Interpolate(z_limits_.first, z_limits_.second, uniform_unit_distribution_(generator_));

        const double x2 = EigenHelpers::Interpolate(x_limits_.first, x_limits_.second, uniform_unit_distribution_(generator_));
        const double y2 = EigenHelpers::Interpolate(y_limits_.first, y_limits_.second, uniform_unit_distribution_(generator_));
        const double z2 = EigenHelpers::Interpolate(z_limits_.first, z_limits_.second, uniform_unit_distribution_(generator_));

        rand_sample.first = Eigen::Vector3d(x1, y1, z1);
        rand_sample.second = Eigen::Vector3d(x2, y2, z2);
    }

//                vis_.visualizePoints("sampled_configuration", {rand_sample.first, rand_sample.second}, Visualizer::Red(), 1);

    return rand_sample;
}


/* const std::function<std::vector<std::pair<T, int64_t>>(const T&, const T&)>& forward_propagation_fn,
 * forward_propagation_fn - given the nearest neighbor and a new target state, returns the states that would grow the tree towards the target
 * SHOULD : collosion checking, constraint violation checking
 Determine the parent index of the new state
 This process deserves some explanation
 The "current relative parent index" is the index of the parent, relative to the list of propagated nodes.
 A negative value means the nearest neighbor in the tree, zero means the first propagated node, and so on.
 NOTE - the relative parent index *must* be lower than the index in the list of prograted nodes
 * i.e. the first node must have a negative value, and so on.
 */
std::vector<std::pair<RRTHelper::RRTConfig, int64_t>> RRTHelper::forwardPropogationFunction(
        const RRTConfig& nearest_neighbor,
        const RRTConfig& random_target)
{
//    vis_.visualizeCubes("forward_propogation_start_config", {nearest_neighbor.first.first, nearest_neighbor.first.second}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), 1);
//    vis_.visualizePoints("random_target", {random_target.first.first, random_target.first.second}, Visualizer::Red(), 1);
    vis_.deleteObjects("rubber_band_post_rrt_step");

    std::vector<std::pair<RRTConfig, int64_t>> propagated_states;
    int64_t parent_offset = -1;

    const double total_distance = distance(nearest_neighbor, random_target);
    const uint32_t max_total_steps = (uint32_t)ceil(total_distance / step_size_);
    propagated_states.reserve(max_total_steps);

    const bool rubber_band_verbose = false;
    uint32_t step_index = 0;
    while (step_index < max_total_steps)
    {
        // Using ternary operator here so that we can avoid making copies, and still take advantage of const correctness
        const bool use_nearest_neighbour_as_prev = (parent_offset == -1);
        const RRTConfig& prev_node = (use_nearest_neighbour_as_prev ? nearest_neighbor : propagated_states[parent_offset].first);

        const double ratio = std::min(1.0, (double)(step_index + 1) * step_size_ / total_distance);
        const Eigen::Vector3d gripper_a_interpolated = EigenHelpers::Interpolate(nearest_neighbor.first.first, random_target.first.first, ratio);
        const Eigen::Vector3d gripper_b_interpolated = EigenHelpers::Interpolate(nearest_neighbor.first.second, random_target.first.second, ratio);

        // If the grippers enter collision, then return however far we were able to get
        if ((environment_sdf_.Get3d(gripper_a_interpolated) < 0.0)
                || (environment_sdf_.Get3d(gripper_b_interpolated) < 0.0))
        {
            break;
        }

        VirtualRubberBand next_rubber_band(prev_node.second);
        next_rubber_band.forwardSimulateVirtualRubberBand(
                    gripper_a_interpolated - prev_node.first.first,
                    gripper_b_interpolated - prev_node.first.second,
                    rubber_band_verbose);

        next_rubber_band.visualize("rubber_band_post_rrt_step", rubber_band_safe_color_, rubber_band_overstretched_color_, (int32_t)parent_offset + 2, true);

        // If the rubber band becomes overstretched, then return however far we were able to get
        if (next_rubber_band.isOverstretched())
        {
            break;
        }

        const RRTConfig next_node(std::make_pair(gripper_a_interpolated, gripper_b_interpolated), next_rubber_band);
//                    vis_.visualizeLineStrip("rrt_gripper_tree_gripper_a", {prev_node.first.first, next_node.first.first}, Visualizer::Red(), marker_id_);
//                    vis_.visualizeLineStrip("rrt_gripper_tree_gripper_b", {prev_node.first.second, next_node.first.second}, Visualizer::Blue(), marker_id_);
        propagated_states.push_back(std::pair<RRTConfig, int64_t>(next_node, parent_offset));

        ++marker_id_;
        ++parent_offset;
        ++step_index;
    }

    return propagated_states;
}

std::vector<RRTHelper::RRTConfig, RRTHelper::Allocator> RRTHelper::rrtPlan(
        const RRTConfig& start,
        const RRTConfig& goal,
        const std::chrono::duration<double>& time_limit)
{
    gripper_goal_positions_ = goal.first;

    const std::function<bool(const RRTConfig&)> goal_reached_fn = [&] (const RRTConfig& node)
    {
        if (distance(node, goal) < goal_reach_radius_)
        {
            // Only accept paths that are different from those on the blacklist
            if (!isBandFirstOrderVisibileToBlacklist(node.second.getVectorRepresentation()))
            {
                return true;
            }
        }
        return false;
    };

    const std::function<RRTConfig(void)> sampling_fn = [&] ()
    {
        RRTConfig sample_config(posPairSampling(), start.second);
        return sample_config;
    };

    const std::function<int64_t(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, Allocator>>&, const RRTConfig&)> nearest_neighbor_fn = [&] (
            const std::vector<simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, Allocator>>& nodes,
            const RRTConfig& config )
    {
        return nearestNeighbour(nodes, config);
    };

    const std::function<std::vector<std::pair<RRTConfig, int64_t>>(const RRTConfig&, const RRTConfig&)> forward_propagation_fn = [&] (
            const RRTConfig& nearest_neighbor,
            const RRTConfig& random_target )
    {
        return forwardPropogationFunction(nearest_neighbor, random_target);
    };


    const auto rrt_results = simple_rrt_planner::SimpleHybridRRTPlanner::Plan(
                start,
                nearest_neighbor_fn,
                goal_reached_fn,
                sampling_fn,
                forward_propagation_fn,
                time_limit);

    const auto smoothing_results = rrtShortcutSmooth(rrt_results.first);

    std::cout << PrettyPrint::PrettyPrint(rrt_results.second, true, "\n") << std::endl;
    std::cout << PrettyPrint::PrettyPrint(smoothing_results.second, true, "\n") << std::endl;

    return smoothing_results.first;
}

///////////////////////////////////////////////////////////////////////////////////////
// Helper function for goal check
///////////////////////////////////////////////////////////////////////////////////////

void RRTHelper::addBandToBlacklist(const EigenHelpers::VectorVector3d& band)
{
    blacklisted_goal_rubber_bands_.push_back(band);
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band) const
{
    for (size_t idx = 0; idx < blacklisted_goal_rubber_bands_.size(); idx++)
    {
        const EigenHelpers::VectorVector3d& blacklisted_path = blacklisted_goal_rubber_bands_[idx];

        // Checks if the straight line between elements of the two paths is collision free
        const auto straight_line_collision_check_fn = [&] (const ssize_t blacklisted_path_ind, const ssize_t test_band_ind)
        {
            assert(0 <= blacklisted_path_ind && blacklisted_path_ind < (ssize_t)blacklisted_path.size());
            assert(0 <= test_band_ind && test_band_ind < (ssize_t)test_band.size());

            const Eigen::Vector3d& first_node = blacklisted_path[blacklisted_path_ind];
            const Eigen::Vector3d& second_node = test_band[test_band_ind];

            const ssize_t num_steps = (ssize_t)std::ceil((second_node - first_node).norm() / environment_sdf_.GetResolution());

            // We don't need to check the endpoints as they are already checked as part of the rubber band process
            for (ssize_t ind = 1; ind < num_steps; ++ind)
            {
                const double ratio = (double)ind / (double)num_steps;
                const Eigen::Vector3d interpolated_point = EigenHelpers::Interpolate(first_node, second_node, ratio);
                if (environment_sdf_.Get3d(interpolated_point) < 0.0)
                {
                    return false;
                }
            }

            return true;
        };

        // If we've found a first order deformation, then we are similar to a blacklisted item
        const bool visualize = false;
        if (arc_utilities::FirstOrderDeformation::CheckFirstOrderDeformation(blacklisted_path.size(), test_band.size(), straight_line_collision_check_fn, visualize))
        {
            return true;
        }
    }

    return false;
}

///////////////////////////////////////////////////////////////////////////////////////
// Helper function for shortcut smoothing
///////////////////////////////////////////////////////////////////////////////////////

static bool gripperPositionsAreApproximatelyEqual(
        const std::pair<Eigen::Vector3d, Eigen::Vector3d>& a_node,
        const std::pair<Eigen::Vector3d, Eigen::Vector3d>& b_node)
{
    const Eigen::Vector3d& transA_first = a_node.first;
    const Eigen::Vector3d& transB_first = b_node.first;

    const Eigen::Vector3d& transA_second = a_node.second;
    const Eigen::Vector3d& transB_second = b_node.second;

    bool is_equal = true;
    is_equal &= transA_first.isApprox(transB_first);
    is_equal &= transA_second.isApprox(transB_second);
    return is_equal;
}

static bool gripperPositionsAreApproximatelyEqual(
        const RRTHelper::RRTConfig& a_node,
        const RRTHelper::RRTConfig& b_node)
{
    return gripperPositionsAreApproximatelyEqual(a_node.first, b_node.first);
}

static bool bandEndpointsMatchGripperPositions(
        const VirtualRubberBand& band,
        const std::pair<Eigen::Vector3d, Eigen::Vector3d> grippers)
{
    const EigenHelpers::VectorVector3d& rubber_band_pos = band.getVectorRepresentation();
    const auto rubber_band_endpoints = std::make_pair(rubber_band_pos.front(), rubber_band_pos.back());
    return gripperPositionsAreApproximatelyEqual(grippers, rubber_band_endpoints);
}

static std::pair<bool, std::vector<RRTHelper::RRTConfig, RRTHelper::Allocator>> forwardSimulateGrippersPath(
        VirtualRubberBand rubber_band,
        const std::vector<RRTHelper::RRTConfig, RRTHelper::Allocator>& path,
        const size_t start_index,
        const size_t end_index)
{
    assert(start_index <= end_index);
    assert(end_index <= path.size() - 1);

    // Verify that the endpoints of the rubber band match the start of the grippers path
    assert(bandEndpointsMatchGripperPositions(rubber_band, path[start_index].first));

    // Collect the results for use by the rrtShortcutSmooth function
    std::vector<RRTHelper::RRTConfig, RRTHelper::Allocator> resulting_path;
    resulting_path.push_back(RRTHelper::RRTConfig(path[start_index].first, rubber_band));

    bool band_is_overstretched = rubber_band.isOverstretched();
    size_t path_idx = start_index + 1;
    const bool rubber_band_verbose = false;
    while (!band_is_overstretched && path_idx <= end_index)
    {
        // Forward simulate the band
        const auto& starting_grippers_pos = path[path_idx - 1].first;
        const auto& ending_grippers_pos = path[path_idx].first;
        rubber_band.forwardSimulateVirtualRubberBand(
                    ending_grippers_pos.first - starting_grippers_pos.first,
                    ending_grippers_pos.second - starting_grippers_pos.second,
                    rubber_band_verbose);

        rubber_band.visualize("rubber_band_checking_rest_of_path", Visualizer::Yellow(), Visualizer::Cyan(), (int32_t)path_idx, true);

        // Store the band in the results
        resulting_path.push_back(RRTHelper::RRTConfig(ending_grippers_pos, rubber_band));

        // Record if the band is overstretched
        band_is_overstretched = rubber_band.isOverstretched();
        ++path_idx;
    }

    // Double check that if the band is not overstetched, then we've reached the endpoint of the path
    if (!band_is_overstretched)
    {
        const EigenHelpers::VectorVector3d ending_rubber_band_pos = rubber_band.getVectorRepresentation();
        const auto rubber_band_endpoints = std::make_pair(ending_rubber_band_pos.front(), ending_rubber_band_pos.back());
        assert(gripperPositionsAreApproximatelyEqual(path[end_index].first, rubber_band_endpoints));
    }

    return std::make_pair(band_is_overstretched, resulting_path);
}

std::pair<std::vector<RRTHelper::RRTConfig, RRTHelper::Allocator>, std::map<std::string, double>> RRTHelper::rrtShortcutSmooth(
        const std::vector<RRTConfig, Allocator>& path)
{
    std::map<std::string, double> statistics;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    std::vector<RRTConfig, Allocator> current_path = path;
    uint32_t num_iterations = 0;
    uint32_t failed_iterations = 0;

    visualize(current_path);
    while (num_iterations < max_smoothing_iterations_ && failed_iterations < max_failed_smoothing_iterations_ && current_path.size() > 2)
    {
//        usleep(100000);
        vis_.deleteObjects("rubber_band_checking_rest_of_path");
        vis_.deleteObjects("rubber_band_post_rrt_step");
//        usleep(100000);

        // Attempt a shortcut
        const int64_t base_index = (int64_t)std::uniform_int_distribution<size_t>(0, current_path.size() - 1)(generator_);

        // Compute the offset index
        // We want to sample the start and goal slightly more frequently, so allow "overshoots" of endpoints for the offset
        const int64_t min_delta = std::max(-base_index - max_shortcut_index_distance_ / 10, -max_shortcut_index_distance_);
        const int64_t max_delta = std::min((int64_t)current_path.size() - base_index - 1 + max_shortcut_index_distance_ / 10, max_shortcut_index_distance_);
        const int64_t offset_delta = std::uniform_int_distribution<int64_t>(min_delta, max_delta)(generator_);
        // Clamp to the boundaries of the current path
        const int64_t second_index = arc_helpers::ClampValue(base_index + offset_delta, (int64_t)0, (int64_t)current_path.size() - 1);

        // Get start & end indices
        const size_t start_index = (size_t)std::min(base_index, second_index);
        const size_t end_index = (size_t)std::max(base_index, second_index);

        // We know start_index <= end_index, this essentially checks if start+1 == end or start == end
        if (start_index + 1 >= end_index)
        {
            ++num_iterations;
            continue;
        }

        // Check if the edge is valid
        const RRTConfig& start_config = current_path[start_index];
        const EigenHelpers::VectorVector3d& start_band = start_config.second.getVectorRepresentation();
        const RRTConfig& end_config = current_path[end_index];
        const EigenHelpers::VectorVector3d& end_band = end_config.second.getVectorRepresentation();

        start_config.second.visualize("rubber_band_smoothing_start", Visualizer::Red(), rubber_band_overstretched_color_, 1, true);
        vis_.visualizeCubes("rubber_band_smoothing_start", {start_band.front(), start_band.back()}, Eigen::Vector3d(0.02, 0.02, 0.02), Visualizer::Red(), 1000);
        end_config.second.visualize("rubber_band_smoothing_end", Visualizer::Red(), rubber_band_overstretched_color_, 1, true);
        vis_.visualizeCubes("rubber_band_smoothing_end", {end_band.front(), end_band.back()}, Eigen::Vector3d(0.02, 0.02, 0.02), Visualizer::Red(), 1000);

        const std::vector<std::pair<RRTHelper::RRTConfig, int64_t>> smoothing_propogation_results = forwardPropogationFunction(start_config, end_config);
        const bool edge_valid =
                smoothing_propogation_results.size() > 0 &&
                !smoothing_propogation_results.back().first.second.isOverstretched() &&
                bandEndpointsMatchGripperPositions(smoothing_propogation_results.back().first.second, current_path[end_index].first);
        if (edge_valid)
        {
            // We still need to check that the rubber band can still reach the goal correctly from this state,
            // so we'll forward propogate along the rest of the trajectory to check feasibility
            const VirtualRubberBand& final_rubber_band_after_smoothing = smoothing_propogation_results.back().first.second;
            const std::pair<bool, std::vector<RRTHelper::RRTConfig, RRTHelper::Allocator>> end_of_smoothing_to_goal_results
                    = forwardSimulateGrippersPath(
                        final_rubber_band_after_smoothing,
                        current_path,
                        end_index,
                        current_path.size() - 1);

            // If we were able to forward simulate without overstretch, then check if the resulting rubber band has been blacklisted
            if (end_of_smoothing_to_goal_results.first == false)
            {
                if (!isBandFirstOrderVisibileToBlacklist(end_of_smoothing_to_goal_results.second.back().second.getVectorRepresentation()))
                {
                    // Allocate space for the total smoothed path
                    std::vector<RRTConfig, Allocator> smoothed_path;
                    smoothed_path.reserve(current_path.size() - (end_index - start_index) + smoothing_propogation_results.size());

                    // Insert the starting unchanged part of the path
                    smoothed_path.insert(smoothed_path.begin(), current_path.begin(), current_path.begin() + start_index + 1);

                    // Insert the smoothed portion
                    for (size_t idx = 0; idx < smoothing_propogation_results.size(); ++idx)
                    {
                        smoothed_path.push_back(smoothing_propogation_results[idx].first);
                    }

                    // Insert the changed end of the path with the new rubber band - gripper positions are identical
                    smoothed_path.insert(smoothed_path.end(), end_of_smoothing_to_goal_results.second.begin() + 1, end_of_smoothing_to_goal_results.second.end());

                    // Record the change and re-visualize
                    current_path = smoothed_path;
                    visualize(current_path);
                }
            }
            else
            {
                ++failed_iterations;
            }
        }
        else
        {
            ++failed_iterations;
        }

        ++num_iterations;
    }

    // Record the statistics and return the result
    std::chrono::time_point<std::chrono::high_resolution_clock> cur_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> smoothing_time(cur_time - start_time);

    statistics["smoothing_time"] = smoothing_time.count();
    statistics["num_iterations"] = (double)num_iterations;
    statistics["failed_iterations"] = (double)failed_iterations;

    return std::make_pair(current_path, statistics);
}


///////////////////////////////////////////////////////////////////////////////////////
// Visualization and other debugging tools
///////////////////////////////////////////////////////////////////////////////////////


void RRTHelper::visualize(const std::vector<RRTConfig, Allocator>& path) const
{
    vis_.deleteObjects("rubber_band_rrt_solution");
    vis_.deleteObjects("gripper_a_rrt_solution");
    vis_.deleteObjects("gripper_b_rrt_solution");
//    usleep(100000);

    for (size_t ind = 0; ind < path.size(); ++ind)
    {
        const RRTConfig& config = path[ind];
        const auto& gripper_positions = config.first;
        const VirtualRubberBand& rubber_band = config.second;

        rubber_band.visualize("rubber_band_rrt_solution", Visualizer::Yellow(), rubber_band_overstretched_color_, (int32_t)ind + 1, true);
        vis_.visualizeCubes("gripper_a_rrt_solution", {gripper_positions.first}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Blue(), (int32_t)ind + 1);
        vis_.visualizeCubes("gripper_b_rrt_solution", {gripper_positions.second}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), (int32_t)ind + 1);
    }
}
