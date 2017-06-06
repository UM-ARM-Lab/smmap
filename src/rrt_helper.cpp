#include "smmap/rrt_helper.h"

#include <thread>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>

using namespace smmap;


///////////////////////////////////////////////////////////////////////////////////////
// Helper function for assertion testing
///////////////////////////////////////////////////////////////////////////////////////

static bool gripperPositionsAreApproximatelyEqual(
        const RRTGrippersRepresentation& c1,
        const RRTGrippersRepresentation& c2)
{
    const Eigen::Vector3d& c1_first_gripper     = c1.first;
    const Eigen::Vector3d& c1_second_gripper    = c1.second;
    const Eigen::Vector3d& c2_first_gripper     = c2.first;
    const Eigen::Vector3d& c2_second_gripper    = c2.second;

    bool is_equal = true;
    is_equal &= c1_first_gripper.isApprox(c2_first_gripper);
    is_equal &= c1_second_gripper.isApprox(c2_second_gripper);
    return is_equal;
}

static bool gripperPositionsAreApproximatelyEqual(
        const RRTConfig& c1,
        const RRTConfig& c2)
{
    return gripperPositionsAreApproximatelyEqual(c1.getGrippers(), c2.getGrippers());
}

static bool bandEndpointsMatchGripperPositions(
        const VirtualRubberBand& band,
        const RRTGrippersRepresentation grippers)
{
    const EigenHelpers::VectorVector3d& rubber_band_pos = band.getVectorRepresentation();
    const RRTGrippersRepresentation rubber_band_endpoints(rubber_band_pos.front(), rubber_band_pos.back());
    return gripperPositionsAreApproximatelyEqual(grippers, rubber_band_endpoints);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTConfig functions                      ////////////////////
///////////////////////////////////////////////////////////////////////////////////////

RRTConfig::RRTConfig(
        const RRTGrippersRepresentation& grippers_position,
        const VirtualRubberBand& band,
        const bool is_visible_to_blacklist)
    : grippers_position_(grippers_position)
    , band_(band)
    , is_visible_to_blacklist_(is_visible_to_blacklist)
{}

const RRTGrippersRepresentation& RRTConfig::getGrippers() const
{
    return grippers_position_;
}

const VirtualRubberBand& RRTConfig::getBand() const
{
    return band_;
}

bool RRTConfig::isVisibleToBlacklist() const
{
    return is_visible_to_blacklist_;
}

// Returned distance is the Euclidian distance of two grippers pos
double RRTConfig::distance(const RRTConfig& other) const
{
    return RRTConfig::Distance(*this, other);
}

double RRTConfig::Distance(const RRTConfig& c1, const RRTConfig& c2)
{
    return RRTConfig::Distance(c1.getGrippers(), c2.getGrippers());
}

double RRTConfig::Distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2)
{
    const Eigen::Vector3d& c1_first_gripper     = c1.first;
    const Eigen::Vector3d& c1_second_gripper    = c1.second;
    const Eigen::Vector3d& c2_first_gripper     = c2.first;
    const Eigen::Vector3d& c2_second_gripper    = c2.second;
    return std::sqrt((c1_first_gripper - c2_first_gripper).squaredNorm() + (c1_second_gripper - c2_second_gripper).squaredNorm());
}

bool RRTConfig::operator==(const RRTConfig& other) const
{
    if (!gripperPositionsAreApproximatelyEqual(grippers_position_, other.grippers_position_))
    {
        return false;
    }

    const auto& this_band_as_vector = band_.getVectorRepresentation();
    const auto& other_band_as_vector = other.band_.getVectorRepresentation();
    if (this_band_as_vector.size() != other_band_as_vector.size())
    {
        return false;
    }

    for (size_t idx = 0; idx < this_band_as_vector.size(); ++idx)
    {
        if (!this_band_as_vector[idx].isApprox(other_band_as_vector[idx]))
        {
            return false;
        }
    }

    return true;
}


std::size_t std::hash<smmap::RRTConfig>::operator()(const smmap::RRTConfig& rrt_config) const
{
    std::size_t seed = 0;
    std::hash_combine(seed, rrt_config.getGrippers());

    const EigenHelpers::VectorVector3d& band_ = rrt_config.getBand().getVectorRepresentation();
    for (size_t idx = 0; idx < band_.size(); ++idx)
    {
        std::hash_combine(seed, band_[idx]);
    }
    return seed;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTHelper functions                      ////////////////////
///////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////////////////////////////////////////

RRTHelper::RRTHelper(
        const sdf_tools::SignedDistanceField& environment_sdf,
        const Visualizer& vis,
        std::mt19937_64& generator,
        const double x_limits_lower,
        const double x_limits_upper,
        const double y_limits_lower,
        const double y_limits_upper,
        const double z_limits_lower,
        const double z_limits_upper,
        const double step_size,
        const double goal_reach_radius,
        const double homotopy_distance_penalty,
        const int64_t max_shortcut_index_distance,
        const uint32_t max_smoothing_iterations,
        const uint32_t max_failed_smoothing_iterations)
    : x_limits_(x_limits_lower,x_limits_upper)
    , y_limits_(y_limits_lower,y_limits_upper)
    , z_limits_(z_limits_lower,z_limits_upper)
    , step_size_(step_size)
    , goal_reach_radius_(goal_reach_radius)
    , homotopy_distance_penalty_(homotopy_distance_penalty)
    , max_shortcut_index_distance_(max_shortcut_index_distance)
    , max_smoothing_iterations_(max_smoothing_iterations)
    , max_failed_smoothing_iterations_(max_failed_smoothing_iterations)
    , uniform_unit_distribution_(0.0, 1.0)
    , environment_sdf_(environment_sdf)
    , vis_(vis)
    , generator_(generator)
    , rubber_band_safe_color_(Visualizer::Black())
    , rubber_band_overstretched_color_(Visualizer::Cyan())
{
    assert(max_shortcut_index_distance > 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for extral RRT planning class
///////////////////////////////////////////////////////////////////////////////////////////////

int64_t RRTHelper::nearestNeighbour(
        const std::vector<ExternalRRTState>& nodes,
        const RRTConfig& config)
{
    const std::function<double(const ExternalRRTState&, const RRTConfig&)> basic_distance_fn = [&] (
            const ExternalRRTState& rrt_state,
            const RRTConfig& rrt_config)
    {
        return RRTConfig::Distance(rrt_state.GetValueImmutable(), rrt_config);
    };

    // If we sampled the goal, then distance is also a function of "homotopy class"
    const std::function<double(const ExternalRRTState&, const RRTConfig&)> goal_sampled_distance_fn = [&] (
            const ExternalRRTState& rrt_state,
            const RRTConfig& rrt_config)
    {
        const auto blacklist_itr = goal_expansion_nn_blacklist_.find(rrt_state.GetValueImmutable());
        const bool goal_blacklisted = blacklist_itr != goal_expansion_nn_blacklist_.end();
        if (goal_blacklisted)
        {
            return std::numeric_limits<double>::infinity();
        }
        else
        {
            const double basic_distance = RRTConfig::Distance(rrt_state.GetValueImmutable(), rrt_config);
            const double homotopy_penalty = rrt_state.GetValueImmutable().isVisibleToBlacklist() ? homotopy_distance_penalty_ : 0.0;
            return basic_distance + homotopy_penalty;
        }
    };

    // Determine which distance function to use
    const bool goal_is_target_config = gripperPositionsAreApproximatelyEqual(grippers_goal_position_, config.getGrippers());
    const auto distance_fn = goal_is_target_config ? goal_sampled_distance_fn : basic_distance_fn;

    const size_t K = 1;
    const auto nn_results = arc_helpers::GetKNearestNeighbors(nodes, config, distance_fn, K);
    const int64_t nn_idx = nn_results[0].first;

    // Blacklist this config from being selected again as the nearest neighbour to the goal
    if (goal_is_target_config)
    {
        goal_expansion_nn_blacklist_.insert(nodes[nn_idx].GetValueImmutable());
    }

    return nn_idx;
}

// const std::function<T(void)>& sampling_fn,
inline RRTGrippersRepresentation RRTHelper::posPairSampling()
{
    RRTGrippersRepresentation rand_sample;
    const bool sample_goal = uniform_unit_distribution_(generator_) < 0.1;

    if (sample_goal)
    {
        rand_sample = grippers_goal_position_;
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


/*
 * forward_propagation_fn - given the nearest neighbor and a new target state, returns the states that would grow the tree towards the target
 * SHOULD : collision checking, constraint violation checking
            Determine the parent index of the new state
 This process deserves some explanation
 The "current relative parent index" is the index of the parent, relative to the list of propagated nodes.
 A negative value means the nearest neighbor in the tree, zero means the first propagated node, and so on.
 NOTE - the relative parent index *must* be lower than the index in the list of prograted nodes
 * i.e. the first node must have a negative value, and so on.
 */
std::vector<std::pair<RRTConfig, int64_t>> RRTHelper::forwardPropogationFunction(
        const RRTConfig& nearest_neighbor,
        const RRTConfig& random_target) const
{
//    static int32_t marker_id = 1;

    // Visualization
    {
        vis_.visualizeCubes(
                    "forward_propogation_start_config",
                    {nearest_neighbor.getGrippers().first},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    Visualizer::Magenta(),
                    1);
        vis_.visualizeCubes(
                    "forward_propogation_start_config",
                    {nearest_neighbor.getGrippers().second},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    Visualizer::Cyan(),
                    5);

        nearest_neighbor.getBand().visualize(
                    "forward_propogation_start_config",
                    Visualizer::Green(),
                    Visualizer::Green(),
                    10,
                    true);

        vis_.visualizeCubes(
                    "random_target",
                    {random_target.getGrippers().first},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    Visualizer::Magenta(),
                    1);
        vis_.visualizeCubes(
                    "random_target",
                    {random_target.getGrippers().second},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    Visualizer::Cyan(),
                    5);
    }

    vis_.deleteObjects("rubber_band_post_rrt_step");

    std::vector<std::pair<RRTConfig, int64_t>> propagated_states;
    int64_t parent_offset = -1;

    const double total_distance = RRTConfig::Distance(nearest_neighbor, random_target);
    const uint32_t max_total_steps = (uint32_t)ceil(total_distance / step_size_);
    propagated_states.reserve(max_total_steps);

    const RRTGrippersRepresentation& starting_grippers_position = nearest_neighbor.getGrippers();
    const RRTGrippersRepresentation& target_grippers_position = random_target.getGrippers();

//    if (gripperPositionsAreApproximatelyEqual(grippers_goal_position_, target_grippers_position))
//    {
//        std::cerr << "Propogating towards goal:\n";
//        std::cerr << "Goal:   " << PrettyPrint::PrettyPrint(grippers_goal_position_, true, " ") << std::endl;
//        std::cerr << "Target: " << PrettyPrint::PrettyPrint(target_grippers_position, true, " ") << std::endl;
//        std::cerr << std::flush;
//    }

    const bool rubber_band_verbose = false;
    uint32_t step_index = 0;
    while (step_index < max_total_steps)
    {
        // Using ternary operator here so that we can avoid making copies, and still take advantage of const correctness
        const bool use_nearest_neighbour_as_prev = (parent_offset == -1);
        const RRTConfig& prev_node = (use_nearest_neighbour_as_prev ? nearest_neighbor : propagated_states[parent_offset].first);
        const VirtualRubberBand& prev_rubber_band = prev_node.getBand();

        const double ratio = std::min(1.0, (double)(step_index + 1) * step_size_ / total_distance);
        const Eigen::Vector3d gripper_a_interpolated = EigenHelpers::Interpolate(starting_grippers_position.first, target_grippers_position.first, ratio);
        const Eigen::Vector3d gripper_b_interpolated = EigenHelpers::Interpolate(starting_grippers_position.second, target_grippers_position.second, ratio);
        const RRTGrippersRepresentation next_grippers_position(gripper_a_interpolated, gripper_b_interpolated);

        // If the grippers enter collision, then return however far we were able to get
        if ((environment_sdf_.Get3d(gripper_a_interpolated) < 0.0) || (environment_sdf_.Get3d(gripper_b_interpolated) < 0.0))
        {
            break;
        }

        VirtualRubberBand next_rubber_band(prev_rubber_band);
        next_rubber_band.forwardSimulateVirtualRubberBandToEndpointTargets(
                    next_grippers_position.first,
                    next_grippers_position.second,
                    rubber_band_verbose);

        // If the rubber band was "pushed" out of collision at the endpoints during the trajectory,
        // then allow the band to resettle on the grippers position
        if (!bandEndpointsMatchGripperPositions(next_rubber_band, next_grippers_position))
        {
            next_rubber_band.forwardSimulateVirtualRubberBandToEndpointTargets(
                        next_grippers_position.first,
                        next_grippers_position.second,
                        rubber_band_verbose);
        }
        // If we are still able to get to the next target position after retrying,
        // then return however far we were able to get
        if (!bandEndpointsMatchGripperPositions(next_rubber_band, next_grippers_position))
        {
            break;
        }

        next_rubber_band.visualize("rubber_band_post_rrt_step", rubber_band_safe_color_, rubber_band_overstretched_color_, (int32_t)parent_offset + 2, true);

        // If the rubber band becomes overstretched, then return however far we were able to get
        if (next_rubber_band.isOverstretched())
        {
            break;
        }

        const RRTConfig next_node(
                    next_grippers_position,
                    next_rubber_band,
                    isBandFirstOrderVisibileToBlacklist(next_rubber_band));
        propagated_states.push_back(std::pair<RRTConfig, int64_t>(next_node, parent_offset));

//        if (next_node.isVisibleToBlacklist())
//        {
//            vis_.visualizeLineStrip("rrt_gripper_tree_gripper_a", {prev_grippers_position.first, gripper_a_interpolated}, Visualizer::Magenta(), marker_id);
//            vis_.visualizeLineStrip("rrt_gripper_tree_gripper_b", {prev_grippers_position.second, gripper_b_interpolated}, Visualizer::Cyan(), marker_id);
//            ++marker_id;
//        }
//        else
//        {
//            vis_.visualizeLineStrip("rrt_gripper_tree_gripper_a", {prev_grippers_position.first, gripper_a_interpolated}, Visualizer::Red(), marker_id);
//            vis_.visualizeLineStrip("rrt_gripper_tree_gripper_b", {prev_grippers_position.second, gripper_b_interpolated}, Visualizer::Blue(), marker_id);
//            ++marker_id;
//        }
//        std::this_thread::sleep_for(std::chrono::duration<double>(0.01));

        ++parent_offset;
        ++step_index;
    }

    return propagated_states;
}

/*
 * Builds the helper functions needed by simple_rrt_planner and invokes the planner (and shortcut smoother)
 */
std::vector<RRTConfig, RRTAllocator> RRTHelper::rrtPlan(
        const RRTConfig& start,
        const RRTGrippersRepresentation& grippers_goal,
        const std::chrono::duration<double>& time_limit)
{
    visualizeBlacklist();

    grippers_goal_position_ = grippers_goal;

    const std::function<bool(const RRTConfig&)> goal_reached_fn = [&] (const RRTConfig& node)
    {
        if (RRTConfig::Distance(node.getGrippers(), grippers_goal) < goal_reach_radius_)
        {
            // Only accept paths that are different from those on the blacklist
            if (!node.isVisibleToBlacklist())
            {
                return true;
            }
        }
        return false;
    };

    const std::function<RRTConfig(void)> sampling_fn = [&] ()
    {
        const RRTConfig sample_config(posPairSampling(), start.getBand(), false);
        return sample_config;
    };

    const std::function<int64_t(const std::vector<ExternalRRTState>&, const RRTConfig&)> nearest_neighbor_fn = [&] (
            const std::vector<ExternalRRTState>& nodes,
            const RRTConfig& config )
    {
        const int64_t neighbour_idx = nearestNeighbour(nodes, config);
        return neighbour_idx;
    };

    const std::function<std::vector<std::pair<RRTConfig, int64_t>>(const RRTConfig&, const RRTConfig&)> forward_propagation_fn = [&] (
            const RRTConfig& nearest_neighbor,
            const RRTConfig& random_target )
    {
        const std::vector<std::pair<RRTConfig, int64_t>> propotation_results =
                forwardPropogationFunction(nearest_neighbor, random_target);
        return propotation_results;
    };


    const auto rrt_results = simple_rrt_planner::SimpleHybridRRTPlanner::Plan(
                start,
                nearest_neighbor_fn,
                goal_reached_fn,
                sampling_fn,
                forward_propagation_fn,
                time_limit);

    const auto smoothing_results = rrtShortcutSmooth(rrt_results.first);

    std::cout << "SimpleRRT Statistics:\n" << PrettyPrint::PrettyPrint(rrt_results.second, true, "\n") << std::endl;
    std::cout << "ShortcutSmoothing Statistics:\n" << PrettyPrint::PrettyPrint(smoothing_results.second, true, "\n") << std::endl;

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
    static ros::NodeHandle nh;
    static Visualizer vis(nh, "first_order_visibility_visualization_marker", "first_order_visibility_visualization_marker_vector");

    EigenHelpers::VectorVector3d collision_points;
    const bool visualize = false;
    if (visualize)
    {
        vis.visualizePoints(
                    "first_order_vis_collision",
                    collision_points,
                    Visualizer::Red(),
                    1,
                    1.0);
        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
    }

    for (size_t idx = 0; idx < blacklisted_goal_rubber_bands_.size(); idx++)
    {
        const EigenHelpers::VectorVector3d& blacklisted_path = blacklisted_goal_rubber_bands_[idx];

        // Checks if the straight line between elements of the two paths is collision free
        const auto straight_line_collision_check_fn = [&] (
                const ssize_t blacklisted_path_ind,
                const ssize_t test_band_ind)
        {
            assert(0 <= blacklisted_path_ind && blacklisted_path_ind < (ssize_t)blacklisted_path.size());
            assert(0 <= test_band_ind && test_band_ind < (ssize_t)test_band.size());

            const Eigen::Vector3d& first_node = blacklisted_path[blacklisted_path_ind];
            const Eigen::Vector3d& second_node = test_band[test_band_ind];

            if (visualize)
            {
                vis_.visualizeLineStrip("first_order_vis_check", {first_node, second_node}, Visualizer::White(), 2);
            }

            const ssize_t num_steps = (ssize_t)std::ceil((second_node - first_node).norm() / environment_sdf_.GetResolution());

            // We don't need to check the endpoints as they are already checked as part of the rubber band process
            for (ssize_t ind = 1; ind < num_steps; ++ind)
            {
                const double ratio = (double)ind / (double)num_steps;
                const Eigen::Vector3d interpolated_point = EigenHelpers::Interpolate(first_node, second_node, ratio);
                if (environment_sdf_.Get3d(interpolated_point) < 0.0)
                {
                    if (visualize)
                    {
                        collision_points.push_back(Eigen::Vector3d((double)blacklisted_path_ind, (double)test_band_ind, 0.0));
                        vis.visualizePoints(
                                    "first_order_vis_collision",
                                    collision_points,
                                    Visualizer::Red(),
                                    1,
                                    1.0);
                        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
                    }
                    return false;
                }
            }

            return true;
        };

        // If we've found a first order deformation, then we are similar to a blacklisted item
        if (arc_utilities::FirstOrderDeformation::CheckFirstOrderDeformation(
                blacklisted_path.size(),
                test_band.size(),
                straight_line_collision_check_fn,
                visualize))
        {
            return true;
        }
    }

    return false;
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const VirtualRubberBand& test_band) const
{
    return isBandFirstOrderVisibileToBlacklist(test_band.getVectorRepresentation());
}

///////////////////////////////////////////////////////////////////////////////////////
// Helper function for shortcut smoothing
///////////////////////////////////////////////////////////////////////////////////////

std::pair<bool, std::vector<RRTConfig, RRTAllocator>> RRTHelper::forwardSimulateGrippersPath(
        VirtualRubberBand rubber_band,
        const std::vector<RRTConfig, RRTAllocator>& path,
        const size_t start_index,
        const size_t end_index) const
{
    assert(start_index <= end_index);
    assert(end_index <= path.size() - 1);

    // Verify that the endpoints of the rubber band match the start of the grippers path
    if (!bandEndpointsMatchGripperPositions(rubber_band, path[start_index].getGrippers()))
    {
        std::cout << "Inside forwardSimulateGrippersPath\n";
        std::cout << "initial rubber band endpoints:\n"
                  << PrettyPrint::PrettyPrint(rubber_band.getVectorRepresentation().front()) << "        "
                  << PrettyPrint::PrettyPrint(rubber_band.getVectorRepresentation().back()) << std::endl;

        std::cout << "path gripper positions:\n"
                  << PrettyPrint::PrettyPrint(path[start_index].getGrippers().first) << "        "
                  << PrettyPrint::PrettyPrint(path[start_index].getGrippers().second) << std::endl;

        assert(bandEndpointsMatchGripperPositions(rubber_band, path[start_index].getGrippers()));
    }

    // Collect the results for use by the rrtShortcutSmooth function
    std::vector<RRTConfig, RRTAllocator> resulting_path;
    resulting_path.push_back(RRTConfig(path[start_index].getGrippers(), rubber_band, isBandFirstOrderVisibileToBlacklist(rubber_band)));

    bool band_is_overstretched = rubber_band.isOverstretched();
    size_t path_idx = start_index + 1;
    const bool rubber_band_verbose = false;
    while (!band_is_overstretched && path_idx <= end_index)
    {
        // Forward simulate the band
        const auto& ending_grippers_pos = path[path_idx].getGrippers();
        rubber_band.forwardSimulateVirtualRubberBandToEndpointTargets(
                    ending_grippers_pos.first,
                    ending_grippers_pos.second,
                    rubber_band_verbose);

        rubber_band.visualize("rubber_band_checking_rest_of_path", Visualizer::Yellow(), Visualizer::Cyan(), (int32_t)path_idx, true);

        // Store the band in the results
        resulting_path.push_back(RRTConfig(ending_grippers_pos, rubber_band, isBandFirstOrderVisibileToBlacklist(rubber_band)));

        // Record if the band is overstretched
        band_is_overstretched = rubber_band.isOverstretched();
        ++path_idx;
    }

    // Double check that if the band is not overstetched, then we've reached the endpoint of the path
    if (!band_is_overstretched)
    {
        const EigenHelpers::VectorVector3d ending_rubber_band_pos = rubber_band.getVectorRepresentation();
        const auto rubber_band_endpoints = std::make_pair(ending_rubber_band_pos.front(), ending_rubber_band_pos.back());
        assert(gripperPositionsAreApproximatelyEqual(path[end_index].getGrippers(), rubber_band_endpoints));
    }

    return std::make_pair(band_is_overstretched, resulting_path);
}

std::pair<std::vector<RRTConfig, RRTAllocator>, std::map<std::string, double>> RRTHelper::rrtShortcutSmooth(
        const std::vector<RRTConfig, RRTAllocator>& path) const
{
    std::map<std::string, double> statistics;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();

    std::vector<RRTConfig, RRTAllocator> current_path = path;
    uint32_t num_iterations = 0;
    uint32_t failed_iterations = 0;

    visualize(current_path);
    while (num_iterations < max_smoothing_iterations_ && failed_iterations < max_failed_smoothing_iterations_ && current_path.size() > 2)
    {
        ++num_iterations;

        vis_.deleteObjects("rubber_band_checking_rest_of_path");
        vis_.deleteObjects("rubber_band_post_rrt_step");

        // Attempt a shortcut
        const int64_t base_index = (int64_t)std::uniform_int_distribution<size_t>(0, current_path.size() - 1)(generator_);

        // Compute the offset index
        // We want to sample the start and goal slightly more frequently, so allow "overshoots" of endpoints for the offset
        const int64_t min_delta = std::max(-base_index - max_shortcut_index_distance_ / 10, -max_shortcut_index_distance_);
        const int64_t max_delta = std::min((int64_t)current_path.size() - base_index - 1 + max_shortcut_index_distance_ / 10, max_shortcut_index_distance_);
        const int64_t offset_delta = std::uniform_int_distribution<int64_t>(min_delta, max_delta)(generator_);
        // Clamp to the boundaries of the current path
        const int64_t second_index = arc_helpers::ClampValue(base_index + offset_delta, (int64_t)0, (int64_t)current_path.size() - 1);

        // Get start and end indices to establish the direction of the shortcut
        const size_t start_index = (size_t)std::min(base_index, second_index);
        const size_t end_index = (size_t)std::max(base_index, second_index);

        // We know start_index <= end_index, this essentially checks if start == end or start + 1 == end
        if (start_index + 1 >= end_index)
        {
            continue;
        }

        // Check if the edge is valid
        const RRTConfig& smoothing_start_config = current_path[start_index];
        const EigenHelpers::VectorVector3d& start_band = smoothing_start_config.getBand().getVectorRepresentation();
        const RRTConfig& smoothing_target_end_config = current_path[end_index];
        const EigenHelpers::VectorVector3d& end_band = smoothing_target_end_config.getBand().getVectorRepresentation();

        smoothing_start_config.getBand().visualize("rubber_band_smoothing_start", Visualizer::Red(), rubber_band_overstretched_color_, 1, true);
        vis_.visualizeCubes("rubber_band_smoothing_start", {start_band.front(), start_band.back()}, Eigen::Vector3d(0.02, 0.02, 0.02), Visualizer::Red(), 1000);
        smoothing_target_end_config.getBand().visualize("rubber_band_smoothing_end", Visualizer::Red(), rubber_band_overstretched_color_, 1, true);
        vis_.visualizeCubes("rubber_band_smoothing_end", {end_band.front(), end_band.back()}, Eigen::Vector3d(0.02, 0.02, 0.02), Visualizer::Red(), 1000);

        // Forward simulate the rubber band along the straight line between gripper positions
        const std::vector<std::pair<RRTConfig, int64_t>> smoothing_propogation_results = forwardPropogationFunction(smoothing_start_config, smoothing_target_end_config);

        // Check if the rubber band gets overstretched while propogating the grippers on the new path
        if (smoothing_propogation_results.size() == 0 ||
            !gripperPositionsAreApproximatelyEqual(smoothing_propogation_results.back().first.getGrippers(), smoothing_target_end_config.getGrippers()))
        {
            ++failed_iterations;
            continue;
        }

        // We still need to check that the rubber band can still reach the goal correctly from this state,
        // so we'll forward propogate along the rest of the trajectory to check feasibility
        const std::pair<bool, std::vector<RRTConfig, RRTAllocator>> end_of_smoothing_to_goal_results =
                forwardSimulateGrippersPath(
                    smoothing_propogation_results.back().first.getBand(),
                    current_path,
                    end_index,
                    current_path.size() - 1);

        // Check if the rubber band gets overstretched or ends up in a blacklisted first order homotopy class
        // while following the tail of the starting trajectory
        const bool final_band_at_goal_overstretched = end_of_smoothing_to_goal_results.first;
        if (final_band_at_goal_overstretched ||
            isBandFirstOrderVisibileToBlacklist(end_of_smoothing_to_goal_results.second.back().getBand()))
        {
            ++failed_iterations;
            continue;
        }

        ////////// Smoothing success - Create the new smoothed path ////////////////////////////////////////////////////

        // Allocate space for the total smoothed path
        std::vector<RRTConfig, RRTAllocator> smoothed_path;
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

void RRTHelper::visualize(const std::vector<RRTConfig, RRTAllocator>& path) const
{
    vis_.deleteObjects("rubber_band_rrt_solution");
    vis_.deleteObjects("gripper_a_rrt_solution");
    vis_.deleteObjects("gripper_b_rrt_solution");

    for (size_t ind = 0; ind < path.size(); ++ind)
    {
        const RRTConfig& config = path[ind];
        const RRTGrippersRepresentation& gripper_positions = config.getGrippers();
        const VirtualRubberBand& rubber_band = config.getBand();

        rubber_band.visualize("rubber_band_rrt_solution", Visualizer::Yellow(), rubber_band_overstretched_color_, (int32_t)ind + 1, true);
        vis_.visualizeCubes("gripper_a_rrt_solution", {gripper_positions.first}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Red(), (int32_t)ind + 1);
        vis_.visualizeCubes("gripper_b_rrt_solution", {gripper_positions.second}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Blue(), (int32_t)ind + 1);
        std::this_thread::sleep_for(std::chrono::duration<double>(1e-5));
    }
}

void RRTHelper::visualizeBlacklist() const
{
    for (size_t idx = 0; idx < blacklisted_goal_rubber_bands_.size(); ++idx)
    {
        vis_.visualizeLineStrip("blacklisted_rubber_bands", blacklisted_goal_rubber_bands_[idx], Visualizer::Yellow(), (int32_t)idx + 1);
    }
}
