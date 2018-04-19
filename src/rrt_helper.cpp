#include "smmap/rrt_helper.h"

#include <thread>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/simple_dtw.hpp>
#include <arc_utilities/timing.hpp>

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;

void print(const RRTRobotRepresentation& config)
{
    std::cout << config.first.transpose() << "        " << config.second.transpose() << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper function for assertion testing
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

static bool robotConfigurationsAreApproximatelyEqual(
        const RRTRobotRepresentation& r1,
        const RRTRobotRepresentation& r2)
{
    bool is_equal = true;
    is_equal &= r1.first.isApprox(r2.first);
    is_equal &= r1.second.isApprox(r2.second);
    return is_equal;
}

static bool bandEndpointsMatchGripperPositions(
        const RubberBand& band,
        const RRTGrippersRepresentation& grippers)
{
    return gripperPositionsAreApproximatelyEqual(grippers, band.getEndpoints());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTConfig functions                      /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RRTConfig::RRTConfig(
        const RRTGrippersRepresentation& grippers_position,
        const RRTRobotRepresentation& robot_configuration,
        const size_t unique_forward_propogation_idx,
        const RubberBand& band,
        const bool is_visible_to_blacklist)
    : grippers_position_(grippers_position)
    , robot_configuration_(robot_configuration)
    , unique_forward_propogation_idx_(unique_forward_propogation_idx)
    , band_(band)
    , is_visible_to_blacklist_(is_visible_to_blacklist)
{}

const RRTGrippersRepresentation& RRTConfig::getGrippers() const
{
    return grippers_position_;
}

const RRTRobotRepresentation& RRTConfig::getRobotConfiguration() const
{
    return robot_configuration_;
}

size_t RRTConfig::getUniqueForwardPropogationIndex() const
{
    return unique_forward_propogation_idx_;
}

const RubberBand& RRTConfig::getBand() const
{
    return band_;
}

bool RRTConfig::isVisibleToBlacklist() const
{
    return is_visible_to_blacklist_;
}

// Returned distance is the Euclidian distance of two grippers posistions
double RRTConfig::distance(const RRTConfig& other) const
{
    return RRTConfig::distance(*this, other);
}

double RRTConfig::distance(const RRTConfig& c1, const RRTConfig& c2)
{
#ifdef DTW_DISTANCE
    const auto distance_fn = [] (const Eigen::Vector3d p1, const Eigen::Vector3d p2)
    {
        return (p1 - p2).norm();
    };
    return simple_dtw::ComputeDTWDistance(c1.getBand().getVectorRepresentation(), c2.getBand().getVectorRepresentation(), distance_fn);
#endif
    return RRTConfig::distance(c1.getGrippers(), c2.getGrippers());
}

double RRTConfig::distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2)
{
    const Eigen::Vector3d& c1_first_gripper     = c1.first;
    const Eigen::Vector3d& c1_second_gripper    = c1.second;
    const Eigen::Vector3d& c2_first_gripper     = c2.first;
    const Eigen::Vector3d& c2_second_gripper    = c2.second;
    return std::sqrt((c1_first_gripper - c2_first_gripper).squaredNorm() +
                     (c1_second_gripper - c2_second_gripper).squaredNorm());
}

double RRTConfig::distance(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2)
{
    const Eigen::VectorXd& r1_first_arm     = r1.first;
    const Eigen::VectorXd& r1_second_arm    = r1.second;
    const Eigen::VectorXd& r2_first_arm     = r2.first;
    const Eigen::VectorXd& r2_second_arm    = r2.second;
    return std::sqrt((r1_first_arm - r2_first_arm).squaredNorm() +
                     (r1_second_arm - r2_second_arm).squaredNorm());
}

// Only calculates the distance travelled by the grippers, not the entire band
double RRTConfig::grippersPathDistance(const std::vector<RRTConfig, RRTAllocator>& path, const size_t start_index, const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());
    double path_distance = 0;
    for (size_t idx = start_index; idx < end_index; ++idx)
    {
        path_distance += RRTConfig::distance(path[idx].getGrippers(), path[idx + 1].getGrippers());
    }
    return path_distance;
}

double RRTConfig::robotPathDistance(const std::vector<RRTConfig, RRTAllocator>& path, const size_t start_index, const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());
    double path_distance = 0;
    for (size_t idx = start_index; idx < end_index; ++idx)
    {
        path_distance += RRTConfig::distance(path[idx].getRobotConfiguration(), path[idx + 1].getRobotConfiguration());
    }
    return path_distance;
}

std::string RRTConfig::print() const
{
    std::stringstream out;
    out << PrettyPrint::PrettyPrint(grippers_position_, true, " ")
        << "    " << robot_configuration_.first.transpose() << "  " << robot_configuration_.second.transpose();
    return out.str();
}

bool RRTConfig::operator==(const RRTConfig& other) const
{
    if (!gripperPositionsAreApproximatelyEqual(grippers_position_, other.grippers_position_))
    {
        return false;
    }

    if (!robotConfigurationsAreApproximatelyEqual(robot_configuration_, other.robot_configuration_))
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

    if (unique_forward_propogation_idx_ != other.unique_forward_propogation_idx_)
    {
        return false;
    }

    return true;
}

std::size_t std::hash<smmap::RRTConfig>::operator()(const smmap::RRTConfig& rrt_config) const
{
    std::size_t seed = 0;
    std::hash_combine(seed, rrt_config.getGrippers());
    std::hash_combine(seed, rrt_config.getRobotConfiguration());
    std::hash_combine(seed, rrt_config.getUniqueForwardPropogationIndex());

    const EigenHelpers::VectorVector3d& band_ = rrt_config.getBand().getVectorRepresentation();
    for (size_t idx = 0; idx < band_.size(); ++idx)
    {
        std::hash_combine(seed, band_[idx]);
    }
    return seed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTHelper functions                      /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RRTHelper::RRTHelper(
        const RobotInterface::Ptr robot,
        const sdf_tools::SignedDistanceField& environment_sdf,
        const Visualizer::Ptr vis,
        std::mt19937_64& generator,
        const std::shared_ptr<PRMHelper>& prm_helper,
        const Eigen::Isometry3d& task_aligned_frame,
        const Eigen::Vector3d& task_aligned_lower_limits,
        const Eigen::Vector3d& task_aligned_upper_limits,
        const double max_gripper_step_size,
        const double max_robot_dof_step_size,
        const double min_robot_dof_step_size,
        const double max_gripper_rotation,
        const double goal_bias,
        const double goal_reach_radius,
        const double gripper_min_distance_to_obstacles,
        const double homotopy_distance_penalty,
        const int64_t max_shortcut_index_distance,
        const uint32_t max_smoothing_iterations,
        const uint32_t max_failed_smoothing_iterations,
        const bool visualization_enabled)
    : robot_(robot)
    , environment_sdf_(environment_sdf)

    , vis_(vis)
    , visualization_enabled_globally_(visualization_enabled)
    , band_safe_color_(Visualizer::Black())
    , band_overstretched_color_(Visualizer::Cyan())
    , gripper_a_tree_color_(Visualizer::Magenta())
    , gripper_b_tree_color_(Visualizer::Orange())

    , task_aligned_frame_(task_aligned_frame)
    , task_aligned_lower_limits_(task_aligned_lower_limits)
    , task_aligned_upper_limits_(task_aligned_upper_limits)
    , max_gripper_step_size_(max_gripper_step_size)
    , max_robot_dof_step_size_(max_robot_dof_step_size)
    , min_robot_dof_step_size_(min_robot_dof_step_size)
    , max_gripper_rotation_(max_gripper_rotation)
    , goal_bias_(goal_bias)
    , goal_reach_radius_(goal_reach_radius)
    , homotopy_distance_penalty_(homotopy_distance_penalty)
    , gripper_min_distance_to_obstacles_(gripper_min_distance_to_obstacles)

    , max_shortcut_index_distance_(max_shortcut_index_distance)
    , max_smoothing_iterations_(max_smoothing_iterations)
    , max_failed_smoothing_iterations_(max_failed_smoothing_iterations)

    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)
    , uniform_shortcut_smoothing_int_distribution_(1, 4)
    , prm_helper_(prm_helper)

    , total_sampling_time_(NAN)
    , total_nearest_neighbour_time_(NAN)
    , total_everything_included_forward_propogation_time_(NAN)
    , total_band_forward_propogation_time_(NAN)
    , total_first_order_vis_propogation_time_(NAN)
{
    assert(task_aligned_lower_limits_.x() <= task_aligned_upper_limits_.x());
    assert(task_aligned_lower_limits_.y() <= task_aligned_upper_limits_.y());
    assert(task_aligned_lower_limits_.z() <= task_aligned_upper_limits_.z());
    assert(max_gripper_step_size_ > 0.0);
    assert(goal_reach_radius_ > 0.0);
    assert(homotopy_distance_penalty_ >= 0.0);
    assert(max_shortcut_index_distance_ > 0);
    assert(gripper_min_distance_to_obstacles_ > 0.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for extral RRT planning class
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int64_t RRTHelper::nearestNeighbour(
        const std::vector<ExternalRRTState>& nodes,
        const RRTConfig& config)
{
    Stopwatch stopwatch;

    arc_helpers::DoNotOptimize(nodes);
    int64_t nn_idx = nearestNeighbour_internal(nodes, config);
    arc_helpers::DoNotOptimize(nn_idx);

    const double nn_time = stopwatch(READ);
    total_nearest_neighbour_time_ += nn_time;

    return nn_idx;
}

int64_t RRTHelper::nearestNeighbour_internal(
        const std::vector<ExternalRRTState>& nodes,
        const RRTConfig& config)
{
    assert(nodes.size() >= 1);

    // If we did not sample the goal, then just use the normal distance metric
    // We cannot use "auto" here because of the ternary later that choses between these 2 function types
    const std::function<double(const ExternalRRTState&, const RRTConfig&)> basic_distance_fn = [&] (
            const ExternalRRTState& rrt_state, const RRTConfig& rrt_config)
    {
        return RRTConfig::distance(rrt_state.GetValueImmutable(), rrt_config);
    };
    // If we sampled the goal, then distance is also a function of "homotopy class"
    // We cannot use "auto" here because of the ternary later that choses between these 2 function types
    const std::function<double(const ExternalRRTState&, const RRTConfig&)> goal_sampled_distance_fn = [&] (
            const ExternalRRTState& rrt_state, const RRTConfig& rrt_config)
    {
        const double basic_distance = basic_distance_fn(rrt_state.GetValueImmutable(), rrt_config);

        const auto blacklist_itr = goal_expansion_nn_blacklist_.find(rrt_state.GetValueImmutable());
        const bool state_blacklisted = (blacklist_itr != goal_expansion_nn_blacklist_.end());
        if (state_blacklisted)
        {
            return NN_BLACKLIST_DISTANCE + basic_distance;
        }
        else
        {
            return basic_distance;
        }
    };

    // Determine which distance function to use
    const bool goal_is_target_config = gripperPositionsAreApproximatelyEqual(grippers_goal_position_, config.getGrippers());
    const auto& distance_fn = goal_is_target_config ? goal_sampled_distance_fn : basic_distance_fn;

    const size_t K = 1;
    const auto nn_results = arc_helpers::GetKNearestNeighbors(nodes, config, distance_fn, K);
    assert(nn_results.size() == K);
    const int64_t nn_idx = nn_results[0].first;

    // Blacklist this config from being selected again as the nearest neighbour to the goal
    if (goal_is_target_config)
    {
        goal_expansion_nn_blacklist_.insert(nodes[nn_idx].GetValueImmutable());
    }

    return nn_idx;
}

RRTConfig RRTHelper::configSampling()
{
    Stopwatch stopwatch;

    arc_helpers::DoNotOptimize(grippers_goal_position_);
#ifdef PRM_SAMPLING
    const RRTConfig sample = prmBasedSampling_internal();
#else
    const RRTConfig sample(posPairSampling_internal(), starting_robot_configuration_, next_unique_forward_propogation_idx_, *starting_band_, false);
#endif
    arc_helpers::DoNotOptimize(sample.getGrippers());

    const double sampling_time = stopwatch(READ);
    total_sampling_time_ += sampling_time;
    next_unique_forward_propogation_idx_++;

    return sample;
}

RRTConfig RRTHelper::prmBasedSampling_internal()
{
    const RRTGrippersRepresentation rand_grippers_sample = posPairSampling_internal();
    const bool goal_is_target_config = gripperPositionsAreApproximatelyEqual(grippers_goal_position_, rand_grippers_sample);

    // If we've sampled the goal, then keep sampling until the result is not visible to the blacklist
    EigenHelpers::VectorVector3d band_path;
    do
    {
        band_path = prm_helper_->getRandomPath(rand_grippers_sample.first, rand_grippers_sample.second);
    }
    while (goal_is_target_config && isBandFirstOrderVisibileToBlacklist(band_path));

    RubberBand band(*starting_band_);
    band.setPointsWithoutSmoothing(band_path);
    band.visualize(PRMHelper::PRM_RANDOM_PATH_NS, Visualizer::Orange(), Visualizer::Orange(), 1, visualization_enabled_globally_);

    return RRTConfig(rand_grippers_sample, starting_robot_configuration_, next_unique_forward_propogation_idx_, band, false);
}

RRTGrippersRepresentation RRTHelper::posPairSampling_internal()
{
    RRTGrippersRepresentation rand_sample;
    const bool sample_goal = uniform_unit_distribution_(generator_) < goal_bias_;

    if (sample_goal)
    {
        rand_sample = grippers_goal_position_;
    }
    else
    {
        do
        {
            const double x1 = EigenHelpers::Interpolate(task_aligned_lower_limits_.x(), task_aligned_upper_limits_.x(), uniform_unit_distribution_(generator_));
            const double y1 = EigenHelpers::Interpolate(task_aligned_lower_limits_.y(), task_aligned_upper_limits_.y(), uniform_unit_distribution_(generator_));
            const double z1 = EigenHelpers::Interpolate(task_aligned_lower_limits_.z(), task_aligned_upper_limits_.z(), uniform_unit_distribution_(generator_));
            rand_sample.first = Eigen::Vector3d(x1, y1, z1);
        }
        while (environment_sdf_.EstimateDistance3d(task_aligned_frame_ * rand_sample.first).first < gripper_min_distance_to_obstacles_);

        // We want to only sample within a radius max_grippers_distance_, and within the world extents; to do so
        // uniformly, we sample from an axis aligned box limited by R and the world extents, rejecting samples that lie
        // outside a radius max_grippers_distance_
        const double x2_min = std::max(task_aligned_lower_limits_.x(), rand_sample.first.x() - max_grippers_distance_);
        const double x2_max = std::min(task_aligned_upper_limits_.x(), rand_sample.first.x() + max_grippers_distance_);
        const double y2_min = std::max(task_aligned_lower_limits_.y(), rand_sample.first.y() - max_grippers_distance_);
        const double y2_max = std::min(task_aligned_upper_limits_.y(), rand_sample.first.y() + max_grippers_distance_);
        const double z2_min = std::max(task_aligned_lower_limits_.z(), rand_sample.first.z() - max_grippers_distance_);
        const double z2_max = std::min(task_aligned_upper_limits_.z(), rand_sample.first.z() + max_grippers_distance_);

        bool valid = false;
        do
        {
            const double x2 = EigenHelpers::Interpolate(x2_min, x2_max, uniform_unit_distribution_(generator_));
            const double y2 = EigenHelpers::Interpolate(y2_min, y2_max, uniform_unit_distribution_(generator_));
            const double z2 = EigenHelpers::Interpolate(z2_min, z2_max, uniform_unit_distribution_(generator_));
            rand_sample.second = Eigen::Vector3d(x2, y2, z2);
            valid = (rand_sample.first - rand_sample.second).norm() <= max_grippers_distance_;
        }
        while (!valid || environment_sdf_.EstimateDistance3d(task_aligned_frame_ * rand_sample.second).first < gripper_min_distance_to_obstacles_);
    }

    rand_sample.first = task_aligned_frame_ * rand_sample.first;
    rand_sample.second = task_aligned_frame_ * rand_sample.second;

    return rand_sample;
}

bool RRTHelper::goalReached(const RRTConfig& node)
{
    if (RRTConfig::distance(node.getGrippers(), grippers_goal_position_) < goal_reach_radius_)
    {
        if (visualization_enabled_globally_)
        {
            vis_->visualizeLineStrip(RRT_GOAL_TESTING_NS, node.getBand().getVectorRepresentation(), Visualizer::White(), 1, 0.01);
            ros::spinOnce();
            std::this_thread::sleep_for(std::chrono::duration<double>(0.01));
        }

        // Only accept paths that are different from those on the blacklist
        if (!isBandFirstOrderVisibileToBlacklist(node.getBand()))
        {
            return true;
        }
    }
    return false;
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
        const RRTConfig& random_target,
        const bool calculate_first_order_vis,
        const bool visualization_enabled_locally)
{
    // Note: This is used only for visualization purposes
    const bool target_is_goal_config = (RRTConfig::distance(random_target.getGrippers(), grippers_goal_position_) < goal_reach_radius_);
    if (visualization_enabled_globally_ && visualization_enabled_locally)// && target_is_goal_config && false)
    {
        vis_->visualizeCubes(
                    RRT_FORWARD_PROP_START_NS,
                    {nearest_neighbor.getGrippers().first},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    gripper_a_tree_color_,
                    1);
        vis_->visualizeCubes(
                    RRT_FORWARD_PROP_START_NS,
                    {nearest_neighbor.getGrippers().second},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    gripper_b_tree_color_,
                    5);

        nearest_neighbor.getBand().visualize(
                    RRT_FORWARD_PROP_START_NS,
                    Visualizer::Green(),
                    Visualizer::Green(),
                    10,
                    true);

        vis_->visualizeCubes(
                    RRT_SAMPLE_NS,
                    {random_target.getGrippers().first},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    gripper_a_tree_color_,
                    1);
        vis_->visualizeCubes(
                    RRT_SAMPLE_NS,
                    {random_target.getGrippers().second},
                    Eigen::Vector3d(0.01, 0.01, 0.01),
                    gripper_b_tree_color_,
                    5);

        vis_->deleteObjects(RRT_FORWARD_PROP_STEPS_NS, 1, 2);
    }

//    std::cout << "Nearest Neighbour Config:\n" << nearest_neighbor.print() << std::endl;
//    std::cout << "Random Target Config:\n" << random_target.print() << std::endl;

    Stopwatch function_wide_stopwatch;
    Stopwatch stopwatch;

    static EigenHelpers::VectorVector3d band_visualization_line_start_points;
    static EigenHelpers::VectorVector3d band_visualization_line_end_points;
    static EigenHelpers::VectorVector3d gripper_a_tree_start_points;
    static EigenHelpers::VectorVector3d gripper_a_tree_end_points;
    static EigenHelpers::VectorVector3d gripper_b_tree_start_points;
    static EigenHelpers::VectorVector3d gripper_b_tree_end_points;
    static int32_t marker_id = 1;
    static size_t num_bands_in_visualize_list = 0;

    const RRTGrippersRepresentation& starting_grippers_position = nearest_neighbor.getGrippers();
    const RRTRobotRepresentation& starting_robot_configuration = nearest_neighbor.getRobotConfiguration();

    // Extract the target gripper pose and corresponding robot configuration
    const RRTGrippersRepresentation& target_grippers_position = random_target.getGrippers();
    RRTRobotRepresentation target_robot_configuration = starting_robot_configuration;
    if (planning_for_whole_robot_)
    {
        AllGrippersSinglePose target_grippers_poses = starting_grippers_poses_;
        target_grippers_poses[0].translation() = target_grippers_position.first;
        target_grippers_poses[1].translation() = target_grippers_position.second;

        const auto ik_solutions = robot_->getIkSolutions(target_grippers_poses);
        assert(ik_solutions.size() == 2);

        const size_t num_solutions_gripper_a = ik_solutions[0].size();
        const size_t num_solutions_gripper_b = ik_solutions[1].size();

        if (num_solutions_gripper_a == 0 || num_solutions_gripper_b == 0)
        {
            std::cout << "Fwd prop never started as there are no IK solutions: Gripper A: " << num_solutions_gripper_a << " Gripper B: " << num_solutions_gripper_b << std::endl;
            return std::vector<std::pair<RRTConfig, int64_t>>();
        }

        const std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> individual_arm_config_distance_fn =
                [] (const Eigen::VectorXd& c1, const Eigen::VectorXd& c2)
        {
            return (c1 - c2).norm();
        };

//        std::cout << "Getting nearest IK solution" << std::endl;

        const size_t K = 1;
        const auto gripper_a_nn_results = arc_helpers::GetKNearestNeighbors(ik_solutions[0], nearest_neighbor.getRobotConfiguration().first, individual_arm_config_distance_fn, K);
        assert(gripper_a_nn_results.size() == K);
        const int64_t gripper_a_nn_idx = gripper_a_nn_results[0].first;
        const auto gripper_b_nn_results = arc_helpers::GetKNearestNeighbors(ik_solutions[1], nearest_neighbor.getRobotConfiguration().second, individual_arm_config_distance_fn, K);
        assert(gripper_b_nn_results.size() == K);
        const int64_t gripper_b_nn_idx = gripper_b_nn_results[0].first;

        target_robot_configuration.first = ik_solutions[0][gripper_a_nn_idx];
        target_robot_configuration.second = ik_solutions[1][gripper_b_nn_idx];

//        std::cout << "Starting robot config: "; print(starting_robot_configuration);
//        std::cout << "Nearest solution:      "; print(target_robot_configuration);
    }

    // Continue advancing the grippers until the grippers collide or the band overstretches
    const bool rubber_band_verbose = visualization_enabled_globally_ && visualization_enabled_locally;
    int64_t parent_offset = -1;
    uint32_t step_index = 0;
    std::vector<std::pair<RRTConfig, int64_t>> propagated_states;

    if (planning_for_whole_robot_)
    {
        // Allocate space for potential children
        const double total_distance = RRTConfig::distance(nearest_neighbor.getRobotConfiguration(), target_robot_configuration);
        const uint32_t max_total_steps = (uint32_t)ceil(total_distance / max_robot_dof_step_size_);
        propagated_states.reserve(max_total_steps);

        while (step_index < max_total_steps)
        {
            stopwatch(RESET);
            // Using ternary operator here so that we can avoid making copies, and still take advantage of const correctness
            const bool use_nearest_neighbour_as_prev = (parent_offset == -1);
            const RRTConfig prev_node = (use_nearest_neighbour_as_prev ? nearest_neighbor : propagated_states[parent_offset].first);
            const RubberBand& prev_band = prev_node.getBand();
            const auto& prev_robot_config = prev_node.getRobotConfiguration();


            // Interpolate in joint space to find the translation of the grippers
//            std::cout << "Interpolating robot config" << std::endl;
            const double ratio = std::min(1.0, (double)(step_index + 1) * max_gripper_step_size_ / total_distance);
            const Eigen::VectorXd arm_a_interpolated = EigenHelpers::Interpolate(starting_robot_configuration.first, target_robot_configuration.first, ratio);
            const Eigen::VectorXd arm_b_interpolated = EigenHelpers::Interpolate(starting_robot_configuration.second, target_robot_configuration.second, ratio);
            const RRTRobotRepresentation next_robot_configuration(arm_a_interpolated, arm_b_interpolated);

//            std::cout << "Converting to gripper poses" << std::endl;
            const AllGrippersSinglePose next_grippers_poses = robot_->getGrippersPoses(next_robot_configuration);
            const RRTGrippersRepresentation next_grippers_position(next_grippers_poses[0].translation(), next_grippers_poses[1].translation());

            // Old Jacobian based movement
            if (false)
            {
////            RRTRobotRepresentation next_robot_configuration = prev_robot_config;
////            RRTGrippersRepresentation next_grippers_position = starting_grippers_position;

////            const auto& prev_grippers_position = prev_node.getGrippers();
////            const auto gripper_a_desired_motion = target_grippers_position.first - prev_grippers_position.first;
////            const auto gripper_b_desired_motion = target_grippers_position.second - prev_grippers_position.second;

////            AllGrippersSinglePoseDelta grippers_desired_motion(2, kinematics::Vector6d::Zero());
////            grippers_desired_motion[0].head<3>() = gripper_a_desired_motion;
////            grippers_desired_motion[1].head<3>() = gripper_b_desired_motion;

//            // Get the matching robot motion, limiting the size of the step based on
//            auto robot_motion = robot_->mapGripperMotionToRobotMotion(prev_robot_config, grippers_desired_motion);
//            if (robot_motion.first.norm() > max_robot_dof_step_size_)
//            {
//                robot_motion.first = robot_motion.first.normalized() * max_robot_dof_step_size_;
//            }
//            if (robot_motion.second.norm() > max_robot_dof_step_size_)
//            {
//                robot_motion.second = robot_motion.second.normalized() * max_robot_dof_step_size_;
//            }

//            // Check if the robot motion is small enough to stop propogation
//            if (robot_motion.first.norm() < min_robot_dof_step_size_ && robot_motion.second.norm() < min_robot_dof_step_size_)
//            {
//                std::cerr << "Fwd prop stopped due to No meaningful motion towards target\n"
//                          << robot_motion.first.transpose() << "    " << robot_motion.second.transpose() << std::endl;
//                break;
//            }

//            next_robot_configuration.first += robot_motion.first;
//            next_robot_configuration.second += robot_motion.second;

//            // Check if we violated joint limits
//            const Eigen::VectorXd lower_limits_first  = robot_->joint_lower_limits_.head(robot_motion.first.size());
//            const Eigen::VectorXd lower_limits_second = robot_->joint_lower_limits_.tail(robot_motion.second.size());
//            const Eigen::VectorXd upper_limits_first  = robot_->joint_upper_limits_.head(robot_motion.first.size());
//            const Eigen::VectorXd upper_limits_second = robot_->joint_upper_limits_.tail(robot_motion.second.size());
//            if (((next_robot_configuration.first  - lower_limits_first).array()  <= 0.0).any() ||
//                ((next_robot_configuration.second - lower_limits_second).array() <= 0.0).any() ||
//                ((upper_limits_first  - next_robot_configuration.first).array()  <= 0.0).any() ||
//                ((upper_limits_second - next_robot_configuration.second).array() <= 0.0).any())
//            {
//                std::cerr << "Fwd prop stopped due to joint limits:\n"
//                          << "Robot config:  " << next_robot_configuration.first.transpose() << "    " << next_robot_configuration.second.transpose() << std::endl;
//                break;
//            }
            }


//            std::cout << "Checking robot collision" << std::endl;

            // Check if we entered collision
            if (robot_->checkRobotCollision(next_robot_configuration))
            {
                if (visualization_enabled_locally)
                {
                    std::cerr << "Fwd prop stopped due to robot collision:\n"
                              << "Robot config:  " << next_robot_configuration.first.transpose() << "    " << next_robot_configuration.second.transpose() << std::endl;
                }
                break;
            }

//            std::cout << "Checking gripper overrotation" << std::endl;

            // Check if we rotated the grippers too much
            {
                const double gripper_a_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_[0].rotation(), next_grippers_poses[0].rotation());
                const double gripper_b_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_[1].rotation(), next_grippers_poses[1].rotation());
                if (gripper_a_rotation_dist > max_gripper_rotation_ || gripper_b_rotation_dist > max_gripper_rotation_)
                {
                    std::cerr << "Fwd prop stopped due to gripper overrotation:\n"
                              << "Gripper rotation dists: " << gripper_a_rotation_dist << " " << gripper_b_rotation_dist << std::endl;
                    break;
                }
            }

//            std::cout << "Checking grippers moving outside of planning arena" << std::endl;

            // If the grippers move outside of the planning arena (possible due to non-linearities), then stop
            {
                auto task_frame_next_grippers_position = next_grippers_position;
                task_frame_next_grippers_position.first = task_aligned_frame_.inverse() * task_frame_next_grippers_position.first;
                task_frame_next_grippers_position.second = task_aligned_frame_.inverse() * task_frame_next_grippers_position.second;

                if ((task_frame_next_grippers_position.first.array() > task_aligned_upper_limits_.array()).any() ||
                    (task_frame_next_grippers_position.first.array() < task_aligned_lower_limits_.array()).any() ||
                    (task_frame_next_grippers_position.second.array() > task_aligned_upper_limits_.array()).any() ||
                    (task_frame_next_grippers_position.second.array() < task_aligned_lower_limits_.array()).any())
                {
                    std::cerr << "Fwd prop stopped due to grippers moving outside the planning arena:\n";
                    std::cout << "Task aligned Next grippers positions: " << PrettyPrint::PrettyPrint(task_frame_next_grippers_position, true, " ") << std::endl;
                    std::cout << "Next grippers positions:              " << PrettyPrint::PrettyPrint(next_grippers_position, true, " ") << std::endl;
                    break;
                }

                assert((task_frame_next_grippers_position.first.array() <= task_aligned_upper_limits_.array()).all());
                assert((task_frame_next_grippers_position.first.array() >= task_aligned_lower_limits_.array()).all());
                assert((task_frame_next_grippers_position.second.array() <= task_aligned_upper_limits_.array()).all());
                assert((task_frame_next_grippers_position.second.array() >= task_aligned_lower_limits_.array()).all());
            }

//            std::cout << "Checking band propogation" << std::endl;


            // Duplicated from below


            // Forward simulate the rubber band to test this transition
            RubberBand next_band(prev_band);
            next_band.forwardPropagateRubberBandToEndpointTargets(
                        next_grippers_position.first,
                        next_grippers_position.second,
                        rubber_band_verbose);
            const double band_forward_propogation_time = stopwatch(READ);
            total_band_forward_propogation_time_ += band_forward_propogation_time;

            // If we are still able to get to the next target position after retrying,
            // then return however far we were able to get
            if (!bandEndpointsMatchGripperPositions(next_band, next_grippers_position))
            {
                if (visualization_enabled_locally)
                {
                    std::cerr << "Fwd prop stopped due to rubber band endpoints not matching:\n"
                              << "Grippers Pos:       " << PrettyPrint::PrettyPrint(next_grippers_position) << std::endl
                              << "Band Endpoints Pos: " << PrettyPrint::PrettyPrint(next_band.getEndpoints()) << std::endl;
                }
                break;
            }

            // If the rubber band becomes overstretched, then return however far we were able to get
            if (next_band.isOverstretched())
            {
                if (visualization_enabled_globally_ && visualization_enabled_locally)
                {
//                    next_band.visualize(RRT_FORWARD_PROP_STEPS_NS, Visualizer::Blue(), Visualizer::Cyan(), 2, true);
                    std::cerr<< "Fwd prop stopped due to band overstretch: " << EigenHelpers::CalculateTotalDistance(next_band.getVectorRepresentation()) << std::endl;
                }
                break;
            }

            if (visualization_enabled_globally_ && visualization_enabled_locally)
            {
                ++num_bands_in_visualize_list;
                const EigenHelpers::VectorVector3d band = next_band.getVectorRepresentation();
                for (size_t band_idx = 0; band_idx + 1 < band.size(); ++band_idx)
                {
                    band_visualization_line_start_points.push_back(band[band_idx]);
                    band_visualization_line_end_points.push_back(band[band_idx + 1]);
                }

                if (num_bands_in_visualize_list >= 20 || target_is_goal_config)
                {
                    vis_->visualizeLines(RRT_FORWARD_PROP_STEPS_NS, band_visualization_line_start_points, band_visualization_line_end_points, Visualizer::Blue(), 1);
                    band_visualization_line_start_points.clear();
                    band_visualization_line_end_points.clear();
                    num_bands_in_visualize_list = 0;
                }
            }

            bool is_first_order_visible = false;
            if (calculate_first_order_vis)
            {
                is_first_order_visible = isBandFirstOrderVisibileToBlacklist(next_band);
            }

            // The new configuation is valid, add it to the tree
            const RRTConfig next_node(
                        next_grippers_position,
                        next_robot_configuration,
                        nearest_neighbor.getUniqueForwardPropogationIndex(),
                        next_band,
                        is_first_order_visible);
            propagated_states.push_back(std::pair<RRTConfig, int64_t>(next_node, parent_offset));

            if (visualization_enabled_globally_ && visualization_enabled_locally)
            {
                const RRTGrippersRepresentation& prev_grippers_position = prev_node.getGrippers();

                gripper_a_tree_start_points.push_back(prev_grippers_position.first);
                gripper_b_tree_start_points.push_back(prev_grippers_position.second);
                gripper_a_tree_end_points.push_back(next_grippers_position.first);
                gripper_b_tree_end_points.push_back(next_grippers_position.second);

                if (gripper_a_tree_start_points.size() >= 100 || target_is_goal_config)
                {
                    vis_->visualizeLines(RRT_TREE_GRIPPER_A_NS, gripper_a_tree_start_points, gripper_a_tree_end_points, gripper_a_tree_color_, marker_id);
                    vis_->visualizeLines(RRT_TREE_GRIPPER_B_NS, gripper_b_tree_start_points, gripper_b_tree_end_points, gripper_b_tree_color_, marker_id);
                    gripper_a_tree_start_points.clear();
                    gripper_b_tree_start_points.clear();
                    gripper_a_tree_end_points.clear();
                    gripper_b_tree_end_points.clear();
                    ++marker_id;
                }
            }

            ++parent_offset;
            ++step_index;
        }
    }
    else
    {
        // Allocate space for potential children
        const double total_distance = RRTConfig::distance(nearest_neighbor.getGrippers(), random_target.getGrippers());
        const uint32_t max_total_steps = (uint32_t)ceil(total_distance / max_gripper_step_size_);
        propagated_states.reserve(max_total_steps);

        while (step_index < max_total_steps)
        {
            stopwatch(RESET);
            // Using ternary operator here so that we can avoid making copies, and still take advantage of const correctness
            const bool use_nearest_neighbour_as_prev = (parent_offset == -1);
            const RRTConfig& prev_node = (use_nearest_neighbour_as_prev ? nearest_neighbor : propagated_states[parent_offset].first);
            const RubberBand& prev_band = prev_node.getBand();
            const auto& prev_robot_config = prev_node.getRobotConfiguration();


            const double ratio = std::min(1.0, (double)(step_index + 1) * max_gripper_step_size_ / total_distance);
            const Eigen::Vector3d gripper_a_interpolated = EigenHelpers::Interpolate(starting_grippers_position.first, target_grippers_position.first, ratio);
            const Eigen::Vector3d gripper_b_interpolated = EigenHelpers::Interpolate(starting_grippers_position.second, target_grippers_position.second, ratio);
            const RRTGrippersRepresentation next_grippers_position = RRTGrippersRepresentation(gripper_a_interpolated, gripper_b_interpolated);
            const RRTRobotRepresentation next_robot_configuration = prev_robot_config;

            // If the grippers collide with each other, then return however far we are able to get
            if ((gripper_a_interpolated - gripper_b_interpolated).norm() < gripper_min_distance_to_obstacles_)
            {
                if (visualization_enabled_locally)
                {
                    std::cerr << "Fwd prop stopped due to gripper collision with other gripper:\n"
                              << "First Pos:  " << PrettyPrint::PrettyPrint(gripper_a_interpolated)
                              << "    Second Pos: " << PrettyPrint::PrettyPrint(gripper_b_interpolated)
                              << "    Dist: " << (gripper_a_interpolated - gripper_b_interpolated).norm() << std::endl;
                }
                break;
            }

            // If the grippers enter collision, then return however far we were able to get
            if ((environment_sdf_.EstimateDistance3d(gripper_a_interpolated).first < gripper_min_distance_to_obstacles_) ||
                (environment_sdf_.EstimateDistance3d(gripper_b_interpolated).first < gripper_min_distance_to_obstacles_) ||
                (environment_sdf_.DistanceToBoundary3d(gripper_a_interpolated).first < gripper_min_distance_to_obstacles_) ||
                (environment_sdf_.DistanceToBoundary3d(gripper_b_interpolated).first < gripper_min_distance_to_obstacles_))
            {
                if (visualization_enabled_locally)
                {
                    std::cerr << "Fwd prop stopped due to gripper collision with environment:\n"
                              << "First Pos:  " << PrettyPrint::PrettyPrint(gripper_a_interpolated) << " Dist: " << environment_sdf_.EstimateDistance3d(gripper_a_interpolated).first << std::endl
                              << "Second Pos: " << PrettyPrint::PrettyPrint(gripper_b_interpolated) << " Dist: " << environment_sdf_.EstimateDistance3d(gripper_b_interpolated).first << std::endl;
                }
                break;
            }

            // Forward simulate the rubber band to test this transition
            RubberBand next_band(prev_band);
            next_band.forwardPropagateRubberBandToEndpointTargets(
                        next_grippers_position.first,
                        next_grippers_position.second,
                        rubber_band_verbose);
            const double band_forward_propogation_time = stopwatch(READ);
            total_band_forward_propogation_time_ += band_forward_propogation_time;

            // If we are still able to get to the next target position after retrying,
            // then return however far we were able to get
            if (!bandEndpointsMatchGripperPositions(next_band, next_grippers_position))
            {
                if (visualization_enabled_locally)
                {
                    std::cerr << "Fwd prop stopped due to rubber band endpoints not matching:\n"
                              << "Grippers Pos:       " << PrettyPrint::PrettyPrint(next_grippers_position) << std::endl
                              << "Band Endpoints Pos: " << PrettyPrint::PrettyPrint(next_band.getEndpoints()) << std::endl;
                }
                break;
            }

            // If the rubber band becomes overstretched, then return however far we were able to get
            if (next_band.isOverstretched())
            {
                if (visualization_enabled_globally_ && visualization_enabled_locally)
                {
                    next_band.visualize(RRT_FORWARD_PROP_STEPS_NS, Visualizer::Blue(), Visualizer::Cyan(), 2, true);
                    std::cerr<< "Fwd prop stopped due to band overstretch: " << EigenHelpers::CalculateTotalDistance(next_band.getVectorRepresentation()) << std::endl;
                }
                break;
            }

            if (visualization_enabled_globally_ && visualization_enabled_locally)
            {
                ++num_bands_in_visualize_list;
                const EigenHelpers::VectorVector3d band = next_band.getVectorRepresentation();
                for (size_t band_idx = 0; band_idx + 1 < band.size(); ++band_idx)
                {
                    band_visualization_line_start_points.push_back(band[band_idx]);
                    band_visualization_line_end_points.push_back(band[band_idx + 1]);
                }

                if (num_bands_in_visualize_list >= 20 || target_is_goal_config)
                {
                    vis_->visualizeLines(RRT_FORWARD_PROP_STEPS_NS, band_visualization_line_start_points, band_visualization_line_end_points, Visualizer::Blue(), 1);
                    band_visualization_line_start_points.clear();
                    band_visualization_line_end_points.clear();
                    num_bands_in_visualize_list = 0;
                }
            }

            bool is_first_order_visible = false;
            if (calculate_first_order_vis)
            {
                is_first_order_visible = isBandFirstOrderVisibileToBlacklist(next_band);
            }
            const RRTConfig next_node(
                        next_grippers_position,
                        next_robot_configuration,
                        nearest_neighbor.getUniqueForwardPropogationIndex(),
                        next_band,
                        is_first_order_visible);
            propagated_states.push_back(std::pair<RRTConfig, int64_t>(next_node, parent_offset));

            if (visualization_enabled_globally_ && visualization_enabled_locally)
            {
                const RRTGrippersRepresentation& prev_grippers_position = prev_node.getGrippers();

                gripper_a_tree_start_points.push_back(prev_grippers_position.first);
                gripper_b_tree_start_points.push_back(prev_grippers_position.second);
                gripper_a_tree_end_points.push_back(next_grippers_position.first);
                gripper_b_tree_end_points.push_back(next_grippers_position.second);

                if (gripper_a_tree_start_points.size() >= 100 || target_is_goal_config)
                {
                    vis_->visualizeLines(RRT_TREE_GRIPPER_A_NS, gripper_a_tree_start_points, gripper_a_tree_end_points, gripper_a_tree_color_, marker_id);
                    vis_->visualizeLines(RRT_TREE_GRIPPER_B_NS, gripper_b_tree_start_points, gripper_b_tree_end_points, gripper_b_tree_color_, marker_id);
                    gripper_a_tree_start_points.clear();
                    gripper_b_tree_start_points.clear();
                    gripper_a_tree_end_points.clear();
                    gripper_b_tree_end_points.clear();
                    ++marker_id;
                }
            }

            ++parent_offset;
            ++step_index;
        }
    }


    const double everything_included_forward_propogation_time = function_wide_stopwatch(READ);
    total_everything_included_forward_propogation_time_ += everything_included_forward_propogation_time;

    std::cout << "Total forward propogated states: " << propagated_states.size() << std::endl;

//    int tmp;
//    std::cin >> tmp;

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
    grippers_goal_position_ = grippers_goal;
    max_grippers_distance_ = start.getBand().maxSafeLength();
    starting_band_.reset(new RubberBand(start.getBand()));
    starting_robot_configuration_ = start.getRobotConfiguration();

    planning_for_whole_robot_ =
            starting_robot_configuration_.first.size() != 0 &&
            starting_robot_configuration_.second.size() != 0;

    if (planning_for_whole_robot_)
    {
        starting_grippers_poses_ = robot_->getGrippersPoses(starting_robot_configuration_);
        assert(starting_grippers_poses_.size() == 2);
    }

    if (visualization_enabled_globally_)
    {
        visualizeBlacklist();
    }

    // Double check that the input goal location isn't immediately impossible
    if ((environment_sdf_.EstimateDistance3d(grippers_goal_position_.first).first < gripper_min_distance_to_obstacles_) ||
        (environment_sdf_.EstimateDistance3d(grippers_goal_position_.second).first < gripper_min_distance_to_obstacles_) ||
        ((grippers_goal_position_.first - grippers_goal_position_.second).norm() > max_grippers_distance_))
    {
        std::cerr << "Unfeasible goal location: " << PrettyPrint::PrettyPrint(grippers_goal) << std::endl;
        std::cerr << "Collision Distances: " << environment_sdf_.EstimateDistance3d(grippers_goal.first).first << " " << environment_sdf_.EstimateDistance3d(grippers_goal.second).first << std::endl;
        std::cerr << "Max allowable distnace: " << max_grippers_distance_ << " Distance beteween goal grippers: " << (grippers_goal_position_.first - grippers_goal_position_.second).norm() << std::endl;
        assert(false && "Unfeasible goal location");
    }

    // Build the functions that are needed by SimpleHybridRRTPlanner
    const auto goal_reached_fn = [&] (const RRTConfig& node)
    {
        const bool goal_reached = goalReached(node);
        return goal_reached;
    };
    const auto sampling_fn = [&] ()
    {
        return configSampling();
    };
    const auto nearest_neighbor_fn = [&] (const std::vector<ExternalRRTState>& nodes, const RRTConfig& config)
    {
        const int64_t neighbour_idx = nearestNeighbour(nodes, config);
        return neighbour_idx;
    };
    const auto forward_propagation_fn = [&] (const RRTConfig& nearest_neighbor, const RRTConfig& random_target)
    {
        const bool local_visualization_enabled = true;
        const bool calculate_first_order_vis = false;
        const std::vector<std::pair<RRTConfig, int64_t>> propogation_results =
                forwardPropogationFunction(nearest_neighbor, random_target, calculate_first_order_vis, local_visualization_enabled);
        return propogation_results;
    };

    total_sampling_time_ = 0.0;
    total_nearest_neighbour_time_ = 0.0;
    total_everything_included_forward_propogation_time_ = 0.0;
    total_band_forward_propogation_time_ = 0.0;
    total_first_order_vis_propogation_time_ = 0.0;

    next_unique_forward_propogation_idx_ = 1;

    ROS_INFO_NAMED("rrt", "Starting SimpleHybridRRTPlanner");

    // Call the actual planner
    const auto rrt_results = simple_rrt_planner::SimpleHybridRRTPlanner::Plan<RRTConfig, RRTAllocator>(
                start, nearest_neighbor_fn, goal_reached_fn, sampling_fn, forward_propagation_fn, time_limit);

    if (visualization_enabled_globally_)
    {
        vis_->clearVisualizationsBullet();
    }

    statistics_["planning0_sampling_time                                 "] = total_sampling_time_;
    statistics_["planning1_nearest_neighbour_time                        "] = total_nearest_neighbour_time_;
    statistics_["planning2_forward_propogation_band_sim_time             "] = total_band_forward_propogation_time_;
    statistics_["planning3_forward_propogation_first_order_vis_time      "] = total_first_order_vis_propogation_time_;
    statistics_["planning4_forward_propogation_everything_included_time  "] = total_everything_included_forward_propogation_time_;
    statistics_["planning5_total_time                                    "] = rrt_results.second.at("planning_time");

    std::cout << "\nSimpleRRT Statistics:\n" << PrettyPrint::PrettyPrint(rrt_results.second, false, "\n") << std::endl << std::endl;
    ROS_INFO_NAMED("rrt", "Starting Shortcut Smoothing");

    const bool visualize_rrt_smoothing = true;
    const auto smoothed_path = rrtShortcutSmooth(rrt_results.first, visualize_rrt_smoothing);

    std::cout << PrettyPrint::PrettyPrint(smoothed_path.back().getBand().getVectorRepresentation(), false, "\n") << std::endl << std::endl;

    if (visualization_enabled_globally_)
    {
        vis_->deleteObjects(RRT_BLACKLISTED_GOAL_BANDS_NS, 1, 2);
    }

    std::cout << "RRT Helper Internal Statistics:\n" << PrettyPrint::PrettyPrint(statistics_, false, "\n") << std::endl << std::endl;

    return smoothed_path;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper function for goal check
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void RRTHelper::addBandToBlacklist(const EigenHelpers::VectorVector3d& band)
{
    blacklisted_goal_rubber_bands_.push_back(band);
}

void RRTHelper::clearBlacklist()
{
    blacklisted_goal_rubber_bands_.clear();
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band) const
{
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

//            if (visualize)
//            {
//                vis_->visualizeLineStrip("first_order_vis_check", {first_node, second_node}, Visualizer::White(), 2);
//            }

            const ssize_t num_steps = (ssize_t)std::ceil((second_node - first_node).norm() / environment_sdf_.GetResolution());

            // We don't need to check the endpoints as they are already checked as part of the rubber band process
            for (ssize_t ind = 1; ind < num_steps; ++ind)
            {
                const double ratio = (double)ind / (double)num_steps;
                const Eigen::Vector3d interpolated_point = EigenHelpers::Interpolate(first_node, second_node, ratio);
                if (environment_sdf_.Get3d(interpolated_point) < 0.0)
                {
//                    if (visualize)
//                    {
//                        collision_points.push_back(Eigen::Vector3d((double)blacklisted_path_ind, (double)test_band_ind, 0.0));
//                        vis.visualizePoints(
//                                    "first_order_vis_collision",
//                                    collision_points,
//                                    Visualizer::Red(),
//                                    1,
//                                    1.0);
//                        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
//                    }
                    return false;
                }
            }

            return true;
        };

        // If we've found a first order deformation, then we are similar to a blacklisted item
        if (FirstOrderDeformation::CheckFirstOrderDeformation(
                blacklisted_path.size(),
                test_band.size(),
                straight_line_collision_check_fn))
        {
            return true;
        }
    }

    return false;
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const RubberBand& test_band)
{
    Stopwatch stopwatch;
    auto vector_representation = test_band.getVectorRepresentation();

    arc_helpers::DoNotOptimize(vector_representation);
    const bool is_first_order_visible = isBandFirstOrderVisibileToBlacklist(vector_representation);
    arc_helpers::DoNotOptimize(is_first_order_visible);

    const double first_order_vis_time = stopwatch(READ);
    total_first_order_vis_propogation_time_ += first_order_vis_time;

    return is_first_order_visible;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for shortcut smoothing
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static EigenHelpers::VectorVector3d findFirstGripperWaypoints(
        const std::vector<RRTConfig, RRTAllocator>& path,
        const size_t start_index,
        const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());

    // The start of the path is clearly the first 'kink'
    EigenHelpers::VectorVector3d gripper_path_kinks(1, path[start_index].getGrippers().first);

    size_t last_kink = start_index;
    Eigen::Vector3d last_kink_gripper_position = path[last_kink].getGrippers().first;
    double path_distance = 0.0;

    // We don't include the last index because it is clearly the last 'kink'
    for (size_t idx = start_index; idx < end_index - 1; ++idx)
    {
//        std::cout << "Curr Idx: " << idx << " Grippers: " << PrettyPrint::PrettyPrint(path[idx].getGrippers()) << std::endl;
//        std::cout << "Next Idx: " << idx  + 1 << " Grippers: " << PrettyPrint::PrettyPrint(path[idx + 1].getGrippers()) << std::endl;

        const Eigen::Vector3d& current_gripper_position = path[idx].getGrippers().first;
        const Eigen::Vector3d& next_gripper_position    = path[idx + 1].getGrippers().first;
        path_distance += (next_gripper_position - current_gripper_position).norm();
        const double straight_line_distance = (next_gripper_position - last_kink_gripper_position).norm();

        // If the straight line distance between the start and the next gripper does not match the path distance, then the current node is a kink
        if (!EigenHelpers::IsApprox(straight_line_distance, path_distance, 1e-6))
        {
            last_kink = idx;
            last_kink_gripper_position = path[last_kink].getGrippers().first;
            path_distance = (next_gripper_position - current_gripper_position).norm();
            gripper_path_kinks.push_back(last_kink_gripper_position);
        }
    }
    gripper_path_kinks.push_back(path[end_index].getGrippers().first);

    return gripper_path_kinks;
}

static EigenHelpers::VectorVector3d findSecondGripperWaypoints(
        const std::vector<RRTConfig, RRTAllocator>& path,
        const size_t start_index,
        const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());

//    std::cout << "Getting second gripper waypoints: Start: " << start_index << " End: " << end_index << " Path Size: " << path.size() << std::endl;

    // The start of the path is clearly the first 'kink'
    EigenHelpers::VectorVector3d gripper_path_kinks(1, path[start_index].getGrippers().second);

    size_t last_kink = start_index;
    Eigen::Vector3d last_kink_gripper_position = path[last_kink].getGrippers().second;
    double path_distance = 0;

    // We don't include the last index because it is clearly the last 'kink'
    for (size_t idx = start_index; idx < end_index - 1; ++idx)
    {
//        std::cout << "Curr Idx: " << idx << " Grippers: " << PrettyPrint::PrettyPrint(path[idx].getGrippers()) << std::endl;
//        std::cout << "Next Idx: " << idx  + 1 << " Grippers: " << PrettyPrint::PrettyPrint(path[idx + 1].getGrippers()) << std::endl;

        const Eigen::Vector3d& current_gripper_position = path[idx].getGrippers().second;
        const Eigen::Vector3d& next_gripper_position    = path[idx + 1].getGrippers().second;
        path_distance += (next_gripper_position - current_gripper_position).norm();
        const double straight_line_distance = (next_gripper_position - last_kink_gripper_position).norm();

        // If the straight line distance between the start and the next gripper does not match the path distance, then the current node is a kink
        if (!EigenHelpers::IsApprox(straight_line_distance, path_distance, 1e-6))
        {
            last_kink = idx;
            last_kink_gripper_position = current_gripper_position;
            path_distance = (next_gripper_position - current_gripper_position).norm();
            gripper_path_kinks.push_back(last_kink_gripper_position);
        }
    }
    gripper_path_kinks.push_back(path[end_index].getGrippers().second);

    return gripper_path_kinks;
}

static EigenHelpers::VectorVector3d createOtherGripperWaypoints(
        const EigenHelpers::VectorVector3d& given_gripper_waypoints,
        const Eigen::Vector3d& start_point,
        const Eigen::Vector3d& end_point)
{
    const size_t num_waypoints = given_gripper_waypoints.size();
    assert(num_waypoints >= 2);

    EigenHelpers::VectorVector3d other_gripper_waypoints;
    other_gripper_waypoints.reserve(num_waypoints);
    other_gripper_waypoints.push_back(start_point);

    // We will need to "space out" the distance between start_point and end_point to match those of the given waypoints
    // Note that we've already inserted the first waypoint, and we'll insert the last manually as well
    const std::vector<double> cummulative_distances = EigenHelpers::CalculateCumulativeDistances(given_gripper_waypoints);
    for (size_t idx = 1; idx < num_waypoints - 1; ++idx)
    {
        const double ratio = cummulative_distances[idx] / cummulative_distances.back();
        const auto next_waypoint = EigenHelpers::Interpolate(start_point, end_point, ratio);
        other_gripper_waypoints.push_back(next_waypoint);
    }
    other_gripper_waypoints.push_back(end_point);

    assert(other_gripper_waypoints.size() == num_waypoints);
    return other_gripper_waypoints;
}

/**
 * @brief RRTHelper::forwardSimulateGrippersPath
 *   Forward simulates the rubber band starting the grippers at position path[start_index]
 *   and ending at the end of the path. Used by rrtShortcutSmooth.
 * @param rubber_band
 * @param path
 * @param start_index
 * @param end_index
 * @return A vector of RRTConfig of at most (end_index - start_index) elements; includes path[start_index].
 */
std::pair<bool, std::vector<RRTConfig, RRTAllocator>> RRTHelper::forwardSimulateGrippersPath(
        const std::vector<RRTConfig, RRTAllocator>& path,
        const size_t start_index,
        RubberBand rubber_band)
{
    Stopwatch function_wide_stopwatch;
    Stopwatch stopwatch;

    assert(start_index < path.size());
    assert(!planning_for_whole_robot_);

    // Verify that the endpoints of the rubber band match the start of the grippers path
    if (!bandEndpointsMatchGripperPositions(rubber_band, path[start_index].getGrippers()))
    {
        std::cerr << "Inside forwardSimulateGrippersPath\n";
        std::cerr << "initial rubber band endpoints:\n"
                  << PrettyPrint::PrettyPrint(rubber_band.getEndpoints()) << std::endl;

        std::cerr << "path gripper positions:\n"
                  << PrettyPrint::PrettyPrint(path[start_index].getGrippers()) << std::endl;

        assert(false && "Band endpoints do not match recorded gripper positions");
    }

    // Collect the results for use by the rrtShortcutSmooth function
    std::vector<RRTConfig, RRTAllocator> resulting_path;
    // Put the start position on the path
    {
        resulting_path.reserve(path.size() - start_index);
        const bool is_first_order_visible = isBandFirstOrderVisibileToBlacklist(rubber_band);
        resulting_path.push_back(
                    RRTConfig(path[start_index].getGrippers(),
                              path[start_index].getRobotConfiguration(),
                              path[start_index].getUniqueForwardPropogationIndex(),
                              rubber_band,
                              is_first_order_visible));
    }

    // Advance the grippers, simulating the rubber band until we reach the end of the path, or the band is overstretched
    bool band_is_overstretched = rubber_band.isOverstretched();
    bool band_got_stuck = false;
    size_t path_idx = start_index + 1;
    const bool rubber_band_verbose = false && visualization_enabled_globally_;
    while (!band_is_overstretched && !band_got_stuck && path_idx < path.size())
    {
        // Forward simulate the band
        stopwatch(RESET);
        const auto& ending_grippers_pos = path[path_idx].getGrippers();
        rubber_band.forwardPropagateRubberBandToEndpointTargets(
                    ending_grippers_pos.first,
                    ending_grippers_pos.second,
                    rubber_band_verbose);
        const double forward_propogation_time = stopwatch(READ);
        total_band_forward_propogation_time_ += forward_propogation_time;

//        rubber_band.visualize(RRT_SHORTCUT_REMAINDER_NS, Visualizer::Yellow(), Visualizer::Cyan(), (int32_t)path_idx, true);

        // Store the band in the results
//        stopwatch(RESET);
//        const bool is_first_order_visible = isBandFirstOrderVisibileToBlacklist(rubber_band);
//        const double first_order_vis_time = stopwatch(READ);
//        total_first_order_vis_propogation_time_ += first_order_vis_time;
        resulting_path.push_back(RRTConfig(ending_grippers_pos, RRTRobotRepresentation(), path[path_idx].getUniqueForwardPropogationIndex(), rubber_band, false));

        // Record if the band is overstretched
        band_is_overstretched = rubber_band.isOverstretched();
        band_got_stuck = !gripperPositionsAreApproximatelyEqual(path[path_idx].getGrippers(), rubber_band.getEndpoints());

        ++path_idx;
    }

    // Double check that if the band is not overstetched, then we've reached the endpoint of the path
    // If we haven't, then the band got stuck.
    const bool success = !band_is_overstretched && !band_got_stuck;
    if (success)
    {
        const RRTGrippersRepresentation rubber_band_endpoints = rubber_band.getEndpoints();
        assert(gripperPositionsAreApproximatelyEqual(path.back().getGrippers(), rubber_band_endpoints));
    }

    const double everything_included_forward_propogation_time = function_wide_stopwatch(READ);
    total_everything_included_forward_propogation_time_ += everything_included_forward_propogation_time;
    return std::make_pair(success, resulting_path);
}

std::vector<RRTConfig, RRTAllocator> RRTHelper::rrtShortcutSmooth(
        std::vector<RRTConfig, RRTAllocator> path,
        const bool visualization_enabled_locally)
{
    Stopwatch function_wide_stopwatch;

    uint32_t num_iterations = 0;
    uint32_t failed_iterations = 0;
    total_everything_included_forward_propogation_time_ = 0.0;
    total_band_forward_propogation_time_ = 0.0;
    total_first_order_vis_propogation_time_ = 0.0;

    if (visualization_enabled_globally_ && visualization_enabled_locally)
    {
        visualizePath(path);
    }

    // The main smoothing loop
    while (path.size() > 2 &&
           num_iterations < max_smoothing_iterations_ &&
           failed_iterations < max_failed_smoothing_iterations_)
    {
        ++num_iterations;

        if (visualization_enabled_globally_ && visualization_enabled_locally)
        {
            vis_->deleteObjects(RRT_FORWARD_PROP_STEPS_NS, 1, 2);
        }

        ///////////////////// Determine which nodes to try to shortcut between /////////////////////////////////////////

        const int64_t base_index = (int64_t)std::uniform_int_distribution<size_t>(0, path.size() - 1)(generator_);

        // Compute the offset index
        // We want to sample the start and goal slightly more frequently, so allow "overshoots" of endpoints for the offset
        const int64_t min_delta = std::max(-base_index - max_shortcut_index_distance_ / 10, -max_shortcut_index_distance_);
        const int64_t max_delta = std::min((int64_t)path.size() - base_index - 1 + max_shortcut_index_distance_ / 10, max_shortcut_index_distance_);
        const int64_t offset_delta = std::uniform_int_distribution<int64_t>(min_delta, max_delta)(generator_);
        // Clamp to the boundaries of the current path
        const int64_t second_index = arc_helpers::ClampValue(base_index + offset_delta, (int64_t)0, (int64_t)path.size() - 1);

        // Get start and end indices to establish the direction of the shortcut
        const size_t smoothing_start_index = (size_t)std::min(base_index, second_index);
        const size_t smoothing_end_index = (size_t)std::max(base_index, second_index);

        const auto& smoothing_start_config = path[smoothing_start_index];
        const auto& smoothing_end_config = path[smoothing_end_index];

        ///////////////////// Determine if a shortcut is even possible /////////////////////////////////////////////////

        // We know start_index <= end_index, this essentially checks if start == end or start + 1 == end
        if (smoothing_start_index + 1 >= smoothing_end_index)
        {
            continue;
        }
        // Distance checks are determined once we know what type of smoothing we are attempting, first, second, or both

        ///////////////////// Attempte a shortcut //////////////////////////////////////////////////////////////////////

        // Create a structure to hold the results which will get filled by each part of the if/else chain
        std::vector<std::pair<RRTConfig, int64_t>> smoothing_propogation_results;

        if (planning_for_whole_robot_)
        {
            #warning "Smoothing disabled for whole robot motion!!!"
            break;



            const double max_robot_dof_step_size = robot_->max_dof_velocity_norm_ * robot_->dt_;

            const auto& start_robot_config = smoothing_start_config.getRobotConfiguration();
            const auto& end_robot_config = smoothing_end_config.getRobotConfiguration();

            // Check if the edge possibly can be smoothed
            const double minimum_distance = RRTConfig::distance(start_robot_config, end_robot_config);
            const double path_distance = RRTConfig::robotPathDistance(path, smoothing_start_index, smoothing_end_index);
            // Essentially this checks if there is a kink in the path
            if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
            {
                continue;
            }

            const Eigen::VectorXd gripper_a_total_delta = end_robot_config.first  - start_robot_config.first;
            const Eigen::VectorXd gripper_b_total_delta = end_robot_config.second - start_robot_config.second;
            const double gripper_a_delta_norm = gripper_a_total_delta.norm();
            const double gripper_b_delta_norm = gripper_b_total_delta.norm();

            const double total_distance = std::max(gripper_a_delta_norm, gripper_b_delta_norm);
            const uint32_t max_total_steps = (uint32_t)ceil(total_distance / max_robot_dof_step_size);
            smoothing_propogation_results.reserve(max_total_steps);

            uint32_t step_index = 0;
            while (step_index < max_total_steps)
            {
                const double ratio = std::min(1.0, (double)(step_index + 1) * max_gripper_step_size_ / total_distance);
                const RRTRobotRepresentation next_robot_configuration(
                            EigenHelpers::Interpolate(start_robot_config.first,  end_robot_config.first,  ratio),
                            EigenHelpers::Interpolate(start_robot_config.second, end_robot_config.second, ratio));

                // Check if we violated joint limits
                const Eigen::VectorXd lower_limits_first  = robot_->joint_lower_limits_.head(next_robot_configuration.first.size());
                const Eigen::VectorXd lower_limits_second = robot_->joint_lower_limits_.tail(next_robot_configuration.second.size());
                const Eigen::VectorXd upper_limits_first  = robot_->joint_upper_limits_.head(next_robot_configuration.first.size());
                const Eigen::VectorXd upper_limits_second = robot_->joint_upper_limits_.tail(next_robot_configuration.second.size());
                if (((next_robot_configuration.first  - lower_limits_first).array()  <= 0.0).any() ||
                    ((next_robot_configuration.second - lower_limits_second).array() <= 0.0).any() ||
                    ((upper_limits_first  - next_robot_configuration.first).array()  <= 0.0).any() ||
                    ((upper_limits_second - next_robot_configuration.second).array() <= 0.0).any())
                {
                    assert(false && "This should not be possible");
                    break;
                }

                // Check if we entered collision
                if (robot_->checkRobotCollision(next_robot_configuration))
                {
                    break;
                }

                // Check if we rotated the grippers too much
                const auto next_grippers_poses = robot_->getGrippersPoses(next_robot_configuration);
                const double gripper_a_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_[0], next_grippers_poses[0]);
                const double gripper_b_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_[1], next_grippers_poses[1]);
                if (gripper_a_rotation_dist > max_gripper_rotation_ || gripper_b_rotation_dist > max_gripper_rotation_)
                {
                    break;
                }

                const RRTGrippersRepresentation next_grippers_position(
                            next_grippers_poses[0].translation(),
                            next_grippers_poses[1].translation());
            }
        }
        else
        {
            // First determine which type of smoothing we are doing, both grippers, or a single gripper
            // On a 1 or a 2, smooth both grippers,
            // On a 3 smooth the first gripper only,
            // On a 4 smooth the second gripper only
            const int smoothing_type = uniform_shortcut_smoothing_int_distribution_(generator_);

            if (smoothing_type == 1 || smoothing_type == 2)
            {
                // Check if the edge possibly can be smoothed
                const double minimum_distance = RRTConfig::distance(smoothing_start_config.getGrippers(), smoothing_end_config.getGrippers());
                const double path_distance = RRTConfig::grippersPathDistance(path, smoothing_start_index, smoothing_end_index);
                // Essentially this checks if there is a kink in the path
                if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
                {
                    continue;
                }

    //            if (visualization_enabled_globally_ && visualization_enabled_locally)
    //            {
    //                const RRTGrippersRepresentation& start_band_endpoints = smoothing_start_config.getBand().getEndpoints();
    //                const RRTGrippersRepresentation& end_band_endpoints = smoothing_target_end_config.getBand().getEndpoints();

    //                vis_->visualizeCubes(RRT_SHORTCUT_FIRST_GRIPPER_NS, {start_band_endpoints.first, end_band_endpoints.first}, Eigen::Vector3d(0.01, 0.01, 0.01), gripper_a_tree_color_, 10);
    //                vis_->visualizeCubes(RRT_SHORTCUT_SECOND_GRIPPER_NS, {start_band_endpoints.second, end_band_endpoints.second}, Eigen::Vector3d(0.01, 0.01, 0.01), gripper_b_tree_color_, 10);
    //                ros::spinOnce();
    //                std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
    //            }

                // Forward simulate the rubber band along the straight line between gripper positions
                next_unique_forward_propogation_idx_++;
                const RRTConfig target_config(
                            smoothing_end_config.getGrippers(),
                            smoothing_end_config.getRobotConfiguration(),
                            next_unique_forward_propogation_idx_,
                            smoothing_end_config.getBand(),
                            smoothing_end_config.isVisibleToBlacklist());

                const bool local_visualization_enabled = false;
                const bool calculate_first_order_vis = false;
                smoothing_propogation_results = forwardPropogationFunction(smoothing_start_config, target_config, calculate_first_order_vis, local_visualization_enabled);
            }
            else if (smoothing_type == 3 || smoothing_type == 4)
            {
                // Once we know the fixed waypoints, then we do smoothing between these waypoints
                const EigenHelpers::VectorVector3d current_waypoints_first_gripper =
                        findFirstGripperWaypoints(path, smoothing_start_index, smoothing_end_index);
                const EigenHelpers::VectorVector3d current_waypoints_second_gripper =
                        findSecondGripperWaypoints(path, smoothing_start_index, smoothing_end_index);

                EigenHelpers::VectorVector3d target_waypoints_first_gripper;
                EigenHelpers::VectorVector3d target_waypoints_second_gripper;
                // Smooth the first gripper
                if (smoothing_type == 3)
                {
                    const auto& first_gripper_start_pos = smoothing_start_config.getGrippers().first;
                    const auto& first_gripper_end_pos = smoothing_end_config.getGrippers().first;

                    // Check if there is room for improvement for the first gripper
                    const double minimum_distance = (first_gripper_end_pos - first_gripper_start_pos).norm();
                    const double path_distance = EigenHelpers::CalculateTotalDistance(current_waypoints_first_gripper);
                    // Essentially this checks if there is a kink in the path
                    if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
                    {
                        continue;
                    }

                    // Follow the waypoints of the second gripper
                    target_waypoints_second_gripper = current_waypoints_second_gripper;
                    // Create new waypoints in a stright line for the first gripper
                    target_waypoints_first_gripper = createOtherGripperWaypoints(
                                target_waypoints_second_gripper,
                                first_gripper_start_pos,
                                first_gripper_end_pos);
                }
                // Smooth the second gripper
                else
                {
                    const auto& second_gripper_start_pos = smoothing_start_config.getGrippers().second;
                    const auto& second_gripper_end_pos = smoothing_end_config.getGrippers().second;

                    // Check if there is room for improvement for the second gripper
                    const double minimum_distance = (second_gripper_end_pos - second_gripper_start_pos).norm();
                    const double path_distance = EigenHelpers::CalculateTotalDistance(current_waypoints_first_gripper);
                    // Essentially this checks if there is a kink in the path
                    if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
                    {
                        continue;
                    }

                    // Follow the waypoints of the first gripper
                    target_waypoints_first_gripper = current_waypoints_first_gripper;
                    // Create new waypoints in a stright line for the second gripper
                    target_waypoints_second_gripper = createOtherGripperWaypoints(
                                target_waypoints_first_gripper,
                                second_gripper_start_pos,
                                second_gripper_end_pos);
                }

    //            if (visualization_enabled_globally_ && visualization_enabled_locally)
    //            {
    //                vis_->visualizeCubes(RRT_SHORTCUT_FIRST_GRIPPER_NS, target_waypoints_first_gripper, Eigen::Vector3d(0.01, 0.01, 0.01), gripper_a_tree_color_, 10);
    //                vis_->visualizeCubes(RRT_SHORTCUT_SECOND_GRIPPER_NS, target_waypoints_second_gripper, Eigen::Vector3d(0.01, 0.01, 0.01), gripper_b_tree_color_, 10);
    //                ros::spinOnce();
    //                std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
    //            }

                // Make a guess about the number of nodes we'll end up using
                assert(target_waypoints_first_gripper.size() == target_waypoints_second_gripper.size());
                const size_t num_waypoints = target_waypoints_first_gripper.size();
                smoothing_propogation_results.reserve(10 * num_waypoints);

                // Now that we have the waypoints, start building the smoothed path, exiting early if we encouter an infeasible configuration
                RRTConfig interm_start_config = smoothing_start_config;
                for (size_t waypoint_idx = 1; waypoint_idx < num_waypoints; ++waypoint_idx)
                {
                    next_unique_forward_propogation_idx_++;
                    const RRTConfig forward_prop_target_config(
                                RRTGrippersRepresentation(
                                    target_waypoints_first_gripper[waypoint_idx],
                                    target_waypoints_second_gripper[waypoint_idx]),
                                RRTRobotRepresentation(),
                                next_unique_forward_propogation_idx_,
                                path.front().getBand(),
                                false);

    //                if (visualization_enabled_globally_)
    //                {
    //                    vis_->visualizeCubes(RRT_SHORTCUT_START_NS, {interm_start_config.getGrippers().first, interm_start_config.getGrippers().second, forward_prop_target_config.getGrippers().first, forward_prop_target_config.getGrippers().second}, Eigen::Vector3d(0.02, 0.02, 0.02), Visualizer::Yellow(), 5);
    //                    ros::spinOnce();
    //                    std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
    //                }

                    const bool local_visualization_enabled = false;
                    const bool calculate_first_order_vis = false;
                    const auto interm_forward_prop_results = forwardPropogationFunction(interm_start_config, forward_prop_target_config, calculate_first_order_vis, local_visualization_enabled);
                    const auto& target_gripper_position = forward_prop_target_config.getGrippers();
                    const auto& last_gripper_position = interm_forward_prop_results.back().first.getGrippers();

                    // Check if the rubber band gets overstretched while propogating the grippers on the new path
                    if (interm_forward_prop_results.size() == 0 ||
                        !gripperPositionsAreApproximatelyEqual(last_gripper_position, target_gripper_position))
                    {
    //                    // If we have overstretched or otherwise have problems, exit the loop
    //                    // We will let the exterior check for success keep track of the failed iteration count
    //                    std::cerr << "fwd prop target:    " << PrettyPrint::PrettyPrint(forward_prop_target_config.getGrippers()) << std::endl;
    //                    std::cerr << "target_grippers:    " << PrettyPrint::PrettyPrint(target_gripper_position) << std::endl;
    //                    std::cerr << "interm_fwd_prop size: " << interm_forward_prop_results.size();
    //                    std::cerr << "resulting_grippers: " << PrettyPrint::PrettyPrint(interm_forward_prop_results, false, "\n") << std::endl;
    //                    std::cerr << std::flush;
    //                    std::cerr << std::flush;

                        break;
                    }

                    // If we were succesful, then insert the results, and move the interm start config forward by a waypoint
                    smoothing_propogation_results.insert(smoothing_propogation_results.end(), interm_forward_prop_results.begin(), interm_forward_prop_results.end());
                    interm_start_config = smoothing_propogation_results.back().first;
                }

    //            if (visualization_enabled_globally_)
    //            {
    //                vis_->deleteObjects(RRT_SHORTCUT_START_NS, 5, 6);
    //                ros::spinOnce();
    //                std::this_thread::sleep_for(std::chrono::duration<double>(0.1));
    //            }
            }
            else
            {
                assert(false && "Smoothing type was something other than [1, 4], this ougth to be impossible");
            }

            // Check if the rubber band gets overstretched while propogating the grippers on the new path
            const auto& target_gripper_position = smoothing_end_config.getGrippers();
            const auto& last_gripper_position = smoothing_propogation_results.back().first.getGrippers();
            if (smoothing_propogation_results.size() == 0 ||
                !gripperPositionsAreApproximatelyEqual(last_gripper_position, target_gripper_position))
            {
                ++failed_iterations;
                continue;
            }

            // We still need to check that the rubber band can still reach the goal correctly from this state,
            // so we'll forward propogate along the rest of the trajectory to check feasibility
            const std::pair<bool, std::vector<RRTConfig, RRTAllocator>> end_of_smoothing_to_goal_results =
                    forwardSimulateGrippersPath(path, smoothing_end_index, smoothing_propogation_results.back().first.getBand());
            const bool final_band_at_goal_success = end_of_smoothing_to_goal_results.first;
            const auto& end_of_smoothing_to_goal_path_ = end_of_smoothing_to_goal_results.second;

            // Check if the rubber band gets overstretched or ends up in a blacklisted first order
            // homotopy class while following the tail of the starting trajectory
            {
                const auto& final_node_of_smoothing = end_of_smoothing_to_goal_path_.back();
                const bool final_band_visible_to_blacklist = isBandFirstOrderVisibileToBlacklist(final_node_of_smoothing.getBand());
                if (!final_band_at_goal_success || final_band_visible_to_blacklist)
                {
                    ++failed_iterations;
                    continue;
                }
            }

            ///////////////////// Smoothing success - Create the new smoothed path /////////////////////////////////////////

            // Allocate space for the total smoothed path
            std::vector<RRTConfig, RRTAllocator> smoothed_path;
            smoothed_path.reserve((smoothing_start_index  + 1) + smoothing_propogation_results.size() + (end_of_smoothing_to_goal_path_.size() - 1));

            // Insert the starting unchanged part of the path
            smoothed_path.insert(smoothed_path.begin(), path.begin(), path.begin() + smoothing_start_index + 1);

            // Insert the smoothed portion
            for (size_t idx = 0; idx < smoothing_propogation_results.size(); ++idx)
            {
                smoothed_path.push_back(smoothing_propogation_results[idx].first);
            }

            // Insert the changed end of the path with the new rubber band - gripper positions are identical
            smoothed_path.insert(smoothed_path.end(), end_of_smoothing_to_goal_path_.begin() + 1, end_of_smoothing_to_goal_path_.end());

            // Record the change and re-visualize
            path = smoothed_path;
            if (visualization_enabled_globally_ && visualization_enabled_locally)
            {
                visualizePath(path);
            }
        }

    //    if (visualization_enabled_globally_ && visualization_enabled_locally)
    //    {
    //        vis_->deleteObjects(RRT_SHORTCUT_FIRST_GRIPPER_NS, 1, 21);
    //        vis_->deleteObjects(RRT_SHORTCUT_SECOND_GRIPPER_NS, 1, 21);
    //    }
    //    vis_->clearVisualizationsBullet();
    }

    // Record the statistics and return the result
    const double smoothing_time = function_wide_stopwatch(READ);

    statistics_["smoothing0_failed_iterations                            "] = (double)failed_iterations;
    statistics_["smoothing1_iterations                                   "] = (double)num_iterations;
    statistics_["smoothing2_forward_propogation_band_sim_time            "] = total_band_forward_propogation_time_;
    statistics_["smoothing3_forward_propogation_first_order_vis_time     "] = total_first_order_vis_propogation_time_;
    statistics_["smoothing4_forward_propogation_everything_included_time "] = total_everything_included_forward_propogation_time_;
    statistics_["smoothing5_total_time                                   "] = smoothing_time;

    return path;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Visualization and other debugging tools
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void RRTHelper::visualizePath(const std::vector<RRTConfig, RRTAllocator>& path) const
{
    if (visualization_enabled_globally_)
    {
        EigenHelpers::VectorVector3d gripper_a_cubes;
        EigenHelpers::VectorVector3d gripper_b_cubes;
        gripper_a_cubes.reserve(path.size());
        gripper_b_cubes.reserve(path.size());

        EigenHelpers::VectorVector3d line_start_points;
        EigenHelpers::VectorVector3d line_end_points;

        for (int32_t ind = 0; ind < (int32_t)path.size(); ++ind)
        {
            const RRTConfig& config = path[ind];
            const RRTGrippersRepresentation& gripper_positions = config.getGrippers();
            const RubberBand& rubber_band = config.getBand();

            gripper_a_cubes.push_back(gripper_positions.first);
            gripper_b_cubes.push_back(gripper_positions.second);

            const EigenHelpers::VectorVector3d band = rubber_band.getVectorRepresentation();
            for (size_t band_idx = 0; band_idx + 1< band.size(); ++band_idx)
            {
                line_start_points.push_back(band[band_idx]);
                line_end_points.push_back(band[band_idx + 1]);
            }
        }

        vis_->visualizeCubes(RRT_SOLUTION_GRIPPER_A_NS, gripper_a_cubes, Eigen::Vector3d(0.005, 0.005, 0.005), gripper_a_tree_color_, 1);
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
        vis_->visualizeCubes(RRT_SOLUTION_GRIPPER_B_NS, gripper_b_cubes, Eigen::Vector3d(0.005, 0.005, 0.005), gripper_b_tree_color_, 1);
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
//        vis_->visualizeLines(RRT_SOLUTION_RUBBER_BAND_NS, line_start_points, line_end_points, Visualizer::Yellow(), 1);
        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
    }
}

void RRTHelper::visualizeBlacklist() const
{
    EigenHelpers::VectorVector3d line_start_points;
    EigenHelpers::VectorVector3d line_end_points;

    for (size_t idx = 0; idx < blacklisted_goal_rubber_bands_.size(); ++idx)
    {
        const auto& band = blacklisted_goal_rubber_bands_[idx];
        for (size_t band_idx = 1; band_idx < band.size(); ++band_idx)
        {
            line_start_points.push_back(band[band_idx - 1]);
            line_end_points.push_back(band[band_idx]);
        }
    }

    vis_->visualizeLines(RRT_BLACKLISTED_GOAL_BANDS_NS, line_start_points, line_end_points, Visualizer::Red(), 1, 0.01);
}
