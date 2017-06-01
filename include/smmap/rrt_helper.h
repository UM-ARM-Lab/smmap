#ifndef RRT_HELPER_H
#define RRT_HELPER_H

#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <arc_utilities/shortcut_smoothing.hpp>
#include <uncertainty_planning_core/simple_samplers.hpp>

#include "smmap/visualization_tools.h"
#include "smmap/virtual_rubber_band.h"

namespace smmap
{
    class RRTHelper
    {
        private:
            const std::pair<double, double> x_limits_;
            const std::pair<double, double> y_limits_;
            const std::pair<double, double> z_limits_;
            const double goal_reach_radius_;
            const double step_size_;
            const double max_shortcut_fraction_;
            const uint32_t max_iterations_;
            const uint32_t max_failed_iterations_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;

            const sdf_tools::SignedDistanceField& environment_sdf_;
            const Visualizer& vis_;
            int32_t marker_id_;
            const std_msgs::ColorRGBA rubber_band_safe_color_;
            const std_msgs::ColorRGBA rubber_band_overstretched_color_;

            std::pair<Eigen::Vector3d, Eigen::Vector3d> gripper_goal_positions_;

        public:
            typedef std::pair<std::pair<Eigen::Vector3d, Eigen::Vector3d>, VirtualRubberBand> RRTConfig;
            typedef std::allocator<RRTConfig> Allocator;


            RRTHelper(const sdf_tools::SignedDistanceField& environment_sdf,
                      const Visualizer& vis,
                      const double step_size = 1.0,
                      const double x_limits_lower = -10.0,
                      const double x_limits_upper = 10.0,
                      const double y_limits_lower = -10.0,
                      const double y_limits_upper = 10.0,
                      const double z_limits_lower = -10.0,
                      const double z_limits_upper = 10.0,
                      const double goal_reach_radius = 1.0,
                      const double max_shortcut_fraction = 1.0,
                      const uint32_t max_iterations = 200,
                      const uint32_t max_failed_iterations = 500)
                : x_limits_(x_limits_lower,x_limits_upper)
                , y_limits_(y_limits_lower,y_limits_upper)
                , z_limits_(z_limits_lower,z_limits_upper)
                , goal_reach_radius_(goal_reach_radius)
                , step_size_(step_size)
                , max_shortcut_fraction_(max_shortcut_fraction)
                , max_iterations_(max_iterations)
                , max_failed_iterations_(max_failed_iterations)
                , uniform_unit_distribution_(0.0, 1.0)
                , environment_sdf_(environment_sdf)
                , vis_(vis)
                , marker_id_(1)
                , rubber_band_safe_color_(Visualizer::Black())
                , rubber_band_overstretched_color_(arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 1.0f, 1.0f, 1.0f))
            {}

            ///////////////////////////////////////////////////////////////////////////////////////////////
            // Helper function for original rrt planning
            ///////////////////////////////////////////////////////////////////////////////////////////////

            // returned distance is the euclidian distance of two grippers pos
            double distance(
                    const RRTConfig& a_node,
                    const RRTConfig& b_node) const
            {
                const Eigen::Vector3d& transA_first = a_node.first.first;
                const Eigen::Vector3d& transB_first = b_node.first.first;
                const Eigen::Vector3d& transA_second = a_node.first.second;
                const Eigen::Vector3d& transB_second = b_node.first.second;
                return (transA_first - transB_first).norm() + (transA_second - transB_second).norm();
            }

            int64_t nearestNeighbour(
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
            template <typename RNG>
            inline std::pair<Eigen::Vector3d, Eigen::Vector3d> posPairSampling(
                    RNG& prng)
            {
                std::pair<Eigen::Vector3d, Eigen::Vector3d> rand_sample;
                const bool sample_goal = uniform_unit_distribution_(prng) < 0.1;

                if (sample_goal)
                {
                    rand_sample = gripper_goal_positions_;
                }
                else
                {
                    const double x1 = EigenHelpers::Interpolate(x_limits_.first, x_limits_.second, uniform_unit_distribution_(prng));
                    const double y1 = EigenHelpers::Interpolate(y_limits_.first, y_limits_.second, uniform_unit_distribution_(prng));
                    const double z1 = EigenHelpers::Interpolate(z_limits_.first, z_limits_.second, uniform_unit_distribution_(prng));

                    const double x2 = EigenHelpers::Interpolate(x_limits_.first, x_limits_.second, uniform_unit_distribution_(prng));
                    const double y2 = EigenHelpers::Interpolate(y_limits_.first, y_limits_.second, uniform_unit_distribution_(prng));
                    const double z2 = EigenHelpers::Interpolate(z_limits_.first, z_limits_.second, uniform_unit_distribution_(prng));

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
            std::vector<std::pair<RRTConfig, int64_t>> forwardPropogationFunction(
                    const RRTConfig& nearest_neighbor,
                    const RRTConfig& random_target)
            {
                vis_.visualizeCubes("forward_propogation_start_config", {nearest_neighbor.first.first, nearest_neighbor.first.second}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), 1);
                vis_.visualizePoints("random_target", {random_target.first.first, random_target.first.second}, Visualizer::Red(), 1);

                for (int32_t id = 1; id < 50; id++)
                {
                    nearest_neighbor.second.visualize("rubber_band_post_rrt_step", rubber_band_safe_color_, rubber_band_overstretched_color_, id, true);
                }

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

                    next_rubber_band.visualize("rubber_band_post_rrt_step", Visualizer::Black(), arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 1.0f, 1.0f, 1.0f), (int32_t)parent_offset + 2, true);

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

                for (int32_t id = 1; id < 50; id++)
                {
                    nearest_neighbor.second.visualize("rubber_band_post_rrt_step", rubber_band_safe_color_, rubber_band_overstretched_color_, id, true);
                }

                return propagated_states;
            }

            template<typename RNG>
            std::vector<RRTConfig, Allocator> rrtPlan (const RRTConfig& start,
                                                       const RRTConfig& goal,
                                                       const std::chrono::duration<double>& time_limit,
                                                       RNG& rng)
            {
                gripper_goal_positions_ = goal.first;

                const std::function<bool(const RRTConfig&)> goal_reached_fn = [&] (const RRTConfig& node)
                {
                    if (distance(node, goal) < goal_reach_radius_)
                    {
                        return true;
                    }
                    return false;
                };

                const std::function<RRTConfig(void)> sampling_fn = [&] ()
                {
                    RRTConfig sample_config(posPairSampling(rng), start.second);
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

                std::cout << PrettyPrint::PrettyPrint(rrt_results.second, true, "\n") << std::endl;
                return rrt_results.first;
            }

            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper function for shortcur smoothing
            ///////////////////////////////////////////////////////////////////////////////////////

            /* const std::function<bool(const Configuration&, const Configuration&)>& edge_validity_check_fn
             */


            template<typename RNG>
            std::vector<RRTConfig, Allocator> rrtShortcurSmooth (const std::vector<RRTConfig, Allocator>& path,
                                                                 RNG & rng)
            {
                const std::function<bool(const RRTConfig&, const RRTConfig&)> edge_validity_check_fn = [&] (
                        const RRTConfig& start_node,
                        const RRTConfig& end_node)
                {
                    vis_.visualizeCubes("shortcut_smoothing_start_config", {start_node.first.first, start_node.first.second}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), 1);
                    vis_.visualizePoints("shortcut_smoothing_end_config", {end_node.first.first, end_node.first.second}, Visualizer::Red(), 1);

                    for (int32_t id = 1; id < 50; id++)
                    {
                        start_node.second.visualize("rubber_band_post_rrt_step", rubber_band_safe_color_, rubber_band_overstretched_color_, id, true);
                    }

                    std::vector<std::pair<RRTConfig, int64_t>> propagated_states;
                    int64_t parent_offset = -1;

                    const double total_distance = distance(start_node, end_node);
                    const uint32_t max_total_steps = (uint32_t)ceil(total_distance / step_size_);
                    propagated_states.reserve(max_total_steps);

                    const bool rubber_band_verbose = false;
                    uint32_t step_index = 0;

                    while (step_index < max_total_steps)
                    {
                        // Using ternary operator here so that we can avoid making copies, and still take advantage of const correctness
                        const bool use_start_node_as_prev = (parent_offset == -1);
                        const RRTConfig& prev_node = (use_start_node_as_prev ? start_node : propagated_states[parent_offset].first);

                        const double ratio = std::min(1.0, (double)(step_index + 1) * step_size_ / total_distance);
                        const Eigen::Vector3d gripper_a_interpolated = EigenHelpers::Interpolate(start_node.first.first, end_node.first.first, ratio);
                        const Eigen::Vector3d gripper_b_interpolated = EigenHelpers::Interpolate(start_node.first.second, end_node.first.second, ratio);

                        // If the grippers enter collision, shortcut smoothing fails
                        if ((environment_sdf_.Get3d(gripper_a_interpolated) < 0.0)
                                || (environment_sdf_.Get3d(gripper_b_interpolated) < 0.0))
                        {
                            return false;
                        }

                        VirtualRubberBand next_rubber_band(prev_node.second);
                        next_rubber_band.forwardSimulateVirtualRubberBand(
                                    gripper_a_interpolated - prev_node.first.first,
                                    gripper_b_interpolated - prev_node.first.second,
                                    rubber_band_verbose);

                        next_rubber_band.visualize("rubber_band_in_smoothing_step", Visualizer::Black(), arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>::MakeFromFloatColors(0.0f, 1.0f, 1.0f, 1.0f), (int32_t)parent_offset + 2, true);

                        // If the rubber band becomes overstretched, shortcut smoothing fails
                        if (next_rubber_band.isOverstretched())
                        {
                            return false;
                        }

                        const RRTConfig next_node(std::make_pair(gripper_a_interpolated, gripper_b_interpolated), next_rubber_band);
    //                    vis_.visualizeLineStrip("rrt_gripper_tree_gripper_a", {prev_node.first.first, next_node.first.first}, Visualizer::Red(), marker_id_);
    //                    vis_.visualizeLineStrip("rrt_gripper_tree_gripper_b", {prev_node.first.second, next_node.first.second}, Visualizer::Blue(), marker_id_);
                        propagated_states.push_back(std::pair<RRTConfig, int64_t>(next_node, parent_offset));

                        ++marker_id_;
                        ++parent_offset;
                        ++step_index;
                    }
                    return true;
                };


                const auto rrt_shortcut_results = shortcut_smoothing::ShortcutSmoothPath(
                            path,
                            max_iterations_,
                            max_failed_iterations_,
                            max_shortcut_fraction_,
                            edge_validity_check_fn,
                            rng);

                return rrt_shortcut_results;
            }


            ///////////////////////////////////////////////////////////////////////////////////////
            // Visualization and other debugging tools
            ///////////////////////////////////////////////////////////////////////////////////////


            void visualize(const std::vector<RRTConfig, Allocator>& path)
            {
                for (size_t ind = 0; ind < path.size(); ++ind)
                {
                    const RRTConfig& config = path[ind];
                    const auto& gripper_positions = config.first;
                    const VirtualRubberBand& rubber_band = config.second;

                    rubber_band.visualize("rubber_band_rrt_solution", rubber_band_safe_color_, rubber_band_overstretched_color_, (int32_t)ind + 1, true);
                    vis_.visualizeCubes("gripper_a_rrt_solution", {gripper_positions.first}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Blue(), (int32_t)ind + 1);
                    vis_.visualizeCubes("gripper_b_rrt_solution", {gripper_positions.second}, Eigen::Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), (int32_t)ind + 1);
                }
            }
    };
}


#endif // ifndef RRT_HELPER_H
