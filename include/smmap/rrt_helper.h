#ifndef RRT_HELPER_H
#define RRT_HELPER_H

//#include <iostream>
//#include <stdio.h>
//#include <vector>


#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <uncertainty_planning_core/simple_samplers.hpp>

#include "smmap/virtual_rubber_band.h"

namespace smmap
{
    class RRTHelper
    {
        public:
            typedef std::pair<std::pair<Eigen::Vector3d, Eigen::Vector3d>, VirtualRubberBand> RRTConfig;
            typedef std::allocator<RRTConfig> Allocator;

            #warning "Replace these magic numbers
            RRTHelper(const double step_size = 1.0)
                : step_size_(step_size)
            {
            }


            // returned distance is the euclidian distance of two grippers pos
            // ?????????????? Here I simply define it as the sum of two grippers distance, ????????????????????????
            double distance(
                    const RRTConfig& a_node,
                    const RRTConfig& b_node)
            {
                const Eigen::Vector3d& transA_first = a_node.first.first;
                const Eigen::Vector3d& transB_first = b_node.first.first;
                const Eigen::Vector3d& transA_second = a_node.first.second;
                const Eigen::Vector3d& transB_second = b_node.first.second;
                return (transA_first - transB_first).norm() + (transA_second - transB_second).norm();
            }

            int64_t nearestNeighbour(
                    const std::vector<simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, Allocator>>& nodes,
                    const RRTConfig& config)
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

            /* const std::function<bool(const T&)>& goal_reached_fn,
             * goal_reached_fn - return if a given state meets the goal conditions (for example, within a radius of a goal state)
             const std::function<void(SimpleRRTPlannerState<T, Allocator>&)>& goal_reached_callback_fn,
             CAN USE DISTANCE FUNCTION INSTEAD
            */


            // const std::function<T(void)>& sampling_fn,
            // * state_sampling_fn - returns a new state (randomly- or deterministically-sampled)
            // ????????? Should I do forward simulation to make sure it is feasible from current configuration ?????
//            template <typename Generator>
//            inline std::pair<Eigen::Affine3d, Eigen::Affine3d> SE3PairSampling(
//                    Generator& prng,
//                    simple_samplers::SimpleSE3BaseSampler se3_sampler)
//            {
//                std::srand(std::time(0));
//                std::pair<Eigen::Affine3d, Eigen::Affine3d> rand_sample;

//                rand_sample.first = se3_sampler.Sample(prng);
//                rand_sample.second = se3_sampler.Sample(prng);

//                return rand_sample;
//            }


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
            // Overstretch checking has been in sendNextCommand(), need constraint violation, and collision checking
            inline std::vector<std::pair<RRTConfig, int64_t>> forwardPropogationFunction(
                    const RRTConfig& nearest_neighbor,
                    const RRTConfig& random_target)
            {
                std::vector<std::pair<RRTConfig, int64_t>> propagated_states;
                int64_t parent_offset = -1;

                const double total_distance = distance(nearest_neighbor, random_target);
                const uint32_t max_total_steps = (uint32_t)ceil(total_distance / step_size_);

                const bool rubber_band_verbose = false;
                uint32_t step_index = 0;
                bool completed = false;
                while ((completed == false) && (step_index < max_total_steps))
                {
                    // Using ternary operator here so that we can avoid making copies, and still take advantage of const correctness
                    const RRTConfig& prev_node = parent_offset == -1 ? nearest_neighbor : propagated_states[parent_offset].first;

                    const double ratio = std::min(1.0, (double)(step_index + 1) * step_size_ / total_distance);
                    const Eigen::Vector3d gripper_a_interpolated = EigenHelpers::Interpolate(nearest_neighbor.first.first, random_target.first.first, ratio);
                    const Eigen::Vector3d gripper_b_interpolated = EigenHelpers::Interpolate(nearest_neighbor.first.second, random_target.first.second, ratio);

                    // TODO: collision check gripper positions

                    VirtualRubberBand next_rubber_band(prev_node.second);
                    next_rubber_band.forwardSimulateVirtualRubberBand(
                                gripper_a_interpolated - prev_node.first.first,
                                gripper_b_interpolated - prev_node.first.second,
                                rubber_band_verbose);

                    // TODO: check rubber band overstretch

                    const RRTConfig next_node(std::make_pair(gripper_a_interpolated, gripper_b_interpolated), next_rubber_band);
                    propagated_states.push_back(std::make_pair(next_node, parent_offset));

                    ++parent_offset;
                    ++step_index;
                }

                return propagated_states;
            }

        private:
            const double step_size_;


//             const std::function<bool(void)>& termination_check_fn
//             * termination_check_fn - returns if the planner should terminate (for example, if it has exceeded time/space limits)



//            const auto distance_fn = [&] (
//                    const simple_rrt_planner::SimpleRRTPlannerState<RRTHelper::RRTConfig, RRTHelper::Allocator>& rrt_state,
//                    const RRTHelper::RRTConfig& rrt_config)
//            {
//                return rrt_helper.distance(rrt_state.GetValueImmutable(), rrt_config);
//            };

//            const auto nearest_neighbor_fn = [&] (
//                    const std::vector<simple_rrt_planner::SimpleRRTPlannerState<RRTHelper::RRTConfig, RRTHelper::Allocator>>& nodes,
//                    const RRTHelper::RRTConfig& config)
//            {
//                return rrt_helper.nearestNeighbour(nodes, config);
//            };
//            const auto results = simple_rrt_planner::SimpleHybridRRTPlanner::Plan<rrtConfig, std::allocator<rrtConfig>>(start, goal, nearest_neighbor_fn, goal_reached_fn, state_sampling_fn, forward_propagation_fn, goal_bias, time_limit, rng);

    };
}






#endif
