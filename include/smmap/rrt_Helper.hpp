#ifndef RRT_HELPER_H
#define RRT_HELPER_H

//#include <iostream>
//#include <stdio.h>
//#include <vector>


#include <arc_utilities/arc_helpers.hpp>
#include "smmap/virtual_rubber_band.h"

#include <arc_utilities/simple_rrt_planner.hpp>
#include <uncertainty_planning_core/include/uncertainty_planning_core/simple_samplers.hpp>

#define ERRORTHRESHOLD 0.002
#define GOALERROR 0.3
#define PI 3.141592653
#define STEPSIZE 0.3
#define SAMPLEBIAS 0.1
#define NUMITERATION 200
#define DOFCONFIG 6
#define STRETCHING_THRESHOLD 0

namespace smmap {

    typedef std::pair<std::pair<Eigen::Affine3d, Eigen::Affine3d>, VirtualRubberBand> rrtConfig;

    namespace Rrt_function{

        // * nearest_neighbor_fn - given all nodes explored so far, and a new state, return the index of the "closest" node
        template<typename T, typename Allocator = std::allocator<T>>
        inline int64_t nearestNeighbor(
                const std::vector<T, Allocator>& nodes,
                const T& config,
                const std::function<double(const T&, const T&)>& distance_fn)
        {
            int64_t res = -1;
            double best_distance = INFINITY;
            for (size_t inode = 0; inode < nodes.size(); cluster++)
            {
                const T& neighbor = nodes[inode];
                const double distance = distance_fn(neighbor, config);
                if (distance < best_distance)
                {
                    best_distance = distance;
                    res = (int64_t)inode;
                }
            }
            assert(res >= 0);
            return res;
        }

        // distance function for T = rrtConfig
        // std::pair<std::pair<Eigen::Affine3d, Eigen::Affine3d>, VirtualRubberBand>
        // ??????? QUESTION: SHOULD I FORWARD SIMULATING RUBBERBAND TO SEE IF THE TWO STATES ARE TRANSFERABLE ??????
        double withBandDistance(const rrtConfig& Anode, const rrtConfig& Bnode)
        {
            double distance = 0;


            return distance;
        }

        // returned distance is the euclidian distance of two grippers pos
        // ?????????????? Here I simply define it as the sum of two grippers distance, ????????????????????????
        std::pair<double, double> affine3dPairDistance(
                const std::pair<Eigen::Affine3d, Eigen::Affine3d>& Anode,
                const std::pair<Eigen::Affine3d, Eigen::Affine3d>& Bnode)
        {
            std::pair<double, double> distance = 0;
            Eigen::Vector3d transA_first = Anode.first.translation();
            Eigen::Vector3d transB_first = Bnode.first.translation();
            Eigen::Vector3d transA_second = Anode.second.translation();
            Eigen::Vector3d transB_second = Bnode.second.translation();
            distance.first = (transA_first-transB_first).norm();
            distance.second = (transA_second-transB_second).norm();
            return distance;
        }


        // const std::function<void(SimpleRRTPlannerState<T, Allocator>&, SimpleRRTPlannerState<T, Allocator>&)>& state_added_fn
        // * state_added_fn - callback function that takes (parent, child) for each extension
        template<typename T, typename Allocator = std::allocator<T>>
        inline void stateAdd(
                simple_rrt_planner::SimpleHybridRRTPlanner<T,Allocator>& parent,
                simple_rrt_planner::SimpleHybridRRTPlanner<T,Allocator>& child)
        {


        }


        /* const std::function<bool(const T&)>& goal_reached_fn,
         * goal_reached_fn - return if a given state meets the goal conditions (for example, within a radius of a goal state)
         const std::function<void(SimpleRRTPlannerState<T, Allocator>&)>& goal_reached_callback_fn,
         CAN USE DISTANCE FUNCTION INSTEAD
        */


        // const std::function<T(void)>& sampling_fn,
        // * state_sampling_fn - returns a new state (randomly- or deterministically-sampled)
        // ????????? Should I do forward simulation to make sure it is feasible from current configuration ?????
        template <typename Generator>
        inline std::pair<Eigen::Affine3d, Eigen::Affine3d> se3PairSampling(
                Generator& prng,
                simple_samplers::SimpleSE3BaseSampler se3_sampler)
        {
            std::srand(std::time(0));
            std::pair<Eigen::Affine3d, Eigen::Affine3d> rand_sample;

            rand_sample.first = se3_sampler.Sample(prng);
            rand_sample.second = se3_sampler.Sample(prng);

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
        // Overstretch checking has been in sendNextCommand(), need constraint violation, and collision checking
        inline std::vector<std::pair<rrtConfig, int64_t>> rrtConfig_takeStep(
                const rrtConfig& nearest_neighbor,
                const std::pair<Eigen::Affine3d, Eigen::Affine3d>& random_target,
                const std::function<std::pair<Eigen::Affine3d, Eigen::Affine3d>(const std::pair<Eigen::Affine3d, Eigen::Affine3d>&, const std::pair<Eigen::Affine3d, Eigen::Affine3d>&, const double)>& step_fn,
                const double step_size)
        {
            std::vector<std::pair<rrtConfig, int64_t>> propagation;
            const std::pair<Eigen::Affine3d, Eigen::Affine3d> start_pos = nearest_neighbor.first;
            const std::pair<Eigen::Affine3d, Eigen::Affine3d> step_to_take = step_fn(start_pos, random_target, step_size);

            // ???? share_ptr ?????????????
            VirtualRubberBand band_copy = nearest_neighbor.second.get();

            rrtConfig& last_node = nearest_neighbor;
            // keep stepping if not within one stepsize to random_target
            while(affine3dPairDistance(start_pos,random_target)>step_size)
            {


            }



            return propagation;
        }

        // one candidate for step_fn above, generate uniform step based on euclidian distance
        // I defined the step distance as the some of step distances of two grippers
        std::pair<Eigen::Affine3d, Eigen::Affine3d> getPairEuclidianStep(
                const std::pair<Eigen::Affine3d, Eigen::Affine3d>& start,
                const std::pair<Eigen::Affine3d, Eigen::Affine3d>& goal,
                const double step_size)
        {
            std::pair<Eigen::Affine3d, Eigen::Affine3d> step_to_take;

            const Eigen::Vector3d first_vector = goal.first.translation()-start.first.translation();
            const Eigen::Vector3d second_vector = goal.second.translation()-start.second.translation();
            const std::pair<double, double> pair_distance = affine3dPairDistance(start,goal);
            const double sum_distance = pair_distance.first + pair_distance.second;

            if(sum_distance < step_size)
            {   return goal;  }
            step_to_take.first = first_vector * step_size/sum_distance;
            step_to_take.second = second_vector * step_size/sum_distance;
            return step_to_take;
        }


        // const std::function<bool(void)>& termination_check_fn
        // * termination_check_fn - returns if the planner should terminate (for example, if it has exceeded time/space limits)




    }




}






#endif
