#ifndef RRT_HELPER_H
#define RRT_HELPER_H

//#include <arc_utilities/arc_helpers.hpp>

#include <random>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>

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
            const int64_t max_shortcut_index_distance_;
            const uint32_t max_smoothing_iterations_;
            const uint32_t max_failed_smoothing_iterations_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;

            const sdf_tools::SignedDistanceField& environment_sdf_;
            const Visualizer& vis_;
            std::mt19937_64& generator_;

            int32_t marker_id_;
            const std_msgs::ColorRGBA rubber_band_safe_color_;
            const std_msgs::ColorRGBA rubber_band_overstretched_color_;

            std::pair<Eigen::Vector3d, Eigen::Vector3d> gripper_goal_positions_;

        public:
            typedef std::pair<std::pair<Eigen::Vector3d, Eigen::Vector3d>, VirtualRubberBand> RRTConfig;
            typedef std::allocator<RRTConfig> Allocator;


            RRTHelper(
                    const sdf_tools::SignedDistanceField& environment_sdf,
                    const Visualizer& vis,
                    std::mt19937_64& generator,
                    const double step_size = 1.0,
                    const double x_limits_lower = -10.0,
                    const double x_limits_upper = 10.0,
                    const double y_limits_lower = -10.0,
                    const double y_limits_upper = 10.0,
                    const double z_limits_lower = -10.0,
                    const double z_limits_upper = 10.0,
                    const double goal_reach_radius = 1.0,
                    const int64_t max_shortcut_index_distance = 100,
                    const uint32_t max_smoothing_iterations = 200,
                    const uint32_t max_failed_smoothing_iterations = 500);

            std::vector<RRTConfig, Allocator> rrtPlan (
                    const RRTConfig& start,
                    const RRTConfig& goal,
                    const std::chrono::duration<double>& time_limit);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Visualization and other debugging tools
            ///////////////////////////////////////////////////////////////////////////////////////

            void visualize(const std::vector<RRTConfig, Allocator>& path);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper function for original rrt planning
            ///////////////////////////////////////////////////////////////////////////////////////
        private:
            // returned distance is the Euclidian distance of two grippers pos
            double distance(
                    const RRTConfig& a_node,
                    const RRTConfig& b_node) const;

            int64_t nearestNeighbour(
                    const std::vector<simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, Allocator>>& nodes,
                    const RRTConfig& config) const;

            std::pair<Eigen::Vector3d, Eigen::Vector3d> posPairSampling();

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
                    const RRTConfig& random_target);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper function for shortcut smoothing
            ///////////////////////////////////////////////////////////////////////////////////////

            std::pair<std::vector<RRTConfig, Allocator>, std::map<std::string, double>> rrtShortcutSmooth(
                    const std::vector<RRTConfig, Allocator>& path);

    };
}


#endif // ifndef RRT_HELPER_H
