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
    class RRTConfig;
}

namespace std
{
    template<>
    struct hash<smmap::RRTConfig>
    {
        std::size_t operator()(const smmap::RRTConfig& rrt_config) const;
    };
}

namespace smmap
{
    typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> RRTGrippersRepresentation;

    class RRTConfig
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            RRTConfig(
                    const RRTGrippersRepresentation& grippers_position,
                    const VirtualRubberBand& band,
                    const bool is_visible_to_blacklist);

            const RRTGrippersRepresentation& getGrippers() const;
            const VirtualRubberBand& getBand() const;
            bool isVisibleToBlacklist() const;

            // returned distance is the Euclidian distance of two grippers pos
            double distance(const RRTConfig& other) const;
            static double Distance(const RRTConfig& c1, const RRTConfig& c2);
            static double Distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2);

            bool operator==(const RRTConfig& other) const;

        private:

            RRTGrippersRepresentation grippers_position_;
            VirtualRubberBand band_;
            bool is_visible_to_blacklist_;
    };
    typedef std::allocator<RRTConfig> RRTAllocator;
    typedef simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, RRTAllocator> ExternalRRTState;

    class RRTHelper
    {
        public:
            RRTHelper(
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
                    const double homotopy_distance_penalty = 1e3,
                    const int64_t max_shortcut_index_distance = 100,
                    const uint32_t max_smoothing_iterations = 200,
                    const uint32_t max_failed_smoothing_iterations = 500);

            std::vector<RRTConfig, RRTAllocator> rrtPlan(
                    const RRTConfig& start,
                    const RRTGrippersRepresentation& grippers_goal,
                    const std::chrono::duration<double>& time_limit);

            void addBandToBlacklist(const EigenHelpers::VectorVector3d& band);

            bool isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band) const;
            bool isBandFirstOrderVisibileToBlacklist(const VirtualRubberBand& test_band) const;

            ///////////////////////////////////////////////////////////////////////////////////////
            // Visualization and other debugging tools
            ///////////////////////////////////////////////////////////////////////////////////////

            void visualize(const std::vector<RRTConfig, RRTAllocator>& path) const;

            void visualizeBlacklist() const;

            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper function for original rrt planning
            ///////////////////////////////////////////////////////////////////////////////////////
        private:
            std::unordered_set<RRTConfig> goal_expansion_nn_blacklist_;

            int64_t nearestNeighbour(
                    const std::vector<ExternalRRTState>& nodes,
                    const RRTConfig& config);

            RRTGrippersRepresentation posPairSampling();

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
                    const RRTConfig& random_target) const;

            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper function for shortcut smoothing
            ///////////////////////////////////////////////////////////////////////////////////////

            std::pair<bool, std::vector<RRTConfig, RRTAllocator>> forwardSimulateGrippersPath(
                    VirtualRubberBand rubber_band,
                    const std::vector<RRTConfig, RRTAllocator>& path,
                    const size_t start_index,
                    const size_t end_index) const;

            std::pair<std::vector<RRTConfig, RRTAllocator>, std::map<std::string, double>> rrtShortcutSmooth(
                    const std::vector<RRTConfig, RRTAllocator>& path) const;


        private:
            const std::pair<double, double> x_limits_;
            const std::pair<double, double> y_limits_;
            const std::pair<double, double> z_limits_;
            const double step_size_;
            const double goal_reach_radius_;
            const double homotopy_distance_penalty_;

            const int64_t max_shortcut_index_distance_;
            const uint32_t max_smoothing_iterations_;
            const uint32_t max_failed_smoothing_iterations_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;

            const sdf_tools::SignedDistanceField& environment_sdf_;
            const Visualizer& vis_;
            std::mt19937_64& generator_;

            const std_msgs::ColorRGBA rubber_band_safe_color_;
            const std_msgs::ColorRGBA rubber_band_overstretched_color_;

            RRTGrippersRepresentation grippers_goal_position_;

            std::vector<EigenHelpers::VectorVector3d> blacklisted_goal_rubber_bands_;

    };
}

#endif // ifndef RRT_HELPER_H
