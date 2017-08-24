#ifndef RRT_HELPER_H
#define RRT_HELPER_H

//#include <arc_utilities/arc_helpers.hpp>

#include <random>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>

#include "smmap/visualization_tools.h"
#include "smmap/virtual_rubber_band.h"
#include "smmap/prm_helper.h"

namespace smmap
{
    class RRTConfig;
    typedef std::allocator<RRTConfig> RRTAllocator;
    typedef simple_rrt_planner::SimpleRRTPlannerState<RRTConfig, RRTAllocator> ExternalRRTState;
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
            static double distance(const RRTConfig& c1, const RRTConfig& c2);
            static double distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2);
            static double pathDistance(const std::vector<RRTConfig, RRTAllocator>& path, const size_t start_index, const size_t end_index);

            bool operator==(const RRTConfig& other) const;

        private:

            RRTGrippersRepresentation grippers_position_;
            VirtualRubberBand band_;
            bool is_visible_to_blacklist_;
    };

    class RRTHelper
    {
        public:
            static constexpr double NN_BLACKLIST_DISTANCE = (std::numeric_limits<double>::max() - 1e10);

            // Topic names used for publishing visualization data
            static constexpr auto RRT_BLACKLISTED_GOAL_BANDS_NS  = "rrt_blacklisted_goal_bands";
            static constexpr auto RRT_GOAL_TESTING_NS            = "rrt_goal_testing";

            static constexpr auto RRT_TREE_GRIPPER_A_NS          = "rrt_tree_gripper_a";
            static constexpr auto RRT_TREE_GRIPPER_B_NS          = "rrt_tree_gripper_b";

            static constexpr auto RRT_SAMPLE_NS                  = "rrt_sample";
            static constexpr auto RRT_FORWARD_PROP_START_NS      = "rrt_forward_prop_start";
            static constexpr auto RRT_FORWARD_PROP_STEPS_NS      = "rrt_forward_prop_steps";

            static constexpr auto RRT_SOLUTION_GRIPPER_A_NS      = "rrt_solution_gripper_a";
            static constexpr auto RRT_SOLUTION_GRIPPER_B_NS      = "rrt_solution_gripper_b";
            static constexpr auto RRT_SOLUTION_RUBBER_BAND_NS    = "rrt_solution_rubber_band";

            static constexpr auto RRT_SHORTCUT_FIRST_GRIPPER_NS  = "rrt_shortcut_first_gripper";
            static constexpr auto RRT_SHORTCUT_SECOND_GRIPPER_NS = "rrt_shortcut_second_gripper";

            RRTHelper(
                    const sdf_tools::SignedDistanceField& environment_sdf,
                    const Visualizer& vis,
                    std::mt19937_64& generator,
                    const std::shared_ptr<PRMHelper>& prm_helper,
                    const Eigen::Vector3d planning_world_lower_limits,
                    const Eigen::Vector3d planning_world_upper_limits,
                    const double max_step_size,
                    const double goal_bias,
                    const double goal_reach_radius,
                    const double gripper_min_distance_to_obstacles,
                    const double homotopy_distance_penalty,
                    const int64_t max_shortcut_index_distance,
                    const uint32_t max_smoothing_iterations,
                    const uint32_t max_failed_smoothing_iterations,
                    const bool visualization_enabled);

            std::vector<RRTConfig, RRTAllocator> rrtPlan(
                    const RRTConfig& start,
                    const RRTGrippersRepresentation& grippers_goal,
                    const std::chrono::duration<double>& time_limit);

            void addBandToBlacklist(const EigenHelpers::VectorVector3d& band);

            bool isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band) const;
            bool isBandFirstOrderVisibileToBlacklist(const VirtualRubberBand& test_band);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Visualization and other debugging tools
            ///////////////////////////////////////////////////////////////////////////////////////

            void visualizePath(const std::vector<RRTConfig, RRTAllocator>& path) const;

            void visualizeBlacklist() const;

        private:
            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper functions and data for internal rrt planning algorithm
            ///////////////////////////////////////////////////////////////////////////////////////

            std::unordered_set<RRTConfig> goal_expansion_nn_blacklist_;

            int64_t nearestNeighbour(
                    const std::vector<ExternalRRTState>& nodes,
                    const RRTConfig& config);

            RRTGrippersRepresentation posPairSampling();

            RRTConfig configSampling();

            bool goalReached(const RRTConfig& node);

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
                    const RRTConfig& random_target,
                    const bool calculate_first_order_vis,
                    const bool visualization_enabled_locally);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper function for shortcut smoothing
            ///////////////////////////////////////////////////////////////////////////////////////

            std::pair<bool, std::vector<RRTConfig, RRTAllocator>> forwardSimulateGrippersPath(
                    const std::vector<RRTConfig, RRTAllocator>& path,
                    const size_t start_index,
                    VirtualRubberBand rubber_band);

            std::vector<RRTConfig, RRTAllocator> rrtShortcutSmooth(
                    std::vector<RRTConfig, RRTAllocator> path,
                    const bool visualization_enabled_locally);


        private:
            const Eigen::Vector3d planning_world_lower_limits_;
            const Eigen::Vector3d planning_world_upper_limits_;
            const double max_step_size_;
            const double goal_bias_;
            const double goal_reach_radius_;
            const double homotopy_distance_penalty_;

            const int64_t max_shortcut_index_distance_;
            const uint32_t max_smoothing_iterations_;
            const uint32_t max_failed_smoothing_iterations_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;
            std::uniform_int_distribution<int> uniform_shortcut_smoothing_int_distribution_;
            std::shared_ptr<PRMHelper> prm_helper_;

            std::mt19937_64& generator_;
            const sdf_tools::SignedDistanceField& environment_sdf_;
            const Visualizer& vis_;
            const bool visualization_enabled_globally_;
            const std_msgs::ColorRGBA band_safe_color_;
            const std_msgs::ColorRGBA band_overstretched_color_;

            std::unique_ptr<VirtualRubberBand> starting_band_;
            RRTGrippersRepresentation grippers_goal_position_;
            double max_grippers_distance_;
            std::vector<EigenHelpers::VectorVector3d> blacklisted_goal_rubber_bands_;

            const double gripper_min_distance_to_obstacles_;

            // Planning and Smoothing statistics
            std::map<std::string, double> statistics_;
            double total_nearest_neighbour_time_;
            double total_everything_included_forward_propogation_time_;
            double total_band_forward_propogation_time_;
            double total_first_order_vis_propogation_time_;
    };
}

#endif // ifndef RRT_HELPER_H
