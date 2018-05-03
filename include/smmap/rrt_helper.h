#ifndef RRT_HELPER_H
#define RRT_HELPER_H

#include <random>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <smmap_utilities/visualization_tools.h>

#include "smmap/rubber_band.hpp"
#include "smmap/prm_helper.h"
#include "smmap/robot_interface.hpp"

namespace smmap
{
    class RRTNode;
    typedef std::allocator<RRTNode> RRTAllocator;
    typedef simple_rrt_planner::SimpleRRTPlannerState<RRTNode, RRTAllocator> ExternalRRTState;
}

// Needed for the goal extension blacklist
namespace std
{
    template<>
    struct hash<smmap::RRTNode>
    {
        std::size_t operator()(const smmap::RRTNode& rrt_config) const;
    };
}

namespace smmap
{
    typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> RRTGrippersRepresentation;
    typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> RRTRobotRepresentation;

    class RRTNode
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            RRTNode();

            RRTNode(
                    const RRTGrippersRepresentation& grippers_position,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band,
                    const size_t unique_forward_propogation_idx);

            RRTNode(
                    const RRTGrippersRepresentation& grippers_position,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band,
                    const size_t unique_forward_propogation_idx,
                    const int64_t parent_index);

            RRTNode(
                    const RRTGrippersRepresentation& grippers_position,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band,
                    const size_t unique_forward_propogation_idx,
                    const int64_t parent_index,
                    const std::vector<int64_t>& child_indices);

            bool isInitialized() const;

            const RRTGrippersRepresentation& getGrippers() const;
            const RRTRobotRepresentation& getRobotConfiguration() const;
            const RubberBand::Ptr& getBand() const;
            size_t getUniqueForwardPropogationIndex() const;

            int64_t getParentIndex() const;
            void setParentIndex(const int64_t parent_index);

            const std::vector<int64_t>& getChildIndices() const;
            void clearChildIndicies();
            void addChildIndex(const int64_t child_index);
            void removeChildIndex(const int64_t child_index);

            double distance(const RRTNode& other) const;
            static double distance(const RRTNode& c1, const RRTNode& c2);
            static double distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2);
            static double distance(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2);

            static double grippersPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index);
            static double robotPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index);



            std::string print() const;

            bool operator==(const RRTNode& other) const;

            uint64_t serialize(std::vector<uint8_t>& buffer) const;
//            std::pair<RRTConfig, uint64_t> deserialize(const std::vector<uint8_t>& buffer, const uint64_t current);

            static uint64_t Serialize(const RRTNode& config, std::vector<uint8_t>& buffer);
            static std::pair<RRTNode, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current, const RubberBand& starting_band);

        private:

            RRTGrippersRepresentation grippers_position_;
            RRTRobotRepresentation robot_configuration_;
            RubberBand::Ptr band_;

            // Book keeping
            size_t unique_forward_propogation_idx_;
            int64_t parent_index_;
            std::vector<int64_t> child_indices_;
            bool initialized_;
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
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const RobotInterface::Ptr robot,
                    const sdf_tools::SignedDistanceField& environment_sdf,
                    const smmap_utilities::Visualizer::Ptr vis,
                    const std::shared_ptr<std::mt19937_64>& generator,
                    const PRMHelper::Ptr& prm_helper,
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
                    const bool visualization_enabled);

            std::vector<RRTNode, RRTAllocator> rrtPlan(
                    const RRTNode& start,
                    const RRTGrippersRepresentation& grippers_goal,
                    const std::chrono::duration<double>& time_limit);

            void addBandToBlacklist(const EigenHelpers::VectorVector3d& band);
            void clearBlacklist();

            bool isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band) const;
            bool isBandFirstOrderVisibileToBlacklist(const RubberBand& test_band);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Visualization and other debugging tools
            ///////////////////////////////////////////////////////////////////////////////////////

            void visualizePath(const std::vector<RRTNode, RRTAllocator>& path) const;

            void visualizeBlacklist() const;

            void storePath(const std::vector<RRTNode, RRTAllocator>& path) const;

            std::vector<RRTNode, RRTAllocator> loadStoredPath() const;

            bool useStoredPath() const;

        private:
            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper functions and data for internal rrt planning algorithm
            ///////////////////////////////////////////////////////////////////////////////////////

            std::unordered_set<RRTNode> goal_expansion_nn_blacklist_;

            int64_t nearestNeighbour(
                    const std::vector<ExternalRRTState>& nodes,
                    const RRTNode& config);

            // Used for timing purposes
            // https://stackoverflow.com/questions/37786547/enforcing-statement-order-in-c
            int64_t nearestNeighbour_internal(
                    const std::vector<ExternalRRTState>& nodes,
                    const RRTNode& config);

            RRTNode configSampling();
            // Used for timing purposes
            // https://stackoverflow.com/questions/37786547/enforcing-statement-order-in-c
            RRTNode prmBasedSampling_internal();
            RRTGrippersRepresentation posPairSampling_internal();
            RRTRobotRepresentation robotConfigPairSampling_internal();

            bool goalReached(const RRTNode& node);

            const std::pair<bool, RRTRobotRepresentation> projectToValidConfig(
                    const RRTRobotRepresentation& configuration,
                    const AllGrippersSinglePose& poses) const;

            /* const std::function<std::vector<std::pair<T, int64_t>>(const T&, const T&)>& forward_propagation_fn,
             * forward_propagation_fn - given the nearest neighbor and a new target state, returns the states that would grow the tree towards the target
             * SHOULD : collision checking, constraint violation checking
             Determine the parent index of the new state
             This process deserves some explanation
             The "current relative parent index" is the index of the parent, relative to the list of propagated nodes.
             A negative value means the nearest neighbor in the tree, zero means the first propagated node, and so on.
             NOTE - the relative parent index *must* be lower than the index in the list of prograted nodes
             * i.e. the first node must have a negative value, and so on.
             */
            std::vector<std::pair<RRTNode, int64_t>> forwardPropogationFunction(
                    const RRTNode& nearest_neighbor,
                    const RRTNode& random_target,
                    const bool visualization_enabled_locally);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper function for shortcut smoothing
            ///////////////////////////////////////////////////////////////////////////////////////

            std::pair<bool, std::vector<RRTNode, RRTAllocator>> forwardSimulateGrippersPath(
                    const std::vector<RRTNode, RRTAllocator>& path,
                    const size_t start_index,
                    RubberBand rubber_band);

            std::vector<RRTNode, RRTAllocator> rrtShortcutSmooth(
                    std::vector<RRTNode, RRTAllocator> path,
                    const bool visualization_enabled_locally);


        private:
            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            const RobotInterface::Ptr robot_;
            // TODO: replace this with a shared pointer
            const sdf_tools::SignedDistanceField environment_sdf_;

            const smmap_utilities::Visualizer::Ptr vis_;
            const bool visualization_enabled_globally_;
            const std_msgs::ColorRGBA band_safe_color_;
            const std_msgs::ColorRGBA band_overstretched_color_;
            const std_msgs::ColorRGBA gripper_a_tree_color_;
            const std_msgs::ColorRGBA gripper_b_tree_color_;

            const Eigen::Isometry3d task_aligned_frame_transform_;
            const Eigen::Isometry3d task_aligned_frame_inverse_transform_;
            const Eigen::Vector3d task_aligned_lower_limits_;
            const Eigen::Vector3d task_aligned_upper_limits_;
            const double max_gripper_step_size_;
            const double max_robot_dof_step_size_;
            const double min_robot_dof_step_size_;
            const double max_gripper_rotation_;
            const double goal_bias_;
            const double goal_reach_radius_;
            const double homotopy_distance_penalty_;
            const double gripper_min_distance_to_obstacles_;

            const int64_t max_shortcut_index_distance_;
            const uint32_t max_smoothing_iterations_;
            const uint32_t max_failed_smoothing_iterations_;

            const std::shared_ptr<std::mt19937_64> generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;
            std::uniform_int_distribution<int> uniform_shortcut_smoothing_int_distribution_;
            std::uniform_int_distribution<size_t> arm_a_goal_config_int_distribution_;
            std::uniform_int_distribution<size_t> arm_b_goal_config_int_distribution_;
            PRMHelper::Ptr prm_helper_;


            // Set/updated on each call of "rrtPlan"
            bool planning_for_whole_robot_;
            std::unique_ptr<RubberBand> starting_band_;
            AllGrippersSinglePose starting_grippers_poses_;
            RRTRobotRepresentation starting_robot_configuration_;

            RRTGrippersRepresentation grippers_goal_position_;
            double max_grippers_distance_;
            std::vector<EigenHelpers::VectorVector3d> blacklisted_goal_rubber_bands_;

            std::pair<ssize_t, ssize_t> arm_dof_;
            std::vector<Eigen::VectorXd> arm_a_goal_configurations_;
            std::vector<Eigen::VectorXd> arm_b_goal_configurations_;
            RRTRobotRepresentation robot_joint_limits_upper_;
            RRTRobotRepresentation robot_joint_limits_lower_;

            // Planning and Smoothing statistics
            std::map<std::string, double> statistics_;
            double total_sampling_time_;
            double total_nearest_neighbour_time_;
            double total_crrt_projection_time_;
            double total_band_forward_propogation_time_;
            double total_first_order_vis_propogation_time_;
            double total_everything_included_forward_propogation_time_;

            // Used to augment the state, duplicates work done by external code, but oh well
            size_t next_unique_forward_propogation_idx_;
    };
}

#endif // ifndef RRT_HELPER_H
