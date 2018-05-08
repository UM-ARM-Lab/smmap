#ifndef RRT_HELPER_H
#define RRT_HELPER_H

#include <random>
#include <arc_utilities/eigen_helpers.hpp>
#include <smmap_utilities/visualization_tools.h>

#include "smmap/rubber_band.hpp"
#include "smmap/prm_helper.h"
#include "smmap/robot_interface.hpp"

namespace smmap
{
    class RRTNode;
    typedef Eigen::aligned_allocator<RRTNode> RRTAllocator;
    typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> RRTGrippersRepresentation;
    typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> RRTRobotRepresentation;

    class RRTNode
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            RRTNode();

            RRTNode(const RRTGrippersRepresentation& grippers_position,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band);

            RRTNode(const RRTGrippersRepresentation& grippers_position,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band,
                    const int64_t parent_index);

            RRTNode(const RRTGrippersRepresentation& grippers_position,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band,
                    const int64_t parent_index,
                    const std::vector<int64_t>& child_indices);

            bool isInitialized() const;

            const RRTGrippersRepresentation& getGrippers() const;
            const RRTRobotRepresentation& getRobotConfiguration() const;
            const RubberBand::Ptr& getBand() const;

            int64_t getParentIndex() const;
            void setParentIndex(const int64_t parent_index);

            const std::vector<int64_t>& getChildIndices() const;
            void clearChildIndicies();
            void addChildIndex(const int64_t child_index);
            void removeChildIndex(const int64_t child_index);

            bool isBlacklisted() const;
            void blacklist();

            double distance(const RRTNode& other) const;
            static double distance(const RRTNode& c1, const RRTNode& c2);
            static double distanceSquared(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2);
            static double distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2);
            static double distanceSquared(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2);
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
            int64_t parent_index_;
            std::vector<int64_t> child_indices_;
            bool initialized_;
            bool blacklisted_;
    };

    class RRTHelper
    {
        public:
            static constexpr double NN_BLACKLIST_DISTANCE = std::numeric_limits<double>::max() - 1e10;

            // Topic names used for publishing visualization data
            static constexpr auto RRT_BLACKLISTED_GOAL_BANDS_NS  = "rrt_blacklisted_goal_bands";
            static constexpr auto RRT_GOAL_TESTING_NS            = "rrt_goal_testing";

            static constexpr auto RRT_TREE_GRIPPER_A_NS          = "rrt_tree_gripper_a";
            static constexpr auto RRT_TREE_GRIPPER_B_NS          = "rrt_tree_gripper_b";
            static constexpr auto RRT_TREE_BAND_NS               = "rrt_tree_band";

            static constexpr auto RRT_SAMPLE_NS                  = "rrt_sample";
            static constexpr auto RRT_FORWARD_PROP_START_NS      = "rrt_forward_prop_start";

            static constexpr auto RRT_SOLUTION_GRIPPER_A_NS      = "rrt_solution_gripper_a";
            static constexpr auto RRT_SOLUTION_GRIPPER_B_NS      = "rrt_solution_gripper_b";
            static constexpr auto RRT_SOLUTION_RUBBER_BAND_NS    = "rrt_solution_rubber_band";

            RRTHelper(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const RobotInterface::Ptr robot,
                    const sdf_tools::SignedDistanceField& environment_sdf,
                    const smmap_utilities::Visualizer::Ptr vis,
                    const std::shared_ptr<std::mt19937_64>& generator,
                    const PRMHelper::Ptr& prm_helper,
                    const bool using_cbirrt_style_projection,
                    const size_t forward_tree_extend_iterations,
                    const size_t backward_tree_extend_iterations,
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

            std::vector<RRTNode, RRTAllocator> plan(
                    const RRTNode& start,
                    const RRTGrippersRepresentation& grippers_goal,
                    const std::chrono::duration<double>& time_limit);

            static std::vector<Eigen::VectorXd> ConvertRRTPathToRobotPath(const std::vector<RRTNode, RRTAllocator>& path);
            static bool CheckTreeLinkage(const std::vector<RRTNode, RRTAllocator>& nodes);
            static std::vector<RRTNode, RRTAllocator> ExtractSolutionPath(
                    const std::vector<RRTNode, RRTAllocator>& tree,
                    const int64_t goal_node_idx);

            void addBandToBlacklist(const EigenHelpers::VectorVector3d& band);
            void clearBlacklist();

            bool isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band) const;
            bool isBandFirstOrderVisibileToBlacklist(const RubberBand& test_band);

            ///////////////////////////////////////////////////////////////////////////////////////
            // Visualization and other debugging tools
            ///////////////////////////////////////////////////////////////////////////////////////

            void visualizeTree(
                    const std::vector<RRTNode, RRTAllocator>& tree,
                    const size_t start_idx,
                    const std::string ns_a,
                    const std::string ns_b,
                    const std::string ns_band,
                    const int id_a,
                    const int id_b,
                    const int id_band,
                    const std_msgs::ColorRGBA& color_a,
                    const std_msgs::ColorRGBA& color_b,
                    const std_msgs::ColorRGBA& color_band,
                    const bool draw_band) const;

            void visualizePath(const std::vector<RRTNode, RRTAllocator>& path) const;

            void visualizeBlacklist() const;

            void storePath(const std::vector<RRTNode, RRTAllocator>& path) const;

            std::vector<RRTNode, RRTAllocator> loadStoredPath() const;

            bool useStoredPath() const;

        private:
            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper functions and data for internal rrt planning algorithm
            ///////////////////////////////////////////////////////////////////////////////////////

            int64_t nearestNeighbour(
                    const std::vector<RRTNode, RRTAllocator>& tree,
                    const RRTNode& config);

            // Used for timing purposes
            // https://stackoverflow.com/questions/37786547/enforcing-statement-order-in-c
            int64_t nearestNeighbour_internal(
                    const std::vector<RRTNode, RRTAllocator>& tree,
                    const RRTNode& config) const;

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

            size_t forwardPropogationFunction(std::vector<RRTNode, RRTAllocator>& tree_to_extend,
                    const int64_t& nearest_neighbor_idx,
                    const RRTNode& target,
                    const bool extend_band,
                    const size_t max_projected_new_states,
                    const bool visualization_enabled_locally);

            std::vector<RRTNode, RRTAllocator> planningMainLoop();

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

        public:
//            const std_msgs::ColorRGBA band_safe_color_;
//            const std_msgs::ColorRGBA band_overstretched_color_;
            const std_msgs::ColorRGBA gripper_a_forward_tree_color_;
            const std_msgs::ColorRGBA gripper_b_forward_tree_color_;
            const std_msgs::ColorRGBA gripper_a_backward_tree_color_;
            const std_msgs::ColorRGBA gripper_b_backward_tree_color_;
            const std_msgs::ColorRGBA band_tree_color_;

        private:
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
            const bool using_cbirrt_style_projection_;
            const size_t forward_tree_extend_iterations_;
            const size_t backward_tree_extend_iterations_;


            // Set/updated on each call of "rrtPlan"
            bool planning_for_whole_robot_;
            RubberBand::Ptr starting_band_;
            AllGrippersSinglePose starting_grippers_poses_;
            RRTRobotRepresentation starting_robot_configuration_;

            std::vector<EigenHelpers::VectorVector3d> blacklisted_goal_rubber_bands_;
            RRTGrippersRepresentation grippers_goal_position_;
            double max_grippers_distance_;
            std::chrono::duration<double> time_limit_;

            std::pair<ssize_t, ssize_t> arm_dof_;
            std::vector<Eigen::VectorXd> arm_a_goal_configurations_;
            std::vector<Eigen::VectorXd> arm_b_goal_configurations_;
            RRTRobotRepresentation robot_joint_limits_upper_;
            RRTRobotRepresentation robot_joint_limits_lower_;

            bool path_found_;
            bool forward_iteration_;
            std::vector<RRTNode, RRTAllocator> forward_tree_;
            // Note that the band portion of the backward tree is invalid
            std::vector<RRTNode, RRTAllocator> backward_tree_;

            // Planning and Smoothing statistics
            std::map<std::string, double> statistics_;
            double total_sampling_time_;
            double total_nearest_neighbour_time_;
            double total_projection_time_;
            double total_collision_check_time_;
            double total_band_forward_propogation_time_;
            double total_first_order_vis_propogation_time_;
            double total_everything_included_forward_propogation_time_;
    };
}

#endif // ifndef RRT_HELPER_H
