#ifndef RRT_HELPER_H
#define RRT_HELPER_H

#include <random>
#include <arc_utilities/eigen_helpers.hpp>
#include <smmap_utilities/visualization_tools.h>

#include <flann/flann.hpp>

#include "smmap/rubber_band.hpp"
#include "smmap/robot_interface.hpp"

namespace flann
{
    /**
     * Squared Euclidean distance functor, optimized version
     */
    template<class T>
    struct L2_weighted
    {
        typedef bool is_kdtree_distance;

        typedef T ElementType;
        typedef T ResultType;
        typedef Eigen::Matrix<ElementType, Eigen::Dynamic, 1> VectorX;

        L2_weighted(const Eigen::VectorXd& dof_weights)
        {
            dof_weights_.resizeLike(dof_weights);
            dof_weights2_.resizeLike(dof_weights);
            for (int i = 0; i < dof_weights.rows(); ++i)
            {
                dof_weights_(i) = (ElementType)dof_weights(i);
                dof_weights2_(i) = (ElementType)(dof_weights(i) * dof_weights(i));
            }
        }

        /**
         *  Compute the squared Euclidean distance between two vectors.
         *
         *	The computation of squared root at the end is omitted for
         *	efficiency.
         */
        template <typename Iterator1, typename Iterator2>
        ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
        {
            (void)worst_dist;
            const Eigen::Map<const VectorX> a_vec(a, size);
            const Eigen::Map<const VectorX> b_vec(b, size);
            auto delta = (a_vec - b_vec).cwiseProduct(dof_weights_);
            return delta.squaredNorm();
        }

        /**
         *	Partial euclidean distance, using just one dimension. This is used by the
         *	kd-tree when computing partial distances while traversing the tree.
         *
         *	Squared root is omitted for efficiency.
         */
        template <typename U, typename V>
        inline ResultType accum_dist(const U& a, const V& b, int ind) const
        {
            return (a-b) * (a-b) * dof_weights2_(ind);
        }

    private:
        VectorX dof_weights_;
        VectorX dof_weights2_;
    };
}

namespace smmap
{
    class RRTNode;
    typedef Eigen::aligned_allocator<RRTNode> RRTAllocator;
    typedef std::pair<Eigen::Isometry3d, Eigen::Isometry3d> RRTGrippersRepresentation;
    typedef Eigen::VectorXd RRTRobotRepresentation;
    typedef flann::KDTreeSingleIndex<flann::L2_weighted<float>> NNIndexType;

    class RRTNode
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            RRTNode();

            RRTNode(const RRTGrippersRepresentation& grippers_poses,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band);

            RRTNode(const RRTGrippersRepresentation& grippers_poses,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band,
                    const double cost_to_come,
                    const int64_t parent_index);

            RRTNode(const RRTGrippersRepresentation& grippers_poses,
                    const RRTRobotRepresentation& robot_configuration,
                    const RubberBand::Ptr& band,
                    const double cost_to_come,
                    const int64_t parent_index,
                    const std::vector<int64_t>& child_indices,
                    const std::vector<int64_t>& other_tree_target_indices_blacklist);

            bool isInitialized() const;

            const RRTGrippersRepresentation& grippers() const;
            const RRTRobotRepresentation& robotConfiguration() const;
            const RubberBand::Ptr& band() const;
            double costToCome() const;

            int64_t getParentIndex() const;
            void setParentIndex(const int64_t parent_index);

            const std::vector<int64_t>& getChildIndices() const;
            void clearChildIndicies();
            void addChildIndex(const int64_t child_index);
            void removeChildIndex(const int64_t child_index);

            const std::vector<int64_t>& getOtherTreeBlacklistIndices() const;
            void clearOtherTreeBlacklistIndices();
            void addOtherTreeBlacklistIndex(const int64_t blacklist_index);
            void removeOtherTreeBlacklistIndex(const int64_t blacklist_index);

            std::string print() const;

            bool operator==(const RRTNode& other) const;

            uint64_t serialize(std::vector<uint8_t>& buffer) const;
            static uint64_t Serialize(const RRTNode& config, std::vector<uint8_t>& buffer);
            static std::pair<RRTNode, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current, const RubberBand& starting_band);

        private:

            RRTGrippersRepresentation grippers_poses_;
            RRTRobotRepresentation robot_configuration_;
            RubberBand::Ptr band_;
            double cost_to_come_;

            // Book keeping
            int64_t parent_index_;
            std::vector<int64_t> child_indices_;
            std::vector<int64_t> other_tree_target_indices_blacklist_;
            bool initialized_;
    };

    class RRTDistance
    {
        public:
            static const RRTRobotRepresentation& GetJointWeights();
            static void SetJointWeights(const RRTRobotRepresentation joint_weights);

            static double DistanceSquared(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2);
            static double Distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2);
            static double DistanceSquared(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2);
            static double Distance(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2);

            static double GrippersPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index);
            static double RobotPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index);

        private:
            static RRTRobotRepresentation joint_weights_;
    };

    class RRTHelper
    {
        public:
            static constexpr double NN_BLACKLIST_DISTANCE = std::numeric_limits<double>::max() - 1e10;

            // Topic names used for publishing visualization data
            static constexpr char RRT_BLACKLISTED_GOAL_BANDS_NS[]   = "rrt_blacklisted_goal_bands";
            static constexpr char RRT_GOAL_TESTING_NS[]               = "rrt_goal_testing";

            static constexpr char RRT_FORWARD_TREE_GRIPPER_A_NS[]   = "rrt_forward_tree_gripper_a";
            static constexpr char RRT_FORWARD_TREE_GRIPPER_B_NS[]   = "rrt_forward_tree_gripper_b";
            static constexpr char RRT_BACKWARD_TREE_GRIPPER_A_NS[]  = "rrt_backward_tree_gripper_a";
            static constexpr char RRT_BACKWARD_TREE_GRIPPER_B_NS[]  = "rrt_backward_tree_gripper_b";
            static constexpr char RRT_TREE_BAND_NS[]                = "rrt_tree_band";

            static constexpr char RRT_SAMPLE_NS[]                   = "rrt_sample";
            static constexpr char RRT_FORWARD_PROP_START_NS[]       = "rrt_forward_prop_start";

            static constexpr char RRT_SOLUTION_GRIPPER_A_NS[]       = "rrt_solution_gripper_a";
            static constexpr char RRT_SOLUTION_GRIPPER_B_NS[]       = "rrt_solution_gripper_b";
            static constexpr char RRT_SOLUTION_RUBBER_BAND_NS[]     = "rrt_solution_rubber_band";

            RRTHelper(
                    // Robot/environment related parameters
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const RobotInterface::Ptr robot,
                    const bool planning_for_whole_robot,
                    const sdf_tools::SignedDistanceField::ConstPtr environment_sdf,
                    const XYZGrid& work_space_grid,
                    const std::shared_ptr<std::mt19937_64>& generator,
                    // Planning algorithm parameters
                    const bool using_cbirrt_style_projection,
                    const size_t forward_tree_extend_iterations,
                    const size_t backward_tree_extend_iterations,
                    const size_t kd_tree_grow_threshold,
                    const bool use_brute_force_nn,
                    const double goal_bias,
                    const double best_near_radius,
                    // Smoothing parameters
                    const int64_t max_shortcut_index_distance,
                    const uint32_t max_smoothing_iterations,
                    const uint32_t max_failed_smoothing_iterations,
                    // Task defined parameters
                    const Eigen::Isometry3d& task_aligned_frame,
                    const Eigen::Vector3d& task_aligned_lower_limits,
                    const Eigen::Vector3d& task_aligned_upper_limits,
                    const double max_gripper_step_size,
                    const double max_robot_dof_step_size,
                    const double min_robot_dof_step_size,
                    const double max_gripper_rotation,
                    const double goal_reach_radius,
                    const double gripper_min_distance_to_obstacles,
                    const double band_distance2_scaling_factor,
                    const size_t band_max_points,
                    // Visualization
                    const smmap_utilities::Visualizer::Ptr vis,
                    const bool visualization_enabled);

            std::vector<RRTNode, RRTAllocator> plan(
                    const RRTNode& start,
                    const RRTGrippersRepresentation& grippers_goal_poses,
                    const std::chrono::duration<double>& time_limit);


            static std::vector<Eigen::VectorXd> ConvertRRTPathToRobotPath(
                    const std::vector<RRTNode, RRTAllocator>& path);

            static bool CheckTreeLinkage(
                    const std::vector<RRTNode, RRTAllocator>& tree);

            static std::vector<RRTNode, RRTAllocator> ExtractSolutionPath(
                    const std::vector<RRTNode, RRTAllocator>& tree,
                    const int64_t goal_node_idx);


            void addBandToBlacklist(const EigenHelpers::VectorVector3d& band);
            void clearBlacklist();

            bool isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band_input) const;
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

            void visualizeBothTrees() const;

            void deleteTreeVisualizations() const;

            void visualizePath(const std::vector<RRTNode, RRTAllocator>& path) const;

            void visualizeBlacklist() const;

            void storePath(const std::vector<RRTNode, RRTAllocator>& path, std::string file_path = "") const;

            std::vector<RRTNode, RRTAllocator> loadStoredPath(std::string file_path = "") const;

            bool useStoredPath() const;

        private:
            ///////////////////////////////////////////////////////////////////////////////////////
            // Helper functions and data for internal rrt planning algorithm
            ///////////////////////////////////////////////////////////////////////////////////////

            size_t rebuildNNIndex(
                    NNIndexType& index,
                    std::vector<float>& nn_raw_data,
                    const std::vector<RRTNode, RRTAllocator>& tree,
                    const size_t new_data_start_idx);

            // Used for timing purposes
            // https://stackoverflow.com/questions/37786547/enforcing-statement-order-in-c
            int64_t nearestNeighbour(
                    const bool use_forward_tree,
                    const RRTNode& config);

            std::pair<int64_t, double> nearestNeighbourRobotSpace(
                    const bool use_forward_tree,
                    const RRTNode& config);

            int64_t nearestBestNeighbourFullSpace(
                    const RRTNode& config);

            RRTNode configSampling(const bool sample_band);
            // Used for timing purposes
            // https://stackoverflow.com/questions/37786547/enforcing-statement-order-in-c
            RRTGrippersRepresentation posPairSampling_internal();
            RRTRobotRepresentation robotConfigPairSampling_internal();
            EigenHelpers::VectorVector3d bandSampling_internal();

            bool goalReached(const RRTNode& node);

            const std::pair<bool, RRTRobotRepresentation> projectToValidConfig(
                    const RRTRobotRepresentation& configuration,
                    const AllGrippersSinglePose& poses,
                    const bool project_to_rotation_bound,
                    const bool project_to_translation_bound) const;

            size_t forwardPropogationFunction(std::vector<RRTNode, RRTAllocator>& tree_to_extend,
                    const int64_t& nearest_neighbor_idx,
                    const RRTNode& target,
                    const bool extend_band,
                    const size_t max_projected_new_states,
                    const bool visualization_enabled_locally);

            size_t connectForwardTree(const RRTNode& target, const bool is_random);
            size_t connectBackwardTree(const RRTNode& target, const bool is_random);
            size_t connectForwardTreeToBackwardTreeBispace(const int64_t last_node_idx_in_forward_tree_branch);
            size_t connectBackwardTreeToForwardTreeBidirectional();

            void followBackwardTree(
                    const size_t forward_tree_node_idx,
                    const size_t backward_tree_node_idx);

            void planningMainLoopBispace();
            void planningMainLoopBidirectional();

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
            const bool planning_for_whole_robot_;
            const sdf_tools::SignedDistanceField::ConstPtr environment_sdf_;
            const XYZGrid work_space_grid_;
            const std::shared_ptr<std::mt19937_64> generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;


        private:
            const Eigen::Isometry3d task_aligned_frame_transform_;
            const Eigen::Isometry3d task_aligned_frame_inverse_transform_;
            const Eigen::Vector3d task_aligned_lower_limits_;
            const Eigen::Vector3d task_aligned_upper_limits_;

            const RRTRobotRepresentation robot_joint_lower_limits_;
            const RRTRobotRepresentation robot_joint_upper_limits_;
            const RRTRobotRepresentation robot_joint_weights_;
            const ssize_t total_dof_;

            const double max_gripper_step_size_;
            const double max_robot_dof_step_size_;
            const double min_robot_dof_step_size_;
            const double max_gripper_rotation_;
            const double goal_bias_;
            const double goal_reach_radius_;
            const double gripper_min_distance_to_obstacles_;

            // Used for double layer NN check
            const double band_distance2_scaling_factor_;
            const size_t band_max_points_;
            const double band_max_dist2_;

            const bool using_cbirrt_style_projection_;
            const size_t forward_tree_extend_iterations_;
            const size_t backward_tree_extend_iterations_;
            const bool use_brute_force_nn_;
            const size_t kd_tree_grow_threshold_;
            const double best_near_radius2_;

            const int64_t max_shortcut_index_distance_;
            const uint32_t max_smoothing_iterations_;
            const uint32_t max_failed_smoothing_iterations_;
            std::uniform_int_distribution<int> uniform_shortcut_smoothing_int_distribution_;


            // Set/updated on each call of "rrtPlan"
            RubberBand::Ptr starting_band_;
            RRTGrippersRepresentation starting_grippers_poses_;
            RRTRobotRepresentation starting_robot_configuration_;

            std::vector<EigenHelpers::VectorVector3d> blacklisted_goal_rubber_bands_;
            double max_grippers_distance_;
            std::chrono::duration<double> time_limit_;
            RRTGrippersRepresentation grippers_goal_poses_;

            std::vector<RRTNode, RRTAllocator> forward_tree_;
            // Note that the band portion of the backward tree is invalid
            std::vector<RRTNode, RRTAllocator> backward_tree_;

            std::vector<float> forward_nn_raw_data_;
            std::vector<float> backward_nn_raw_data_;
            std::shared_ptr<NNIndexType> forward_nn_index_;
            std::shared_ptr<NNIndexType> backward_nn_index_;
            size_t forward_next_idx_to_add_to_nn_dataset_;
            size_t backward_next_idx_to_add_to_nn_dataset_;


            // TODO: address this hack - used to help track what node to connect to the backward tree from
            int64_t forward_tree_nearest_neighbour_idx_;


            // Planning and Smoothing statistics
            std::map<std::string, double> planning_statistics_;
            std::map<std::string, double> smoothing_statistics_;
            double total_sampling_time_;
            double total_nearest_neighbour_index_building_time_;
            double total_nearest_neighbour_index_searching_time_;
            double total_nearest_neighbour_linear_searching_time_;
            double total_nearest_neighbour_radius_searching_time_;
            double total_nearest_neighbour_best_searching_time_;
            double total_nearest_neighbour_time_;
            double total_forward_kinematics_time_;
            double total_projection_time_;
            double total_collision_check_time_;
            double total_band_forward_propogation_time_;
            double total_first_order_vis_propogation_time_;
            double total_everything_included_forward_propogation_time_;

            size_t forward_random_samples_useful_;
            size_t forward_random_samples_useless_;
            size_t backward_random_samples_useful_;
            size_t backward_random_samples_useless_;
            size_t forward_connection_attempts_useful_;
            size_t forward_connection_attempts_useless_;
            size_t forward_connections_made_;
            size_t backward_connection_attempts_useful_;
            size_t backward_connection_attempts_useless_;
            size_t backward_connections_made_;

            bool path_found_;
            int64_t goal_idx_in_forward_tree_;
            std::chrono::time_point<std::chrono::steady_clock> start_time_;


            // Visualization
            const smmap_utilities::Visualizer::Ptr vis_;
            const bool visualization_enabled_globally_;
            const std_msgs::ColorRGBA gripper_a_forward_tree_color_;
            const std_msgs::ColorRGBA gripper_b_forward_tree_color_;
            const std_msgs::ColorRGBA gripper_a_backward_tree_color_;
            const std_msgs::ColorRGBA gripper_b_backward_tree_color_;
            const std_msgs::ColorRGBA band_tree_color_;
            // Used in the forward propagation function
            int32_t tree_marker_id_;
            size_t forward_tree_next_visualized_node_;
            size_t backward_tree_next_visualized_node_;
    };
}

#endif // ifndef RRT_HELPER_H
