#ifndef RRT_HELPER_H
#define RRT_HELPER_H

#include <random>
#include <arc_utilities/eigen_helpers.hpp>
#include <smmap_utilities/visualization_tools.h>

#include <flann/flann.hpp>

#include "smmap/rubber_band.hpp"
#include "smmap/prm_helper.h"
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
         *	This is highly optimised, with loop unrolling, as it is one
         *	of the most expensive inner loops.
         *
         *	The computation of squared root at the end is omitted for
         *	efficiency.
         */
        template <typename Iterator1, typename Iterator2>
        ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const
        {
//            (void)size;
//            ResultType result = 0.0;
//            ResultType diff0, diff1, diff2, diff3;
//            Iterator1 start = a;
//            ElementType const * w = &dof_weights_[0]; // pointer to a const ElementType

//            /* Process 4 items with each loop for efficiency. */
//            while (a < start + 12)
//            {
//                diff0 = (a[0] - b[0]) * w[0];
//                diff1 = (a[1] - b[1]) * w[1];
//                diff2 = (a[2] - b[2]) * w[2];
//                diff3 = (a[3] - b[3]) * w[3];
//                result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
//                a += 4;
//                b += 4;
//                w += 4;

//                if ((worst_dist > 0) && (result > worst_dist))
//                {
//                    return result;
//                }
//            }
//            /* Process last 2 values */
//            diff0 = (a[0] - b[0]) * w[0];
//            diff1 = (a[1] - b[1]) * w[1];
//            result += diff0 * diff0 + diff1 * diff1;

//            return result;

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
                    const int64_t parent_index);

            RRTNode(const RRTGrippersRepresentation& grippers_poses,
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

            RRTGrippersRepresentation grippers_poses_;
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
                    const sdf_tools::SignedDistanceField::ConstPtr environment_sdf,
                    const PRMHelper::Ptr& prm_helper,
                    const std::shared_ptr<std::mt19937_64>& generator,
                    // Planning algorithm parameters
                    const bool using_cbirrt_style_projection,
                    const size_t forward_tree_extend_iterations,
                    const size_t backward_tree_extend_iterations,
                    const size_t kd_tree_grow_threshold,
                    const bool use_brute_force_nn,
                    const double goal_bias,
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
                    const std::chrono::duration<double>& time_limit,
                    const bool planning_for_whole_robot);


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

            void storePath(const std::vector<RRTNode, RRTAllocator>& path) const;

            std::vector<RRTNode, RRTAllocator> loadStoredPath() const;

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

            int64_t nearestNeighbour(
                    const bool use_forward_tree,
                    const RRTNode& config);

            // Used for timing purposes
            // https://stackoverflow.com/questions/37786547/enforcing-statement-order-in-c
            int64_t nearestNeighbour_internal(
                    const bool use_forward_tree,
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
            const sdf_tools::SignedDistanceField::ConstPtr environment_sdf_;
            PRMHelper::Ptr prm_helper_;
            const std::shared_ptr<std::mt19937_64> generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;


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
            const double gripper_min_distance_to_obstacles_;

            // Use for double layer NN check
            const double band_distance2_scaling_factor_;
            const size_t band_max_points_;
            const double band_max_dist2_;

            const bool using_cbirrt_style_projection_;
            const size_t forward_tree_extend_iterations_;
            const size_t backward_tree_extend_iterations_;
            const bool use_brute_force_nn_;
            const size_t kd_tree_grow_threshold_;

            const int64_t max_shortcut_index_distance_;
            const uint32_t max_smoothing_iterations_;
            const uint32_t max_failed_smoothing_iterations_;
            std::uniform_int_distribution<int> uniform_shortcut_smoothing_int_distribution_;


            // Set/updated on each call of "rrtPlan"
            bool planning_for_whole_robot_;
            ssize_t total_dof_;
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


            // Planning and Smoothing statistics
            std::map<std::string, double> planning_statistics_;
            std::map<std::string, double> smoothing_statistics_;
            double total_sampling_time_;
            double total_nearest_neighbour_index_building_time_;
            double total_nearest_neighbour_index_searching_time_;
            double total_nearest_neighbour_linear_searching_time_;
            double total_nearest_neighbour_radius_searching_time_;
            double total_nearest_neighbour_time_;
            double total_projection_time_;
            double total_collision_check_time_;
            double total_band_forward_propogation_time_;
            double total_first_order_vis_propogation_time_;
            double total_everything_included_forward_propogation_time_;


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
