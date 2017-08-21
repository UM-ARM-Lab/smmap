#ifndef TASK_SPECIFICATION_H
#define TASK_SPECIFICATION_H

#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <Eigen/Dense>
#include <arc_utilities/dijkstras.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <sdf_tools/sdf.hpp>
#include <deformable_manipulation_experiment_params/task_enums.h>
#include <deformable_manipulation_experiment_params/xyzgrid.h>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/task_function_pointer_types.h"
#include "smmap/visualization_tools.h"
#include "smmap/timing.hpp"

namespace smmap
{
    class TaskSpecification
    {
        public:
            typedef std::shared_ptr<TaskSpecification> Ptr;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Static builder function
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            static TaskSpecification::Ptr MakeTaskSpecification(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph);

        public:

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Constructor to initialize objects that all TaskSpecifications share
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            TaskSpecification(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const DeformableType deformable_type,
                    const TaskType task_type,
                    const bool is_dijkstras_type_task = false);

            TaskSpecification(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    Visualizer vis,
                    const DeformableType deformable_type,
                    const TaskType task_type,
                    const bool is_dijkstras_type_task = false);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual function wrappers
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            void visualizeDeformableObject(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeDeformableObject(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const;

            double calculateError(
                    const WorldState& world_state);

            /**
             * @brief calculateObjectDesiredDelta
             * @param world_state
             * @return return.first is the desired movement of the object
             *         return.second is the importance of that part of the movement
             */
            ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta(
                    const WorldState& world_state);

            bool taskDone(
                    const WorldState& world_state);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Helper functions
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            double defaultDeformability() const;        // k
            double collisionScalingFactor() const;      // beta (or k2)
            double maxStretchFactor() const;        // lambda
            double maxTime() const;                     // max simulation time when scripting things

            bool stretchingConstraintViolated(
                    const ssize_t first_node_ind,
                    const Eigen::Vector3d& first_node,
                    const ssize_t second_node_ind,
                    const Eigen::Vector3d& second_node) const;

            /**
             * @brief calculateStretchingCorrectionDeltaFullyConnected
             * @param world_state
             * @return
             */
            ObjectDeltaAndWeight calculateStretchingCorrectionDeltaFullyConnected(
                    const ObjectPointSet& object_configuration,
                    bool visualize) const;

            /**
             * @brief calculateStretchingCorrectionDeltaPairwise
             * @param object_configuration
             * @param visualize
             * @return
             */
            ObjectDeltaAndWeight calculateStretchingCorrectionDeltaPairwise(
                    const ObjectPointSet& object_configuration,
                    bool visualize) const;

            /**
             * @brief computeStretchingCorrection
             * @param object_configuration
             * @return
             */
            ObjectDeltaAndWeight calculateStretchingCorrectionDelta(
                    const WorldState& world_state,
                    bool visualize) const;

            /**
             * @brief combineErrorCorrectionAndStretchingCorrection
             * @param error_correction
             * @param stretching_correction
             * @return
             */
            ObjectDeltaAndWeight combineErrorCorrectionAndStretchingCorrection(
                    const ObjectDeltaAndWeight& error_correction,
                    const ObjectDeltaAndWeight& stretching_correction) const;


            ObjectDeltaAndWeight calculateDesiredDirection(const WorldState& world_state);

            std::vector<ssize_t> getNodeNeighbours(const ssize_t node) const;

            const std::vector<long>& getGripperAttachedNodesIndices(const size_t gripper_idx) const;

        private:
            // Data needed to avoid re-calculating the first desired step repeatedly
            std::atomic_bool first_step_calculated_;
            std::mutex first_step_mtx_;
            double first_step_last_simtime_calced_;
            ObjectDeltaAndWeight first_step_desired_motion_;
            ObjectDeltaAndWeight first_step_error_correction_;
            ObjectDeltaAndWeight first_step_stretching_correction_;

            // Data needed to avoid re-calculating the current error repeatedly
            std::atomic_bool current_error_calculated_;
            std::mutex current_error_mtx_;
            double current_error_last_simtime_calced_;
            double current_error_;

        public:
            // Records of task and deformable type if various visualizers or whatever need them
            const DeformableType deformable_type_;
            const TaskType task_type_;
            const bool is_dijkstras_type_task_;

        protected:
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Prevent deletion of base pointer
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            ~TaskSpecification() {}

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Objects shared by all task specifications
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            Visualizer vis_;

            const std::vector<GripperData> grippers_data_;
            const Eigen::MatrixXd object_initial_node_distance_;
            const ssize_t num_nodes_;


            const double default_deformability_;        // k
            const double collision_scaling_factor_;     // beta (or k2)
            const double max_stretch_factor_;           // used to be lambda
            const double max_time_;                     // max simulation time when scripting things

        private:
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that each task specification must provide
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const = 0;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const = 0;

            virtual double calculateError_impl(
                    const WorldState& world_state) = 0;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) = 0;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const = 0;

            virtual bool taskDone_impl(
                    const WorldState& world_state) = 0;
    };

    class CoverageTask : public TaskSpecification
    {
        public:
            CoverageTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const DeformableType deformable_type,
                    const TaskType task_type,
                    const bool is_dijkstras_type_task);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Publically viewable variables
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Note that work_space_grid_ and the environment_sdf_ are using different
            // resolutions due to the way the SDF is created in CustomScene
            const XYZGrid work_space_grid_;
            const sdf_tools::SignedDistanceField environment_sdf_;

            bool pointIsCovered(const ssize_t cover_idx, const Eigen::Vector3d& test_point) const;

            /// Stores the points that we are trying to cover with the rope
            const ObjectPointSet cover_points_;
            const ObjectPointSet cover_point_normals_;
            const ssize_t num_cover_points_;

            const double error_threshold_along_normal_;
            const double error_threshold_distance_to_normal_;
            const double error_threshold_task_done_;
    };


    class DirectCoverageTask : public CoverageTask
    {
        public:
            DirectCoverageTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const DeformableType deformable_type,
                    const TaskType task_type);

        private:
            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) final;

            virtual double calculateError_impl(
                    const WorldState& world_state) final;
    };

    class DijkstrasCoverageTask : public CoverageTask
    {
        public:
            struct Correspondences
            {
                public:
                    Correspondences(const ssize_t num_nodes)
                        : correspondences_(num_nodes)
                        , correspondences_next_step_(num_nodes)
                        , correspondences_distances_(num_nodes)
                        , correspondences_is_covered_(num_nodes)
                    {
                        assert(num_nodes > 0);
                    }

                    std::vector<ssize_t> uncovered_target_points_idxs_;     // Indices of the taget points that are uncovered
                    std::vector<double> uncovered_target_points_distances_; // Distance to the deformable object for each uncovered target point

                    std::vector<std::vector<ssize_t>> correspondences_;                     // Vector of size num_nodes_, each entry is a list indices into the cover_points_ data
                    std::vector<EigenHelpers::VectorVector3d> correspondences_next_step_;   // Vector of size num_nodes_, each entry is a list of "next steps" if moving towards the corresponding target
                    std::vector<std::vector<double>> correspondences_distances_;            // Vector of size num_nodes_, each entry is a list of distances to the corresponding cover_point as listed in correspondences_
                    std::vector<std::vector<bool>> correspondences_is_covered_;             // Vector of size num_nodes_, each entry is a list which desecribes if the corresponding cover_point is already "covered"
            };

            DijkstrasCoverageTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const DeformableType deformable_type,
                    const TaskType task_type);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual function wrappers
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            const Correspondences& getCoverPointCorrespondences(
                    const WorldState& world_state);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Interface functions used externally
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            std::vector<EigenHelpers::VectorVector3d> findPathFromObjectToTarget(
                    const WorldState& world_state,
                    const size_t max_steps);

            ObjectDeltaAndWeight calculateErrorCorrectionDeltaFixedCorrespondences(
                    const WorldState& world_state,
                    const std::vector<std::vector<ssize_t>>& correspondences);

            std::vector<double> averageDijkstrasDistanceBetweenGrippersAndClusters(
                    const Eigen::Isometry3d& gripper_pose,
                    const std::vector<ssize_t>& cover_indices,
                    const std::vector<uint32_t>& cluster_labels,
                    const uint32_t num_clusters) const;

        protected:
            /// Free space graph that creates a vector field for the deformable object to follow
            arc_dijkstras::Graph<Eigen::Vector3d> free_space_graph_;

        private:
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that we implement
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            virtual double calculateError_impl(
                    const WorldState& world_state) final;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) final;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that others need to write
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            virtual Correspondences getCoverPointCorrespondences_impl(
                    const WorldState& world_state) const = 0;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Private helpers
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            bool saveDijkstrasResults();
            bool loadDijkstrasResults();

            EigenHelpers::VectorVector3d followCoverPointAssignments(
                    Eigen::Vector3d current_pos,
                    const std::vector<ssize_t>& cover_point_assignments,
                    const size_t maximum_itterations) const;

        protected:
            /// Map between cover point indices and graph indices, with distances
            std::vector<int64_t> cover_ind_to_free_space_graph_ind_;

            /// Dijkstras results, indexed by goal index, then current node index - each entry is a (next_node, distance to goal) pair
            std::vector<std::pair<std::vector<int64_t>, std::vector<double>>> dijkstras_results_;

            // Data needed to avoid re-calculating the correspondences repeatedly
            std::atomic_bool current_correspondences_calculated_;
            std::mutex current_correspondences_mtx_;
            double current_correspondences_last_simtime_calced_;
            Correspondences current_correspondences_;
    };


    class DistanceBasedCorrespondencesTask : public DijkstrasCoverageTask
    {
        public:
            DistanceBasedCorrespondencesTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const DeformableType deformable_type,
                    const TaskType task_type);

        private:
            virtual Correspondences getCoverPointCorrespondences_impl(
                    const WorldState& world_state) const final;

            std::tuple<ssize_t, double, ssize_t, bool> findNearestObjectPoint(
                    const WorldState& world_state,
                    const ssize_t cover_idx) const;
    };

    class FixedCorrespondencesTask : public DijkstrasCoverageTask
    {
        public:
            FixedCorrespondencesTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const DeformableType deformable_type,
                    const TaskType task_type);

        protected:
            std::vector<std::vector<ssize_t>> correspondences_internal_fixed_;

        private:
            virtual Correspondences getCoverPointCorrespondences_impl(
                    const WorldState& world_state) const final;
    };


    class LineNeighbours
    {
        public:
            LineNeighbours(const ssize_t num_nodes);
            std::vector<ssize_t> getNodeNeighbours(const ssize_t node) const;

        private:
            const ssize_t num_nodes_;
    };

    class Grid4Neighbours
    {
        public:
            Grid4Neighbours(const ssize_t num_nodes, const ssize_t stride);
            std::vector<ssize_t> getNodeNeighbours(const ssize_t node) const;

        private:
            const ssize_t num_nodes_;
            const ssize_t stride_;
    };
}

#endif // TASK_SPECIFICATION_H
