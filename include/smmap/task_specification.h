#ifndef TASK_SPECIFICATION_H
#define TASK_SPECIFICATION_H

#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <Eigen/Dense>
#include <arc_utilities/dijkstras.hpp>
#include <sdf_tools/sdf.hpp>
#include <deformable_manipulation_experiment_params/task_enums.h>
#include <deformable_manipulation_experiment_params/xyzgrid.h>
#include <smmap_utilities/visualization_tools.h>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/task_function_pointer_types.h"


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
                    ros::NodeHandle& ph,
                    smmap_utilities::Visualizer::Ptr vis);

        public:

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Constructor to initialize objects that all TaskSpecifications share
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            TaskSpecification(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    smmap_utilities::Visualizer::Ptr vis,
                    const bool is_dijkstras_type_task = false);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual function wrappers
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            void visualizeDeformableObject(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeDeformableObject(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1) const;

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
            double maxStretchFactor() const;            // lambda
            double maxBandLength() const;
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


            DesiredDirection calculateDesiredDirection(const WorldState& world_state);

            std::vector<ssize_t> getNodeNeighbours(const ssize_t node) const;

            const std::vector<long>& getGripperAttachedNodesIndices(const size_t gripper_idx) const;

        private:
            // Data needed to avoid re-calculating the first desired step repeatedly
            std::atomic_bool first_step_calculated_;
            std::mutex first_step_mtx_;
            double first_step_last_simtime_calced_;
            DesiredDirection first_step_desired_motion_;

            // Data needed to avoid re-calculating the current error repeatedly
            std::atomic_bool current_error_calculated_;
            std::mutex current_error_mtx_;
            double current_error_last_simtime_calced_;
            double current_error_;

            // Rescale the desired motion so that it is less absurd
            const double desired_motion_scaling_factor_;

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
            smmap_utilities::Visualizer::Ptr vis_;

            const std::vector<GripperData> grippers_data_;
            const Eigen::MatrixXd object_initial_node_distance_;
            const ssize_t num_nodes_;

            // TODO: Move these to the controller, not really part of the task anymore
            const double default_deformability_;        // k
            const double collision_scaling_factor_;     // beta (or k2)
            const double max_stretch_factor_;           // used to be lambda
            const double max_band_length_;              // Function of initial distance between the grippers, and max_stretch_factor_ - cached value to allow for arbitrary starting setups of the environment from bullet/sim
            const double max_time_;                     // max simulation time when scripting things

        private:
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that have a default implementation
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id) const;

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id) const;

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that each task specification must provide
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            virtual double calculateError_impl(
                    const WorldState& world_state) = 0;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) = 0;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const = 0;

            virtual bool taskDone_impl(
                    const WorldState& world_state) = 0;
    };

    /**
     * @brief The ModelAccuracyTestTask class. Essentially this is just a dummy placeholder for the task, and all output is meaningless
     */
    class ModelAccuracyTestTask : public TaskSpecification
    {
        public:
            ModelAccuracyTestTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual double calculateError_impl(
                    const WorldState& world_state) override final;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) override final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const override final;

            virtual bool taskDone_impl(
                    const WorldState& world_state) override final;
    };

    class CoverageTask : public TaskSpecification
    {
        public:
            CoverageTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    smmap_utilities::Visualizer::Ptr vis,
                    const bool is_dijkstras_type_task);

            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Publically viewable variables
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            // Note that work_space_grid_ and the environment_sdf_ are using different
            // resolutions due to the way the SDF is created in CustomScene
            const sdf_tools::SignedDistanceField::ConstPtr sdf_;
            const XYZGrid work_space_grid_;

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
                    smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) override final;

            virtual double calculateError_impl(
                    const WorldState& world_state) override final;
    };

    class DijkstrasCoverageTask : public CoverageTask
    {
        public:
            typedef std::shared_ptr<DijkstrasCoverageTask> Ptr;
            typedef std::shared_ptr<DijkstrasCoverageTask> ConstPtr;

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

                    std::vector<std::vector<ssize_t>> correspondences_;                     // Vector of size num_nodes_, each entry is a list of indices into the cover_points_ data
                    std::vector<EigenHelpers::VectorVector3d> correspondences_next_step_;   // Vector of size num_nodes_, each entry is a list of "next steps" if moving towards the corresponding target
                    std::vector<std::vector<double>> correspondences_distances_;            // Vector of size num_nodes_, each entry is a list of distances to the corresponding cover_point as listed in correspondences_
                    std::vector<std::vector<bool>> correspondences_is_covered_;             // Vector of size num_nodes_, each entry is a list which desecribes if the corresponding cover_point is already "covered"
            };

            DijkstrasCoverageTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    smmap_utilities::Visualizer::Ptr vis);

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

            void visualizeFreeSpaceGraph() const;

            void visualizeIndividualDijkstrasResult(
                    const size_t cover_idx,
                    const Eigen::Vector3d& querry_loc) const;

        protected:
            /// Free space graph that creates a vector field for the deformable object to follow
            arc_dijkstras::Graph<Eigen::Vector3d> free_space_graph_;

        private:
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that we implement
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////

            virtual double calculateError_impl(
                    const WorldState& world_state) override final;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) override final;

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

            Eigen::Vector3d sumVectorFields(
                    const std::vector<ssize_t>& cover_point_assignments,
                    const Eigen::Vector3d& querry_loc) const;

            EigenHelpers::VectorVector3d followCoverPointAssignments(
                    const Eigen::Vector3d& starting_pos,
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
            const bool visualize_correspondences_;
    };


    class DistanceBasedCorrespondencesTask : public DijkstrasCoverageTask
    {
        public:
            DistanceBasedCorrespondencesTask(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual Correspondences getCoverPointCorrespondences_impl(
                    const WorldState& world_state) const override final;

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
                    smmap_utilities::Visualizer::Ptr vis);

        protected:
            std::vector<std::vector<ssize_t>> correspondences_internal_fixed_;

        private:
            virtual Correspondences getCoverPointCorrespondences_impl(
                    const WorldState& world_state) const override final;
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

    // Helper structure to convert between x ind, y ind, and node ind for cloth;
    // TODO: Update this structure to be a "Grid8Neighbours" class, similar to above
    /*
    struct NodeXYInd
    {
        NodeXYInd()
        {}

        NodeXYInd(ssize_t x_num, ssize_t y_num)
            : num_x_steps_(x_num)
            , num_y_steps_(y_num)
            , num_nodes_(x_num * y_num)
        {}

        void SetNodeXYInd(ssize_t x_num, ssize_t y_num)
        {
            num_x_steps_ = x_num;
            num_y_steps_ = y_num;
            num_nodes_ = x_num * y_num;
        }

        ssize_t GetNodeInd(ssize_t x_ind, ssize_t y_ind)
        {
            assert(NodeInBound(x_ind, y_ind)||"xy_ind out of bound");
            return y_ind * num_x_steps_ + x_ind;
        }

        // first element is x ind, second is y ind
        std::pair<ssize_t, ssize_t> GetXYInd(ssize_t node_ind)
        {
            assert(NodeInBound(node_ind)||"node_ind out of bound");
            std::pair<ssize_t, ssize_t> xy_ind;
            xy_ind.second = node_ind / num_x_steps_;
            xy_ind.first = node_ind - xy_ind.second * num_x_steps_;
            return xy_ind;
        }

        // the size of vector is always 8, value is -1 if out of bound.
        // Layout :
        //   3  2  1
        //   4  X  0
        //   5  6  7
        std::vector<ssize_t> Neighbor8Ind(ssize_t node_ind)
        {
            const int num_neighbor = 8;
            std::vector<ssize_t> nearest_8_neighbor(num_neighbor, -1);

            if (NodeInBound(node_ind + 1))
            { nearest_8_neighbor.at(0) = node_ind + 1;}

            if (NodeInBound(node_ind + 1 + num_x_steps_))
            { nearest_8_neighbor.at(1) = node_ind + 1 + num_x_steps_;}

            if (NodeInBound(node_ind + num_x_steps_))
            { nearest_8_neighbor.at(2) = node_ind + num_x_steps_;}

            if (NodeInBound(node_ind - 1 + num_x_steps_))
            { nearest_8_neighbor.at(3) = node_ind - 1 + num_x_steps_;}

            if (NodeInBound(node_ind - 1))
            { nearest_8_neighbor.at(4) = node_ind - 1;}

            if (NodeInBound(node_ind - 1 - num_x_steps_))
            { nearest_8_neighbor.at(5) = node_ind - 1 - num_x_steps_;}

            if (NodeInBound(node_ind - num_x_steps_))
            { nearest_8_neighbor.at(6) = node_ind - num_x_steps_;}

            if (NodeInBound(node_ind + 1 - num_x_steps_))
            { nearest_8_neighbor.at(7) = node_ind + 1 - num_x_steps_;}

            return nearest_8_neighbor;
        }

        bool NodeInBound(ssize_t node_ind)
        {
            if ((node_ind >=0) && (node_ind < num_nodes_))
                return true;
            return false;
        }

        bool NodeInBound(ssize_t x_ind, ssize_t y_ind)
        {
            if ((x_ind >=0) && (x_ind < num_x_steps_))
            {
                if (((y_ind >=0) && (y_ind < num_y_steps_)))
                    return true;
            }
            return false;
        }

        ssize_t num_x_steps_;
        ssize_t num_y_steps_;
        ssize_t num_nodes_;
    };
    */
}

#endif // TASK_SPECIFICATION_H
