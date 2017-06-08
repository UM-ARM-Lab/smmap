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
            ////////////////////////////////////////////////////////////////////
            // Static helper functions - could be private given how they are
            // used but making public as they are static
            ////////////////////////////////////////////////////////////////////

            static double CalculateErrorWithTheshold(
                    const ObjectPointSet& target_points,
                    const ObjectPointSet& deformable_object,
                    const double minimum_threshold);

            static ObjectDeltaAndWeight CalculateObjectErrorCorrectionDeltaWithThreshold(
                    const ObjectPointSet& target_points,
                    const ObjectPointSet& deformable_object,
                    const double minimum_threshold);

        public:
            typedef std::shared_ptr<TaskSpecification> Ptr;

            ////////////////////////////////////////////////////////////////////
            // Constructor to initialize objects that all TaskSpecifications share
            ////////////////////////////////////////////////////////////////////

            TaskSpecification(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type, const bool is_dijkstras_type_task = false);
            TaskSpecification(ros::NodeHandle& nh, Visualizer vis, const DeformableType deformable_type, const TaskType task_type, const bool is_dijkstras_type_task = false);

            ////////////////////////////////////////////////////////////////////
            // Static builder function
            ////////////////////////////////////////////////////////////////////

            static TaskSpecification::Ptr MakeTaskSpecification(
                    ros::NodeHandle& nh);

            ////////////////////////////////////////////////////////////////////
            // Virtual function wrappers
            ////////////////////////////////////////////////////////////////////

            double defaultDeformability() const;        // k
            double collisionScalingFactor() const;      // beta (or k2)
            double stretchingThreshold() const;         // lambda
            double maxTime() const;                     // max simulation time when scripting things

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
                    const ObjectPointSet& object_configuration) const;

            /**
             * @brief calculateObjectDesiredDelta
             * @param world_state
             * @return return.first is the desired movement of the object
             *         return.second is the importance of that part of the movement
             */
            ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta(
                    const WorldState& world_state) const;

            ////////////////////////////////////////////////////////////////////
            // Helper functions
            // TODO: Should these be virtual? virtual final?
            ////////////////////////////////////////////////////////////////////

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

        private:
            ObjectDeltaAndWeight first_step_desired_motion_;
            ObjectDeltaAndWeight first_step_error_correction_;
            ObjectDeltaAndWeight first_step_stretching_correction_;
            std::atomic_bool first_step_calculated_;
            std::mutex first_step_mtx_;
            double sim_time_last_time_first_step_calced_;


        public:
            // Records of task and deformable type if various visualizers or whatever need them
            const DeformableType deformable_type_;
            const TaskType task_type_;
            const bool is_dijkstras_type_task_;

        protected:
            ////////////////////////////////////////////////////////////////////
            // Prevent deletion of base pointer
            ////////////////////////////////////////////////////////////////////

            ~TaskSpecification() {}

            ////////////////////////////////////////////////////////////////////
            // Objects shared by all task specifications
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            Visualizer vis_;
            const Eigen::MatrixXd object_initial_node_distance_;
            const ssize_t num_nodes_;


            const double default_deformability_;        // k
            const double collision_scaling_factor_;     // beta (or k2)
            const double stretching_threshold_;         // lambda
            const double max_time_;                     // max simulation time when scripting things

        private:
            ////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that each task specification must provide
            ////////////////////////////////////////////////////////////////////////////////

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
                    const ObjectPointSet& object_configuration) const = 0;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const = 0;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const = 0;
    };

    class CoverageTask : public TaskSpecification
    {
        public:
            CoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type, const bool is_dijkstras_type_task);

            double getErrorThreshold() const;

        protected:
            virtual double getErrorThreshold_impl() const = 0;

            /// Stores the points that we are trying to cover with the rope
            const ObjectPointSet cover_points_;
            const ssize_t num_cover_points_;

        private:

            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration) const final;
    };

    class DirectCoverageTask : public CoverageTask
    {
        public:
            DirectCoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type);

        private:
            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const final;
    };

    class DijkstrasCoverageTask : public CoverageTask
    {
        public:
            DijkstrasCoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type);

            std::pair<std::vector<EigenHelpers::VectorVector3d>, std::vector<std::vector<ssize_t>>> findPathFromObjectToTarget(const ObjectPointSet& object_configuration, const double minimum_threshold, const size_t max_ittr) const;
            ObjectDeltaAndWeight getErrorCorrectionVectorsAndWeights(const ObjectPointSet& object_configuration, const std::vector<std::vector<ssize_t>>& cover_point_assignments) const;

        protected:
            bool saveDijkstrasResults();
            bool loadDijkstrasResults();

            std::tuple<ssize_t, double, ssize_t> findNearestObjectPoint(const ObjectPointSet& object_configuration, const ssize_t cover_ind) const;
            EigenHelpers::VectorVector3d followCoverPointAssignments(Eigen::Vector3d current_pos, const std::vector<ssize_t>& cover_point_assignments, const size_t maximum_itterations) const;

            ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_Dijkstras(const ObjectPointSet& object_configuration, const double minimum_threshold) const;

            /// Free space graph that creates a vector field for the deformable object to follow
            arc_dijkstras::Graph<Eigen::Vector3d> free_space_graph_;

        public:
            // Note that work_space_grid_ and the environment_sdf_ are using different resolutions due to the way the sdf
            // is created in CustomScene
            const XYZGrid work_space_grid_;
            const sdf_tools::SignedDistanceField environment_sdf_;

        private:
            /// Map between cover point indices and graph indices, with distances
            std::vector<int64_t> cover_ind_to_free_space_graph_ind_;
            /// Dijkstras results, indexed by goal index, then current node index - each entry is a (next_node, distance to goal) pair
            std::vector<std::pair<std::vector<int64_t>, std::vector<double>>> dijkstras_results_;

        private:
            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
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
