#ifndef TASK_SPECIFICATION_H
#define TASK_SPECIFICATION_H

#include <chrono>
#include <memory>
#include <Eigen/Dense>
#include <arc_utilities/dijkstras.hpp>
#include <smmap_experiment_params/task_enums.h>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/task_function_pointer_types.h"
#include "smmap/visualization_tools.h"

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

            TaskSpecification(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type);
            TaskSpecification(ros::NodeHandle& nh, Visualizer vis, const DeformableType deformable_type, const TaskType task_type);

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
            double stretchingScalingThreshold() const;  // lambda
            double maxTime() const;                     // max simulation time when scripting things
            double errorHistoryThreshold() const;

            bool terminateTask(const WorldState& world_state, const double error);

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

            Eigen::VectorXd projectObjectDelta(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta) const;

            ////////////////////////////////////////////////////////////////////
            // Helper functions
            // TODO: Should these be virtual? virtual final?
            ////////////////////////////////////////////////////////////////////

            /**
             * @brief calculateStretchingCorrectionDelta
             * @param world_state
             * @return
             */
            ObjectDeltaAndWeight calculateStretchingCorrectionDelta(
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
             * @brief calculateStretchingError
             * @param object_configuration
             * @return
             */
            double calculateStretchingError(
                    const ObjectPointSet& object_configuration) const;

            /**
             * @brief calculateStretchingError
             * @param world_state
             * @return
             */
            double calculateStretchingError(
                    const WorldState& world_state) const;

            /**
             * @brief combineErrorCorrectionAndStretchingCorrection
             * @param error_correction
             * @param stretching_correction
             * @return
             */
            ObjectDeltaAndWeight combineErrorCorrectionAndStretchingCorrection(
                    const ObjectDeltaAndWeight& error_correction,
                    const ObjectDeltaAndWeight& stretching_correction) const;

            // Records of task and deformable type if various visualizers or whatever need them
            const DeformableType deformable_type_;
            const TaskType task_type_;

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

            Eigen::VectorXd error_history_;
            ssize_t next_error_history_ind_;
            bool error_history_buffer_full_;
            bool task_done_;

        private:
            ////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that each task specification must provide
            ////////////////////////////////////////////////////////////////////////////////

            virtual double deformability_impl() const = 0;              // k
            virtual double collisionScalingFactor_impl() const = 0;     // beta (or k2)
            virtual double stretchingScalingThreshold_impl() const = 0; // lambda
            virtual double maxTime_impl() const = 0;                    // max simulation time when scripting things
            virtual double errorHistoryThreshold_impl() const = 0;

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

            virtual Eigen::VectorXd projectObjectDelta_impl(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta) const = 0;

    };

    class CoverageTask : public TaskSpecification
    {
        public:
            CoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type)
                : TaskSpecification(nh, deformable_type, task_type)
                , cover_points_(GetCoverPoints(nh))
                , num_cover_points_(cover_points_.cols())
            {}

            double getErrorThreshold() const
            {
                return getErrorThreshold_impl();
            }

        protected:
            virtual double getErrorThreshold_impl() const = 0;

            /// Stores the points that we are trying to cover with the rope
            const ObjectPointSet cover_points_;
            const ssize_t num_cover_points_;

        private:

            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration) const final
            {
                return CalculateErrorWithTheshold(cover_points_, current_configuration, getErrorThreshold());
            }
    };

    class DirectCoverageTask : public CoverageTask
    {
        public:
            DirectCoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type)
                : CoverageTask(nh, deformable_type, task_type)
            {}

        private:
            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const final
            {
                ROS_INFO_NAMED("direct_coverage_task" , "Finding 'best' object delta");
                return CalculateObjectErrorCorrectionDeltaWithThreshold(
                            cover_points_, world_state.object_configuration_, getErrorThreshold());
            }
    };

    class DijkstrasCoverageTask : public CoverageTask
    {
        public:
            DijkstrasCoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type)
                : CoverageTask(nh, deformable_type, task_type)
                , world_x_min_(GetWorldXMin(nh))
                , world_x_step_(GetWorldXStep(nh))
                , world_x_num_steps_(GetWorldXNumSteps(nh))
                , world_y_min_(GetWorldYMin(nh))
                , world_y_step_(GetWorldYStep(nh))
                , world_y_num_steps_(GetWorldYNumSteps(nh))
                , world_z_min_(GetWorldZMin(nh))
                , world_z_step_(GetWorldZStep(nh))
                , world_z_num_steps_(GetWorldZNumSteps(nh))
            {
                GetFreeSpaceGraph(nh, free_space_graph_, cover_ind_to_free_space_graph_ind_);
                assert(cover_ind_to_free_space_graph_ind_.size() == (size_t)num_cover_points_);

//                EigenHelpers::VectorVector3d graph_nodes;
//                graph_nodes.reserve(free_space_graph_.GetNodesImmutable().size());
//                for (size_t node_ind = 0; node_ind < free_space_graph_.GetNodesImmutable().size(); node_ind++)
//                {
//                    graph_nodes.push_back(free_space_graph_.GetNodeImmutable(node_ind).GetValueImmutable());
//                }
//                ROS_INFO_STREAM_NAMED("coverage_task", "Visualizing " << graph_nodes.size() << " graph nodes");
//                auto blue = Visualizer::Blue();
//                blue.a = 0.2f;
//                vis_.visualizePoints("graph_nodes", graph_nodes, blue);

                ROS_INFO_STREAM_NAMED("coverage_task", "Generating " << num_cover_points_ << " Dijkstra's solutions");
                const std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
                dijkstras_results_.resize(num_cover_points_);
                #pragma omp parallel for schedule(guided)
                for (size_t cover_ind = 0; cover_ind < (size_t)num_cover_points_; cover_ind++)
                {
                    const int64_t free_space_graph_ind = cover_ind_to_free_space_graph_ind_[cover_ind];
                    auto result = arc_dijkstras::SimpleDijkstrasAlgorithm<Eigen::Vector3d>::PerformDijkstrasAlgorithm(free_space_graph_, free_space_graph_ind);
                    dijkstras_results_[cover_ind] = result.second;
                }
                const std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
                ROS_INFO_STREAM_NAMED("coverage_task", "Found solutions in " << duration << " seconds");
            }

        protected:
            ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_Dijkstras(
                    const ObjectPointSet& object_configuration, const double minimum_threshold) const
            {
                ROS_INFO_NAMED("coverage_task" , "Finding 'best' object delta");

                ObjectDeltaAndWeight desired_object_delta(num_nodes_ * 3);

                const std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
                // For every cover point, find the nearest deformable object point
                for (ssize_t cover_ind = 0; cover_ind < num_cover_points_; cover_ind++)
                {
                    const Eigen::Vector3d& cover_point = cover_points_.col(cover_ind);

                    // Find the closest deformable object point
                    double min_dist = std::numeric_limits<double>::infinity();
                    ssize_t min_ind = -1;
                    ssize_t target_point_ind_in_graph = -1;

                    for (ssize_t deformable_ind = 0; deformable_ind < num_nodes_; deformable_ind++)
                    {
                        const Eigen::Vector3d& deformable_point = object_configuration.col(deformable_ind);
                        const double straight_line_distance_squared = (cover_point - deformable_point).squaredNorm();

                        int64_t target_ind;
                        double graph_dist;

                        // If we are more than a grid cell away from the cover point, then lookup our position in the rest of the grid
                        if (straight_line_distance_squared > 2.0 * std::min({world_x_min_, world_y_min_, world_z_min_}))
                        {
                            const ssize_t deformable_point_ind_in_graph = worldPosToGridIndex(deformable_point);
                            target_ind = dijkstras_results_[(size_t)cover_ind].first[deformable_point_ind_in_graph];
                            graph_dist = dijkstras_results_[(size_t)cover_ind].second[deformable_point_ind_in_graph];
                        }
                        // Otherwise, use the cover point directly
                        else
                        {
                            target_ind = cover_ind_to_free_space_graph_ind_[cover_ind];
                            graph_dist = std::sqrt(straight_line_distance_squared);
                        }

                        // If we've found something closer, update our records
                        if (graph_dist < min_dist)
                        {
                            min_dist = graph_dist;
                            min_ind= deformable_ind;
                            target_point_ind_in_graph = target_ind;
                        }
                    }

                    // If we are at least some minimum threshold away, use this
                    // cover point as a "pull" on the nearest deformable point
                    if (min_dist > minimum_threshold)
                    {
                        const Eigen::Vector3d& closest_point = object_configuration.col(min_ind);
                        const Eigen::Vector3d& target_point = free_space_graph_.GetNodeImmutable(target_point_ind_in_graph).GetValueImmutable();

                        desired_object_delta.delta.segment<3>(min_ind * 3) =
                                desired_object_delta.delta.segment<3>(min_ind * 3)
                                + (target_point - closest_point);

                        const double weight = std::max(desired_object_delta.weight(min_ind * 3), min_dist);
                        desired_object_delta.weight(min_ind * 3) = weight;
                        desired_object_delta.weight(min_ind * 3 + 1) = weight;
                        desired_object_delta.weight(min_ind * 3 + 2) = weight;
                    }
                }
                const std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
                const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
                ROS_INFO_STREAM_NAMED("coverage_task", "Found best delta in " << duration << " milliseconds");

                return desired_object_delta;
            }

            #pragma message "These 3 functions are duplicated from CustomScene()"
            ssize_t xyzIndexToGridIndex(const ssize_t x_ind, const ssize_t y_ind, const ssize_t z_ind) const
            {
                // If the point is in the grid, return the index
                if ((0 <= x_ind && x_ind < world_x_num_steps_)
                    && (0 <= y_ind && y_ind < world_y_num_steps_)
                    && (0 <= z_ind && z_ind < world_z_num_steps_))
                {
                    return (x_ind * world_y_num_steps_ + y_ind) * world_z_num_steps_ + z_ind;
                }
                // Otherwise return -1
                else
                {
                    return -1;
                }
            }

            ssize_t worldPosToGridIndex(const double x, const double y, const double z) const
            {
                const int64_t x_ind = std::lround((x - world_x_min_) / world_x_step_);
                const int64_t y_ind = std::lround((y - world_y_min_) / world_y_step_);
                const int64_t z_ind = std::lround((z - world_z_min_) / world_z_step_);

                return xyzIndexToGridIndex(
                            arc_helpers::ClampValue(x_ind, 0L, world_x_num_steps_ - 1),
                            arc_helpers::ClampValue(y_ind, 0L, world_y_num_steps_ - 1),
                            arc_helpers::ClampValue(z_ind, 0L, world_z_num_steps_ - 1));
            }

            ssize_t worldPosToGridIndex(const Eigen::Vector3d& vec) const
            {
                return worldPosToGridIndex(vec(0), vec(1), vec(2));
            }

        protected:
            /// Free space graph that creates a vector field for the deformable object to follow
            arc_dijkstras::Graph<Eigen::Vector3d> free_space_graph_;

            /// Variables describing the extents of the graph
            const double world_x_min_;
            const double world_x_step_;
            const int64_t world_x_num_steps_;

            const double world_y_min_;
            const double world_y_step_;
            const int64_t world_y_num_steps_;

            const double world_z_min_;
            const double world_z_step_;
            const int64_t world_z_num_steps_;

            /// Map between cover point indices and graph indices, with distances
            std::vector<int64_t> cover_ind_to_free_space_graph_ind_;
            /// Dijkstras results, indexed by goal index, then current node index - each entry is a (next_node, distance to goal) pair
            std::vector<std::pair<std::vector<int64_t>, std::vector<double>>> dijkstras_results_;

        private:
            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const final
            {
                ROS_INFO_NAMED("dijkstras_coverage_task" , "Finding 'best' object delta");
                return calculateObjectErrorCorrectionDelta_Dijkstras(world_state.object_configuration_, getErrorThreshold());
            }
    };
}

#endif // TASK_SPECIFICATION_H
