#ifndef TASK_SPECIFICATION_IMPLEMENTIONS_HPP
#define TASK_SPECIFICATION_IMPLEMENTIONS_HPP

#include <chrono>
#include <Eigen/Dense>
#include <arc_utilities/dijkstras.hpp>
#include "smmap/task_specification.h"
#include "smmap/point_reflector.hpp"
#include "smmap/ros_communication_helpers.hpp"

namespace smmap
{
    class CoverageTask : public TaskSpecification
    {
        public:
            CoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type)
                : TaskSpecification(nh, deformable_type, task_type)
                , cover_points_(GetCoverPoints(nh))
                , num_cover_points_(cover_points_.cols())
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
            /// Stores the points that we are trying to cover with the rope
            const ObjectPointSet cover_points_;
            ssize_t num_cover_points_;

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
    };

    /**
     * @brief The RopeCylinderCoverage class
     */
    class RopeCylinderCoverage : public CoverageTask
    {
        public:
            RopeCylinderCoverage(ros::NodeHandle& nh)
                : CoverageTask(nh, DeformableType::ROPE, TaskType::CYLINDER_COVERAGE)
                , cylinder_com_(GetCylinderCenterOfMassX(nh), GetCylinderCenterOfMassY(nh))
                , cylinder_radius_(GetCylinderRadius(nh))
                , rope_radius_(GetRopeRadius(nh))
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 10.0; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return  200.0; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.005; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 15.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const
            {
                vis.visualizeRope(marker_name, object_configuration, color);
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const
            {
                vis.visualizeRope(marker_name, object_configuration, colors);
            }

            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration) const
            {
                return CalculateErrorWithTheshold(cover_points_, current_configuration, 0.01);
            }

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const
            {
                ROS_INFO_NAMED("rope_coverage_task" , "Finding 'best' object delta");
                return calculateObjectErrorCorrectionDelta_Dijkstras(world_state.object_configuration_, 0.01);
            }

            virtual Eigen::VectorXd projectObjectDelta_impl(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta) const
            {
                #pragma message "Rope cylinder projection function makes a lot of assumptions - movements are small, will only penetrate the sides, etc."

                #pragma omp parallel for
                for (ssize_t point_ind = 0; point_ind < num_nodes_; point_ind++)
                {
                    const Eigen::Vector2d new_pos = object_configuration.block<2, 1>(0, point_ind)
                            + object_delta.segment<2>(point_ind * 3);

                    const Eigen::Vector2d vector_from_com = new_pos - cylinder_com_;
                    if (vector_from_com.norm() < cylinder_radius_ + rope_radius_)
                    {
                        const Eigen::Vector2d adjusted_pos = cylinder_com_ +
                                vector_from_com.normalized() * (cylinder_radius_ + rope_radius_);

                        object_delta.segment<2>(point_ind * 3) =
                                adjusted_pos - object_configuration.block<2, 1>(0, point_ind);
                    }
                }

                return object_delta;
            }

        private:
            /// Center of the cylinder in the plane defined by the table
            const Eigen::Vector2d cylinder_com_;

            /// Radious of the cylinder
            const double cylinder_radius_;

            /// Radius of the rope
            const double rope_radius_;
    };

    /**
     * @brief The ClothCylinderCoverage class
     */
    class ClothCylinderCoverage : public CoverageTask
    {
        public:
            ClothCylinderCoverage(ros::NodeHandle& nh)
                : CoverageTask(nh, DeformableType::CLOTH, TaskType::CYLINDER_COVERAGE)
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 14.0; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return  1000.0; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.03; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 12.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const
            {
                vis.visualizeCloth(marker_name, object_configuration, color);
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const
            {
                vis.visualizeCloth(marker_name, object_configuration, colors);
            }

            virtual double calculateError_impl(
                    const ObjectPointSet &current_configuration) const
            {
                return CalculateErrorWithTheshold(cover_points_, current_configuration, 0.002);
            }

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const
            {
                ROS_INFO_NAMED("cloth_cylinder_coverage" , "Finding 'best' cloth delta");
                return calculateObjectErrorCorrectionDelta_Dijkstras(world_state.object_configuration_, 0.002);
//                return CalculateObjectErrorCorrectionDeltaWithThreshold(
//                            cover_points_, world_state.object_configuration_, 0.002);
            }

            virtual Eigen::VectorXd projectObjectDelta_impl(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta) const
            {
                assert(false && "This function is not specified for this experiment yet");
                (void)object_configuration;
                (void)object_delta;
            }

        private:
    };

    /**
     * @brief The ClothTableCoverage class
     */
    class ClothTableCoverage : public CoverageTask
    {
        public:
            ClothTableCoverage(ros::NodeHandle& nh)
                : CoverageTask(nh, DeformableType::CLOTH, TaskType::TABLE_COVERAGE)
                , table_min_x_(GetTableSurfaceX(nh) - GetTableHalfExtentsX(nh))
                , table_max_x_(GetTableSurfaceX(nh) + GetTableHalfExtentsX(nh))
                , table_min_y_(GetTableSurfaceY(nh) - GetTableHalfExtentsY(nh))
                , table_max_y_(GetTableSurfaceY(nh) + GetTableHalfExtentsY(nh))
                , table_z_(GetTableSurfaceZ(nh))
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 14.0; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return  1000.0; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.03; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 2.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const
            {
                vis.visualizeCloth(marker_name, object_configuration, color);
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const
            {
                vis.visualizeCloth(marker_name, object_configuration, colors);
            }

            virtual double calculateError_impl(
                    const ObjectPointSet &current_configuration) const
            {
                return CalculateErrorWithTheshold(cover_points_, current_configuration, 0.002);
            }

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const
            {
                ROS_INFO_NAMED("cloth_table_coverage" , "Finding 'best' cloth delta");
                return calculateObjectErrorCorrectionDelta_Dijkstras(world_state.object_configuration_, 0.002);
//                return CalculateObjectErrorCorrectionDeltaWithThreshold(
//                            cover_points_, world_state.object_configuration_, 0.002);
            }

            virtual Eigen::VectorXd projectObjectDelta_impl(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta) const
            {
                #pragma message "Cloth Table projection function makes a lot of assumptions - movements are small, will only penetrate the top, etc."

                #pragma omp parallel for
                for (ssize_t point_ind = 0; point_ind < object_configuration.cols(); point_ind++)
                {
                    const Eigen::Vector3d new_pos = object_configuration.col(point_ind)
                            + object_delta.segment<3>(point_ind * 3);

                    // TODO: move out of the table sideways?
                    // TODO: use Calder's SDF/collision resolution stuff?

                    // check if the new positition is in the same "vertical column" as the table
                    if (table_min_x_ <= new_pos(0) && new_pos(0) <= table_max_x_
                         && table_min_y_ <= new_pos(1) && new_pos(1) <= table_max_y_)
                    {
                        // Check if the new point position penetrated the object
                        // Note that I am only checking "downwards" penetratraion as this task should never even consider having the other type
                        if (new_pos(2) < table_z_)
                        {
                            object_delta(point_ind * 3 + 2) = table_z_ - object_configuration(2, point_ind);
                        }
                    }

                }

                return object_delta;
            }

        private:
            const double table_min_x_;
            const double table_max_x_;
            const double table_min_y_;
            const double table_max_y_;
            const double table_z_;
    };

    /**
     * @brief The ClothColabFolding class
     */
    class ClothColabFolding : public TaskSpecification
    {
        public:
            ClothColabFolding(ros::NodeHandle& nh)
                : TaskSpecification(nh, DeformableType::CLOTH, TaskType::COLAB_FOLDING)
                , point_reflector_(createPointReflector(nh))
                , mirror_map_(createMirrorMap(nh, point_reflector_))
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 0.7*20; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return 1000.0; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.03; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 6.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const
            {
                vis.visualizeCloth(marker_name, object_configuration, color);
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const
            {
                vis.visualizeCloth(marker_name, object_configuration, colors);
            }

            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration) const
            {
                double error = 0;

                for (std::map<long, long>::const_iterator ittr = mirror_map_.begin(); ittr != mirror_map_.end(); ittr++)
                {
                    error += (current_configuration.col(ittr->second) -
                               point_reflector_.reflect(current_configuration.col(ittr->first))).norm();
                }

                return error;
            }

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const
            {
                ROS_INFO_NAMED("cloth_colab_folding" , "Finding 'best' cloth delta");
                const ObjectPointSet& object_configuration = world_state.object_configuration_;

                ObjectDeltaAndWeight desired_cloth_delta(object_configuration.cols() * 3);

                long robot_cloth_points_ind = 0;
                for (std::map<long, long>::const_iterator ittr = mirror_map_.begin();
                      ittr != mirror_map_.end(); ittr++, robot_cloth_points_ind++)
                {
                    desired_cloth_delta.delta.segment<3>(ittr->second * 3) =
                            point_reflector_.reflect(object_configuration.block<3, 1>(0, ittr->first))
                            - object_configuration.block<3, 1>(0, ittr->second);

                    const double weight = desired_cloth_delta.delta.segment<3>(ittr->second * 3).norm();
                    desired_cloth_delta.weight(ittr->second * 3) = weight;
                    desired_cloth_delta.weight(ittr->second * 3 + 1) = weight;
                    desired_cloth_delta.weight(ittr->second * 3 + 2) = weight;
                }

                return desired_cloth_delta;
            }

            virtual Eigen::VectorXd projectObjectDelta_impl(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta) const
            {
                (void)object_configuration;
                return object_delta;
            }

        private:
            const PointReflector point_reflector_;
            PointReflector createPointReflector(ros::NodeHandle& nh)
            {
                ROS_INFO_NAMED("cloth_colab_folding_task" , "Getting mirror line");

                // Get the initial configuration of the object
                ros::ServiceClient mirror_line_client =
                    nh.serviceClient<smmap_msgs::GetMirrorLine>(GetMirrorLineTopic(nh));

                mirror_line_client.waitForExistence();

                smmap_msgs::GetMirrorLine mirror_line_data;
                mirror_line_client.call(mirror_line_data);

                return PointReflector(mirror_line_data.response.mid_x,
                                       mirror_line_data.response.min_y,
                                       mirror_line_data.response.max_y);
            }

            const std::map<long, long> mirror_map_;
            static std::map<long, long> createMirrorMap(
                    ros::NodeHandle& nh,
                    const PointReflector& point_reflector)
            {
                ObjectPointSet object_initial_configuration =
                    GetObjectInitialConfiguration(nh);

                ROS_INFO_NAMED("cloth_colab_folding", "Finding point correspondences");

                std::map<long, long> mirror_map;
                for (long node_ind = 0; node_ind < object_initial_configuration.cols(); node_ind++)
                {
                    // for every node on one side of the mirror line, find the closest match on the other side
                    // Note that nodes that have an x value > than mid_x are on the manual gripper side
                    if (object_initial_configuration(0, node_ind) > point_reflector.get_mid_x())
                    {
                        long mirror_ind = closestPointInSet(object_initial_configuration,
                                point_reflector.reflect(object_initial_configuration.block<3, 1>(0, node_ind)));

                        mirror_map[node_ind] = mirror_ind;
                    }
                }

                return mirror_map;
            }
    };

    /**
     * @brief The ClothWAFR class
     */
    class ClothWAFR : public CoverageTask
    {
        public:
            ClothWAFR(ros::NodeHandle& nh)
                : CoverageTask(nh, DeformableType::CLOTH, TaskType::WAFR)
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 14.0; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return  1000.0; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.03; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 15.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const
            {
                vis.visualizeCloth(marker_name, object_configuration, color);
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const
            {
                vis.visualizeCloth(marker_name, object_configuration, colors);
            }

            virtual double calculateError_impl(
                    const ObjectPointSet &current_configuration) const
            {
                return CalculateErrorWithTheshold(cover_points_, current_configuration, 0.002);
            }

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const
            {
                ROS_INFO_NAMED("cloth_wafr_coverage" , "Finding 'best' cloth delta");
                return calculateObjectErrorCorrectionDelta_Dijkstras(world_state.object_configuration_, 0.002);
//                return CalculateObjectErrorCorrectionDeltaWithThreshold(
//                            cover_points_, world_state.object_configuration_, 0.002);
            }

            virtual Eigen::VectorXd projectObjectDelta_impl(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta) const
            {
                assert(false && "This function is not specified for this experiment yet");
                (void)object_delta;
                (void)object_configuration;
            }

        private:
    };
}

#endif // TASK_SPECIFICATION_IMPLEMENTIONS_HPP
