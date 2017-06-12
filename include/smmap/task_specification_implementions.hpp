#ifndef TASK_SPECIFICATION_IMPLEMENTIONS_HPP
#define TASK_SPECIFICATION_IMPLEMENTIONS_HPP

#include "smmap/task_specification.h"
#include "smmap/point_reflector.hpp"

namespace smmap
{
    /**
     * @brief The RopeCylinderCoverage class
     */
    class RopeCylinderCoverage : public DirectCoverageTask
    {
        public:
            RopeCylinderCoverage(ros::NodeHandle& nh)
                : DirectCoverageTask(nh, DeformableType::ROPE, TaskType::ROPE_CYLINDER_COVERAGE)
                , neighbours_(num_nodes_)
            {}

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.01;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const LineNeighbours neighbours_;
    };

    /**
     * @brief The ClothCylinderCoverage class
     */
    class ClothCylinderCoverage : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothCylinderCoverage(ros::NodeHandle& nh)
                : DistanceBasedCorrespondencesTask(nh, DeformableType::CLOTH, TaskType::CLOTH_CYLINDER_COVERAGE)
                , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
            {}

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.002;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothTableCoverage class
     */
    class ClothTableCoverage : public DirectCoverageTask
    {
        public:
            ClothTableCoverage(ros::NodeHandle& nh)
                : DirectCoverageTask(nh, DeformableType::CLOTH, TaskType::CLOTH_TABLE_COVERAGE)
                , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
            {}

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.002;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothColabFolding class
     */
    class ClothColabFolding : public TaskSpecification
    {
        public:
            ClothColabFolding(ros::NodeHandle& nh)
                : TaskSpecification(nh, DeformableType::CLOTH, TaskType::CLOTH_COLAB_FOLDING)
                , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
                , point_reflector_(createPointReflector(nh))
                , mirror_map_(createMirrorMap(nh, point_reflector_))
            {}

        private:
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

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const Grid4Neighbours neighbours_;

        private:
            const PointReflector point_reflector_;
            PointReflector createPointReflector(ros::NodeHandle& nh)
            {
                ROS_INFO_NAMED("cloth_colab_folding_task" , "Getting mirror line");

                // Get the initial configuration of the object
                ros::ServiceClient mirror_line_client =
                    nh.serviceClient<deformable_manipulation_msgs::GetMirrorLine>(GetMirrorLineTopic(nh));

                mirror_line_client.waitForExistence();

                deformable_manipulation_msgs::GetMirrorLine mirror_line_data;
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
                        long mirror_ind = ClosestPointInSet(object_initial_configuration,
                                point_reflector.reflect(object_initial_configuration.col(node_ind)));

                        mirror_map[node_ind] = mirror_ind;
                    }
                }

                return mirror_map;
            }
    };

    /**
     * @brief The ClothWAFR class
     */
    class ClothWAFR : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothWAFR(ros::NodeHandle& nh)
                : DistanceBasedCorrespondencesTask(nh, DeformableType::CLOTH, TaskType::CLOTH_WAFR)
                , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
            {}

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.002;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothSinglePole class
     */
    class ClothSinglePole : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothSinglePole(ros::NodeHandle& nh)
                : DistanceBasedCorrespondencesTask(nh, DeformableType::CLOTH, TaskType::CLOTH_SINGLE_POLE)
                , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
            {}

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.002;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothWall class
     */
    class ClothWall : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothWall(ros::NodeHandle& nh)
                : DistanceBasedCorrespondencesTask(nh, DeformableType::CLOTH, TaskType::CLOTH_WALL)
                , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
            {}

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.002;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothDoubleSlit class
     */
    class ClothDoubleSlit : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothDoubleSlit(ros::NodeHandle& nh)
                : DistanceBasedCorrespondencesTask(nh, DeformableType::CLOTH, TaskType::CLOTH_DOUBLE_SLIT)
                , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
            {}

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.002;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The RopeMaze class
     */
    class RopeMaze : public FixedCorrespondencesTask
    {
        public:
            RopeMaze(ros::NodeHandle& nh)
                : FixedCorrespondencesTask(nh, DeformableType::ROPE, TaskType::ROPE_MAZE)
                , neighbours_(num_nodes_)
            {
                setCorrespondences();
            }

        private:
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

            virtual double getErrorThreshold_impl() const
            {
                return 0.01;
            }

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const
            {
                return neighbours_.getNodeNeighbours(node);
            }

            virtual void setCorrespondences_impl() final
            {
                assert(num_nodes_ == num_cover_points_);

                correspondences_.clear();
                correspondences_.reserve(num_nodes_);

                for (ssize_t idx = 0; idx < num_nodes_; ++idx)
                {
                    correspondences_.push_back({idx});
                }
            }

            const LineNeighbours neighbours_;
    };
}

#endif // TASK_SPECIFICATION_IMPLEMENTIONS_HPP
