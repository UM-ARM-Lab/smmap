#ifndef TASK_SPECIFICATION_IMPLEMENTIONS_H
#define TASK_SPECIFICATION_IMPLEMENTIONS_H

#include "smmap/task_specification.h"
#include "smmap/point_reflector.hpp"

namespace smmap
{
    /**
     * @brief The ClothColabFolding class
     */
    class ClothColabFolding : public TaskSpecification
    {
        public:
            ClothColabFolding(ros::NodeHandle& nh, ros::NodeHandle& ph, smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual double calculateError_impl(
                    const WorldState& world_state) final;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            virtual bool taskDone_impl(
                    const WorldState& world_state) final;

            const Grid4Neighbours neighbours_;

        private:
            const PointReflector point_reflector_;
            static PointReflector createPointReflector(ros::NodeHandle& nh);

            const std::map<long, long> mirror_map_;
            static std::map<long, long> createMirrorMap(ros::NodeHandle& nh, const PointReflector& point_reflector);
    };


    /**
     * @brief The RopeCylinderCoverage class
     */
    class RopeCylinderCoverage : public DirectCoverageTask
    {
        public:
            RopeCylinderCoverage(ros::NodeHandle& nh, ros::NodeHandle& ph, smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            virtual bool taskDone_impl(
                    const WorldState& world_state) final;

            const LineNeighbours neighbours_;
    };

    /**
     * @brief The ClothTableCoverage class
     */
    class ClothTableCoverage : public DirectCoverageTask
    {
        public:
            ClothTableCoverage(ros::NodeHandle& nh, ros::NodeHandle& ph, smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            virtual bool taskDone_impl(
                    const WorldState& world_state) final;

            const Grid4Neighbours neighbours_;
    };


    /**
     * @brief The ClothDistanceBasedCorrespondences class. Uses the Dijkstra's
     * field to determine distances and directions to manipulate objects.
     * Correspondences are determined based on a dynamic "what's nearest right
     * now basis.
     */
    class ClothDistanceBasedCorrespondences : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothDistanceBasedCorrespondences(ros::NodeHandle& nh, ros::NodeHandle& ph, smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            virtual bool taskDone_impl(
                    const WorldState& world_state) final;

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The RopeFixedCorrespondences class. Uses the Dijkstra's
     * field to determine distances and directions to manipulate objects.
     * Correspondences are determined based on a fixed apriori assignment.
     */
    class RopeFixedCorrespondences : public FixedCorrespondencesTask
    {
        public:
            RopeFixedCorrespondences(ros::NodeHandle& nh, ros::NodeHandle& ph, smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            virtual bool taskDone_impl(
                    const WorldState& world_state) final;

            const LineNeighbours neighbours_;
    };

    /**
     * @brief The ClothFixedCorrespondences class. Uses the Dijkstra's
     * field to determine distances and directions to manipulate objects.
     * Correspondences are determined based on a fixed apriori assignment.
     */
    class ClothFixedCorrespondences : public FixedCorrespondencesTask
    {
        public:
            ClothFixedCorrespondences(ros::NodeHandle& nh, ros::NodeHandle& ph, smmap_utilities::Visualizer::Ptr vis);

        private:
            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            virtual bool taskDone_impl(
                    const WorldState& world_state) final;

            // TODO: Not clear that this will be the case moving forward - are all nodes in a grid?
            const Grid4Neighbours neighbours_;
    };
}

#endif // TASK_SPECIFICATION_IMPLEMENTIONS_H
