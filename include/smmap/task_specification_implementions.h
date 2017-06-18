#ifndef TASK_SPECIFICATION_IMPLEMENTIONS_HPP
#define TASK_SPECIFICATION_IMPLEMENTIONS_HPP

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
            ClothColabFolding(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual double calculateError_impl(
                    const WorldState& world_state) const final;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

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
            RopeCylinderCoverage(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const LineNeighbours neighbours_;
    };

    /**
     * @brief The ClothTableCoverage class
     */
    class ClothTableCoverage : public DirectCoverageTask
    {
        public:
            ClothTableCoverage(ros::NodeHandle& nh);

        private:
            void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const Grid4Neighbours neighbours_;
    };


    /**
     * @brief The ClothCylinderCoverage class
     */
    class ClothCylinderCoverage : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothCylinderCoverage(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothWAFR class
     */
    class ClothWAFR : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothWAFR(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothWall class
     */
    class ClothWall : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothWall(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothSinglePole class
     */
    class ClothSinglePole : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothSinglePole(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const Grid4Neighbours neighbours_;
    };

    /**
     * @brief The ClothDoubleSlit class
     */
    class ClothDoubleSlit : public DistanceBasedCorrespondencesTask
    {
        public:
            ClothDoubleSlit(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const Grid4Neighbours neighbours_;
    };


    /**
     * @brief The RopeMaze class
     */
    class RopeMaze : public FixedCorrespondencesTask
    {
        public:
            RopeMaze(ros::NodeHandle& nh);

        private:
            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const final;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const final;

            virtual std::vector<ssize_t> getNodeNeighbours_impl(const ssize_t node) const final;

            const LineNeighbours neighbours_;
    };
}

#endif // TASK_SPECIFICATION_IMPLEMENTIONS_HPP
