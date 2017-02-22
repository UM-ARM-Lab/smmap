#ifndef VISUALIZATION_TOOLS_H
#define VISUALIZATION_TOOLS_H

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <smmap_experiment_params/ros_params.hpp>

#include "smmap/trajectory.hpp"

namespace smmap
{
    class Visualizer
    {
        public:
            static void InitializeStandardColors();
            static std_msgs::ColorRGBA Red();
            static std_msgs::ColorRGBA Green();
            static std_msgs::ColorRGBA Blue();
            static std_msgs::ColorRGBA Black();
            static std_msgs::ColorRGBA Magenta();

        public:
            Visualizer(ros::NodeHandle& nh);

            void visualizePoints(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d points,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 0) const;

            void visualizePoints(const std::string& marker_name,
                    const EigenHelpers::VectorVector3d points,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id) const;

            void visualizeRope(
                    const std::string& marker_name,
                    const ObjectPointSet& rope,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeRope(
                    const std::string& marker_name,
                    const ObjectPointSet& rope,
                    const std::vector<std_msgs::ColorRGBA>& colors) const;

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std::vector<std_msgs::ColorRGBA>& colors) const;

            visualization_msgs::MarkerArray::_markers_type createGripperMarker(
                    const std::string& marker_name,
                    const Eigen::Affine3d& eigen_pose,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeGripper(
                    const std::string& marker_name,
                    const Eigen::Affine3d& eigen_pose,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeGrippers(
                    const std::string& marker_name,
                    const EigenHelpers::VectorAffine3d eigen_poses,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeObjectDelta(
                    const std::string& marker_name,
                    const ObjectPointSet& current,
                    const ObjectPointSet& desired,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const geometry_msgs::Point& start,
                    const geometry_msgs::Point& end,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Vector3d& start,
                    const Eigen::Vector3d& end,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Affine3d &start,
                    const Eigen::Affine3d &end,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeLines(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& start,
                    const EigenHelpers::VectorVector3d& end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 0) const;

            void visualizeXYZTrajectory(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& point_sequence,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 0) const;

            void deletePoints(
                    const std::string& marker_name,
                    const int32_t id = 0) const;

            void deleteXYZTrajectory(
                    const std::string& marker_name,
                    const int32_t id = 0) const;

        private:
            const std::string world_frame_name_;
            const double gripper_apperture_;
            mutable ros::Publisher visualization_marker_pub_;
            mutable ros::Publisher visualization_marker_array_pub_;

            static bool standard_colors_initialized_;
            static std_msgs::ColorRGBA red_;
            static std_msgs::ColorRGBA green_;
            static std_msgs::ColorRGBA blue_;
            static std_msgs::ColorRGBA black_;
            static std_msgs::ColorRGBA magenta_;
    };
}

#endif // VISUALIZATION_TOOLS_H
