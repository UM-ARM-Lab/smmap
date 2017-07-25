#ifndef VISUALIZATION_TOOLS_H
#define VISUALIZATION_TOOLS_H

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <deformable_manipulation_experiment_params/ros_params.hpp>

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
            static std_msgs::ColorRGBA Yellow();
            static std_msgs::ColorRGBA Cyan();
            static std_msgs::ColorRGBA White();
            static std_msgs::ColorRGBA Silver();
            static std_msgs::ColorRGBA Coral();
            static std_msgs::ColorRGBA Olive();
            static std_msgs::ColorRGBA Orange();

        public:
            Visualizer(ros::NodeHandle& nh, ros::NodeHandle& ph);

            Visualizer(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const std::string& marker_topic,
                    const std::string& marker_array_topic);

            void clearVisualizationsBullet() const;

            void deleteObjects(
                    const std::string& marker_name,
                    const int32_t start_id = 0,
                    const int32_t end_id = 1024) const;

            void visualizePoints(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1,
                    const double scale = 0.005) const;

            void visualizePoints(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1,
                    const double scale = 0.005) const;

            void visualizeCubes(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& points,
                    const Eigen::Vector3d& scale,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeRope(
                    const std::string& marker_name,
                    const ObjectPointSet& rope,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeRope(
                    const std::string& marker_name,
                    const ObjectPointSet& rope,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1) const;

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std::vector<std_msgs::ColorRGBA>& colors,
                    const int32_t id = 1) const;

            visualization_msgs::MarkerArray::_markers_type createGripperMarker(
                    const std::string& marker_name,
                    const Eigen::Affine3d& eigen_pose,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeGripper(
                    const std::string& marker_name,
                    const Eigen::Affine3d& eigen_pose,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeGrippers(
                    const std::string& marker_name,
                    const EigenHelpers::VectorAffine3d eigen_poses,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

            void visualizeObjectDelta(
                    const std::string& marker_name,
                    const ObjectPointSet& current,
                    const ObjectPointSet& desired,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 2) const;


            // Visualization tool for object wrench (1D rope). --- Added by Mengyao
            // Q: What should be the right id?
            void visualizeObjectForce(
                    const std::string& marker_name,
                    const ObjectPointSet& current,
                    const std::vector<Eigen::Vector3d>& force,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 4) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const geometry_msgs::Point& start,
                    const geometry_msgs::Point& end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 3) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Vector3d& start,
                    const Eigen::Vector3d& end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 3) const;

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Affine3d &start,
                    const Eigen::Affine3d &end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 3) const;

            void visualizeLines(const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& start,
                    const EigenHelpers::VectorVector3d& end,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1,
                    const double scale = 0.001) const;

            void visualizeLineStrip(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& point_sequence,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1,
                    const double scale = 0.001) const;

            void visualizeXYZTrajectory(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& point_sequence,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id = 1) const;

        private:
            const bool disable_all_visualizations_;
            mutable ros::ServiceClient clear_markers_srv_;
            mutable ros::Publisher visualization_marker_pub_;
            mutable ros::Publisher visualization_marker_vector_pub_;

            // Data needed to properly create visualizations and markers
            const std::string world_frame_name_;
            const double gripper_apperture_;

            static bool standard_colors_initialized_;
            static std_msgs::ColorRGBA red_;
            static std_msgs::ColorRGBA green_;
            static std_msgs::ColorRGBA blue_;
            static std_msgs::ColorRGBA black_;
            static std_msgs::ColorRGBA magenta_;
            static std_msgs::ColorRGBA yellow_;
            static std_msgs::ColorRGBA cyan_;
            static std_msgs::ColorRGBA white_;
            static std_msgs::ColorRGBA silver_;
            static std_msgs::ColorRGBA coral_;
            static std_msgs::ColorRGBA olive_;
            static std_msgs::ColorRGBA orange_;
    };
}

#endif // VISUALIZATION_TOOLS_H
