#ifndef VISUALIZATION_TOOLS_H
#define VISUALIZATION_TOOLS_H

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include "smmap/trajectory.hpp"
#include "smmap/ros_params.hpp"

namespace smmap
{
    class Visualizer
    {
        public:
            Visualizer( ros::NodeHandle& nh );

            void visualizeRope(
                    const std::string& marker_name,
                    const ObjectPointSet& rope,
                    const std_msgs::ColorRGBA& color);

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std_msgs::ColorRGBA& color );

            void visualizeCloth(
                    const std::string& marker_name,
                    const ObjectPointSet& cloth,
                    const std::vector< std_msgs::ColorRGBA >& colors );

            void visualizeGripper(
                    const std::string& marker_name,
                    const geometry_msgs::Pose& pose,
                    const std_msgs::ColorRGBA& color );


            void visualizeObjectDelta(
                    const std::string& marker_name,
                    const ObjectPointSet& current,
                    const ObjectPointSet& desired );

            void visualizeTranslation(
                    const std::string& marker_name,
                    const geometry_msgs::Point& start,
                    const geometry_msgs::Point& end,
                    const std_msgs::ColorRGBA& color );

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Vector3d& start,
                    const Eigen::Vector3d& end,
                    const std_msgs::ColorRGBA& color );

            void visualizeTranslation(
                    const std::string& marker_name,
                    const Eigen::Affine3d &start,
                    const Eigen::Affine3d &end,
                    const std_msgs::ColorRGBA& color );

            void visualizeLines(
                    const std::string& marker_name,
                    const EigenHelpers::VectorVector3d& start,
                    const EigenHelpers::VectorVector3d& end,
                    const std_msgs::ColorRGBA& color );

        private:
            ros::Publisher visualization_marker_pub_;
            ros::Publisher visualization_marker_array_pub_;

    };
}

#endif // VISUALIZATION_TOOLS_H
