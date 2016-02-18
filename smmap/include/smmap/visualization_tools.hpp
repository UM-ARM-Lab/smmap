#ifndef VISUALIZATION_TOOLS_HPP
#define VISUALIZATION_TOOLS_HPP

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include "smmap/trajectory.hpp"
#include "smmap/ros_params.hpp"

namespace smmap {
    class Visualizer
    {
        public:
            Visualizer( ros::NodeHandle& nh )
            {
                // Publish visualization request markers
                visualization_marker_pub_ =
                        nh.advertise< visualization_msgs::Marker >( GetVisualizationMarkerTopic( nh ), 10 );

                visualization_marker_array_pub_ =
                        nh.advertise< visualization_msgs::MarkerArray >( GetVisualizationMarkerArrayTopic( nh ), 10 );
            }

            void visualizeRope(
                    const ObjectPointSet& rope,
                    const std_msgs::ColorRGBA& color,
                    const std::string& name )
            {
                visualization_msgs::Marker marker;

                marker.type = visualization_msgs::Marker::LINE_STRIP;
                marker.ns = name;
                marker.id = 0;
                marker.scale.x = 0.1;
                marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint( rope );
                marker.colors = std::vector< std_msgs::ColorRGBA >( (size_t)rope.cols(), color );
                visualization_marker_pub_.publish( marker );

                marker.type = visualization_msgs::Marker::SPHERE;
                marker.id = 1;
                marker.scale.x = 0.01;
                visualization_marker_pub_.publish( marker );
            }

            void visualizeCloth(
                    const ObjectPointSet& cloth,
                    const std_msgs::ColorRGBA& color,
                    const std::string& name )
            {
                std::vector< std_msgs::ColorRGBA > colors( (size_t)cloth.cols(), color );

                visualizeCloth( cloth, colors, name );
            }

            void visualizeCloth(
                    const ObjectPointSet& cloth,
                    const std::vector< std_msgs::ColorRGBA >& colors,
                    const std::string& name )
            {
                visualization_msgs::Marker marker;

                marker.type = visualization_msgs::Marker::POINTS;
                marker.ns = name;
                marker.id = 0;
                marker.scale.x = 0.002;
                marker.scale.y = 0.002;
                marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint( cloth );
                marker.colors = colors;

                visualization_marker_pub_.publish( marker );
            }

            void visualizeGripper(
                    const geometry_msgs::Pose& pose,
                    const std_msgs::ColorRGBA& color,
                    const std::string& name )
            {
                // TODO: do
            }

        private:
            ros::Publisher visualization_marker_pub_;
            ros::Publisher visualization_marker_array_pub_;

    };
}

#endif // VISUALIZATION_TOOLS_HPP
