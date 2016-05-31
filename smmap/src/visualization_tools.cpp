#include "smmap/visualization_tools.h"

using namespace smmap;

Visualizer::Visualizer(ros::NodeHandle& nh)
{
    // Publish visualization request markers
    visualization_marker_pub_ =
            nh.advertise<visualization_msgs::Marker>(GetVisualizationMarkerTopic(nh), 10);

    visualization_marker_array_pub_ =
            nh.advertise<visualization_msgs::MarkerArray>(GetVisualizationMarkerArrayTopic(nh), 10);
}

void Visualizer::visualizeRope(
        const std::string& marker_name,
        const ObjectPointSet& rope,
        const std_msgs::ColorRGBA& color) const
{
    std::vector<std_msgs::ColorRGBA> colors((size_t)rope.cols(), color);

    visualizeRope(marker_name, rope, colors);
}

void Visualizer::visualizeRope(
        const std::string& marker_name,
        const ObjectPointSet& rope,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.15;
    marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint(rope);
    marker.colors = colors;
    visualization_marker_pub_.publish(marker);

    marker.type = visualization_msgs::Marker::SPHERE;
    marker.id = 1;
    marker.scale.x = 0.015;
    visualization_marker_pub_.publish(marker);
}


void Visualizer::visualizeCloth(
        const std::string& marker_name,
        const ObjectPointSet& cloth,
        const std_msgs::ColorRGBA& color) const
{
    std::vector<std_msgs::ColorRGBA> colors((size_t)cloth.cols(), color);

    visualizeCloth(marker_name, cloth, colors);
}

void Visualizer::visualizeCloth(
        const std::string& marker_name,
        const ObjectPointSet& cloth,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::POINTS;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.005;
    marker.scale.y = 0.005;
    marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint(cloth);
    marker.colors = colors;

    visualization_marker_pub_.publish(marker);
}

void Visualizer::visualizeGripper(
        const std::string& marker_name,
        const geometry_msgs::Pose& pose,
        const std_msgs::ColorRGBA& color) const
{
    // TODO: do
    (void)marker_name;
    (void)pose;
    (void)color;
    assert(false && "Visualizer::visualizeGripper is not yet implemented.");
}


void Visualizer::visualizeObjectDelta(
        const std::string& marker_name,
        const ObjectPointSet& current,
        const ObjectPointSet& desired) const
{
    visualization_msgs::Marker marker;
    std_msgs::ColorRGBA color;

    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points.reserve((size_t)current.cols() * 2);
    marker.colors.reserve((size_t)current.cols() * 2);
    for (ssize_t col = 0; col < current.cols(); col++)
    {
        color.r = 0;//(1.0 + std::cos(2*M_PI*(double)col/15.0)) / 3;
        color.g = 1;//(1.0 + std::cos(2*M_PI*(double)(col+5)/15.0)) / 3;
        color.b = 0;//(1.0 + std::cos(2*M_PI*double(col+10)/15.0)) / 3;
        color.a = 1;

        marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(current.col(col)));
        marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(desired.col(col)));
        marker.colors.push_back(color);
        marker.colors.push_back(color);
    }

    visualization_marker_pub_.publish(marker);
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const geometry_msgs::Point& start,
        const geometry_msgs::Point& end,
        const std_msgs::ColorRGBA& color) const
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points.push_back(start);
    marker.points.push_back(end);
    marker.colors.push_back(color);
    marker.colors.push_back(color);

    visualization_marker_pub_.publish(marker);
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        const std_msgs::ColorRGBA& color) const
{
    visualizeTranslation(marker_name,
                          EigenHelpersConversions::EigenVector3dToGeometryPoint(start),
                          EigenHelpersConversions::EigenVector3dToGeometryPoint(end),
                          color);
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const Eigen::Affine3d &start,
        const Eigen::Affine3d &end,
        const std_msgs::ColorRGBA& color) const
{
    Visualizer::visualizeTranslation(marker_name,
                          start.translation(),
                          end.translation(),
                          color);
}

void Visualizer::visualizeLines(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& start,
        const EigenHelpers::VectorVector3d& end,
        const std_msgs::ColorRGBA& color) const
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;

    for (size_t ind = 0; ind < start.size(); ind++)
    {
        marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(start[ind]));
        marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(end[ind]));
        marker.colors.push_back(color);
        marker.colors.push_back(color);
    }

    visualization_marker_pub_.publish(marker);
}
