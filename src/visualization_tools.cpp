#include "smmap/visualization_tools.h"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

bool Visualizer::standard_colors_initialized_ = false;
std_msgs::ColorRGBA Visualizer::red_;
std_msgs::ColorRGBA Visualizer::green_;
std_msgs::ColorRGBA Visualizer::blue_;
std_msgs::ColorRGBA Visualizer::black_;
std_msgs::ColorRGBA Visualizer::magenta_;

void Visualizer::InitializeStandardColors()
{
    red_.r = 1.0;
    red_.g = 0.0;
    red_.b = 0.0;
    red_.a = 1.0;

    green_.r = 0.0;
    green_.g = 1.0;
    green_.b = 0.0;
    green_.a = 1.0;

    blue_.r = 0.0;
    blue_.g = 0.0;
    blue_.b = 1.0;
    blue_.a = 1.0;

    black_.r = 0.0;
    black_.g = 0.0;
    black_.b = 0.0;
    black_.a = 1.0;

    magenta_.r = 1.0f;
    magenta_.g = 0.0f;
    magenta_.b = 1.0f;
    magenta_.a = 1.0f;

    standard_colors_initialized_ = true;
}

std_msgs::ColorRGBA Visualizer::Red()
{
    assert(standard_colors_initialized_);
    return red_;
}

std_msgs::ColorRGBA Visualizer::Green()
{
    assert(standard_colors_initialized_);
    return green_;
}

std_msgs::ColorRGBA Visualizer::Blue()
{
    assert(standard_colors_initialized_);
    return blue_;
}

std_msgs::ColorRGBA Visualizer::Black()
{
    assert(standard_colors_initialized_);
    return black_;
}

std_msgs::ColorRGBA Visualizer::Magenta()
{
    assert(standard_colors_initialized_);
    return magenta_;
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

Visualizer::Visualizer(ros::NodeHandle& nh)
    : world_frame_name_(GetWorldFrameName())
    , gripper_apperture_(GetGripperApperture(nh))
{
    InitializeStandardColors();

    // Publish visualization request markers
    visualization_marker_pub_ =
            nh.advertise<visualization_msgs::Marker>(GetVisualizationMarkerTopic(nh), 3000);

    visualization_marker_array_pub_ =
            nh.advertise<visualization_msgs::MarkerArray>(GetVisualizationMarkerArrayTopic(nh), 3000);
}

void Visualizer::visualizePoints(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d points,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    std::vector<std_msgs::ColorRGBA> colors(points.size(), color);

    visualizePoints(marker_name, points, colors, id);
}

void Visualizer::visualizePoints(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d points,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id) const
{
    visualization_msgs::Marker marker;

    marker.header.frame_id = world_frame_name_;

    marker.type = visualization_msgs::Marker::POINTS;
    marker.ns = marker_name;
    marker.id = id;
    marker.scale.x = 0.005;
    marker.scale.y = 0.005;
    marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(points);
    marker.colors = colors;

    visualization_marker_pub_.publish(marker);
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

    marker.header.frame_id = world_frame_name_;

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

    marker.header.frame_id = world_frame_name_;

    marker.type = visualization_msgs::Marker::POINTS;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.005;
    marker.scale.y = 0.005;
    marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint(cloth);
    marker.colors = colors;

    visualization_marker_pub_.publish(marker);
}

visualization_msgs::MarkerArray::_markers_type Visualizer::createGripperMarker(
        const std::string& marker_name,
        const Eigen::Affine3d& eigen_pose,
        const std_msgs::ColorRGBA& color) const
{
    visualization_msgs::MarkerArray::_markers_type markers;
    markers.resize(2);

    markers[0].header.frame_id = world_frame_name_;

    markers[0].type = visualization_msgs::Marker::CUBE;
    markers[0].ns = marker_name;
    markers[0].id = 0;
    markers[0].scale.x = 0.03;
    markers[0].scale.y = 0.03;
    markers[0].scale.z = 0.01;
    markers[0].pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(eigen_pose * Eigen::Translation3d(0.0, 0.0, gripper_apperture_));
    markers[0].color = color;

    markers[1].header.frame_id = world_frame_name_;

    markers[1].type = visualization_msgs::Marker::CUBE;
    markers[1].ns = marker_name;
    markers[1].id = 1;
    markers[1].scale.x = 0.03;
    markers[1].scale.y = 0.03;
    markers[1].scale.z = 0.01;
    markers[1].pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(eigen_pose * Eigen::Translation3d(0.0, 0.0, -gripper_apperture_));
    markers[1].color = color;

    return markers;
}

void Visualizer::visualizeGripper(
        const std::string& marker_name,
        const Eigen::Affine3d& eigen_pose,
        const std_msgs::ColorRGBA& color) const
{
    visualization_msgs::MarkerArray marker;

    marker.markers = createGripperMarker(marker_name, eigen_pose, color);

    visualization_marker_array_pub_.publish(marker);
}

void Visualizer::visualizeGrippers(
        const std::string& marker_name,
        const EigenHelpers::VectorAffine3d eigen_poses,
        const std_msgs::ColorRGBA& color) const
{
    visualization_msgs::MarkerArray marker;
    marker.markers.reserve(2 * eigen_poses.size());

    for (size_t gripper_ind = 0; gripper_ind < eigen_poses.size(); ++gripper_ind)
    {
        const auto m = createGripperMarker(marker_name, eigen_poses[gripper_ind], color);
        marker.markers.insert(marker.markers.end(), m.begin(), m.end());

        marker.markers[2*gripper_ind].id = (int)(2*gripper_ind);
        marker.markers[2*gripper_ind + 1].id = (int)(2*gripper_ind + 1);
    }

    visualization_marker_array_pub_.publish(marker);
}


void Visualizer::visualizeObjectDelta(
        const std::string& marker_name,
        const ObjectPointSet& current,
        const ObjectPointSet& desired,
        const std_msgs::ColorRGBA& color) const
{
    visualization_msgs::Marker marker;

    marker.header.frame_id = world_frame_name_;

    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points.reserve((size_t)current.cols() * 2);
    marker.colors.reserve((size_t)current.cols() * 2);
    for (ssize_t col = 0; col < current.cols(); col++)
    {
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

    marker.header.frame_id = world_frame_name_;

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
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    assert(start.size() == end.size());

    visualization_msgs::Marker marker;

    marker.header.frame_id = world_frame_name_;

    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.ns = marker_name;
    marker.id = id;
    marker.scale.x = 0.001;

    for (size_t ind = 0; ind < start.size(); ind++)
    {
        marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(start[ind]));
        marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(end[ind]));
        marker.colors.push_back(color);
        marker.colors.push_back(color);
    }

    visualization_marker_pub_.publish(marker);
}

void Visualizer::visualizeLineStrip(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& point_sequence,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    visualization_msgs::Marker marker;

    marker.header.frame_id = world_frame_name_;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = id;
    marker.scale.x = 0.001;

    marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(point_sequence);
    marker.colors = std::vector<std_msgs::ColorRGBA>(marker.points.size(), color);

    visualization_marker_pub_.publish(marker);
}

void Visualizer::visualizeXYZTrajectory(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& point_sequence,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    visualizeLineStrip(marker_name, point_sequence, color, id);
}

void Visualizer::deletePoints(
        const std::string& marker_name,
        const int32_t id) const
{
    assert(false && "There is something wrong with these functions as far as bullet is concerned");
    visualization_msgs::Marker marker;

    marker.header.frame_id = world_frame_name_;
    marker.action = visualization_msgs::Marker::DELETE;

    marker.type = visualization_msgs::Marker::POINTS;
    marker.ns = marker_name;
    marker.id = id;

    visualization_marker_pub_.publish(marker);
}


void Visualizer::deleteXYZTrajectory(
        const std::string& marker_name,
        const int32_t id) const
{
    assert(false && "There is something wrong with these functions as far as bullet is concerned");
    visualization_msgs::Marker marker;

    marker.header.frame_id = world_frame_name_;
    marker.action = visualization_msgs::Marker::DELETE;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = id;

    visualization_marker_pub_.publish(marker);
}
