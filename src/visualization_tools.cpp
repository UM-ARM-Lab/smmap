#include "smmap/visualization_tools.h"

#include <thread>
#include <std_srvs/Empty.h>

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
std_msgs::ColorRGBA Visualizer::yellow_;
std_msgs::ColorRGBA Visualizer::cyan_;
std_msgs::ColorRGBA Visualizer::white_;
std_msgs::ColorRGBA Visualizer::silver_;
std_msgs::ColorRGBA Visualizer::coral_;
std_msgs::ColorRGBA Visualizer::olive_;
std_msgs::ColorRGBA Visualizer::orange_;

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

    yellow_.r = 1.0f;
    yellow_.g = 1.0f;
    yellow_.b = 0.0f;
    yellow_.a = 1.0f;

    cyan_.r = 0.0f;
    cyan_.g = 1.0f;
    cyan_.b = 1.0f;
    cyan_.a = 1.0f;

    white_.r = 1.0f;
    white_.g = 1.0f;
    white_.b = 1.0f;
    white_.a = 1.0f;

    silver_.r = 0.75f;
    silver_.g = 0.75f;
    silver_.b = 0.75f;
    silver_.a = 1.0f;

    coral_.r = 0.8f;
    coral_.g = 0.36f;
    coral_.b = 0.27f;
    coral_.a = 1.0f;

    olive_.r = 0.31f;
    olive_.g = 0.31f;
    olive_.b = 0.18f;
    olive_.a = 1.0f;

    orange_.r = 0.8f;
    orange_.g = 0.2f;
    orange_.b = 0.2f;
    orange_.a = 1.0f;

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

std_msgs::ColorRGBA Visualizer::Yellow()
{
    assert(standard_colors_initialized_);
    return yellow_;
}

std_msgs::ColorRGBA Visualizer::Cyan()
{
    assert(standard_colors_initialized_);
    return cyan_;
}

std_msgs::ColorRGBA Visualizer::White()
{
    assert(standard_colors_initialized_);
    return white_;
}

std_msgs::ColorRGBA Visualizer::Silver()
{
    assert(standard_colors_initialized_);
    return silver_;
}

std_msgs::ColorRGBA Visualizer::Coral()
{
    assert(standard_colors_initialized_);
    return coral_;
}

std_msgs::ColorRGBA Visualizer::Olive()
{
    assert(standard_colors_initialized_);
    return olive_;
}

std_msgs::ColorRGBA Visualizer::Orange()
{
    assert(standard_colors_initialized_);
    return orange_;
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

Visualizer::Visualizer(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph)
    : Visualizer(
          nh,
          ph,
          GetVisualizationMarkerTopic(nh),
          GetVisualizationMarkerArrayTopic(nh))
{}

Visualizer::Visualizer(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const std::string& marker_topic,
        const std::string& marker_array_topic)
    : disable_all_visualizations_(GetDisableAllVisualizations(ph))
    , clear_markers_srv_(nh.serviceClient<std_srvs::Empty>(GetClearVisualizationsTopic(nh), true))
    , world_frame_name_(GetWorldFrameName())
    , gripper_apperture_(GetGripperApperture(nh))
{
    InitializeStandardColors();

    if (!disable_all_visualizations_)
    {
        clear_markers_srv_.waitForExistence();

        // Publish visualization request markers
        visualization_marker_pub_        = nh.advertise<visualization_msgs::Marker>(marker_topic, 100);
        visualization_marker_vector_pub_ = nh.advertise<visualization_msgs::MarkerArray>(marker_array_topic, 100);
    }
}

void Visualizer::clearVisualizationsBullet() const
{
    if (!disable_all_visualizations_)
    {
        std_srvs::Empty srv_data;

        if (!clear_markers_srv_.call(srv_data))
        {
            ROS_ERROR_NAMED("visualizer", "Unable to clear visualization data");
        }
    }
}

void Visualizer::deleteObjects(
        const std::string& marker_name,
        const int32_t start_id,
        const int32_t end_id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.action = visualization_msgs::Marker::DELETE;
        marker.ns = marker_name;

        for (int32_t id = start_id; id < end_id; ++id)
        {
            marker.id = id;
            marker.header.stamp = ros::Time::now();
            visualization_marker_pub_.publish(marker);

            if (id % 100 == 0)
            {
                ros::spinOnce();
                std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
            }
        }

        ros::spinOnce();
        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
    }
}

void Visualizer::visualizePoints(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const std_msgs::ColorRGBA& color,
        const int32_t id, const double scale) const
{
    if (!disable_all_visualizations_)
    {
        std::vector<std_msgs::ColorRGBA> colors(points.size(), color);

        visualizePoints(marker_name, points, colors, id, scale);
    }
}

void Visualizer::visualizePoints(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id,
        const double scale) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::POINTS;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = scale;
        marker.scale.y = scale;
        marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(points);
        marker.colors = colors;

        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

void Visualizer::visualizeCubes(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& points,
        const Eigen::Vector3d& scale,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::CUBE_LIST;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale = EigenHelpersConversions::EigenVector3dToGeometryVector3(scale);
        marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(points);
        marker.color = color;

        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

void Visualizer::visualizeRope(
        const std::string& marker_name,
        const ObjectPointSet& rope,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors((size_t)rope.cols(), color);
        visualizeRope(marker_name, rope, colors, id);
    }
}

void Visualizer::visualizeRope(
        const std::string& marker_name,
        const ObjectPointSet& rope,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.005;
        marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint(rope);
        marker.colors = colors;
        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);

        marker.type = visualization_msgs::Marker::SPHERE;
        marker.id = id + 1;
        marker.scale.x = 0.015;
        marker.scale.y = 0.015;
        marker.scale.z = 0.015;
        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

void Visualizer::visualizeCloth(
        const std::string& marker_name,
        const ObjectPointSet& cloth,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        const std::vector<std_msgs::ColorRGBA> colors((size_t)cloth.cols(), color);
        visualizeCloth(marker_name, cloth, colors, id);
    }
}

void Visualizer::visualizeCloth(
        const std::string& marker_name,
        const ObjectPointSet& cloth,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::POINTS;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.005;
        marker.scale.y = 0.005;
        marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint(cloth);
        marker.colors = colors;

        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

visualization_msgs::MarkerArray::_markers_type Visualizer::createGripperMarker(
        const std::string& marker_name,
        const Eigen::Affine3d& eigen_pose,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    visualization_msgs::MarkerArray::_markers_type markers;
    markers.resize(2);

    markers[0].header.frame_id = world_frame_name_;

    markers[0].type = visualization_msgs::Marker::CUBE;
    markers[0].ns = marker_name;
    markers[0].id = id;
    //markers[0].scale.x = 0.03;
    markers[0].scale.x = 0.01;
    markers[0].scale.y = 0.03;
    markers[0].scale.z = 0.01;
    markers[0].pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(eigen_pose * Eigen::Translation3d(0.0, 0.0, gripper_apperture_));
    markers[0].color = color;

    markers[1].header.frame_id = world_frame_name_;

    markers[1].type = visualization_msgs::Marker::CUBE;
    markers[1].ns = marker_name;
    markers[1].id = id + 1;
//    markers[1].scale.x = 0.03;
    markers[0].scale.x = 0.01;
    markers[1].scale.y = 0.03;
    markers[1].scale.z = 0.01;
    markers[1].pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(eigen_pose * Eigen::Translation3d(0.0, 0.0, -gripper_apperture_));
    markers[1].color = color;

    markers[0].header.stamp = ros::Time::now();
    markers[1].header.stamp = markers[0].header.stamp;
    return markers;
}

void Visualizer::visualizeGripper(
        const std::string& marker_name,
        const Eigen::Affine3d& eigen_pose,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::MarkerArray marker;
        marker.markers = createGripperMarker(marker_name, eigen_pose, color, id);
        visualization_marker_pub_.publish(marker.markers[0]);
        visualization_marker_pub_.publish(marker.markers[1]);
    }
}

void Visualizer::visualizeGrippers(
        const std::string& marker_name,
        const EigenHelpers::VectorAffine3d eigen_poses,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        for (size_t gripper_ind = 0; gripper_ind < eigen_poses.size(); ++gripper_ind)
        {
            visualizeGripper(marker_name, eigen_poses[gripper_ind], color, id + (int32_t)(2*gripper_ind + 1));
        }
    }
}

void Visualizer::visualizeObjectDelta(
        const std::string& marker_name,
        const ObjectPointSet& current,
        const ObjectPointSet& desired,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.001;
        marker.points.reserve((size_t)current.cols() * 2);
        marker.colors.reserve((size_t)current.cols() * 2);
        for (ssize_t col = 0; col < current.cols(); col++)
        {
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(current.col(col)));
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(desired.col(col)));
            marker.colors.push_back(color);
            marker.colors.push_back(color);
        }

        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const geometry_msgs::Point& start,
        const geometry_msgs::Point& end,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = 0.01;
        marker.points.push_back(start);
        marker.points.push_back(end);
        marker.colors.push_back(color);
        marker.colors.push_back(color);

        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& end,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualizeTranslation(
                    marker_name,
                    EigenHelpersConversions::EigenVector3dToGeometryPoint(start),
                    EigenHelpersConversions::EigenVector3dToGeometryPoint(end),
                    color,
                    id);
    }
}

void Visualizer::visualizeTranslation(
        const std::string& marker_name,
        const Eigen::Affine3d &start,
        const Eigen::Affine3d &end,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        Visualizer::visualizeTranslation(
                    marker_name,
                    start.translation(),
                    end.translation(),
                    color,
                    id);
    }
}

void Visualizer::visualizeLines(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& start,
        const EigenHelpers::VectorVector3d& end,
        const std_msgs::ColorRGBA& color,
        const int32_t id,
        const double scale) const
{
    if (!disable_all_visualizations_)
    {
        assert(start.size() == end.size());

        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = scale;

        for (size_t ind = 0; ind < start.size(); ind++)
        {
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(start[ind]));
            marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(end[ind]));
            marker.colors.push_back(color);
            marker.colors.push_back(color);
        }

        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

void Visualizer::visualizeLineStrip(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& point_sequence,
        const std_msgs::ColorRGBA& color,
        const int32_t id,
        const double scale) const
{
    if (!disable_all_visualizations_)
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = world_frame_name_;

        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.ns = marker_name;
        marker.id = id;
        marker.scale.x = scale;

        marker.points = EigenHelpersConversions::VectorEigenVector3dToVectorGeometryPoint(point_sequence);
        marker.colors = std::vector<std_msgs::ColorRGBA>(marker.points.size(), color);

        marker.header.stamp = ros::Time::now();
        visualization_marker_pub_.publish(marker);
    }
}

void Visualizer::visualizeXYZTrajectory(
        const std::string& marker_name,
        const EigenHelpers::VectorVector3d& point_sequence,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    if (!disable_all_visualizations_)
    {
        visualizeLineStrip(marker_name, point_sequence, color, id);
    }
}
