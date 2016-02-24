#include "smmap/visualization_tools.h"

using namespace smmap;

Visualizer::Visualizer( ros::NodeHandle& nh )
{
    // Publish visualization request markers
    visualization_marker_pub_ =
            nh.advertise< visualization_msgs::Marker >( GetVisualizationMarkerTopic( nh ), 10 );

    visualization_marker_array_pub_ =
            nh.advertise< visualization_msgs::MarkerArray >( GetVisualizationMarkerArrayTopic( nh ), 10 );
}

void Visualizer::visualizeRope(
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

void Visualizer::visualizeCloth(
        const ObjectPointSet& cloth,
        const std_msgs::ColorRGBA& color,
        const std::string& name )
{
    std::vector< std_msgs::ColorRGBA > colors( (size_t)cloth.cols(), color );

    visualizeCloth( cloth, colors, name );
}

void Visualizer::visualizeCloth(
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

void Visualizer::visualizeGripper(
        const geometry_msgs::Pose& pose,
        const std_msgs::ColorRGBA& color,
        const std::string& name )
{
    // TODO: do
}


void Visualizer::visualizeObjectDelta( const std::string& marker_name,
                           const ObjectPointSet& current,
                           const ObjectPointSet& desired )
{
    visualization_msgs::Marker marker;
    std_msgs::ColorRGBA color;

    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points.reserve( (size_t)current.cols() * 2 );
    marker.colors.reserve( (size_t)current.cols() * 2 );
    for ( long col = 0; col < current.cols(); col++ )
    {
        color.r = 0;//(1.0 + std::cos( 2*M_PI*(double)col/15.0 )) / 3;
        color.g = 1;//(1.0 + std::cos( 2*M_PI*(double)(col+5)/15.0 )) / 3;
        color.b = 0;//(1.0 + std::cos( 2*M_PI*double(col+10)/15.0 )) / 3;
        color.a = 1;

        marker.points.push_back( EigenHelpersConversions::EigenVector3dToGeometryPoint( current.block< 3, 1 >( 0, col ) ) );
        marker.points.push_back( EigenHelpersConversions::EigenVector3dToGeometryPoint( desired.block< 3, 1 >( 0, col ) ) );
        marker.colors.push_back( color );
        marker.colors.push_back( color );
    }

    visualization_marker_pub_.publish( marker );
}

void Visualizer::visualizeTranslation( const std::string& marker_name,
                           const geometry_msgs::Point& start,
                           const geometry_msgs::Point& end,
                           const std_msgs::ColorRGBA& color )
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points.push_back( start );
    marker.points.push_back( end );
    marker.colors.push_back( color );
    marker.colors.push_back( color );

    visualization_marker_pub_.publish( marker );
}

void Visualizer::visualizeTranslation( const std::string& marker_name,
                           const Eigen::Vector3d& start,
                           const Eigen::Vector3d& end,
                           const std_msgs::ColorRGBA& color )
{
    visualizeTranslation( marker_name,
                          EigenHelpersConversions::EigenVector3dToGeometryPoint( start ),
                          EigenHelpersConversions::EigenVector3dToGeometryPoint( end ),
                          color );
}

void Visualizer::visualizeTranslation( const std::string& marker_name,
                           const Eigen::Affine3d &start,
                           const Eigen::Affine3d &end,
                           const std_msgs::ColorRGBA& color )
{
    Visualizer::visualizeTranslation( marker_name,
                          start.translation(),
                          end.translation(),
                          color );
}

void Visualizer::visualizeLines( const std::string& marker_name,
                     const EigenHelpers::VectorVector3d& start,
                     const EigenHelpers::VectorVector3d& end,
                     const std_msgs::ColorRGBA& color )
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;

    for ( size_t ind = 0; ind < start.size(); ind++ )
    {
        marker.points.push_back( EigenHelpersConversions::EigenVector3dToGeometryPoint( start[ind] ) );
        marker.points.push_back( EigenHelpersConversions::EigenVector3dToGeometryPoint( end[ind] ) );
        marker.colors.push_back( color );
        marker.colors.push_back( color );
    }

    visualization_marker_pub_.publish( marker );
}
