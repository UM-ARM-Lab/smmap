#include "smmap/gripper_collision_checker.h"
#include "smmap/ros_params.hpp"

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <smmap_msgs/messages.h>

using namespace smmap;
using namespace EigenHelpersConversions;

GripperCollisionChecker::GripperCollisionChecker( ros::NodeHandle& nh )
{
    collision_checker_client_ =
        nh.serviceClient< smmap_msgs::GetGripperCollisionReport >( GetGripperCollisionCheckTopic( nh ) );

    collision_checker_client_.waitForExistence();
}

std::vector< CollisionData > GripperCollisionChecker::gripperCollisionCheck(
        const AllGrippersSinglePose& gripper_poses )
{
    smmap_msgs::GetGripperCollisionReport collision_report_ros;
    collision_report_ros.request.pose =
            VectorAffine3dToVectorGeometryPose( gripper_poses );

    if ( !collision_checker_client_.call( collision_report_ros ) )
    {
        ROS_FATAL_NAMED( "gripper collision check", "Unabled to retrieve gripper collision report." );
    }

    std::vector< CollisionData > collision_report_eigen;
    collision_report_eigen.reserve( gripper_poses.size() );

    for ( size_t gripper_ind = 0; gripper_ind < gripper_poses.size(); gripper_ind++ )
    {
        collision_report_eigen.push_back(
                    CollisionData(
                        EigenHelpersConversions::GeometryPointToEigenVector3d(
                            collision_report_ros.response.gripper_nearest_point_to_obstacle[gripper_ind] ),
                        EigenHelpersConversions::GeometryVector3ToEigenVector3d(
                            collision_report_ros.response.obstacle_surface_normal[gripper_ind] ),
                        collision_report_ros.response.gripper_distance_to_obstacle[gripper_ind] ) );
    }

    return collision_report_eigen;
}
