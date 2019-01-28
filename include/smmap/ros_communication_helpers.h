#ifndef ROS_COMMUNICATION_HELPERS_H
#define ROS_COMMUNICATION_HELPERS_H

#include <arc_utilities/dijkstras.hpp>
#include <sdf_tools/sdf.hpp>
#include <smmap_utilities/grippers.h>

#define CHECK_FRAME_NAME(logger, expected, given)                                                       \
    if ((given) != (expected))                                                                          \
    {                                                                                                   \
        ROS_FATAL_STREAM_NAMED((logger), __func__ << " response data in incorrect frame. Expecting '"   \
                               << (expected) << "', got '" << (given) << "'.");                         \
        throw_arc_exception(std::invalid_argument, "Invalid frame name");                               \
    }

namespace smmap
{
    std::vector<smmap_utilities::GripperData> GetGrippersData(ros::NodeHandle& nh);

    smmap_utilities::ObjectPointSet GetObjectInitialConfiguration(ros::NodeHandle& nh);

    std::vector<geometry_msgs::Pose> GetRopeNodeTransforms(ros::NodeHandle& nh);

    smmap_utilities::ObjectPointSet GetCoverPoints(ros::NodeHandle& nh);

    smmap_utilities::ObjectPointSet GetCoverPointNormals(ros::NodeHandle& nh);

    // TODO: replace these out params with something else
    void GetFreeSpaceGraph(
            ros::NodeHandle& nh,
            arc_dijkstras::Graph<Eigen::Vector3d>& free_space_graph,
            std::vector<int64_t>& cover_ind_to_free_space_graph_ind);

    sdf_tools::SignedDistanceField::ConstPtr GetEnvironmentSDF(ros::NodeHandle& nh);
}

#endif // ROS_COMMUNICATION_HELPERS_H
