#include "smmap/task_specification_implementions.h"


using namespace smmap;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Colab folding
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ClothColabFolding::ClothColabFolding(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : TaskSpecification(nh, ph, DeformableType::CLOTH, TaskType::CLOTH_COLAB_FOLDING)
    , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
    , point_reflector_(createPointReflector(nh))
    , mirror_map_(createMirrorMap(nh, point_reflector_))
{}

void ClothColabFolding::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeCloth(marker_name, object_configuration, color);
}

void ClothColabFolding::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeCloth(marker_name, object_configuration, colors);
}

double ClothColabFolding::calculateError_impl(
        const WorldState& world_state)
{
    const ObjectPointSet& current_configuration = world_state.object_configuration_;

    double error = 0;
    for (std::map<long, long>::const_iterator ittr = mirror_map_.begin(); ittr != mirror_map_.end(); ++ittr)
    {
        error += (current_configuration.col(ittr->second) -
                   point_reflector_.reflect(current_configuration.col(ittr->first))).norm();
    }

    return error;
}

ObjectDeltaAndWeight ClothColabFolding::calculateObjectErrorCorrectionDelta_impl(
        const WorldState& world_state)
{
    ROS_INFO_NAMED("cloth_colab_folding" , "Finding 'best' cloth delta");
    const ObjectPointSet& object_configuration = world_state.object_configuration_;

    ObjectDeltaAndWeight desired_cloth_delta(object_configuration.cols() * 3);

    long robot_cloth_points_ind = 0;
    for (std::map<long, long>::const_iterator ittr = mirror_map_.begin();
          ittr != mirror_map_.end(); ++ittr, ++robot_cloth_points_ind)
    {
        desired_cloth_delta.delta.segment<3>(ittr->second * 3) =
                point_reflector_.reflect(object_configuration.block<3, 1>(0, ittr->first))
                - object_configuration.block<3, 1>(0, ittr->second);

        const double weight = desired_cloth_delta.delta.segment<3>(ittr->second * 3).norm();
        desired_cloth_delta.weight(ittr->second * 3) = weight;
        desired_cloth_delta.weight(ittr->second * 3 + 1) = weight;
        desired_cloth_delta.weight(ittr->second * 3 + 2) = weight;
    }

    return desired_cloth_delta;
}

std::vector<ssize_t> ClothColabFolding::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

PointReflector ClothColabFolding::createPointReflector(ros::NodeHandle& nh)
{
    ROS_INFO_NAMED("cloth_colab_folding_task" , "Getting mirror line");

    // Get the initial configuration of the object
    ros::ServiceClient mirror_line_client =
        nh.serviceClient<deformable_manipulation_msgs::GetMirrorLine>(GetMirrorLineTopic(nh));

    mirror_line_client.waitForExistence();

    deformable_manipulation_msgs::GetMirrorLine mirror_line_data;
    mirror_line_client.call(mirror_line_data);

    return PointReflector(mirror_line_data.response.mid_x,
                           mirror_line_data.response.min_y,
                           mirror_line_data.response.max_y);
}

std::map<long, long> ClothColabFolding::createMirrorMap(ros::NodeHandle& nh, const PointReflector& point_reflector)
{
    ObjectPointSet object_initial_configuration =
        GetObjectInitialConfiguration(nh);

    ROS_INFO_NAMED("cloth_colab_folding", "Finding point correspondences");

    std::map<long, long> mirror_map;
    for (long node_ind = 0; node_ind < object_initial_configuration.cols(); node_ind++)
    {
        // for every node on one side of the mirror line, find the closest match on the other side
        // Note that nodes that have an x value > than mid_x are on the manual gripper side
        if (object_initial_configuration(0, node_ind) > point_reflector.get_mid_x())
        {
            long mirror_ind = ClosestPointInSet(object_initial_configuration,
                    point_reflector.reflect(object_initial_configuration.col(node_ind)));

            mirror_map[node_ind] = mirror_ind;
        }
    }

    return mirror_map;
}

bool ClothColabFolding::taskDone_impl(
        const WorldState& world_state)
{
    (void)world_state;
    return false;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rope cylinder coverage
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RopeCylinderCoverage::RopeCylinderCoverage(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : DirectCoverageTask(nh, ph, DeformableType::ROPE, TaskType::ROPE_CYLINDER_COVERAGE)
    , neighbours_(num_nodes_)
{}

void RopeCylinderCoverage::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeRope(marker_name, object_configuration, color);
}

void RopeCylinderCoverage::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeRope(marker_name, object_configuration, colors);
}

std::vector<ssize_t> RopeCylinderCoverage::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool RopeCylinderCoverage::taskDone_impl(
        const WorldState& world_state)
{
    return calculateError(world_state) < error_threshold_task_done_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cloth cylinder coverage - unused
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ClothCylinderCoverage::ClothCylinderCoverage(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : DistanceBasedCorrespondencesTask(nh, ph, DeformableType::CLOTH, TaskType::CLOTH_CYLINDER_COVERAGE)
    , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
{
    assert(false && "This task has not been looked at in over a year");
}

void ClothCylinderCoverage::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeCloth(marker_name, object_configuration, color);
}

void ClothCylinderCoverage::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeCloth(marker_name, object_configuration, colors);
}

std::vector<ssize_t> ClothCylinderCoverage::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool ClothCylinderCoverage::taskDone_impl(
        const WorldState& world_state)
{
    return calculateError(world_state) < error_threshold_task_done_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cloth table coverage
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ClothTableCoverage::ClothTableCoverage(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : DirectCoverageTask(nh, ph, DeformableType::CLOTH, TaskType::CLOTH_TABLE_COVERAGE)
    , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
{}

void ClothTableCoverage::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeCloth(marker_name, object_configuration, color);
}

void ClothTableCoverage::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeCloth(marker_name, object_configuration, colors);
}

std::vector<ssize_t> ClothTableCoverage::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool ClothTableCoverage::taskDone_impl(
        const WorldState& world_state)
{
    return calculateError(world_state) < error_threshold_task_done_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cloth WAFR coverage
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ClothWAFR::ClothWAFR(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : DistanceBasedCorrespondencesTask(nh, ph, DeformableType::CLOTH, TaskType::CLOTH_WAFR)
    , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
{}

void ClothWAFR::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeCloth(marker_name, object_configuration, color);
}

void ClothWAFR::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeCloth(marker_name, object_configuration, colors);
}

std::vector<ssize_t> ClothWAFR::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool ClothWAFR::taskDone_impl(
        const WorldState& world_state)
{
    assert(false && "This task needs to some revisions before it is used.");
    (void)world_state;
    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cloth wall
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ClothWall::ClothWall(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : DistanceBasedCorrespondencesTask(nh, ph, DeformableType::CLOTH, TaskType::CLOTH_WALL)
    , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
{}

void ClothWall::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeCloth(marker_name, object_configuration, color);
}

void ClothWall::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeCloth(marker_name, object_configuration, colors);
}

std::vector<ssize_t> ClothWall::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool ClothWall::taskDone_impl(
        const WorldState& world_state)
{
    assert(false && "This task needs to some revisions before it is used.");
    (void)world_state;
    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cloth Single Pole
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ClothSinglePole::ClothSinglePole(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : DistanceBasedCorrespondencesTask(nh, ph, DeformableType::CLOTH, TaskType::CLOTH_SINGLE_POLE)
    , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
{}

void ClothSinglePole::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeCloth(marker_name, object_configuration, color);
}

void ClothSinglePole::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeCloth(marker_name, object_configuration, colors);
}

std::vector<ssize_t> ClothSinglePole::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool ClothSinglePole::taskDone_impl(
        const WorldState& world_state)
{
    return calculateError(world_state) < error_threshold_task_done_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Cloth double slit
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ClothDoubleSlit::ClothDoubleSlit(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : DistanceBasedCorrespondencesTask(nh, ph, DeformableType::CLOTH, TaskType::CLOTH_DOUBLE_SLIT)
    , neighbours_(num_nodes_, GetClothNumControlPointsX(nh))
{}

void ClothDoubleSlit::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeCloth(marker_name, object_configuration, color);
}

void ClothDoubleSlit::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeCloth(marker_name, object_configuration, colors);
}

std::vector<ssize_t> ClothDoubleSlit::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool ClothDoubleSlit::taskDone_impl(
        const WorldState& world_state)
{
    return calculateError(world_state) < error_threshold_task_done_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rope maze
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RopeMaze::RopeMaze(ros::NodeHandle& nh, ros::NodeHandle& ph)
    : FixedCorrespondencesTask(nh, ph, DeformableType::ROPE, TaskType::ROPE_MAZE)
    , neighbours_(num_nodes_)
{
    assert(num_nodes_ == num_cover_points_);

    correspondences_internal_fixed_.clear();
    correspondences_internal_fixed_.reserve(num_nodes_);

    for (ssize_t idx = 0; idx < num_nodes_; ++idx)
    {
        correspondences_internal_fixed_.push_back({idx});
    }
}

void RopeMaze::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    vis.visualizeRope(marker_name, object_configuration, color);
}

void RopeMaze::visualizeDeformableObject_impl(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    vis.visualizeRope(marker_name, object_configuration, colors);
}

std::vector<ssize_t> RopeMaze::getNodeNeighbours_impl(const ssize_t node) const
{
    return neighbours_.getNodeNeighbours(node);
}

bool RopeMaze::taskDone_impl(
        const WorldState& world_state)
{
    return calculateError(world_state) < error_threshold_task_done_;
}
