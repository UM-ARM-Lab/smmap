#include "smmap/task_specification.h"
#include "smmap/task_specification_implementions.hpp"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Static builder function
////////////////////////////////////////////////////////////////////////////////

TaskSpecification::Ptr TaskSpecification::MakeTaskSpecification(
        ros::NodeHandle& nh )
{
    TaskType task_type = GetTaskType( nh );
    DeformableType deformable_type = GetDeformableType( nh );

    if ( deformable_type == DeformableType::ROPE && task_type == TaskType::COVERAGE )
    {
        return std::make_shared< RopeCoverage >( RopeCoverage( nh ) );
    }
    else if ( deformable_type == DeformableType::CLOTH && task_type == TaskType::COVERAGE )
    {
        return std::make_shared< ClothTableCoverage >( ClothTableCoverage( nh ) );
    }
    else if ( deformable_type == DeformableType::CLOTH && task_type == TaskType::COLAB_FOLDING )
    {
        return std::make_shared< ClothColabFolding >( ClothColabFolding( nh ) );
    }
    else
    {
        throw new std::invalid_argument( "Invalid task and deformable pair in createErrorFunction(), this should not be possible" );
    }
}

////////////////////////////////////////////////////////////////////////////////
// Virtual function wrappers
////////////////////////////////////////////////////////////////////////////////

double TaskSpecification::getDeformability() const
{
    return getDeformability_impl();
}

double TaskSpecification::getCollisionScalingFactor() const
{
    return getCollisionScalingFactor_impl();
}

double TaskSpecification::getStretchingScalingThreshold() const
{
    return getStretchingScalingThreshold_impl();
}

double TaskSpecification::maxTime() const
{
    return maxTime_impl();
}

void TaskSpecification::visualizeDeformableObject(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    visualizeDeformableObject_impl( vis, marker_name, object_configuration, color );
}

double TaskSpecification::calculateError(
        const ObjectPointSet& object_configuration ) const
{
    return calculateError_impl( object_configuration );
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> TaskSpecification::calculateObjectDesiredDelta(
        const WorldState& world_state ) const
{
    return calculateObjectDesiredDelta_impl( world_state );
}
