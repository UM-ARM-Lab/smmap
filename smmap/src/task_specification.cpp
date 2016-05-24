#include <arc_utilities/arc_exceptions.hpp>

#include "smmap/task_specification.h"
#include "smmap/task_specification_implementions.hpp"

using namespace smmap;

////////////////////////////////////////////////////////////////////
// Constructor to initialize objects that all TaskSpecifications share
////////////////////////////////////////////////////////////////////

TaskSpecification::TaskSpecification(ros::NodeHandle& nh)
    : TaskSpecification(nh, Visualizer(nh))
{}

TaskSpecification::TaskSpecification(ros::NodeHandle& nh, Visualizer vis)
    : vis_(vis)
    , object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh)))
    , num_nodes_(object_initial_node_distance_.cols())
{}

////////////////////////////////////////////////////////////////////////////////
// Static builder function
////////////////////////////////////////////////////////////////////////////////

TaskSpecification::Ptr TaskSpecification::MakeTaskSpecification(
        ros::NodeHandle& nh)
{
    TaskType task_type = GetTaskType(nh);
    DeformableType deformable_type = GetDeformableType(nh);

    if (deformable_type == DeformableType::ROPE && task_type == TaskType::CYLINDER_COVERAGE)
    {
        return std::make_shared<RopeCylinderCoverage>(RopeCylinderCoverage(nh));
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::TABLE_COVERAGE)
    {
        return std::make_shared<ClothTableCoverage>(ClothTableCoverage(nh));
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::CYLINDER_COVERAGE)
    {
        return std::make_shared<ClothCylinderCoverage>(ClothCylinderCoverage(nh));
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::COLAB_FOLDING)
    {
        return std::make_shared<ClothColabFolding>(ClothColabFolding(nh));
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::WAFR)
    {
        return std::make_shared<ClothWAFR>(ClothWAFR(nh));
    }
    else
    {
        throw new arc_exceptions::invalid_argument("Invalid task and deformable pair in createErrorFunction(), this should not be possible", __FILE__, __LINE__);
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
    visualizeDeformableObject_impl(vis, marker_name, object_configuration, color);
}

void TaskSpecification::visualizeDeformableObject(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    visualizeDeformableObject_impl(vis, marker_name, object_configuration, colors);
}

double TaskSpecification::calculateError(
        const ObjectPointSet& object_configuration) const
{
    return calculateError_impl(object_configuration);
}

ObjectDeltaAndWeight TaskSpecification::calculateObjectErrorCorrectionDelta(
        const WorldState& world_state) const
{
    return calculateObjectErrorCorrectionDelta_impl(world_state);
}

Eigen::VectorXd TaskSpecification::projectObjectDelta(
        const ObjectPointSet& object_configuration,
        Eigen::VectorXd object_delta) const
{
    return projectObjectDelta_impl(object_configuration, object_delta);
}

/**
 * @brief TaskSpecification::calculateStretchingCorrectionDelta
 * @param object_configuration
 * @return
 */
ObjectDeltaAndWeight TaskSpecification::calculateStretchingCorrectionDelta(
        const ObjectPointSet& object_configuration,
        bool visualize) const
{
    ObjectDeltaAndWeight stretching_correction (num_nodes_ * 3);

    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(object_configuration);

    const double stretching_correction_threshold = getStretchingScalingThreshold();

    EigenHelpers::VectorVector3d start_points;
    EigenHelpers::VectorVector3d end_points;

    for (long first_node = 0; first_node < num_nodes_; first_node++)
    {
        for (long second_node = first_node + 1; second_node < num_nodes_; second_node++)
        {
            const double max_distance = stretching_correction_threshold + object_initial_node_distance_(first_node, second_node);
            if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
            {
                const double node_distance_delta = std::sqrt(node_squared_distance(first_node, second_node)) - object_initial_node_distance_(first_node, second_node);
                // The correction vector points from the first node to the second node,
                // and is half the length of the "extra" distance
                const Eigen::Vector3d correction_vector = 0.5 * node_distance_delta
                        * (object_configuration.block<3, 1>(0, second_node)
                            - object_configuration.block<3, 1>(0, first_node));

                stretching_correction.delta.segment<3>(3 * first_node) += correction_vector;
                stretching_correction.delta.segment<3>(3 * second_node) -= correction_vector;

                stretching_correction.weight(3 * first_node) += 1;
                stretching_correction.weight(3 * first_node + 1) += 1;
                stretching_correction.weight(3 * first_node + 2) += 1;
                stretching_correction.weight(3 * second_node) += 1;
                stretching_correction.weight(3 * second_node + 1) += 1;
                stretching_correction.weight(3 * second_node + 2) += 1;

                if (visualize)
                {
                    start_points.push_back(object_configuration.block<3, 1>(0, first_node));
                    end_points.push_back(object_configuration.block<3, 1>(0, first_node) + correction_vector);

                    start_points.push_back(object_configuration.block<3, 1>(0, second_node));
                    end_points.push_back(object_configuration.block<3, 1>(0, first_node) - correction_vector);
                }
            }
        }
    }

    if (visualize && start_points.size() > 0)
    {
        std_msgs::ColorRGBA blue;
        blue.r = 0.0f;
        blue.g = 0.0f;
        blue.b = 1.0f;
        blue.a = 1.0f;
        vis_.visualizeLines("stretching_lines", start_points, end_points, blue);
    }

    // Normalize the weights so that changing the number of nodes doesn't affect
    // the weights too much; i.e. minimize the effect of the level of discretization
//    stretching_correction.second /= (double)num_nodes_;

    return stretching_correction;
}

/**
 * @brief TaskSpecification::calculateStretchingCorrectionDelta
 * @param world_state
 * @return
 */
ObjectDeltaAndWeight TaskSpecification::calculateStretchingCorrectionDelta(
        const WorldState& world_state,
        bool visualize) const
{
    return calculateStretchingCorrectionDelta(world_state.object_configuration_, visualize);
}

/**
 * @brief TaskSpecification::calculateStretchingError
 * @param object_configuration
 * @return
 */
double TaskSpecification::calculateStretchingError(
        const ObjectPointSet& object_configuration) const
{
    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(object_configuration);

    const double stretching_correction_threshold = getStretchingScalingThreshold();

    ssize_t squared_error = 0;
    #pragma omp parallel for reduction(+ : squared_error)
    for (ssize_t first_node = 0; first_node < num_nodes_; first_node++)
    {
        // A node is never overstretched relative to itself, so we can start at the next node
        int node_overstretches = 0;
        for (ssize_t second_node = 0; second_node < num_nodes_; second_node++)
        {
            if (first_node != second_node)
            {
                const double max_distance = stretching_correction_threshold + object_initial_node_distance_(first_node, second_node);
                if (node_squared_distance(first_node, second_node) > (max_distance * max_distance))
                {
                    node_overstretches++;
                }
            }
        }
        squared_error += node_overstretches * node_overstretches;
    }

    return std::sqrt((double)squared_error) / (double)(num_nodes_ * num_nodes_);
}

/**
 * @brief TaskSpecification::calculateStretchingError
 * @param world_state
 * @return
 */
double TaskSpecification::calculateStretchingError(
        const WorldState& world_state) const
{
    return calculateStretchingError(world_state.object_configuration_);
}

/**
 * @brief TaskSpecification::combineErrorCorrectionAndStretchingCorrection
 * @param error_correction
 * @param stretching_correction
 * @return
 */
// TODO: this probably doesn't belong in this class
ObjectDeltaAndWeight TaskSpecification::combineErrorCorrectionAndStretchingCorrection(
        const ObjectDeltaAndWeight& error_correction,
        const ObjectDeltaAndWeight& stretching_correction) const
{
    ObjectDeltaAndWeight combined(num_nodes_ * 3);

//    std::cout << "Max error:      " << error_correction.second.maxCoeff() << std::endl
//              << "Sum error:      " << error_correction.second.sum() << std::endl
//              << "Max stretching: " << stretching_correction.second.maxCoeff() << std::endl
//              << "Sum stretching: " << stretching_correction.second.sum() << std::endl;


    for (long ind = 0; ind < num_nodes_ * 3; ind += 3)
    {
        const double stretching_importance =
                1.0 - std::exp(-10.0*1e-3 * stretching_correction.weight(ind));

        assert(stretching_importance >= 0.0);
        assert(stretching_importance <= 1.0);

        // Calculate the combined object delta
        combined.delta.segment<3>(ind) =
                stretching_importance * stretching_correction.delta.segment<3>(ind)
                + (1.0 - stretching_importance) * error_correction.delta.segment<3>(ind);

        // Calculate the combined node weights
        combined.weight.segment<3>(ind) =
                stretching_importance * stretching_correction.weight.segment<3>(ind)
                + (1.0 - stretching_importance) * error_correction.weight.segment<3>(ind);
    }

//    combined.first = error_correction.first + stretching_correction.first;
//    combined.second = Eigen::VectorXd::Ones(num_nodes_ * 3);

    // Normalize the weights for later use
    const double combined_normalizer = combined.weight.maxCoeff();
    assert(combined_normalizer > 0);
    combined.weight /= combined_normalizer;

    return combined;
}
