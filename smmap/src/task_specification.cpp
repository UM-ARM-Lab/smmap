#include <arc_utilities/arc_exceptions.hpp>

#include "smmap/task_specification.h"
#include "smmap/task_specification_implementions.hpp"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Static helper functions - could be private given how they are
// used but making public as they are static - probably should be moved out of class
////////////////////////////////////////////////////////////////////////////////

double TaskSpecification::CalculateErrorWithTheshold(
        const ObjectPointSet& target_points,
        const ObjectPointSet& deformable_object,
        const double minimum_threshold)
{
    Eigen::VectorXd error(target_points.cols());

    // for every cover point, find the nearest deformable object point
    #pragma omp parallel for
    for (ssize_t target_ind = 0; target_ind < target_points.cols(); target_ind++)
    {
        const Eigen::Vector3d& target_point = target_points.col(target_ind);

        // find the closest deformable object point
        double min_dist_squared = std::numeric_limits<double>::infinity();
        for (ssize_t deformable_ind = 0; deformable_ind < deformable_object.cols(); deformable_ind++)
        {
            const Eigen::Vector3d& deformable_point = deformable_object.col(deformable_ind);
            const double new_dist_squared = (target_point - deformable_point).squaredNorm();
            min_dist_squared = std::min(new_dist_squared, min_dist_squared);
        }

        if (std::sqrt(min_dist_squared) >= minimum_threshold)
        {
            error(target_ind) = std::sqrt(min_dist_squared);
        }
        else
        {
            error(target_ind) = 0;
        }
    }

    return error.sum();
}

ObjectDeltaAndWeight TaskSpecification::CalculateObjectErrorCorrectionDeltaWithThreshold(
        const ObjectPointSet& target_points,
        const ObjectPointSet& deformable_object,
        const double minimum_threshold)
{
    const ssize_t num_nodes = deformable_object.cols();
    ObjectDeltaAndWeight desired_object_delta(num_nodes * 3);

    // for every target point, find the nearest deformable object point
    for (ssize_t target_ind = 0; target_ind < target_points.cols(); target_ind++)
    {
        const Eigen::Vector3d& target_point = target_points.col(target_ind);

        // find the closest deformable object point
        ssize_t min_ind = -1;
        double min_dist_squared = std::numeric_limits<double>::infinity();
        for (ssize_t deformable_ind = 0; deformable_ind < num_nodes; deformable_ind++)
        {
            const Eigen::Vector3d& deformable_point = deformable_object.col(deformable_ind);
            const double new_dist_squared = (target_point - deformable_point).squaredNorm();
            if (new_dist_squared < min_dist_squared)
            {
                min_dist_squared = new_dist_squared;
                min_ind = deformable_ind;
            }
        }

        const double min_dist = std::sqrt(min_dist_squared);
        if (min_dist > minimum_threshold)
        {
            desired_object_delta.delta.segment<3>(min_ind * 3) =
                    desired_object_delta.delta.segment<3>(min_ind * 3)
                    + (target_point - deformable_object.col(min_ind));

            const double weight = std::max(desired_object_delta.weight(min_ind * 3), min_dist);
            desired_object_delta.weight(min_ind * 3) = weight;
            desired_object_delta.weight(min_ind * 3 + 1) = weight;
            desired_object_delta.weight(min_ind * 3 + 2) = weight;
        }
    }

    return desired_object_delta;
}

////////////////////////////////////////////////////////////////////
// Constructor to initialize objects that all TaskSpecifications share
////////////////////////////////////////////////////////////////////

TaskSpecification::TaskSpecification(ros::NodeHandle& nh, DeformableType deformable_type, TaskType task_type)
    : TaskSpecification(nh, Visualizer(nh), deformable_type, task_type)
{}

TaskSpecification::TaskSpecification(ros::NodeHandle& nh, Visualizer vis,DeformableType deformable_type, TaskType task_type)
    : deformable_type_(deformable_type)
    , task_type_(task_type)
    , vis_(vis)
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
        return std::make_shared<RopeCylinderCoverage>(nh);
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::TABLE_COVERAGE)
    {
        return std::make_shared<ClothTableCoverage>(nh);
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::CYLINDER_COVERAGE)
    {
        return std::make_shared<ClothCylinderCoverage>(nh);
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::COLAB_FOLDING)
    {
        return std::make_shared<ClothColabFolding>(nh);
    }
    else if (deformable_type == DeformableType::CLOTH && task_type == TaskType::WAFR)
    {
        return std::make_shared<ClothWAFR>(nh);
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

    for (ssize_t first_node = 0; first_node < num_nodes_; first_node++)
    {
        for (ssize_t second_node = first_node + 1; second_node < num_nodes_; second_node++)
        {
            const double max_distance = stretching_correction_threshold + object_initial_node_distance_(first_node, second_node);
            if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
            {
                const double node_distance_delta = std::sqrt(node_squared_distance(first_node, second_node)) - object_initial_node_distance_(first_node, second_node);
                assert(node_distance_delta > stretching_correction_threshold);
                // The correction vector points from the first node to the second node,
                // and is half the length of the "extra" distance
                const Eigen::Vector3d correction_vector = 0.5 * node_distance_delta
                        * (object_configuration.block<3, 1>(0, second_node)
                            - object_configuration.block<3, 1>(0, first_node));

                stretching_correction.delta.segment<3>(3 * first_node) += correction_vector;
                stretching_correction.delta.segment<3>(3 * second_node) -= correction_vector;

                // Set the weight to be the stretch distance of the worst offender
                const double first_node_max_stretch = std::max(stretching_correction.weight(3 * first_node), node_distance_delta);
                stretching_correction.weight(3 * first_node) = first_node_max_stretch;
                stretching_correction.weight(3 * first_node + 1) = first_node_max_stretch;
                stretching_correction.weight(3 * first_node + 2) = first_node_max_stretch;

                // Set the weight to be the stretch distance of the worst offender
                const double second_node_max_stretch = std::max(stretching_correction.weight(3 * second_node), node_distance_delta);
                stretching_correction.weight(3 * second_node) = second_node_max_stretch;
                stretching_correction.weight(3 * second_node + 1) = second_node_max_stretch;
                stretching_correction.weight(3 * second_node + 2) = second_node_max_stretch;

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
        vis_.visualizeLines("stretching_lines", start_points, end_points, Visualizer::Blue());
    }

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

//    std::cout << "Max error:      " << error_correction.weight.maxCoeff() << std::endl
//              << "Avg error:      " << error_correction.weight.sum() / (double)(num_nodes_ * 3) << std::endl
//              << "Sum error:      " << error_correction.weight.sum() << std::endl
//              << "Max stretching: " << stretching_correction.weight.maxCoeff() << std::endl
//              << "Avg stretching: " << stretching_correction.weight.sum() / (double)(num_nodes_ * 3) << std::endl
//              << "Sum stretching: " << stretching_correction.weight.sum() << std::endl;
//    std::cout << std::endl;

    for (ssize_t ind = 0; ind < num_nodes_ * 3; ind += 3)
    {
        const Eigen::Vector3d error_correction_perpendicular =
                EigenHelpers::VectorRejection(stretching_correction.delta.segment<3>(ind),
                                               error_correction.delta.segment<3>(ind) );

        combined.delta.segment<3>(ind) = stretching_correction.delta.segment<3>(ind) + error_correction_perpendicular;
        combined.weight.segment<3>(ind) = stretching_correction.weight.segment<3>(ind) + error_correction.weight.segment<3>(ind);
    }

    // Normalize the weights for later use
    const double combined_normalizer = combined.weight.maxCoeff();
    assert(combined_normalizer > 0);
    combined.weight /= combined_normalizer;

    return combined;
}
