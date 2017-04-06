#include <arc_utilities/arc_exceptions.hpp>
#include <arc_utilities/log.hpp>
#include <deformable_manipulation_msgs/messages.h>

#include "smmap/test_specification.h"
//#include "smmap/task_specification_implementions.hpp"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Static helper functions - could be private given how they are
// used but making public as they are static - probably should be moved out of class
////////////////////////////////////////////////////////////////////////////////


// This function should calculate error at each step
// error = norm(p_Delta_real-p_Delta_model)
// p_Delta_real = p_current-p_last
// p_Delta_model = J*q_Delta_last; The result could be obtained from calculateDesiredDirection
// Also stored as (Last)first_step_desired_motion_;
// It is called by CalculateError_impl in CalculateError, minimum_threshold depend on test
double CalculateErrorWithTheshold(
        const ObjectPointSet& real_delta_p,
        ObjectDeltaAndWeight& model_delta_p,
        const double minimum_threshold)
{
    Eigen::VectorXd error(real_delta_p);
    const Eigen::VectorXd& model_delta_p_value = model_delta_p.delta;

    // for every cover point, find the nearest deformable object point
    #pragma omp parallel for
    for (ssize_t real_ind = 0; real_ind < real_delta_p.cols(); ++real_ind)
    {
        const Eigen::Vector3d& real_point = real_delta_p.col(real_ind);
        const Eigen::Vector3d& model_point = model_delta_p_value.segment<3>(real_ind*3);
        const double point_error = (real_point-model_point).squaredNorm();

        if (std::sqrt(point_error) >= minimum_threshold)
        { error(real_ind) = std::sqrt(point_error); }
        else{ error(real_ind) = 0; }
    }

    return error.sum();
}


////////////////////////////////////////////////////////////////////
// Constructor to initialize objects that all TaskSpecifications share
////////////////////////////////////////////////////////////////////

TestSpecification::TestSpecification(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type)
    : TestSpecification(nh, Visualizer(nh), deformable_type, task_type)
{}

TestSpecification::TestSpecification(ros::NodeHandle& nh, Visualizer vis, const DeformableType deformable_type, const TaskType task_type)
    : first_step_calculated_(false)
    , deformable_type_(deformable_type)
    , task_type_(task_type)
    , is_dijkstras_type_task_(false)
    , nh_(nh)
    , vis_(vis)
    , object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh)))
    , num_nodes_(object_initial_node_distance_.cols())
    , grippers_data_(GetGrippersData(nh))
{}


////////////////////////////////////////////////////////////////////////////////
// Static builder function
////////////////////////////////////////////////////////////////////////////////

TestSpecification::Ptr TestSpecification::MakeTaskSpecification(ros::NodeHandle& nh)
{
    const TaskType task_type = GetTaskType(nh);

#warning "More test case to be declared in MakeTaskSpecification function for test_specification class"
    // TODO: DEFINE MORE SUB CLASS FOR SPECIFIC TEST CASE DESING
    switch (task_type)
    {
    /*
        case TaskType::ROPE_CYLINDER_COVERAGE:
            return std::make_shared<RopeCylinderCoverage>(nh);

        case TaskType::CLOTH_TABLE_COVERAGE:
            return std::make_shared<ClothTableCoverage>(nh);

        case TaskType::CLOTH_CYLINDER_COVERAGE:
            return std::make_shared<ClothCylinderCoverage>(nh);

        case TaskType::CLOTH_COLAB_FOLDING:
            return std::make_shared<ClothColabFolding>(nh);

        case TaskType::CLOTH_WAFR:
            return std::make_shared<ClothWAFR>(nh);

        case TaskType::CLOTH_SINGLE_POLE:
            return std::make_shared<ClothSinglePole>(nh);

        case TaskType::CLOTH_WALL:
            return std::make_shared<ClothWall>(nh);

        case TaskType::CLOTH_DOUBLE_SLIT:
            return std::make_shared<ClothDoubleSlit>(nh);

        case TaskType::ROPE_MAZE:
            return std::make_shared<RopeMaze>(nh);

        default:
            throw_arc_exception(std::invalid_argument, "Invalid task type in MakeTaskSpecification(), this should not be possible");
            return nullptr;
    */
    }
}


////////////////////////////////////////////////////////////////////////////////
// Virtual function wrappers
////////////////////////////////////////////////////////////////////////////////

double TestSpecification::defaultDeformability() const
{
    return deformability_impl();
}

double TestSpecification::collisionScalingFactor() const
{
    return collisionScalingFactor_impl();
}

double TestSpecification::stretchingScalingThreshold() const
{
    return stretchingScalingThreshold_impl();
}

double TestSpecification::maxTime() const
{
    return maxTime_impl();
}

//////////// Mengyao: Initialize delta_q ///////////////////////////

void TestSpecification::initializeGripperDelta(ros::NodeHandle& nh) const
{
    return initializeGripperDelta_impl(nh);
}

void TestSpecification::visualizeDeformableObject(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color) const
{
    visualizeDeformableObject_impl(vis, marker_name, object_configuration, color);
}

void TestSpecification::visualizeDeformableObject(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors) const
{
    visualizeDeformableObject_impl(vis, marker_name, object_configuration, colors);
}

///////////////////////////////////////////////////////////////////////////////////
// Can define specific error here, otherwise, could just use CalculateErrorWithTheshold
///////////////////////////////////////////////////////////////////////////////////

double TestSpecification::calculateError(
        const ObjectPointSet& real_delta_p,
        ObjectDeltaAndWeight& model_delta_p) const
{
    return calculateError_impl(real_delta_p, model_delta_p);
}

///////////////////////////////////////////////////////////////////////////////////
// Object Error is planned one, or set by q (gripper motion), should implement one
// That can map delta_q to delta_p, specifically, only need the velocity of start and end
//////////////////////////////////////////////////////////////////
ObjectDeltaAndWeight TestSpecification::calculateObjectErrorCorrectionDelta(
        const WorldState& world_state) const
{
    return calculateObjectErrorCorrectionDelta_impl(world_state);
}

// COPY DIRECTLY FROM TASKSPECIFICATION
/**
 * @brief TaskSpecification::calculateStretchingCorrectionDelta
 * @param object_configuration
 * @return
 */
ObjectDeltaAndWeight TestSpecification::calculateStretchingCorrectionDelta(
        const ObjectPointSet& object_configuration,
        bool visualize) const
{
    ObjectDeltaAndWeight stretching_correction (num_nodes_ * 3);

    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(object_configuration);

    const double stretching_correction_threshold = stretchingScalingThreshold();

    EigenHelpers::VectorVector3d start_points;
    EigenHelpers::VectorVector3d end_points;

    for (ssize_t first_node = 0; first_node < num_nodes_; ++first_node)
    {
        for (ssize_t second_node = first_node + 1; second_node < num_nodes_; ++second_node)
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
ObjectDeltaAndWeight TestSpecification::calculateStretchingCorrectionDelta(
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
double TestSpecification::calculateStretchingError(
        const ObjectPointSet& object_configuration) const
{
    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(object_configuration);

    const double stretching_correction_threshold = stretchingScalingThreshold();

    ssize_t squared_error = 0;
    #pragma omp parallel for reduction(+ : squared_error)
    for (ssize_t first_node = 0; first_node < num_nodes_; ++first_node)
    {
        // A node is never overstretched relative to itself, so we can start at the next node
        int node_overstretches = 0;
        for (ssize_t second_node = 0; second_node < num_nodes_; ++second_node)
        {
            if (first_node != second_node)
            {
                const double max_distance = stretching_correction_threshold + object_initial_node_distance_(first_node, second_node);
                if (node_squared_distance(first_node, second_node) > (max_distance * max_distance))
                {
                    ++node_overstretches;
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
double TestSpecification::calculateStretchingError(
        const WorldState& world_state) const
{
    return calculateStretchingError(world_state.object_configuration_);
}

// THIS FUNCTION IS TO COMBINE THE CALCULATED/ PLANNED WITH STRETCHING CORRECTION
// Don't acually need the stretching correction for test model utility though
// can hide off the defiection correction term in FUNCTION  ObjectDesiredDirection()
/**
 * @brief TaskSpecification::combineErrorCorrectionAndStretchingCorrection
 * @param error_correction
 * @param stretching_correction
 * @return
 */
ObjectDeltaAndWeight TestSpecification::combineErrorCorrectionAndStretchingCorrection(
        const ObjectDeltaAndWeight& error_correction,
        const ObjectDeltaAndWeight& stretching_correction) const
{
    ObjectDeltaAndWeight combined(num_nodes_ * 3);

    for (ssize_t node_ind = 0; node_ind < num_nodes_ * 3; node_ind += 3)
    {
        const Eigen::Vector3d error_correction_perpendicular =
                EigenHelpers::VectorRejection(stretching_correction.delta.segment<3>(node_ind),
                                               error_correction.delta.segment<3>(node_ind) );

        combined.delta.segment<3>(node_ind) = stretching_correction.delta.segment<3>(node_ind) + error_correction_perpendicular;
        combined.weight.segment<3>(node_ind) = stretching_correction.weight.segment<3>(node_ind) + error_correction.weight.segment<3>(node_ind);
    }

    // Normalize the weights for later use
    const double combined_normalizer = combined.weight.maxCoeff();
    if (combined_normalizer > 0)
    {
        combined.weight /= combined_normalizer;
    }

    return combined;
}

//////////////////////////////////////////////////////////////////////////////////////////
// The function that combine the planned delata_p (object error) with stretching correction
// Don't acually need the stretching correction for test model utility though
// can hide off the defiection correction term in this function
//////////////////////////////////////////////////////////////////////////////////////////

ObjectDeltaAndWeight TestSpecification::calculateDesiredDirection(const WorldState& world_state)
{
    if (sim_time_last_time_first_step_calced_ != world_state.sim_time_)
    {
        first_step_calculated_.store(false);
    }

    if (first_step_calculated_.load())
    {
        return first_step_desired_motion_;
    }
    else
    {
        std::lock_guard<std::mutex> lock(first_step_mtx_);
        if (first_step_calculated_.load())
        {
            return first_step_desired_motion_;
        }
        else
        {
            ROS_INFO_NAMED("task", "Determining desired direction");
            first_step_error_correction_ = calculateObjectErrorCorrectionDelta(world_state);

            first_step_stretching_correction_ = calculateStretchingCorrectionDelta(world_state, false);

            first_step_desired_motion_ = combineErrorCorrectionAndStretchingCorrection(
                        first_step_error_correction_, first_step_stretching_correction_);

            sim_time_last_time_first_step_calced_ = world_state.sim_time_;
            first_step_calculated_.store(true);
            return first_step_desired_motion_;
        }
    }
}






