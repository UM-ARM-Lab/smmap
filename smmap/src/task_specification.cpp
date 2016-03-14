#include "smmap/task_specification.h"
#include "smmap/task_specification_implementions.hpp"

using namespace smmap;

////////////////////////////////////////////////////////////////////
// Constructor to initialize objects that all TaskSpecifications share
////////////////////////////////////////////////////////////////////

TaskSpecification::TaskSpecification( ros::NodeHandle& nh )
    : object_initial_node_distance_( CalculateDistanceMatrix( GetObjectInitialConfiguration( nh ) ) )
    , num_nodes_( object_initial_node_distance_.cols() )
{}

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

void TaskSpecification::visualizeDeformableObject(
        Visualizer& vis,
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector< std_msgs::ColorRGBA >& colors ) const
{
    visualizeDeformableObject_impl( vis, marker_name, object_configuration, colors );
}

double TaskSpecification::calculateError(
        const ObjectPointSet& object_configuration ) const
{
    return calculateError_impl( object_configuration );
}

std::pair< Eigen::VectorXd, Eigen::VectorXd > TaskSpecification::calculateObjectErrorCorrectionDelta(
        const WorldState& world_state ) const
{
    return calculateObjectErrorCorrectionDelta_impl( world_state );
}

/**
 * @brief TaskSpecification::calculateStretchingCorrectionDelta
 * @param world_state
 * @return
 */
std::pair< Eigen::VectorXd, Eigen::VectorXd > TaskSpecification::calculateStretchingCorrectionDelta(
        const WorldState& world_state ) const
{
    std::pair< Eigen::VectorXd, Eigen::VectorXd > stretching_correction =
            std::make_pair( Eigen::VectorXd::Zero( num_nodes_ * 3 ),
                            Eigen::VectorXd::Zero( num_nodes_ * 3 ) );

    const Eigen::MatrixXd node_distance_delta =
            CalculateDistanceMatrix( world_state.object_configuration_ )
            - object_initial_node_distance_;

    const double stretching_correction_threshold = getStretchingScalingThreshold();

    for ( long first_node = 0; first_node < num_nodes_; first_node++)
    {
        for ( long second_node = first_node + 1; second_node < num_nodes_; second_node++)
        {
            if ( node_distance_delta( first_node, second_node ) > stretching_correction_threshold )
            {
                // The correction vector points from the first node to the second node,
                // and is half the length of the "extra" distance
                const Eigen::Vector3d correction_vector = 0.5
                        * node_distance_delta( first_node, second_node )
                        * ( world_state.object_configuration_.block< 3, 1 >( 0, second_node )
                            - world_state.object_configuration_.block< 3, 1 >( 0, first_node ) );

                stretching_correction.first.segment< 3 >( 3 * first_node ) += correction_vector;
                stretching_correction.first.segment< 3 >( 3 * second_node ) -= correction_vector;

                stretching_correction.second( 3 * first_node ) += 1;
                stretching_correction.second( 3 * first_node + 1 ) += 1;
                stretching_correction.second( 3 * first_node + 2 ) += 1;
                stretching_correction.second( 3 * second_node ) += 1;
                stretching_correction.second( 3 * second_node + 1 ) += 1;
                stretching_correction.second( 3 * second_node + 2 ) += 1;
            }
        }
    }

    // Normalize the weights so that changing the number of nodes doesn't affect
    // the weights too much; i.e. minimize the effect of the level of discretization
//    stretching_correction.second /= (double)num_nodes_;

    return stretching_correction;
}

/**
 * @brief TaskSpecification::combineErrorCorrectionAndStretchingCorrection
 * @param error_correction
 * @param stretching_correction
 * @return
 */
// TODO: this probably doesn't belong in this class
std::pair< Eigen::VectorXd, Eigen::VectorXd > TaskSpecification::combineErrorCorrectionAndStretchingCorrection(
        const std::pair< Eigen::VectorXd, Eigen::VectorXd >& error_correction,
        const std::pair< Eigen::VectorXd, Eigen::VectorXd >& stretching_correction ) const
{
    std::pair< Eigen::VectorXd, Eigen::VectorXd > combined =
            std::make_pair( Eigen::VectorXd( num_nodes_ * 3 ),
                            Eigen::VectorXd( num_nodes_ * 3 ) );

//    std::cout << "Max error:      " << error_correction.second.maxCoeff() << std::endl
//              << "Sum error:      " << error_correction.second.sum() << std::endl
//              << "Max stretching: " << stretching_correction.second.maxCoeff() << std::endl
//              << "Sum stretching: " << stretching_correction.second.sum() << std::endl;


    for ( long ind = 0; ind < num_nodes_ * 3; ind += 3 )
    {
        const double stretching_importance =
                1.0 - std::exp( -10.0*1e-3 * stretching_correction.second( ind ) );

        // Calculate the combined object delta
        combined.first.segment< 3 >( ind ) =
                stretching_importance * stretching_correction.first.segment< 3 >( ind )
                + ( 1.0 - stretching_importance ) * error_correction.first.segment< 3 >( ind );

        // Calculate the combined node weights
        combined.second.segment< 3 >( ind ) =
                stretching_importance * stretching_correction.second.segment< 3 >( ind )
                + ( 1.0 - stretching_importance ) * error_correction.second.segment< 3 >( ind );
    }

//    combined.first = error_correction.first + stretching_correction.first;
//    combined.second = Eigen::VectorXd::Ones( num_nodes_ * 3 );

    // Normalize the weights for later use
    const double combined_normalizer = combined.second.maxCoeff();
    assert( combined_normalizer > 0 );
    combined.second /= combined_normalizer;

    return combined;
}
