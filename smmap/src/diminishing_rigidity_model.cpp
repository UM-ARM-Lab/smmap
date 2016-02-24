#include "smmap/diminishing_rigidity_model.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <Eigen/SVD>

#include <ros/ros.h>

using namespace smmap;

// TODO: find a way to accept dynamic/online gripper re-grasping

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

DiminishingRigidityModel::DiminishingRigidityModel( double deformability )
    : DiminishingRigidityModel( deformability, deformability )
{}

DiminishingRigidityModel::DiminishingRigidityModel(
        double translation_deformability,
        double rotation_deformability )
    : translation_deformability_( translation_deformability )
    , rotation_deformability_( rotation_deformability )
{
    if ( !static_data_initialized_.load() )
    {
        throw new std::runtime_error(
                    "You must call SetInitialObjectConfiguration before constructing a DiminishingRigidityModel" );
    }

    if ( translation_deformability < 0 )
    {
        throw new std::invalid_argument( "translation_deformability must be >= 0" );
    }
    if ( rotation_deformability < 0 )
    {
        throw new std::invalid_argument( "rotation_deformability must be >= 0" );
    }
}

////////////////////////////////////////////////////////////////////////////////
// Static helpers
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief DiminishingRigidityModel::SetInitialObjectConfiguration This function
 *          is not thread safe.
 * @param object_initial_configuration
 */
void DiminishingRigidityModel::SetInitialObjectConfiguration(
        const ObjectPointSet& object_initial_configuration )
{
    num_nodes_ = object_initial_configuration.cols();
    object_initial_node_distance_ = distanceMatrix( object_initial_configuration );
    static_data_initialized_.store( true );
}

////////////////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////////////////

void DiminishingRigidityModel::updateModel( const std::vector<WorldState>& feedback )
{
    // This model doesn't do any updates, so tell the compiler that it's okay
    // that these values are unused.
    (void)feedback;
}

/**
 * @brief DiminishingRigidityModel::getPrediction
 * @param object_configuration
 * @param grippers_trajectory
 * @param grippers_velocities Note that velocity 0 moves us from pose 0 to pose 1 for a given gripper
 * @return
 */
ObjectTrajectory DiminishingRigidityModel::getPrediction(
        const WorldState& world_initial_state,
        const AllGrippersPoseTrajectory & grippers_pose_trajectory,
        const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory,
        double dt ) const
{
    assert( grippers_pose_trajectory.size() > 0 );
    assert( grippers_pose_delta_trajectory.size() == grippers_pose_trajectory.size() - 1 );

    ObjectTrajectory object_traj( grippers_pose_trajectory.size() );
    object_traj[0] = world_initial_state.object_configuration_;

    for ( size_t time_ind = 0; time_ind < grippers_pose_delta_trajectory.size(); time_ind++ )
    {
        object_traj[time_ind + 1] = object_traj[time_ind] + getObjectDelta(
                    object_traj[time_ind],
                    grippers_pose_trajectory[time_ind],
                    grippers_pose_delta_trajectory[time_ind],
                    dt );
    }

    return object_traj;
}

/**
 * @brief DiminishingRigidityModel::getFinalConfiguration
 * @param object_configuration
 * @param grippers_trajectory
 * @param grippers_velocities Note that velocity 0 moves us from pose 0 to pose 1 for a given gripper
 * @return
 */
ObjectPointSet DiminishingRigidityModel::getFinalConfiguration(
        const WorldState& world_initial_state,
        const std::vector< AllGrippersSinglePose >& grippers_trajectory,
        const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
        double dt ) const
{
    assert( grippers_trajectory.size() > 0 );
    assert( grippers_velocities.size() == grippers_trajectory.size() - 1 );

    ObjectPointSet final_configuration = world_initial_state.object_configuration_;

    for ( size_t time_ind = 0; time_ind < grippers_velocities.size(); time_ind++ )
    {
        final_configuration += getObjectDelta(
                    final_configuration,
                    grippers_trajectory[time_ind],
                    grippers_velocities[time_ind],
                    dt );
    }

    return final_configuration;
}

/**
 * @brief DiminishingRigidityModel::getObjectDelta
 * @param object_current_configuration
 * @param grippers_pose
 * @param grippers_velocity
 * @param dt
 * @return
 */
ObjectPointSet DiminishingRigidityModel::getObjectDelta(
        const ObjectPointSet& object_initial_configuration,
        const AllGrippersSinglePose & grippers_pose,
        const AllGrippersSingleVelocity& grippers_velocity,
        double dt ) const
{
    const Eigen::MatrixXd J = computeGrippersToObjectJacobian(
                grippers_pose,
                object_initial_configuration );

    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero( num_nodes_ * 3, 1 );

    // Move the object based on the movement of each gripper
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        delta += J.block( 0, 6 * gripper_ind, J.rows(), 6 )
                * grippers_velocity[gripper_ind] * dt;
    }

    delta.resizeLike( object_initial_configuration );
    return delta;
}

std::pair< AllGrippersPoseTrajectory, ObjectTrajectory >
DiminishingRigidityModel::getSuggestedGrippersTrajectory(
        const WorldState& world_initial_state,
        const int planning_horizion,
        const double dt,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale ) const
{
    ROS_INFO_STREAM_NAMED( "diminishing_rigidity_model",
                           "Creating suggested grippers trajectory: " <<
                           " translational_deformability: " << translation_deformability_ <<
                           " rotational_deformability: " << rotation_deformability_ );

    // Make a copy of the world state because we'll mutate it
    // TODO: is this true for all models and thus the prototype should change?
    WorldState world_current_state = world_initial_state;

    const double max_step_size = max_gripper_velocity * dt;

    auto suggested_traj = std::make_pair< AllGrippersPoseTrajectory, ObjectTrajectory >(
                AllGrippersPoseTrajectory( planning_horizion + 1, AllGrippersSinglePose( grippers_data_.size() ) ),
                ObjectTrajectory( planning_horizion + 1, ObjectPointSet( 3, num_nodes_ ) )
                );

    // Initialize the starting point of the trajectory with the current gripper
    // poses and object configuration
    suggested_traj.first[0] = world_current_state.all_grippers_single_pose_;
    suggested_traj.second[0] = world_current_state.object_configuration_;

    // Create a mutable pair of objects that represent the current configuration
    // as either a point set or a stacked vector
//    ObjectPointSet* current_as_point_set = &(suggested_traj.second[0]);
//    Eigen::Map< Eigen::VectorXd, Eigen::Aligned > current_as_vector( current_as_point_set->data(), 3 * num_nodes_ );

    for ( int traj_step = 1; traj_step <= planning_horizion; traj_step++ )
    {
        ////////////////////////////////////////////////////////////////////////
        // Find the velocities of each part of the algorithm
        ////////////////////////////////////////////////////////////////////////

        // Retrieve the desired object velocity (p_dot)
        const std::pair< Eigen::VectorXd, Eigen::MatrixXd > desired_object_velocity
                = task_desired_object_delta_fn_( world_current_state );

        // Recalculate the jacobian at each timestep, because of rotations being non-linear
        const Eigen::MatrixXd J = computeGrippersToObjectJacobian(
                suggested_traj.first[traj_step-1],
                suggested_traj.second[traj_step-1] );

        // Find the least-squares fitting to the desired object velocity
        Eigen::VectorXd grippers_velocity_achieve_goal =
                J.jacobiSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve( desired_object_velocity.first );

        // Find the collision avoidance data that we'll need
        std::vector< CollisionAvoidanceResult > grippers_collision_avoidance_result
                = ComputeGripperObjectAvoidance(
                    world_current_state.gripper_collision_data_,
                    suggested_traj.first[traj_step-1], max_step_size );

        ////////////////////////////////////////////////////////////////////////
        // Combine the velocities into a single command velocity
        ////////////////////////////////////////////////////////////////////////

        for ( long gripper_ind = 0; gripper_ind < (long)grippers_data_.size(); gripper_ind++ )
        {
            kinematics::Vector6d actual_gripper_velocity;
            kinematics::Vector6d desired_gripper_vel = grippers_velocity_achieve_goal.segment< 6 >( gripper_ind * 6 );

            // normalize the achive goal velocity
            const double velocity_norm = GripperVelocity6dNorm( desired_gripper_vel );

            if ( velocity_norm > max_step_size )
            {
                desired_gripper_vel *= max_step_size / velocity_norm;
            }

            // If we need to avoid an obstacle, then use the sliding scale
            const CollisionAvoidanceResult& collision_result =
                    grippers_collision_avoidance_result[(size_t)gripper_ind];
            if ( !std::isinf( collision_result.distance ) )
            {
                 const double collision_severity =
                         std::min( 1.0, std::exp( -obstacle_avoidance_scale * collision_result.distance ) );

                 actual_gripper_velocity =
                         collision_severity * ( collision_result.velocity + collision_result.nullspace_projector * desired_gripper_vel ) +
                         (1 - collision_severity) * desired_gripper_vel;
            }
            // Otherwise use our desired velocity directly
            else
            {
                 actual_gripper_velocity = desired_gripper_vel;
            }

            suggested_traj.first[traj_step][(size_t)gripper_ind] =
                        suggested_traj.first[traj_step - 1][(size_t)gripper_ind] *
                        kinematics::expTwistAffine3d( actual_gripper_velocity, 1 );

            Eigen::MatrixXd object_delta = J.block( 0, 6 * gripper_ind, J.rows(), 6 ) * actual_gripper_velocity;
            object_delta.resizeLike( suggested_traj.second[traj_step] );

            // Assume that our Jacobian is correct, and predict where we will end up (if needed)
            suggested_traj.second[traj_step] = suggested_traj.second[traj_step-1] + object_delta;

            // If we are going to do more steps, mutate the current world state
            if ( traj_step + 1 < planning_horizion )
            {
                world_current_state.object_configuration_ = suggested_traj.second[traj_step];
                world_current_state.all_grippers_single_pose_ = suggested_traj.first[traj_step];
                world_current_state.gripper_collision_data_ = gripper_collision_check_fn_( world_current_state.all_grippers_single_pose_ );
                world_current_state.sim_time_ += dt;
            }
        }
    }

    return suggested_traj;
}

void DiminishingRigidityModel::perturbModel( std::mt19937_64& generator )
{
    translation_deformability_ += perturbation_distribution_( generator );
    rotation_deformability_ += perturbation_distribution_( generator );

    if ( translation_deformability_ < 0 )
    {
        translation_deformability_ = 0;
    }
    if ( rotation_deformability_ < 0 )
    {
        rotation_deformability_ = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Computation helpers
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief DiminishingRigidityModel::computeObjectToGripperJacobian
 * Computes a Jacobian that converts gripper velocities in the individual
 * gripper frames into object velocities in the world frame
 * @param grippers_data
 */
Eigen::MatrixXd DiminishingRigidityModel::computeGrippersToObjectJacobian(
        const AllGrippersSinglePose& grippers_pose,
        const ObjectPointSet& current_configuration ) const
{
    //ROS_DEBUG_NAMED( "diminishing_rigidity_model" , "Computing object Jacobian: Diminishing rigidity k_trans: %f k_rot: %f", translation_deformablity_, rotation_deformability_ );

    const long num_grippers = (long)grippers_pose.size();
    const long num_Jcols = num_grippers * 6;
    const long num_Jrows = num_nodes_ * 3;

    Eigen::MatrixXd J( num_Jrows, num_Jcols );

    // for each gripper
    for ( long gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++ )
    {
        // Get all the data we need for a given gripper
        const Eigen::Matrix3d& gripper_rot = grippers_pose[(size_t)gripper_ind].rotation();

        for ( long node_ind = 0; node_ind < num_nodes_; node_ind++ )
        {
            // TODO: do I need to use the gripper_node_index that is returned by this function?
            const std::pair< long, double > dist_to_gripper
                = getMinimumDistanceToGripper( grippers_data_[(size_t)gripper_ind].node_indices,
                        node_ind, object_initial_node_distance_ );

            const Eigen::Matrix3d& J_trans = gripper_rot;

            J.block< 3, 3 >( node_ind * 3, gripper_ind * 6 ) =
                    std::exp( -translation_deformability_ * dist_to_gripper.second ) * J_trans;

            Eigen::Matrix3d J_rot;

            // Vector from gripper to node
            const Eigen::Vector3d gripper_to_node =
                    current_configuration.block< 3, 1 >( 0, node_ind ) -
                    grippers_pose[(size_t)gripper_ind].translation();

            J_rot.block< 3, 1 >( 0, 0 ) = gripper_rot.block< 3, 1 >( 0, 0 ).cross( gripper_to_node );
            J_rot.block< 3, 1 >( 0, 1 ) = gripper_rot.block< 3, 1 >( 0, 1 ).cross( gripper_to_node );
            J_rot.block< 3, 1 >( 0, 2 ) = gripper_rot.block< 3, 1 >( 0, 2 ).cross( gripper_to_node );

            J.block< 3, 3 >( node_ind * 3, gripper_ind * 6 + 3 ) =
                    std::exp( -rotation_deformability_ * dist_to_gripper.second ) * J_rot;
        }
    }

    return J;
}

////////////////////////////////////////////////////////////////////////////////
// Static member initialization
////////////////////////////////////////////////////////////////////////////////

/// TODO: remove this magic number for the noise generated by this distribution
std::normal_distribution< double > DiminishingRigidityModel::perturbation_distribution_ =
    std::normal_distribution< double >( 0, 0.1 );

std::atomic_bool DiminishingRigidityModel::static_data_initialized_( false );
Eigen::MatrixXd DiminishingRigidityModel::object_initial_node_distance_;
long DiminishingRigidityModel::num_nodes_;
