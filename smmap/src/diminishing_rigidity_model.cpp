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

DiminishingRigidityModel::DiminishingRigidityModel(
        const Task& task,
        double deformability,
        bool use_rotation,
        double obstacle_avoidance_scale,
        double strechting_correction_threshold )
    : DiminishingRigidityModel(
          task,
          deformability,
          deformability,
          use_rotation,
          obstacle_avoidance_scale,
          strechting_correction_threshold )
{}

DiminishingRigidityModel::DiminishingRigidityModel(
        const Task& task,
        double translation_deformability,
        double rotation_deformability,
        bool use_rotation,
        double obstacle_avoidance_scale,
        double strechting_correction_threshold )
    : task_( task )
    , translation_deformability_( translation_deformability )
    , rotation_deformability_( rotation_deformability )
    , use_rotation_( use_rotation )
    , cols_per_gripper_( use_rotation_ ? 6 : 3 )
    , obstacle_avoidance_scale_( obstacle_avoidance_scale )
    , stretching_correction_threshold_( strechting_correction_threshold )
{
    if ( !initialized_.load() )
    {
        throw new std::runtime_error(
                    "You must call SetInitialObjectConfiguration before constructing a DiminishingRigidityModel" );
    }

    if ( obstacle_avoidance_scale <= 0 )
    {
        throw new std::invalid_argument( "obstacle_avoidance_scale must be > 0" );
    }
    if ( translation_deformability < 0 )
    {
        throw new std::invalid_argument( "translation_deformability must be >= 0" );
    }
    if ( use_rotation_ && rotation_deformability < 0 )
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
    initialized_.store( true );
}

////////////////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////////////////

void DiminishingRigidityModel::updateModel( const std::vector<WorldFeedback>& feedback )
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
        const WorldFeedback& current_world_configuration,
        const std::vector< AllGrippersSinglePose >& grippers_trajectory,
        const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
        double dt ) const
{
    assert( grippers_trajectory.size() > 0 );
    assert( grippers_velocities.size() == grippers_trajectory.size() - 1 );

    ObjectTrajectory object_traj( grippers_trajectory.size() );
    object_traj[0] = current_world_configuration.object_configuration_;

    for ( size_t time_ind = 0; time_ind < grippers_velocities.size(); time_ind++ )
    {
        object_traj[time_ind + 1] = object_traj[time_ind] + getObjectDelta(
                    object_traj[time_ind],
                    grippers_trajectory[time_ind],
                    grippers_velocities[time_ind],
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
        const WorldFeedback& current_world_configuration,
        const std::vector< AllGrippersSinglePose >& grippers_trajectory,
        const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
        double dt ) const
{
    assert( grippers_trajectory.size() > 0 );
    assert( grippers_velocities.size() == grippers_trajectory.size() - 1 );

    ObjectPointSet final_configuration = current_world_configuration.object_configuration_;

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
        const ObjectPointSet& object_current_configuration,
        const AllGrippersSinglePose & grippers_pose,
        const AllGrippersSingleVelocity& grippers_velocity,
        double dt ) const
{
    const Eigen::MatrixXd J = computeGrippersToObjectJacobian(
                grippers_pose,
                object_current_configuration );

    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero( num_nodes_ * 3, 1 );

    // Move the object based on the movement of each gripper
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        if ( use_rotation_ )
        {
            delta += J.block( 0, 6*gripper_ind, J.rows(), 6 )
                    * grippers_velocity[gripper_ind] * dt;
        }
        else
        {
            delta += J.block( 0, 3*gripper_ind, J.rows(), 3 )
                    * grippers_velocity[gripper_ind].segment< 3 >( 0 ) * dt;
        }
    }

    delta.resizeLike( object_current_configuration );
    return delta;
}




std::vector< AllGrippersSinglePose > DiminishingRigidityModel::getDesiredGrippersTrajectory(
        const WorldFeedback& world_feedback,
        double max_step_size, size_t num_steps ) const
{
    ROS_INFO_STREAM_NAMED( "diminishing_rigidity_model",
                           "Creating suggested trajectory: " <<
                           " translational_deformability: " << translation_deformability_ <<
                           " rotational_deformability: " << rotation_deformability_ );

    // Initialize the starting point of the trajectory with the current gripper poses
    std::vector< AllGrippersSinglePose > traj( num_steps + 1, AllGrippersSinglePose( grippers_data_.size() ) );
    traj[0] = world_feedback.all_grippers_single_pose_;

    // Create a mutable pair of objects that represent the current configuration
    // as either a point set or a stacked vector
    ObjectPointSet current_as_point_set = world_feedback.object_configuration_;
    Eigen::Map< Eigen::VectorXd, Eigen::Aligned > current_as_vector( current_as_point_set.data(), 3 * num_nodes_ );

    for ( size_t traj_step = 1; traj_step <= num_steps; traj_step++ )
    {
        ////////////////////////////////////////////////////////////////////////
        // Find the velocities of each part of the algorithm
        ////////////////////////////////////////////////////////////////////////

        const Eigen::VectorXd error_correction =
                task_.calculateObjectDesiredVelocity( current_as_point_set );

        const Eigen::VectorXd stretching_correction
                = computeStretchingCorrection( current_as_point_set );

        // Calculate the desired object velocity (p_dot)
        const Eigen::VectorXd desired_object_velocity =
                error_correction +
                stretching_correction;

        // Recalculate the jacobian at each timestep, because of rotations being non-linear
        const Eigen::MatrixXd J = computeGrippersToObjectJacobian(
                traj[traj_step-1],
                current_as_point_set );

        // Find the least-squares fitting to the desired object velocity
        Eigen::VectorXd grippers_velocity_achieve_goal =
                J.jacobiSvd( Eigen::ComputeThinU | Eigen::ComputeThinV ).solve( desired_object_velocity );

        // Find the collision avoidance data that we'll need
        std::vector< CollisionAvoidanceResult > grippers_collision_avoidance_result
                = computeGrippersObjectAvoidance( world_feedback, traj[traj_step - 1], max_step_size );

        ////////////////////////////////////////////////////////////////////////
        // Combine the velocities into a single command velocity
        ////////////////////////////////////////////////////////////////////////

        for ( long gripper_ind = 0; gripper_ind < (long)grippers_data_.size(); gripper_ind++ )
        {
            Eigen::VectorXd actual_gripper_velocity( cols_per_gripper_ );
            Eigen::VectorXd desired_gripper_vel = grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ );

            // normalize the achive goal velocity

            // To account for the scale difference from porting the code over we
            // need to shrink the size of the rotational components when
            // calculating the norm. Expanding the rotational components again is
            // not done at this point as it causes poor performance of the algorithm
            const double velocity_norm = use_rotation_ ?
                GripperVelocity6dNorm( desired_gripper_vel ) :
                desired_gripper_vel.norm();

            if ( velocity_norm > max_step_size )
            {
                desired_gripper_vel *= max_step_size / velocity_norm;
            }

            // If we need to avoid an obstacle, then use the sliding scale
            const CollisionAvoidanceResult& collision_result = grippers_collision_avoidance_result[(size_t)gripper_ind];
            if ( !std::isinf( collision_result.distance ) )
            {
                 const double collision_severity = std::min( 1.0, std::exp( -obstacle_avoidance_scale_* collision_result.distance ) );

                 actual_gripper_velocity =
                         collision_severity * ( collision_result.velocity + collision_result.nullspace_projector * desired_gripper_vel ) +
                         (1 - collision_severity) * desired_gripper_vel;
            }
            // Otherwise use our desired velocity directly
            else
            {
                 actual_gripper_velocity = desired_gripper_vel;
            }

            // Then apply the velocity to the gripper and record the resulting pose
            kinematics::Vector6d gripper_velocity;
            if ( use_rotation_ )
            {
                gripper_velocity = actual_gripper_velocity;
            }
            else
            {
                gripper_velocity << actual_gripper_velocity, 0, 0, 0;
            }

            traj[traj_step][(size_t)gripper_ind] =
                        traj[traj_step - 1][(size_t)gripper_ind] *
                        kinematics::expTwistAffine3d( gripper_velocity, 1 );

            // Assume that our Jacobian is correct, and predict where we will end up (if needed)
            if ( traj_step < num_steps )
            {
                current_as_vector += J.block( 0, cols_per_gripper_*gripper_ind, J.rows(), cols_per_gripper_ ) * actual_gripper_velocity;
            }
        }
    }

    return traj;
}

void DiminishingRigidityModel::perturbModel( std::mt19937_64& generator )
{
    translation_deformability_ += perturbation_distribution( generator );
    rotation_deformability_ += perturbation_distribution( generator );

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
    const long num_Jcols = cols_per_gripper_ * num_grippers;
    const long num_Jrows = 3*num_nodes_;

    Eigen::MatrixXd J( num_Jrows, num_Jcols );

    // for each gripper
    for ( long gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++ )
    {
        // Get all the data we need for a given gripper
        const Eigen::Matrix3d gripper_rot = grippers_pose[(size_t)gripper_ind].rotation();

        for ( long node_ind = 0; node_ind < num_nodes_; node_ind++ )
        {
            // TODO: do I need to use the gripper_node_index that is returned by this function?
            const std::pair< long, double > dist_to_gripper
                = getMinimumDistanceToGripper( grippers_data_[(size_t)gripper_ind].node_indices,
                        node_ind, object_initial_node_distance_ );

            const Eigen::Matrix3d& J_trans = gripper_rot;

            J.block< 3, 3 >( node_ind * 3, gripper_ind * cols_per_gripper_ ) =
                    std::exp( -translation_deformability_ * dist_to_gripper.second ) * J_trans;

            if ( use_rotation_ )
            {
                Eigen::Matrix3d J_rot = Eigen::Matrix3d::Zero();

                // Vector from gripper to node
                const Eigen::Vector3d gripper_to_node =
//                        current_configuration.block< 3, 1 >( 0, node_ind ) -
                        current_configuration.block< 3, 1 >( 0, dist_to_gripper.first ) -
                        grippers_pose[(size_t)gripper_ind].translation();

//                const Eigen::Vector3d gripper_to_nearest_attached_node =
//                        current_configuration.block< 3, 1 >( 0, dist_to_gripper.first ) -
//                        grippers_pose[(size_t)gripper_ind].translation();

                J_rot.block< 3, 1 >( 0, 0 ) = gripper_rot.block< 3, 1 >( 0, 0 ).cross( gripper_to_node );
                J_rot.block< 3, 1 >( 0, 1 ) = gripper_rot.block< 3, 1 >( 0, 1 ).cross( gripper_to_node );
                J_rot.block< 3, 1 >( 0, 2 ) = gripper_rot.block< 3, 1 >( 0, 2 ).cross( gripper_to_node );

                J.block< 3, 3 >( node_ind * 3, gripper_ind * cols_per_gripper_ + 3 ) =
//                        gripper_to_nearest_attached_node.norm() / gripper_to_node.norm() *
                        std::exp( -rotation_deformability_ * dist_to_gripper.second ) * J_rot;
            }
        }
    }

    return J;
}

std::vector< CollisionAvoidanceResult > DiminishingRigidityModel::computeGrippersObjectAvoidance(
        const WorldFeedback& world_feedback,
        const AllGrippersSinglePose& grippers_pose,
        double max_step_size ) const
{
    std::vector< CollisionAvoidanceResult > collision_avoidance_results(
                grippers_data_.size(), CollisionAvoidanceResult( cols_per_gripper_ ) );

    // TODO: deal with multiple traj_steps, multiple avoids?
    // TODO: deal with not having an object to avoid
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
    {
        collision_avoidance_results[gripper_ind].distance = world_feedback.all_grippers_distance_to_obstacle_[gripper_ind];

        // If we have a collision to avoid, then find the vector
        if ( !std::isinf( world_feedback.all_grippers_distance_to_obstacle_[gripper_ind] ) )
        {
            // Create the collision Jacobian
            const Eigen::MatrixXd J_collision = computeCollisionToGripperJacobian(
                        world_feedback.all_grippers_nearest_point_to_obstacle_[gripper_ind], grippers_pose[gripper_ind] );
            const Eigen::MatrixXd J_collision_inv = EigenHelpers::Pinv( J_collision, EigenHelpers::SuggestedRcond() );

            // Create the collision avoidance vector to follow
            const Eigen::Vector3d& avoid_collision_delta = world_feedback.all_grippers_obstacle_surface_normal_[gripper_ind];

            collision_avoidance_results[gripper_ind].velocity =  J_collision_inv * avoid_collision_delta;
            collision_avoidance_results[gripper_ind].velocity /= collision_avoidance_results[gripper_ind].velocity.norm();
            collision_avoidance_results[gripper_ind].velocity *= max_step_size;

            collision_avoidance_results[gripper_ind].nullspace_projector =
                    Eigen::MatrixXd::Identity( cols_per_gripper_, cols_per_gripper_) - J_collision_inv * J_collision;
        }
        // Otherwise, leave the collision result as the default "no collision" state
        else {}
    }

    return collision_avoidance_results;
}

/**
 * @brief DiminishingRigidityModel::computeCollisionToGripperJacobian
 * Computes a Jacobian that converts a gripper velocity in the individual
 * gripper frame into a point velocity in the world frame
 * @param gripper_data
 * @return
 */
Eigen::MatrixXd DiminishingRigidityModel::computeCollisionToGripperJacobian(
        const Eigen::Vector3d& point_on_gripper,
        const Eigen::Affine3d& gripper_pose ) const
{
    Eigen::MatrixXd J_collision = Eigen::MatrixXd::Zero( 3, cols_per_gripper_ );
    const Eigen::Matrix3d gripper_rot = gripper_pose.rotation();

    // Translation - if I move the gripper along its x/y/z-axis, what happens to the given point?
    J_collision.block< 3, 3 >( 0, 0 ) = gripper_rot;

    const Eigen::Vector3d gripper_to_point_in_collision =
            point_on_gripper - gripper_pose.translation();

    // If I rotate the gripper about its x/y/z-axis, what happens to the point in question?
    if ( cols_per_gripper_ == 6 )
    {
        J_collision.block< 3, 1 >( 0, 3 ) = gripper_rot.block< 3, 1 >( 0, 0 ).cross( gripper_to_point_in_collision );
        J_collision.block< 3, 1 >( 0, 4 ) = gripper_rot.block< 3, 1 >( 0, 1 ).cross( gripper_to_point_in_collision );
        J_collision.block< 3, 1 >( 0, 5 ) = gripper_rot.block< 3, 1 >( 0, 2 ).cross( gripper_to_point_in_collision );
    }

    return J_collision;
}

Eigen::VectorXd DiminishingRigidityModel::computeStretchingCorrection(
        const ObjectPointSet& object_current_configuration ) const
{
    Eigen::VectorXd stretching_correction = Eigen::VectorXd::Zero( object_current_configuration.cols() * 3 );

    Eigen::MatrixXd node_distance_delta =
            distanceMatrix( object_current_configuration )
            - object_initial_node_distance_;

    for ( long first_node = 0; first_node < node_distance_delta.rows(); first_node++)
    {
        for ( long second_node = first_node + 1; second_node < node_distance_delta.cols(); second_node++)
        {
            if ( node_distance_delta( first_node, second_node ) > stretching_correction_threshold_ )
            {
                // The correction vector points from the first node to the second node,
                // and is half the length of the "extra" distance
                Eigen::Vector3d correction_vector = 0.5 * node_distance_delta( first_node, second_node )
                        * ( object_current_configuration.block< 3, 1 >( 0, second_node )
                            - object_current_configuration.block< 3, 1 >( 0, first_node ) );

                stretching_correction.segment< 3 >( 3 * first_node ) += correction_vector;
                stretching_correction.segment< 3 >( 3 * second_node ) -= correction_vector;
            }
        }
    }

    return stretching_correction;
}

////////////////////////////////////////////////////////////////////////////////
// Static member initialization
////////////////////////////////////////////////////////////////////////////////

/// TODO: remove this magic number for the noise generated by this distribution
std::normal_distribution< double > DiminishingRigidityModel::perturbation_distribution =
    std::normal_distribution< double >( 0, 0.1 );

Eigen::MatrixXd DiminishingRigidityModel::object_initial_node_distance_;

long DiminishingRigidityModel::num_nodes_;

std::atomic<bool> DiminishingRigidityModel::initialized_( false );
