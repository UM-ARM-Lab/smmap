#include "smmap/diminishing_rigidity_model.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

#include <ros/ros.h>

using namespace smmap;

// TODO: find a way to accept dynamic/online gripper re-grasping

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

DiminishingRigidityModel::DiminishingRigidityModel(
        const VectorGrippersData& grippers_data,
        const ObjectPointSet& object_initial_configuration,
        double rigidity, bool use_rotation,
        double obstacle_avoidance_scale, double strechting_correction_threshold )
    : DiminishingRigidityModel( grippers_data, object_initial_configuration,
                                rigidity, rigidity, use_rotation,
                                obstacle_avoidance_scale, strechting_correction_threshold )
{}

DiminishingRigidityModel::DiminishingRigidityModel(
        const VectorGrippersData& grippers_data,
        const ObjectPointSet& object_initial_configuration,
        double rigidity_translation, double rotation_rigidity, bool use_rotation,
        double obstacle_avoidance_scale, double strechting_correction_threshold )
    : object_initial_configuration_( object_initial_configuration )
    , translation_rigidity_( rigidity_translation )
    , rotation_rigidity_( rotation_rigidity )
    , use_rotation_( use_rotation )
    , cols_per_gripper_( use_rotation_ ? 6 : 3 )
    , obstacle_avoidance_scale_( obstacle_avoidance_scale )
    , stretching_correction_threshold_( strechting_correction_threshold )
{
    if ( obstacle_avoidance_scale <= 0 )
    {
        throw new std::invalid_argument( "obstacle_avoidance_scale must be > 0" );
    }
    if ( rigidity_translation < 0 )
    {
        throw new std::invalid_argument( "translation_rigidity must be >= 0" );
    }
    if ( use_rotation_ && rotation_rigidity < 0 )
    {
        throw new std::invalid_argument( "rotation_rigidity must be >= 0" );
    }

    computeObjectNodeDistanceMatrix();
    computeGrippersToObjectJacobian( grippers_data );
}

////////////////////////////////////////////////////////////////////////////////
// Constructor helpers
////////////////////////////////////////////////////////////////////////////////

void DiminishingRigidityModel::computeObjectNodeDistanceMatrix()
{
    // TODO: replace this ugly hack
    if ( object_initial_node_distance_.size() == 0 )
    {
        ROS_INFO_NAMED( "diminishing_rigidity_model" , "Computing object initial distance matrix" );

        object_initial_node_distance_ = distanceMatrix( object_initial_configuration_ );
    }
}

////////////////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////////////////

void DiminishingRigidityModel::doUpdateModel(
        const VectorGrippersData& grippers_data,
        const AllGrippersTrajectory& grippers_trajectory,
        const std::vector< kinematics::VectorVector6d >& grippers_velocities,
        const ObjectTrajectory& object_trajectory,
        const kinematics::VectorMatrix3Xd& object_velocities )
{
    // This model doesn't do any updates, so tell the compiler that it's okay
    // that these values are unused.
    (void)grippers_data;
    (void)grippers_trajectory;
    (void)grippers_velocities;
    (void)object_trajectory;
    (void)object_velocities;
}

// TODO: update this for the new way of recalculating J.
ObjectTrajectory DiminishingRigidityModel::doGetPrediction(
        const ObjectPointSet& object_configuration,
        VectorGrippersData grippers_data,
        const AllGrippersTrajectory& grippers_trajectory,
        const std::vector< kinematics::VectorVector6d >& grippers_velocities ) const
{
    assert( grippers_trajectory.size() > 0 );
    assert( grippers_velocities.size() == grippers_trajectory.size() );

    ObjectTrajectory object_traj( grippers_trajectory[0].size() );
    object_traj[0] = object_configuration;

    for ( size_t vel_ind = 0; vel_ind < grippers_velocities[0].size(); vel_ind++ )
    {
        // Recalculate the jacobian at each timestep, because of rotations being non-linear
        const Eigen::MatrixXd J = computeGrippersToObjectJacobian( grippers_data );

        // We start from where we were last
        object_traj[vel_ind + 1] = object_traj[vel_ind];

        // Move the object based on the movement of each gripper
        for ( long gripper_ind = 0; gripper_ind < (long)grippers_velocities.size(); gripper_ind++ )
        {
            Eigen::MatrixXd delta;

            // Assume that our Jacobian is correct, and predict where we will end up
            if ( use_rotation_ )
            {
                delta = J.block( 0, 6*gripper_ind, J.rows(), 6 )
                        * grippers_velocities[(size_t)gripper_ind][vel_ind];
            }
            else
            {
                delta = J.block( 0, 3*gripper_ind, J.rows(), 3 )
                        * grippers_velocities[(size_t)gripper_ind][vel_ind].segment< 3 >( 0 );
            }
            delta.resizeLike(object_traj[vel_ind + 1]);
            object_traj[vel_ind + 1] = object_traj[vel_ind + 1] + delta;

            grippers_data[(size_t)gripper_ind].pose = grippers_trajectory[(size_t)gripper_ind][(size_t)vel_ind + 1];
        }
    }

    return object_traj;
}

AllGrippersTrajectory DiminishingRigidityModel::doGetDesiredGrippersTrajectory(
        const ObjectPointSet& object_current_configuration,
        const ObjectPointSet& object_desired_configuration,
        VectorGrippersData grippers_data,
        double max_step_size, size_t num_steps ) const
{
    ROS_INFO_STREAM_NAMED( "diminishing_rigidity_model",
                           "Creating suggested trajectory: " <<
                           " translational_rigidity: " << translation_rigidity_ <<
                           " rotational_rigidity: " << rotation_rigidity_ );

    assert( grippers_data.size() == grippers_data.size() );
    // Initialize the starting point of the trajectory with the current gripper poses
    AllGrippersTrajectory traj( grippers_data.size() );
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data.size(); gripper_ind++ )
    {
        traj[gripper_ind].push_back( grippers_data[gripper_ind].pose );
    }

    // TODO: get rid of this uglyness, needed for a reshape
    Eigen::MatrixXd tmp_current = object_current_configuration;
    Eigen::MatrixXd tmp_desired = object_desired_configuration;
    const Eigen::VectorXd desired = Eigen::Map< Eigen::VectorXd, Eigen::Aligned >( tmp_current.data(), object_desired_configuration.cols() * object_desired_configuration.rows() );
    Eigen::VectorXd current = Eigen::Map< Eigen::VectorXd, Eigen::Aligned >( tmp_desired.data(), object_current_configuration.cols() * object_current_configuration.rows() );

    for ( size_t traj_step = 1; traj_step <= num_steps; traj_step++ )
    {
        ////////////////////////////////////////////////////////////////////////
        // Find the velocities of each part of the algorithm
        ////////////////////////////////////////////////////////////////////////

        // Recalculate the jacobian at each timestep, because of rotations being non-linear
        const Eigen::MatrixXd J = computeGrippersToObjectJacobian( grippers_data );
        // Yes, this is ugly. This is to suppress a warning on type conversion related to Eigen operations
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wconversion"
        const Eigen::MatrixXd J_inv = EigenHelpers::Pinv( J.transpose() * J, EigenHelpers::SuggestedRcond() ) * J.transpose();
        #pragma GCC diagnostic pop

        // pragmas are here to supress some warnings from GCC
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wconversion"
        // Apply the desired object delta Jacobian pseudo-inverse - will normalize later
        Eigen::VectorXd grippers_velocity_achieve_goal = J_inv *
                ( (current - desired) + computeStretchingCorrection( object_current_configuration ) );
        #pragma GCC diagnostic pop

        // Find the collision avoidance data that we'll need
        std::vector< CollisionAvoidanceResult > grippers_collision_avoidance_result
                = computeGrippersObjectAvoidance( grippers_data, max_step_size );

        ////////////////////////////////////////////////////////////////////////
        // Combine the velocities into a single command velocity
        ////////////////////////////////////////////////////////////////////////

        for ( long gripper_ind = 0; gripper_ind < (long)grippers_data.size(); gripper_ind++ )
        {
            Eigen::VectorXd actual_gripper_velocity( cols_per_gripper_ );
            Eigen::VectorXd desired_gripper_vel = grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ );

            // normalize the achive goal velocity
            if ( desired_gripper_vel.norm() > max_step_size )
            {
                desired_gripper_vel = desired_gripper_vel / desired_gripper_vel.norm() * max_step_size;
            }

            // If we need to avoid an obstacle, then use the sliding scale
            const CollisionAvoidanceResult& collision_result = grippers_collision_avoidance_result[(size_t)gripper_ind];
            if ( !std::isinf( collision_result.distance ) )
            {
                 const double collision_severity = std::min( 1.0, std::exp( -obstacle_avoidance_scale_* collision_result.distance ) );

                 std::cout << "collision severity: " << collision_severity << std::endl;

                 actual_gripper_velocity =
                         collision_severity * ( collision_result.velocity + collision_result.nullspace_projector * desired_gripper_vel ) +
                         (1 - collision_severity) * desired_gripper_vel;
            }
            // Otherwise use our desired velocity directly
            else
            {
                 actual_gripper_velocity = desired_gripper_vel;
            }

            // Then apply the velocity to the gripper
            kinematics::Vector6d gripper_velocity;
            if ( use_rotation_ )
            {
                gripper_velocity = actual_gripper_velocity;
            }
            else
            {
                gripper_velocity << actual_gripper_velocity, 0, 0, 0;
            }

            traj[(size_t)gripper_ind].push_back(
                        traj[(size_t)gripper_ind][traj_step - 1] *
                        kinematics::expTwistAffine3d( gripper_velocity, 1 ) );

            grippers_data[(size_t)gripper_ind].pose = traj[(size_t)gripper_ind].back();

            // Assume that our Jacobian is correct, and predict where we will end up
            current += J.block( 0, cols_per_gripper_*gripper_ind, J.rows(), cols_per_gripper_ ) * actual_gripper_velocity;
        }
    }

    return traj;
}

void DiminishingRigidityModel::doPerturbModel( std::mt19937_64& generator )
{
    translation_rigidity_ += perturbation_distribution( generator );
    rotation_rigidity_ += perturbation_distribution( generator );

    if ( translation_rigidity_ < 0 )
    {
        translation_rigidity_ = 0;
    }
    if ( rotation_rigidity_ < 0 )
    {
        rotation_rigidity_ = 0;
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
Eigen::MatrixXd DiminishingRigidityModel::computeGrippersToObjectJacobian( const VectorGrippersData& grippers_data ) const
{
    ROS_DEBUG_NAMED( "diminishing_rigidity_model" , "Computing object Jacobian: Diminishing rigidity k_trans: %f k_rot: %f", translation_rigidity_, rotation_rigidity_ );

    const long num_grippers = (long)grippers_data.size();
    const long num_Jcols = cols_per_gripper_ * num_grippers;

    const long num_nodes = object_initial_configuration_.cols();
    const long num_Jrows = 3*object_initial_configuration_.cols();

    Eigen::MatrixXd J( num_Jrows, num_Jcols );

    // for each gripper
    for ( long gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++ )
    {
        // Get all the data we need for a given gripper
        const std::vector< long >& gripper_node_indices = grippers_data[(size_t)gripper_ind].node_indices;
        const Eigen::Matrix3d gripper_rot = grippers_data[(size_t)gripper_ind].pose.rotation();

        for ( long node_ind = 0; node_ind < num_nodes; node_ind++ )
        {
            // TODO: do I need to use the gripper_node_index that is returned by this function?
            const std::pair< long, double > dist_to_gripper
                = getMinimumDistanceToGripper( gripper_node_indices, node_ind,
                        object_initial_node_distance_ );

            //const Eigen::Matrix3d J_trans = Eigen::Matrix3d::Identity();
            const Eigen::Matrix3d J_trans = gripper_rot;
            J.block< 3, 3 >( node_ind * 3, gripper_ind * cols_per_gripper_ ) =
                    std::exp( -translation_rigidity_ * dist_to_gripper.second ) * J_trans;

            if ( use_rotation_ )
            {
                Eigen::Matrix3d J_rot = Eigen::Matrix3d::Zero();
                // Vector from gripper to node
                const Eigen::Vector3d gripper_to_node =
                        object_initial_configuration_.block< 3, 1 >( 0, node_ind ) -
                        grippers_data[(size_t)gripper_ind].pose.translation();
                J_rot.block< 3, 1 >( 0, 0 ) = gripper_rot.block< 3, 1 >( 0, 0 ).cross( gripper_to_node );
                J_rot.block< 3, 1 >( 0, 1 ) = gripper_rot.block< 3, 1 >( 0, 1 ).cross( gripper_to_node );
                J_rot.block< 3, 1 >( 0, 2 ) = gripper_rot.block< 3, 1 >( 0, 2 ).cross( gripper_to_node );
                J_rot *= 20; // to account for the scale difference from porting the code over

                J.block< 3, 3 >( node_ind * 3, gripper_ind * cols_per_gripper_ + 3 ) =
                        std::exp( -rotation_rigidity_ * dist_to_gripper.second ) * J_rot * 50;
            }
        }
    }

//    std::cout << "Jacobian: translation_rigidity: " << translation_rigidity_ << " rotation_rigidity: " << rotation_rigidity_ << std::endl;
//    std::cout << J << std::endl;

    return J;
}

std::vector< CollisionAvoidanceResult > DiminishingRigidityModel::computeGrippersObjectAvoidance(
        const VectorGrippersData& grippers_data, double max_step_size ) const
{
    std::vector< CollisionAvoidanceResult > collision_avoidance_results( grippers_data.size(), CollisionAvoidanceResult( cols_per_gripper_ ) );

    // TODO: deal with multiple traj_steps, multiple avoids?
    // TODO: deal with not having an object to avoid
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data.size(); gripper_ind++ )
    {
        collision_avoidance_results[gripper_ind].distance = grippers_data[(size_t)gripper_ind].distance_to_obstacle;
        std::cout << "collision dist: " << grippers_data[(size_t)gripper_ind].distance_to_obstacle << std::endl;

        // If we have a collision to avoid, then find the vector
        if ( !std::isinf( grippers_data[(size_t)gripper_ind].distance_to_obstacle ) )
        {
            // Create the collision Jacobian
            const Eigen::MatrixXd J_collision = computeCollisionToGripperJacobian( grippers_data[gripper_ind] );
            const Eigen::MatrixXd J_collision_inv = EigenHelpers::Pinv( J_collision, EigenHelpers::SuggestedRcond() );

            // Create the collision avoidance vector to follow
            const Eigen::Vector3d& avoid_collision_delta = grippers_data[gripper_ind].obstacle_surface_normal;

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
 * Computes a  Jacobian that converts a gripper velocitie in the individual
 * gripper frame into a point velocity in the world frame
 * @param gripper_data
 * @return
 */
Eigen::MatrixXd DiminishingRigidityModel::computeCollisionToGripperJacobian( const GripperData &gripper_data ) const
{
    Eigen::MatrixXd J_collision = Eigen::MatrixXd::Zero( 3, cols_per_gripper_ );

    // Translation - if I move the gripper in x,y,z, what happens to the given point?
    J_collision.block< 3, 3>( 0, 0 ) = gripper_data.pose.rotation();

    // TODO find out of these are at all correct
//    else if(i == 3)
//        transvec =  (gripper->getWorldTransform()*btVector4(1,0,0,0)).cross(
//                    points_in_world_frame[k] - gripper->getWorldTransform().getOrigin());
//    else if(i == 4)
//        transvec =  (gripper->getWorldTransform()*btVector4(0,1,0,0)).cross(
//                    points_in_world_frame[k] - gripper->getWorldTransform().getOrigin());
//    else if(i == 5)
//        transvec =  (gripper->getWorldTransform()*btVector4(0,0,1,0)).cross(
//                    points_in_world_frame[k] - gripper->getWorldTransform().getOrigin());

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
            if ( node_distance_delta( first_node, second_node ) - stretching_correction_threshold_ > 0 )
            {
                Eigen::Vector3d correction_vector = 0.5 * node_distance_delta( first_node, second_node )
                        * ( object_current_configuration.block< 3, 1 >( 0, second_node ) - object_current_configuration.block< 3, 1 >( 0, first_node ) );

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
