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

DiminishingRigidityModel::DiminishingRigidityModel( const VectorGrippersData& grippers_data,
        const ObjectPointSet& object_initial_configuration,
        double obstacle_avoidance_scale, bool use_rotation,
        double rigidity )
    : DiminishingRigidityModel( grippers_data, object_initial_configuration,
                                obstacle_avoidance_scale, use_rotation, rigidity, rigidity )
{}

DiminishingRigidityModel::DiminishingRigidityModel(
        const VectorGrippersData& grippers_data,
        const ObjectPointSet& object_initial_configuration,
        double obstacle_avoidance_scale, bool use_rotation,
        double rigidity_translation, double rotation_rigidity )
    : object_initial_configuration_( object_initial_configuration )
    , obstacle_avoidance_scale_( obstacle_avoidance_scale )
    , translation_rigidity_( rigidity_translation )
    , rotation_rigidity_( rotation_rigidity )
    , use_rotation_( use_rotation )
    , cols_per_gripper_( use_rotation_ ? 6 : 3 )
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
    computeObjectToGripperJacobian( grippers_data );
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

/**
 * @brief DiminishingRigidityModel::computeObjectToGripperJacobian
 * Computes a Jacobian that takes object velocities in the world frame, and computes
 * gripper velocities in the gripper frame
 * @param grippers_data
 */
Eigen::MatrixXd DiminishingRigidityModel::computeObjectToGripperJacobian( const VectorGrippersData& grippers_data ) const
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

                J.block< 3, 3 >( node_ind * 3, gripper_ind * cols_per_gripper_ + 3 ) =
                        std::exp( -rotation_rigidity_ * dist_to_gripper.second ) * J_rot*50;///78.9;///78.886009230677;
            }
        }
    }

    std::cout << "Jacobian: translation_rigidity: " << translation_rigidity_ << " rotation_rigidity: " << rotation_rigidity_ << std::endl;
    std::cout << J << std::endl;

    return J;
}

Eigen::MatrixXd DiminishingRigidityModel::computeCollisionToGripperJacobian( const VectorGrippersData &grippers_data ) const
{
    Eigen::MatrixXd J_collision = Eigen::MatrixXd::Zero( 3, cols_per_gripper_ );

    // Translation
    J_collision.block< 3, 3>( 0, 0 ) = Eigen::Matrix3d::Identity();

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
        const AllGrippersTrajectory& grippers_trajectory,
        const std::vector< kinematics::VectorVector6d >& grippers_velocities ) const
{
    assert( grippers_trajectory.size() > 0 );
    assert( grippers_velocities.size() == grippers_trajectory.size() );

    ObjectTrajectory object_traj( grippers_trajectory[0].size() );
    object_traj[0] = object_configuration;

    for ( size_t vel_ind = 0; vel_ind < grippers_velocities[0].size(); vel_ind++ )
    {
        // create a cols_per_gripper_*num_gripprs by 1 vector that contains all the gripper velocities at this time step
        Eigen::VectorXd combined_gripper_vel( (long)grippers_velocities.size()*cols_per_gripper_ );
        for ( long gripper_ind = 0; gripper_ind < (long)grippers_velocities.size(); gripper_ind++ )
        {
            if ( use_rotation_ )
            {
                combined_gripper_vel.segment< 6 >( gripper_ind * 6 ) =
                    grippers_velocities[(size_t)gripper_ind][vel_ind];
            }
            else
            {
                combined_gripper_vel.segment< 3 >( gripper_ind * 3 ) =
                    grippers_velocities[(size_t)gripper_ind][vel_ind].segment< 3 >( 0 );
            }
        }

        // calculate the velocity of the object given the gripper velocity
        Eigen::MatrixXd delta_obj;// = J*combined_gripper_vel;
        delta_obj.resize( 3, object_configuration.cols() );

        object_traj[vel_ind + 1] = object_traj[vel_ind] + delta_obj;
    }

    return object_traj;
}

AllGrippersTrajectory DiminishingRigidityModel::doGetDesiredGrippersTrajectory(
        const ObjectPointSet& object_current_configuration,
        const ObjectPointSet& object_desired_configuration,
        VectorGrippersData grippers_data,
        double max_step_size, size_t num_steps ) const
{
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
        // Recalculate the jacobian at each timestep, because of rotations being non-linear
        const Eigen::MatrixXd J = computeObjectToGripperJacobian( grippers_data );
        const Eigen::MatrixXd J_inv = EigenHelpers::Pinv( J, EigenHelpers::SuggestedRcond() );

        Eigen::VectorXd avoid_object_collision_delta = Eigen::VectorXd::Zero( 3*(long)grippers_data.size() );
        Eigen::MatrixXd J_collision( 3*(long)grippers_data.size(), cols_per_gripper_ );

        // TODO: deal with multiple traj_steps, multiple avoids?
        // Object avoidance block
        for ( long gripper_ind = 0; gripper_ind < (long)grippers_data.size(); gripper_ind++ )
        {
            // Create the collision Jacobian
            J_collision.block( 3*gripper_ind, 0, 3, cols_per_gripper_ )
                    = computeCollisionToGripperJacobian( grippers_data );

            // TODO: this is checking vs infinity, but bullet is returning BT_LARGE_FLOAT. Fix this.
            // If we have a collision to avoid, then find the vector
            if ( !std::isinf( grippers_data[(size_t)gripper_ind].distance_to_obstacle ) )
            {
                // Create the collision avoidance vector to follow
                Eigen::Vector3d avoid_collision_delta =
                        grippers_data[(size_t)gripper_ind].nearest_point_on_gripper
                        - grippers_data[(size_t)gripper_ind].nearest_point_on_obstacle;

                // If we are already inside the obstacle, then we need to invert the
                // direction of movement
                if ( grippers_data[(size_t)gripper_ind].distance_to_obstacle < 0 )
                {
                    avoid_collision_delta = -avoid_collision_delta;
                }

                // Normalize the step size to avoid the obstacle to avoid potential over/underflow
                avoid_object_collision_delta.segment< 3 >( gripper_ind * 3 ) =
                        avoid_collision_delta / avoid_collision_delta.norm();
            }
            // Otherwise, leave that part of avoid_object_collision_delta at zero
            else {}
        }

        Eigen::MatrixXd J_collision_pinv = EigenHelpers::Pinv( J_collision, EigenHelpers::SuggestedRcond() );
        // pragmas are here to supress some warnings from GCC
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wconversion"
        // Apply the collision Jacobian pseudo-inverse - will normalize later
        Eigen::VectorXd grippers_velocity_avoid_collision = J_collision_pinv * avoid_object_collision_delta;
        // Apply the desired object delta Jacobian pseudo-inverse - will normalize later
        Eigen::VectorXd grippers_velocity_achieve_goal = J_inv * (current - desired);
        #pragma GCC diagnostic pop

        // Then normalize each individual gripper velocity, and combine into a single unified instruction
        Eigen::VectorXd actual_gripper_velocity( cols_per_gripper_ * (long)grippers_data.size() );
        for ( long gripper_ind = 0; gripper_ind < (long)grippers_data.size(); gripper_ind++ )
        {
            // TODO: this is checking vs infinity, but bullet is returning BT_LARGE_FLOAT. Fix this.
            // normalize the avoidance velocity
            if ( !std::isinf( grippers_data[(size_t)gripper_ind].distance_to_obstacle ) )
            {
                grippers_velocity_avoid_collision.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ) =
                        grippers_velocity_avoid_collision.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ )
                        / grippers_velocity_avoid_collision.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ).norm()
                        * max_step_size;
            }

            // normalize the achive goal velocity
            if ( grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ).norm() > max_step_size )
            {
                grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ) =
                        grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ )
                        / grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ).norm()
                        * max_step_size;
            }

            // Last, we combine the gripper velocities
            const double collision_severity = std::min( 1.0,
                        std::exp( -obstacle_avoidance_scale_* grippers_data[(size_t)gripper_ind].distance_to_obstacle ) );

            std::cout << "collision severity: " << collision_severity << std::endl;

            actual_gripper_velocity.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ) =
                    collision_severity * ( grippers_velocity_avoid_collision.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ) +
                                           ( Eigen::MatrixXd::Identity( cols_per_gripper_, cols_per_gripper_ ) - J_collision_pinv * J_collision ) *
                                           grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ ) )
                    + (1 - collision_severity) * grippers_velocity_achieve_goal.segment( gripper_ind * cols_per_gripper_, cols_per_gripper_ );
        }

        // Apply the velocity to each gripper
        for ( long gripper_ind = 0; gripper_ind < (long)grippers_data.size(); gripper_ind++ )
        {
            kinematics::Vector6d gripper_velocity;
            if ( use_rotation_ )
            {
                // TODO: confirm that this is not needed with the new J_
                // First move the translational velocity into the gripper frame
//                actual_gripper_velocity.segment< 3 >( gripper_ind * 6 ) =
//                        traj[(size_t)gripper_ind][traj_step - 1].rotation().transpose()
//                        * actual_gripper_velocity.segment< 3 >( gripper_ind * 6 );

                // then use the translated velocity
                gripper_velocity = actual_gripper_velocity.segment< 6 >( gripper_ind * 6 );
            }
            else
            {
                gripper_velocity << actual_gripper_velocity.segment< 3 >( gripper_ind * 6 ), 0, 0, 0;
            }

            traj[(size_t)gripper_ind].push_back(
                        traj[(size_t)gripper_ind][traj_step - 1] *
                        kinematics::expTwistAffine3d( gripper_velocity, 1 ) );

            grippers_data[(size_t)gripper_ind].pose = traj[(size_t)gripper_ind].back();

        }

        // TODO: do we need to worry about the "rotation vs. translation cancling" above?
        // Assume that our Jacobian is correct, and predict where we will end up
        current += J * actual_gripper_velocity;
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

// Static member initialization
////////////////////////////////////////////////////////////////////////////////

/// TODO: remove this magic number for the noise generated by this distribution
std::normal_distribution< double > DiminishingRigidityModel::perturbation_distribution =
    std::normal_distribution< double >( 0, 0.1 );

Eigen::MatrixXd DiminishingRigidityModel::object_initial_node_distance_;
