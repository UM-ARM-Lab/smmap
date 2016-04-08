#include "smmap/jacobian_model.h"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

JacobianModel::JacobianModel()
{}

////////////////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief JacobianModel::getPrediction
 * @param object_configuration
 * @param grippers_trajectory
 * @param grippers_velocities Note that velocity 0 moves us from pose 0 to pose 1 for a given gripper
 * @return
 */
ObjectTrajectory JacobianModel::getPrediction(
        const WorldState& world_initial_state,
        const AllGrippersPoseTrajectory & grippers_pose_trajectory,
        const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory,
        double dt ) const
{
    assert( grippers_pose_trajectory.size() > 0 );
    assert( grippers_pose_delta_trajectory.size() == grippers_pose_trajectory.size() - 1 );
    (void)dt;

    ObjectTrajectory object_traj( grippers_pose_trajectory.size() );
    object_traj[0] = world_initial_state.object_configuration_;

    for ( size_t time_ind = 0; time_ind < grippers_pose_delta_trajectory.size(); time_ind++ )
    {
        object_traj[time_ind + 1] = object_traj[time_ind] + getObjectDelta(
                    object_traj[time_ind],
                    grippers_pose_trajectory[time_ind],
                    grippers_pose_delta_trajectory[time_ind] );
    }

    return object_traj;
}

/**
 * @brief JacobianModel::getFinalConfiguration
 * @param object_configuration
 * @param grippers_trajectory
 * @param grippers_velocities Note that velocity 0 moves us from pose 0 to pose 1 for a given gripper
 * @return
 */
ObjectPointSet JacobianModel::getFinalConfiguration(
        const WorldState& world_initial_state,
        const AllGrippersPoseTrajectory& gripper_pose_trajectory,
        const AllGrippersPoseDeltaTrajectory& gripper_pose_delta_trajectory,
        double dt ) const
{
    assert( gripper_pose_trajectory.size() > 0 );
    assert( gripper_pose_delta_trajectory.size() == gripper_pose_trajectory.size() - 1 );
    (void)dt;

    ObjectPointSet final_configuration = world_initial_state.object_configuration_;

    for ( size_t time_ind = 0; time_ind < gripper_pose_delta_trajectory.size(); time_ind++ )
    {
        final_configuration += getObjectDelta(
                    final_configuration,
                    gripper_pose_trajectory[time_ind],
                    gripper_pose_delta_trajectory[time_ind] );
    }

    return final_configuration;
}

std::pair< AllGrippersPoseTrajectory, ObjectTrajectory >
JacobianModel::getSuggestedGrippersTrajectory(
        const WorldState& world_initial_state,
        const int planning_horizion,
        const double dt,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale ) const
{
    // Make a copy of the world state because we'll mutate it
    // TODO: is this true for all models and thus the prototype should change?
    WorldState world_current_state = world_initial_state;

    const double max_step_size = max_gripper_velocity * dt;
    const long num_nodes = world_current_state.object_configuration_.cols();

    std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > suggested_traj =
            std::make_pair< AllGrippersPoseTrajectory, ObjectTrajectory >(
                AllGrippersPoseTrajectory(
                    (size_t)planning_horizion + 1,
                    AllGrippersSinglePose( grippers_data_.size() ) ),
                ObjectTrajectory(
                    (size_t)planning_horizion + 1,
                    ObjectPointSet( 3, num_nodes ) ) );

    // Initialize the starting point of the trajectory with the current gripper
    // poses and object configuration
    suggested_traj.first[0] = world_current_state.all_grippers_single_pose_;
    suggested_traj.second[0] = world_current_state.object_configuration_;

    for ( int traj_step = 1; traj_step <= planning_horizion; traj_step++ )
    {
        ////////////////////////////////////////////////////////////////////////
        // Find the velocities of each part of the algorithm
        ////////////////////////////////////////////////////////////////////////

        // Retrieve the desired object velocity (p_dot)
        const std::pair< Eigen::VectorXd, Eigen::VectorXd > desired_object_velocity
                = task_desired_object_delta_fn_( world_current_state );

        // Recalculate the jacobian at each timestep, because of rotations being non-linear
        const Eigen::MatrixXd jacobian = computeGrippersToObjectJacobian(
                suggested_traj.first[(size_t)traj_step-1],
                suggested_traj.second[(size_t)traj_step-1] );

        // Find the least-squares fitting to the desired object velocity
        #warning "More magic numbers - damping threshold and damping coefficient"
        Eigen::VectorXd grippers_velocity_achieve_goal =
                EigenHelpers::WeightedLeastSquaresSolver(
                    jacobian,
                    desired_object_velocity.first,
                    desired_object_velocity.second,
                    1e-3,
                    1e-2 );

        // Find the collision avoidance data that we'll need
        std::vector< CollisionAvoidanceResult > grippers_collision_avoidance_result
                = ComputeGripperObjectAvoidance(
                    world_current_state.gripper_collision_data_,
                    suggested_traj.first[(size_t)traj_step-1], max_step_size );

        // Store the predicted object change for use in later loops
        Eigen::MatrixXd object_delta = Eigen::MatrixXd::Zero( num_nodes * 3, 1 );

        ////////////////////////////////////////////////////////////////////////
        // Combine the velocities into a single command velocity
        ////////////////////////////////////////////////////////////////////////
        for ( long gripper_ind = 0; gripper_ind < (long)grippers_data_.size(); gripper_ind++ )
        {
            kinematics::Vector6d actual_gripper_velocity;
            kinematics::Vector6d desired_gripper_vel =
                    grippers_velocity_achieve_goal.segment< 6 >( gripper_ind * 6 );

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
                         collision_severity * ( collision_result.velocity
                                                + collision_result.nullspace_projector * desired_gripper_vel )
                         + (1 - collision_severity) * desired_gripper_vel;
            }
            // Otherwise use our desired velocity directly
            else
            {
                 actual_gripper_velocity = desired_gripper_vel;
            }

            suggested_traj.first[(size_t)traj_step][(size_t)gripper_ind] =
                        suggested_traj.first[(size_t)traj_step - 1][(size_t)gripper_ind] *
                        kinematics::expTwistAffine3d( actual_gripper_velocity, 1 );

            object_delta += jacobian.block( 0, 6 * gripper_ind, num_nodes * 3, 6 ) * actual_gripper_velocity;
        }

        // Store our prediction in the output data structure
        object_delta.resizeLike( suggested_traj.second[(size_t)traj_step] );

        // Assume that our Jacobian is correct, and predict where we will end up (if needed)
        suggested_traj.second[(size_t)traj_step] = suggested_traj.second[(size_t)traj_step-1] + object_delta;

        ////////////////////////////////////////////////////////////////////////
        // If we are going to do more steps, mutate the current world state
        ////////////////////////////////////////////////////////////////////////

        if ( traj_step + 1 < planning_horizion )
        {
            world_current_state.object_configuration_ =
                    suggested_traj.second[(size_t)traj_step];
            world_current_state.all_grippers_single_pose_ =
                    suggested_traj.first[(size_t)traj_step];
            world_current_state.gripper_collision_data_ =
                    gripper_collision_check_fn_(
                        world_current_state.all_grippers_single_pose_ );
            world_current_state.sim_time_ += dt;
        }
    }

    return suggested_traj;
}

////////////////////////////////////////////////////////////////////////////////
// Computation helpers
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief JacobianModel::getObjectDelta
 * @param object_initial_configuration
 * @param grippers_pose
 * @param grippers_pose_delta
 * @return
 */
ObjectPointSet JacobianModel::getObjectDelta(
        const ObjectPointSet& object_initial_configuration,
        const AllGrippersSinglePose & grippers_pose,
        const AllGrippersSinglePoseDelta& grippers_pose_delta ) const
{
    const Eigen::MatrixXd J = computeGrippersToObjectJacobian(
                grippers_pose,
                object_initial_configuration );

    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero( object_initial_configuration.cols() * 3, 1 );

    // Move the object based on the movement of each gripper
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        delta += J.block( 0, 6 * (long)gripper_ind, J.rows(), 6 )
                * grippers_pose_delta[gripper_ind];
    }

    delta.resizeLike( object_initial_configuration );
    return delta;
}
