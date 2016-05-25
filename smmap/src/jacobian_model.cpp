#include "smmap/jacobian_model.h"
#include "smmap/optimization.hpp"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

JacobianModel::JacobianModel(bool optimize)
    : optimize_(optimize)
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
        double dt) const
{
    assert(grippers_pose_trajectory.size() > 0);
    assert(grippers_pose_delta_trajectory.size() == grippers_pose_trajectory.size() - 1);
    (void)dt;

    ObjectTrajectory object_traj(grippers_pose_trajectory.size());
    object_traj[0] = world_initial_state.object_configuration_;

    for (size_t time_ind = 0; time_ind < grippers_pose_delta_trajectory.size(); time_ind++)
    {
        object_traj[time_ind + 1] = object_traj[time_ind] + getObjectDelta(
                    object_traj[time_ind],
                    grippers_pose_trajectory[time_ind],
                    grippers_pose_delta_trajectory[time_ind]);
    }

    return object_traj;
}






////////////////////////////////////////////////////////////////////////////////
// Liner model solver verion
////////////////////////////////////////////////////////////////////////////////

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
        double dt) const
{
    assert(gripper_pose_trajectory.size() > 0);
    assert(gripper_pose_delta_trajectory.size() == gripper_pose_trajectory.size() - 1);
    (void)dt;

    ObjectPointSet final_configuration = world_initial_state.object_configuration_;

    for (size_t time_ind = 0; time_ind < gripper_pose_delta_trajectory.size(); time_ind++)
    {
        final_configuration += getObjectDelta(
                    final_configuration,
                    gripper_pose_trajectory[time_ind],
                    gripper_pose_delta_trajectory[time_ind]);
    }

    return final_configuration;
}

std::pair<AllGrippersPoseTrajectory, ObjectTrajectory>
JacobianModel::getSuggestedGrippersTrajectory(
        const WorldState& world_initial_state,
        const int planning_horizion,
        const double dt,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale) const
{
    // Make a copy of the world state because we'll mutate it
    // TODO: is this true for all models and thus the prototype should change?
    WorldState world_current_state = world_initial_state;

    const double max_step_size = max_gripper_velocity * dt;
    const ssize_t num_grippers = (ssize_t)grippers_data_.size();
    const long num_nodes = world_current_state.object_configuration_.cols();

    std::pair<AllGrippersPoseTrajectory, ObjectTrajectory> suggested_traj =
            std::make_pair<AllGrippersPoseTrajectory, ObjectTrajectory>(
                AllGrippersPoseTrajectory(
                    (size_t)planning_horizion + 1,
                    AllGrippersSinglePose(grippers_data_.size())),
                ObjectTrajectory(
                    (size_t)planning_horizion + 1,
                    ObjectPointSet(3, num_nodes)));

    // Initialize the starting point of the trajectory with the current gripper
    // poses and object configuration
    suggested_traj.first[0] = world_current_state.all_grippers_single_pose_;
    suggested_traj.second[0] = world_current_state.object_configuration_;

    for (int traj_step = 1; traj_step <= planning_horizion; traj_step++)
    {
        ////////////////////////////////////////////////////////////////////////
        // Find the velocities of each part of the algorithm
        ////////////////////////////////////////////////////////////////////////

        // Retrieve the desired object velocity (p_dot)
        const ObjectDeltaAndWeight desired_object_velocity = task_desired_object_delta_fn_(world_current_state);

        // Recalculate the jacobian at each timestep, because of rotations being non-linear
        const Eigen::MatrixXd jacobian =
                computeGrippersToObjectJacobian(suggested_traj.first[(size_t)traj_step-1], suggested_traj.second[(size_t)traj_step-1]);

        // Find the least-squares fitting to the desired object velocity
        #pragma message "More magic numbers - damping threshold and damping coefficient"
        Eigen::VectorXd grippers_delta_achieve_goal =
                ClampGripperPoseDeltas(
                    EigenHelpers::WeightedLeastSquaresSolver(jacobian, desired_object_velocity.delta, desired_object_velocity.weight, 1e-3, 1e-2),
                    max_step_size);

        ObjectFinalConfigurationPredictionFunctionType prediction_fn = std::bind(
                    &DeformableModel::getFinalConfiguration,
                    this,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    std::placeholders::_3,
                    std::placeholders::_4);

        ErrorFunctionDerivitiveType derivitive_fn = std::bind(
                    &ErrorFunctionNumericalDerivitive,
                    std::placeholders::_1,
                    std::placeholders::_2,
                    std::placeholders::_3,
                    error_fn_,
                    prediction_fn,
                    std::placeholders::_4);

        if (optimize_)
        {
            // Optimize using the least squares result as the seed
            AllGrippersPoseTrajectory optimized_grippers_delta_traj = OptimizeTrajectoryDirectShooting(
                        world_current_state,
                        CalculateGrippersTrajectory(world_current_state.all_grippers_single_pose_, grippers_delta_achieve_goal),
                        error_fn_,
                        derivitive_fn,
                        prediction_fn,
                        max_step_size,
                        dt);
            // Confirm that we get only a single timestep output as we are only passing a single timestep in
            assert(optimized_grippers_delta_traj.size() == 2);
            grippers_delta_achieve_goal = EigenHelpersConversions::VectorEigenVectorToEigenVectorX(CalculateGrippersPoseDeltas(optimized_grippers_delta_traj)[0]);
        }

        // Find the collision avoidance data that we'll need
        const std::vector<CollisionAvoidanceResult> grippers_collision_avoidance_result =
                ComputeGripperObjectAvoidance(world_current_state.gripper_collision_data_, suggested_traj.first[(size_t)traj_step-1], max_step_size);

        // Store the predicted object change for use in later loops
        Eigen::MatrixXd object_delta = Eigen::MatrixXd::Zero(num_nodes * 3, 1);

        ////////////////////////////////////////////////////////////////////////
        // Combine the velocities into a single command velocity
        ////////////////////////////////////////////////////////////////////////
        for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
        {
            const kinematics::Vector6d actual_gripper_delta =
                    CombineDesiredAndObjectAvoidance(
                        grippers_delta_achieve_goal.segment<6>(gripper_ind * 6),
                        grippers_collision_avoidance_result[(size_t)gripper_ind],
                        obstacle_avoidance_scale);

            suggested_traj.first[(size_t)traj_step][(size_t)gripper_ind] =
                        suggested_traj.first[(size_t)traj_step - 1][(size_t)gripper_ind] *
                        kinematics::expTwistAffine3d(actual_gripper_delta, 1);

            object_delta += jacobian.block(0, 6 * gripper_ind, num_nodes * 3, 6) * actual_gripper_delta;
        }

        // Store our prediction in the output data structure
        object_delta.resizeLike(suggested_traj.second[(size_t)traj_step]);

        // Assume that our Jacobian is correct, and predict where we will end up (if needed)
        suggested_traj.second[(size_t)traj_step] = suggested_traj.second[(size_t)traj_step-1] + object_delta;

        ////////////////////////////////////////////////////////////////////////
        // If we are going to do more steps, mutate the current world state
        ////////////////////////////////////////////////////////////////////////

        if (traj_step + 1 < planning_horizion)
        {
            world_current_state.object_configuration_ = suggested_traj.second[(size_t)traj_step];
            world_current_state.all_grippers_single_pose_ = suggested_traj.first[(size_t)traj_step];
            world_current_state.gripper_collision_data_ = gripper_collision_check_fn_(world_current_state.all_grippers_single_pose_);
            world_current_state.sim_time_ += dt;
        }
    }

    return suggested_traj;
}


////////////////////////////////////////////////////////////////////////////////
// Non-linear model solver verion
////////////////////////////////////////////////////////////////////////////////

//std::pair<AllGrippersPoseTrajectory, ObjectTrajectory>
//JacobianModel::getSuggestedGrippersTrajectory(
//        const WorldState& world_initial_state,
//        const int planning_horizion,
//        const double dt,
//        const double max_gripper_velocity,
//        const double obstacle_avoidance_scale) const
//{

//    // Make a copy of the world state because we'll mutate it
//    // TODO: is this true for all models and thus the prototype should change?
//    WorldState world_current_state = world_initial_state;

//    const double max_step_size = max_gripper_velocity * dt;
//    const ssize_t num_grippers = (ssize_t)grippers_data_.size();
//    const long num_nodes = world_initial_state.object_configuration_.cols();

//    auto suggested_traj =
//            std::make_pair<AllGrippersPoseTrajectory, ObjectTrajectory>(
//                AllGrippersPoseTrajectory(
//                    (size_t)planning_horizion + 1,
//                    AllGrippersSinglePose(grippers_data_.size())),
//                ObjectTrajectory(
//                    (size_t)planning_horizion + 1,
//                    ObjectPointSet(3, num_nodes)));

//    // Initialize the starting point of the trajectory with the current gripper
//    // poses and object configuration
//    suggested_traj.first[0] = world_current_state.all_grippers_single_pose_;
//    suggested_traj.second[0] = world_current_state.object_configuration_;

//    for (int traj_step = 1; traj_step <= planning_horizion; traj_step++)
//    {
//        ////////////////////////////////////////////////////////////////////////
//        // Find the velocities of each part of the algorithm
//        ////////////////////////////////////////////////////////////////////////

//        // Retrieve the desired object velocity (p_dot)
//        const std::pair<Eigen::VectorXd, Eigen::VectorXd> desired_object_velocity
//                = task_desired_object_delta_fn_(world_current_state);

//        // Recalculate the jacobian at each timestep, because of rotations being non-linear
//        Eigen::MatrixXd jacobian = computeGrippersToObjectJacobian(
//                suggested_traj.first[(size_t)traj_step-1],
//                suggested_traj.second[(size_t)traj_step-1]);

//        // Find the least-squares fitting to the desired object velocity
//        #warning "More magic numbers - damping threshold and damping coefficient"
//        const double damping_thresh = 1e-3;
//        const double damping_ratio = 1e-2;

//        Eigen::VectorXd grippers_velocity_achieve_goal =
//                EigenHelpers::WeightedLeastSquaresSolver(
//                    jacobian,
//                    desired_object_velocity.first,
//                    desired_object_velocity.second,
//                    damping_thresh,
//                    damping_ratio);
//        grippers_velocity_achieve_goal = ClampGripperVelocities(grippers_velocity_achieve_goal, max_step_size);
//        std::cerr << "\n\n\nInitial singular values\n    " << jacobian.jacobiSvd().singularValues().transpose() << std::endl;
//        std::cerr << "\n\n\n Initial Gripper vel:\n    " << grippers_velocity_achieve_goal.transpose() << std::endl;

//        int ind = 0;
//        double gripper_velocity_change = 0;
//        do
//        {
//            #pragma GCC diagnostic push
//            #pragma GCC diagnostic ignored "-Wconversion"
//            const Eigen::VectorXd predicted_object_delta =
//                    jacobian * grippers_velocity_achieve_goal;
//            #pragma GCC diagnostic pop

//            // project the movement into the constraints of the world
//            const Eigen::VectorXd projected_object_delta =
//                    task_object_delta_projection_fn_(
//                        world_current_state.object_configuration_,
//                        predicted_object_delta);

//           const Eigen::MatrixXd projection_function_gradient =
//                   computeNonlinearProjectionGradient(
//                       world_current_state.object_configuration_,
//                       projected_object_delta,
//                       jacobian,
//                       grippers_velocity_achieve_goal);

//            // find a new gripper movement
//            // Find the least-squares fitting to the "missing" desired object velocity
//            const Eigen::VectorXd grippers_velocity_delta =
//                    EigenHelpers::WeightedLeastSquaresSolver(
//                        projection_function_gradient,
//                        desired_object_velocity.first - projected_object_delta,
//                        desired_object_velocity.second,
//                        damping_thresh,
//                        damping_ratio);
//            Eigen::VectorXd next_grippers_velocity_achieve_goal = grippers_velocity_achieve_goal + grippers_velocity_delta;
//            std::cerr << "Singular values\n    " << jacobian.jacobiSvd().singularValues().transpose() << std::endl;
//            grippers_velocity_achieve_goal = ClampGripperVelocities(next_grippers_velocity_achieve_goal, max_step_size);

//            gripper_velocity_change = MultipleGrippersVelocity6dNorm(next_grippers_velocity_achieve_goal - grippers_velocity_achieve_goal);
//            grippers_velocity_achieve_goal = next_grippers_velocity_achieve_goal;
//            std::cerr << "grippers_vel_change: " << gripper_velocity_change << std::endl;
//            std::cerr << "Gripper vel:\n    " << grippers_velocity_achieve_goal.transpose() << std::endl;
//            std::cerr << std::endl;

//            ind++;
//        }
//        while (ind < 100 && gripper_velocity_change > 1e-9);

//        ////////////////////////////////////////////////////////////////////////
//        // End Jacobian updating
//        ////////////////////////////////////////////////////////////////////////

//        // Find the collision avoidance data that we'll need
//        std::vector<CollisionAvoidanceResult> grippers_collision_avoidance_result
//                = ComputeGripperObjectAvoidance(
//                    world_current_state.gripper_collision_data_,
//                    suggested_traj.first[(size_t)traj_step-1], max_step_size);

//        // Store the predicted object change for use in later loops
//        Eigen::MatrixXd object_delta = Eigen::MatrixXd::Zero(num_nodes * 3, 1);

//        ////////////////////////////////////////////////////////////////////////
//        // Combine the velocities into a single command velocity
//        ////////////////////////////////////////////////////////////////////////
//        for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
//        {
//            kinematics::Vector6d actual_gripper_velocity;
//            const kinematics::Vector6d desired_gripper_vel =
//                    grippers_velocity_achieve_goal.segment<6>(gripper_ind * 6);

//            // If we need to avoid an obstacle, then use the sliding scale
//            const CollisionAvoidanceResult& collision_result =
//                    grippers_collision_avoidance_result[(size_t)gripper_ind];
//            if (!std::isinf(collision_result.distance))
//            {
//                 const double collision_severity =
//                         std::min(1.0, std::exp(-obstacle_avoidance_scale * collision_result.distance));

//                 actual_gripper_velocity =
//                         collision_severity * (collision_result.velocity
//                                                + collision_result.nullspace_projector * desired_gripper_vel)
//                         + (1 - collision_severity) * desired_gripper_vel;
//            }
//            // Otherwise use our desired velocity directly
//            else
//            {
//                 actual_gripper_velocity = desired_gripper_vel;
//            }

//            suggested_traj.first[(size_t)traj_step][(size_t)gripper_ind] =
//                        suggested_traj.first[(size_t)traj_step - 1][(size_t)gripper_ind] *
//                        kinematics::expTwistAffine3d(actual_gripper_velocity, 1);

//            object_delta += jacobian.block(0, 6 * gripper_ind, num_nodes * 3, 6) * actual_gripper_velocity;
//        }

//        // Store our prediction in the output data structure
//        object_delta.resizeLike(suggested_traj.second[(size_t)traj_step]);

//        // Assume that our Jacobian is correct, and predict where we will end up (if needed)
//        suggested_traj.second[(size_t)traj_step] = suggested_traj.second[(size_t)traj_step-1] + object_delta;

//        ////////////////////////////////////////////////////////////////////////
//        // If we are going to do more steps, mutate the current world state
//        ////////////////////////////////////////////////////////////////////////

//        if (traj_step + 1 < planning_horizion)
//        {
//            world_current_state.object_configuration_ =
//                    suggested_traj.second[(size_t)traj_step];
//            world_current_state.all_grippers_single_pose_ =
//                    suggested_traj.first[(size_t)traj_step];
//            world_current_state.gripper_collision_data_ =
//                    gripper_collision_check_fn_(
//                        world_current_state.all_grippers_single_pose_);
//            world_current_state.sim_time_ += dt;
//        }
//    }

//    return suggested_traj;
//}









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
        const AllGrippersSinglePoseDelta& grippers_pose_delta) const
{
    const Eigen::MatrixXd J = computeGrippersToObjectJacobian(
                grippers_pose,
                object_initial_configuration);

    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(object_initial_configuration.cols() * 3, 1);

    // Move the object based on the movement of each gripper
    for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        delta += J.block(0, 6 * (long)gripper_ind, J.rows(), 6)
                * grippers_pose_delta[gripper_ind];
    }

    delta.resizeLike(object_initial_configuration);
    return delta;
}

/**
 * @brief DiminishingRigidityModel::computeNonlinearProjectionGradient
 * @param jacobian
 * @param grippers_velocity
 * @return
 */
Eigen::MatrixXd JacobianModel::computeNonlinearProjectionGradient(
        const ObjectPointSet& current_object_configuration,
        const Eigen::VectorXd& current_object_velocity,
        const Eigen::MatrixXd& jacobian,
        Eigen::VectorXd current_grippers_velocity) const
{
    Eigen::MatrixXd projection_function_jacobian;
    projection_function_jacobian.resizeLike(jacobian);

    // roughly on the order of 1e-5 times smaller than expected gripper velocities
    const double h = 1e-10;

    // Lets try the central difference first;
    for (ssize_t col_ind = 0; col_ind < current_grippers_velocity.rows(); col_ind++)
    {
        current_grippers_velocity(col_ind) += h;

        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wconversion"
        const Eigen::VectorXd next_object_delta =
                task_object_delta_projection_fn_(
                    current_object_configuration,
                    jacobian * current_grippers_velocity);
        #pragma GCC diagnostic pop
        projection_function_jacobian.col(col_ind) = (next_object_delta - current_object_velocity) / h;

        current_grippers_velocity(col_ind) -= h;
    }

    return projection_function_jacobian;
}
