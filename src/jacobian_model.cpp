#include "smmap/jacobian_model.h"
#include "smmap/gurobi_solvers.h"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

JacobianModel::JacobianModel(bool optimize)
    : optimize_(optimize)
{}

Eigen::MatrixXd JacobianModel::computeGrippersToDeformableObjectJacobian(
        const DeformableModelInputData& input_data) const
{
    return computeGrippersToDeformableObjectJacobian_impl(input_data);
}

////////////////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief JacobianModel::getObjectDelta_impl
 * @param input_data
 * @param grippers_pose_delta
 * @return
 */
ObjectPointSet JacobianModel::getObjectDelta_impl(
        const DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& grippers_pose_delta) const
{
    const MatrixXd J = computeGrippersToDeformableObjectJacobian_impl(input_data);

    MatrixXd delta = MatrixXd::Zero(input_data.world_initial_state_.object_configuration_.cols() * 3, 1);

    // Move the object based on the movement of each gripper
    for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        delta += J.block(0, 6 * (ssize_t)gripper_ind, J.rows(), 6) * grippers_pose_delta[gripper_ind];
    }

    delta.resizeLike(input_data.world_initial_state_.object_configuration_);
    return delta;
}


/**
 * @brief JacobianModel::getSuggestedGrippersCommand_impl
 * @param input_data
 * @param max_gripper_velocity
 * @param obstacle_avoidance_scale
 * @return
 */
std::pair<AllGrippersSinglePoseDelta, ObjectPointSet>
JacobianModel::getSuggestedGrippersCommand_impl(
        const DeformableModelInputData& input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale) const
{
    const double max_step_size = max_gripper_velocity * input_data.dt_;
    const size_t num_grippers = grippers_data_.size();
    const ssize_t num_nodes = input_data.world_initial_state_.object_configuration_.cols();

    ////////////////////////////////////////////////////////////////////////
    // Find the velocities of each part of the algorithm
    ////////////////////////////////////////////////////////////////////////

    // Retrieve the desired object velocity (p_dot)
    const ObjectDeltaAndWeight desired_object_velocity =
            input_data.task_desired_object_delta_fn_(input_data.world_initial_state_);

    // Recalculate the jacobian at each timestep, because of rotations being non-linear
    const MatrixXd jacobian = computeGrippersToDeformableObjectJacobian_impl(input_data);

    // Find the least-squares fitting to the desired object velocity
    VectorXd grippers_delta_achieve_goal;
    if (optimize_)
    {
        grippers_delta_achieve_goal =
                minSquaredNorm(jacobian, desired_object_velocity.delta, max_step_size, desired_object_velocity.weight);
    }
    else
    {
        #pragma message "More magic numbers - damping threshold and damping coefficient"
        grippers_delta_achieve_goal =
            ClampGripperPoseDeltas(
                WeightedLeastSquaresSolver(jacobian, desired_object_velocity.delta, desired_object_velocity.weight, 1e-4, 1e-3),
                max_step_size);
    }

    // Find the collision avoidance data that we'll need
    const std::vector<CollisionAvoidanceResult> grippers_collision_avoidance_result =
            ComputeGripperObjectAvoidance(
                input_data.world_initial_state_.gripper_collision_data_,
                input_data.world_initial_state_.all_grippers_single_pose_,
                max_step_size);

    ////////////////////////////////////////////////////////////////////////
    // Combine the velocities into a single command velocity
    ////////////////////////////////////////////////////////////////////////

    std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> suggested_grippers_command(
                AllGrippersSinglePoseDelta(num_grippers),
                ObjectPointSet::Zero(3, num_nodes));

    // Remapped data array
    Map<VectorXd> object_delta_as_vector(suggested_grippers_command.second.data(), suggested_grippers_command.second.size());

    for (size_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
    {
        suggested_grippers_command.first[gripper_ind] =
                CombineDesiredAndObjectAvoidance(
                    grippers_delta_achieve_goal.segment<6>((ssize_t)gripper_ind * 6),
                    grippers_collision_avoidance_result[gripper_ind],
                obstacle_avoidance_scale);

        object_delta_as_vector += jacobian.block(0, 6 * (ssize_t)gripper_ind, num_nodes * 3, 6) * suggested_grippers_command.first[gripper_ind];
    }

    return suggested_grippers_command;
}
