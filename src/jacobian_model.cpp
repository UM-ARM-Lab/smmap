#include "smmap/jacobian_model.h"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

JacobianModel::JacobianModel()
{}

Eigen::MatrixXd JacobianModel::computeGrippersToDeformableObjectJacobian(
        const WorldState& world_state) const
{
    return computeGrippersToDeformableObjectJacobian_impl(world_state);
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
        const WorldState& world_state,
        const AllGrippersSinglePoseDelta& grippers_pose_delta) const
{
    const MatrixXd J = computeGrippersToDeformableObjectJacobian_impl(world_state);

    MatrixXd delta = MatrixXd::Zero(world_state.object_configuration_.cols() * 3, 1);

    // Move the object based on the movement of each gripper
    for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        delta += J.block(0, 6 * (ssize_t)gripper_ind, J.rows(), 6) * grippers_pose_delta[gripper_ind];
    }

    delta.resizeLike(world_state.object_configuration_);
    return delta;
}
