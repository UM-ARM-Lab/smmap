#include "smmap/jacobian_model.h"
#include "smmap/gurobi_solvers.h"
#include "smmap/timing.hpp"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

JacobianModel::JacobianModel()
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
//    Stopwatch stopwatch;
//    stopwatch(RESET);

    const MatrixXd J = computeGrippersToDeformableObjectJacobian_impl(input_data);

    MatrixXd delta = MatrixXd::Zero(input_data.world_current_state_.object_configuration_.cols() * 3, 1);

    // Move the object based on the movement of each gripper
    for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        delta += J.block(0, 6 * (ssize_t)gripper_ind, J.rows(), 6) * grippers_pose_delta[gripper_ind];
    }

    delta.resizeLike(input_data.world_current_state_.object_configuration_);

//    ROS_INFO_STREAM_NAMED("jacobian_model", "Calculated predicted p_dot in  " << stopwatch(READ) << " seconds");
    return delta;
}
