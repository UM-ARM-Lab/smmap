#include "smmap/adaptive_jacobian_model.h"

#include <ros/ros.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpersConversions;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

AdaptiveJacobianModel::AdaptiveJacobianModel(
        const MatrixXd& initial_jacobian,
        const double learning_rate,
        const bool optimize)
    : JacobianModel(optimize)
    , current_jacobian_(initial_jacobian)
    , learning_rate_(learning_rate)
{
    ROS_INFO_STREAM_NAMED("adaptive_jacobian", "Creating adaptive jacobian with learning rate " << learning_rate_);
}

////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////

void AdaptiveJacobianModel::updateModel_impl(const WorldState& previous, const WorldState& next)
{
    const AllGrippersSinglePoseDelta grippers_pose_deltas =
            CalculateGrippersPoseDelta(previous.all_grippers_single_pose_,
                                       next.all_grippers_single_pose_);

    const VectorXd grippers_delta = VectorEigenVectorToEigenVectorX(grippers_pose_deltas);

    if (grippers_delta.squaredNorm() < 1e-20)
    {
        ROS_WARN_STREAM_NAMED("adaptive_jacobian", "Grippers did not move, not updating: squared norm vel: " << grippers_delta.squaredNorm());
    }
    else
    {
        // Suppress a warning on type conversion related to Eigen operations
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
        const Eigen::VectorXd predicted_delta = current_jacobian_ * grippers_delta;
#pragma GCC diagnostic pop

        MatrixXd true_delta = next.object_configuration_ - previous.object_configuration_;
        true_delta.resize(current_jacobian_.rows(), 1);

        // Perform the update
        current_jacobian_ += learning_rate_ *
                (true_delta - predicted_delta) * grippers_delta.transpose()
                / (grippers_delta.transpose() * grippers_delta);
    }
}

Eigen::MatrixXd AdaptiveJacobianModel::computeGrippersToDeformableObjectJacobian_impl(
        const DeformableModelInputData &input_data) const
{
    const AllGrippersSinglePose& grippers_pose = input_data.world_initial_state_.all_grippers_single_pose_;
    const ObjectPointSet& current_configuration = input_data.world_initial_state_.object_configuration_;
//    WorldState& world_state = input_data.world_initial_state_;
    (void)(grippers_pose);
    (void)(current_configuration);
    return current_jacobian_;
}
