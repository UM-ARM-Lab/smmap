#include "smmap/least_squares_controller_with_object_avoidance.h"
#include <smmap_utilities/gurobi_solvers.h>
#include "smmap/jacobian_model.h"

using namespace smmap;
using namespace smmap_utilities;
using namespace Eigen;
using namespace EigenHelpers;

#pragma message "Magic number - damping threshold and damping coefficient"
#define LEAST_SQUARES_DAMPING_THRESHOLD (1e-4)
#define LEAST_SQUARES_DAMPING_VALUE     (1e-3)

LeastSquaresControllerWithObjectAvoidance::LeastSquaresControllerWithObjectAvoidance(
        const DeformableModel::Ptr& model,
        const double obstacle_avoidance_scale,
        const bool optimize)
    : model_(model)
    , obstacle_avoidance_scale_(obstacle_avoidance_scale)
    , optimize_(optimize)
{
    // TODO: Why can't I just put this cast inside the constructor and define model_ to be a JacobianModel::Ptr?
    assert(std::dynamic_pointer_cast<JacobianModel>(model_) != nullptr && "Invalid model type passed to constructor");
}

std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> LeastSquaresControllerWithObjectAvoidance::getGripperMotion_impl(
        const DeformableModel::DeformableModelInputData& input_data,
        const double max_gripper_velocity)
{
    const auto& grippers_data = DeformableModel::GetGrippersData();

    const double max_step_size = max_gripper_velocity * input_data.dt_;
    const size_t num_grippers = grippers_data.size();
    const ssize_t num_nodes = input_data.world_current_state_.object_configuration_.cols();

    ////////////////////////////////////////////////////////////////////////
    // Find the velocities of each part of the algorithm
    ////////////////////////////////////////////////////////////////////////

    // Retrieve the desired object velocity (p_dot)
    const ObjectDeltaAndWeight desired_object_velocity =
            input_data.desired_object_motion_;

    // Recalculate the jacobian at each timestep, because of rotations being non-linear
    const auto test = std::static_pointer_cast<JacobianModel>(model_);
    const MatrixXd jacobian = test->computeGrippersToDeformableObjectJacobian(input_data);

    // Find the least-squares fitting to the desired object velocity
    VectorXd grippers_delta_achieve_goal;
    if (optimize_)
    {
        grippers_delta_achieve_goal =
                minSquaredNormSE3VelocityConstraints(jacobian, desired_object_velocity.delta, max_step_size, desired_object_velocity.weight);
    }
    else
    {
        grippers_delta_achieve_goal =
            ClampGripperPoseDeltas(
                WeightedLeastSquaresSolver(jacobian, desired_object_velocity.delta, desired_object_velocity.weight, LEAST_SQUARES_DAMPING_THRESHOLD, LEAST_SQUARES_DAMPING_VALUE),
                max_step_size);
    }

    // Find the collision avoidance data that we'll need
    const std::vector<CollisionAvoidanceResult> grippers_collision_avoidance_result =
            ComputeGripperObjectAvoidance(
                input_data.world_current_state_.gripper_collision_data_,
                input_data.world_current_state_.all_grippers_single_pose_,
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
                obstacle_avoidance_scale_);

        object_delta_as_vector += jacobian.block(0, 6 * (ssize_t)gripper_ind, num_nodes * 3, 6) * suggested_grippers_command.first[gripper_ind];
    }

    return suggested_grippers_command;
}
