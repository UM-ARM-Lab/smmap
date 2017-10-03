#include "smmap/least_squares_controller_with_object_avoidance.h"
#include "smmap/gurobi_solvers.h"
#include "smmap/jacobian_model.h"

#include "smmap/ros_communication_helpers.hpp"

using namespace smmap;
using namespace Eigen;
using namespace EigenHelpers;

#pragma message "Magic number - damping threshold and damping coefficient"
#define LEAST_SQUARES_DAMPING_THRESHOLD (1e-4)
#define LEAST_SQUARES_DAMPING_VALUE     (1e-3)

LeastSquaresControllerWithObjectAvoidance::LeastSquaresControllerWithObjectAvoidance(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const DeformableModel::Ptr& model,
        const double obstacle_avoidance_scale,
        const bool optimize)
    : object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh)))
    , max_stretch_factor_(GetMaxStretchFactor(ph))
//    , max_grippers_distance_(GetClothYSize(nh) - 0.015)
    , num_grippers_(GetGrippersData(nh).size())
    , model_(model)
    , obstacle_avoidance_scale_(obstacle_avoidance_scale)
    , optimize_(optimize)
{
    // TODO: Why can't I just put this cast inside the constructor and define model_ to be a JacobianModel::Ptr?
    assert(std::dynamic_pointer_cast<JacobianModel>(model_) != nullptr && "Invalid model type passed to constructor");
    if (GetDeformableType(nh) == CLOTH)
    {
        max_grippers_distance_ = GetClothYSize(nh) - 0.015;
    }
    else if (GetDeformableType(nh) == ROPE)
    {
        max_grippers_distance_ = GetRopeSegmentLength(nh) * GetRopeNumLinks(nh);
    }
}

std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> LeastSquaresControllerWithObjectAvoidance::getGripperMotion_impl(
        const DeformableModel::DeformableModelInputData& input_data,
        const double max_gripper_velocity)
{
    const auto& grippers_data = DeformableModel::GetGrippersData();

    const double max_step_size = max_gripper_velocity * input_data.dt_;
    const size_t num_grippers = grippers_data.size();
    const ssize_t num_nodes = input_data.world_current_state_.object_configuration_.cols();

    // Check object current stretching status; only for evaluation of controller performance --- Added by Mengyao
    /*
    if (stretchingViolation(input_data.world_current_state_))
    {
        stretching_violation_count_++;
    }
    */

    ////////////////////////////////////////////////////////////////////////
    // Find the velocities of each part of the algorithm
    ////////////////////////////////////////////////////////////////////////

    // Retrieve the desired object velocity (p_dot)
   // const ObjectDeltaAndWeight desired_object_velocity =
   //         input_data.task_desired_object_delta_fn_(input_data.world_current_state_);
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

// stretching violation detection helper  --- Added by Mengyao
bool LeastSquaresControllerWithObjectAvoidance::stretchingViolation(const WorldState& current_world_state)
{
    const ObjectPointSet& current_object_configuration = current_world_state.object_configuration_;
    const AllGrippersSinglePose& current_grippers_poses = current_world_state.all_grippers_single_pose_;

    bool over_stretch = false;
    double max_stretching = 0.0;

    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(current_object_configuration);
    ssize_t num_nodes = current_object_configuration.cols();

    for (ssize_t first_node = 0; first_node < num_nodes; ++first_node)
    {
        for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
        {
            /*
            double this_stretching_factor = std::sqrt(node_squared_distance(first_node, second_node))
                    / object_initial_node_distance_(first_node, second_node);
            if (this_stretching_factor > max_stretching)
            {
                max_stretching = this_stretching_factor;
            }
            */
            const double max_distance = max_stretch_factor_ * object_initial_node_distance_(first_node, second_node);
            if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
            {
                over_stretch = true;
                break;
            }
        }
        if (over_stretch)
        {
            break;
        }
    }
    /*
    if (num_grippers_ == 2)
    {
        double this_stretching_factor = (current_grippers_poses.at(0).translation()
                - current_grippers_poses.at(1).translation()).norm()
                / max_grippers_distance_;
        if (this_stretching_factor > max_stretching)
        {
            max_stretching = this_stretching_factor;
        }
    }
    current_stretching_factor_ = max_stretching;
    */
    return over_stretch;
}

