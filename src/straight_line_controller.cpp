#include "smmap/straight_line_controller.h"

using namespace smmap;

StraightLineController::StraightLineController(
        ros::NodeHandle& ph,
        const DeformableModel::Ptr& model,
        const RobotInterface::Ptr& robot)
    : DeformableController(robot)
    , model_(model)
{
    kinematics::Vector6d gripper_delta = kinematics::Vector6d::Zero();
    gripper_delta(0) = GetGripperStraightLineMotionTransX(ph);
    gripper_delta(1) = GetGripperStraightLineMotionTransY(ph);
    gripper_delta(2) = GetGripperStraightLineMotionTransZ(ph);
    gripper_delta(3) = GetGripperStraightLineMotionAngularX(ph);
    gripper_delta(4) = GetGripperStraightLineMotionAngularY(ph);
    gripper_delta(5) = GetGripperStraightLineMotionAngularZ(ph);

    const auto num_grippers = robot_->getGrippersData().size();
    grippers_motion_ = AllGrippersSinglePoseDelta(num_grippers, gripper_delta);
}

DeformableController::OutputData StraightLineController::getGripperMotion_impl(
        const InputData& input_data)
{
    AllGrippersSinglePoseDelta cmd(1, kinematics::Vector6d::Zero());

    cmd[0](0) = grippers_motion_[0](0);

    if (input_data.world_current_state_.sim_time_ > 4.3)
    {
        cmd[0](5) = grippers_motion_[0](5);
    }

    if (input_data.world_current_state_.sim_time_ > 4.9)
    {
        cmd[0](5) = 0.0;
    }

    const auto prediction = model_->getObjectDelta(input_data.world_current_state_, cmd);
    return OutputData(cmd, prediction, Eigen::VectorXd(0));
}