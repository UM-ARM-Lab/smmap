#include <smmap/constraint_jacobian_model.h>

#include <iomanip>

using namespace smmap;

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "test_constraint_jacobian");
  auto nh = std::make_shared<ros::NodeHandle>();
  auto sdf = std::make_shared<sdf_tools::SignedDistanceField const>();

  std::vector<CollisionData> no_collisions;
  std::vector<GripperData> grippers;
  grippers.emplace_back("left_tool", std::vector<long>{0});
  ConstraintJacobianModel::SetGrippersData(grippers);
  ObjectPointSet previous_object_state(3, 4);
  previous_object_state << -0.01, 0.01, 0.03, 0.05, 0, 0, 0, 0, 0, 0, 0, 0;
  ObjectPointSet current_object_state(3, 4);
  current_object_state << 0, 0.02, 0.04, 0.06, 0, 0, 0, 0, 0, 0, 0, 0;
  ConstraintJacobianModel::SetInitialObjectConfiguration(current_object_state);

  auto gripper_collision_callback = [&](const AllGrippersSinglePose&) { return no_collisions; };
  ConstraintJacobianModel::SetCallbackFunctions(gripper_collision_callback);

  ConstraintJacobianModel model(nh, 0.1, 0.1, 0.1, sdf);

  Eigen::VectorXd robot_configuration(1);
  robot_configuration(0) = 0;
  AllGrippersSinglePose gripper_poses;
  Eigen::Isometry3d left_gripper_pose = Eigen::Isometry3d::Identity();
  gripper_poses.emplace_back(left_gripper_pose);
  WorldState current_state{current_object_state, gripper_poses, no_collisions};
  WorldState previous_state{previous_object_state, gripper_poses, no_collisions};

  kinematics::Vector6d left_gripper_delta;
  left_gripper_delta << 0.01, 0, 0, 0, 0, 0;
  AllGrippersSinglePoseDelta grippers_pose_delta;
  grippers_pose_delta.emplace_back(left_gripper_delta);

  for (auto i{0}; i < 10; ++i) {
    auto const object_delta = model.getObjectDelta(current_state, grippers_pose_delta);
    auto const next_object_state = current_state.object_configuration_ + object_delta;
    AllGrippersSinglePose next_gripper_state = current_state.all_grippers_single_pose_;
    next_gripper_state[0].translation() += left_gripper_delta.head<3>();

    current_state.object_configuration_ = next_object_state;
    current_state.all_grippers_single_pose_ = next_gripper_state;

    std::cout << "next state: " << std::setprecision(3) << next_object_state << '\n';
  }
}