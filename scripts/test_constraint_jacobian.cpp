#include <smmap/constraint_jacobian_model.h>

using namespace smmap;

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "test_constraint_jacobian");
  auto nh = std::make_shared<ros::NodeHandle>();
  auto sdf = std::make_shared<sdf_tools::SignedDistanceField const>();

  std::vector<GripperData> grippers;
  grippers.emplace_back("left_tool", std::vector<long>{0});
  ConstraintJacobianModel::SetGrippersData(grippers);
  ObjectPointSet initial_state(3, 4);
  initial_state << 0, 0.02, 0.04, 0.06, 0, 0, 0, 0, 0, 0, 0, 0;
  ConstraintJacobianModel::SetInitialObjectConfiguration(initial_state);

  auto gripper_collision_callback = [](const AllGrippersSinglePose&) { return std::vector<CollisionData>{}; };
  ConstraintJacobianModel::SetCallbackFunctions(gripper_collision_callback);

  ConstraintJacobianModel model(nh, 1, 1, 1, sdf);
}