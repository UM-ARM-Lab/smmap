#include <smmap/constraint_jacobian_model.h>

#include <iomanip>
#include <sdf_tools/collision_map.hpp>

using namespace smmap;

int main(int argc, char* argv[]) {
  ros::init(argc, argv, "test_constraint_jacobian");
  auto nh = std::make_shared<ros::NodeHandle>();

  ros::Publisher pub = nh->advertise<visualization_msgs::MarkerArray>("test_cjm", 10);

  std::vector<CollisionData> no_collisions;
  std::vector<GripperData> grippers;
  grippers.emplace_back("left_tool", std::vector<long>{0});
  ConstraintJacobianModel::SetGrippersData(grippers);
  ObjectPointSet current_object_state(3, 8);
  current_object_state << 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
      0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02;
  ConstraintJacobianModel::SetInitialObjectConfiguration(current_object_state);

  auto gripper_collision_callback = [&](const AllGrippersSinglePose&) { return no_collisions; };
  ConstraintJacobianModel::SetCallbackFunctions(gripper_collision_callback);

  sdf_tools::COLLISION_CELL oob_cell;
  oob_cell.occupancy = 0.0;
  oob_cell.component = 0;
  float oob_value = INFINITY;
  int64_t x_shape = 100;
  int64_t y_shape = 100;
  int64_t z_shape = 100;
  Eigen::Isometry3d origin_transform = Eigen::Isometry3d::Identity();
  //  origin_transform(0, 3) = -0.05;
  //  origin_transform(1, 3) = -0.05;
  float res = 0.01f;
  sdf_tools::CollisionMapGrid vg(origin_transform, "world", res, x_shape, y_shape, z_shape, oob_cell);
  sdf_tools::COLLISION_CELL occupied(1);
  for (int64_t xIndex = 2; xIndex < 10; ++xIndex) {
    for (int64_t yIndex = 3; yIndex < 10; ++yIndex) {
      for (int64_t zIndex = 4; zIndex < 10; ++zIndex) {
        vg.SetValue(xIndex, yIndex, zIndex, occupied);
      }
    }
  }

  auto result = vg.ExtractSignedDistanceField(oob_value, false, false);
  auto sdf = result.first;
  auto sdf_ptr = std::make_shared<sdf_tools::SignedDistanceField const>(sdf);

  ConstraintJacobianModel model(nh, 20, 20, 20, sdf_ptr);

  AllGrippersSinglePose gripper_poses;
  Eigen::Isometry3d left_gripper_pose = Eigen::Isometry3d::Identity();
  gripper_poses.emplace_back(left_gripper_pose);
  WorldState current_state{current_object_state, gripper_poses};

  kinematics::Vector6d left_gripper_delta;
  left_gripper_delta << 0, 0.004, 0.007, 0, 0, 0;
  AllGrippersSinglePoseDelta grippers_pose_delta;
  grippers_pose_delta.emplace_back(left_gripper_delta);

  visualization_msgs::Marker spheres;
  spheres.id = 0;
  spheres.ns = "spheres";
  spheres.type = visualization_msgs::Marker::CUBE_LIST;
  spheres.pose.orientation.w = 1;
  spheres.scale.x = res;
  spheres.scale.y = res;
  spheres.scale.z = res;
  spheres.color.a = 0.5;
  spheres.header.frame_id = "world";
  spheres.action = visualization_msgs::Marker::ADD;
  for (int64_t x_idx{0}; x_idx < x_shape; ++x_idx) {
    for (int64_t y_idx{0}; y_idx < y_shape; ++y_idx) {
      for (int64_t z_idx{0}; z_idx < z_shape; ++z_idx) {
        auto location = vg.GridIndexToLocation(x_idx, y_idx, z_idx);
        if (sdf.EstimateDistance3d(location.head<3>()).first <= 0) {
          geometry_msgs::Point point_msg;
          point_msg.x = location.x();
          point_msg.y = location.y();
          point_msg.z = location.z();
          spheres.points.push_back(point_msg);
        }
      }
    }
  }
  visualization_msgs::MarkerArray msg;
  msg.markers.push_back(spheres);

  for (auto i{0}; i < 5; ++i) {
    pub.publish(msg);
    ros::spinOnce();
  }

  for (auto t{0}; t < 7; ++t) {
    auto const object_delta = model.getObjectDelta(current_state, grippers_pose_delta);
    auto const next_object_state = current_state.object_configuration_ + object_delta;
    AllGrippersSinglePose next_gripper_state = current_state.all_grippers_single_pose_;
    next_gripper_state[0].translation() += left_gripper_delta.head<3>();

    current_state.object_configuration_ = next_object_state;
    current_state.all_grippers_single_pose_ = next_gripper_state;

    //    std::cout << "next state:\n" << std::setprecision(3) << next_object_state << '\n';
    visualization_msgs::Marker rope;
    rope.id = 2 + t;
    rope.ns = "rope";
    rope.type = visualization_msgs::Marker::LINE_STRIP;
    rope.pose.orientation.w = 1;
    rope.scale.x = res / 4;
    rope.scale.y = res / 4;
    rope.scale.z = res / 4;
    rope.color.a = 1;
    rope.color.r = 1;
    rope.color.g = 0;
    rope.color.b = 1;
    rope.header.frame_id = "world";
    rope.action = visualization_msgs::Marker::ADD;
    for (auto col{0}; col < next_object_state.cols(); ++col) {
      Eigen::Vector3d p = next_object_state.col(col);
      geometry_msgs::Point point_msg;
      point_msg.x = p.x();
      point_msg.y = p.y();
      point_msg.z = p.z();
      rope.points.push_back(point_msg);
    }
    visualization_msgs::MarkerArray rope_msg;
    rope_msg.markers.push_back(rope);
    pub.publish(rope_msg);
    ros::spinOnce();
  }
}