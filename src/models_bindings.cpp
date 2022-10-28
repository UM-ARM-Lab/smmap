#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <smmap/constraint_jacobian_model.h>

#include <cstdint>

namespace py = pybind11;

using namespace smmap;

smmap::ConstraintJacobianModel make_cjm(const ObjectPointSet &initial_object_state,
                                        const std::vector<GripperData> &grippers) {
  auto nh = std::make_shared<ros::NodeHandle>();
  auto sdf = std::make_shared<sdf_tools::SignedDistanceField const>();

  std::vector<CollisionData> no_collisions;
  ConstraintJacobianModel::SetGrippersData(grippers);
  ConstraintJacobianModel::SetInitialObjectConfiguration(initial_object_state);

  auto gripper_collision_callback = [&](const AllGrippersSinglePose &) { return no_collisions; };
  ConstraintJacobianModel::SetCallbackFunctions(gripper_collision_callback);

  return {nh, 0.1, 0.1, 0.1, sdf};
}

std::vector<WorldState> dynamics(ConstraintJacobianModel const &model, std::vector<WorldState> const &initial_states,
                                 std::vector<AllGrippersPoseDeltaTrajectory> const &action_trajectories) {
  if (initial_states.size() != action_trajectories.size()) {
    throw std::runtime_error("mismatching batch size for states and action_trajectories");
  }

  std::vector<WorldState> outputs;

  for (auto b{0u}; b < initial_states.size(); ++b) {
    auto const &initial_state = initial_states[b];
    auto const &action_trajectory = action_trajectories[b];

    WorldState current_state{initial_state};

    for (auto t{0u}; t < action_trajectory.size(); ++t) {
      auto const &grippers_pose_delta_t = action_trajectory[t];
      auto const object_delta = model.getObjectDelta(current_state, grippers_pose_delta_t);
      auto const next_object_state = current_state.object_configuration_ + object_delta;

      const AllGrippersSinglePose next_gripper_state =
          kinematics::applyTwist(current_state.all_grippers_single_pose_, grippers_pose_delta_t);

      current_state.object_configuration_ = next_object_state;
      current_state.all_grippers_single_pose_ = next_gripper_state;

      outputs.emplace_back(current_state);
    }
  }

  return outputs;
}

PYBIND11_MODULE(pysmmap_models, m) {
  Py_Initialize();

  // Bindings
  py::class_<Eigen::Isometry3d>(m, "Isometry3d").def(py::init<Eigen::Matrix4d>());

  m.def("make_cjm", &make_cjm);
  m.def("dynamics", &dynamics);

  py::class_<WorldState>(m, "WorldState")
      .def(py::init<ObjectPointSet, AllGrippersSinglePose>())
      .def_readwrite("object_configuration", &WorldState::object_configuration_)
      .def_readwrite("all_grippers_single_pose", &WorldState::all_grippers_single_pose_)
      .def("__repr__", [](WorldState const &w) {
        std::stringstream ss;
        ss << "grippers: \n";
        for (auto const &pose : w.all_grippers_single_pose_) {
          ss << pose.matrix() << "\n";
        }
        ss << "\n object: \n";
        ss << " " << w.object_configuration_;
        return ss.str();
      });

  py::class_<ConstraintJacobianModel>(m, "ConstraintJacobianModel")
      .def("getObjectDelta", &ConstraintJacobianModel::getObjectDelta);

  py::class_<GripperData>(m, "GripperData").def(py::init<const std::string &, const std::vector<long> &>());
}