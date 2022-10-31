#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <smmap/constraint_jacobian_model.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <sdf_tools/collision_map.hpp>
#include <utility>

namespace py = pybind11;

using namespace smmap;

smmap::ConstraintJacobianModel make_cjm(const ObjectPointSet &initial_object_state,
                                        const std::vector<GripperData> &grippers,
                                        sdf_tools::SignedDistanceField::ConstPtr const &sdf_ptr,
                                        double const translation_dir_deformability,
                                        double const translation_dis_deformability,
                                        double const rotation_deformability) {
  auto nh = std::make_shared<ros::NodeHandle>();

  assert(initial_object_state.rows() == 3);
  for (auto const &gripper : grippers) {
    for (auto const &i : gripper.node_indices_) {
      assert(i < initial_object_state.cols());
    }
  }

  std::vector<CollisionData> no_collisions;
  ConstraintJacobianModel::SetGrippersData(grippers);
  ConstraintJacobianModel::SetInitialObjectConfiguration(initial_object_state);

  auto gripper_collision_callback = [&](const AllGrippersSinglePose &) { return no_collisions; };
  ConstraintJacobianModel::SetCallbackFunctions(gripper_collision_callback);

  return {nh, translation_dir_deformability, translation_dis_deformability, rotation_deformability, sdf_ptr};
}

smmap::ConstraintJacobianModel make_cjm_no_sdf(const ObjectPointSet &initial_object_state,
                                               const std::vector<GripperData> &grippers,
                                               double const translation_dir_deformability,
                                               double const translation_dis_deformability,
                                               double const rotation_deformability) {
  auto sdf_ptr = std::make_shared<sdf_tools::SignedDistanceField const>();
  return make_cjm(initial_object_state, grippers, sdf_ptr, translation_dir_deformability, translation_dis_deformability,
                  rotation_deformability);
}

smmap::ConstraintJacobianModel make_cjm_with_sdf(const ObjectPointSet &initial_object_state,
                                                 const std::vector<GripperData> &grippers,
                                                 Eigen::Isometry3d const &origin_transform, double res,
                                                 std::vector<std::vector<std::vector<bool>>> vg_data,
                                                 double const translation_dir_deformability,
                                                 double const translation_dis_deformability,
                                                 double const rotation_deformability) {
  auto nh = std::make_shared<ros::NodeHandle>();

  sdf_tools::COLLISION_CELL oob_cell;
  oob_cell.occupancy = 0.0;
  oob_cell.component = 0;
  float oob_value = INFINITY;
  auto x_shape = static_cast<int64_t>(vg_data.size());
  auto y_shape = static_cast<int64_t>(vg_data[0].size());
  auto z_shape = static_cast<int64_t>(vg_data[0][0].size());
  sdf_tools::CollisionMapGrid vg(origin_transform, "world", res, x_shape, y_shape, z_shape, oob_cell);
  for (int64_t x_idx{0}; x_idx < x_shape; ++x_idx) {
    for (int64_t y_idx{0}; y_idx < y_shape; ++y_idx) {
      for (int64_t z_idx{0}; z_idx < z_shape; ++z_idx) {
        auto const value = static_cast<float>(vg_data[x_idx][y_idx][z_idx]);
        sdf_tools::COLLISION_CELL cell(value);
        vg.SetValue(x_idx, y_idx, z_idx, cell);
      }
    }
  }
  auto result = vg.ExtractSignedDistanceField(oob_value, false, false);
  auto sdf = result.first;
  auto sdf_ptr = std::make_shared<sdf_tools::SignedDistanceField const>(sdf);

  return make_cjm(initial_object_state, grippers, sdf_ptr, translation_dir_deformability, translation_dis_deformability,
                  rotation_deformability);
}

std::vector<WorldState> dynamics(ConstraintJacobianModel const &model, std::vector<WorldState> const &initial_states,
                                 std::vector<AllGrippersSinglePoseDelta> const &actions) {
  if (initial_states.size() != actions.size()) {
    throw std::runtime_error("mismatching batch size for states and action_trajectories");
  }

  std::vector<WorldState> outputs;

  for (auto b{0u}; b < initial_states.size(); ++b) {
    auto const &initial_state = initial_states[b];
    auto const &grippers_pose_delta = actions[b];

    WorldState current_state{initial_state};

    auto const object_delta = model.getObjectDelta(current_state, grippers_pose_delta);
    auto const next_object_state = current_state.object_configuration_ + object_delta;

    const AllGrippersSinglePose next_gripper_state =
        kinematics::applyTwist(current_state.all_grippers_single_pose_, grippers_pose_delta);

    WorldState next_state{next_object_state, next_gripper_state};

    outputs.emplace_back(next_state);
  }

  return outputs;
}

PYBIND11_MODULE(pysmmap_models, m) {
  Py_Initialize();

  // Bindings
  py::class_<Eigen::Isometry3d>(m, "Isometry3d")
      .def(py::init<Eigen::Matrix4d>())
      .def("matrix", py::overload_cast<>(&Eigen::Isometry3d::matrix, py::const_))
      .def("__repr__", [](Eigen::Isometry3d const &i) {
        std::stringstream ss;
        ss << i.matrix() << '\n';
        return ss.str();
      });

  m.def("make_cjm_no_sdf", &make_cjm_no_sdf, py::arg("initial_object_state"), py::arg("grippers"),
        py::arg("translation_dir_deformability") = 10.0, py::arg("translation_dis_deformability") = 1.0,
        py::arg("rotation_deformability") = 1.0);
  m.def("make_cjm_with_sdf", &make_cjm_with_sdf, py::arg("initial_object_state"), py::arg("grippers"),
        py::arg("origin_transform"), py::arg("res"), py::arg("vg_data"),
        py::arg("translation_dir_deformability") = 10.0, py::arg("translation_dis_deformability") = 1.0,
        py::arg("rotation_deformability") = 1.0);
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

  py::class_<GripperData>(m, "GripperData")
      .def(py::init<const std::string &, const std::vector<long> &>())
      .def("__repr__", [](GripperData const &g) {
        std::stringstream ss;
        ss << g.name_ << '\n';
        for (auto const &i : g.node_indices_) {
          ss << i << ',';
        }
        ss << '\n';
        return ss.str();
      });
}