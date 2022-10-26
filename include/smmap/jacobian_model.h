#ifndef JACOBIAN_MODEL_H
#define JACOBIAN_MODEL_H

#include "smmap/deformable_model.h"

namespace smmap {
class JacobianModel : public DeformableModel {
 public:
  ////////////////////////////////////////////////////////////////////
  // Constructors and Destructor
  ////////////////////////////////////////////////////////////////////

  JacobianModel(std::shared_ptr<ros::NodeHandle> nh);

  Eigen::MatrixXd computeGrippersToDeformableObjectJacobian(const WorldState& world_state) const;

 protected:
  ////////////////////////////////////////////////////////////////////
  // Static helpers
  ////////////////////////////////////////////////////////////////////

  static void ComputeObjectNodeDistanceMatrix();

 private:
  ////////////////////////////////////////////////////////////////////
  // Virtual functions sub-classes must define
  ////////////////////////////////////////////////////////////////////

  virtual Eigen::MatrixXd computeGrippersToDeformableObjectJacobian_impl(const WorldState& world_state) const = 0;

  ////////////////////////////////////////////////////////////////////
  // Virtual function overrides
  ////////////////////////////////////////////////////////////////////

  virtual ObjectPointSet getObjectDelta_impl(
      const WorldState& world_state, const AllGrippersSinglePoseDelta& grippers_pose_delta) const override final;
};
}  // namespace smmap

#endif  // DIMINISHING_RIGIDITY_MODEL_H
