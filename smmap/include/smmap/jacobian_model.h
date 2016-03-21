#ifndef JACOBIAN_MODEL_H
#define JACOBIAN_MODEL_H

#include "smmap/deformable_model.h"

namespace smmap
{
    // TODO: find a way to accept dynamic/online gripper re-grasping
    class JacobianModel : public DeformableModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            JacobianModel();

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual ObjectTrajectory getPrediction(
                    const WorldState& world_initial_state,
                    const AllGrippersPoseTrajectory& grippers_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory,
                    const double dt ) const;

            virtual ObjectPointSet getFinalConfiguration(
                    const WorldState& world_initial_state,
                    const AllGrippersPoseTrajectory& gripper_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& gripper_pose_delta_trajectory,
                    const double dt ) const ;

            virtual std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > getSuggestedGrippersTrajectory(
                    const WorldState& world_initial_state,
                    const int planning_horizion,
                    const double dt,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale ) const;

        private:

            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            ObjectPointSet getObjectDelta(
                    const ObjectPointSet& object_initial_configuration,
                    const AllGrippersSinglePose& grippers_pose,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta ) const;

            virtual Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const AllGrippersSinglePose& grippers_pose,
                    const ObjectPointSet& current_configuration ) const = 0;

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////
    };
}

#endif // DIMINISHING_RIGIDITY_MODEL_H
