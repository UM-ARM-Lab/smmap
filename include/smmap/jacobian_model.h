#ifndef JACOBIAN_MODEL_H
#define JACOBIAN_MODEL_H

#include "smmap/deformable_model.h"

namespace smmap
{
    class JacobianModel : public DeformableModel
    {
        public:
            struct JacobianInputData
            {
                public:
                    JacobianInputData(
                            const TaskDesiredObjectDeltaFunctionType& task_desired_object_delta_fn,
                            const WorldState& world_initial_state)
                        : task_desired_object_delta_fn_(task_desired_object_delta_fn)
                        , world_initial_state_(world_initial_state)
                    {}

                    const TaskDesiredObjectDeltaFunctionType& task_desired_object_delta_fn_;
                    const WorldState& world_initial_state_;
            };

            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            JacobianModel(bool optimize);

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual ObjectPointSet getObjectDelta(const JacobianInputData &input_data,
                    const double dt) const override final;

            virtual std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getSuggestedGrippersCommand(
                    TaskDesiredObjectDeltaFunctionType task_desired_object_delta_fn,
                    const WorldState& world_initial_state,
                    const double dt,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale) const override final;

        protected:

            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            ObjectPointSet getObjectDelta(const JacobianInputData &input_data,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta) const;

            virtual Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const JacobianInputData& input_data) const = 0;

            /*
            virtual Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const AllGrippersSinglePose& grippers_pose,
                    const ObjectPointSet& current_configuration) const = 0;
            */

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            // Controls if we perform an optimization pass between the weighted pseudo inverse and the gripper collision avoidance
            bool optimize_;
    };
}

#endif // DIMINISHING_RIGIDITY_MODEL_H
