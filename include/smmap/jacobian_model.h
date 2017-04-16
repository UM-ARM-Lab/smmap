#ifndef JACOBIAN_MODEL_H
#define JACOBIAN_MODEL_H

#include "smmap/deformable_model.h"

namespace smmap
{
    class JacobianModel : public DeformableModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            JacobianModel(bool optimize);

            Eigen::MatrixXd computeGrippersToDeformableObjectJacobian(
                    const DeformableModelInputData& input_data) const;

            ////////////////////////////////////////////////////////////////////
            // Null Projection Matrix   -----  EDIT BY Mengyao
            ////////////////////////////////////////////////////////////////////
            Eigen::MatrixXd computeObjectVelocityMask(const ObjectPointSet& current_configuration,
                    const Eigen::MatrixXd &object_p_dot) const;



        protected:
            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

        private:
            ////////////////////////////////////////////////////////////////////
            // Virtual functions sub-classes must define
            ////////////////////////////////////////////////////////////////////

            virtual Eigen::MatrixXd computeGrippersToDeformableObjectJacobian_impl(
                    const DeformableModelInputData& input_data) const = 0;

            ////////////////////////////////////////////////////////////////////
            // Null Projection Matrix   -----  EDIT BY Mengyao
            ////////////////////////////////////////////////////////////////////
            virtual Eigen::MatrixXd computeObjectVelocityMask_impl(
                    const ObjectPointSet& current_configuration,
                    const Eigen::MatrixXd &object_p_dot) const = 0;


            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual ObjectPointSet getObjectDelta_impl(
                    const DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta) const override final;


            virtual std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getSuggestedGrippersCommand_impl(
                    const DeformableModelInputData& input_data,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale) const override final;

            // Mengyao defined: Get ObjectDelta as a stacked vector, the above is the resized one
            virtual ObjectPointSet getProjectedObjectDelta_impl(const DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta, const ObjectPointSet &current_configuration) const override final;


            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            // Controls if we perform an optimization pass between the weighted pseudo inverse and the gripper collision avoidance
            bool optimize_;
    };
}

#endif // DIMINISHING_RIGIDITY_MODEL_H
