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

            JacobianModel();

            Eigen::MatrixXd computeGrippersToDeformableObjectJacobian(
                    const DeformableModelInputData& input_data) const;

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
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual ObjectPointSet getObjectDelta_impl(
                    const DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta) const override final;
    };
}

#endif // DIMINISHING_RIGIDITY_MODEL_H
