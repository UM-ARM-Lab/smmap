#ifndef ADAPTIVE_JACOBIAN_MODEL_H
#define ADAPTIVE_JACOBIAN_MODEL_H

#include "smmap/jacobian_model.h"

namespace smmap
{
    class AdaptiveJacobianModel final : public JacobianModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            AdaptiveJacobianModel(const Eigen::MatrixXd& initial_jacobian,
                                  const double learning_rate,
                                  const bool optimize);

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////


        private:

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel_impl(const WorldState& previous, const WorldState& next) override final;

            virtual Eigen::MatrixXd computeGrippersToDeformableObjectJacobian_impl(const DeformableModelInputData &input_data) const override final;

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            Eigen::MatrixXd current_jacobian_;
            const double learning_rate_;
    };
}

#endif // ADAPTIVE_JACOBIAN_MODEL_H
