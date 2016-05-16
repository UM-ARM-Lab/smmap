#ifndef LEAST_SQUARES_JACOBIAN_MODEL_H
#define LEAST_SQUARES_JACOBIAN_MODEL_H

#include "smmap/jacobian_model.h"

namespace smmap
{
    class LeastSquaresJacobianModel final : public JacobianModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            LeastSquaresJacobianModel(const Eigen::MatrixXd& initial_jacobian,
                                       const long extra_samples);

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel(const std::vector< WorldState >& feedback) override;

            virtual void perturbModel(std::mt19937_64& generator) override;

        private:

            ////////////////////////////////////////////////////////////////////
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            virtual Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const AllGrippersSinglePose& grippers_pose,
                    const ObjectPointSet& current_configuration) const override;

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            Eigen::MatrixXd current_jacobian_;

            long next_buffer_ind_;
            const long buffer_size_;
            bool buffer_full_;
            Eigen::MatrixXd grippers_delta_wide_matrix_;
            Eigen::MatrixXd deformable_delta_wide_matrix_;
    };
}

#endif // LEAST_SQUARES_JACOBIAN_MODEL_H
