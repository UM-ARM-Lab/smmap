#ifndef LEAST_SQUARES_JACOBIAN_MODEL_H
#define LEAST_SQUARES_JACOBIAN_MODEL_H

#include "smmap/jacobian_model.h"

namespace smmap
{
    class LeastSquaresJacobianModel : public JacobianModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            LeastSquaresJacobianModel( const Eigen::MatrixXd& initial_jacobian );

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel( const std::vector< WorldState >& feedback );

            void perturbModel( std::mt19937_64& generator );

        private:

            ////////////////////////////////////////////////////////////////////
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            virtual Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const AllGrippersSinglePose& grippers_pose,
                    const ObjectPointSet& current_configuration ) const;

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
