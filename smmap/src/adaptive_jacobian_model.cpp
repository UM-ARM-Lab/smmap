#include "smmap/adaptive_jacobian_model.h"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

AdaptiveJacobianModel::AdaptiveJacobianModel(
        const Eigen::MatrixXd& initial_jacobian,
        const double learning_rate )
    : current_jacobian_( initial_jacobian )
//    , num_nodes_( initial_jacobian.rows() / 3 )
    , learning_rate_( learning_rate )
{}

////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////

void AdaptiveJacobianModel::updateModel( const std::vector< WorldState >& feedback )
{
    const AllGrippersPoseDeltaTrajectory grippers_pose_deltas =
            CalculateGrippersPoseDeltas( GetGripperTrajectories( feedback ) );

    for ( size_t ind = 0; ind < feedback.size() - 1; ind++ )
    {
        // Collect the data we need
        Eigen::MatrixXd true_delta =
                feedback[ind + 1].object_configuration_
                - feedback[ind].object_configuration_;
        Eigen::MatrixXd predicted_delta = getObjectDelta(
                    feedback[ind].object_configuration_,
                    feedback[ind].all_grippers_single_pose_,
                    grippers_pose_deltas[ind] );

        // Reformat the data into the vectors we need
        true_delta.resize( current_jacobian_.rows(), 1 );
        predicted_delta.resize( current_jacobian_.rows(), 1 );

        Eigen::VectorXd grippers_delta( current_jacobian_.cols() );
        for ( size_t gripper_ind = 0; gripper_ind < grippers_pose_deltas.size(); gripper_ind++ )
        {
            grippers_delta.segment< 6 >( (long)gripper_ind * 6  ) =
                    grippers_pose_deltas[ind][gripper_ind];
        }

        // Perform the update
        current_jacobian_ = current_jacobian_ + learning_rate_ *
                ( true_delta - predicted_delta ) * grippers_delta.transpose()
                / grippers_delta.norm();
    }
}

void AdaptiveJacobianModel::perturbModel( std::mt19937_64& generator )
{}

////////////////////////////////////////////////////////////////////
// Computation helpers
////////////////////////////////////////////////////////////////////

Eigen::MatrixXd AdaptiveJacobianModel::computeGrippersToObjectJacobian(
        const AllGrippersSinglePose& grippers_pose,
        const ObjectPointSet& current_configuration ) const
{
    (void)grippers_pose;
    (void)current_configuration;
    return current_jacobian_;
}
