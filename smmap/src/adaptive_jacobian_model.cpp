#include "smmap/adaptive_jacobian_model.h"


#include <ros/ros.h>

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

AdaptiveJacobianModel::AdaptiveJacobianModel(
        const Eigen::MatrixXd& initial_jacobian,
        const double learning_rate )
    : current_jacobian_( initial_jacobian )
    , learning_rate_( learning_rate )
{
    ROS_INFO_STREAM_NAMED( "adaptive_jacobian", "Creating adaptive jacobian with learning rate " << learning_rate_ );
}

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
        Eigen::VectorXd grippers_delta( current_jacobian_.cols() );
        for ( size_t gripper_ind = 0; gripper_ind < grippers_pose_deltas[ind].size(); gripper_ind++ )
        {
            grippers_delta.segment< 6 >( (long)gripper_ind * 6  ) =
                    grippers_pose_deltas[ind][gripper_ind];
        }

        if ( grippers_delta.squaredNorm() < 1e-20 )
        {
            ROS_WARN_STREAM_NAMED( "adaptive_jacobian", "Grippers did not move, not updating: squared norm vel: " << grippers_delta.squaredNorm() );
        }
        else
        {
            // Suppress a warning on type conversion related to Eigen operations
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wconversion"
            const Eigen::VectorXd predicted_delta = current_jacobian_ * grippers_delta;
            #pragma GCC diagnostic pop

            Eigen::MatrixXd true_delta =
                    feedback[ind + 1].object_configuration_
                    - feedback[ind].object_configuration_;
            true_delta.resize( current_jacobian_.rows(), 1 );

            // Perform the update
            current_jacobian_ += learning_rate_ *
                    ( true_delta - predicted_delta ) * grippers_delta.transpose()
                    / grippers_delta.squaredNorm();
        }
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
