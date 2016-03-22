#include "smmap/least_squares_jacobian_model.h"


#include <ros/ros.h>

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

LeastSquaresJacobianModel::LeastSquaresJacobianModel(
        const Eigen::MatrixXd& initial_jacobian )
    : current_jacobian_( initial_jacobian )
    , next_buffer_ind_( 0 )
    , buffer_size_( initial_jacobian.cols() + 2 )
    , buffer_full_( false )
    , grippers_delta_wide_matrix_( initial_jacobian.cols(), buffer_size_ )
    , deformable_delta_wide_matrix_( initial_jacobian.rows(), buffer_size_ )
{
    ROS_INFO_NAMED( "least_squares_jacobian", "Creating least squares jacobian" );
}

////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////

void LeastSquaresJacobianModel::updateModel( const std::vector< WorldState >& feedback )
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

        Eigen::MatrixXd true_delta =
                feedback[ind + 1].object_configuration_
                - feedback[ind].object_configuration_;
        true_delta.resize( current_jacobian_.rows(), 1 );

        if ( grippers_delta.squaredNorm() < 1e-6 )
        {
            ROS_WARN_STREAM_NAMED( "least_squares_jacobian", "Grippers did not move much, not updating: squared norm vel: " << grippers_delta.squaredNorm() );
        }
        else
        {
            ROS_INFO_NAMED( "least_squares_jacobian", "Adding data to buffer" );
            grippers_delta_wide_matrix_.col( next_buffer_ind_ ) = grippers_delta;

            deformable_delta_wide_matrix_.col( next_buffer_ind_ ) = true_delta;

            next_buffer_ind_++;
            if ( next_buffer_ind_ >= buffer_size_ )
            {
                next_buffer_ind_ = 0;
            }

            if ( next_buffer_ind_ == 0 )
            {
                buffer_full_ = true;
            }
        }
    }

    if ( buffer_full_ )
    {
        ROS_INFO_NAMED( "least_squares_jacobian", "Updating jacobian" );
        current_jacobian_ = deformable_delta_wide_matrix_ *
                EigenHelpers::Pinv( grippers_delta_wide_matrix_, EigenHelpers::SuggestedRcond() );
    }
}

void LeastSquaresJacobianModel::perturbModel( std::mt19937_64& generator )
{}

////////////////////////////////////////////////////////////////////
// Computation helpers
////////////////////////////////////////////////////////////////////

Eigen::MatrixXd LeastSquaresJacobianModel::computeGrippersToObjectJacobian(
        const AllGrippersSinglePose& grippers_pose,
        const ObjectPointSet& current_configuration ) const
{
    (void)grippers_pose;
    (void)current_configuration;
    return current_jacobian_;
}
