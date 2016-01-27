#ifndef DEFORMABLE_MODEL_H
#define DEFORMABLE_MODEL_H

#include <functional>
#include <memory>
#include <random>

#include "smmap/trajectory.hpp"
#include "smmap/gripper_helpers.hpp"
#include "smmap/task.hpp"

namespace smmap
{
    class DeformableModel
    {
        public:
            typedef std::shared_ptr< DeformableModel > Ptr;

            ////////////////////////////////////////////////////////////////////
            /// Virtual functions that define the interface
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel( const std::vector< WorldFeedback >& feedback ) = 0;

            virtual ObjectTrajectory getPrediction(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt ) const = 0;

            virtual ObjectPointSet getFinalConfiguration(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt ) const = 0;

            virtual std::vector< AllGrippersSinglePose > getDesiredGrippersTrajectory(
                    const WorldFeedback& world_feedback,
                    const ObjectPointSet& object_desired_configuration,
                    double max_step_size, size_t num_steps ) const = 0;

            virtual Eigen::VectorXd getObjectiveFunction1stDerivitive(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt,
                    std::function< double( const ObjectPointSet& ) > objective_function ) const
            {
                const double h = 0.0001; // arbitrary step size for numeric differencing
                const size_t num_grippers = grippers_data_.size();
                const size_t num_timesteps = grippers_velocities.size();

                assert( num_timesteps > 0 );
                assert( num_grippers > 0 );

                // Allocate some space to store the results of the differencing.
                Eigen::VectorXd derivitives( num_grippers * 6 * num_timesteps );

                // Note that I am following the math found on the Finite difference page of
                // Wikipedia for "finite difference in several variables"

                std::vector< AllGrippersSingleVelocity > new_grippers_velocities( grippers_velocities );

                // This loop fills out the Jacobian (first derivitive) of the objective function
                for ( long ind = 0; ind < (long)(num_grippers * 6 * num_timesteps); ind++ )
                {
                    const long time_ind = ind / ( num_grippers * 6 );
                    const long vel_ind = ind % ( num_grippers * 6 );

                    std::vector< AllGrippersSinglePose > new_grippers_trajectory;

                    // f(x + h, y)
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) += h;
                    new_grippers_trajectory = CalculateGrippersTrajectory( grippers_trajectory[0], new_grippers_velocities, dt );
                    const double objective_value_x_plus_h = objective_function(
                                getFinalConfiguration( current_world_configuration,
                                                       new_grippers_trajectory,
                                                       new_grippers_velocities,
                                                       dt ) );
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) -= h;

                    // f(x - h, y)
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) -= h;
                    new_grippers_trajectory = CalculateGrippersTrajectory( grippers_trajectory[0], new_grippers_velocities, dt );
                    const double objective_value_x_minus_h = objective_function(
                                getFinalConfiguration( current_world_configuration,
                                                       new_grippers_trajectory,
                                                       new_grippers_velocities,
                                                       dt ) );
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) += h;

                    // f_x = [ f(x + h, y) - f(x - h, y) ] / ( 2h )
                    derivitives( ind ) = ( objective_value_x_plus_h - objective_value_x_minus_h ) / ( 2*h );
                }

                return derivitives;
            }

            virtual std::pair< Eigen::VectorXd, Eigen::MatrixXd > getObjectiveFunction2ndDerivitive(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt,
                    std::function< double( const ObjectPointSet& ) > objective_function ) const
            {
                const double h = 0.0001; // arbitrary step size for numeric differencing
                const size_t num_grippers = grippers_data_.size();
                const size_t num_timesteps = grippers_velocities.size();

                assert( num_timesteps > 0 );
                assert( num_grippers > 0 );

                const double initial_objective_value = objective_function(
                            getFinalConfiguration( current_world_configuration,
                                                   grippers_trajectory,
                                                   grippers_velocities,
                                                   dt ) );

                // Allocate some space to store the results of the differencing.
                std::pair< Eigen::VectorXd, Eigen::MatrixXd > derivitives (
                            Eigen::VectorXd( num_grippers * 6 * num_timesteps ),
                            Eigen::MatrixXd( num_grippers * 6 * num_timesteps, num_grippers * 6 * num_timesteps ) );

                // Note that I am following the math found on the Finite difference page of
                // Wikipedia for "finite difference in several variables"

                // First calculate all of the perturbations for a single variable
                Eigen::VectorXd objective_value_x_plus_h( num_grippers * 6 * num_timesteps );
                Eigen::VectorXd objective_value_x_minus_h( num_grippers * 6 * num_timesteps );
                std::vector< AllGrippersSingleVelocity > new_grippers_velocities( grippers_velocities );

                // This loop fills out the Jacobian (first derivitive) of the objective function
                for ( long ind = 0; ind < (long)(num_grippers * 6 * num_timesteps); ind++ )
                {
                    const long time_ind = ind / ( num_grippers * 6 );
                    const long vel_ind = ind % ( num_grippers * 6 );

                    std::vector< AllGrippersSinglePose > new_grippers_trajectory;

                    // f(x + h, y)
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) += h;
                    new_grippers_trajectory = CalculateGrippersTrajectory(
                            grippers_trajectory[0],
                            new_grippers_velocities,
                            dt );
                    objective_value_x_plus_h( ind ) = objective_function(
                                getFinalConfiguration( current_world_configuration,
                                                       new_grippers_trajectory,
                                                       new_grippers_velocities,
                                                       dt ) );
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) -= h;

                    // f(x - h, y)
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) -= h;
                    new_grippers_trajectory = CalculateGrippersTrajectory(
                                grippers_trajectory[0],
                                new_grippers_velocities,
                                dt );
                    objective_value_x_minus_h( ind ) = objective_function(
                                getFinalConfiguration( current_world_configuration,
                                                       new_grippers_trajectory,
                                                       new_grippers_velocities,
                                                       dt ) );
                    new_grippers_velocities[ time_ind ][ vel_ind / 6 ]( vel_ind  % 6 ) += h;

                    // f_x = [ f(x + h, y) - f(x - h, y) ] / ( 2h )
                    derivitives.first( ind ) = (   objective_value_x_plus_h( ind )
                                                 - objective_value_x_minus_h( ind ) )
                            / ( 2*h );
                }

                // This loop fills out the Hessian (second derivitive) of the objective function
                for ( long row_ind = 0; row_ind < derivitives.second.rows(); row_ind++ )
                {
                    const long row_time_ind = row_ind / ( num_grippers * 6 );
                    const long row_vel_ind = row_ind % ( num_grippers * 6 );

                    // f(x + h, y)
                    const double objective_value_row_plus_h = objective_value_x_plus_h( row_ind );
                    // f(x - h, y)
                    const double objective_value_row_minus_h = objective_value_x_minus_h( row_ind );

                    // f_xx = [ f(x + h, y) - 2 f(x,y) + f(x - h, y) ] / h^2
                    derivitives.second( row_ind, row_ind ) = (   objective_value_row_plus_h
                                                               - 2*initial_objective_value
                                                               + objective_value_row_minus_h )
                            / std::pow( h, 2 );

                    #pragma omp parallel for
                    for ( long col_ind = row_ind + 1; col_ind < derivitives.second.cols(); col_ind++ )
                    {
                        const long col_time_ind = col_ind / ( num_grippers * 6 );
                        const long col_vel_ind = col_ind % ( num_grippers * 6 );

                        // f(x, y + h)
                        const double objective_value_col_plus_h = objective_value_x_plus_h( col_ind );
                        // f(x, y - h)
                        const double objective_value_col_minus_h = objective_value_x_minus_h( col_ind );;

                        std::vector< AllGrippersSinglePose > new_grippers_trajectory;

                        // f(x + h, y + h)
                        new_grippers_velocities[ row_time_ind ][ row_vel_ind / 6 ]( row_vel_ind  % 6 ) += h;
                        new_grippers_velocities[ col_time_ind ][ col_vel_ind / 6 ]( col_vel_ind  % 6 ) += h;
                        new_grippers_trajectory = CalculateGrippersTrajectory(
                                    grippers_trajectory[0],
                                    new_grippers_velocities,
                                    dt );
                        const double objective_value_row_col_plus_h = objective_function(
                                    getFinalConfiguration( current_world_configuration,
                                                           new_grippers_trajectory,
                                                           new_grippers_velocities,
                                                           dt ) );
                        new_grippers_velocities[ row_time_ind ][ row_vel_ind / 6 ]( row_vel_ind  % 6 ) -= h;
                        new_grippers_velocities[ col_time_ind ][ col_vel_ind / 6 ]( col_vel_ind  % 6 ) -= h;

                        // f(x - h, y - h)
                        new_grippers_velocities[ row_time_ind ][ row_vel_ind / 6 ]( row_vel_ind  % 6 ) -= h;
                        new_grippers_velocities[ col_time_ind ][ col_vel_ind / 6 ]( col_vel_ind  % 6 ) -= h;
                        new_grippers_trajectory = CalculateGrippersTrajectory(
                                    grippers_trajectory[0],
                                    new_grippers_velocities,
                                    dt );
                        const double objective_value_row_col_minus_h  = objective_function(
                                    getFinalConfiguration( current_world_configuration,
                                                           new_grippers_trajectory,
                                                           new_grippers_velocities,
                                                           dt ) );
                        new_grippers_velocities[ row_time_ind ][ row_vel_ind / 6 ]( row_vel_ind  % 6 ) += h;
                        new_grippers_velocities[ col_time_ind ][ col_vel_ind / 6 ]( col_vel_ind  % 6 ) += h;

                        // f_xy = [ f(x + h, y + h) - f(x + h, y) - f(x, y + h) + 2f(x, y) - f(x - h, y) - f(x, y - h) + f(x - h, y - h) ] / ( 2 h^2 )
                        derivitives.second( row_ind, col_ind ) = (   objective_value_row_col_plus_h
                                                                   - objective_value_row_plus_h
                                                                   - objective_value_col_plus_h
                                                                   + 2*initial_objective_value
                                                                   - objective_value_row_minus_h
                                                                   - objective_value_col_minus_h
                                                                   + objective_value_row_col_minus_h )
                                / ( 2 * std::pow( h, 2 ) );
                        // f_yx = f_xy
                        derivitives.second( col_ind, row_ind ) = derivitives.second( row_ind, col_ind );
                    }
                }

                return derivitives;
            }

            virtual void perturbModel( std::mt19937_64& generator ) = 0;

            static void UpdateGrippersData( const std::vector< GripperData >& grippers_data )
            {
                grippers_data_ = grippers_data;
            }

        protected:

            ////////////////////////////////////////////////////////////////////
            /// Destructor that prevents "delete pointer to base object"
            ////////////////////////////////////////////////////////////////////

            ~DeformableModel() {}

            static std::vector< GripperData > grippers_data_;
    };

}

#endif // DEFORMABLE_MODEL_H
