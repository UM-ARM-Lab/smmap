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
                    double max_step_size,
                    size_t num_steps ) const = 0;

            virtual Eigen::VectorXd getObjectiveFunction1stDerivitive(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    const double dt,
                    const std::function< double( const ObjectPointSet& ) > objective_function ) const;

            virtual std::pair< Eigen::VectorXd, Eigen::MatrixXd > getObjectiveFunction2ndDerivitive(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    const double dt,
                    const std::function< double( const ObjectPointSet& ) > objective_function ) const;

            virtual void perturbModel( std::mt19937_64& generator ) = 0;

            ////////////////////////////////////////////////////////////////////
            /// Update function for static member
            ////////////////////////////////////////////////////////////////////

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
