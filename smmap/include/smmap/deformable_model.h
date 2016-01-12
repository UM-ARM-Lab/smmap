#ifndef DEFORMABLE_MODEL_H
#define DEFORMABLE_MODEL_H

#include <memory>
#include <random>

#include <kinematics_toolbox/kinematics.h>

#include "smmap/trajectory.hpp"
#include "smmap/gripper_helpers.hpp"

namespace smmap
{
    class DeformableModel
    {
        public:
            typedef std::shared_ptr< DeformableModel > Ptr;

            ////////////////////////////////////////////////////////////////////
            /// Wrappers for virtual functions
            ////////////////////////////////////////////////////////////////////

            void updateModel(
                    const VectorGrippersData& grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const std::vector< kinematics::VectorVector6d >& grippers_velocities,
                    const ObjectTrajectory& object_trajectory,
                    const kinematics::VectorMatrix3Xd& object_velocities )
            {
                doUpdateModel( grippers_data, grippers_trajectory,
                        grippers_velocities, object_trajectory, object_velocities );
            }

            ObjectTrajectory getPrediction(
                    const ObjectPointSet& object_configuration,
                    VectorGrippersData grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const std::vector< kinematics::VectorVector6d >& grippers_velocities ) const
            {
                return doGetPrediction( object_configuration, grippers_data, grippers_trajectory, grippers_velocities );
            }

            AllGrippersTrajectory getDesiredGrippersTrajectory(
                    const ObjectPointSet& object_current_configuration,
                    const ObjectPointSet& object_desired_configuration,
                    VectorGrippersData grippers_data,
                    double max_step_size, size_t num_steps ) const
            {
                return doGetDesiredGrippersTrajectory( object_current_configuration,
                        object_desired_configuration, grippers_data, max_step_size, num_steps );
            }

            void perturbModel( std::mt19937_64& generator )
            {
                doPerturbModel( generator );
            }

        protected:

            ////////////////////////////////////////////////////////////////////
            /// Destructor that prevents "delete pointer to base object"
            ////////////////////////////////////////////////////////////////////

            ~DeformableModel() {}

        private:

            ////////////////////////////////////////////////////////////////////
            /// Virtual functions that need to be overridden by derived classes
            ////////////////////////////////////////////////////////////////////

            virtual void doUpdateModel(
                    const VectorGrippersData& grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const std::vector< kinematics::VectorVector6d >& grippers_velocities,
                    const ObjectTrajectory& object_trajectory,
                    const kinematics::VectorMatrix3Xd& object_velocities ) = 0;

            virtual ObjectTrajectory doGetPrediction(
                    const ObjectPointSet& object_configuration,
                    VectorGrippersData grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const std::vector< kinematics::VectorVector6d >& grippers_velocities ) const = 0;

            virtual AllGrippersTrajectory doGetDesiredGrippersTrajectory(
                    const ObjectPointSet& object_current_configuration,
                    const ObjectPointSet& object_desired_configuration,
                    VectorGrippersData gripper_data,
                    double max_step_size, size_t num_steps ) const = 0;

            virtual void doPerturbModel( std::mt19937_64& generator ) = 0;
    };

}

#endif // DEFORMABLE_MODEL_H
