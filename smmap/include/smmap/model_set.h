#ifndef model_set_h
#define model_set_h

#include <list>
#include <memory>
#include <random>

#include "smmap/deformable_model.h"
#include "smmap/task.hpp"

namespace smmap
{
    class ModelSet
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructor and destructor
            ////////////////////////////////////////////////////////////////////

            ModelSet( const VectorGrippersData& grippers_data,
                      const ObjectPointSet& object_initial_configuration,
                      const Task& task );
            ~ModelSet();

            void updateModels(
                    const VectorGrippersData& grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const ObjectTrajectory& object_trajectory );

            VectorObjectTrajectory makePredictions(
                    VectorGrippersData grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const ObjectPointSet& object_configuration ) const;

            std::vector< std::pair< AllGrippersTrajectory, double > > getDesiredGrippersTrajectories(
                    const ObjectPointSet& object_current_configuration,
                    const ObjectPointSet& object_desired_configuration,
                    const VectorGrippersData& grippers_data,
                    double max_step_size, size_t num_steps );

            const std::vector< double >& getModelConfidence() const;

        private:
            ////////////////////////////////////////////////////////////////////
            // Private helpers
            ////////////////////////////////////////////////////////////////////

            void addModel( DeformableModel::Ptr model );

            std::vector< kinematics::VectorVector6d > calculateGrippersVelocities(
                    const AllGrippersTrajectory& grippers_trajectory ) const;

            kinematics::VectorMatrix3Xd calculateObjectVelocities(
                    const ObjectTrajectory& object_trajectory ) const;

            void evaluateConfidence(
                    VectorGrippersData grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const std::vector< kinematics::VectorVector6d >& grippers_velocities,
                    const ObjectTrajectory& object_trajectory );

            // TODO: move this to *somewhere* else
            const Task& task_;
            VectorGrippersData grippers_data_;
            const ObjectPointSet object_initial_configuration_;
            std::vector< DeformableModel::Ptr > model_list_;
            std::vector< double > model_confidence_;

            std::mt19937_64 rnd_generator_;
    };
}

#endif // model_set_h
