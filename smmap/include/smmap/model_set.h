#ifndef MODEL_SET_H
#define MODEL_SET_H

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

            /**
             * @brief ModelSet
             * @param grippers_data
             * @param object_initial_configuration
             * @param task
             * @param deformability_override If set to -1, do not override the
             * deformablity value. If set >= 0, use this value, otherwise throw
             * an exception
             */
            ModelSet( const std::vector< GripperData >& grippers_data,
                      const ObjectPointSet& object_initial_configuration,
                      const Task& task, double deformability_override = -1 );
            ~ModelSet();

            void updateModels( const std::vector<WorldFeedback>& feedback );

            VectorObjectTrajectory makePredictions(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory ) const;

            std::vector< std::pair< std::vector< AllGrippersSinglePose >, double > >
            getDesiredGrippersTrajectories(
                    const WorldFeedback& world_feedback,
                    const ObjectPointSet& object_desired_configuration,
                    double max_step_size, size_t num_steps );

            const std::vector< double >& getModelConfidence() const;

        private:
            ////////////////////////////////////////////////////////////////////
            // Private helpers
            ////////////////////////////////////////////////////////////////////

            void addModel( DeformableModel::Ptr model );

            void evaluateConfidence( const std::vector<WorldFeedback>& feedback );

            ////////////////////////////////////////////////////////////////////
            // The data for the models themselves
            ////////////////////////////////////////////////////////////////////

            std::vector< DeformableModel::Ptr > model_list_;
            std::vector< double > model_confidence_;

            ////////////////////////////////////////////////////////////////////
            // Utility variables
            ////////////////////////////////////////////////////////////////////

            std::mt19937_64 rnd_generator_;
    };
}

#endif // model_set_h
