#ifndef MODEL_SET_H
#define MODEL_SET_H

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

            ModelSet( const std::vector< GripperData >& grippers_data,
                      const ObjectPointSet& object_initial_configuration,
                      const Task& task );

            ModelSet( const std::vector< GripperData >& grippers_data,
                      const ObjectPointSet& object_initial_configuration,
                      const Task& task,
                      double translational_deformability,
                      double rotational_deformability );
            ~ModelSet();

            void updateModels( const std::vector<WorldFeedback>& feedback );

            VectorObjectTrajectory makePredictions(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector<kinematics::VectorVector6d>& grippers_velocities,
                    double dt ) const;

            std::vector< std::pair< std::vector< AllGrippersSinglePose >, double > >
            getDesiredGrippersTrajectories(
                    const WorldFeedback& world_feedback,
                    const ObjectPointSet& object_desired_configuration,
                    double max_step_size, size_t num_steps );

            std::vector< Eigen::VectorXd >
            getObjectiveFunction1stDerivitive(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt,
                    std::function< double( const ObjectPointSet& ) > objective_function ) const;

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
