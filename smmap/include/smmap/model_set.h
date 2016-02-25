#ifndef MODEL_SET_H
#define MODEL_SET_H

#include "smmap/deformable_model.h"

namespace smmap
{
    class ModelSet
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructor
            ////////////////////////////////////////////////////////////////////

            ModelSet( const UpdateModelUtilityFunctionType& update_model_utility_fn );

            void addModel( DeformableModel::Ptr model );
            void updateModels(
                    const std::vector< WorldState >& world_feedback,
                    const Eigen::VectorXd& weights );

            VectorObjectTrajectory getPredictions(
                    const WorldState& starting_world_state,
                    const AllGrippersPoseTrajectory& grippers_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory,
                    const double dt ) const;

            std::vector< std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > >
            getSuggestedGrippersTrajectories(
                    const WorldState& world_current_state,
                    const int planning_horizion,
                    const double dt,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale,
                    const double stretching_correction_threshold ) const;
/*
            std::vector< Eigen::VectorXd >
            getObjectiveFunction1stDerivitive(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt,
                    std::function< double( const ObjectPointSet& ) > objective_function ) const;

            std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >
            getObjectiveFunction2ndDerivitive(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt,
                    std::function< double( const ObjectPointSet& ) > objective_function ) const;
*/
            const std::vector< double >& getModelUtility() const;

        private:
            ////////////////////////////////////////////////////////////////////
            // The data for the models themselves
            ////////////////////////////////////////////////////////////////////

            std::vector< DeformableModel::Ptr > model_list_;
            std::vector< double > model_utility_;
            const UpdateModelUtilityFunctionType update_model_utility_fn_;

            ////////////////////////////////////////////////////////////////////
            // Utility variables
            ////////////////////////////////////////////////////////////////////

//            std::mt19937_64 rnd_generator_;
    };
}

#endif // model_set_h
