#ifndef DIMINISHING_RIGIDITY_MODEL_H
#define DIMINISHING_RIGIDITY_MODEL_H

#include "smmap/deformable_model.h"

namespace smmap
{
    // TODO: find a way to accept dynamic/online gripper re-grasping
    class DiminishingRigidityModel final : public DeformableModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Static functions to set data for all models
            ////////////////////////////////////////////////////////////////////

            static void SetInitialObjectConfiguration(
                    const ObjectPointSet& object_initial_configuration );

            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            DiminishingRigidityModel( double deformability );

            DiminishingRigidityModel(
                    double translation_deformability,
                    double rotation_deformability );

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel( const std::vector< WorldState >& feedback );

            virtual ObjectTrajectory getPrediction(
                    const WorldState& world_initial_state,
                    const AllGrippersPoseTrajectory& grippers_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& grippers_pose_delta_trajectory,
                    double dt ) const;

            virtual ObjectPointSet getFinalConfiguration(
                    const WorldState& world_initial_state,
                    const AllGrippersPoseTrajectory& gripper_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& gripper_pose_delta_trajectory,
                    double dt ) const ;

            virtual std::pair< AllGrippersPoseTrajectory, ObjectTrajectory > getSuggestedGrippersTrajectory(
                    const WorldState& world_initial_state,
                    const int planning_horizion,
                    const double dt,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale ) const;

            void perturbModel( std::mt19937_64& generator );

        private:

            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            ObjectPointSet getObjectDelta(
                    const ObjectPointSet& object_initial_configuration,
                    const AllGrippersSinglePose & grippers_pose,
                    const AllGrippersSingleVelocity& grippers_velocity,
                    double dt ) const;

            Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const AllGrippersSinglePose& grippers_pose,
                    const ObjectPointSet& current_configuration ) const;

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            static std::atomic_bool static_data_initialized_;
            static std::normal_distribution< double > perturbation_distribution_;
            static Eigen::MatrixXd object_initial_node_distance_;
            static long num_nodes_;

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            double translation_deformability_;
            double rotation_deformability_;
    };
}

#endif // DIMINISHING_RIGIDITY_MODEL_H
