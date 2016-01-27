#ifndef DIMINISHING_RIGIDITY_MODEL_H
#define DIMINISHING_RIGIDITY_MODEL_H

#include "smmap/deformable_model.h"
#include <atomic>

namespace smmap
{
    class DiminishingRigidityModel : public DeformableModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            DiminishingRigidityModel( double deformability,
                                      bool use_rotation,
                                      double obstacle_avoidance_scale,
                                      double strechting_correction_threshold );

            DiminishingRigidityModel( double translation_deformability,
                                      double rotation_deformability,
                                      bool use_rotation,
                                      double obstacle_avoidance_scale,
                                      double strechting_correction_threshold );

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            void updateModel( const std::vector< WorldFeedback >& feedback );

            ObjectTrajectory getPrediction(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt ) const;

            ObjectPointSet getFinalConfiguration(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt ) const;

            ObjectPointSet getObjectDelta(
                    const ObjectPointSet& object_current_configuration,
                    const AllGrippersSinglePose & grippers_pose,
                    const AllGrippersSingleVelocity& grippers_velocity,
                    double dt ) const;

            std::vector< AllGrippersSinglePose > getDesiredGrippersTrajectory(
                    const WorldFeedback& world_feedback,
                    const ObjectPointSet& object_desired_configuration,
                    double max_step_size, size_t num_steps ) const;

            std::pair< Eigen::VectorXd, Eigen::MatrixXd > getObjectiveFunctionDerivitives(
                    const WorldFeedback& current_world_configuration,
                    const std::vector< AllGrippersSinglePose >& grippers_trajectory,
                    const std::vector< AllGrippersSingleVelocity >& grippers_velocities,
                    double dt,
                    std::function< double( const ObjectPointSet& ) > objective_function ) const;

            void perturbModel( std::mt19937_64& generator );

            static void SetInitialObjectConfiguration( const ObjectPointSet& object_initial_configuration );

        private:

            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const AllGrippersSinglePose& grippers_pose,
                    const ObjectPointSet& current_configuration ) const;

            std::vector< CollisionAvoidanceResult > computeGrippersObjectAvoidance(
                    const WorldFeedback& world_feedback,
                    const AllGrippersSinglePose& grippers_pose,
                    double max_step_size ) const;

            Eigen::MatrixXd computeCollisionToGripperJacobian(
                    const Eigen::Vector3d& point_on_gripper,
                    const Eigen::Affine3d& gripper_pose ) const;

            Eigen::VectorXd computeStretchingCorrection(
                    const ObjectPointSet& object_current_configuration ) const ;

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            static std::atomic< bool > initialized_;
            static std::normal_distribution< double > perturbation_distribution;
            static Eigen::MatrixXd object_initial_node_distance_;

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            static long num_nodes_;

            double translation_deformability_;
            double rotation_deformability_;
            bool use_rotation_;
            const long cols_per_gripper_;

            const double obstacle_avoidance_scale_;
            const double stretching_correction_threshold_;
    };
}

#endif // DIMINISHING_RIGIDITY_MODEL_H
