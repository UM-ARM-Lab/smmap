#ifndef diminishing_rigidity_model_h
#define diminishing_rigidity_model_h

#include "smmap/deformable_model.h"

namespace smmap
{
    class DiminishingRigidityModel : public DeformableModel
    {
//        typedef std::shared_ptr< DiminishingRigidityModel > Ptr;

        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            DiminishingRigidityModel(
                    const VectorGrippersData& grippers_data,
                    const ObjectPointSet& object_initial_configuration,
                    double rigidity, bool use_rotation,
                    double obstacle_avoidance_scale, double strechting_correction_threshold );

            DiminishingRigidityModel(
                    const VectorGrippersData& grippers_data,
                    const ObjectPointSet& object_initial_configuration,
                    double rigidity_translation, double rotation_rigidity, bool use_rotation,
                    double obstacle_avoidance_scale, double strechting_correction_threshold );

        private:

            ////////////////////////////////////////////////////////////////////
            // Constructor helpers
            ////////////////////////////////////////////////////////////////////

            void computeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            void doUpdateModel(
                    const VectorGrippersData& grippers_data,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const std::vector< kinematics::VectorVector6d >& grippers_velocities,
                    const ObjectTrajectory& object_trajectory,
                    const kinematics::VectorMatrix3Xd& object_velocities );

            ObjectTrajectory doGetPrediction(
                    const ObjectPointSet& object_configuration,
                    const AllGrippersTrajectory& grippers_trajectory,
                    const std::vector< kinematics::VectorVector6d >& grippers_velocities ) const;

            AllGrippersTrajectory doGetDesiredGrippersTrajectory(
                    const ObjectPointSet& object_current_configuration,
                    const ObjectPointSet& object_desired_configuration,
                    VectorGrippersData grippers_data,
                    double max_step_size, size_t num_steps ) const;

            void doPerturbModel( std::mt19937_64& generator );

            ////////////////////////////////////////////////////////////////////
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            Eigen::MatrixXd computeGrippersToObjectJacobian(
                    const VectorGrippersData& grippers_data ) const;

            std::vector< CollisionAvoidanceResult > computeGrippersObjectAvoidance(
                    const VectorGrippersData& grippers_data,
                    double max_step_size) const;

            Eigen::MatrixXd computeCollisionToGripperJacobian(
                    const GripperData& gripper_data ) const;

            Eigen::VectorXd computeStretchingCorrection(
                    const ObjectPointSet& object_current_configuration ) const ;

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            static std::normal_distribution< double > perturbation_distribution;
            static Eigen::MatrixXd object_initial_node_distance_;

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            const ObjectPointSet object_initial_configuration_;

            double translation_rigidity_;
            double rotation_rigidity_;
            bool use_rotation_;
            const long cols_per_gripper_;

            const double obstacle_avoidance_scale_;
            const double stretching_correction_threshold_;
    };
}

#endif // diminishing_rigidity_model_h
