#ifndef CONSTRAINT_JACOBIAN_MODEL_H
#define CONSTRAINT_JACOBIAN_MODEL_H

#include <sdf_tools/sdf.hpp>

#include "smmap/deformable_model.h"

namespace smmap
{
    class ConstraintJacobianModel final : public DeformableModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            ConstraintJacobianModel(
                    const double translation_dir_deformability,
                    const double translation_dis_deformability,
                    const double rotation_deformability,
//                    const double translational_deformablity,
                    const sdf_tools::SignedDistanceField& environment_sdf);

            ////////////////////////////////////////////////////////////////////
            // Static functions to set data for all models
            ////////////////////////////////////////////////////////////////////

            static void SetInitialObjectConfiguration(
                    const ObjectPointSet& object_initial_configuration);

        private:

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel_impl(
                    const WorldState& previous,
                    const WorldState& next) final override;

            virtual ObjectPointSet getObjectDelta_impl(
                    const WorldState& world_state,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta) const final override;

            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            static std::atomic<bool> static_data_initialized_;
            static Eigen::MatrixXd object_initial_node_distance_;

            /// Indexed first by gripper, second by node i.e. (gripper_ind, node_ind)
            static Eigen::MatrixXd gripper_influence_per_node_;
//            static Eigen::VectorXd sum_of_object_node_to_grippers_distances_;
            static ssize_t num_nodes_;

            ////////////////////////////////////////////////////////////////////
            // Model function parameters
            ////////////////////////////////////////////////////////////////////

            const double translation_dir_deformability_;
            const double translation_dis_deformability_;

            // Analogs of the original version
            const double rotation_deformability_;
//            const double translational_deformability_;

            ////////////////////////////////////////////////////////////////////
            // Obstacle information from sdf tool
            ////////////////////////////////////////////////////////////////////
            const sdf_tools::SignedDistanceField environment_sdf_;
            const double obstacle_threshold_;

            ////////////////////////////////////////////////////////////////////
            // Function to adjust rigidity actually
            ////////////////////////////////////////////////////////////////////

            double dirPropotionalModel(
                    const Eigen::Vector3d node_to_gripper,
                    const Eigen::Vector3d node_v) const;

            double disLinearModel(
                    const double dist_to_gripper,
                    const double dist_rest) const;

            ////////////////////////////////////////////////////////////////////
            // Jacobian and Mask matrix computation
            ////////////////////////////////////////////////////////////////////

            Eigen::MatrixXd computeGrippersToDeformableObjectJacobian(
                    const WorldState& world_state,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta) const;

            // Null Projection
            Eigen::MatrixXd computeObjectVelocityMask(
                    const ObjectPointSet& current_configuration,
                    const Eigen::MatrixXd& object_p_dot) const;
    };
}

#endif // CONSTRAINT_JACOBIAN_MODEL_H
