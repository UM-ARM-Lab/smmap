#ifndef CONSTRAINT_JACOBIAN_MODEL_H
#define CONSTRAINT_JACOBIAN_MODEL_H

#include <sdf_tools/sdf.hpp>

#include "smmap/jacobian_model.h"

namespace smmap
{
    enum RigidityFnType {simpleFn, midleverFn, complicateFn};

    typedef std::function<Eigen::Matrix3d(
            Eigen::Vector3d node_to_gripper, Eigen::Vector3d node_v
            )>
    translationDirectionFnPtr;
    typedef std::function<double(
            const double dist_to_gripper, const double dist_rest
            )>
    translationDistanceFnPtr;

    // TODO: find a way to accept dynamic/online gripper re-grasping
    class ConstraintJacobianModel final: public JacobianModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////
/*
            ConstraintJacobianModel(
                    const double translation_dir_deformability,
                    const double translation_dis_deformability,
                    const double rotation_deformability,
                    const bool optimize);
*/
            ConstraintJacobianModel(
                    const double translation_dir_deformability,
                    const double translation_dis_deformability,
                    const double rotation_deformability,
                    const sdf_tools::SignedDistanceField environment_sdf,
                    const bool optimize);
/*
            ConstraintJacobianModel(
                    const double translation_dir_deformability,
                    const double translation_dis_deformability,
                    const double rotation_deformability,
                    RigidityFnType trans_dir_fn,
                    RigidityFnType trans_dis_fn,
                    const bool optimize);
*/
            ConstraintJacobianModel(
                    const double translation_dir_deformability,
                    const double translation_dis_deformability,
                    const double rotation_deformability,
                    const sdf_tools::SignedDistanceField environment_sdf,
                    RigidityFnType trans_dir_fn,
                    RigidityFnType trans_dis_fn,
                    const bool optimize);

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

            virtual Eigen::MatrixXd computeGrippersToDeformableObjectJacobian_impl(
                    const DeformableModelInputData &input_data) const final override;

            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            static std::atomic_bool static_data_initialized_;
            static Eigen::MatrixXd object_initial_node_distance_;
            static long num_nodes_;

            ////////////////////////////////////////////////////////////////////
            // Model function parameters
            ////////////////////////////////////////////////////////////////////

            const double translation_dir_deformability_;
            const double translation_dis_deformability_;
            const double rotation_deformability_;

            ////////////////////////////////////////////////////////////////////
            // Function mode for gamma and beta function, more to be implemented later
            ////////////////////////////////////////////////////////////////////

            const RigidityFnType trans_dir_type_;
            const RigidityFnType trans_dis_type_;

            ////////////////////////////////////////////////////////////////////
            // Obstacle information from sdf tool
            ////////////////////////////////////////////////////////////////////
            const sdf_tools::SignedDistanceField environment_sdf_;
            const double obstacle_threshold_;

            ////////////////////////////////////////////////////////////////////
            // Function to adjust rigidity actually
            ////////////////////////////////////////////////////////////////////

            Eigen::Matrix3d dirPropotionalModel(
                    const Eigen::Vector3d node_to_gripper,
                    const Eigen::Vector3d node_v) const;
            double disLinearModel(
                    const double dist_to_gripper,
                    const double dist_rest) const;

            ////////////////////////////////////////////////////////////////////
            // Computation of Mask Matrix M, q_dot = pinv(J)*M*P_dot
            ////////////////////////////////////////////////////////////////////
//            Eigen::MatrixXd computeObjectVelocityMask(const ObjectPointSet& current_configuration,
//                    const Eigen::VectorXd &object_p_dot) const;

            Eigen::MatrixXd computeObjectVelocityMask(const ObjectPointSet& current_configuration,
                    const AllGrippersSinglePose& grippers_pose) const;


    };
}

#endif // CONSTRAINT_JACOBIAN_MODEL_H
