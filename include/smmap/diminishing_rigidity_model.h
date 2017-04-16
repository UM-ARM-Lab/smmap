#ifndef DIMINISHING_RIGIDITY_MODEL_H
#define DIMINISHING_RIGIDITY_MODEL_H

#include "smmap/jacobian_model.h"

namespace smmap
{
    // TODO: find a way to accept dynamic/online gripper re-grasping
    class DiminishingRigidityModel final : public JacobianModel
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Constructors and Destructor
            ////////////////////////////////////////////////////////////////////

            DiminishingRigidityModel(
                    const double deformability,
                    const bool optimize);

            DiminishingRigidityModel(
                    const double translation_deformability,
                    const double rotation_deformability,
                    const bool optimize);

            ////////////////////////////////////////////////////////////////////
            // Static functions to set data for all models
            ////////////////////////////////////////////////////////////////////

            static void SetInitialObjectConfiguration(
                    const ObjectPointSet& object_initial_configuration);

        private:

            ////////////////////////////////////////////////////////////////////
            // Static helpers
            ////////////////////////////////////////////////////////////////////

            static void ComputeObjectNodeDistanceMatrix();

            ////////////////////////////////////////////////////////////////////
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel_impl(
                    const WorldState& previous,
                    const WorldState& next) final override;

            virtual Eigen::MatrixXd computeGrippersToDeformableObjectJacobian_impl(
                    const DeformableModelInputData &input_data) const override final;

            // Null Projection --- Edit By Mengyao
            virtual Eigen::MatrixXd computeObjectVelocityMask_impl(
                    const ObjectPointSet& current_configuration,
                    const Eigen::MatrixXd &object_p_dot) const override final;

            ////////////////////////////////////////////////////////////////////
            // Static members
            ////////////////////////////////////////////////////////////////////

            static std::atomic_bool static_data_initialized_;
            static Eigen::MatrixXd object_initial_node_distance_;
            static long num_nodes_;

            ////////////////////////////////////////////////////////////////////
            // Private members
            ////////////////////////////////////////////////////////////////////

            const double translation_deformability_;
            const double rotation_deformability_;
    };
}

#endif // DIMINISHING_RIGIDITY_MODEL_H
