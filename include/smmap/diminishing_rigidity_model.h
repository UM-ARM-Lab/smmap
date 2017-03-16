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
            // Virtual function overrides
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel(const WorldState& previous, const WorldState& next) final override;

            ////////////////////////////////////////////////////////////////////
            // Helper used only by AdaptiveJacobian (at the moment)
            // Find a better way to do this
            ////////////////////////////////////////////////////////////////////

            Eigen::MatrixXd getGrippersToObjectJacobian(const JacobianInputData &input_data) const;

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
            // Computation helpers
            ////////////////////////////////////////////////////////////////////

            virtual Eigen::MatrixXd computeGrippersToObjectJacobian(const JacobianInputData &input_data) const override final;

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
