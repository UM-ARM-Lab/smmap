#ifndef LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
#define LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H

#include "smmap/deformable_controller.hpp"

namespace smmap
{
    class LeastSquaresControllerWithObjectAvoidance : public DeformableController
    {
        public:
            LeastSquaresControllerWithObjectAvoidance(
                    const DeformableModel::Ptr& model,
                    const double obstacle_avoidance_scale_,
                    const bool optimize);

        private:
            virtual std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion_impl(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity) override final;

            const DeformableModel::Ptr model_;
            const double obstacle_avoidance_scale_;
            const bool optimize_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
