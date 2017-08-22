#ifndef DEFORMABLE_CONTROLLER_HPP
#define DEFORMABLE_CONTROLLER_HPP

#include "smmap/deformable_model.h"

namespace smmap
{
    class DeformableController
    {
        public:
            DeformableController()
                : stretching_violation_count_(0)
                , current_stretching_factor_(0.0)
            {}

            typedef std::shared_ptr<DeformableController> Ptr;

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity)
            {
                return getGripperMotion_impl(input_data, max_gripper_velocity);
            }

            long getStretchingViolationCount()
            {
                return stretching_violation_count_;
            }

            double getCurrentStretchingFactor()
            {
                return current_stretching_factor_;
            }

        protected:

            ////////////////////////////////////////////////////////////////////
            // Destructor that prevents "delete pointer to base object"
            ////////////////////////////////////////////////////////////////////

            ~DeformableController() {}

        private:

            virtual std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion_impl(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity) = 0;

        protected:

            long stretching_violation_count_;
            double current_stretching_factor_;
    };
}

#endif // DEFORMABLE_CONTROLLER_HPP
