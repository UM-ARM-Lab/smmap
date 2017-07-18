#ifndef DEFORMABLE_CONTROLLER_HPP
#define DEFORMABLE_CONTROLLER_HPP

#include "smmap/deformable_model.h"

namespace smmap
{
    class DeformableController
    {
        public:
            typedef std::shared_ptr<DeformableController> Ptr;

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity)
            {
                return getGripperMotion_impl(input_data, max_gripper_velocity);
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
    };
}

#endif // DEFORMABLE_CONTROLLER_HPP
