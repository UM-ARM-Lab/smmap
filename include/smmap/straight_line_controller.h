#ifndef STRAIGHT_LINE_CONTROLLER_H
#define STRAIGHT_LINE_CONTROLLER_H

#include "smmap/deformable_controller.h"

namespace smmap
{
    class StraightLineController : public DeformableController
    {
        public:
            StraightLineController(
                    ros::NodeHandle& ph,
                    const DeformableModel::Ptr& model,
                    const RobotInterface::Ptr& robot);

        private:
            virtual OutputData getGripperMotion_impl(const InputData& input_data) override final;

            const DeformableModel::Ptr model_;
            AllGrippersSinglePoseDelta grippers_motion_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
