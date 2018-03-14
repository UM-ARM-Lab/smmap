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
            std::vector<std::pair<std::vector<double>, std::vector<kinematics::Vector6d>>> static_grippers_motions_;
            std::vector<size_t> current_motion_idx_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
