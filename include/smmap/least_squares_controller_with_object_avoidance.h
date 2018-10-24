#ifndef LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
#define LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H

#include "smmap/deformable_controller.h"

namespace smmap
{
    class LeastSquaresControllerWithObjectAvoidance : public DeformableController
    {
        public:
            LeastSquaresControllerWithObjectAvoidance(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const RobotInterface::Ptr& robot,
                    const smmap_utilities::Visualizer::Ptr& vis,
                    const DeformableModel::Ptr& model,
                    const double obstacle_avoidance_scale,
                    const bool optimize);

        private:
            virtual OutputData getGripperMotion_impl(const InputData& input_data) override final;

            const DeformableModel::Ptr model_;
            const double obstacle_avoidance_scale_;
            const bool optimize_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
