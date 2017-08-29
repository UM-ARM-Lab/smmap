#ifndef LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
#define LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H

#include "smmap/deformable_controller.hpp"

namespace smmap
{
    class LeastSquaresControllerWithObjectAvoidance : public DeformableController
    {
        public:
            LeastSquaresControllerWithObjectAvoidance(
                    ros::NodeHandle &nh,
                    ros::NodeHandle &ph,
                    const DeformableModel::Ptr& model,
                    const double obstacle_avoidance_scale_,
                    const bool optimize);

        public:
            // stretching violation detection helper  --- Added by Mengyao
            // Only for detection usage, have no effect on controller performance
            const Eigen::MatrixXd object_initial_node_distance_;
            double max_grippers_distance_;

        private:
            virtual std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion_impl(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity) override final;

        protected:
            // stretching violation detection helper  --- Added by Mengyao
            bool stretchingViolation(const WorldState &current_world_state);


            // stretching violation detection helper  --- Added by Mengyao
            // Only for detection usage, have no effect on controller performance
            double max_stretch_factor_;
            const int num_grippers_;

            const DeformableModel::Ptr model_;
            const double obstacle_avoidance_scale_;
            const bool optimize_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
