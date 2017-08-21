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

        private:
            virtual std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion_impl(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity) override final;

        protected:
            // stretching violation detection helper  --- Added by Mengyao
            bool stretchingViolation(
                    const ObjectPointSet& current_object_configuration) const
            {
                bool over_stretch = false;
                const Eigen::MatrixXd node_squared_distance =
                        CalculateSquaredDistanceMatrix(current_object_configuration);
                ssize_t num_nodes = current_object_configuration.cols();

                for (ssize_t first_node = 0; first_node < num_nodes; ++first_node)
                {
                    for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
                    {
                        const double max_distance = max_stretch_factor_ * object_initial_node_distance_(first_node, second_node);
                        if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
                        {
                            over_stretch = true;
                            return over_stretch;
                        }
                    }
                }

                return over_stretch;
            }


            // stretching violation detection helper  --- Added by Mengyao
            // Only for detection usage, have no effect on controller performance
            const Eigen::MatrixXd object_initial_node_distance_;
            double max_stretch_factor_;

            const DeformableModel::Ptr model_;
            const double obstacle_avoidance_scale_;
            const bool optimize_;
    };
}

#endif // LEAST_SQUARES_CONTROLLER_WITH_OBJECT_AVOIDANCE_H
