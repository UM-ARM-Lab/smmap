#ifndef DEFORMABLE_CONTROLLER_HPP
#define DEFORMABLE_CONTROLLER_HPP

#include <smmap_utilities/visualization_tools.h>
#include "smmap/deformable_model.h"
#include "smmap/robot_interface.hpp"

namespace smmap
{
    class DeformableController
    {
        public:
            typedef std::shared_ptr<DeformableController> Ptr;

            // TODO: Solve problem that constructor can be passed a temporary, which then gets
            // destructed as soon as the ctor exits
            // TODO: Move more data into "world current state" perhaps
            struct InputData
            {
                public:
                    InputData(
                            const WorldState& world_current_state,
                            const DesiredDirection& desired_object_motion,
                            const RobotInterface::Ptr& robot,
                            const Eigen::MatrixXd& robot_jacobian,
                            const bool robot_jacobian_valid,
                            const std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>>& poi_collision_data,
                            const double max_step_size,
                            const double max_robot_dof_step_size)
                        : world_current_state_(world_current_state)
                        , desired_object_motion_(desired_object_motion)
                        , robot_(robot)
                        , robot_jacobian_(robot_jacobian)
                        , robot_jacobian_valid_(robot_jacobian_valid)
                        , poi_collision_data_(poi_collision_data)
                        , max_grippers_step_size_(max_step_size)
                        , max_robot_dof_step_size_(max_robot_dof_step_size)
                    {}

                    const WorldState world_current_state_;
                    const DesiredDirection desired_object_motion_;
                    const RobotInterface::Ptr robot_;
                    const Eigen::MatrixXd robot_jacobian_;
                    const bool robot_jacobian_valid_;
                    const std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> poi_collision_data_;
                    const double max_grippers_step_size_;
                    const double max_robot_dof_step_size_;
            };

            struct OutputData
            {
                public:
                    OutputData()
                    {}

                    OutputData(
                            const size_t num_grippers,
                            const ssize_t num_nodes_on_object,
                            const ssize_t num_robot_dof)
                        : grippers_motion_(num_grippers, kinematics::Vector6d::Zero())
                        , object_motion_(ObjectPointSet::Zero(3, num_nodes_on_object))
                        , robot_dof_motion_(Eigen::VectorXd::Zero(num_robot_dof))
                    {}

                    OutputData(
                            const AllGrippersSinglePoseDelta grippers_motion,
                            const ObjectPointSet object_motion,
                            const Eigen::VectorXd robot_dof_motion)
                        : grippers_motion_(grippers_motion)
                        , object_motion_(object_motion)
                        , robot_dof_motion_(robot_dof_motion)
                    {}

                    AllGrippersSinglePoseDelta grippers_motion_;
                    ObjectPointSet object_motion_;
                    Eigen::VectorXd robot_dof_motion_;
            };

            DeformableController(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    const RobotInterface::Ptr& robot,
                    const smmap_utilities::Visualizer::Ptr& vis)
                : nh_(nh)
                , ph_(ph)
                , robot_(robot)
                , vis_(vis)
            {}

            OutputData getGripperMotion(
                    const InputData& input_data)
            {
                return getGripperMotion_impl(input_data);
            }

        protected:

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            const RobotInterface::Ptr robot_;
            smmap_utilities::Visualizer::Ptr vis_;

            ////////////////////////////////////////////////////////////////////
            // Destructor that prevents "delete pointer to base object"
            ////////////////////////////////////////////////////////////////////

            ~DeformableController() {}

        private:

            virtual OutputData getGripperMotion_impl(
                    const InputData& input_data) = 0;
    };
}

#endif // DEFORMABLE_CONTROLLER_HPP
