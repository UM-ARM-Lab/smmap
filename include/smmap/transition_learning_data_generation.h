#pragma once

#include <random>
#include <memory>

#include <smmap_utilities/visualization_tools.h>
#include "smmap/learned_transitions.h"
#include "smmap/quinlan_rubber_band.h"
#include "smmap/robot_interface.hpp"

namespace smmap
{
    class DataGeneration
    {
    private:
        const std::shared_ptr<ros::NodeHandle> nh_;
        const std::shared_ptr<ros::NodeHandle> ph_;
        const RobotInterface::Ptr robot_;
        const smmap_utilities::Visualizer::ConstPtr vis_;
        const bool visualize_gripper_motion_;

        const unsigned long seed_;
        const std::shared_ptr<std::mt19937_64> generator_;

        // Note that work_space_grid_ and the environment_sdf_ are using different
        // resolutions due to the way the SDF is created in CustomScene
        const sdf_tools::SignedDistanceField::ConstPtr sdf_;
        const XYZGrid work_space_grid_;

        const DeformableType deformable_type_;
        const TaskType task_type_;

        const std::vector<GripperData> grippers_data_;
        const Eigen::MatrixXd object_initial_node_distance_;
        const ssize_t num_nodes_;
        const double max_stretch_factor_;
        const double max_band_length_;

        RubberBand::Ptr band_;
        std::vector<ssize_t> path_between_grippers_through_object_;
        TransitionEstimation::Ptr transition_estimator_;

    public:
        DataGeneration(
                std::shared_ptr<ros::NodeHandle> nh,
                std::shared_ptr<ros::NodeHandle> ph,
                RobotInterface::Ptr robot,
                const smmap_utilities::Visualizer::ConstPtr& vis);

        void initialize(const WorldState& world_state);
        void initializeBand(const WorldState& world_state);

        void visualizeDeformableObject(
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std_msgs::ColorRGBA& color,
                const int32_t id = 1) const;

        void visualizeDeformableObject(
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std::vector<std_msgs::ColorRGBA>& colors,
                const int32_t id = 1) const;
    };
}
