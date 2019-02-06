#pragma once

#include <random>
#include <memory>

#include <smmap_utilities/visualization_tools.h>
#include "smmap/learned_transitions.h"
#include "smmap/quinlan_rubber_band.h"
#include "smmap/robot_interface.h"

namespace smmap
{
    struct TransitionSimulationRecord
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        TransitionEstimation::StateTransition template_;
        smmap_utilities::ObjectPointSet template_band_surface_;

        Eigen::Isometry3d center_of_rotation_;
        Eigen::Isometry3d transform_applied_;

        TransitionEstimation::StateTransition tested_;
        smmap_utilities::ObjectPointSet tested_band_surface_;

        uint64_t serializeSelf(std::vector<uint8_t>& buffer) const;

        static uint64_t Serialize(
                const TransitionSimulationRecord& test_results,
                std::vector<uint8_t>& buffer);

        static std::pair<TransitionSimulationRecord, uint64_t> Deserialize(
                const std::vector<uint8_t>& buffer,
                const uint64_t current,
                const RubberBand& template_band);

        bool operator==(const TransitionSimulationRecord& other) const;

        void visualize(const smmap_utilities::Visualizer::ConstPtr& vis) const;
    };

    class TransitionTesting
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
        const double gripper_min_distance_to_obstacles_;
        const Eigen::Isometry3d experiment_center_of_rotation_;

        const DeformableType deformable_type_;
        const TaskType task_type_;

        RubberBand::Ptr band_;
        TransitionEstimation::Ptr transition_estimator_;

        const std::string data_folder_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        TransitionTesting(
                std::shared_ptr<ros::NodeHandle> nh,
                std::shared_ptr<ros::NodeHandle> ph,
                RobotInterface::Ptr robot,
                const smmap_utilities::Visualizer::ConstPtr& vis);

    private:
        Eigen::Isometry3d calculateExperimentCenterOfRotation();
        void initialize(const WorldState& world_state);
        void initializeBand(const WorldState& world_state);

    public:
        void runTests(const bool generate_new_test_data);

        void visualizeDeformableObject(
                const std::string& marker_name,
                const smmap_utilities::ObjectPointSet& object_configuration,
                const std_msgs::ColorRGBA& color,
                const int32_t id = 1) const;

        void visualizeDeformableObject(
                const std::string& marker_name,
                const smmap_utilities::ObjectPointSet& object_configuration,
                const std::vector<std_msgs::ColorRGBA>& colors,
                const int32_t id = 1) const;

    private:
        class DataGeneration
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            DataGeneration(const TransitionTesting& framework);
            void generateTestData(
                    std::mt19937_64& generator,
                    const std::string& data_folder);

        private:
            const TransitionTesting& framework_;
            arc_helpers::RandomRotationGenerator random_rotation_distribution_;

            // Stored here because Eigen + tuple = bad
            EigenHelpers::VectorIsometry3d random_test_rope_nodes_start_;
            smmap_utilities::AllGrippersSinglePose random_test_starting_gripper_poses_;
            smmap_utilities::AllGrippersSinglePose random_test_ending_gripper_poses_;
            Eigen::Isometry3d random_test_transform_applied_;

            void generateRandomTest(
                    std::mt19937_64& generator,
                    const TransitionEstimation::StateTransition& trans);
        };
        friend class DataGeneration;
    };
}
