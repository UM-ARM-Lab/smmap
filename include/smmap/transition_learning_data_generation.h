#pragma once

#include <random>
#include <memory>
#include <string>

#include <arc_utilities/arc_helpers.hpp>
#include <deformable_manipulation_msgs/TransitionTest.h>
#include <smmap_utilities/visualization_tools.h>
#include "smmap/learned_transitions.h"
#include "smmap/quinlan_rubber_band.h"
#include "smmap/robot_interface.h"
#include "smmap/band_rrt.h"

namespace smmap
{
    struct TransitionSimulationRecord
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        TransitionEstimation::StateTransition template_;
        ObjectPointSet template_band_surface_;

        TransitionEstimation::StateTransition tested_;
        ObjectPointSet tested_band_surface_;

        TransitionEstimation::TransitionAdaptationResult adaptation_result_;

        uint64_t serializeSelf(std::vector<uint8_t>& buffer) const;

        static uint64_t Serialize(
                const TransitionSimulationRecord& test_results,
                std::vector<uint8_t>& buffer);

        static std::pair<TransitionSimulationRecord, uint64_t> Deserialize(
                const std::vector<uint8_t>& buffer,
                const uint64_t current,
                const RubberBand& template_band);

        bool operator==(const TransitionSimulationRecord& other) const;

        void visualize(const Visualizer::Ptr& vis) const;
    };

    class TransitionTesting
    {
    private:
        const std::shared_ptr<ros::NodeHandle> nh_;
        const std::shared_ptr<ros::NodeHandle> ph_;
        const RobotInterface::Ptr robot_;
        const Visualizer::Ptr vis_;
        const bool visualize_gripper_motion_;

        const unsigned long seed_;
        const std::shared_ptr<std::mt19937_64> generator_;

        // Note that work_space_grid_ and the environment_sdf_ are using different
        // resolutions due to the way the SDF is created in CustomScene
        const sdf_tools::SignedDistanceField::ConstPtr sdf_;
        const XYZGrid work_space_grid_;

        std::shared_ptr<BandRRT::WorldParams> world_params_;
        BandRRT::PlanningParams planning_params_;
        BandRRT::SmoothingParams smoothing_params_;
        BandRRT::TaskParams task_params_;

        Eigen::Isometry3d gripper_a_starting_pose_;
        Eigen::Isometry3d gripper_b_starting_pose_;
        Eigen::Vector3d gripper_a_action_vector_;
        Eigen::Vector3d gripper_b_action_vector_;
        const Eigen::Isometry3d experiment_center_of_rotation_;

        const DeformableType deformable_type_;
        const TaskType task_type_;

        const WorldState initial_world_state_;
        RubberBand::Ptr band_;
        TransitionEstimation::Ptr transition_estimator_;

        const std::string data_folder_;
        const std::string sim_test_result_suffix_;
        const std::string prediction_result_suffix_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        TransitionTesting(
                std::shared_ptr<ros::NodeHandle> nh,
                std::shared_ptr<ros::NodeHandle> ph,
                RobotInterface::Ptr robot,
                Visualizer::Ptr vis);

    private:
        void initialize(const WorldState& world_state);
        void initializeBand(const WorldState& world_state);
        void initializeRRTParams();
        void clampGripperDeltas(
                Eigen::Ref<Eigen::Vector3d> a_delta,
                Eigen::Ref<Eigen::Vector3d> b_delta) const;
        std::vector<std::string> getDataFileList();
        TransitionSimulationRecord loadSimRecord(const std::string& filename);

    public:
        void runTests(const bool generate_new_test_data);

    private:
        class DataGeneration
        {
        public:
            DataGeneration(const TransitionTesting& framework);
            void generateTestData(const std::string& data_folder);

        private:
            const TransitionTesting& fw_;

            AllGrippersPoseTrajectory generateTestPath(
                    const AllGrippersSinglePose& gripper_target_poses);
        };
        friend class DataGeneration;

        class DataProcessing
        {
        public:
        private:
        };
        friend class DataProcessing;
    };
}
