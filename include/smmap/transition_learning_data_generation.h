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

namespace smmap
{
    struct TransitionSimulationRecord
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        TransitionEstimation::StateTransition template_;
        ObjectPointSet template_band_surface_;

        Eigen::Isometry3d center_of_rotation_;
        Eigen::Isometry3d transform_applied_;

        TransitionEstimation::StateTransition tested_;
        ObjectPointSet tested_band_surface_;

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
        const double gripper_min_distance_to_obstacles_;

        Eigen::Isometry3d gripper_a_starting_pose_;
        Eigen::Isometry3d gripper_a_ending_pose_;
        Eigen::Isometry3d gripper_b_starting_pose_;
        Eigen::Isometry3d gripper_b_ending_pose_;
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
        std::vector<std::string> getDataFileList();
        TransitionSimulationRecord loadSimRecord(const std::string& filename);

    public:
        void runTests(const bool generate_new_test_data);

    private:
        class DataGeneration
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            DataGeneration(const TransitionTesting& framework);
            void generateTestData(const std::string& data_folder);

        private:
            const TransitionTesting& framework_;
            arc_helpers::RandomRotationGenerator random_rotation_distribution_;

            // Stored here because Eigen + tuple = bad
            EigenHelpers::VectorIsometry3d random_test_rope_nodes_start_;
            AllGrippersSinglePose random_test_starting_gripper_poses_;
            AllGrippersSinglePose random_test_ending_gripper_poses_;
            Eigen::Isometry3d random_test_transform_applied_;

            AllGrippersPoseTrajectory generateTestPath(
                    const AllGrippersSinglePose& gripper_target_poses);

            void generateRandomTest(
                    std::mt19937_64& generator,
                    const TransitionEstimation::StateTransition& trans);
        };
        friend class DataGeneration;

        class SE3Prediction
        {
        public:
            SE3Prediction(const TransitionTesting& framework);

            std::map<std::string, std::vector<RubberBand>> predictAll(
                    const TransitionEstimation::StateTransition& stored_trans,
                    const RubberBand& test_band_start,
                    const PairGripperPositions& ending_gripper_positions);

            void predictBasedOnPlannedBand(
                    const TransitionEstimation::StateTransition& stored_trans);

            void predictBasedOnExecutedBand(
                    const TransitionEstimation::StateTransition& stored_trans);

            void visualizePrediction();

            bool prediction_valid_;
            const TransitionTesting& framework_;
            static constexpr auto BASENAME = "SE3_ALIGNMENT";

            RubberBand::ConstPtr test_band_;
            PairGripperPositions test_action_;

            size_t num_gripper_steps_;
            std::vector<RubberBand> stored_bands_;
            ObjectPointSet warping_target_points_;

            ObjectPointSet warping_template_points_planned_;
            ObjectPointSet template_planned_band_aligned_to_target_;
            ObjectPointSet stored_bands_planned_aligned_to_target_;

            ObjectPointSet warping_template_points_executed_;
            ObjectPointSet template_executed_band_aligned_to_target_;
            ObjectPointSet stored_bands_executed_aligned_to_target_;

            std::map<std::string, std::vector<RubberBand>> results_;
        };
        friend class SE3Prediction;

        class TPSPrediction
        {
        public:
            TPSPrediction(const TransitionTesting& framework);

            std::map<std::string, std::vector<RubberBand>> predictAll(
                    const TransitionEstimation::StateTransition& stored_trans,
                    const RubberBand& starting_band,
                    const PairGripperPositions& action);

            const TransitionTesting& framework_;
            static constexpr auto BASENAME = "TPS_WARPING";
            std::vector<RubberBand> stored_bands_;
            std::map<std::string, std::vector<RubberBand>> results_;
        };
        friend class TPSPrediction;
    };
}
