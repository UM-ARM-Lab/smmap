#pragma once

#include <random>
#include <memory>

#include <smmap_utilities/visualization_tools.h>
#include "smmap/learned_transitions.h"
#include "smmap/quinlan_rubber_band.h"
#include "smmap/robot_interface.h"

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

        RubberBand::Ptr band_;
        TransitionEstimation::Ptr transition_estimator_;

    public:
        struct TransitionTestResults
        {
        public:
            TransitionEstimation::StateTransition template_;
            smmap_utilities::ObjectPointSet tps_control_points_;
            smmap_utilities::ObjectPointSet template_band_surface_;

            TransitionEstimation::StateTransition tested_;
            smmap_utilities::ObjectPointSet tps_target_points_;

            smmap_utilities::ObjectPointSet predicted_final_band_surface_;
            smmap_utilities::ObjectPointSet final_band_surface_;

            uint64_t serializeSelf(std::vector<uint8_t>& buffer) const;

            static uint64_t Serialize(
                    const TransitionTestResults& test_results,
                    std::vector<uint8_t>& buffer)
            {
                const auto starting_bytes = buffer.size();
                uint64_t bytes_written = 0;
                bytes_written += test_results.template_.serializeSelf(buffer);
                bytes_written += arc_utilities::SerializeEigen(test_results.tps_control_points_, buffer);
                bytes_written += arc_utilities::SerializeEigen(test_results.template_band_surface_, buffer);
                bytes_written += test_results.tested_.serializeSelf(buffer);
                bytes_written += arc_utilities::SerializeEigen(test_results.tps_target_points_, buffer);
                bytes_written += arc_utilities::SerializeEigen(test_results.predicted_final_band_surface_, buffer);
                bytes_written += arc_utilities::SerializeEigen(test_results.final_band_surface_, buffer);
                const auto ending_bytes = buffer.size();
                assert(ending_bytes - starting_bytes == bytes_written);
                const auto deserialized = Deserialize(buffer, starting_bytes, *test_results.template_.starting_state_.rubber_band_);
                assert(test_results == deserialized.first);
                assert(bytes_written = deserialized.second);
                return bytes_written;;
            }

            static std::pair<TransitionTestResults, uint64_t> Deserialize(
                    const std::vector<uint8_t>& buffer,
                    const uint64_t current,
                    const RubberBand& template_band)
            {
                TransitionTestResults result;
                uint64_t bytes_read = 0;

                const auto template_deserialized = TransitionEstimation::StateTransition::Deserialize(buffer, current + bytes_read, template_band);
                result.template_ = template_deserialized.first;
                bytes_read += template_deserialized.second;

                const auto tps_control_points_deserialized = arc_utilities::DeserializeEigen<smmap_utilities::ObjectPointSet>(buffer, current + bytes_read);
                result.tps_control_points_ = tps_control_points_deserialized.first;
                bytes_read = tps_control_points_deserialized.second;

                const auto template_band_surface_deserialized = arc_utilities::DeserializeEigen<smmap_utilities::ObjectPointSet>(buffer, current + bytes_read);
                result.template_band_surface_ = template_band_surface_deserialized.first;
                bytes_read = template_band_surface_deserialized.second;

                const auto tested_deserialized = TransitionEstimation::StateTransition::Deserialize(buffer, current + bytes_read, template_band);
                result.tested_ = tested_deserialized.first;
                bytes_read += tested_deserialized.second;

                const auto tps_target_points_deserialized = arc_utilities::DeserializeEigen<smmap_utilities::ObjectPointSet>(buffer, current + bytes_read);
                result.tps_target_points_ = tps_target_points_deserialized.first;
                bytes_read = tps_target_points_deserialized.second;

                const auto predicted_final_band_surface_deserialized = arc_utilities::DeserializeEigen<smmap_utilities::ObjectPointSet>(buffer, current + bytes_read);
                result.predicted_final_band_surface_ = predicted_final_band_surface_deserialized.first;
                bytes_read = predicted_final_band_surface_deserialized.second;

                const auto final_band_surface_deserialized = arc_utilities::DeserializeEigen<smmap_utilities::ObjectPointSet>(buffer, current + bytes_read);
                result.final_band_surface_ = final_band_surface_deserialized.first;
                bytes_read = final_band_surface_deserialized.second;

                return {result, bytes_read};
            }

            bool operator==(const TransitionTestResults& other) const
            {
                if (template_ != other.template_)
                {
                    return false;
                }
                if ((tps_control_points_.array() != other.tps_control_points_.array()).any())
                {
                    return false;
                }
                if ((template_band_surface_.array() != other.template_band_surface_.array()).any())
                {
                    return false;
                }
                if (tested_ != other.tested_)
                {
                    return false;
                }
                if ((tps_target_points_.array() != other.tps_target_points_.array()).any())
                {
                    return false;
                }
                if ((predicted_final_band_surface_.array() != other.predicted_final_band_surface_.array()).any())
                {
                    return false;
                }
                if ((final_band_surface_.array() != other.final_band_surface_.array()).any())
                {
                    return false;
                }
                return true;
            }
        };

        DataGeneration(
                std::shared_ptr<ros::NodeHandle> nh,
                std::shared_ptr<ros::NodeHandle> ph,
                RobotInterface::Ptr robot,
                const smmap_utilities::Visualizer::ConstPtr& vis);

    private:
        void initialize(const WorldState& world_state);
        void initializeBand(const WorldState& world_state);

    public:
        void runTests();

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
    };
}
