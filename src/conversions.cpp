#include "smmap/conversions.h"

namespace dmm = deformable_manipulation_msgs;

namespace smmap
{
    std::vector<TransitionEstimation::StateMicrostepsPair> ToTrajectory(
            const WorldState& initial_world_state,
            const RRTPath& path,
            const dmm::TransitionTest& test,
            const dmm::TransitionTestResult& test_result)
    {
        static ros::NodeHandle nh;
        static const auto simsteps_per_gripper_cmd = ROSHelpers::GetParamRequiredDebugLog<int>(nh, "deform_simulator_node/num_simsteps_per_gripper_command", __func__).Get();

        const bool has_last_action = test.final_num_substeps > 0;
        const auto path_num_steps = path.size();
        const auto path_total_substeps = std::accumulate(test.path_num_substeps.begin(), test.path_num_substeps.end(), test.final_num_substeps);
        const auto path_cummulative_substeps = [&]
        {
            auto res = std::vector<int>(path_num_steps);
            std::partial_sum(test.path_num_substeps.begin(), test.path_num_substeps.end(), res.begin());
            return res;
        }();

        // Make sure that the data is in the format we're expecting
        assert(test.path_num_substeps.size() == path_num_steps);
        // I.e.; the path starts at the same place that the grippers are already at
        assert(test.path_num_substeps.at(0) == 0);

        if ((int)test_result.microsteps_all.size() != (path_total_substeps * simsteps_per_gripper_cmd))
        {
            ROS_WARN_STREAM_NAMED("conversions", "Only a partial trajectory exists.");
        }

        const auto total_steps = has_last_action ? path_num_steps + 1 : path_num_steps;
        std::vector<TransitionEstimation::StateMicrostepsPair> trajectory;
        trajectory.reserve(total_steps);

        // Add the first state with no history
        {
            const TransitionEstimation::State tes =
            {
                initial_world_state.object_configuration_,
                std::make_shared<RubberBand>(*path.front().band()),
                std::make_shared<RubberBand>(*path.front().band()),
                initial_world_state.rope_node_transforms_
            };
            trajectory.push_back({tes, std::vector<WorldState>(0)});
        }

        // Add the rest of the states other than the last step
        for (size_t idx = 1; idx < path_num_steps; ++idx)
        {
            const auto microsteps_start_idx = path_cummulative_substeps.at(idx - 1) * simsteps_per_gripper_cmd;
            const auto microsteps_end_idx = path_cummulative_substeps.at(idx) * simsteps_per_gripper_cmd;

            // If the grippers do not move, then skip this part of the path
            if (microsteps_end_idx == microsteps_start_idx)
            {
                continue;
            }

            // Ensure that we don't overflow the end of the vector (assuming one set of data at the end for the "last step")
            // Given the earlier assertions; only a logic error or an imcomplete trajectory would trigger this
            if ((test_result.microsteps_all.begin() + microsteps_start_idx > test_result.microsteps_all.end() - (test.final_num_substeps * simsteps_per_gripper_cmd)) ||
                (test_result.microsteps_all.begin() + microsteps_end_idx   > test_result.microsteps_all.end() - (test.final_num_substeps * simsteps_per_gripper_cmd)))
            {
                ROS_WARN_STREAM_NAMED("conversions", "Results only contains " << idx << " path steps. Path steps anticipated: " << path_num_steps);
                return trajectory;
            }

            const std::vector<dmm::WorldState> dmm_microsteps(
                        test_result.microsteps_all.begin() + microsteps_start_idx,
                        test_result.microsteps_all.begin() + microsteps_end_idx);
            const auto microsteps = ConvertToEigenFeedback(dmm_microsteps);

            const auto planned_band = std::make_shared<RubberBand>(*path[idx].band());
            const TransitionEstimation::State tes =
            {
                microsteps.back().object_configuration_,
                RubberBand::BandFromWorldState(microsteps.back(), *planned_band),
                planned_band,
                microsteps.back().rope_node_transforms_,
            };
            // Shortcut if the band becomes overstretched
            if (tes.rubber_band_->isOverstretched())
            {
                ROS_WARN_STREAM_NAMED("conversions", "Band overstretched at index " << idx << ". Path steps anticipated: " << path_num_steps);
                return trajectory;
            }
            trajectory.push_back({tes, microsteps});
        }

        // Propagate the planned band the last step, and record the resulting state
        if (has_last_action)
        {
            const WorldState end = ConvertToEigenFeedback(test_result.microsteps_last_action.back());

            auto planned_band = std::make_shared<RubberBand>(*path.back().band());
            planned_band->forwardPropagate(ToGripperPositions(end.all_grippers_single_pose_), false);
            const auto tes = TransitionEstimation::State
            {
                end.object_configuration_,
                RubberBand::BandFromWorldState(end, *planned_band),
                planned_band,
                end.rope_node_transforms_
            };
            if (tes.rubber_band_->isOverstretched())
            {
                ROS_WARN_STREAM_NAMED("to_traj", "Band overstretched at last action.");
                return trajectory;
            }
            trajectory.push_back({tes, ConvertToEigenFeedback(test_result.microsteps_last_action)});
        }


        return trajectory;
    }
}
