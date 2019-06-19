#include "smmap/transition_learning_data_generation.h"
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/serialization_ros.hpp>
#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/log.hpp>
#include <arc_utilities/math_helpers.hpp>
#include <smmap_utilities/neighbours.h>
#include <boost/filesystem.hpp>
#include <deformable_manipulation_experiment_params/conversions.hpp>
#include <deformable_manipulation_experiment_params/utility.hpp>
#include <deformable_manipulation_msgs/GenerateTransitionDataAction.h>

#include "smmap/band_rrt.h"

using namespace arc_utilities;
using namespace arc_helpers;
using namespace Eigen;
using namespace EigenHelpers;
using namespace EigenHelpersConversions;
namespace dmm = deformable_manipulation_msgs;
using ColorBuilder = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>;

////////////////////////////////////////////////////////////////////////////////
//          Conversions and Random Helpers
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    static std::string ToString(const Eigen::Vector3d& mat)
    {
        std::stringstream ss;
        ss << mat.x() << "_" << mat.y() << "_" << mat.z() ;
        return ss.str();
    }

    static TransitionEstimation::StateTransition ToStateTransition(
            const dmm::TransitionTestResult& test,
            const RRTPath& path)
    {
        const auto template_band = *path.back().band();

        const WorldState start = ConvertToEigenFeedback(test.start_after_following_path);
        const std::vector<WorldState> microsteps_all = ConvertToEigenFeedback(test.microsteps_all);
        const std::vector<WorldState> microsteps_last = ConvertToEigenFeedback(test.microsteps_last_action);
        const WorldState& end = microsteps_all.back();

        const auto start_state = TransitionEstimation::State
        {
            start.object_configuration_,
            RubberBand::BandFromWorldState(start, template_band),
            std::make_shared<RubberBand>(*path.back().band()),
            start.rope_node_transforms_
        };

        // Propagate the planned band the last step
        auto band = std::make_shared<RubberBand>(*path.back().band());
        band->forwardPropagate(ToGripperPositions(end.all_grippers_single_pose_), false);
        const auto end_state = TransitionEstimation::State
        {
            end.object_configuration_,
            RubberBand::BandFromWorldState(end, template_band),
            band,
            end.rope_node_transforms_
        };

        std::vector<RubberBand::Ptr> microsteps_last_bands;
        microsteps_last_bands.reserve(microsteps_last.size());
        for (size_t idx = 0; idx < microsteps_last.size(); ++idx)
        {
            microsteps_last_bands.push_back(std::make_shared<RubberBand>(template_band));
            if (!microsteps_last_bands.back()->resetBand(microsteps_last[idx]))
            {
                throw_arc_exception(std::runtime_error, "Unable to extract surface");
            }
        }

        return TransitionEstimation::StateTransition
        {
            start_state,
            end_state,
            start_state.planned_rubber_band_->getEndpoints(),
            end_state.planned_rubber_band_->getEndpoints(),
            microsteps_last,
            microsteps_last_bands
        };
    }

    typedef std::pair<TransitionEstimation::State, std::vector<WorldState>> StateMicrostepsPair;
    std::vector<StateMicrostepsPair> TransitionTesting::toTrajectory(
            const dmm::TransitionTestResult& test_result,
            const RRTPath& path)
    {
        const auto gripper_a_ending_pose = GeometryPoseToEigenIsometry3d(test_result.microsteps_last_action.back().gripper_poses.at(0));
        const auto gripper_b_ending_pose = GeometryPoseToEigenIsometry3d(test_result.microsteps_last_action.back().gripper_poses.at(1));
        const auto test = robot_->toRosTransitionTest(
                    initial_world_state_.rope_node_transforms_,
                    initial_world_state_.all_grippers_single_pose_,
                    RRTPathToGrippersPoseTrajectory(path),
                    {gripper_a_ending_pose, gripper_b_ending_pose});
        const auto path_num_steps = path.size();
        const auto simsteps_per_gripper_cmd = ROSHelpers::GetParamRequiredDebugLog<int>(*nh_, "deform_simulator_node/num_simsteps_per_gripper_command", __func__).Get();
        const auto path_total_substeps = std::accumulate(test.path_num_substeps.begin(), test.path_num_substeps.end(), test.final_num_substeps);
        const auto path_cummulative_substeps = [&]
        {
            auto res = std::vector<int>(path_num_steps);
            std::partial_sum(test.path_num_substeps.begin(), test.path_num_substeps.end(), res.begin());
            return res;
        }();

        // Make sure that the data is in the format we're expecting
        assert(test.path_num_substeps.size() == path_num_steps);
        assert((int)test_result.microsteps_all.size() == (path_total_substeps * simsteps_per_gripper_cmd));
        assert(test.path_num_substeps.at(0) == 0); // I.e.; the path starts at the same place that the grippers are already at

        // Add an extra state for the start, remove one to exclude the 'dud' first step in the path,
        // and add one for the last step after reaching the end of the path
        std::vector<StateMicrostepsPair> trajectory;
        trajectory.reserve(path_num_steps + 1);

        // Add the first state with no history
        {
            const TransitionEstimation::State tes =
            {
                initial_world_state_.object_configuration_,
                std::make_shared<RubberBand>(*initial_band_),
                std::make_shared<RubberBand>(*initial_band_),
                initial_world_state_.rope_node_transforms_
            };
            trajectory.push_back({tes, std::vector<WorldState>(0)});
        }

        // Add the rest of the states other than the last step
        for (size_t idx = 1; idx < path_num_steps; ++idx)
        {
            const auto microsteps_start_idx = path_cummulative_substeps[idx - 1] * simsteps_per_gripper_cmd;
            const auto microsteps_end_idx = path_cummulative_substeps[idx] * simsteps_per_gripper_cmd;

            if (microsteps_end_idx == microsteps_start_idx)
            {
                continue;
            }

            // Ensure that we don't overflow the end of the vector (assuming one set of data at the end for the "last step")
            // Given the earlier assertions; only a logic error in this function would trigger this
            assert(test_result.microsteps_all.begin() + microsteps_start_idx <= test_result.microsteps_all.end() - (test.final_num_substeps * simsteps_per_gripper_cmd));
            assert(test_result.microsteps_all.begin() + microsteps_end_idx <= test_result.microsteps_all.end() - (test.final_num_substeps * simsteps_per_gripper_cmd));

            const std::vector<dmm::WorldState> dmm_microsteps(
                        test_result.microsteps_all.begin() + microsteps_start_idx,
                        test_result.microsteps_all.begin() + microsteps_end_idx);
            const auto microsteps = ConvertToEigenFeedback(dmm_microsteps);

            const TransitionEstimation::State tes =
            {
                microsteps.back().object_configuration_,
                RubberBand::BandFromWorldState(microsteps.back(), *initial_band_),
                std::make_shared<RubberBand>(*path[idx].band()),
                microsteps.back().rope_node_transforms_,
            };
            trajectory.push_back({tes, microsteps});
        }

        // Propagate the planned band the last step, and record the resulting state
        {
            const WorldState end = ConvertToEigenFeedback(test_result.microsteps_last_action.back());

            auto planned_band = std::make_shared<RubberBand>(*path.back().band());
            planned_band->forwardPropagate(ToGripperPositions(VectorGeometryPoseToVectorIsometry3d(test.final_gripper_targets)), false);
            const auto tes = TransitionEstimation::State
            {
                end.object_configuration_,
                RubberBand::BandFromWorldState(end, *initial_band_),
                planned_band,
                end.rope_node_transforms_
            };
            trajectory.push_back({tes, ConvertToEigenFeedback(test_result.microsteps_last_action)});
        }

        return trajectory;
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Transition Simulation Record
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    uint64_t TransitionSimulationRecord::serializeSelf(std::vector<uint8_t>& buffer) const
    {
        const auto starting_bytes = buffer.size();
        uint64_t bytes_written = 0;
        bytes_written += template_.serialize(buffer);
        bytes_written += arc_utilities::SerializeEigen(template_band_surface_, buffer);
        bytes_written += tested_.serialize(buffer);
        bytes_written += arc_utilities::SerializeEigen(tested_band_surface_, buffer);
        bytes_written += adaptation_result_.serialize(buffer);

        const auto ending_bytes = buffer.size();
        assert(ending_bytes - starting_bytes == bytes_written);
        const auto deserialized = Deserialize(buffer, starting_bytes, *template_.starting_state_.rubber_band_);
        assert(bytes_written = deserialized.second);
        assert(*this == deserialized.first);
        return bytes_written;;
    }

    uint64_t TransitionSimulationRecord::Serialize(
            const TransitionSimulationRecord& test_results,
            std::vector<uint8_t>& buffer)
    {
        return test_results.serializeSelf(buffer);
    }

    std::pair<TransitionSimulationRecord, uint64_t> TransitionSimulationRecord::Deserialize(
            const std::vector<uint8_t>& buffer,
            const uint64_t current,
            const RubberBand& template_band)
    {
        uint64_t bytes_read = 0;

        const auto template_deserialized =
                TransitionEstimation::StateTransition::Deserialize(buffer, current + bytes_read, template_band);
        bytes_read += template_deserialized.second;

        const auto template_band_surface_deserialized =
                arc_utilities::DeserializeEigen<ObjectPointSet>(buffer, current + bytes_read);
        bytes_read += template_band_surface_deserialized.second;

        const auto tested_deserialized =
                TransitionEstimation::StateTransition::Deserialize(buffer, current + bytes_read, template_band);
        bytes_read += tested_deserialized.second;

        const auto tested_band_surface_deserialized =
                arc_utilities::DeserializeEigen<ObjectPointSet>(buffer, current + bytes_read);
        bytes_read += tested_band_surface_deserialized.second;

        const auto adaptation_result_deserialized =
                TransitionEstimation::TransitionAdaptationResult::Deserialize(buffer, current + bytes_read, template_band);
        bytes_read += adaptation_result_deserialized.second;

        TransitionSimulationRecord record =
        {
            template_deserialized.first,
            template_band_surface_deserialized.first,
            tested_deserialized.first,
            tested_band_surface_deserialized.first,
            adaptation_result_deserialized.first
        };
        return {record, bytes_read};
    }

    bool TransitionSimulationRecord::operator==(const TransitionSimulationRecord& other) const
    {
        if (template_ != template_)
        {
            return false;
        }
        if (template_band_surface_ != other.template_band_surface_)
        {
            return false;
        }
        if (tested_ != other.tested_)
        {
            return false;
        }
        if (tested_band_surface_ != other.tested_band_surface_)
        {
            return false;
        }
        if (adaptation_result_ != other.adaptation_result_)
        {
            return false;
        }
        return true;
    }

    std::vector<Visualizer::NamespaceId> TransitionSimulationRecord::visualize(
            const std::string& basename,
            const Visualizer::Ptr& vis) const
    {
        std::vector<Visualizer::NamespaceId> marker_ids;

        constexpr bool vis_template_starting_band = true;
        constexpr bool vis_template_ending_executed_band = true;
        constexpr bool vis_template_executed_band_surface = true;
        constexpr bool vis_test_start_planned_band = true;
        constexpr bool vis_test_executed_band = true;
        constexpr bool vis_test_executed_band_surface = true;
        constexpr bool vis_adapt_default_next_band = true;
        constexpr bool vis_adapt_target_band_and_action = true;
        constexpr bool vis_adapt_template_band_and_action = true;
        constexpr bool vis_adapt_template_aligned = true;
        constexpr bool vis_adapt_next_band_points_to_smooth = true;
        constexpr bool vis_adapt_transformed_band_surface_points = true;
        constexpr bool vis_adapt_retightend_band_surface = true;
        constexpr bool vis_adapt_final_result = true;

        // Template - starting planned band
        if (vis_template_starting_band)
        {
            const auto color = Visualizer::Green();
            const auto name = basename + "template__start";
            const auto new_ids = template_.starting_state_.planned_rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Template - ending executed band
        if (vis_template_ending_executed_band)
        {
            const auto color = Visualizer::Cyan();
            const auto name = basename + "template__executed";
            const auto new_ids = template_.ending_state_.rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Template - Executed band surface
        if (vis_template_executed_band_surface)
        {
            const auto start_color = Visualizer::Green();
            const auto end_color = Visualizer::Cyan();
            const auto name = basename + "template__band_surface";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, template_band_surface_, template_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Test - start planned band
        if (vis_test_start_planned_band)
        {
            const auto color = Visualizer::Yellow();
            const auto name = basename + "tested__start";
            const auto new_ids = tested_.starting_state_.planned_rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Test - exectued band
        if (vis_test_executed_band)
        {
            const auto color = Visualizer::Orange();
            const auto name = basename + "tested__executed";
            const auto new_ids = tested_.ending_state_.rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Test - Executed band surface
        if (vis_test_executed_band_surface)
        {
            const auto start_color = Visualizer::Yellow();
            const auto end_color = Visualizer::Orange();
            const auto name = basename + "tested__band_surface";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, tested_band_surface_, tested_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - default next band
        if (vis_adapt_default_next_band)
        {
            const auto color = Visualizer::Red();
            const auto name = basename + "adaptation__default_next_band";
            const auto new_ids = adaptation_result_.default_next_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - target band and action
        if (vis_adapt_target_band_and_action)
        {
            const auto color = Visualizer::Yellow();
            const auto name = basename + "adaptation__target_points_to_match";
            std::vector<std_msgs::ColorRGBA> colors;
            const auto num_divs = (adaptation_result_.target_points_to_match_.cols() - 1);
            for (ssize_t idx = 0; idx <= num_divs; ++idx)
            {
                colors.push_back(InterpolateColor(color, Visualizer::Red(), (float)idx / (float)num_divs));
            }
            const auto new_ids = vis->visualizePoints(name, adaptation_result_.target_points_to_match_, colors, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - template band and action
        if (vis_adapt_template_band_and_action)
        {
            const auto color = Visualizer::Green();
            const auto name = basename + "adaptation__template_points_to_align";
            std::vector<std_msgs::ColorRGBA> colors;
            const auto num_divs = adaptation_result_.template_points_to_align_.cols() - 1;
            for (ssize_t idx = 0; idx <= num_divs; ++idx)
            {
                colors.push_back(InterpolateColor(color, Visualizer::Red(), (float)idx / (float)num_divs));
            }
            const auto new_ids = vis->visualizePoints(name, adaptation_result_.template_points_to_align_, colors, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - template aligned
        if (vis_adapt_template_aligned)
        {
            const auto color = Visualizer::Magenta();
            const auto name = basename + "adaptation__template_aligned_to_target";
            std::vector<std_msgs::ColorRGBA> colors;
            const auto num_divs = adaptation_result_.template_planned_band_aligned_to_target_.cols() - 1;
            for (ssize_t idx = 0; idx <= num_divs; ++idx)
            {
                colors.push_back(InterpolateColor(color, Visualizer::Red(), (float)idx / (float)num_divs));
            }
            const auto new_ids = vis->visualizePoints(name, adaptation_result_.template_planned_band_aligned_to_target_, colors, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - next_band_points_to_smooth_
        if (vis_adapt_next_band_points_to_smooth)
        {
            const auto color = Visualizer::Seafoam();
            const auto name = basename + "adaptation__next_band_points_to_smooth";
            const auto new_ids = vis->visualizePoints(name, adaptation_result_.next_band_points_to_smooth_, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - transformed_band_surface_points
        if (vis_adapt_transformed_band_surface_points)
        {
            const auto start_color = Visualizer::Blue();
            const auto end_color = Visualizer::Seafoam();
            const auto name = basename + "adaptation__transformed_band_surface_points";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, adaptation_result_.transformed_band_surface_points_, template_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - re-tightened band surface
        if (vis_adapt_retightend_band_surface)
        {
            const auto start_color = Visualizer::Olive();
            const auto end_color = Visualizer::Coral();
            const auto name = basename + "adaptation__tightened_transformed_bands_surface";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, adaptation_result_.tightened_transformed_bands_surface_, template_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - final result
        if (vis_adapt_final_result)
        {
            const auto color = Visualizer::Coral();
            const auto name = basename + "adaptation__result";
            const auto new_ids = adaptation_result_.result_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }

        return marker_ids;
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Transition Testing
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    TransitionTesting::TransitionTesting(
            std::shared_ptr<ros::NodeHandle> nh,
            std::shared_ptr<ros::NodeHandle> ph,
            RobotInterface::Ptr robot,
            Visualizer::Ptr vis)
        : nh_(std::move(nh))
        , ph_(std::move(ph))
        , robot_(std::move(robot))
        , vis_(std::move(vis))
        , visualize_gripper_motion_(!GetDisableAllVisualizations(*ph_) && GetVisualizeGripperMotion(*ph_))

        , seed_(GetPlannerSeed(*ph_))
        , generator_(std::make_shared<std::mt19937_64>(seed_))

        , sdf_(GetEnvironmentSDF(*nh_))
        , work_space_grid_(sdf_->GetOriginTransform(),
                           sdf_->GetFrame(),
                           GetWorldXStep(*nh_),
                           GetWorldYStep(*nh_),
                           GetWorldZStep(*nh_),
                           GetWorldXNumSteps(*nh_),
                           GetWorldYNumSteps(*nh_),
                           GetWorldZNumSteps(*nh_))

        , gripper_a_starting_pose_(GetPoseFromParamSerer(*ph_, "gripper_a_test_start", true))
        , gripper_b_starting_pose_(GetPoseFromParamSerer(*ph_, "gripper_b_test_start", true))
        , gripper_a_action_vector_(GetVector3FromParamServer(*ph_, "gripper_a_action_vector"))
        , gripper_b_action_vector_(GetVector3FromParamServer(*ph_, "gripper_b_action_vector"))
        , experiment_center_of_rotation_(Isometry3d(Translation3d(GetVector3FromParamServer(*ph_, "experiment_cor"))))

        , deformable_type_(GetDeformableType(*nh_))
        , task_type_(GetTaskType(*nh_))
        , initial_world_state_(robot_->start())

        , data_folder_(ROSHelpers::GetParam<std::string>(*ph_, "data_folder", "/tmp/transition_learning_data_generation"))

        , next_vis_prefix_(0)
        , next_vis_id_sub_(nh_->subscribe("transition_vis/set_next_vis_id", 1, &TransitionTesting::setNextVisId, this))
        , remove_visualization_(nh_->advertiseService("transition_vis/remove_visualization", &TransitionTesting::removeVisualizationCallback, this))

        , source_valid_(false)
        , set_transition_adaptation_source_(nh_->advertiseService("transition_vis/set_transition_adaptation_source", &TransitionTesting::setTransitionAdaptationSourceCallback, this))
        , add_transition_adaptation_visualization_(nh_->advertiseService("transition_vis/add_transition_adaptation_visualization", &TransitionTesting::addTransitionAdaptationVisualizationCallback, this))

        , add_mistake_example_visualization_(nh_->advertiseService("transition_vis/add_mistake_example_visualization", &TransitionTesting::addMistakeExampleVisualizationCallback, this))
    {
        std::srand((unsigned int)seed_);
        initialize(initial_world_state_);

        // Used for generating data
        gripper_a_starting_pose_.linear() = initial_world_state_.all_grippers_single_pose_[0].linear();
        gripper_b_starting_pose_.linear() = initial_world_state_.all_grippers_single_pose_[1].linear();
        clampGripperDeltas(gripper_a_action_vector_, gripper_b_action_vector_);

        vis_->visualizeAxes("center_of_rotation",   experiment_center_of_rotation_, 0.1, 0.005, 1);
        vis_->visualizeAxes("gripper_a_start",      gripper_a_starting_pose_,       0.1, 0.005, 1);
        vis_->visualizeAxes("gripper_b_start",      gripper_b_starting_pose_,       0.1, 0.005, 1);
        vis_->visualizeAxes("gripper_a_end",        Translation3d(gripper_a_action_vector_) * gripper_a_starting_pose_, 0.1, 0.005, 1);
        vis_->visualizeAxes("gripper_b_end",        Translation3d(gripper_b_action_vector_) * gripper_b_starting_pose_, 0.1, 0.005, 1);
    }

    void TransitionTesting::initialize(const WorldState& world_state)
    {
        initializeBand(world_state);
        transition_estimator_ = std::make_shared<TransitionEstimation>(
                    nh_, ph_, sdf_, work_space_grid_, vis_, *initial_band_);
        initializeRRTParams();
    }

    void TransitionTesting::initializeBand(const WorldState& world_state)
    {
        // Extract the maximum distance between the grippers
        // This assumes that the starting position of the grippers is at the maximum "unstretched" distance
        const auto& grippers_starting_poses = world_state.all_grippers_single_pose_;
        const double max_calced_band_length =
                (grippers_starting_poses[0].translation() - grippers_starting_poses[1].translation()).norm()
                * GetMaxStretchFactor(*ph_);
        const auto max_band_length = GetMaxBandLength(*ph_);
        ROS_ERROR_STREAM_COND_NAMED(!CloseEnough(max_calced_band_length, max_band_length, 1e-3),
                                    "data_generation",
                                    "Calc'd max band distance is: " << max_calced_band_length <<
                                    " but the ros param saved distance is " << max_band_length <<
                                    ". Double check the stored value in the roslaunch file.");

        // Find the shortest path through the object, between the grippers, while following nodes of the object.
        // Used to determine the starting position of the rubber band at each timestep
        const auto num_nodes = world_state.object_configuration_.cols();
        std::function<std::vector<ssize_t>(const ssize_t node)> neighbour_fn;
        switch (deformable_type_)
        {
            case ROPE:
            {
                LineNeighbours neighbours_calc(num_nodes);
                neighbour_fn = [neighbours_calc] (const ssize_t node)
                {
                    return neighbours_calc.getNodeNeighbours(node);
                };
                break;
            }
            case CLOTH:
            {
                Grid4Neighbours neighbours_calc(num_nodes, GetClothNumControlPointsX(*nh_));
                neighbour_fn = [neighbours_calc] (const ssize_t node)
                {
                    return neighbours_calc.getNodeNeighbours(node);
                };
                break;
            }
            default:
                throw_arc_exception(std::invalid_argument, "Invalid deformable type; this should not be possible");
        }

        // Create the initial rubber band
        const double resampled_band_max_pointwise_dist = work_space_grid_.minStepDimension() / 2.0;
        const size_t upsampled_band_num_points = GetRRTBandMaxPoints(*ph_);

        initial_band_ = std::make_shared<RubberBand>(
                    nh_,
                    ph_,
                    vis_,
                    sdf_,
                    work_space_grid_,
                    neighbour_fn,
                    world_state,
                    resampled_band_max_pointwise_dist,
                    upsampled_band_num_points,
                    max_band_length);
    }

    void TransitionTesting::initializeRRTParams()
    {
        // "World" params used by planning
        world_params_ = std::make_shared<BandRRT::WorldParams>(BandRRT::WorldParams
        {
            robot_,
            false,
            sdf_,
            work_space_grid_,
            transition_estimator_,
            generator_
        });

        // Algorithm parameters
        const auto use_cbirrt_style_projection      = GetUseCBiRRTStyleProjection(*ph_);
        const auto forward_tree_extend_iterations   = GetRRTForwardTreeExtendIterations(*ph_);
        const auto backward_tree_extend_iterations  = GetRRTBackwardTreeExtendIterations(*ph_);
        const auto kd_tree_grow_threshold           = GetRRTKdTreeGrowThreshold(*ph_);
        const auto use_brute_force_nn               = GetRRTUseBruteForceNN(*ph_);
        const auto goal_bias                        = GetRRTGoalBias(*ph_);
        const auto best_near_radius                 = GetRRTBestNearRadius(*ph_);
        const auto feasibility_dist_scale_factor    = GetRRTFeasibilityDistanceScaleFactor(*ph_);
        assert(!use_cbirrt_style_projection && "CBiRRT style projection is no longer supported");
        planning_params_ =
        {
            forward_tree_extend_iterations,
            backward_tree_extend_iterations,
            use_brute_force_nn,
            kd_tree_grow_threshold,
            best_near_radius * best_near_radius,
            goal_bias,
            feasibility_dist_scale_factor
        };

        // Smoothing parameters
        const auto max_shortcut_index_distance      = GetRRTMaxShortcutIndexDistance(*ph_);
        const auto max_smoothing_iterations         = GetRRTMaxSmoothingIterations(*ph_);
        const auto max_failed_smoothing_iterations  = GetRRTMaxFailedSmoothingIterations(*ph_);
        const auto smoothing_band_dist_threshold    = GetRRTSmoothingBandDistThreshold(*ph_);
        smoothing_params_ =
        {
            max_shortcut_index_distance,
            max_smoothing_iterations,
            max_failed_smoothing_iterations,
            smoothing_band_dist_threshold
        };

        // Task defined parameters
        const auto task_aligned_frame = robot_->getWorldToTaskFrameTf();
        const auto task_frame_lower_limits = Vector3d(
                    GetRRTPlanningXMinBulletFrame(*ph_),
                    GetRRTPlanningYMinBulletFrame(*ph_),
                    GetRRTPlanningZMinBulletFrame(*ph_));
        const auto task_frame_upper_limits = Vector3d(
                    GetRRTPlanningXMaxBulletFrame(*ph_),
                    GetRRTPlanningYMaxBulletFrame(*ph_),
                    GetRRTPlanningZMaxBulletFrame(*ph_));
        const auto max_gripper_step_size                = work_space_grid_.minStepDimension();
        const auto max_robot_step_size                  = GetRRTMaxRobotDOFStepSize(*ph_);
        const auto min_robot_step_size                  = GetRRTMinRobotDOFStepSize(*ph_);
        const auto max_gripper_rotation                 = GetRRTMaxGripperRotation(*ph_); // only matters for real robot
        const auto goal_reached_radius                  = work_space_grid_.minStepDimension();
        const auto min_gripper_distance_to_obstacles    = GetRRTMinGripperDistanceToObstacles(*ph_); // only matters for simulation
        const auto band_distance2_scaling_factor        = GetRRTBandDistance2ScalingFactor(*ph_);
        const auto upsampled_band_num_points            = GetRRTBandMaxPoints(*ph_);
        task_params_ =
        {
            task_aligned_frame,
            task_frame_lower_limits,
            task_frame_upper_limits,
            max_gripper_step_size,
            max_robot_step_size,
            min_robot_step_size,
            max_gripper_rotation,
            goal_reached_radius,
            min_gripper_distance_to_obstacles,
            band_distance2_scaling_factor,
            upsampled_band_num_points
        };
    }

    void TransitionTesting::clampGripperDeltas(Ref<Vector3d> a_delta, Ref<Vector3d> b_delta) const
    {
        const double distance = std::sqrt(a_delta.squaredNorm() + b_delta.squaredNorm());
        if (distance > task_params_.max_gripper_step_size_)
        {
            a_delta *= (task_params_.max_gripper_step_size_ / distance);
            b_delta *= (task_params_.max_gripper_step_size_ / distance);
        }
    }

    std::vector<std::string> TransitionTesting::getDataFileList()
    {
        std::vector<std::string> files;
        const boost::filesystem::path p(data_folder_ + "/cannonical_straight_test");
        const boost::filesystem::recursive_directory_iterator start(p);
        const boost::filesystem::recursive_directory_iterator end;
        for (auto itr = start; itr != end; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->status()))
            {
                const auto filename = itr->path().string();
                // Only warn about file types that are not expected
                if (filename.find("compressed") == std::string::npos)
                {
                    ROS_WARN_STREAM("Ignoring file: " << filename);
                }
                if (filename.find("test_results.compressed") != std::string::npos)
                {
                    files.push_back(filename);
                }
            }
        }
        std::sort(files.begin(), files.end());
        ROS_INFO_STREAM("Found " << files.size() << " possible data files in " << data_folder_);
        return files;
    }

    void TransitionTesting::runTests(const bool generate_test_data,
                                     const bool generate_last_step_transition_approximations,
                                     const bool generate_meaningful_mistake_examples)
    {
        if (generate_test_data)
        {
            Stopwatch stopwatch;
            generateTestData();
            std::cout << "Data generation time taken: " << stopwatch(READ) << std::endl;
        }

        if (generate_last_step_transition_approximations)
        {
            Stopwatch stopwatch;
            generateLastStepTransitionApproximations();
            std::cout << "Last step transition approximations time taken: " << stopwatch(READ) << std::endl;
        }

        if (generate_meaningful_mistake_examples)
        {
            Stopwatch stopwatch;
            generateMeaningfulMistakeExamples();
            std::cout << "Finding meaningful mistake examples time taken: " << stopwatch(READ) << std::endl;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Data Generation
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    // Note: this includes "no perturbation" so that it can be combined "cartesian-product" style
    static VectorVector3d Vec3dPerturbations(const double max_magnitude, const int num_divisions)
    {
        VectorVector3d perturbations;
        perturbations.reserve((size_t)(std::pow(2 * num_divisions + 1, 3)));
        for (int x_idx = -num_divisions; x_idx <= num_divisions; ++x_idx)
        {
            const double x_delta = max_magnitude * x_idx / num_divisions;
            for (int y_idx = -num_divisions; y_idx <= num_divisions; ++y_idx)
            {
                const double y_delta = max_magnitude * y_idx / num_divisions;
                for (int z_idx = -num_divisions; z_idx <= num_divisions; ++z_idx)
                {
                    const double z_delta = max_magnitude * z_idx / num_divisions;
                    perturbations.push_back(Vector3d(x_delta, y_delta, z_delta));
                }
            }
        }
        return perturbations;
    }

    void TransitionTesting::generateTestData()
    {
        const auto num_threads = GetNumOMPThreads();
        std::vector<dmm::TransitionTest> tests;
        std::vector<std::string> filenames;
        tests.reserve(num_threads);
        filenames.reserve(num_threads);

        // Ignore the feedback as the action sever saves the results to file anyway
        const auto feedback_callback = [&] (const size_t test_id, const dmm::TransitionTestResult& result)
        {
            (void)test_id;
            (void)result;
        };

        //// Generate the canonical example ////////////////////////////////////
        {
            const std::string folder(data_folder_ + "/cannonical_straight_test");
            const std::string test_id("/unmodified");
            const std::string test_results_filename = folder + test_id + "__test_results.compressed";
            const std::string path_to_start_filename = folder + test_id + "__path_to_start.compressed";
            arc_utilities::CreateDirectory(folder);

            if (!boost::filesystem::is_regular_file(test_results_filename))
            {
                Isometry3d gripper_a_ending_pose_ = Translation3d(gripper_a_action_vector_) * gripper_a_starting_pose_;
                Isometry3d gripper_b_ending_pose_ = Translation3d(gripper_b_action_vector_) * gripper_b_starting_pose_;

                // Generate a path and convert the test to a ROS format (if needed)
                const RRTPath path_to_start_of_test = loadOrGeneratePath(
                            path_to_start_filename,
                            {gripper_a_starting_pose_, gripper_b_starting_pose_});

                const auto canonical_test = robot_->toRosTransitionTest(
                            initial_world_state_.rope_node_transforms_,
                            initial_world_state_.all_grippers_single_pose_,
                            RRTPathToGrippersPoseTrajectory(path_to_start_of_test),
                            {gripper_a_ending_pose_, gripper_b_ending_pose_});

                // Add the test to the list waiting to be executed
                tests.push_back(canonical_test);
                filenames.push_back(test_results_filename);
            }
        }

        //// Generate versions with perturbed gripper start positions //////////
        {
            const auto max_magnitude = ROSHelpers::GetParamRequired<double>(*ph_, "perturbations/gripper_positions/max_magnitude", __func__).GetImmutable();
            const auto num_divisions = ROSHelpers::GetParamRequired<int>(*ph_, "perturbations/gripper_positions/num_divisions", __func__).GetImmutable();
            const auto perturbations = Vec3dPerturbations(max_magnitude, num_divisions);
            std::cout << "Num position perturbations: " << perturbations.size() * perturbations.size()<< std::endl;
            #pragma omp parallel for
            for (size_t a_idx = 0; a_idx < perturbations.size(); ++a_idx)
            {
                const Isometry3d gripper_a_starting_pose = Translation3d(perturbations[a_idx]) * gripper_a_starting_pose_;
                const Isometry3d gripper_a_ending_pose = Translation3d(gripper_a_action_vector_) * gripper_a_starting_pose;

                const std::string folder(data_folder_ +
                                         "/cannonical_straight_test"
                                         "/perturbed_gripper_start_positions"
                                         "/gripper_a_" + ToString(perturbations[a_idx]));
                arc_utilities::CreateDirectory(folder);

                for (size_t b_idx = 0; b_idx < perturbations.size(); ++b_idx)
                {
                    try
                    {
                        const Isometry3d gripper_b_starting_pose = Translation3d(perturbations[b_idx]) * gripper_b_starting_pose_;
                        const Isometry3d gripper_b_ending_pose = Translation3d(gripper_b_action_vector_) * gripper_b_starting_pose;

                        const std::string test_id("/gripper_b_" + ToString(perturbations[b_idx]));
                        const std::string test_results_filename = folder + test_id + "__test_results.compressed";
                        const std::string path_to_start_filename = folder + test_id + "__path_to_start.compressed";

                        if (!boost::filesystem::is_regular_file(test_results_filename))
                        {
                            // Generate a path and convert the test to a ROS format (if needed)
                            const RRTPath path_to_start_of_test = loadOrGeneratePath(
                                        path_to_start_filename,
                                        {gripper_a_starting_pose, gripper_b_starting_pose});

                            const auto test = robot_->toRosTransitionTest(
                                        initial_world_state_.rope_node_transforms_,
                                        initial_world_state_.all_grippers_single_pose_,
                                        RRTPathToGrippersPoseTrajectory(path_to_start_of_test),
                                        {gripper_a_ending_pose, gripper_b_ending_pose});

                            #pragma omp critical
                            {
                                // Add the test to the list waiting to be executed
                                tests.push_back(test);
                                filenames.push_back(test_results_filename);

                                // Execute the tests if tehre are enough to run
                                if (tests.size() == num_threads)
                                {
                                    robot_->generateTransitionData(tests, filenames, feedback_callback, false);
                                    tests.clear();
                                    filenames.clear();
                                }
                            }
                        }
                    }
                    catch (const std::runtime_error& ex)
                    {
                        ROS_ERROR_STREAM_NAMED("data_generation", "Unable to plan with perturbation"
                                               << " a: " << perturbations[a_idx].transpose()
                                               << " b: " << perturbations[b_idx].transpose()
                                               << " Message: " << ex.what());
                    }
                }
            }
        }

        //// Generate versions with perturbed action vectors ///////////////////
        {
            const auto max_magnitude = ROSHelpers::GetParamRequired<double>(*ph_, "perturbations/action_vectors/max_magnitude", __func__).GetImmutable();
            const auto num_divisions = ROSHelpers::GetParamRequired<int>(*ph_, "perturbations/action_vectors/num_divisions", __func__).GetImmutable();
            const auto perturbations = Vec3dPerturbations(max_magnitude, num_divisions);
            std::cout << "Num action perturbations: " << perturbations.size() * perturbations.size()<< std::endl;
            #pragma omp parallel for
            for (size_t a_idx = 0; a_idx < perturbations.size(); ++a_idx)
            {
                const std::string folder(data_folder_ +
                                         "/cannonical_straight_test"
                                         "/perturbed_gripper_action_vectors"
                                         "/gripper_a_" + ToString(perturbations[a_idx]));
                arc_utilities::CreateDirectory(folder);

                const Vector3d gripper_a_action_vector = gripper_a_action_vector_ + perturbations[a_idx];
                for (size_t b_idx = 0; b_idx < perturbations.size(); ++b_idx)
                {
                    try
                    {
                        const Vector3d gripper_b_action_vector = gripper_b_action_vector_ + perturbations[b_idx];
                        Vector3d gripper_a_action_vector_normalized = gripper_a_action_vector;
                        Vector3d gripper_b_action_vector_normalized = gripper_b_action_vector;
                        clampGripperDeltas(gripper_a_action_vector_normalized, gripper_b_action_vector_normalized);

                        const Isometry3d gripper_a_ending_pose = Translation3d(gripper_a_action_vector_normalized) * gripper_a_starting_pose_;
                        const Isometry3d gripper_b_ending_pose = Translation3d(gripper_b_action_vector_normalized) * gripper_b_starting_pose_;

                        const std::string test_id("/gripper_b_" + ToString(perturbations[b_idx]));
                        const std::string test_results_filename = folder + test_id + "__test_results.compressed";
                        const std::string path_to_start_filename = folder + test_id + "__path_to_start.compressed";

                        if (!boost::filesystem::is_regular_file(test_results_filename))
                        {
                            // Generate a path and convert the test to a ROS format (if needed)
                            const RRTPath path_to_start_of_test = loadOrGeneratePath(
                                        path_to_start_filename,
                                        {gripper_a_starting_pose_, gripper_b_starting_pose_});

                            const auto test = robot_->toRosTransitionTest(
                                        initial_world_state_.rope_node_transforms_,
                                        initial_world_state_.all_grippers_single_pose_,
                                        RRTPathToGrippersPoseTrajectory(path_to_start_of_test),
                                        {gripper_a_ending_pose, gripper_b_ending_pose});

                            #pragma omp critical
                            {
                                // Add the test to the list waiting to be executed
                                tests.push_back(test);
                                filenames.push_back(test_results_filename);

                                // Execute the tests if tehre are enough to run
                                if (tests.size() == num_threads)
                                {
                                    robot_->generateTransitionData(tests, filenames, feedback_callback, false);
                                    tests.clear();
                                    filenames.clear();
                                }
                            }

                        }
                    }
                    catch (const std::runtime_error& ex)
                    {
                        ROS_ERROR_STREAM_NAMED("data_generation", "Unable to plan with perturbation"
                                               << " a: " << perturbations[a_idx].transpose()
                                               << " b: " << perturbations[b_idx].transpose()
                                               << " Message: " << ex.what());
                    }
                }
            }
        }

        // Run an tests left over
        if (tests.size() != 0)
        {
            robot_->generateTransitionData(tests, filenames, feedback_callback, false);
            tests.clear();
            filenames.clear();
        }
    }

    RRTPath TransitionTesting::loadOrGeneratePath(
            const std::string& filename,
            const AllGrippersSinglePose& gripper_target_poses)
    {
        if (boost::filesystem::is_regular_file(filename))
        {
            const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(filename);
            const auto deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
            {
                return RRTNode::Deserialize(buf, cur, *initial_band_);
            };
            const auto path_deserialized = DeserializeVector<RRTNode, Eigen::aligned_allocator<RRTNode>>(buffer, 0, deserializer);
            return path_deserialized.first;
        }
        else
        {
            const auto path = generateTestPath(gripper_target_poses);

            // Save the generated path to file
            std::vector<uint8_t> buffer;
            SerializeVector<RRTNode>(path, buffer, &RRTNode::Serialize);
            ZlibHelpers::CompressAndWriteToFile(buffer, filename);

            return path;
        }
    }

    RRTPath TransitionTesting::generateTestPath(
            const AllGrippersSinglePose& gripper_target_poses)
    {
        // Pass in all the config values that the RRT needs; for example goal bias, step size, etc.
        auto band_rrt = BandRRT(nh_,
                                ph_,
                                *world_params_,
                                planning_params_,
                                smoothing_params_,
                                task_params_,
                                vis_,
                                false);

        const auto gripper_config = RRTGrippersRepresentation(
                    initial_world_state_.all_grippers_single_pose_[0],
                    initial_world_state_.all_grippers_single_pose_[1]);

        RRTRobotRepresentation robot_config(6);
        robot_config.head<3>() = gripper_config.first.translation();
        robot_config.tail<3>() = gripper_config.second.translation();

        const auto rubber_band = RubberBand::BandFromWorldState(
                    initial_world_state_, *initial_band_);

        const RRTNode start_config(
                    gripper_config,
                    robot_config,
                    rubber_band);

        const std::chrono::duration<double> time_limit(GetRRTTimeout(*ph_));

        const auto policy = band_rrt.plan(start_config,
                                          {gripper_target_poses[0], gripper_target_poses[1]},
                                          time_limit);
        if (policy.size() == 0)
        {
            throw_arc_exception(std::runtime_error, "No path returned by RRT.");
        }
        else if (policy.size() > 1)
        {
            throw_arc_exception(std::runtime_error, "Multiple paths returned by RRT. Weird.");
        }
        return policy[0].first;
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Generate Last Step Approximations
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    void TransitionTesting::generateLastStepTransitionApproximations()
    {
        // Setup the transition data source to generate transition approximations from
        dmm::TransitionTestingVisualizationRequest req;
        req.data = "cannonical_straight_test/unmodified__test_results.compressed";
        dmm::TransitionTestingVisualizationResponse res;
        setTransitionAdaptationSourceCallback(req, res);
        assert(source_valid_);

        enum
        {
            FILENAME,
            ERROR_STRING,
            TEMPLATE_MISALIGNMENT_EUCLIDEAN,
            DEFAULT_VS_ADAPTATION_FOH,
            DEFAULT_VS_ADAPTATION_EUCLIDEAN,
            BAND_TIGHTEN_DELTA,
            SOURCE_NUM_FOH_CHANGES,
            RESULT_NUM_FOH_CHANGES,
            TRUE_VS_DEFAULT_FOH,
            TRUE_VS_DEFAULT_EUCLIDEAN,
            TRUE_VS_ADAPTATION_FOH,
            TRUE_VS_ADAPTATION_EUCLIDEAN,
            PLANNED_VS_ACTUAL_START_FOH,
            PLANNED_VS_ACTUAL_START_EUCLIDEAN,
            DUMMY_ITEM
        };
        Log::Log logger(data_folder_ + "/cannonical_straight_test/generate_last_step_transition_approximations.csv", false);
        LOG(logger, "FILENAME, "
                    "ERROR_STRING, "
                    "TEMPLATE_MISALIGNMENT_EUCLIDEAN, "
                    "DEFAULT_VS_ADAPTATION_FOH, "
                    "DEFAULT_VS_ADAPTATION_EUCLIDEAN, "
                    "BAND_TIGHTEN_DELTA, "
                    "SOURCE_NUM_FOH_CHANGES, "
                    "RESULT_NUM_FOH_CHANGES, "
                    "TRUE_VS_DEFAULT_FOH, "
                    "TRUE_VS_DEFAULT_EUCLIDEAN, "
                    "TRUE_VS_ADAPTATION_FOH, "
                    "TRUE_VS_ADAPTATION_EUCLIDEAN, "
                    "PLANNED_VS_ACTUAL_START_FOH, "
                    "PLANNED_VS_ACTUAL_START_EUCLIDEAN");
        const auto files = getDataFileList();
        #pragma omp parallel for
        for (size_t idx = 0; idx < files.size(); ++idx)
        {
            const auto& test_result_file = files[idx];
            std::vector<std::string> dists_etc(DUMMY_ITEM, "");
            dists_etc[FILENAME] = test_result_file.substr(data_folder_.length() + 1);
            try
            {
                // Load the path that generated the test
                const RRTPath path_to_start = [&]
                {
                    const auto path_to_start_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__path_to_start.compressed";
                    const auto decompressed_path = ZlibHelpers::LoadFromFileAndDecompress(path_to_start_file);
                    const auto node_deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
                    {
                        return RRTNode::Deserialize(buf, cur, *initial_band_);
                    };
                    return DeserializeVector<RRTNode, aligned_allocator<smmap::RRTNode>>(decompressed_path, 0, node_deserializer).first;
                }();

                // Load the test record
                const dmm::TransitionTestResult test_result = [&]
                {
                    const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(test_result_file);
                    return arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(buffer, 0).first.test_result;
                }();

                // Load the resulting transition, if needed generate it first
                const TransitionEstimation::StateTransition test_transition = [&]
                {
                    const auto test_transition_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__test_transition.compressed";
                    if (!boost::filesystem::is_regular_file(test_transition_file))
                    {
                        const auto transition = ToStateTransition(test_result, path_to_start);

                        std::vector<uint8_t> output_buffer;
                        transition.serialize(output_buffer);
                        ZlibHelpers::CompressAndWriteToFile(output_buffer, test_transition_file);
                        return transition;
                    }
                    else
                    {
                        const auto test_transition_buffer = ZlibHelpers::LoadFromFileAndDecompress(test_transition_file);
                        return TransitionEstimation::StateTransition::Deserialize(test_transition_buffer, 0, *initial_band_).first;
                    }
                }();

                // Load the adaptation record, if needed generate it first
                const TransitionEstimation::TransitionAdaptationResult adaptation_record = [&]
                {
                    const auto adaptation_record_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__adaptation_record.compressed";
                    if (!boost::filesystem::is_regular_file(adaptation_record_file))
                    {
                        const auto ar = transition_estimator_->generateTransition(
                                    source_transition_,
                                    *test_transition.starting_state_.planned_rubber_band_,
                                    test_transition.ending_gripper_positions_);

                        std::vector<uint8_t> output_buffer;
                        ar.serialize(output_buffer);
                        ZlibHelpers::CompressAndWriteToFile(output_buffer, adaptation_record_file);
                        return ar;
                    }
                    else
                    {
                        const auto adaptation_record_buffer = ZlibHelpers::LoadFromFileAndDecompress(adaptation_record_file);
                        return TransitionEstimation::TransitionAdaptationResult::Deserialize(adaptation_record_buffer, 0, *initial_band_).first;
                    }
                }();

                const auto test_band_start = RubberBand::BandFromWorldState(ConvertToEigenFeedback(test_result.start_after_following_path), *initial_band_);
                if (test_band_start->isOverstretched())
                {
                    throw_arc_exception(std::runtime_error, "Starting configuration of test band is overstretched");
                }
                const auto test_band_end = RubberBand::BandFromWorldState(ConvertToEigenFeedback(test_result.microsteps_last_action.back()), *initial_band_);

                dists_etc[TEMPLATE_MISALIGNMENT_EUCLIDEAN] = std::to_string(adaptation_record.template_misalignment_dist_);
                dists_etc[DEFAULT_VS_ADAPTATION_FOH] = std::to_string(adaptation_record.default_band_foh_result_);
                dists_etc[DEFAULT_VS_ADAPTATION_EUCLIDEAN] = std::to_string(adaptation_record.default_band_dist_);

                dists_etc[BAND_TIGHTEN_DELTA] = std::to_string(adaptation_record.band_tighten_delta_);
                dists_etc[SOURCE_NUM_FOH_CHANGES] = std::to_string(source_num_foh_changes_);
                dists_etc[RESULT_NUM_FOH_CHANGES] = std::to_string(adaptation_record.num_foh_changes_);

                dists_etc[TRUE_VS_DEFAULT_FOH] = std::to_string(transition_estimator_->checkFirstOrderHomotopy(*adaptation_record.default_next_band_, *test_band_end));
                dists_etc[TRUE_VS_DEFAULT_EUCLIDEAN] = std::to_string(adaptation_record.default_next_band_->distance(*test_band_end));

                dists_etc[TRUE_VS_ADAPTATION_FOH] = std::to_string(transition_estimator_->checkFirstOrderHomotopy(*adaptation_record.result_, *test_band_end));
                dists_etc[TRUE_VS_ADAPTATION_EUCLIDEAN] = std::to_string(adaptation_record.result_->distance(*test_band_end));

                dists_etc[PLANNED_VS_ACTUAL_START_FOH] = std::to_string(transition_estimator_->checkFirstOrderHomotopy(*test_transition.starting_state_.planned_rubber_band_, *test_transition.starting_state_.rubber_band_));
                dists_etc[PLANNED_VS_ACTUAL_START_EUCLIDEAN] = std::to_string(test_transition.starting_state_.planned_rubber_band_->distance(*test_transition.starting_state_.rubber_band_));
            }
            catch (const std::exception& ex)
            {
                ROS_ERROR_STREAM("Error parsing idx: " << idx << " file: " << test_result_file << ": " << ex.what());
                dists_etc[ERROR_STRING] = ex.what();
            }

            LOG(logger, PrettyPrint::PrettyPrint(dists_etc, false, ", "));
        }
    }

    bool TransitionTesting::setTransitionAdaptationSourceCallback(
            dmm::TransitionTestingVisualizationRequest& req,
            dmm::TransitionTestingVisualizationResponse& res)
    {
        (void)res;

        source_valid_ = false;

        const std::string test_result_file = data_folder_ + "/" + req.data;
        const auto path_to_start_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__path_to_start.compressed";
        const auto decompressed_path = ZlibHelpers::LoadFromFileAndDecompress(path_to_start_file);
        const auto node_deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
        {
            return RRTNode::Deserialize(buf, cur, *initial_band_);
        };
        const auto path_to_start = DeserializeVector<RRTNode, aligned_allocator<smmap::RRTNode>>(decompressed_path, 0, node_deserializer).first;
        const auto decompressed_test_result = ZlibHelpers::LoadFromFileAndDecompress(test_result_file);
        const auto test_result = arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(decompressed_test_result, 0).first.test_result;

        source_file_ = req.data;
        source_transition_ = ToStateTransition(test_result, path_to_start);
        source_band_surface_ = RubberBand::AggregateBandPoints(source_transition_.microstep_band_history_);

        std::vector<bool> foh_values;
        for (size_t idx = 0; idx < source_transition_.microstep_band_history_.size() - 1; ++idx)
        {
            RubberBand::Ptr b1 = source_transition_.microstep_band_history_[idx];
            RubberBand::Ptr b2 = source_transition_.microstep_band_history_[idx + 1];
            foh_values.push_back(transition_estimator_->checkFirstOrderHomotopy(*b1, *b2));
        }
        source_num_foh_changes_ = 0;
        for (size_t idx = 0; idx < foh_values.size() - 1; ++idx)
        {
            if (foh_values[idx] != foh_values[idx + 1])
            {
                ++source_num_foh_changes_;
            }
        }

        // Ensure all bands have been upsampled and resampled to avoid race conditions in multithreading later
        source_transition_.starting_state_.rubber_band_->upsampleBand();
        source_transition_.starting_state_.rubber_band_->resampleBand();
        source_transition_.starting_state_.planned_rubber_band_->upsampleBand();
        source_transition_.starting_state_.planned_rubber_band_->resampleBand();
        source_transition_.ending_state_.rubber_band_->upsampleBand();
        source_transition_.ending_state_.rubber_band_->resampleBand();
        source_transition_.ending_state_.planned_rubber_band_->upsampleBand();
        source_transition_.ending_state_.planned_rubber_band_->resampleBand();

        source_valid_ = true;
        ROS_INFO_STREAM("Source transition set to " << req.data);
        return true;
    }

    bool TransitionTesting::addTransitionAdaptationVisualizationCallback(
            deformable_manipulation_msgs::TransitionTestingVisualizationRequest& req,
            deformable_manipulation_msgs::TransitionTestingVisualizationResponse& res)
    {
        if (!source_valid_)
        {
            ROS_WARN_NAMED("data_visualization", "Visualization requested, but transition source is invalid");
            res.response = "Visualization requested, but transition source is invalid";
            return false;
        }

        const std::string test_result_file = data_folder_ + "/" + req.data;
        const auto path_to_start_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__path_to_start.compressed";
        const auto decompressed_path = ZlibHelpers::LoadFromFileAndDecompress(path_to_start_file);
        const auto node_deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
        {
            return RRTNode::Deserialize(buf, cur, *initial_band_);
        };
        const auto path_to_start = DeserializeVector<RRTNode, aligned_allocator<smmap::RRTNode>>(decompressed_path, 0, node_deserializer).first;
        const auto decompressed_test_result = ZlibHelpers::LoadFromFileAndDecompress(test_result_file);
        const auto test_result = arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(decompressed_test_result, 0).first.test_result;

        const auto test_transition = ToStateTransition(test_result, path_to_start);
        const auto adaptation_record = transition_estimator_->generateTransition(
                    source_transition_,
                    *test_transition.starting_state_.planned_rubber_band_,
                    test_transition.ending_gripper_positions_);

        const auto sim_record = TransitionSimulationRecord
        {
            source_transition_,
            RubberBand::AggregateBandPoints(source_transition_.microstep_band_history_),
            test_transition,
            RubberBand::AggregateBandPoints(test_transition.microstep_band_history_),
            adaptation_record
        };
        res.response = std::to_string(next_vis_prefix_);
        visid_to_markers_[res.response] = sim_record.visualize(std::to_string(next_vis_prefix_) + "__", vis_);
        ++next_vis_prefix_;

        const auto test_band_end = RubberBand::BandFromWorldState(ConvertToEigenFeedback(test_result.microsteps_last_action.back()), *initial_band_);
        ROS_INFO_STREAM("Added vis id: " << res.response << " for file " << req.data << std::endl
                        << "Template alignment dist:      " << adaptation_record.template_misalignment_dist_ << std::endl
                        << "Default band FOH:             " << adaptation_record.default_band_foh_result_ << std::endl
                        << "Default band dist:            " << adaptation_record.default_band_dist_ << std::endl
                        << "Band tighten delta:           " << adaptation_record.band_tighten_delta_ << std::endl
                        << "Source FOH changes:           " << source_num_foh_changes_ << std::endl
                        << "Adaptation FOH changes:       " << adaptation_record.num_foh_changes_ << std::endl
                        << "True vs default FOH:          " << transition_estimator_->checkFirstOrderHomotopy(*adaptation_record.default_next_band_, *test_band_end) << std::endl
                        << "True vs default dist:         " << adaptation_record.default_next_band_->distance(*test_band_end) << std::endl
                        << "True vs adaptation FOH:       " << transition_estimator_->checkFirstOrderHomotopy(*adaptation_record.result_, *test_band_end) << std::endl
                        << "True vs adaptation dist:      " << adaptation_record.result_->distance(*test_band_end) << std::endl
                        << "Planned vs actual start FOH:  " << transition_estimator_->checkFirstOrderHomotopy(*test_transition.starting_state_.planned_rubber_band_, *test_transition.starting_state_.rubber_band_) << std::endl
                        << "Planned vs actual start dist: " << test_transition.starting_state_.planned_rubber_band_->distance(*test_transition.starting_state_.rubber_band_) << std::endl);
        return true;
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Find Meaningful Mistakes
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    void TransitionTesting::generateMeaningfulMistakeExamples()
    {
        enum
        {
            FILENAME,
            ERROR_STRING,
            PLANNED_VS_EXECUTED_START_EUCLIDEAN,
            PLANNED_VS_EXECUTED_END_EUCLIDEAN,
            START_VS_END_EUCLIDEN_PLANNED,
            START_VS_END_EUCLIDEN_EXECUTED,
            FOH_RESULTS,
            NUM_FOH_CHANGES,
            DUMMY_ITEM
        };
        Log::Log logger(data_folder_ + "/cannonical_straight_test/generate_meaningful_mistake_examples.csv", false);
        LOG(logger, "FILENAME, "
                    "ERROR_STRING, "
                    "PLANNED_VS_EXECUTED_START_EUCLIDEAN, "
                    "PLANNED_VS_EXECUTED_END_EUCLIDEAN, "
                    "START_VS_END_EUCLIDEN_PLANNED, "
                    "START_VS_END_EUCLIDEN_EXECUTED, "
                    "FOH_RESULTS, "
                    "NUM_FOH_CHANGES");

        const auto files = getDataFileList();
        #pragma omp parallel for
        for (size_t idx = 0; idx < files.size(); ++idx)
        {
            const auto& test_result_file = files[idx];
            std::vector<std::string> dists_etc(DUMMY_ITEM, "");
            dists_etc[FILENAME] = test_result_file.substr(data_folder_.length() + 1);
            try
            {
                // Load the transition example if possible, otherwise generate it
                const TransitionEstimation::StateTransition transition = [&]
                {
                    const std::string example_mistake_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__example_mistake.compressed";

                    if (!boost::filesystem::is_regular_file(example_mistake_file))
                    {
                        // Load the path that generated the test
                        const RRTPath path_to_start = [&]
                        {
                            const auto path_to_start_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__path_to_start.compressed";
                            const auto decompressed_path = ZlibHelpers::LoadFromFileAndDecompress(path_to_start_file);
                            const auto node_deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
                            {
                                return RRTNode::Deserialize(buf, cur, *initial_band_);
                            };
                            return DeserializeVector<RRTNode, aligned_allocator<smmap::RRTNode>>(decompressed_path, 0, node_deserializer).first;
                        }();

                        // Load the test record
                        const dmm::TransitionTestResult test_result = [&]
                        {
                            const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(test_result_file);
                            return arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(buffer, 0).first.test_result;
                        }();

                        const auto trajectory = toTrajectory(test_result, path_to_start);
                        const auto example = transition_estimator_->findMostRecentBadTransition(trajectory).Get();

                        std::vector<uint8_t> buffer;
                        example.serialize(buffer);
                        ZlibHelpers::CompressAndWriteToFile(buffer, example_mistake_file);
                        return example;
                    }
                    else
                    {
                        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(example_mistake_file);
                        return TransitionEstimation::StateTransition::Deserialize(buffer, 0, *initial_band_).first;
                    }
                }();

                std::vector<int> foh_values;
                for (size_t step_idx = 0; step_idx < transition.microstep_band_history_.size() - 1; ++step_idx)
                {
                    RubberBand::Ptr b1 = transition.microstep_band_history_[step_idx];
                    RubberBand::Ptr b2 = transition.microstep_band_history_[step_idx + 1];
                    foh_values.push_back(transition_estimator_->checkFirstOrderHomotopy(*b1, *b2));
                }
                int num_foh_changes = 0;
                for (size_t step_idx = 0; step_idx < foh_values.size() - 1; ++step_idx)
                {
                    if (foh_values[step_idx] != foh_values[step_idx + 1])
                    {
                        ++num_foh_changes;
                    }
                }

                dists_etc[PLANNED_VS_EXECUTED_START_EUCLIDEAN] = std::to_string(transition.starting_state_.planned_rubber_band_->distance(*transition.starting_state_.rubber_band_));
                dists_etc[PLANNED_VS_EXECUTED_END_EUCLIDEAN] = std::to_string(transition.ending_state_.planned_rubber_band_->distance(*transition.ending_state_.rubber_band_));

                dists_etc[START_VS_END_EUCLIDEN_PLANNED] = std::to_string(transition.starting_state_.planned_rubber_band_->distance(*transition.ending_state_.planned_rubber_band_));
                dists_etc[START_VS_END_EUCLIDEN_EXECUTED] = std::to_string(transition.starting_state_.rubber_band_->distance(*transition.ending_state_.rubber_band_));

                dists_etc[FOH_RESULTS] = PrettyPrint::PrettyPrint(foh_values, false, "");
                dists_etc[NUM_FOH_CHANGES] = std::to_string(num_foh_changes);
            }
            catch (const std::exception& ex)
            {
                ROS_ERROR_STREAM("Error parsing idx: " << idx << " file: " << test_result_file << ": " << ex.what());
                dists_etc[ERROR_STRING] = ex.what();
            }

            LOG(logger, PrettyPrint::PrettyPrint(dists_etc, false, ", "));
        }
    }

    bool TransitionTesting::addMistakeExampleVisualizationCallback(
            deformable_manipulation_msgs::TransitionTestingVisualizationRequest& req,
            deformable_manipulation_msgs::TransitionTestingVisualizationResponse& res)
    {
        const std::string test_result_file = data_folder_ + "/" + req.data;
        const auto path_to_start_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__path_to_start.compressed";
        const auto example_mistake_file = test_result_file.substr(0, test_result_file.find("__test_results.compressed")) + "__example_mistake.compressed";

        // Load the path that generated the test
        const RRTPath path_to_start = [&]
        {
            const auto decompressed_path = ZlibHelpers::LoadFromFileAndDecompress(path_to_start_file);
            const auto node_deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
            {
                return RRTNode::Deserialize(buf, cur, *initial_band_);
            };
            return DeserializeVector<RRTNode, aligned_allocator<smmap::RRTNode>>(decompressed_path, 0, node_deserializer).first;
        }();

        // Load the test record
        const dmm::TransitionTestResult test_result = [&]
        {
            const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(test_result_file);
            return arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(buffer, 0).first.test_result;
        }();

        // Load the transition example if possible
        const TransitionEstimation::StateTransition transition = [&]
        {
            if (!boost::filesystem::is_regular_file(example_mistake_file))
            {
                const auto trajectory = toTrajectory(test_result, path_to_start);
                const auto example = transition_estimator_->findMostRecentBadTransition(trajectory).Get();

                std::vector<uint8_t> buffer;
                example.serialize(buffer);
                ZlibHelpers::CompressAndWriteToFile(buffer, example_mistake_file);
                return example;
            }
            else
            {
                const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(example_mistake_file);
                return TransitionEstimation::StateTransition::Deserialize(buffer, 0, *initial_band_).first;
            }
        }();

        // Determine the FOH and distance values along the band surface
        Matrix2Xd dist_and_foh_values(2, transition.microstep_band_history_.size() - 1);
        for (size_t step_idx = 0; step_idx < transition.microstep_band_history_.size() - 1; ++step_idx)
        {
            RubberBand::Ptr b1 = transition.microstep_band_history_[step_idx];
            RubberBand::Ptr b2 = transition.microstep_band_history_[step_idx + 1];
            dist_and_foh_values(0, step_idx) = b1->distance(*b2);
            dist_and_foh_values(1, step_idx) = transition_estimator_->checkFirstOrderHomotopy(*b1, *b2);
        }
        int num_foh_changes = 0;
        for (ssize_t step_idx = 0; step_idx < dist_and_foh_values.cols() - 1; ++step_idx)
        {
            if (dist_and_foh_values(1, step_idx) != dist_and_foh_values(1, step_idx + 1))
            {
                ++num_foh_changes;
            }
        }

        res.response = std::to_string(-1);

        ROS_INFO_STREAM("Added vis id: " << res.response << " for file " << req.data << std::endl
                        << "Planned vs executed start dist:     " << transition.starting_state_.planned_rubber_band_->distance(*transition.starting_state_.rubber_band_) << std::endl
                        << "Planned vs executed end dist:       " << transition.ending_state_.planned_rubber_band_->distance(*transition.ending_state_.rubber_band_) << std::endl
                        << "Start vs end dist planned:          " << transition.starting_state_.planned_rubber_band_->distance(*transition.ending_state_.planned_rubber_band_) << std::endl
                        << "Start vs end dist executed:         " << transition.starting_state_.rubber_band_->distance(*transition.ending_state_.rubber_band_) << std::endl
                        << "Distance and FOH values along band surface:\n" << dist_and_foh_values.transpose() << std::endl
                        << "Num FOH changes:                    " << num_foh_changes << std::endl);

        return true;
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Generic Visualization
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    void TransitionTesting::setNextVisId(const std_msgs::Int32& msg)
    {
        next_vis_prefix_ = msg.data;
        ROS_INFO_STREAM("Next vis id: " << next_vis_prefix_);
    }

    bool TransitionTesting::removeVisualizationCallback(
            deformable_manipulation_msgs::TransitionTestingVisualizationRequest& req,
            deformable_manipulation_msgs::TransitionTestingVisualizationResponse& res)
    {
        (void)res;
        try
        {
            const auto markers_nsid = visid_to_markers_.at(req.data);
            for (const auto& nsid : markers_nsid)
            {
                vis_->deleteObject(nsid.first, nsid.second);
            }
            visid_to_markers_.erase(req.data);
            ROS_INFO_STREAM("Removed vis id: " << req.data);
            return true;
        }
        catch (...)
        {
            res.response = "Invalid vis id";
            return false;
        }
    }
}
