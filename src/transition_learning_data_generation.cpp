#include "smmap/transition_learning_data_generation.h"
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/serialization_ros.hpp>
#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/log.hpp>
#include <arc_utilities/math_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <sdf_tools/collision_map.hpp>
#include <smmap_utilities/neighbours.h>
#include <boost/filesystem.hpp>
#include <deformable_manipulation_experiment_params/conversions.hpp>
#include <deformable_manipulation_experiment_params/utility.hpp>
#include <deformable_manipulation_msgs/GenerateTransitionDataAction.h>
#include <smmap_utilities/grippers.h>

#include "smmap/band_rrt.h"
#include "smmap/parabola.h"

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

    std::vector<TransitionTesting::StateMicrostepsPair> TransitionTesting::toTrajectory(
            const dmm::TransitionTestResult& test_result,
            const RRTPath& path)
    {
        // Designed to address different ways of generating data (with or without a last action)
        const AllGrippersSinglePose grippers_ending_poses = test_result.microsteps_last_action.size() > 0
                ? VectorGeometryPoseToVectorIsometry3d(test_result.microsteps_last_action.back().gripper_poses)
                : ToGripperPoseVector(path.back().grippers());

        const auto test = robot_->toRosTransitionTest(
                    initial_world_state_.rope_node_transforms_, // Doesn't matter what this is because we do not read anything calculated from this
                    ToGripperPoseVector(path.front().grippers()),
                    RRTPathToGrippersPoseTrajectory(path),
                    grippers_ending_poses);
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

//        std::cout << "Path back grippers:                  " << PrettyPrint::PrettyPrint(ToGripperPositions(path.back().grippers()), true, " ") << std::endl;
//        std::cout << "grippers ending poses (as positions: " << PrettyPrint::PrettyPrint(ToGripperPositions(grippers_ending_poses), true, " ") << std::endl;
//        std::cout << "Path total substeps: " << path_total_substeps << std::endl;

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

            // If the grippers do not move, then skip this part of the path
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
            // Shortcut if the band becomes overstretched
            if (tes.rubber_band_->isOverstretched())
            {
                ROS_WARN_STREAM_NAMED("to_traj", "Band overstretched at index " << idx);
                return trajectory;
            }
            trajectory.push_back({tes, microsteps});
        }

        // Propagate the planned band the last step, and record the resulting state
        if (test_result.microsteps_last_action.size() > 0)
        {
            const WorldState end = ConvertToEigenFeedback(test_result.microsteps_last_action.back());

            auto planned_band = std::make_shared<RubberBand>(*path.back().band());
            planned_band->forwardPropagate(ToGripperPositions(grippers_ending_poses), false);
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
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Template - ending executed band
        if (vis_template_ending_executed_band)
        {
            const auto color = Visualizer::Cyan();
            const auto name = basename + "template__executed";
            const auto new_ids = template_.ending_state_.rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Template - Executed band surface
        if (vis_template_executed_band_surface)
        {
            const auto start_color = Visualizer::Green();
            const auto end_color = Visualizer::Cyan();
            const auto name = basename + "template__band_surface";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, template_band_surface_, template_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Test - start planned band
        if (vis_test_start_planned_band)
        {
            const auto color = Visualizer::Yellow();
            const auto name = basename + "tested__start";
            const auto new_ids = tested_.starting_state_.planned_rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Test - exectued band
        if (vis_test_executed_band)
        {
            const auto color = Visualizer::Orange();
            const auto name = basename + "tested__executed";
            const auto new_ids = tested_.ending_state_.rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Test - Executed band surface
        if (vis_test_executed_band_surface)
        {
            const auto start_color = Visualizer::Yellow();
            const auto end_color = Visualizer::Orange();
            const auto name = basename + "tested__band_surface";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, tested_band_surface_, tested_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Adaptation process - default next band
        if (vis_adapt_default_next_band)
        {
            const auto color = Visualizer::Red();
            const auto name = basename + "adaptation__default_next_band";
            const auto new_ids = adaptation_result_.default_next_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
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
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
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
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
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
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Adaptation process - next_band_points_to_smooth_
        if (vis_adapt_next_band_points_to_smooth)
        {
            const auto color = Visualizer::Seafoam();
            const auto name = basename + "adaptation__next_band_points_to_smooth";
            const auto new_ids = vis->visualizePoints(name, adaptation_result_.next_band_points_to_smooth_, color, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Adaptation process - transformed_band_surface_points
        if (vis_adapt_transformed_band_surface_points)
        {
            const auto start_color = Visualizer::Blue();
            const auto end_color = Visualizer::Seafoam();
            const auto name = basename + "adaptation__transformed_band_surface_points";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, adaptation_result_.transformed_band_surface_points_, template_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Adaptation process - re-tightened band surface
        if (vis_adapt_retightend_band_surface)
        {
            const auto start_color = Visualizer::Olive();
            const auto end_color = Visualizer::Coral();
            const auto name = basename + "adaptation__tightened_transformed_bands_surface";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, adaptation_result_.tightened_transformed_bands_surface_, template_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
        }
        // Adaptation process - final result
        if (vis_adapt_final_result)
        {
            const auto color = Visualizer::Coral();
            const auto name = basename + "adaptation__result";
            const auto new_ids = adaptation_result_.result_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(), new_ids.begin(), new_ids.end());
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
        : nh_(nh)
        , ph_(ph)
        , robot_(robot)
        , vis_(vis)
        , disable_visualizations_(GetDisableAllVisualizations(*ph_))
        , visualize_gripper_motion_(!disable_visualizations_ && GetVisualizeGripperMotion(*ph_))

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

        #warning("Hard coded mistake dist thresh here")
        , mistake_dist_thresh_(0.5)
        , add_mistake_example_visualization_(nh_->advertiseService("transition_vis/add_mistake_example_visualization", &TransitionTesting::addMistakeExampleVisualizationCallback, this))

        , classifier_scaler_(nh_, ph_)
        , transition_mistake_classifier_(nh_, ph_)
        , add_classification_example_visualization_(nh_->advertiseService("transition_vis/add_classification_example_visualization", &TransitionTesting::addClassificationExampleVisualizationCallback, this))
    {
        std::srand((unsigned int)seed_);
        initialize(initial_world_state_);

        // Used for generating data
        gripper_a_starting_pose_.linear() = initial_world_state_.all_grippers_single_pose_[0].linear();
        gripper_b_starting_pose_.linear() = initial_world_state_.all_grippers_single_pose_[1].linear();
        clampGripperDeltas(gripper_a_action_vector_, gripper_b_action_vector_);

        if (visualize_gripper_motion_)
        {
            vis_->visualizeAxes("center_of_rotation",   experiment_center_of_rotation_, 0.1, 0.005, 1);
            vis_->visualizeAxes("gripper_a_start",      gripper_a_starting_pose_,       0.1, 0.005, 1);
            vis_->visualizeAxes("gripper_b_start",      gripper_b_starting_pose_,       0.1, 0.005, 1);
            vis_->visualizeAxes("gripper_a_end",        Translation3d(gripper_a_action_vector_) * gripper_a_starting_pose_, 0.1, 0.005, 1);
            vis_->visualizeAxes("gripper_b_end",        Translation3d(gripper_b_action_vector_) * gripper_b_starting_pose_, 0.1, 0.005, 1);
        }
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
        world_params_ = std::make_shared<const BandRRT::WorldParams>(BandRRT::WorldParams
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

        band_rrt_vis_ = std::make_shared<const BandRRT>(nh_,
                                                        ph_,
                                                        *world_params_,
                                                        planning_params_,
                                                        smoothing_params_,
                                                        task_params_,
                                                        vis_,
                                                        false);
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
        ROS_INFO_STREAM("Finding data files in folder: " << data_folder_);

        std::vector<std::string> files;
        const boost::filesystem::path p(data_folder_);
        const boost::filesystem::recursive_directory_iterator start(p);
        const boost::filesystem::recursive_directory_iterator end;
        for (auto itr = start; itr != end; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->status()))
            {
                const auto filename = itr->path().string();
                // Only warn about file types that are not expected
                if (filename.find("compressed") == std::string::npos &&
                    filename.find("failed") == std::string::npos &&
                    filename.find("classification_features") == std::string::npos)
                {
                    ROS_WARN_STREAM("Ignoring file: " << filename);
                }
                if (filename.find("__test_results.compressed") != std::string::npos)
                {
                    // Strip off the extra string for simpler use later
                    files.push_back(filename.substr(0, filename.find("__test_results.compressed")));
                }
            }
        }
        std::sort(files.begin(), files.end());
        ROS_INFO_STREAM("Found " << files.size() << " possible data files in " << data_folder_);
        return files;
    }

    void TransitionTesting::runTests(const bool generate_test_data,
                                     const bool generate_last_step_transition_approximations,
                                     const bool generate_meaningful_mistake_examples,
                                     const bool generate_features)
    {
        if (generate_test_data)
        {
            Stopwatch stopwatch;
            ROS_INFO("Generating test data via Bullet");
            generateTestData();
            ROS_INFO_STREAM("Data generation time taken: " << stopwatch(READ));
        }

        data_files_ = getDataFileList();

        if (generate_last_step_transition_approximations)
        {
            ROS_INFO("Generating last step transition approximations");
            Stopwatch stopwatch;
            generateLastStepTransitionApproximations();
            ROS_INFO_STREAM("Last step transition approximations time taken: " << stopwatch(READ));
        }

        if (generate_meaningful_mistake_examples)
        {
            ROS_INFO("Generating meaningful mistake examples");
            Stopwatch stopwatch;
            generateMeaningfulMistakeExamples();
            ROS_INFO_STREAM("Finding meaningful mistake examples time taken: " << stopwatch(READ));
        }

        if (generate_features)
        {
            Stopwatch stopwatch;
            ROS_INFO("Generating transition features");
            generateFeatures();
            ROS_INFO_STREAM("Generate features time taken: " << stopwatch(READ));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Data Saving/Loading
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    void TransitionTesting::savePath(const RRTPath& path, const std::string& filename) const
    {
        std::vector<uint8_t> buffer;
        SerializeVector<RRTNode>(path, buffer, &RRTNode::Serialize);
        ZlibHelpers::CompressAndWriteToFile(buffer, filename);
    }

    RRTPath TransitionTesting::loadPath(const std::string& filename) const
    {
        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(filename);
        const auto deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
        {
            return RRTNode::Deserialize(buf, cur, *initial_band_);
        };
        const auto path_deserialized = DeserializeVector<RRTNode, Eigen::aligned_allocator<RRTNode>>(buffer, 0, deserializer);
        return path_deserialized.first;
    }

    void TransitionTesting::saveTestResult(const dmm::TransitionTestResult& test_result, const std::string& filename) const
    {
        std::vector<uint8_t> buffer;
        arc_utilities::RosMessageSerializationWrapper(test_result, buffer);
        ZlibHelpers::CompressAndWriteToFile(buffer, filename);
    }

    dmm::TransitionTestResult TransitionTesting::loadTestResult(const std::string& filename) const
    {
        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(filename);
        return arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(buffer, 0).first.test_result;
    }

    void TransitionTesting::saveStateTransition(const TransitionEstimation::StateTransition& state, const std::string& filename) const
    {
        std::vector<uint8_t> buffer;
        state.serialize(buffer);
        ZlibHelpers::CompressAndWriteToFile(buffer, filename);
    }

    TransitionEstimation::StateTransition TransitionTesting::loadStateTransition(const std::string& filename) const
    {
        const auto test_transition_buffer = ZlibHelpers::LoadFromFileAndDecompress(filename);
        return TransitionEstimation::StateTransition::Deserialize(test_transition_buffer, 0, *initial_band_).first;
    }

    void TransitionTesting::saveAdaptationResult(const TransitionEstimation::TransitionAdaptationResult& result, const std::string& filename) const
    {
        std::vector<uint8_t> buffer;
        result.serialize(buffer);
        ZlibHelpers::CompressAndWriteToFile(buffer, filename);
    }

    TransitionEstimation::TransitionAdaptationResult TransitionTesting::loadAdaptationResult(const std::string& filename) const
    {
        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(filename);
        return TransitionEstimation::TransitionAdaptationResult::Deserialize(buffer, 0, *initial_band_).first;
    }

    void TransitionTesting::saveTrajectory(const std::vector<StateMicrostepsPair>& trajectory, const std::string& filename) const
    {
        const auto microsteps_serializer = [] (const std::vector<WorldState>& microsteps, std::vector<uint8_t>& buf)
        {
            return arc_utilities::SerializeVector<WorldState>(microsteps, buf, &WorldState::Serialize);
        };
        const auto item_serializer = [&] (const StateMicrostepsPair& item, std::vector<uint8_t>& buf)
        {
            return arc_utilities::SerializePair<StateMicrostepsPair::first_type, StateMicrostepsPair::second_type>(item, buf, &TransitionEstimation::State::Serialize, microsteps_serializer);
        };
        std::vector<uint8_t> buffer;
        arc_utilities::SerializeVector<StateMicrostepsPair>(trajectory, buffer, item_serializer);
        ZlibHelpers::CompressAndWriteToFile(buffer, filename);
    }

    std::vector<TransitionTesting::StateMicrostepsPair> TransitionTesting::loadTrajectory(const std::string& filename) const
    {
        const auto state_deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
        {
            return TransitionEstimation::State::Deserialize(buf, cur, *initial_band_);
        };
        const auto microsteps_deserializer = [] (const std::vector<uint8_t>& buf, const uint64_t cur)
        {
            return arc_utilities::DeserializeVector<WorldState>(buf, cur, &WorldState::Deserialize);
        };
        const auto item_deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t cur)
        {
            return arc_utilities::DeserializePair<StateMicrostepsPair::first_type, StateMicrostepsPair::second_type>(buf, cur, state_deserializer, microsteps_deserializer);
        };
        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(filename);
        return arc_utilities::DeserializeVector<StateMicrostepsPair>(buffer, 0, item_deserializer).first;
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

        //// Generate the canonical example ////////////////////////////////////
        {
            const std::string folder(data_folder_);
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
            std::cerr << "Num position perturbations: " << perturbations.size() * perturbations.size()<< std::endl;
            #pragma omp parallel for
            for (size_t a_idx = 0; a_idx < perturbations.size(); ++a_idx)
            {
                const Isometry3d gripper_a_starting_pose = Translation3d(perturbations[a_idx]) * gripper_a_starting_pose_;
                const Isometry3d gripper_a_ending_pose = Translation3d(gripper_a_action_vector_) * gripper_a_starting_pose;

                const std::string folder(data_folder_ +
                                         "/perturbed_gripper_start_positions"
                                         "/gripper_a_" + ToString(perturbations[a_idx]));
                arc_utilities::CreateDirectory(folder);

                for (size_t b_idx = 0; b_idx < perturbations.size(); ++b_idx)
                {
                    const Isometry3d gripper_b_starting_pose = Translation3d(perturbations[b_idx]) * gripper_b_starting_pose_;
                    const Isometry3d gripper_b_ending_pose = Translation3d(gripper_b_action_vector_) * gripper_b_starting_pose;

                    const std::string test_id("/gripper_b_" + ToString(perturbations[b_idx]));
                    const std::string test_results_filename = folder + test_id + "__test_results.compressed";
                    const std::string path_to_start_filename = folder + test_id + "__path_to_start.compressed";
                    const std::string failure_file = folder + test_id + "__path_to_start.failure";

                    // Check for the file flag that indicates that this test is not possible
                    if (boost::filesystem::is_regular_file(failure_file))
                    {
                        continue;
                    }

                    try
                    {
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
                                    // Ignore the feedback as the action sever saves the results to file anyway
                                    robot_->generateTransitionData(tests, filenames, nullptr, false);
                                    tests.clear();
                                    filenames.clear();
                                }
                            }
                        }
                    }
                    catch (const std::runtime_error& ex)
                    {
                        Log::Log failure_logger(failure_file, true);
                        ARC_LOG_STREAM(failure_logger, "Unable to plan with perturbation"
                                   << " a: " << perturbations[a_idx].transpose()
                                   << " b: " << perturbations[b_idx].transpose()
                                   << " Message: " << ex.what());
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
            std::cerr << "Num action perturbations: " << perturbations.size() * perturbations.size()<< std::endl;
            #pragma omp parallel for
            for (size_t a_idx = 0; a_idx < perturbations.size(); ++a_idx)
            {
                const std::string folder(data_folder_ +
                                         "/perturbed_gripper_action_vectors"
                                         "/gripper_a_" + ToString(perturbations[a_idx]));
                arc_utilities::CreateDirectory(folder);

                const Vector3d gripper_a_action_vector = gripper_a_action_vector_ + perturbations[a_idx];
                for (size_t b_idx = 0; b_idx < perturbations.size(); ++b_idx)
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
                    const std::string failure_file = folder + test_id + "__path_to_start.failure";

                    // Check for the file flag that indicates that this test is not possible
                    if (boost::filesystem::is_regular_file(failure_file))
                    {
                        continue;
                    }

                    try
                    {
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
                                    // Ignore the feedback as the action sever saves the results to file anyway
                                    robot_->generateTransitionData(tests, filenames, nullptr, false);
                                    tests.clear();
                                    filenames.clear();
                                }
                            }

                        }
                    }
                    catch (const std::runtime_error& ex)
                    {
                        Log::Log failure_logger(failure_file, true);
                        ARC_LOG_STREAM(failure_logger, "Unable to plan with perturbation"
                                   << " a: " << perturbations[a_idx].transpose()
                                   << " b: " << perturbations[b_idx].transpose()
                                   << " Message: " << ex.what());
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
            // Ignore the feedback as the action sever saves the results to file anyway
            robot_->generateTransitionData(tests, filenames, nullptr, false);
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
            return loadPath(filename);
        }
        else
        {
            const auto path = generateTestPath(gripper_target_poses);
            savePath(path, filename);
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
        req.data = "unmodified__test_results.compressed";
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
        Log::Log logger(data_folder_ + "/generate_last_step_transition_approximations.csv", false);
        ARC_LOG(logger, "FILENAME, "
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
        #pragma omp parallel for
        for (size_t idx = 0; idx < data_files_.size(); ++idx)
        {
            const auto& experiment = data_files_[idx];
            const auto test_result_file =       experiment + "__test_results.compressed";
            const auto path_to_start_file =     experiment + "__path_to_start.compressed";
            const auto test_transition_file =   experiment + "__test_transition.compressed";
            const auto adaptation_result_file = experiment + "__adaptation_record.compressed";
            const auto failure_file =           experiment + "__adaptation_record.failed";

            std::vector<std::string> dists_etc(DUMMY_ITEM, "");
            dists_etc[FILENAME] = test_result_file.substr(data_folder_.length() + 1);
            try
            {
                // Check for the file flag that indicatest that this test is not possible
                if (boost::filesystem::is_regular_file(failure_file))
                {
                    continue;
                }

                // Load the test result itself
                const dmm::TransitionTestResult test_result = loadTestResult(test_result_file);

                // Load the resulting transition, if needed generate it first
                const TransitionEstimation::StateTransition test_transition = [&]
                {
                    if (!boost::filesystem::is_regular_file(test_transition_file))
                    {
                        // Load the path that generated the test
                        const RRTPath path_to_start = loadPath(path_to_start_file);
                        // Generate the transition at the end of the path
                        const auto transition = ToStateTransition(test_result, path_to_start);
                        saveStateTransition(transition, test_transition_file);
                        return transition;
                    }
                    else
                    {
                        return loadStateTransition(test_transition_file);
                    }
                }();

                // Load the adaptation record, if needed generate it first
                const TransitionEstimation::TransitionAdaptationResult adaptation_result = [&]
                {
                    if (!boost::filesystem::is_regular_file(adaptation_result_file))
                    {
                        const auto ar = transition_estimator_->generateTransition(
                                    source_transition_,
                                    *test_transition.starting_state_.planned_rubber_band_,
                                    test_transition.ending_gripper_positions_);
                        saveAdaptationResult(ar, adaptation_result_file);
                        return ar;
                    }
                    else
                    {
                        return loadAdaptationResult(adaptation_result_file);
                    }
                }();

                const auto test_band_start = RubberBand::BandFromWorldState(ConvertToEigenFeedback(test_result.start_after_following_path), *initial_band_);
                if (test_band_start->isOverstretched())
                {
                    throw_arc_exception(std::runtime_error, "Starting configuration of test band is overstretched");
                }
                const auto test_band_end = RubberBand::BandFromWorldState(ConvertToEigenFeedback(test_result.microsteps_last_action.back()), *initial_band_);

                dists_etc[TEMPLATE_MISALIGNMENT_EUCLIDEAN] = std::to_string(adaptation_result.template_misalignment_dist_);
                dists_etc[DEFAULT_VS_ADAPTATION_FOH] = std::to_string(adaptation_result.default_band_foh_result_);
                dists_etc[DEFAULT_VS_ADAPTATION_EUCLIDEAN] = std::to_string(adaptation_result.default_band_dist_);

                dists_etc[BAND_TIGHTEN_DELTA] = std::to_string(adaptation_result.band_tighten_delta_);
                dists_etc[SOURCE_NUM_FOH_CHANGES] = std::to_string(source_num_foh_changes_);
                dists_etc[RESULT_NUM_FOH_CHANGES] = std::to_string(adaptation_result.num_foh_changes_);

                dists_etc[TRUE_VS_DEFAULT_FOH] = std::to_string(transition_estimator_->checkFirstOrderHomotopy(*adaptation_result.default_next_band_, *test_band_end));
                dists_etc[TRUE_VS_DEFAULT_EUCLIDEAN] = std::to_string(adaptation_result.default_next_band_->distance(*test_band_end));

                dists_etc[TRUE_VS_ADAPTATION_FOH] = std::to_string(transition_estimator_->checkFirstOrderHomotopy(*adaptation_result.result_, *test_band_end));
                dists_etc[TRUE_VS_ADAPTATION_EUCLIDEAN] = std::to_string(adaptation_result.result_->distance(*test_band_end));

                dists_etc[PLANNED_VS_ACTUAL_START_FOH] = std::to_string(transition_estimator_->checkFirstOrderHomotopy(*test_transition.starting_state_.planned_rubber_band_, *test_transition.starting_state_.rubber_band_));
                dists_etc[PLANNED_VS_ACTUAL_START_EUCLIDEAN] = std::to_string(test_transition.starting_state_.planned_rubber_band_->distance(*test_transition.starting_state_.rubber_band_));
            }
            catch (const std::exception& ex)
            {
                Log::Log failure_logger(failure_file, true);
                ARC_LOG_STREAM(failure_logger, "Error parsing idx: " << idx << " file: " << test_result_file << ": " << ex.what());
                ROS_ERROR_STREAM_NAMED("last_step_approximations", "Error parsing idx: " << idx << " file: " << test_result_file << ": " << ex.what());
                dists_etc[ERROR_STRING] = ex.what();
            }

            ARC_LOG(logger, PrettyPrint::PrettyPrint(dists_etc, false, ", "));
        }
    }

    bool TransitionTesting::setTransitionAdaptationSourceCallback(
            dmm::TransitionTestingVisualizationRequest& req,
            dmm::TransitionTestingVisualizationResponse& res)
    {
        (void)res;

        source_valid_ = false;

        const std::string delimiter = "__test_results.compressed";
        const auto pos = req.data.find(delimiter);
        const auto experiment = data_folder_ + "/" + req.data.substr(0, pos);

        const auto test_result_file =       experiment + "__test_results.compressed";
        const auto path_to_start_file =     experiment + "__path_to_start.compressed";
        const auto test_transition_file =   experiment + "__test_transition.compressed";

        // Load the resulting transition, if needed generate it first
        source_file_ = req.data;
        source_transition_ = [&]
        {
            if (!boost::filesystem::is_regular_file(test_transition_file))
            {
                const RRTPath path_to_start = loadPath(path_to_start_file);
                const dmm::TransitionTestResult test_result = loadTestResult(test_result_file);

                const auto transition = ToStateTransition(test_result, path_to_start);
                saveStateTransition(transition, test_transition_file);
                return transition;
            }
            else
            {
                return loadStateTransition(test_transition_file);
            }
        }();
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
            dmm::TransitionTestingVisualizationRequest& req,
            dmm::TransitionTestingVisualizationResponse& res)
    {
        if (!source_valid_)
        {
            ROS_WARN_NAMED("data_visualization", "Visualization requested, but transition source is invalid");
            res.response = "Visualization requested, but transition source is invalid";
            return false;
        }

        const std::string delimiter = "__test_results.compressed";
        const auto pos = req.data.find(delimiter);
        const auto experiment = data_folder_ + "/" + req.data.substr(0, pos);

        const auto test_result_file =       experiment + "__test_results.compressed";
        const auto path_to_start_file =     experiment + "__path_to_start.compressed";
        const auto test_transition_file =   experiment + "__test_transition.compressed";

        // Load the test result itself
        const dmm::TransitionTestResult test_result = loadTestResult(test_result_file);

        // Load the resulting transition, if needed generate it first
        const TransitionEstimation::StateTransition test_transition = [&]
        {
            if (!boost::filesystem::is_regular_file(test_transition_file))
            {
                // Load the path that generated the test
                const RRTPath path_to_start = loadPath(path_to_start_file);
                // Generate the transition at the end of the path
                const auto transition = ToStateTransition(test_result, path_to_start);
                saveStateTransition(transition, test_transition_file);
                return transition;
            }
            else
            {
                return loadStateTransition(test_transition_file);
            }
        }();

        // Don't use any saved files as we could be using a different source transition
        const TransitionEstimation::TransitionAdaptationResult adaptation_result =
                transition_estimator_->generateTransition(
                    source_transition_,
                    *test_transition.starting_state_.planned_rubber_band_,
                    test_transition.ending_gripper_positions_);

        const auto sim_record = TransitionSimulationRecord
        {
            source_transition_,
            RubberBand::AggregateBandPoints(source_transition_.microstep_band_history_),
            test_transition,
            RubberBand::AggregateBandPoints(test_transition.microstep_band_history_),
            adaptation_result
        };

        // Remove any existing visualization at this id (if there is one)
        {
            dmm::TransitionTestingVisualizationRequest dmmreq;
            dmmreq.data = std::to_string(next_vis_prefix_);
            dmm::TransitionTestingVisualizationResponse dmmres;
            removeVisualizationCallback(dmmreq, dmmres);
        }

        res.response = std::to_string(next_vis_prefix_);
        visid_to_markers_[res.response] = sim_record.visualize(std::to_string(next_vis_prefix_) + "__", vis_);
        ++next_vis_prefix_;

        const auto test_band_end = RubberBand::BandFromWorldState(ConvertToEigenFeedback(test_result.microsteps_last_action.back()), *initial_band_);
        ROS_INFO_STREAM("Added vis id: " << res.response << " for file " << req.data << std::endl
                        << "Template alignment dist:      " << adaptation_result.template_misalignment_dist_ << std::endl
                        << "Default band FOH:             " << adaptation_result.default_band_foh_result_ << std::endl
                        << "Default band dist:            " << adaptation_result.default_band_dist_ << std::endl
                        << "Band tighten delta:           " << adaptation_result.band_tighten_delta_ << std::endl
                        << "Source FOH changes:           " << source_num_foh_changes_ << std::endl
                        << "Adaptation FOH changes:       " << adaptation_result.num_foh_changes_ << std::endl
                        << "True vs default FOH:          " << transition_estimator_->checkFirstOrderHomotopy(*adaptation_result.default_next_band_, *test_band_end) << std::endl
                        << "True vs default dist:         " << adaptation_result.default_next_band_->distance(*test_band_end) << std::endl
                        << "True vs adaptation FOH:       " << transition_estimator_->checkFirstOrderHomotopy(*adaptation_result.result_, *test_band_end) << std::endl
                        << "True vs adaptation dist:      " << adaptation_result.result_->distance(*test_band_end) << std::endl
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
            LARGEST_FOH_CHANGE_DIST,
            DUMMY_ITEM
        };
        Log::Log logger(data_folder_ + "/generate_meaningful_mistake_examples.csv", false);
        ARC_LOG(logger, "FILENAME, "
                    "ERROR_STRING, "
                    "PLANNED_VS_EXECUTED_START_EUCLIDEAN, "
                    "PLANNED_VS_EXECUTED_END_EUCLIDEAN, "
                    "START_VS_END_EUCLIDEN_PLANNED, "
                    "START_VS_END_EUCLIDEN_EXECUTED, "
                    "FOH_RESULTS, "
                    "LARGEST_FOH_CHANGE_DIST, "
                    "NUM_FOH_CHANGES");

        #pragma omp parallel for
        for (size_t idx = 0; idx < data_files_.size(); ++idx)
        {
            const auto& experiment = data_files_[idx];
            const auto test_result_file =       experiment + "__test_results.compressed";
            const auto path_to_start_file =     experiment + "__path_to_start.compressed";
            const auto example_mistake_file =   experiment + "__example_mistake.compressed";
            const auto failure_file =           experiment + "__example_mistake.failed";

            std::vector<std::string> dists_etc(DUMMY_ITEM, "");
            dists_etc[FILENAME] = test_result_file.substr(data_folder_.length() + 1);
            try
            {
                // Check for the file flag that indicatest that this test is not possible
                if (boost::filesystem::is_regular_file(failure_file))
                {
                    continue;
                }

                // Load the transition example if possible, otherwise generate it
                const TransitionEstimation::StateTransition transition = [&]
                {
                    if (!boost::filesystem::is_regular_file(example_mistake_file))
                    {
                        // Load the path that generated the test
                        const RRTPath path_to_start = loadPath(path_to_start_file);

                        // Load the test record
                        const dmm::TransitionTestResult test_result = loadTestResult(test_result_file);

                        const auto trajectory = toTrajectory(test_result, path_to_start);
                        const auto example = transition_estimator_->findMostRecentBadTransition(trajectory).Get();
                        saveStateTransition(example, example_mistake_file);
                        return example;
                    }
                    else
                    {
                        return loadStateTransition(example_mistake_file);
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
                double largest_foh_change_dist = 0.0;
                for (size_t foh_idx = 0; foh_idx < foh_values.size() - 1; ++foh_idx)
                {
                    if (foh_values[foh_idx] != foh_values[foh_idx + 1])
                    {
                        ++num_foh_changes;

                        RubberBand::Ptr b1 = transition.microstep_band_history_[foh_idx];
                        RubberBand::Ptr b2 = transition.microstep_band_history_[foh_idx + 1];
                        const double foh_change_dist = b1->distance(*b2);
                        largest_foh_change_dist = std::max(largest_foh_change_dist, foh_change_dist);
                    }
                }

                dists_etc[PLANNED_VS_EXECUTED_START_EUCLIDEAN] = std::to_string(transition.starting_state_.planned_rubber_band_->distance(*transition.starting_state_.rubber_band_));
                dists_etc[PLANNED_VS_EXECUTED_END_EUCLIDEAN] = std::to_string(transition.ending_state_.planned_rubber_band_->distance(*transition.ending_state_.rubber_band_));

                dists_etc[START_VS_END_EUCLIDEN_PLANNED] = std::to_string(transition.starting_state_.planned_rubber_band_->distance(*transition.ending_state_.planned_rubber_band_));
                dists_etc[START_VS_END_EUCLIDEN_EXECUTED] = std::to_string(transition.starting_state_.rubber_band_->distance(*transition.ending_state_.rubber_band_));

                dists_etc[FOH_RESULTS] = PrettyPrint::PrettyPrint(foh_values, false, "");
                dists_etc[NUM_FOH_CHANGES] = std::to_string(num_foh_changes);
                dists_etc[LARGEST_FOH_CHANGE_DIST] = std::to_string(largest_foh_change_dist);
            }
            catch (const std::exception& ex)
            {
                Log::Log failure_logger(failure_file, true);
                ARC_LOG_STREAM(failure_logger, "Error parsing idx: " << idx << " file: " << test_result_file << ": " << ex.what());
                ROS_ERROR_STREAM_NAMED("meaningful_mistake", "Error parsing idx: " << idx << " file: " << test_result_file << ": " << ex.what());
                dists_etc[ERROR_STRING] = ex.what();
            }

            ARC_LOG(logger, PrettyPrint::PrettyPrint(dists_etc, false, ", "));
        }
    }

    bool TransitionTesting::addMistakeExampleVisualizationCallback(
            dmm::TransitionTestingVisualizationRequest& req,
            dmm::TransitionTestingVisualizationResponse& res)
    {
        const std::string delimiter = "__test_results.compressed";
        const auto pos = req.data.find(delimiter);
        const auto experiment = data_folder_ + "/" + req.data.substr(0, pos);

        const auto test_result_file =       experiment + "__test_results.compressed";
        const auto path_to_start_file =     experiment + "__path_to_start.compressed";
        const auto example_mistake_file =   experiment + "__example_mistake.compressed";
        const auto trajectory_file =        experiment + "__trajectory.compressed";

        // Load the path that generated the test
        const RRTPath path_to_start = loadPath(path_to_start_file);

        // Load the trajectory if possible, otherwise generate it
        const std::vector<StateMicrostepsPair> trajectory = [&]
        {
            if (!boost::filesystem::is_regular_file(trajectory_file))
            {
                const dmm::TransitionTestResult test_result = loadTestResult(test_result_file);
                const auto traj = toTrajectory(test_result, path_to_start);
                saveTrajectory(traj, trajectory_file);
                return traj;
            }
            else
            {
                return loadTrajectory(trajectory_file);
            }
        }();

        // Load the transition example if possible, otherwise generate it
        const TransitionEstimation::StateTransition transition = [&]
        {
            if (!boost::filesystem::is_regular_file(example_mistake_file))
            {
                const auto example = transition_estimator_->findMostRecentBadTransition(trajectory).Get();
                saveStateTransition(example, example_mistake_file);
                return example;
            }
            else
            {
                return loadStateTransition(example_mistake_file);
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

        // Visualization
        {
            std::vector<Visualizer::NamespaceId> marker_ids;
            const std::string ns_prefix = std::to_string(next_vis_prefix_) + "__";

            // Remove any existing visualization at this id (if there is one)
            {
                dmm::TransitionTestingVisualizationRequest dmmreq;
                dmmreq.data = std::to_string(next_vis_prefix_);
                dmm::TransitionTestingVisualizationResponse dmmres;
                removeVisualizationCallback(dmmreq, dmmres);
            }

            // Planned Path
            {
                const auto draw_bands = true;
                const auto path_ids = band_rrt_vis_->visualizePath(path_to_start, ns_prefix + "PLANNED_", 1, draw_bands);

                const auto gripper_a_last_id = vis_->visualizeCubes(ns_prefix + "PLANNED_" + BandRRT::RRT_PATH_GRIPPER_A_NS, {trajectory.back().first.planned_rubber_band_->getEndpoints().first}, Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), 2);
                const auto gripper_b_last_id = vis_->visualizeCubes(ns_prefix + "PLANNED_" + BandRRT::RRT_PATH_GRIPPER_B_NS, {trajectory.back().first.planned_rubber_band_->getEndpoints().second}, Vector3d(0.005, 0.005, 0.005), Visualizer::Red(), 2);

                marker_ids.insert(marker_ids.end(), path_ids.begin(), path_ids.end());
                marker_ids.insert(marker_ids.end(), gripper_a_last_id.begin(), gripper_a_last_id.end());
                marker_ids.insert(marker_ids.end(), gripper_b_last_id.begin(), gripper_b_last_id.end());

            }

            // Actual Path
            {
                for (size_t idx = 0; idx < trajectory.size(); ++idx)
                {
                    const auto& state = trajectory[idx].first;
                    const auto new_ids = state.rubber_band_->visualize(ns_prefix + "EXECUTED_BAND", Visualizer::Yellow(), Visualizer::Yellow(), (int32_t)(idx + 1));
                    marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                }
            }

            // Discovered mistake
            {
                // Add the first band surface band
                {
                    const bool foh = dist_and_foh_values(1, 0);
                    const auto color = foh ? Visualizer::Green() : Visualizer::Red();
                    const auto ns = foh ? ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_SAME" : ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_DIFF";
                    const auto new_ids = transition.microstep_band_history_.back()->visualize(ns, color, color, (int)(dist_and_foh_values.cols() + 1));
                    marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                }
                // Add the "middle" band surface bands
                for (size_t step_idx = 1; step_idx < transition.microstep_band_history_.size() - 1; ++step_idx)
                {
                    const auto ratio = (float)(step_idx) / (float)(transition.microstep_band_history_.size() - 1);
                    const bool foh = (bool)dist_and_foh_values(1, step_idx - 1) && (bool)dist_and_foh_values(1, step_idx);
                    const auto color = foh
                            ? InterpolateColor(Visualizer::Green(), Visualizer::Cyan(), ratio)
                            : InterpolateColor(Visualizer::Red(), Visualizer::Magenta(), ratio);
                    const auto ns = foh ? ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_SAME" : ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_DIFF";
                    const auto new_ids = transition.microstep_band_history_[step_idx]->visualize(ns, color, color, (int)(step_idx + 1));
                    marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                }
                // Add the last band surface band
                {
                    const bool foh = dist_and_foh_values(1, dist_and_foh_values.cols() - 1);
                    const auto color = foh ? Visualizer::Cyan() : Visualizer::Magenta();
                    const auto ns = foh ? ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_SAME" : ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_DIFF";
                    const auto new_ids = transition.microstep_band_history_.back()->visualize(ns, color, color, (int)(dist_and_foh_values.cols() + 1));
                    marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                }

                // Add the planned vs executed start and end bands on their own namespaces
                {
                    const auto new_ids1 = transition.starting_state_.planned_rubber_band_->visualize(
                                ns_prefix + "MISTAKE_START_PLANNED",
                                Visualizer::Green(),
                                Visualizer::Green(),
                                1);
                    const auto new_ids2 = transition.starting_state_.rubber_band_->visualize(
                                ns_prefix + "MISTAKE_START_EXECUTED",
                                Visualizer::Red(),
                                Visualizer::Red(),
                                1);
                    const auto new_ids3 = transition.ending_state_.planned_rubber_band_->visualize(
                                ns_prefix + "MISTAKE_END_PLANNED",
                                Visualizer::Olive(),
                                Visualizer::Olive(),
                                1);
                    const auto new_ids4 = transition.ending_state_.rubber_band_->visualize(
                                ns_prefix + "MISTAKE_END_EXECUTED",
                                Visualizer::Orange(),
                                Visualizer::Orange(),
                                1);

                    marker_ids.insert(marker_ids.begin(), new_ids1.begin(), new_ids1.end());
                    marker_ids.insert(marker_ids.begin(), new_ids2.begin(), new_ids2.end());
                    marker_ids.insert(marker_ids.begin(), new_ids3.begin(), new_ids3.end());
                    marker_ids.insert(marker_ids.begin(), new_ids4.begin(), new_ids4.end());
                }
            }

            res.response = std::to_string(next_vis_prefix_);
            visid_to_markers_[res.response] = marker_ids;
            ++next_vis_prefix_;
        }

        ROS_INFO_STREAM("Added vis id: " << res.response << " for file " << req.data << std::endl
                        << "Planned vs executed start FOH:      " << transition_estimator_->checkFirstOrderHomotopy(*transition.starting_state_.planned_rubber_band_, *transition.starting_state_.rubber_band_) << std::endl
                        << "Planned vs executed start dist:     " << transition.starting_state_.planned_rubber_band_->distance(*transition.starting_state_.rubber_band_) << std::endl
                        << "Planned vs executed end FOH:        " << transition_estimator_->checkFirstOrderHomotopy(*transition.ending_state_.planned_rubber_band_, *transition.ending_state_.rubber_band_) << std::endl
                        << "Planned vs executed end dist:       " << transition.ending_state_.planned_rubber_band_->distance(*transition.ending_state_.rubber_band_) << std::endl
                        << "Start vs end dist planned:          " << transition.starting_state_.planned_rubber_band_->distance(*transition.ending_state_.planned_rubber_band_) << std::endl
                        << "Start vs end dist executed:         " << transition.starting_state_.rubber_band_->distance(*transition.ending_state_.rubber_band_) << std::endl
                        << "Num FOH changes:                    " << num_foh_changes << std::endl
                        << "Distance and FOH values along band surface:\n" << dist_and_foh_values.transpose() << std::endl);

        return true;
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Generate data for offline learning testing of features etc
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    void TransitionTesting::generateFeatures()
    {
        const std::vector<std::string> feature_names =
        {
            std::string("GRIPPER_A_PRE_X"),
            std::string("GRIPPER_A_PRE_Y"),
            std::string("GRIPPER_A_PRE_Z"),
            std::string("GRIPPER_B_PRE_X"),
            std::string("GRIPPER_B_PRE_Y"),
            std::string("GRIPPER_B_PRE_Z"),
            std::string("GRIPPER_A_POST_X"),
            std::string("GRIPPER_A_POST_Y"),
            std::string("GRIPPER_A_POST_Z"),
            std::string("GRIPPER_B_POST_X"),
            std::string("GRIPPER_B_POST_Y"),
            std::string("GRIPPER_B_POST_Z"),

            std::string("GRIPPER_DELTA_LENGTH_PRE"),
            std::string("GRIPPER_DELTA_LENGTH_POST"),

            std::string("MAX_BAND_LENGTH"),
            std::string("STARTING_BAND_LENGTH"),
            std::string("ENDING_DEFAULT_BAND_LENGTH"),

//            std::string("STARTING_MAJOR_AXIS_LENGTH"),
//            std::string("STARTING_MINOR_AXIS_LENGTH"),
//            std::string("ENDING_MAJOR_AXIS_LENGTH"),
//            std::string("ENDING_MINOR_AXIS_LENGTH"),

            std::string("SLICE_NUM_CONNECTED_COMPONENTS_PRE"),
            std::string("SLICE_NUM_CONNECTED_COMPONENTS_POST"),
            std::string("SLICE_NUM_CONNECTED_COMPONENTS_DELTA"),

            std::string("SLICE_NUM_FREE_CONNECTED_COMPONENTS_PRE"),
            std::string("SLICE_NUM_FREE_CONNECTED_COMPONENTS_POST"),
            std::string("SLICE_NUM_FREE_CONNECTED_COMPONENTS_DELTA"),

            std::string("SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_PRE"),
            std::string("SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_POST"),
            std::string("SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_DELTA")
        };

//        Log::Log logger(data_folder_ + "/meaningful_mistake_features.csv", false);

        int num_examples_delta_eq_0 = 0;
        int num_examples_delta_neq_0 = 0;

        #pragma omp parallel for
        for (size_t file_idx = 0; file_idx < data_files_.size(); ++file_idx)
        {
            const auto& experiment = data_files_[file_idx];
            const auto test_result_file =            experiment + "__test_results.compressed";
            const auto path_to_start_file =          experiment + "__path_to_start.compressed";
            const auto trajectory_file =             experiment + "__trajectory.compressed";
            const auto features_file =               experiment + "__classification_features.csv";
            const auto features_complete_flag_file = experiment + "__classification_features.complete";
//            const auto features_metadata_file =      experiment + "__classification_metadata.csv";

            Log::Log logger(features_file, false);

            try
            {
                if (!boost::filesystem::is_regular_file(features_complete_flag_file))
                {
                    const RRTPath path_to_start = loadPath(path_to_start_file);

                    // Load the trajectory if possible, otherwise generate it
                    const std::vector<StateMicrostepsPair> trajectory = [&]
                    {
                        if (!boost::filesystem::is_regular_file(trajectory_file))
                        {
                            const dmm::TransitionTestResult test_result = loadTestResult(test_result_file);
                            const auto traj = toTrajectory(test_result, path_to_start);
                            saveTrajectory(traj, trajectory_file);
                            return traj;
                        }
                        else
                        {
                            return loadTrajectory(trajectory_file);
                        }
                    }();

                    // Step through the trajectory, looking for cases where the prediction goes
                    // from homotopy match to homotopy mismatch and large Euclidean distance

                    assert(trajectory.size() > 0);
                    bool start_foh = transition_estimator_->checkFirstOrderHomotopy(
                                *trajectory[0].first.planned_rubber_band_,
                                *trajectory[0].first.rubber_band_);

                    for (size_t idx = 1; idx < trajectory.size(); ++idx)
                    {
                        const auto& start_state = trajectory[idx - 1].first;
                        const auto& end_state = trajectory[idx].first;
                        const bool end_foh = transition_estimator_->checkFirstOrderHomotopy(
                                    *end_state.planned_rubber_band_,
                                    *end_state.rubber_band_);
                        const auto dist = end_state.planned_rubber_band_->distance(*end_state.rubber_band_);

                        // Only compute the microstep band history if we're going to actually use it
                        std::vector<RubberBand::Ptr> microstep_band_history;
                        if (!disable_visualizations_)
                        {
                            microstep_band_history = transition_estimator_->reduceMicrostepsToBands(trajectory[idx].second);
                        }

                        const TransitionEstimation::StateTransition transition
                        {
                            start_state,
                            end_state,
                            start_state.planned_rubber_band_->getEndpoints(),
                            end_state.planned_rubber_band_->getEndpoints(),
                            trajectory[idx].second,
                            microstep_band_history
                        };
                        const auto features = extractFeatures(transition);
                        assert(feature_names.size() == features.size());
                        const bool mistake = (start_foh && !end_foh) && (dist > mistake_dist_thresh_);
                        ARC_LOG_STREAM(logger, std::to_string(mistake) << ", " << PrettyPrint::PrettyPrint(features, false, ", "));

                        if (!disable_visualizations_)
                        {
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

                            // Visualization
//                            if (!disable_visualizations_)
                            {
                                std::vector<Visualizer::NamespaceId> marker_ids;
                                const std::string ns_prefix = std::to_string(next_vis_prefix_) + "__";

                                // Remove any existing visualization at this id (if there is one)
                                {
                                    dmm::TransitionTestingVisualizationRequest dmmreq;
                                    dmmreq.data = std::to_string(next_vis_prefix_);
                                    dmm::TransitionTestingVisualizationResponse dmmres;
                                    removeVisualizationCallback(dmmreq, dmmres);
                                }

                                // Planned Path
                                {
                                    const auto draw_bands = true;
                                    const auto path_ids = band_rrt_vis_->visualizePath(path_to_start, ns_prefix + "PLANNED_", 1, draw_bands);

                                    const auto gripper_a_last_id = vis_->visualizeCubes(ns_prefix + "PLANNED_" + BandRRT::RRT_PATH_GRIPPER_A_NS, {trajectory.back().first.planned_rubber_band_->getEndpoints().first}, Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), 2);
                                    const auto gripper_b_last_id = vis_->visualizeCubes(ns_prefix + "PLANNED_" + BandRRT::RRT_PATH_GRIPPER_B_NS, {trajectory.back().first.planned_rubber_band_->getEndpoints().second}, Vector3d(0.005, 0.005, 0.005), Visualizer::Red(), 2);

                                    marker_ids.insert(marker_ids.end(), path_ids.begin(), path_ids.end());
                                    marker_ids.insert(marker_ids.end(), gripper_a_last_id.begin(), gripper_a_last_id.end());
                                    marker_ids.insert(marker_ids.end(), gripper_b_last_id.begin(), gripper_b_last_id.end());
                                }

                                // Actual Path
                                {
                                    for (size_t path_idx = 0; path_idx < trajectory.size(); ++path_idx)
                                    {
                                        const auto& state = trajectory[path_idx].first;
                                        const auto new_ids = state.rubber_band_->visualize(ns_prefix + "EXECUTED_BAND", Visualizer::Yellow(), Visualizer::Yellow(), (int32_t)(path_idx + 1));
                                        marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                                    }
                                }

                                // Transition under consideration
                                {
                                    // Add the first band surface band
                                    {
                                        const bool foh = dist_and_foh_values(1, 0);
                                        const auto color = foh ? Visualizer::Green() : Visualizer::Red();
                                        const auto ns = foh ? ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_SAME" : ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_DIFF";
                                        const auto new_ids = transition.microstep_band_history_.back()->visualize(ns, color, color, (int)(dist_and_foh_values.cols() + 1));
                                        marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                                    }
                                    // Add the "middle" band surface bands
                                    for (size_t step_idx = 1; step_idx < transition.microstep_band_history_.size() - 1; ++step_idx)
                                    {
                                        const auto ratio = (float)(step_idx) / (float)(transition.microstep_band_history_.size() - 1);
                                        const bool foh = (bool)dist_and_foh_values(1, step_idx - 1) && (bool)dist_and_foh_values(1, step_idx);
                                        const auto color = foh
                                                ? InterpolateColor(Visualizer::Green(), Visualizer::Cyan(), ratio)
                                                : InterpolateColor(Visualizer::Red(), Visualizer::Magenta(), ratio);
                                        const auto ns = foh ? ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_SAME" : ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_DIFF";
                                        const auto new_ids = transition.microstep_band_history_[step_idx]->visualize(ns, color, color, (int)(step_idx + 1));
                                        marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                                    }
                                    // Add the last band surface band
                                    {
                                        const bool foh = dist_and_foh_values(1, dist_and_foh_values.cols() - 1);
                                        const auto color = foh ? Visualizer::Cyan() : Visualizer::Magenta();
                                        const auto ns = foh ? ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_SAME" : ns_prefix + "MISTAKE_EXECUTED_BAND_SURFACE_FOH_DIFF";
                                        const auto new_ids = transition.microstep_band_history_.back()->visualize(ns, color, color, (int)(dist_and_foh_values.cols() + 1));
                                        marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
                                    }

                                    // Add the planned vs executed start and end bands on their own namespaces
                                    {
                                        const auto new_ids1 = transition.starting_state_.planned_rubber_band_->visualize(
                                                    ns_prefix + "MISTAKE_START_PLANNED",
                                                    Visualizer::Green(),
                                                    Visualizer::Green(),
                                                    1);
                                        const auto new_ids2 = transition.starting_state_.rubber_band_->visualize(
                                                    ns_prefix + "MISTAKE_START_EXECUTED",
                                                    Visualizer::Red(),
                                                    Visualizer::Red(),
                                                    1);
                                        const auto new_ids3 = transition.ending_state_.planned_rubber_band_->visualize(
                                                    ns_prefix + "MISTAKE_END_PLANNED",
                                                    Visualizer::Olive(),
                                                    Visualizer::Olive(),
                                                    1);
                                        const auto new_ids4 = transition.ending_state_.rubber_band_->visualize(
                                                    ns_prefix + "MISTAKE_END_EXECUTED",
                                                    Visualizer::Orange(),
                                                    Visualizer::Orange(),
                                                    1);

                                        marker_ids.insert(marker_ids.begin(), new_ids1.begin(), new_ids1.end());
                                        marker_ids.insert(marker_ids.begin(), new_ids2.begin(), new_ids2.end());
                                        marker_ids.insert(marker_ids.begin(), new_ids3.begin(), new_ids3.end());
                                        marker_ids.insert(marker_ids.begin(), new_ids4.begin(), new_ids4.end());
                                    }
                                }

    //                            res.response = std::to_string(next_vis_prefix_);
                                visid_to_markers_[std::to_string(next_vis_prefix_)] = marker_ids;
    //                            ++next_vis_prefix_;

    //                            ROS_INFO_STREAM("Added vis id: " << std::to_string(next_vis_prefix_) << " for file " << test_result_file << std::endl
    //                                            << "Planned vs executed start FOH:      " << start_foh << std::endl
    //                                            << "Planned vs executed start dist:     " << transition.starting_state_.planned_rubber_band_->distance(*transition.starting_state_.rubber_band_) << std::endl
    //                                            << "Planned vs executed end FOH:        " << end_foh << std::endl
    //                                            << "Planned vs executed end dist:       " << transition.ending_state_.planned_rubber_band_->distance(*transition.ending_state_.rubber_band_) << std::endl
    //                                            << "Start vs end dist planned:          " << transition.starting_state_.planned_rubber_band_->distance(*transition.ending_state_.planned_rubber_band_) << std::endl
    //                                            << "Start vs end dist executed:         " << transition.starting_state_.rubber_band_->distance(*transition.ending_state_.rubber_band_) << std::endl
    //                                            << "Num FOH changes:                    " << num_foh_changes << std::endl
    //                                            << "Distance and FOH values along band surface:\n" << dist_and_foh_values.transpose() << std::endl);

                                if (features[SLICE_NUM_FREE_CONNECTED_COMPONENTS_DELTA] != std::to_string(0) ||
                                    features[SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_DELTA] != std::to_string(0))
                                {
                                    assert(features.size() == feature_names.size());
                                    ++num_examples_delta_neq_0;
                                    ROS_INFO_STREAM_NAMED("features", "Examples with equal number of components: " << num_examples_delta_eq_0 << "   Not equal components: " << num_examples_delta_neq_0);
                                    ROS_INFO_STREAM_NAMED("features", test_result_file);
                                    ROS_INFO_STREAM_NAMED("features", "  Transition at idx: " << idx << " with distance " << dist);
                                    for (size_t i = 0; i < feature_names.size(); ++i)
                                    {
                                        ROS_INFO_STREAM_NAMED("features", "  " << /* std::left << */ std::setw(48) << feature_names[i] << ": " << features[i]);
                                    }
                                    PressAnyKeyToContinue();
                                }
                                else
                                {
                                    ++num_examples_delta_eq_0;
                                }
                            }
                        }
                    }

                    // Create the flag file indicating that this file has been parsed, including the day and time stamp
                    Log::Log flag_file(features_complete_flag_file, true);
                    (void)flag_file;
                }
            }
            catch (const std::exception& ex)
            {
                ROS_ERROR_STREAM("Error parsing idx: " << file_idx << " file: " << test_result_file << ": " << ex.what());
            }
        }
    }

    std::vector<std::string> TransitionTesting::extractFeatures(
            const TransitionEstimation::StateTransition& transition) const
    {
//        static double time = 0.0;
//        static size_t calls = 0;
//        ++calls;

//        Stopwatch sw;

        const Vector3d mid_point_pre =
                (transition.starting_gripper_positions_.first +
                 transition.starting_gripper_positions_.second) / 2.0;

        const Vector3d mid_point_post =
                (transition.ending_gripper_positions_.first +
                 transition.ending_gripper_positions_.second) / 2.0;

        const Vector3d midpoint_translation =  mid_point_post - mid_point_pre;

        const Vector3d mid_to_gripper_b_pre =
                transition.starting_gripper_positions_.second -
                mid_point_pre;

//        const Vector3d mid_to_gripper_b_post =
//                (transition.ending_gripper_positions_.second -
//                 mid_point_post);

        // Transform the starting gipper positions to "neutral":
        //      Centered on the origin (midpoint of the starting gripper positions)
        //      World-z defines the transformed z-direction
        //      +'ve x pointing towards gripper b
        //          If the grippers are directly in line with the z-axis
        //          then use the (average) direction of motion to define the y-axis
        //              If the direction of motion is directly in line with the z-axis
        //              then use use world x and y to define the rotation
        const Isometry3d origin = [&]
        {
            Vector3d x_axis;
            Vector3d y_axis;
            const Vector3d z_axis = Vector3d::UnitZ();
            if (!mid_to_gripper_b_pre.normalized().isApprox(Vector3d::UnitZ()))
            {
                x_axis = Vector3d(mid_to_gripper_b_pre.x(), mid_to_gripper_b_pre.y(), 0.0).normalized();
                y_axis = z_axis.cross(x_axis).normalized();
            }
            else
            {
                if (!midpoint_translation.normalized().isApprox(Vector3d::UnitZ()))
                {
                    y_axis = Vector3d(midpoint_translation.x(), midpoint_translation.y(), 0.0);
                    x_axis = y_axis.cross(x_axis).normalized();
                }
                else
                {
                    x_axis = Vector3d::UnitX();
                    y_axis = Vector3d::UnitY();
                }
            }

            return Isometry3d((Matrix4d() << x_axis, y_axis, z_axis, mid_point_pre,
                                             0.0,    0.0,    0.0,    1.0).finished());
        }();
        const Isometry3d inv_origin = origin.inverse();

        const Vector3d gripper_a_pre = inv_origin * transition.starting_gripper_positions_.first;
        const Vector3d gripper_b_pre = inv_origin * transition.starting_gripper_positions_.second;
        const Vector3d gripper_a_post = inv_origin * transition.ending_gripper_positions_.first;
        const Vector3d gripper_b_post = inv_origin * transition.ending_gripper_positions_.second;

        const double band_length_pre = transition.starting_state_.planned_rubber_band_->totalLength();
        const double default_band_length_post = transition.ending_state_.planned_rubber_band_->totalLength();

        const double dmax = initial_band_->maxSafeLength();
//        const double major_axis_length_pre = dmax / 2.0;
//        const double minor_axis_length_pre = std::sqrt(dmax * dmax / 4.0 - mid_to_gripper_b_pre.squaredNorm());
//        const double major_axis_length_post = dmax / 2.0;
//        const double minor_axis_length_post = std::sqrt(dmax * dmax / 4.0 - mid_to_gripper_b_post.squaredNorm());

        const double resolution = work_space_grid_.minStepDimension() / 2.0;
        sdf_tools::CollisionMapGrid collision_grid_pre = ExtractParabolaSlice(*sdf_, resolution, transition.starting_gripper_positions_, dmax);//, vis_);
        sdf_tools::CollisionMapGrid collision_grid_post = ExtractParabolaSlice(*sdf_, resolution, transition.ending_gripper_positions_, dmax);//, vis_);

        if (!disable_visualizations_)
        {
            auto collision_grid_marker_pre = collision_grid_pre.ExportForDisplay(Visualizer::Red(), Visualizer::Green(), Visualizer::Blue());
            auto collision_grid_marker_post = collision_grid_post.ExportForDisplay(Visualizer::Orange(), Visualizer::Seafoam(), Visualizer::Blue());
//            auto collision_grid_marker_pre = collision_grid_pre.ExportForDisplay(Visualizer::Red(0.2f), Visualizer::Green(0.2f), Visualizer::Blue(0.2f));
//            auto collision_grid_marker_post = collision_grid_post.ExportForDisplay(Visualizer::Orange(0.2f), Visualizer::Seafoam(0.2f), Visualizer::Blue(0.2f));
            collision_grid_marker_pre.ns = "collision_grid_pre";
            collision_grid_marker_post.ns = "collision_grid_post";
            vis_->publish(collision_grid_marker_pre);
            vis_->publish(collision_grid_marker_post);
        }

        const auto components_pre = collision_grid_pre.ExtractConnectedComponents();
        const auto components_post = collision_grid_post.ExtractConnectedComponents();
        const auto num_connected_components_pre = components_pre.size();
        const auto num_connected_components_post = components_post.size();

        // Figure out how many "occupied" components there are, and how many "empty" components there are (pre)
        size_t num_free_components_pre = 0;
        size_t num_occupied_components_pre = 0;
        for (size_t idx = 0; idx < components_pre.size(); ++idx)
        {
            const auto grid_idx = components_pre[idx].at(0);
            const auto occupancy = collision_grid_pre.GetImmutable(grid_idx).first.occupancy;
            if (occupancy < 0.5)
            {
                ++num_free_components_pre;
            }
            else if (occupancy > 0.5)
            {
                ++num_occupied_components_pre;
            }
            else
            {
                assert(false && "Unknown cell in grid");
            }
        }

        // Figure out how many "occupied" components there are, and how many "empty" components there are (post)
        size_t num_free_components_post = 0;
        size_t num_occupied_components_post = 0;
        for (size_t idx = 0; idx < components_post.size(); ++idx)
        {
            const auto grid_idx = components_post[idx].at(0);
            const auto occupancy = collision_grid_post.GetImmutable(grid_idx).first.occupancy;
            if (occupancy < 0.5)
            {
                ++num_free_components_post;
            }
            else if (occupancy > 0.5)
            {
                ++num_occupied_components_post;
            }
            else
            {
                assert(false && "Unknown cell in grid");
            }
        }

        std::vector<std::string> features(FEATURES_DUMMY_ITEM, "");

        features[GRIPPER_A_PRE_X] = std::to_string(gripper_a_pre.x());
        features[GRIPPER_A_PRE_Y] = std::to_string(gripper_a_pre.y());
        features[GRIPPER_A_PRE_Z] = std::to_string(gripper_a_pre.z());
        features[GRIPPER_B_PRE_X] = std::to_string(gripper_b_pre.x());
        features[GRIPPER_B_PRE_Y] = std::to_string(gripper_b_pre.y());
        features[GRIPPER_B_PRE_Z] = std::to_string(gripper_b_pre.z());
        features[GRIPPER_A_POST_X] = std::to_string(gripper_a_post.x());
        features[GRIPPER_A_POST_Y] = std::to_string(gripper_a_post.y());
        features[GRIPPER_A_POST_Z] = std::to_string(gripper_a_post.z());
        features[GRIPPER_B_POST_X] = std::to_string(gripper_b_post.x());
        features[GRIPPER_B_POST_Y] = std::to_string(gripper_b_post.y());
        features[GRIPPER_B_POST_Z] = std::to_string(gripper_b_post.z());

        features[GRIPPER_DELTA_LENGTH_PRE]  = std::to_string((gripper_a_pre - gripper_b_pre).norm());
        features[GRIPPER_DELTA_LENGTH_POST] = std::to_string((gripper_a_post - gripper_b_post).norm());

        features[MAX_BAND_LENGTH]            = std::to_string(dmax);
        features[STARTING_BAND_LENGTH]       = std::to_string(band_length_pre);
        features[ENDING_DEFAULT_BAND_LENGTH] = std::to_string(default_band_length_post);

//        features[STARTING_MAJOR_AXIS_LENGTH] = std::to_string(major_axis_length_pre);
//        features[STARTING_MINOR_AXIS_LENGTH] = std::to_string(minor_axis_length_pre);
//        features[ENDING_MAJOR_AXIS_LENGTH]   = std::to_string(major_axis_length_post);
//        features[ENDING_MINOR_AXIS_LENGTH]   = std::to_string(minor_axis_length_post);

        features[SLICE_NUM_CONNECTED_COMPONENTS_PRE]            = std::to_string(num_connected_components_pre);
        features[SLICE_NUM_CONNECTED_COMPONENTS_POST]           = std::to_string(num_connected_components_post);
        features[SLICE_NUM_CONNECTED_COMPONENTS_DELTA]          = std::to_string((int)num_connected_components_post - (int)num_connected_components_pre);

        features[SLICE_NUM_FREE_CONNECTED_COMPONENTS_PRE]       = std::to_string(num_free_components_pre);
        features[SLICE_NUM_FREE_CONNECTED_COMPONENTS_POST]      = std::to_string(num_free_components_post);
        features[SLICE_NUM_FREE_CONNECTED_COMPONENTS_DELTA]     = std::to_string((int)num_free_components_post - (int)num_free_components_pre);

        features[SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_PRE]   = std::to_string(num_occupied_components_pre);
        features[SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_POST]  = std::to_string(num_occupied_components_post);
        features[SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_DELTA] = std::to_string((int)num_occupied_components_post - (int)num_occupied_components_pre);

//        time += sw(READ);
//        std::cerr << "Calls: " << calls
//                  << "    Time: " << time << std::endl;

        return features;
    }

    bool TransitionTesting::addClassificationExampleVisualizationCallback(
            dmm::TransitionTestingVisualizationRequest& req,
            dmm::TransitionTestingVisualizationResponse& res)
    {
        // Separate out the data source and the transition index. Assumed string format is
        // relative/path/to/file/test_identifier__classification_features.csv : 38
        const std::string delimiter = "__classification_features.csv : ";
        const auto pos = req.data.find(delimiter);
        const auto experiment = data_folder_ + "/" + req.data.substr(0, pos);
        const auto traj_idx = std::stoi(req.data.substr(pos + delimiter.size()));
        assert(traj_idx > 0);

        const auto test_result_file =   experiment + "__test_results.compressed";
        const auto path_to_start_file = experiment + "__path_to_start.compressed";
        const auto trajectory_file =    experiment + "__trajectory.compressed";

        const RRTPath path_to_start = loadPath(path_to_start_file);

        // Load the trajectory if possible, otherwise generate it
        const std::vector<StateMicrostepsPair> trajectory = [&]
        {
            if (!boost::filesystem::is_regular_file(trajectory_file))
            {
                const dmm::TransitionTestResult test_result = loadTestResult(test_result_file);
                const auto traj = toTrajectory(test_result, path_to_start);
                saveTrajectory(traj, trajectory_file);
                return traj;
            }
            else
            {
                return loadTrajectory(trajectory_file);
            }
        }();

        std::vector<Visualizer::NamespaceId> marker_ids;
        const std::string ns_prefix = std::to_string(next_vis_prefix_) + "__";

        // Planned Path
        {
            const auto draw_bands = true;
            const auto path_ids = band_rrt_vis_->visualizePath(path_to_start, ns_prefix + "PLANNED_", 1, draw_bands);

            const auto gripper_a_last_id = vis_->visualizeCubes(ns_prefix + "PLANNED_" + BandRRT::RRT_PATH_GRIPPER_A_NS, {trajectory.back().first.planned_rubber_band_->getEndpoints().first}, Vector3d(0.005, 0.005, 0.005), Visualizer::Magenta(), 2);
            const auto gripper_b_last_id = vis_->visualizeCubes(ns_prefix + "PLANNED_" + BandRRT::RRT_PATH_GRIPPER_B_NS, {trajectory.back().first.planned_rubber_band_->getEndpoints().second}, Vector3d(0.005, 0.005, 0.005), Visualizer::Red(), 2);

            marker_ids.insert(marker_ids.end(), path_ids.begin(), path_ids.end());
            marker_ids.insert(marker_ids.end(), gripper_a_last_id.begin(), gripper_a_last_id.end());
            marker_ids.insert(marker_ids.end(), gripper_b_last_id.begin(), gripper_b_last_id.end());
        }

        // Actual Path
        {
            for (size_t path_idx = 0; path_idx < trajectory.size(); ++path_idx)
            {
                const auto& state = trajectory[path_idx].first;
                const auto new_ids = state.rubber_band_->visualize(ns_prefix + "EXECUTED_BAND", Visualizer::Yellow(), Visualizer::Yellow(), (int32_t)(path_idx + 1));
                marker_ids.insert(marker_ids.begin(), new_ids.begin(), new_ids.end());
            }
        }

        // Specific transition
        const auto& start_state = trajectory[traj_idx - 1].first;
        const auto& end_state = trajectory[traj_idx].first;

        const bool start_foh = transition_estimator_->checkFirstOrderHomotopy(
                    *start_state.planned_rubber_band_,
                    *end_state.rubber_band_);
        const bool end_foh = transition_estimator_->checkFirstOrderHomotopy(
                    *end_state.planned_rubber_band_,
                    *end_state.rubber_band_);
        const auto start_dist = start_state.planned_rubber_band_->distance(*start_state.rubber_band_);
        const auto end_dist = end_state.planned_rubber_band_->distance(*end_state.rubber_band_);

        std::vector<RubberBand::Ptr> microstep_band_history = transition_estimator_->reduceMicrostepsToBands(trajectory[traj_idx].second);

        const TransitionEstimation::StateTransition transition
        {
            start_state,
            end_state,
            start_state.planned_rubber_band_->getEndpoints(),
            end_state.planned_rubber_band_->getEndpoints(),
            trajectory[traj_idx].second,
            microstep_band_history
        };
        const auto features = extractFeatures(transition);
        const bool mistake = (start_foh && !end_foh) && (end_dist > mistake_dist_thresh_);
        const bool predicted_mistake = transition_mistake_classifier_.predict(
                    classifier_scaler_(
                        transition_estimator_->transitionFeatures(
                            *start_state.planned_rubber_band_, *end_state.planned_rubber_band_, false)));

        res.response = std::to_string(next_vis_prefix_);
        visid_to_markers_[res.response] = marker_ids;
        ++next_vis_prefix_;

        ROS_INFO_STREAM("Added vis id: " << res.response << " for file " << req.data << std::endl
                        << "Planned vs executed start FOH:          " << start_foh << std::endl
                        << "Planned vs executed start dist:         " << start_dist << std::endl
                        << "Planned vs executed end FOH:            " << end_foh << std::endl
                        << "Planned vs executed end dist:           " << end_dist << std::endl
                        << "Mistake:                                " << std::boolalpha <<  mistake << std::endl
                        << "Predicted mistake:                      " << std::boolalpha <<  predicted_mistake << std::endl
//                        << "Gripper a transformed start position:   " << features[GRIPPER_A_PRE_X] << ", " << features[GRIPPER_A_PRE_Y] << ", " << features[GRIPPER_A_PRE_Z] << std::endl
//                        << "Gripper b transformed start position:   " << features[GRIPPER_B_PRE_X] << ", " << features[GRIPPER_B_PRE_Y] << ", " << features[GRIPPER_B_PRE_Z] << std::endl
//                        << "Gripper a transformed end position:     " << features[GRIPPER_A_POST_X] << ", " << features[GRIPPER_A_POST_Y] << ", " << features[GRIPPER_A_POST_Z] << std::endl
//                        << "Gripper b transformed end position:     " << features[GRIPPER_B_POST_X] << ", " << features[GRIPPER_B_POST_Y] << ", " << features[GRIPPER_B_POST_Z] << std::endl
                        << "Gripper seperation pre:                 " << features[GRIPPER_DELTA_LENGTH_PRE] << std::endl
                        << "Gripper seperation post:                " << features[GRIPPER_DELTA_LENGTH_POST] << std::endl
//                        << "Band length max:                        " << features[MAX_BAND_LENGTH] << std::endl
                        << "Band length pre:                        " << features[STARTING_BAND_LENGTH] << std::endl
                        << "Band length post:                       " << features[ENDING_DEFAULT_BAND_LENGTH] << std::endl
                        << "Num connected components pre:           " << features[SLICE_NUM_CONNECTED_COMPONENTS_PRE] << std::endl
                        << "Num connected components post:          " << features[SLICE_NUM_CONNECTED_COMPONENTS_POST] << std::endl
                        << "Num connected components delta:         " << features[SLICE_NUM_CONNECTED_COMPONENTS_DELTA] << std::endl
                        << "Num free connected components pre:      " << features[SLICE_NUM_FREE_CONNECTED_COMPONENTS_PRE] << std::endl
                        << "Num free connected components post:     " << features[SLICE_NUM_FREE_CONNECTED_COMPONENTS_POST] << std::endl
                        << "Num free connected components delta:    " << features[SLICE_NUM_FREE_CONNECTED_COMPONENTS_DELTA] << std::endl
                        << "Num filled connected components pre:    " << features[SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_PRE] << std::endl
                        << "Num filled connected components post:   " << features[SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_POST] << std::endl
                        << "Num filled connected components delta:  " << features[SLICE_NUM_OCCUPIED_CONNECTED_COMPONENTS_DELTA] << std::endl);

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
            dmm::TransitionTestingVisualizationRequest& req,
            dmm::TransitionTestingVisualizationResponse& res)
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
            ROS_DEBUG_STREAM("Removed vis id: " << req.data);
            return true;
        }
        catch (...)
        {
            res.response = "Invalid vis id";
            return false;
        }
    }
}
