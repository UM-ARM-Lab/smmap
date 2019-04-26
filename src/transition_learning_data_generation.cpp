#include "smmap/transition_learning_data_generation.h"
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/thin_plate_spline.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/serialization_ros.hpp>
#include <arc_utilities/filesystem.hpp>
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
namespace dmm = deformable_manipulation_msgs;

////////////////////////////////////////////////////////////////////////////////
//          Random Helpers
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
            RubberBand band)
    {
        const auto template_band = band;

        const WorldState start = ConvertToEigenFeedback(test.start_after_following_path);
        const std::vector<WorldState> microsteps_all = ConvertToEigenFeedback(test.microsteps_all);
        const std::vector<WorldState> microsteps_last = ConvertToEigenFeedback(test.microsteps_last_action);
        const WorldState& end = microsteps_all.back();

        // Run the template band through the gripper actions up to the start of of transition test,
        // updating the band only when the grippers move to account for the "microstepping".
        // This gets us the planned band
        auto prev_endpoints = band.getEndpoints();
        for (size_t idx = 0; idx < microsteps_all.size() - microsteps_last.size(); ++idx)
        {
            const auto next_positions = ToGripperPositions(microsteps_all[idx].all_grippers_single_pose_);
            if (next_positions != prev_endpoints)
            {
                band.forwardPropagate(next_positions, false);
                prev_endpoints = next_positions;
            }
        }

        const auto start_state = TransitionEstimation::State
        {
            start.object_configuration_,
            RubberBand::BandFromWorldState(start, template_band),
            std::make_shared<RubberBand>(band),
            start.rope_node_transforms_
        };

        // Propagate the planned band the last step
        band.forwardPropagate(ToGripperPositions(end.all_grippers_single_pose_), false);
        const auto end_state = TransitionEstimation::State
        {
            end.object_configuration_,
            RubberBand::BandFromWorldState(end, template_band),
            std::make_shared<RubberBand>(band),
            end.rope_node_transforms_
        };

        std::vector<RubberBand::Ptr> microsteps_bands;
        microsteps_bands.reserve(microsteps_last.size());
        for (size_t idx = 0; idx < microsteps_last.size(); ++idx)
        {
            microsteps_bands.push_back(std::make_shared<RubberBand>(template_band));
            if (!microsteps_bands.back()->resetBand(microsteps_last[idx]))
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
            microsteps_bands
        };
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

        // Template - starting planned band
        {
            const auto color = Visualizer::Green();
            const auto name = basename + "template__start";
            const auto new_ids = template_.starting_state_.planned_rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Template - ending executed band
        {
            const auto color = Visualizer::Cyan();
            const auto name = basename + "template__executed";
            const auto new_ids = template_.ending_state_.rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Template - Executed band surface
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
        {
            const auto color = Visualizer::Yellow();
            const auto name = basename + "tested__start";
            const auto new_ids = tested_.starting_state_.planned_rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Test - start planned band
        {
            const auto color = Visualizer::Orange();
            const auto name = basename + "tested__executed";
            const auto new_ids = tested_.ending_state_.rubber_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Test - Executed band surface
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
        {
            const auto color = Visualizer::Red();
            const auto name = basename + "adaptation__default_next_band";
            const auto new_ids = adaptation_result_.default_next_band_->visualize(name, color, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - template aligned
        {
            const auto color = Visualizer::Magenta();
            const auto name = basename + "adaptation__template_aligned_to_target";
            const auto new_ids = vis->visualizePoints(name, adaptation_result_.template_planned_band_aligned_to_target_, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - next_band_points_to_smooth_
        {
            const auto color = Visualizer::Seafoam();
            const auto name = basename + "adaptation__next_band_points_to_smooth";
            const auto new_ids = vis->visualizePoints(name, adaptation_result_.next_band_points_to_smooth_, color, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - next_band_points_to_smooth_
        {
            const auto start_color = Visualizer::Blue();
            const auto end_color = Visualizer::Seafoam();
            const auto name = basename + "adaptation__transformed_band_surface_points";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, adaptation_result_.transformed_band_surface_points_, tested_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - re-tightened band surface
        {
            const auto start_color = Visualizer::Olive();
            const auto end_color = Visualizer::Coral();
            const auto name = basename + "adaptation__tightened_transformed_bands_surface";
            const auto new_ids = RubberBand::VisualizeBandSurface(vis, adaptation_result_.tightened_transformed_bands_surface_, tested_.microstep_band_history_.size(), start_color, end_color, name, 1);
            marker_ids.insert(marker_ids.end(),
                              std::make_move_iterator(new_ids.begin()),
                              std::make_move_iterator(new_ids.end()));
        }
        // Adaptation process - final result
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
        , sim_test_result_suffix_("_results.compressed")
        , prediction_result_suffix_("_prediction.compressed")
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
                    nh_, ph_, sdf_, work_space_grid_, vis_, *band_);
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

        band_ = std::make_shared<RubberBand>(
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
        const auto substr_len = sim_test_result_suffix_.size();

        std::vector<std::string> files;
        const boost::filesystem::path p(data_folder_);
        const boost::filesystem::directory_iterator start(p);
        const boost::filesystem::directory_iterator end;
        for (auto itr = start; itr != end; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->status()))
            {
                const auto filename = itr->path().string();
                if (filename.substr(filename.size() - substr_len, std::string::npos) == sim_test_result_suffix_)
                {
                    files.push_back(filename);
                }
                else
                {
                    ROS_WARN_STREAM("Ignoring file: " << filename);
                }
            }
        }
        std::sort(files.begin(), files.end());
        ROS_INFO_STREAM("Found " << files.size() << " possible data files in " << data_folder_);
        return files;
    }

    TransitionSimulationRecord TransitionTesting::loadSimRecord(const std::string& filename)
    {
        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(filename);
        const auto record = TransitionSimulationRecord::Deserialize(buffer, 0, *band_);
        if (record.second != buffer.size())
        {
            throw_arc_exception(std::invalid_argument, "Buffer size mismatch: " + filename);
        }
        return record.first;
    }

    void TransitionTesting::runTests(const bool generate_new_test_data)
    {

        if (generate_new_test_data)
        {
            Stopwatch stopwatch;
            DataGeneration data_generator(*this);
            data_generator.generateTestData(data_folder_);
            std::cout << "Time taken: " << stopwatch(READ) << std::endl;
        }

        auto data_processing = DataProcessing(*this);
        {
            dmm::TransitionTestingVisualizationRequest req;
            req.data = "cannonical_straight_test/unmodified.compressed";
            dmm::TransitionTestingVisualizationResponse res;
            data_processing.setSourceCallback(req, res);
        }
        {
            dmm::TransitionTestingVisualizationRequest req;
            req.data = "cannonical_straight_test/perturbed_gripper_start_positions/gripper_a_-0.1_-0.1_-0.1/gripper_b_-0.1_-0.1_-0.1.compressed";
            dmm::TransitionTestingVisualizationResponse res;
            data_processing.addVisualizationCallback(req, res);
        }
        PressAnyKeyToContinue("visualization testing");

//        const auto files = getDataFileList();
//        for (const auto& file : files)
//        {
//            try
//            {
//                vis_->deleteAll();
//                const TransitionSimulationRecord sim_record = loadSimRecord(file);
//            }
//            catch (const std::exception& ex)
//            {
//                ROS_ERROR_STREAM("Error parsing file: " << file << ": " << ex.what());
//            }
//        }
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

    TransitionTesting::DataGeneration::DataGeneration(
            const TransitionTesting& framework)
        : fw_(framework)
    {}

    void TransitionTesting::DataGeneration::generateTestData(
            const std::string& data_folder)
    {
        const auto num_threads = GetNumOMPThreads();
        std::vector<dmm::TransitionTest> tests;
        std::vector<std::string> filenames;
        tests.reserve(num_threads);
        filenames.reserve(num_threads);

//        std::vector<dmm::TransitionTestResult> results(tests.size());
        const auto feedback_callback = [&] (const size_t test_id, const dmm::TransitionTestResult& result)
        {
//            std::cout << data_folder + "/" + test_descriptions[test_id] + ".compressed" << std::endl;
////            results[test_id] = result;
//            std::vector<uint8_t> buffer;
//            RosMessageSerializationWrapper(result, buffer);
//            ZlibHelpers::CompressAndWriteToFile(buffer, data_folder + "//" + test_descriptions[test_id] + ".compressed");
            (void)test_id;
            (void)result;
        };

        //// Generate the canonical example ////////////////////////////////////
        {
            arc_utilities::CreateDirectory(data_folder + "/cannonical_straight_test");
            const std::string filename(data_folder +
                                       "/cannonical_straight_test"
                                       "/unmodified.compressed");
            if (!boost::filesystem::is_regular_file(filename))
            {
                Isometry3d gripper_a_ending_pose_ = Translation3d(fw_.gripper_a_action_vector_) * fw_.gripper_a_starting_pose_;
                Isometry3d gripper_b_ending_pose_ = Translation3d(fw_.gripper_b_action_vector_) * fw_.gripper_b_starting_pose_;
                const auto canonical_test = fw_.robot_->toRosTransitionTest(
                            fw_.initial_world_state_.rope_node_transforms_,
                            fw_.initial_world_state_.all_grippers_single_pose_,
                            generateTestPath({fw_.gripper_a_starting_pose_, fw_.gripper_b_starting_pose_}),
                            {gripper_a_ending_pose_, gripper_b_ending_pose_});
                tests.push_back(canonical_test);
                filenames.push_back(filename);
            }
        }

        //// Generate versions with perturbed gripper start positions //////////
        {
            const auto max_magnitude = ROSHelpers::GetParamRequired<double>(*fw_.ph_, "perturbations/gripper_positions/max_magnitude", __func__).GetImmutable();
            const auto num_divisions = ROSHelpers::GetParamRequired<int>(*fw_.ph_, "perturbations/gripper_positions/num_divisions", __func__).GetImmutable();
            const auto perturbations = Vec3dPerturbations(max_magnitude, num_divisions);
            std::cout << "Num position perturbations: " << perturbations.size() * perturbations.size()<< std::endl;
            #pragma omp parallel for
            for (size_t a_idx = 0; a_idx < perturbations.size(); ++a_idx)
            {
                const Isometry3d gripper_a_starting_pose = Translation3d(perturbations[a_idx]) * fw_.gripper_a_starting_pose_;
                const Isometry3d gripper_a_ending_pose = Translation3d(fw_.gripper_a_action_vector_) * gripper_a_starting_pose;
                for (size_t b_idx = 0; b_idx < perturbations.size(); ++b_idx)
                {
                    try
                    {
                        const Isometry3d gripper_b_starting_pose = Translation3d(perturbations[b_idx]) * fw_.gripper_b_starting_pose_;
                        const Isometry3d gripper_b_ending_pose = Translation3d(fw_.gripper_b_action_vector_) * gripper_b_starting_pose;

                        const std::string folder(data_folder +
                                                 "/cannonical_straight_test"
                                                 "/perturbed_gripper_start_positions"
                                                 "/gripper_a_" + ToString(perturbations[a_idx]));
                        const std::string filename(folder +
                                                   "/gripper_b_" + ToString(perturbations[b_idx]) +
                                                   ".compressed");
                        arc_utilities::CreateDirectory(folder);

                        if (!boost::filesystem::is_regular_file(filename))
                        {
                            const auto test = fw_.robot_->toRosTransitionTest(
                                        fw_.initial_world_state_.rope_node_transforms_,
                                        fw_.initial_world_state_.all_grippers_single_pose_,
                                        generateTestPath({gripper_a_starting_pose, gripper_b_starting_pose}),
                                        {gripper_a_ending_pose, gripper_b_ending_pose});

                            #pragma omp critical
                            {
                                // Add the test to the list waiting to be executed
                                tests.push_back(test);
                                filenames.push_back(filename);

                                // Execute the tests if tehre are enough to run
                                if (tests.size() == num_threads)
                                {
                                    fw_.robot_->generateTransitionData(tests, filenames, feedback_callback, false);
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
            const auto max_magnitude = ROSHelpers::GetParamRequired<double>(*fw_.ph_, "perturbations/action_vectors/max_magnitude", __func__).GetImmutable();
            const auto num_divisions = ROSHelpers::GetParamRequired<int>(*fw_.ph_, "perturbations/action_vectors/num_divisions", __func__).GetImmutable();
            const auto perturbations = Vec3dPerturbations(max_magnitude, num_divisions);
            std::cout << "Num action perturbations: " << perturbations.size() * perturbations.size()<< std::endl;
            #pragma omp parallel for
            for (size_t a_idx = 0; a_idx < perturbations.size(); ++a_idx)
            {
                const Vector3d gripper_a_action_vector = fw_.gripper_a_action_vector_ + perturbations[a_idx];
                for (size_t b_idx = 0; b_idx < perturbations.size(); ++b_idx)
                {
                    try
                    {
                        const Vector3d gripper_b_action_vector = fw_.gripper_b_action_vector_ + perturbations[b_idx];
                        Vector3d gripper_a_action_vector_normalized = gripper_a_action_vector;
                        Vector3d gripper_b_action_vector_normalized = gripper_b_action_vector;
                        fw_.clampGripperDeltas(gripper_a_action_vector_normalized, gripper_b_action_vector_normalized);

                        const Isometry3d gripper_a_ending_pose = Translation3d(gripper_a_action_vector_normalized) * fw_.gripper_a_starting_pose_;
                        const Isometry3d gripper_b_ending_pose = Translation3d(gripper_b_action_vector_normalized) * fw_.gripper_b_starting_pose_;

                        const std::string folder(data_folder +
                                                 "/cannonical_straight_test"
                                                 "/perturbed_gripper_action_vectors"
                                                 "/gripper_a_" + ToString(perturbations[a_idx]));
                        const std::string filename(folder +
                                                   "/gripper_b_" + ToString(perturbations[b_idx]) +
                                                   ".compressed");
                        arc_utilities::CreateDirectory(folder);

                        if (!boost::filesystem::is_regular_file(filename))
                        {
                            const auto test = fw_.robot_->toRosTransitionTest(
                                        fw_.initial_world_state_.rope_node_transforms_,
                                        fw_.initial_world_state_.all_grippers_single_pose_,
                                        generateTestPath({fw_.gripper_a_starting_pose_, fw_.gripper_b_starting_pose_}),
                                        {gripper_a_ending_pose, gripper_b_ending_pose});

                            #pragma omp critical
                            {
                                // Add the test to the list waiting to be executed
                                tests.push_back(test);
                                filenames.push_back(filename);

                                // Execute the tests if tehre are enough to run
                                if (tests.size() == num_threads)
                                {
                                    fw_.robot_->generateTransitionData(tests, filenames, feedback_callback, false);
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
            fw_.robot_->generateTransitionData(tests, filenames, feedback_callback, false);
            tests.clear();
            filenames.clear();
        }
    }

    AllGrippersPoseTrajectory TransitionTesting::DataGeneration::generateTestPath(
            const AllGrippersSinglePose& gripper_target_poses)
    {
        // Pass in all the config values that the RRT needs; for example goal bias, step size, etc.
        auto band_rrt = BandRRT(fw_.nh_,
                                fw_.ph_,
                                *fw_.world_params_,
                                fw_.planning_params_,
                                fw_.smoothing_params_,
                                fw_.task_params_,
                                fw_.vis_,
                                false);

        const auto gripper_config = RRTGrippersRepresentation(
                    fw_.initial_world_state_.all_grippers_single_pose_[0],
                    fw_.initial_world_state_.all_grippers_single_pose_[1]);

        RRTRobotRepresentation robot_config(6);
        robot_config.head<3>() = gripper_config.first.translation();
        robot_config.tail<3>() = gripper_config.second.translation();

        const auto rubber_band = RubberBand::BandFromWorldState(
                    fw_.initial_world_state_, *fw_.band_);

        const RRTNode start_config(
                    gripper_config,
                    robot_config,
                    rubber_band);

        const std::chrono::duration<double> time_limit(GetRRTTimeout(*fw_.ph_));

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
        return RRTPathToGrippersPoseTrajectory(policy[0].first);
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Data Processing
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    TransitionTesting::DataProcessing::DataProcessing(
            const TransitionTesting& framework)
        : fw_(framework)
        , set_source_(fw_.nh_->advertiseService("transition_vis/set_source", &DataProcessing::setSourceCallback, this))
        , add_visualization_(fw_.nh_->advertiseService("transition_vis/add_visualization", &DataProcessing::addVisualizationCallback, this))
        , remove_visualization_(fw_.nh_->advertiseService("transition_vis/remove_visualization", &DataProcessing::removeVisualizationCallback, this))
        , source_valid_(false)
        , next_vis_prefix_(0)
    {}

    bool TransitionTesting::DataProcessing::setSourceCallback(
            dmm::TransitionTestingVisualizationRequest& req,
            dmm::TransitionTestingVisualizationResponse& res)
    {
        (void)res;

        const std::string fullpath = fw_.data_folder_ + "/" + req.data;
        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(fullpath);
        const auto test_result = arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(buffer, 0).first.test_result;

        source_file_ = req.data;
        source_transition_ = ToStateTransition(test_result, *fw_.band_);
        source_band_surface_ = RubberBand::AggregateBandPoints(source_transition_.microstep_band_history_);
        source_valid_ = true;

        return true;
    }

    bool TransitionTesting::DataProcessing::addVisualizationCallback(
            deformable_manipulation_msgs::TransitionTestingVisualizationRequest& req,
            deformable_manipulation_msgs::TransitionTestingVisualizationResponse& res)
    {
        if (!source_valid_)
        {
            ROS_WARN_NAMED("data_visualization", "Visualization requested, but transition source is invalid");
            res.response = "Visualization requested, but transition source is invalid";
            return false;
        }

        const std::string fullpath = fw_.data_folder_ + "/" + req.data;
        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(fullpath);
        const auto test_result = arc_utilities::RosMessageDeserializationWrapper<dmm::GenerateTransitionDataFeedback>(buffer, 0).first.test_result;

        const auto test_transition = ToStateTransition(test_result, *fw_.band_);
        const auto adaptation_record = fw_.transition_estimator_->generateTransition(
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
        res.response = source_file_ + "____" + req.data;
        visid_to_markers_[res.response] = sim_record.visualize(std::to_string(next_vis_prefix_) + "__", fw_.vis_);
        ++next_vis_prefix_;
        return true;
    }

    bool TransitionTesting::DataProcessing::removeVisualizationCallback(
            deformable_manipulation_msgs::TransitionTestingVisualizationRequest& req,
            deformable_manipulation_msgs::TransitionTestingVisualizationResponse& res)
    {
        (void)res;
        try
        {
            const auto markers_nsid = visid_to_markers_.at(req.data);
            for (const auto& nsid : markers_nsid)
            {
                fw_.vis_->deleteObjects(nsid.first, nsid.second, nsid.second + 1);
            }
            visid_to_markers_.erase(req.data);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
}
