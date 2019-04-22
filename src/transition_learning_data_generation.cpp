#include "smmap/transition_learning_data_generation.h"
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/thin_plate_spline.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/serialization_ros.hpp>
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

////////////////////////////////////////////////////////////////////////////////
//          Random Helpers
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    static std::pair<RubberBand::Ptr,
              std::pair<PairGripperPositions,
                        PairGripperPositions>> BandAndGrippersFromTpsPoints(
            const ObjectPointSet& tps_points,
            const RubberBand& band_template)
    {
        const size_t num_band_points = band_template.upsampleBand().size();
        assert(tps_points.size() == (ssize_t)num_band_points + 4);

        // Extract the band portion
        VectorVector3d band_points(num_band_points);
        for (size_t band_idx = 0; band_idx < num_band_points; ++band_idx)
        {
            band_points[band_idx] = tps_points.col(band_idx);
        }
        const auto band = std::make_shared<RubberBand>(band_template);
        band->setPointsAndSmooth(band_points);

        // Extract the grippers portion
        const PairGripperPositions grippers_start =
        {
            tps_points.col(num_band_points + 0),
            tps_points.col(num_band_points + 1)
        };
        const PairGripperPositions grippers_end =
        {
            tps_points.col(num_band_points + 2),
            tps_points.col(num_band_points + 3)
        };
        return {band, {grippers_start, grippers_end}};
    }

    static std::pair<AllGrippersSinglePose, AllGrippersSinglePose> ExtractGripperPosesFromTransition(
            const TransitionEstimation::StateTransition& trans)
    {
        AllGrippersSinglePose starting_gripper_poses(2);
        starting_gripper_poses[0] = trans.starting_state_.rope_node_transforms_.front();
        starting_gripper_poses[0].translation() = trans.starting_gripper_positions_.first;
        starting_gripper_poses[1] = trans.starting_state_.rope_node_transforms_.back();
        starting_gripper_poses[1].translation() = trans.starting_gripper_positions_.second;
        AllGrippersSinglePose target_gripper_poses = starting_gripper_poses;
        target_gripper_poses[0].translation() = trans.ending_gripper_positions_.first;
        target_gripper_poses[1].translation() = trans.ending_gripper_positions_.second;
        return {starting_gripper_poses, target_gripper_poses};
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Transition Test Results
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    uint64_t TransitionSimulationRecord::serializeSelf(std::vector<uint8_t>& buffer) const
    {
        const auto starting_bytes = buffer.size();
        uint64_t bytes_written = 0;
        bytes_written += template_.serializeSelf(buffer);
        bytes_written += arc_utilities::SerializeEigen(template_band_surface_, buffer);
        bytes_written += arc_utilities::SerializeEigen(center_of_rotation_, buffer);
        bytes_written += arc_utilities::SerializeEigen(transform_applied_, buffer);
        bytes_written += tested_.serializeSelf(buffer);
        bytes_written += arc_utilities::SerializeEigen(tested_band_surface_, buffer);

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

        const auto center_of_rotation_deserialized =
                arc_utilities::DeserializeEigen<Isometry3d>(buffer, current + bytes_read);
        bytes_read += center_of_rotation_deserialized.second;

        const auto transform_applied_deserialized =
                arc_utilities::DeserializeEigen<Isometry3d>(buffer, current + bytes_read);
        bytes_read += transform_applied_deserialized.second;

        const auto tested_deserialized =
                TransitionEstimation::StateTransition::Deserialize(buffer, current + bytes_read, template_band);
        bytes_read += tested_deserialized.second;

        const auto tested_band_surface_deserialized =
                arc_utilities::DeserializeEigen<ObjectPointSet>(buffer, current + bytes_read);
        bytes_read += tested_band_surface_deserialized.second;

        TransitionSimulationRecord record =
        {
            template_deserialized.first,
            template_band_surface_deserialized.first,
            center_of_rotation_deserialized.first,
            transform_applied_deserialized.first,
            tested_deserialized.first,
            tested_band_surface_deserialized.first
        };
        return {record, bytes_read};
    }

    bool TransitionSimulationRecord::operator==(const TransitionSimulationRecord& other) const
    {
        if (template_ != template_)
        {
            return false;
        }
        if ((template_band_surface_.array() != other.template_band_surface_.array()).any())
        {
            return false;
        }
        if ((center_of_rotation_.matrix().array() != other.center_of_rotation_.matrix().array()).any())
        {
            return false;
        }
        if ((transform_applied_.matrix().array() != other.transform_applied_.matrix().array()).any())
        {
            return false;
        }
        if (tested_ != other.tested_)
        {
            return false;
        }
        if ((tested_band_surface_.array() != other.tested_band_surface_.array()).any())
        {
            return false;
        }
        return true;
    }

    void TransitionSimulationRecord::visualize(const Visualizer::Ptr& vis) const
    {
        const size_t points_per_band = template_.starting_state_.rubber_band_->upsampleBand().size();
        const size_t num_bands = template_band_surface_.cols() / points_per_band;
        TransitionEstimation::VisualizeTransition(vis, template_, 1, "template_");
        TransitionEstimation::VisualizeTransition(vis, tested_, 1, "tested_");
        vis->visualizeAxes("center_of_rotation", center_of_rotation_, 0.1, 0.005, 1);
        RubberBand::VisualizeBandSurface(vis, template_band_surface_,   num_bands, Visualizer::Blue(), Visualizer::Red(),     "band_surface_template", 1);
        RubberBand::VisualizeBandSurface(vis, tested_band_surface_,     num_bands, Visualizer::Cyan(), Visualizer::Magenta(), "band_surface_tested",   1);
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
        vis_->visualizeAxes("gripper_a_end",        gripper_a_starting_pose_ * Translation3d(gripper_a_action_vector_), 0.1, 0.005, 1);
        vis_->visualizeAxes("gripper_b_end",        gripper_b_starting_pose_ * Translation3d(gripper_b_action_vector_), 0.1, 0.005, 1);
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

    void TransitionTesting::clampGripperDeltas(Ref<Vector3d> a_delta, Ref<Vector3d> b_delta)
    {
        const double distance = std::sqrt(a_delta.squaredNorm() + b_delta.squaredNorm());
        if (distance > task_params_.max_gripper_step_size_)
        {
            gripper_a_action_vector_ *= (task_params_.max_gripper_step_size_ / distance);
            gripper_b_action_vector_ *= (task_params_.max_gripper_step_size_ / distance);
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

        PressAnyKeyToContinue("Tested canonical position with perturbations");

        const auto files = getDataFileList();
        for (const auto& file : files)
        {
            try
            {
                vis_->deleteAll();

                const TransitionSimulationRecord sim_record = loadSimRecord(file);
//                sim_record.visualize(vis_);
//                PressKeyToContinue("Sim Record Vis ");

//                SE3Prediction se3_predictor(*this);
//                se3_predictor.predictAll(
//                            sim_record.template_,
//                            *sim_record.tested_.starting_state_.rubber_band_,
//                            sim_record.tested_.ending_gripper_positions_);
//                se3_predictor.visualizePrediction();
//                PressKeyToContinue("SE3 Prediction Vis ");

                RubberBand b = *sim_record.tested_.starting_state_.rubber_band_;
//                b.forwardPropagate(b.getEndpoints(), false);
                vis_->visualizePoints("TESTED_TRUE_RESULT",
                                      sim_record.tested_.ending_state_.rubber_band_->upsampleBand(),
                                      Visualizer::White(), 1, 0.002);

                transition_estimator_->estimateTransitions(
                            b,
                            sim_record.tested_.ending_gripper_positions_);
            }
            catch (const std::exception& ex)
            {
                ROS_ERROR_STREAM("Error parsing file: " << file << ": " << ex.what());
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Data Generation
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    static std::string ToString(const Eigen::Vector3d& mat)
    {
        std::stringstream ss;
        ss << mat.x() << "_" << mat.y() << "_" << mat.z() ;
        return ss.str();
    }

    static VectorVector3d Vec3dPerturbations(const double max_magnitude, const int num_divisions)
    {
        VectorVector3d perturbations;
        perturbations.reserve((size_t)(std::pow(2 * num_divisions + 1, 3) - 1));
        for (int x_idx = -num_divisions; x_idx <= num_divisions; ++x_idx)
        {
            const double x_delta = max_magnitude * x_idx / num_divisions;
            for (int y_idx = -num_divisions; y_idx <= num_divisions; ++y_idx)
            {
                const double y_delta = max_magnitude * y_idx / num_divisions;
                for (int z_idx = -num_divisions; z_idx <= num_divisions; ++z_idx)
                {
                    if (!(x_idx == 0 && y_idx == 0 && z_idx == 0))
                    {
                        const double z_delta = max_magnitude * z_idx / num_divisions;
                        perturbations.push_back(Vector3d(x_delta, y_delta, z_delta));
                    }
                }
            }
        }
        return perturbations;
    }

    TransitionTesting::DataGeneration::DataGeneration(
            const TransitionTesting& framework)
        : framework_(framework)
        , random_rotation_distribution_()
    {}

    void TransitionTesting::DataGeneration::generateTestData(
            const std::string& data_folder)
    {
        std::vector<deformable_manipulation_msgs::TransitionTest> tests;
        std::vector<std::string> filenames;

//        std::vector<deformable_manipulation_msgs::TransitionTestResult> results(tests.size());
        const auto feedback_callback = [&] (const size_t test_id, const deformable_manipulation_msgs::TransitionTestResult& result)
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
            Isometry3d gripper_a_ending_pose_ = framework_.gripper_a_starting_pose_ * Translation3d(framework_.gripper_a_action_vector_);
            Isometry3d gripper_b_ending_pose_ = framework_.gripper_b_starting_pose_ * Translation3d(framework_.gripper_b_action_vector_);
            const auto canonical_test = framework_.robot_->toRosTransitionTest(
                        framework_.initial_world_state_.rope_node_transforms_,
                        framework_.initial_world_state_.all_grippers_single_pose_,
                        generateTestPath({framework_.gripper_a_starting_pose_, framework_.gripper_b_starting_pose_}),
                        {gripper_a_ending_pose_, gripper_b_ending_pose_});
            tests.push_back(canonical_test);
            filenames.push_back("cannonical_straight_test");
        }

        //// Generate versions with perturbed gripper start positions //////////

        const auto gripper_positions_perturbations_max_magnitude = ROSHelpers::GetParamRequired<double>(*framework_.ph_, "perturbations/gripper_positions/max_magnitude", __func__).GetImmutable();
        const auto gripper_positions_perturbations_num_divisions = ROSHelpers::GetParamRequired<int>(*framework_.ph_, "perturbations/gripper_positions/num_divisions", __func__).GetImmutable();
        const auto gripper_positions_perturbations = Vec3dPerturbations(gripper_positions_perturbations_max_magnitude, gripper_positions_perturbations_num_divisions);
        std::cout << "Num total perturbations: " << gripper_positions_perturbations.size() << std::endl;
        const auto num_threads = GetNumOMPThreads();
        #pragma omp parallel for
        for (size_t a_idx = 0; a_idx < gripper_positions_perturbations.size(); ++a_idx)
        {
            Isometry3d gripper_a_starting_pose = framework_.gripper_a_starting_pose_ * Translation3d(gripper_positions_perturbations[a_idx]);
            Isometry3d gripper_a_ending_pose = gripper_a_starting_pose * Translation3d(framework_.gripper_a_action_vector_);
            for (size_t b_idx = 0; b_idx < gripper_positions_perturbations.size(); ++b_idx)
            {
                try
                {
                    const Isometry3d gripper_b_starting_pose = framework_.gripper_b_starting_pose_ * Translation3d(gripper_positions_perturbations[b_idx]);
                    const Isometry3d gripper_b_ending_pose = gripper_b_starting_pose * Translation3d(framework_.gripper_b_action_vector_);

                    const std::string filename(data_folder +
                                               "/cannonical_straight_test_"
                                               "_perturbed_gripper_start_positions_"
                                               "_gripper_a_" + ToString(gripper_positions_perturbations[a_idx]) +
                                               "_gripper_b_" + ToString(gripper_positions_perturbations[b_idx]) +
                                               ".compressed");

                    const auto test = framework_.robot_->toRosTransitionTest(
                                framework_.initial_world_state_.rope_node_transforms_,
                                framework_.initial_world_state_.all_grippers_single_pose_,
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
                            framework_.robot_->generateTransitionData(tests, filenames, feedback_callback, false);
                            tests.clear();
                            filenames.clear();
                        }
                    }
                }
                catch (const std::runtime_error& ex)
                {
                    ROS_ERROR_STREAM_NAMED("data_generation", "Unable to plan with perturbation"
                                           << " a: " << gripper_positions_perturbations[a_idx].transpose()
                                           << " b: " << gripper_positions_perturbations[b_idx].transpose()
                                           << " Message: " << ex.what());
                }
            }
        }

        // Run an tests left over
        if (tests.size() != 0)
        {
            framework_.robot_->generateTransitionData(tests, filenames, feedback_callback, false);
            tests.clear();
            filenames.clear();
        }
    }

    AllGrippersPoseTrajectory TransitionTesting::DataGeneration::generateTestPath(
            const AllGrippersSinglePose& gripper_target_poses)
    {
        // Pass in all the config values that the RRT needs; for example goal bias, step size, etc.
        auto band_rrt = BandRRT(framework_.nh_,
                                framework_.ph_,
                                *framework_.world_params_,
                                framework_.planning_params_,
                                framework_.smoothing_params_,
                                framework_.task_params_,
                                framework_.vis_,
                                false);

        const auto gripper_config = RRTGrippersRepresentation(
                    framework_.initial_world_state_.all_grippers_single_pose_[0],
                    framework_.initial_world_state_.all_grippers_single_pose_[1]);

        RRTRobotRepresentation robot_config(6);
        robot_config.head<3>() = gripper_config.first.translation();
        robot_config.tail<3>() = gripper_config.second.translation();

        const auto rubber_band = RubberBand::BandFromWorldState(
                    framework_.initial_world_state_, *framework_.band_);

        const RRTNode start_config(
                    gripper_config,
                    robot_config,
                    rubber_band);

        const std::chrono::duration<double> time_limit(GetRRTTimeout(*framework_.ph_));

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

    void TransitionTesting::DataGeneration::generateRandomTest(
            std::mt19937_64 generator,
            const TransitionEstimation::StateTransition& trans)
    {
        static const double grippers_delta_random_max = 0.025;     // meters
        static const double translation_delta_random_max = 0.1;    // meters
        static const double rotation_delta_max = 0.1;              // radians
        const auto max_gripper_step_size = framework_.task_params_.max_gripper_step_size_;

        // Transform all the data by the random translation and rotation of the test frame
        bool valid = false;
        while (!valid)
        {
            // Overridden if something renders this invalid
            valid = true;

            // Transform the whole band + gripper motion by some random amount
            {
                const Translation3d band_translation_offset((Vector3d::Random() * translation_delta_random_max));
                Quaterniond band_rotation_offset = random_rotation_distribution_.GetQuaternion(generator);
                while (2* std::acos(band_rotation_offset.w()) > rotation_delta_max)
                {
                    band_rotation_offset = random_rotation_distribution_.GetQuaternion(generator);
                }
                const Isometry3d random_test_transform = band_translation_offset * band_rotation_offset;
                random_test_transform_applied_ =
                        framework_.experiment_center_of_rotation_
                        * random_test_transform
                        * framework_.experiment_center_of_rotation_.inverse();

                random_test_rope_nodes_start_ = TransformData(random_test_transform_applied_, trans.starting_state_.rope_node_transforms_);

                const auto template_gripper_poses = ExtractGripperPosesFromTransition(trans);
                random_test_starting_gripper_poses_ = TransformData(random_test_transform_applied_, template_gripper_poses.first);
                random_test_ending_gripper_poses_ = TransformData(random_test_transform_applied_, template_gripper_poses.first);

                if (framework_.sdf_->EstimateDistance3d(random_test_starting_gripper_poses_[0].translation()).first < framework_.task_params_.gripper_min_distance_to_obstacles_)
                {
                    valid = false;
                }
                if (framework_.sdf_->EstimateDistance3d(random_test_starting_gripper_poses_[1].translation()).first < framework_.task_params_.gripper_min_distance_to_obstacles_)
                {
                    valid = false;
                }
            }

            // Transform the target position of the grippers by some random amount
            {
                const auto delta =
                        Sub(ToGripperPositions(random_test_ending_gripper_poses_),
                            ToGripperPositions(random_test_starting_gripper_poses_));

                auto test_delta = delta;
                test_delta.first += Vector3d::Random() * grippers_delta_random_max;
                test_delta.second += Vector3d::Random() * grippers_delta_random_max;
                const double test_delta_norm = std::sqrt(test_delta.first.squaredNorm() + test_delta.second.squaredNorm());
                if (test_delta_norm > max_gripper_step_size)
                {
                    test_delta.first *= (max_gripper_step_size / test_delta_norm);
                    test_delta.second *= (max_gripper_step_size / test_delta_norm);
                }
                random_test_ending_gripper_poses_[0].translation() += test_delta.first;
                random_test_ending_gripper_poses_[1].translation() += test_delta.first;

                if (framework_.sdf_->EstimateDistance3d(random_test_ending_gripper_poses_[0].translation()).first < framework_.task_params_.gripper_min_distance_to_obstacles_)
                {
                    valid = false;
                }
                if (framework_.sdf_->EstimateDistance3d(random_test_ending_gripper_poses_[1].translation()).first < framework_.task_params_.gripper_min_distance_to_obstacles_)
                {
                    valid = false;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Prediction
////////////////////////////////////////////////////////////////////////////////

namespace smmap
{
    TransitionTesting::SE3Prediction::SE3Prediction(
            const TransitionTesting &framework)
        : prediction_valid_(false)
        , framework_(framework)
    {}

    std::map<std::string, std::vector<RubberBand>> TransitionTesting::SE3Prediction::predictAll(
            const TransitionEstimation::StateTransition& stored_trans,
            const RubberBand& test_band_start,
            const PairGripperPositions& ending_gripper_positions)
    {
        test_band_ = std::make_shared<RubberBand>(test_band_start);
        test_action_ = ending_gripper_positions;

        num_gripper_steps_ = stored_trans.microstep_state_history_.size() / 4;
        stored_bands_.clear();
        stored_bands_.reserve(stored_trans.microstep_state_history_.size() + 1);
        stored_bands_.push_back(*stored_trans.starting_state_.rubber_band_);
        for (const auto& state : stored_trans.microstep_state_history_)
        {
            RubberBand temp_band(test_band_start);
            assert(temp_band.resetBand(state));
            stored_bands_.push_back(temp_band);
        }

        warping_target_points_ = RubberBand::PointsFromBandAndGrippers(
                    test_band_start,
                    test_band_start.getEndpoints(),
                    ending_gripper_positions,
                    num_gripper_steps_);

        predictBasedOnExecutedBand(stored_trans);
        predictBasedOnPlannedBand(stored_trans);
        prediction_valid_ = true;
        return results_;
    }

    void TransitionTesting::SE3Prediction::predictBasedOnPlannedBand(
            const TransitionEstimation::StateTransition& stored_trans)
    {
        static const std::string transform_definition_name = "__PLANNED_BAND__";

        // Extract the best SE(3) transform that transforms the template points
        // (memorized data) into the target points (test data)
        warping_template_points_planned_ = RubberBand::PointsFromBandAndGrippers(
                    *stored_trans.starting_state_.planned_rubber_band_,
                    stored_trans.starting_gripper_positions_,
                    stored_trans.ending_gripper_positions_,
                    num_gripper_steps_);

        const Isometry3d transform =
                Isometry3d(umeyama(warping_template_points_planned_, warping_target_points_, false));
        template_planned_band_aligned_to_target_ = transform * warping_template_points_planned_;
        stored_bands_planned_aligned_to_target_ = transform * RubberBand::AggregateBandPoints(stored_bands_);

        // Apply the best SE(3) transform to the memorized stored bands
        // Project the transformed points out of collision and retighten
        std::vector<RubberBand> transformed_bands_from_stored_bands;
        transformed_bands_from_stored_bands.reserve(stored_bands_.size());
        for (size_t idx = 0; idx < stored_bands_.size(); ++idx)
        {
            // Transform the stored band into the test band space
            const auto& stored_band = stored_bands_[idx];
            const auto transformed_band = TransformData(transform, stored_band.getVectorRepresentation());
            // Move the endpoints to the line along the test action vector
            const double ratio = (double)idx / (double)(stored_bands_.size() - 1);
            const auto gripper_targets = Interpolate(test_band_->getEndpoints(), test_action_, ratio);
            const auto points_to_smooth = RubberBand::PointsFromBandPointsAndGripperTargets(transformed_band, gripper_targets, 1);

            RubberBand band(stored_band);
            assert(band.setPointsAndSmooth(points_to_smooth));
            transformed_bands_from_stored_bands.push_back(band);
        }
        results_[BASENAME + transform_definition_name + "STORED_BANDS"] = transformed_bands_from_stored_bands;
    }

    void TransitionTesting::SE3Prediction::predictBasedOnExecutedBand(
            const TransitionEstimation::StateTransition& stored_trans)
    {
        static const std::string transform_definition_name = "__EXECUTED_BAND__";

        // Extract the best SE(3) transform that transforms the template points
        // (memorized data) into the target points (test data)
        warping_template_points_executed_ = RubberBand::PointsFromBandAndGrippers(
                    *stored_trans.starting_state_.rubber_band_,
                    stored_trans.starting_gripper_positions_,
                    stored_trans.ending_gripper_positions_,
                    num_gripper_steps_);

        const Isometry3d transform =
                Isometry3d(umeyama(warping_template_points_executed_, warping_target_points_, false));
        template_executed_band_aligned_to_target_ = transform * warping_template_points_executed_;
        stored_bands_executed_aligned_to_target_ = transform * RubberBand::AggregateBandPoints(stored_bands_);

        // Apply the best SE(3) transform to the memorized stored bands
        // Project the transformed points out of collision and retighten
        std::vector<RubberBand> transformed_bands_from_stored_bands;
        transformed_bands_from_stored_bands.reserve(stored_bands_.size());
        for (size_t idx = 0; idx < stored_bands_.size(); ++idx)
        {
            // Transform the stored band into the test band space
            const auto& stored_band = stored_bands_[idx];
            const auto transformed_band = TransformData(transform, stored_band.getVectorRepresentation());
            // Move the endpoints to the line along the test action vector
            const double ratio = (double)idx / (double)(stored_bands_.size() - 1);
            const auto gripper_targets = Interpolate(test_band_->getEndpoints(), test_action_, ratio);
            const auto points_to_smooth = RubberBand::PointsFromBandPointsAndGripperTargets(transformed_band, gripper_targets, 1);

            RubberBand band(stored_band);
            assert(band.setPointsAndSmooth(points_to_smooth));
            transformed_bands_from_stored_bands.push_back(band);
        }
        results_[BASENAME + transform_definition_name + "STORED_BANDS"] = transformed_bands_from_stored_bands;
    }

    void TransitionTesting::SE3Prediction::visualizePrediction()
    {
        assert(prediction_valid_);


        framework_.vis_->visualizePoints("TEMPLATE_POINTS_EXECUTED",
                                         warping_template_points_executed_,
                                         Visualizer::Green(), 1, 0.002);

        framework_.vis_->visualizePoints("TEMPLATE_POINTS_PLANNED",
                                         warping_template_points_planned_,
                                         Visualizer::Green(), 1, 0.002);

        framework_.vis_->visualizePoints("TARGET_POINTS",
                                         warping_target_points_,
                                         Visualizer::Yellow(), 1, 0.002);

        framework_.vis_->visualizePoints("TEMPLATE_POINTS_EXECUTED_ALIGNED",
                                         template_executed_band_aligned_to_target_,
                                         Visualizer::Olive(), 1, 0.002);

        framework_.vis_->visualizePoints("TEMPLATE_POINTS_PLANNED_ALIGNED",
                                         template_planned_band_aligned_to_target_,
                                         Visualizer::Olive(), 1, 0.002);

        RubberBand::VisualizeBandSurface(framework_.vis_,
                                         stored_bands_,
                                         Visualizer::Blue(),
                                         Visualizer::Red(),
                                         "stored_bands", 1);


        RubberBand::VisualizeBandSurface(framework_.vis_,
                                         stored_bands_executed_aligned_to_target_,
                                         stored_bands_.size(),
                                         Visualizer::Silver(),
                                         Visualizer::White(),
                                         "stored_bands_EXECUTED_ALIGNED", 1);

        RubberBand::VisualizeBandSurface(framework_.vis_,
                                         stored_bands_executed_aligned_to_target_,
                                         stored_bands_.size(),
                                         Visualizer::Silver(),
                                         Visualizer::White(),
                                         "stored_bands_PLANNED_ALIGNED", 1);

        for (const auto& result : results_)
        {
            RubberBand::VisualizeBandSurface(framework_.vis_,
                                             result.second,
                                             Visualizer::Seafoam(),
                                             Visualizer::Orange(),
                                             result.first, 1);
        }
    }
}
