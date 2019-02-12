#include "smmap/transition_learning_data_generation.h"
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/thin_plate_spline.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <smmap_utilities/neighbours.h>
#include <boost/filesystem.hpp>
#include <deformable_manipulation_experiment_params/conversions.hpp>

using namespace smmap;
using namespace arc_utilities;
using namespace arc_helpers;
using namespace Eigen;
using namespace EigenHelpers;

////////////////////////////////////////////////////////////////////////////////
//          Random Helpers
////////////////////////////////////////////////////////////////////////////////

int GetNumberOfDigits(int i)
{
    if (i < 0)
    {
        i = -i;
    }
    return i > 0 ? (int)std::log10((double) i) + 1 : 1;
}

ObjectPointSet PointsFromBandAndGrippers(
        const RubberBand& band,
        const PairGripperPositions& grippers_start,
        const PairGripperPositions& grippers_end)
{
    const VectorVector3d& band_points = band.upsampleBand();
    ObjectPointSet result(3, band_points.size() + 4);
    for (size_t band_idx = 0; band_idx < band_points.size(); ++band_idx)
    {
        result.col(band_idx) = band_points[band_idx];
    }
    result.col(band_points.size() + 0) = grippers_start.first;
    result.col(band_points.size() + 1) = grippers_start.second;
    result.col(band_points.size() + 2) = grippers_end.first;
    result.col(band_points.size() + 3) = grippers_end.second;
    return result;
}

std::pair<RubberBand::Ptr,
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

VectorVector3d ExtractPositionsFromPoses(const VectorIsometry3d& poses)
{
    VectorVector3d positions(poses.size());
    for (size_t idx = 0; idx < poses.size(); ++idx)
    {
        positions[idx] = poses[idx].translation();
    }
    return positions;
}

ObjectPointSet ExtractPointSetFromPoses(const VectorIsometry3d& poses)
{
    ObjectPointSet pointset(3, poses.size());
    for (size_t idx = 0; idx < poses.size(); ++idx)
    {
        pointset.col(idx) = poses[idx].translation();
    }
    return pointset;
}

std::pair<AllGrippersSinglePose, AllGrippersSinglePose> ExtractGripperPosesFromTransition(
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

RubberBand::Ptr BandFromNodeTransformsAndGrippers(
        const VectorIsometry3d& node_transforms,
        const PairGripperPositions& grippers_position,
        const RubberBand& template_band)
{
    auto band = std::make_shared<RubberBand>(template_band);
    band->resetBand(ExtractPointSetFromPoses(node_transforms),
                    grippers_position);
    return band;
}

RubberBand::Ptr BandFromWorldState(
        const WorldState& world_state,
        const RubberBand& template_band)
{
    auto band = std::make_shared<RubberBand>(template_band);
    band->resetBand(world_state);
    return band;
}

void VisualizeBandSurface(
        const Visualizer::ConstPtr& vis,
        const ObjectPointSet& band_surface,
        const size_t points_per_band,
        const std_msgs::ColorRGBA& start_color,
        const std_msgs::ColorRGBA& end_color,
        const std::string& ns,
        const int32_t id = 1)
{
    assert(points_per_band > 0);
    assert(band_surface.cols() % points_per_band == 0);
    const size_t num_bands = band_surface.cols() / points_per_band;
    assert(num_bands > 1);

    std::vector<std_msgs::ColorRGBA> colors;
    for (size_t band_idx = 0; band_idx < num_bands; ++band_idx)
    {
        const float ratio = (float)(band_idx) / (float)(num_bands - 1);
        const auto color = InterpolateColor(start_color, end_color, ratio);
        colors.insert(colors.end(), points_per_band, color);
    }
    vis->visualizePoints(ns, band_surface, colors, id, 0.002);
}

void VisualizeBandSurface(
        const Visualizer::ConstPtr& vis,
        const std::vector<RubberBand>& band_surface,
        const std_msgs::ColorRGBA& start_color,
        const std_msgs::ColorRGBA& end_color,
        const std::string& ns,
        const int32_t id = 1)
{
    if (band_surface.empty())
    {
        return;
    }
    const auto points_per_band = band_surface[0].upsampleBand().size();
    const auto num_bands = band_surface.size();
    ObjectPointSet points(3, num_bands * points_per_band);
    for (size_t band_idx = 0; band_idx < num_bands; ++band_idx)
    {
        points.block(0, band_idx * points_per_band, 3, points_per_band)
                = EigenHelpersConversions::VectorEigenVector3dToEigenMatrix3Xd(
                    band_surface[band_idx].upsampleBand());
    }
    VisualizeBandSurface(vis, points, points_per_band, start_color, end_color, ns, id);
}

VectorVector3d TransformData(
        const Eigen::Isometry3d& transform,
        const VectorVector3d& data)
{
    VectorVector3d retval;
    retval.reserve(data.size());
    for (const auto& vec : data)
    {
        retval.push_back(transform * vec);
    }
    return retval;
}

VectorIsometry3d TransformData(
        const Eigen::Isometry3d& transform,
        const VectorIsometry3d& data)
{
    VectorIsometry3d retval;
    retval.reserve(data.size());
    for (const auto& pose : data)
    {
        retval.push_back(transform * pose);
    }
    return retval;
}

std::vector<CollisionData> TransformData(
        const Eigen::Isometry3d& transform,
        const std::vector<CollisionData>& data)
{
    std::vector<CollisionData> retval;
    retval.reserve(data.size());
    for (const auto& collision_data : data)
    {
        retval.push_back(CollisionData(
                             transform * collision_data.nearest_point_to_obstacle_,
                             transform * collision_data.obstacle_surface_normal_,
                             collision_data.distance_to_obstacle_));
    }
    return retval;
}

WorldState TransformWorldState(
        const Eigen::Isometry3d& transform,
        const WorldState& state)
{
    WorldState new_state;
    new_state.object_configuration_ = transform * state.object_configuration_;
    new_state.rope_node_transforms_ = TransformData(transform, state.rope_node_transforms_);
    new_state.all_grippers_single_pose_ = TransformData(transform, state.all_grippers_single_pose_);
    new_state.robot_configuration_ = state.robot_configuration_;
    new_state.robot_configuration_valid_ = state.robot_configuration_valid_;
    new_state.gripper_collision_data_ = TransformData(transform, state.gripper_collision_data_);
    new_state.sim_time_ = state.sim_time_;
    return new_state;
}

////////////////////////////////////////////////////////////////////////////////
//          Transition Test Results
////////////////////////////////////////////////////////////////////////////////


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

void TransitionSimulationRecord::visualize(const Visualizer::ConstPtr& vis) const
{
    const size_t points_per_band = template_.starting_state_.rubber_band_->upsampleBand().size();
    TransitionEstimation::VisualizeTransition(vis, template_, 1, "template_");
    TransitionEstimation::VisualizeTransition(vis, tested_, 1, "tested_");
    vis->visualizeAxes("center_of_rotation", center_of_rotation_, 0.1, 0.005, 1);
    VisualizeBandSurface(vis, template_band_surface_,   points_per_band, Visualizer::Blue(),  Visualizer::Red(),     "band_surface_template",   1);
    VisualizeBandSurface(vis, tested_band_surface_,     points_per_band, Visualizer::Cyan(),  Visualizer::Magenta(), "band_surface_tested",     1);
}


////////////////////////////////////////////////////////////////////////////////
//          Transition Testing
////////////////////////////////////////////////////////////////////////////////


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
    , gripper_min_distance_to_obstacles_(GetRRTMinGripperDistanceToObstacles(*ph_))
    , experiment_center_of_rotation_(calculateExperimentCenterOfRotation())

    , deformable_type_(GetDeformableType(*nh_))
    , task_type_(GetTaskType(*nh_))
    , data_folder_(ROSHelpers::GetParam<std::string>(*ph_, "data_folder", "/tmp/transition_learning_data_generation"))
    , sim_test_result_suffix_("_results.compressed")
    , prediction_result_suffix_("_prediction.compressed")
{
    std::srand((unsigned int)seed_);
    initialize(robot_->start());
    vis_->visualizeAxes("center_of_rotation", experiment_center_of_rotation_, 0.1, 0.005, 1);
}

Isometry3d TransitionTesting::calculateExperimentCenterOfRotation()
{
    using namespace ROSHelpers;

    const Vector3d world_min(
                GetWorldXMinBulletFrame(*nh_),
                GetWorldYMinBulletFrame(*nh_),
                GetWorldZMinBulletFrame(*nh_));

    const Vector3d world_max(
                GetWorldXMaxBulletFrame(*nh_),
                GetWorldYMaxBulletFrame(*nh_),
                GetWorldZMaxBulletFrame(*nh_));

//    const double wall_thickness = GetWorldResolution(*nh_) * 4.0;
    const Vector3d world_center = (world_max + world_min) / 2.0;
//    const Vector3d world_size = world_max - world_min;

    const double task_progress_wall_width =         GetParamRequired<double>(*nh_, "task_progress_wall_width", __func__).GetImmutable();
    const double task_progress_wall_x_com =         GetParamRequired<double>(*nh_, "task_progress_wall_x_com", __func__).GetImmutable();
    const double gripper_separator_lower_height =   GetParamRequired<double>(*nh_, "gripper_separator_lower_height", __func__).GetImmutable();

    double hook_length =                            GetParamRequired<double>(*nh_, "hook_length", __func__).GetImmutable();
    double hook_radius =                            GetParamRequired<double>(*nh_, "hook_radius", __func__).GetImmutable();
    double hook_com_offset_y =                      GetParamRequired<double>(*nh_, "hook_com_offset_y", __func__).GetImmutable();

    const Vector3d hook_half_extents(
                hook_length / 2.0f,
                hook_radius,
                hook_radius);

    const Vector3d hook_com(
                task_progress_wall_x_com - task_progress_wall_width / 2.0f - hook_half_extents.x(),
                world_center.y() + hook_com_offset_y,
                world_min.z() + gripper_separator_lower_height - hook_radius);

    return Isometry3d(Translation3d(
                hook_com.x() - hook_half_extents.x() + work_space_grid_.minStepDimension() / 4.0,
                hook_com.y(),
                hook_com.z()));
}

void TransitionTesting::initialize(const WorldState& world_state)
{
    initializeBand(world_state);
    transition_estimator_ = std::make_shared<TransitionEstimation>(
                nh_, ph_, sdf_, work_space_grid_, vis_, *band_);
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

    // Find the shortest path through the object, between the grippers, while follow nodes of the object.
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
        throw_arc_exception(std::invalid_argument,
                            "Buffer size mismatch: " + filename);
    }
    return record.first;
}

void TransitionTesting::runTests(const bool generate_new_test_data)
{
    if (generate_new_test_data)
    {
        DataGeneration data_generator(*this);
        data_generator.generateTestData(*generator_, data_folder_);
    }

    const auto files = getDataFileList();
    for (const auto& file : files)
    {
        try
        {
            const auto sim_record = loadSimRecord(file);

            SE3Prediction se3_predictor(*this);
            se3_predictor.predictAll(
                        sim_record.template_,
                        *sim_record.tested_.starting_state_.rubber_band_,
                        sim_record.tested_.ending_gripper_positions_);
            se3_predictor.visualizePrediction();
            std::cout << "Press any key to continue " << std::flush;
            auto key = arc_helpers::GetChar();
            if (key != '\n')
            {
                std::cout << std::endl;
            }
        }
        catch (const std::exception& ex)
        {
            ROS_ERROR_STREAM("Error parsing file: " << file << ": " << ex.what());
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Data Generation
////////////////////////////////////////////////////////////////////////////////

TransitionTesting::DataGeneration::DataGeneration(
        const TransitionTesting& framework)
    : framework_(framework)
    , random_rotation_distribution_()
{}

void TransitionTesting::DataGeneration::generateTestData(
        std::mt19937_64& generator,
        const std::string& data_folder)
{
    const auto& transitions = framework_.transition_estimator_->transitions();

    ROS_INFO_STREAM("Visualizing " << transitions.size() << " transitions");
    for (size_t idx = 0; idx < transitions.size(); ++idx)
    {
        const TransitionEstimation::StateTransition& trans = transitions[idx];
        const size_t points_per_band = trans.starting_state_.rubber_band_->upsampleBand().size();

        auto bands = framework_.transition_estimator_->reduceMicrostepsToBands(
                    trans.microstep_state_history_);
        bands.insert(bands.begin(), trans.starting_state_.rubber_band_);
        const auto band_surface = RubberBand::AggregateBandPoints(ToConstPtr(bands));

        VisualizeBandSurface(framework_.vis_, band_surface, points_per_band, Visualizer::Blue(), Visualizer::Red(), "band_surface_template", 1);
        framework_.transition_estimator_->visualizeTransition(trans, 1, "transition_testing_");

        // Test transitions with random changes
        {
            const int num_random_tests = 500;
            for (int i = 0; i < num_random_tests; ++i)
            {
                // Stores the test parameters in class variables
                generateRandomTest(generator, trans);

                ////// Gather simulated results ////////////////////////////////////////////////////////////////////////

                const std::pair<WorldState, std::vector<WorldState>> test_result =
                        framework_.robot_->testRobotMotionMicrosteps(
                            random_test_rope_nodes_start_,
                            random_test_starting_gripper_poses_,
                            random_test_ending_gripper_poses_,
                            (int)trans.microstep_state_history_.size() / 4);
                const auto& start_after_settling = test_result.first;
                const auto& microsteps = test_result.second;

                // Rejection sampling
                if (microsteps.empty())
                {
                    --i;
                    ROS_INFO("Rejecting sample, band starts overstretched");
                    continue;
                }

                const auto random_test_band_start = BandFromWorldState(
                            start_after_settling,
                            *trans.starting_state_.rubber_band_);
                // More rejection sampling
                if (random_test_band_start->isOverstretched())
                {
                    --i;
                    ROS_INFO("Rejecting sample, rope starts in collision");
                    continue;
                }
                auto test_bands = framework_.transition_estimator_->reduceMicrostepsToBands(microsteps);
                test_bands.insert(test_bands.begin(), random_test_band_start);
                const auto test_band_surface = RubberBand::AggregateBandPoints(ToConstPtr(test_bands));

                // Add each band with a different color, ranging from cyan (early in the history) to magenta (late in the history)
                VisualizeBandSurface(framework_.vis_, test_band_surface, points_per_band, Visualizer::Cyan(), Visualizer::Magenta(), "band_surface_simulation", 1);

                ////// Compile the results into a single structure and save to file ////////////////////////////////////
                {
                    TransitionSimulationRecord transition_test_results;
                    transition_test_results.template_ = trans;
                    transition_test_results.template_band_surface_ = band_surface;

                    transition_test_results.center_of_rotation_ = framework_.experiment_center_of_rotation_;
                    transition_test_results.transform_applied_ = random_test_transform_applied_;

                    transition_test_results.tested_.starting_state_                     = trans.starting_state_;
                    transition_test_results.tested_.starting_gripper_positions_         = trans.starting_gripper_positions_;
                    transition_test_results.tested_.ending_state_.deform_config_        = microsteps.back().object_configuration_;
                    transition_test_results.tested_.ending_state_.rubber_band_          = std::make_shared<RubberBand>(*test_bands.back());
                    transition_test_results.tested_.ending_state_.planned_rubber_band_  = std::make_shared<RubberBand>(*test_bands.back());
                    transition_test_results.tested_.ending_state_.rope_node_transforms_ = microsteps.back().rope_node_transforms_;
                    transition_test_results.tested_.ending_gripper_positions_           = ToGripperPositions(random_test_starting_gripper_poses_);
                    transition_test_results.tested_.microstep_state_history_            = microsteps;

                    transition_test_results.tested_band_surface_ = test_band_surface;

                    const int trans_idx_width = GetNumberOfDigits(static_cast<int>(transitions.size()));
                    const int test_idx_width = GetNumberOfDigits(num_random_tests);
                    std::vector<uint8_t> buffer;
                    transition_test_results.serializeSelf(buffer);
                    std::stringstream path;
                    path << data_folder
                         << "/trans_" << std::setfill('0') << std::setw(trans_idx_width) << idx
                         << "_random_test_" << std::setfill('0') << std::setw(test_idx_width) << i
                         << framework_.sim_test_result_suffix_;
                    ZlibHelpers::CompressAndWriteToFile(buffer, path.str());
                }
            }
        }
    }
}

void TransitionTesting::DataGeneration::generateRandomTest(
        std::mt19937_64& generator,
        const TransitionEstimation::StateTransition& trans)
{
    static const double grippers_delta_random_max = 0.025;     // meters
    static const double translation_delta_random_max = 0.1;    // meters
    static const double rotation_delta_max = 0.1;              // radians
    const auto max_gripper_step_size = framework_.work_space_grid_.minStepDimension();

    // Transform all the data by the random translation and rotation of the test frame
    bool valid = false;
    while (!valid)
    {
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

            random_test_rope_nodes_start_.resize(trans.starting_state_.rope_node_transforms_.size());
            for (size_t idx = 0; idx < random_test_rope_nodes_start_.size(); ++ idx)
            {
                random_test_rope_nodes_start_[idx] = random_test_transform_applied_ * trans.starting_state_.rope_node_transforms_[idx];
            }
            const auto template_gripper_poses = ExtractGripperPosesFromTransition(trans);

            random_test_starting_gripper_poses_.resize(2);
            random_test_starting_gripper_poses_[0] = random_test_transform_applied_ * template_gripper_poses.first[0];
            random_test_starting_gripper_poses_[1] = random_test_transform_applied_ * template_gripper_poses.first[1];
            random_test_ending_gripper_poses_.resize(2);
            random_test_ending_gripper_poses_[0]   = random_test_transform_applied_ * template_gripper_poses.second[0];
            random_test_ending_gripper_poses_[1]   = random_test_transform_applied_ * template_gripper_poses.second[1];

            valid = true;
            if (framework_.sdf_->EstimateDistance3d(random_test_starting_gripper_poses_[0].translation()).first < framework_.gripper_min_distance_to_obstacles_)
            {
                valid = false;
            }
            if (framework_.sdf_->EstimateDistance3d(random_test_starting_gripper_poses_[1].translation()).first < framework_.gripper_min_distance_to_obstacles_)
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

            if (framework_.sdf_->EstimateDistance3d(random_test_ending_gripper_poses_[0].translation()).first < framework_.gripper_min_distance_to_obstacles_)
            {
                valid = false;
            }
            if (framework_.sdf_->EstimateDistance3d(random_test_ending_gripper_poses_[1].translation()).first < framework_.gripper_min_distance_to_obstacles_)
            {
                valid = false;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//          Prediction
////////////////////////////////////////////////////////////////////////////////

TransitionTesting::SE3Prediction::SE3Prediction(
        const TransitionTesting &framework)
    : prediction_valid_(false)
    , framework_(framework)
{}

std::map<std::string, std::vector<RubberBand>> TransitionTesting::SE3Prediction::predictAll(
        const TransitionEstimation::StateTransition& stored_trans,
        const RubberBand& band,
        const PairGripperPositions& action)
{
    stored_bands_.clear();
    stored_bands_.reserve(stored_trans.microstep_state_history_.size() + 1);
    stored_bands_.push_back(band);
    for (const auto& state : stored_trans.microstep_state_history_)
    {
        RubberBand temp_band(band);
        temp_band.resetBand(state);
        stored_bands_.push_back(temp_band);
    }
    VisualizeBandSurface(framework_.vis_,
                         stored_bands_,
                         Visualizer::Blue(),
                         Visualizer::Red(),
                         "stored_bands",
                         1);

    warping_target_points_ = PointsFromBandAndGrippers(
                band,
                band.getEndpoints(),
                action);
    framework_.vis_->visualizePoints("TARGET_POINTS",
                                     warping_target_points_,
                                     Visualizer::Yellow(), 1, 0.002);

    arc_helpers::InteractiveWaitToContinue();

    predictBasedOnPlannedBand(stored_trans);
    predictBasedOnExecutedBand(stored_trans);
    prediction_valid_ = true;
    return results_;
}

void TransitionTesting::SE3Prediction::predictBasedOnPlannedBand(
        const TransitionEstimation::StateTransition& stored_trans)
{
    const std::string transform_definition_name = "__PLANNED_BAND__";



    warping_template_points_planned_ = PointsFromBandAndGrippers(
                *stored_trans.starting_state_.planned_rubber_band_,
                stored_trans.starting_gripper_positions_,
                stored_trans.ending_gripper_positions_);


    const Isometry3d transform =
            Isometry3d(Eigen::umeyama(warping_template_points_planned_, warping_target_points_, false));
    template_planned_band_aligned_to_target_ = transform * warping_template_points_planned_;



    framework_.vis_->visualizePoints("TEMPLATE_POINTS_PLANNED",
                                     warping_template_points_planned_,
                                     Visualizer::Green(), 1, 0.002);
    framework_.vis_->visualizePoints("TEMPLATE_POINTS_PLANNED_ALIGNED",
                                     template_planned_band_aligned_to_target_,
                                     Visualizer::Olive(), 1, 0.002);

    arc_helpers::InteractiveWaitToContinue();


    std::vector<RubberBand> transformed_bands_from_stored_world_state;
    transformed_bands_from_stored_world_state.reserve(stored_trans.microstep_state_history_.size() + 1);
    for (const auto state : stored_trans.microstep_state_history_)
    {
        RubberBand band(stored_bands_[0]);
        band.resetBand(TransformWorldState(transform, state));
        transformed_bands_from_stored_world_state.push_back(band);
    }
    results_[BASENAME + transform_definition_name + "WORLD_STATES"] = transformed_bands_from_stored_world_state;

    std::vector<RubberBand> transformed_bands_from_stored_bands;
    transformed_bands_from_stored_bands.reserve(stored_trans.microstep_state_history_.size() + 1);
    for (const auto stored_band : stored_bands_)
    {
        RubberBand band(stored_band);
        band.setPointsAndSmooth(TransformData(transform, stored_band.upsampleBand()));
        transformed_bands_from_stored_bands.push_back(band);
    }
    results_[BASENAME + transform_definition_name + "STORED_BANDS"] = transformed_bands_from_stored_bands;
}

void TransitionTesting::SE3Prediction::predictBasedOnExecutedBand(
        const TransitionEstimation::StateTransition& stored_trans)
{
    const std::string transform_definition_name = "__EXECUTED_BAND__";

    warping_template_points_executed_ = PointsFromBandAndGrippers(
                *stored_trans.starting_state_.rubber_band_,
                stored_trans.starting_gripper_positions_,
                stored_trans.ending_gripper_positions_);

    const Isometry3d transform =
            Isometry3d(Eigen::umeyama(warping_template_points_executed_, warping_target_points_, false));
    template_executed_band_aligned_to_target_ = transform * warping_template_points_executed_;


    std::vector<RubberBand> transformed_bands_from_stored_world_state;
    transformed_bands_from_stored_world_state.reserve(stored_trans.microstep_state_history_.size() + 1);
    for (const auto state : stored_trans.microstep_state_history_)
    {
        RubberBand band(stored_bands_[0]);
        band.resetBand(TransformWorldState(transform, state));
        transformed_bands_from_stored_world_state.push_back(band);
    }
    results_[BASENAME + transform_definition_name + "WORLD_STATES"] = transformed_bands_from_stored_world_state;

    std::vector<RubberBand> transformed_bands_from_stored_bands;
    transformed_bands_from_stored_bands.reserve(stored_trans.microstep_state_history_.size() + 1);
    for (const auto stored_band : stored_bands_)
    {
        RubberBand band(stored_band);
        band.setPointsAndSmooth(TransformData(transform, stored_band.upsampleBand()));
        transformed_bands_from_stored_bands.push_back(band);
    }
    results_[BASENAME + transform_definition_name + "STORED_BANDS"] = transformed_bands_from_stored_bands;
}

void TransitionTesting::SE3Prediction::visualizePrediction()
{
    assert(prediction_valid_);

    VisualizeBandSurface(framework_.vis_,
                         stored_bands_,
                         Visualizer::Blue(),
                         Visualizer::Red(),
                         "stored_bands",
                         1);

    framework_.vis_->visualizePoints("TARGET_POINTS",
                                     warping_template_points_planned_,
                                     Visualizer::Yellow(), 1, 0.002);

    framework_.vis_->visualizePoints("TEMPLATE_POINTS_PLANNED",
                                     warping_template_points_planned_,
                                     Visualizer::Green(), 1, 0.002);

    framework_.vis_->visualizePoints("TEMPLATE_POINTS_PLANNED_ALIGNED",
                                     template_planned_band_aligned_to_target_,
                                     Visualizer::Olive(), 1, 0.002);

    framework_.vis_->visualizePoints("TEMPLATE_POINTS_EXECUTED",
                                     warping_template_points_executed_,
                                     Visualizer::White(), 1, 0.002);

    framework_.vis_->visualizePoints("TEMPLATE_POINTS_EXECUTED_ALIGNED",
                                     template_executed_band_aligned_to_target_,
                                     Visualizer::Silver(), 1, 0.002);

    for (const auto& result : results_)
    {
        VisualizeBandSurface(framework_.vis_,
                             result.second,
                             Visualizer::Cyan(),
                             Visualizer::Orange(),
                             result.first,
                             1);
    }
}

