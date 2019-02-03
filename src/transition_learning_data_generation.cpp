#include "smmap/transition_learning_data_generation.h"
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/thin_plate_spline.hpp>
#include <smmap_utilities/neighbours.h>

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;
using namespace arc_helpers;
using namespace Eigen;
using namespace EigenHelpers;

ObjectPointSet TpsPointsFromBandAndGrippers(
        const RubberBand::ConstPtr& band,
        const TransitionEstimation::GripperPositions& grippers_start,
        const TransitionEstimation::GripperPositions& grippers_end)
{
    const VectorVector3d band_points = band->upsampleBand();
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
          std::pair<TransitionEstimation::GripperPositions,
                    TransitionEstimation::GripperPositions>> BandAndGrippersFromTpsPoints(
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
    const TransitionEstimation::GripperPositions grippers_start =
    {
        tps_points.col(num_band_points + 0),
        tps_points.col(num_band_points + 1)
    };
    const TransitionEstimation::GripperPositions grippers_end =
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

RubberBand::Ptr BandFromNodeTransformsAndGrippers(
        const VectorIsometry3d& node_transforms,
        const TransitionEstimation::GripperPositions& grippers_position,
        const RubberBand& template_band)
{
    auto band = std::make_shared<RubberBand>(template_band);
    band->resetBand(ExtractPointSetFromPoses(node_transforms),
                    grippers_position.first,
                    grippers_position.second);
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

TransitionEstimation::GripperPositions ToGripperPositions(const VectorIsometry3d& poses)
{
    assert(poses.size() == 2);
    return {poses[0].translation(), poses[1].translation()};
}

DataGeneration::DataGeneration(
        std::shared_ptr<ros::NodeHandle> nh,
        std::shared_ptr<ros::NodeHandle> ph,
        RobotInterface::Ptr robot,
        const Visualizer::ConstPtr& vis)
    : nh_(nh)
    , ph_(ph)
    , robot_(robot)
    , vis_(vis)
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

    , deformable_type_(GetDeformableType(*nh))
    , task_type_(GetTaskType(*nh))
{
    initialize(robot_->start());
}

void DataGeneration::initialize(const WorldState& world_state)
{
    initializeBand(world_state);
    transition_estimator_ = std::make_shared<TransitionEstimation>(
                nh_, ph_, sdf_, work_space_grid_, vis_, *band_);
}

void DataGeneration::initializeBand(const WorldState& world_state)
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

void DataGeneration::runTests()
{
    const auto& transitions = transition_estimator_->transitions();

    ROS_INFO_STREAM("Visualizing " << transitions.size() << " transitions");
    for (size_t idx = 0; idx < transitions.size(); ++idx)
    {
        const TransitionEstimation::StateTransition& trans = transitions[idx];

        auto bands = transition_estimator_->reduceMicrostepsToBands(
                    trans.microstep_state_history_);
        bands.insert(bands.begin(), trans.starting_state_.rubber_band_);
        const auto band_surface = RubberBand::AggregateBandPoints(bands);

        const auto tps_template_points = TpsPointsFromBandAndGrippers(
                    trans.starting_state_.rubber_band_,
                    trans.starting_gripper_positions_,
                    trans.ending_gripper_positions_);

        // Visualization
        {
            // Add each band with a different color, ranging from
            // blue (early in the history) to red (late in the history)
            std::vector<std_msgs::ColorRGBA> colors;
            for (size_t band_idx = 0; band_idx < bands.size(); ++band_idx)
            {
                const float ratio = (float)(band_idx) / (float)(bands.size() - 1) ;
                const auto color = InterpolateColor(Visualizer::Blue(), Visualizer::Red(), ratio);
                colors.insert(colors.end(), bands[band_idx]->upsampleBand().size(), color);
            }

            vis_->visualizePoints("band_surface", band_surface, colors, 1, 0.002);
            transition_estimator_->visualizeTransition(trans, 1, "transition_testing_");
        }

        AllGrippersSinglePose starting_gripper_poses(2);
        starting_gripper_poses[0] = trans.starting_state_.rope_node_transforms_.front();
        starting_gripper_poses[0].translation() = trans.starting_gripper_positions_.first;
        starting_gripper_poses[1] = trans.starting_state_.rope_node_transforms_.back();
        starting_gripper_poses[1].translation() = trans.starting_gripper_positions_.second;
        AllGrippersSinglePose target_gripper_poses = starting_gripper_poses;
        target_gripper_poses[0].translation() = trans.ending_gripper_positions_.first;
        target_gripper_poses[1].translation() = trans.ending_gripper_positions_.second;

        const auto delta =
                Sub(trans.ending_gripper_positions_,
                    trans.starting_gripper_positions_);
        const double delta_norm = std::sqrt(delta.first.squaredNorm() + delta.second.squaredNorm());
        const auto max_gripper_step_size = work_space_grid_.minStepDimension();

        // Test transitions with random changes
        {
            const int num_random_tests = 100;
            const double random_max = 0.025;
            for (int i = 0; i < num_random_tests; ++i)
            {
                ////// Generate a random test based on the recorded transition /////////////////////////////////////////

                auto test_delta = delta;
                test_delta.first += Vector3d::Random() * random_max;
                test_delta.second += Vector3d::Random() * random_max;
                const double test_delta_norm = std::sqrt(test_delta.first.squaredNorm() + test_delta.second.squaredNorm());
                if (test_delta_norm > max_gripper_step_size)
                {
                    test_delta.first *= (max_gripper_step_size / test_delta_norm);
                    test_delta.second *= (max_gripper_step_size / test_delta_norm);
                }

                AllGrippersSinglePose random_test_ending_gripper_poses = starting_gripper_poses;
                random_test_ending_gripper_poses[0].translation() += test_delta.first;
                random_test_ending_gripper_poses[1].translation() += test_delta.first;

                // Specify the starting configuration, and gripper movement
                const auto random_test_rope_nodes_start             = trans.starting_state_.rope_node_transforms_;
                const auto random_test_starting_gripper_poses       = starting_gripper_poses;
                const auto random_test_starting_grippers_positions  = ToGripperPositions(starting_gripper_poses);
                const auto random_test_ending_grippers_positions    = ToGripperPositions(random_test_ending_gripper_poses);

                ////// Gather simulated results ////////////////////////////////////////////////////////////////////////

                const std::pair<WorldState, std::vector<WorldState>> test_result =
                        robot_->testRobotMotionMicrosteps(
                            random_test_rope_nodes_start,
                            random_test_starting_gripper_poses,
                            random_test_ending_gripper_poses,
                            (int)trans.microstep_state_history_.size() / 4);
                const auto& start_after_settling = test_result.first;
                const auto& microsteps = test_result.second;

                const auto random_test_band_start = BandFromWorldState(
                            start_after_settling,
                            *trans.starting_state_.rubber_band_);
                auto test_bands = transition_estimator_->reduceMicrostepsToBands(microsteps);
                test_bands.insert(test_bands.begin(), random_test_band_start);
                const auto test_band_surface = RubberBand::AggregateBandPoints(test_bands);

                ////// Gather tps predicted results ////////////////////////////////////////////////////////////////////

                const auto tps_target_points = TpsPointsFromBandAndGrippers(
                            random_test_band_start,
                            random_test_starting_grippers_positions,
                            random_test_ending_grippers_positions);

                vis_->visualizePoints("tps_template", tps_template_points, Visualizer::Red(), 1, 0.002);
                vis_->visualizePoints("tps_target", tps_target_points, Visualizer::Blue(), 1, 0.002);

                ThinPlateSpline<3> tps_warp(tps_template_points, tps_target_points);
                const auto tps_band_surface_prediction = tps_warp.interpolate(band_surface);

                // Visualization
                {
                    // Add each band with a different color, ranging from
                    // green (early in the history) to orange (late in the history)
                    std::vector<std_msgs::ColorRGBA> colors;
                    for (size_t band_idx = 0; band_idx < test_bands.size(); ++band_idx)
                    {
                        const float ratio = (float)(band_idx) / (float)(test_bands.size() - 1) ;
                        const auto color = InterpolateColor(Visualizer::Green(), Visualizer::Orange(), ratio);
                        colors.insert(colors.end(), test_bands[band_idx]->upsampleBand().size(), color);
                    }

                    vis_->visualizePoints("band_surface_tps_prediction", tps_band_surface_prediction, colors, 1, 0.002);
                }

                ////// Compile the results into a single structure /////////////////////////////////////////////////////

                TransitionTestResults transition_test_results;
                transition_test_results.template_ = trans;
                transition_test_results.tps_control_points_ = TpsPointsFromBandAndGrippers(
                            trans.starting_state_.rubber_band_,
                            trans.starting_gripper_positions_,
                            trans.ending_gripper_positions_);
                transition_test_results.template_band_surface_ = band_surface;

                transition_test_results.tested_.starting_state_ = trans.starting_state_;
                                transition_test_results.tested_.starting_gripper_positions_ = trans.starting_gripper_positions_;
                transition_test_results.tested_.ending_state_.deform_config_ = microsteps.back().object_configuration_;
                transition_test_results.tested_.ending_state_.rubber_band_ = test_bands.back();
                transition_test_results.tested_.ending_state_.planned_rubber_band_ = std::make_shared<RubberBand>(*test_bands.back());
                transition_test_results.tested_.ending_state_.rope_node_transforms_ = microsteps.back().rope_node_transforms_;
                transition_test_results.tested_.ending_gripper_positions_ = ToGripperPositions(random_test_starting_gripper_poses);
                transition_test_results.tested_.microstep_state_history_ = microsteps;

                transition_test_results.predicted_final_band_surface_ = tps_band_surface_prediction;
                transition_test_results.final_band_surface_ = test_band_surface;

                // Visualization
                {
                    // Add each band with a different color, ranging from
                    // cyan (early in the history) to magenta (late in the history)
                    std::vector<std_msgs::ColorRGBA> colors;
                    for (size_t band_idx = 0; band_idx < test_bands.size(); ++band_idx)
                    {
                        const float ratio = (float)(band_idx) / (float)(test_bands.size() - 1) ;
                        const auto color = InterpolateColor(Visualizer::Cyan(), Visualizer::Magenta(), ratio);
                        colors.insert(colors.end(), test_bands[band_idx]->upsampleBand().size(), color);
                    }

                    vis_->visualizePoints("band_surface_test", test_band_surface, colors, 1, 0.002);
                }

//                std::cout << "    Press any key to continue " << std::flush;
//                GetChar();
//                std::cout << std::endl;
            }
        }



//        std::cout << "Number of bands: " << trans.microstep_state_history_.size() + 1
//                  << "  Gripper endpoint distances: " << delta.first.norm() << "  " << delta.second.norm()
//                  << "  Net norm: " << delta_norm << "  Max norm:  " << max_gripper_step_size
//                  << "    Press any key to continue " << std::flush;
//        GetChar();
//        std::cout << std::endl;

    }
}

void DataGeneration::visualizeDeformableObject(
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    switch (deformable_type_)
    {
        case ROPE:
            vis_->visualizeRope(marker_name, object_configuration, color, id);
            break;

        case CLOTH:
            vis_->visualizeCloth(marker_name, object_configuration, color, id);

        default:
            assert(false && "Imposibru!");
    }
}

void DataGeneration::visualizeDeformableObject(
        const std::string& marker_name,
        const ObjectPointSet& object_configuration,
        const std::vector<std_msgs::ColorRGBA>& colors,
        const int32_t id) const
{
    switch (deformable_type_)
    {
        case ROPE:
            vis_->visualizeRope(marker_name, object_configuration, colors, id);
            break;

        case CLOTH:
            vis_->visualizeCloth(marker_name, object_configuration, colors, id);
            break;

        default:
            assert(false && "Imposibru!");
    }
}
