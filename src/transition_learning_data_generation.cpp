#include "smmap/transition_learning_data_generation.h"
#include <arc_utilities/arc_helpers.hpp>
#include <smmap_utilities/neighbours.h>

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;
using namespace arc_helpers;
using namespace EigenHelpers;

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
    , grippers_data_(GetGrippersData(*nh_))
    , object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(*nh_)))
    , num_nodes_(object_initial_node_distance_.cols())
    , max_stretch_factor_(GetMaxStretchFactor(*ph_))
    , max_band_length_(GetMaxBandLength(*ph_))
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
            * max_stretch_factor_;
    ROS_ERROR_STREAM_COND_NAMED(!CloseEnough(max_calced_band_length, max_band_length_, 1e-3),
                                "data_generation",
                                "Calc'd max band distance is: " << max_calced_band_length <<
                                " but the ros param saved distance is " << max_band_length_ <<
                                ". Double check the stored value in the roslaunch file.");

    // Find the shortest path through the object, between the grippers, while follow nodes of the object.
    // Used to determine the starting position of the rubber band at each timestep

    std::function<std::vector<ssize_t>(const ssize_t node)> neighbour_fn;
    switch (deformable_type_)
    {
        case ROPE:
        {
            LineNeighbours neighbours_calc(num_nodes_);
            neighbour_fn = [neighbours_calc] (const ssize_t node)
            {
                return neighbours_calc.getNodeNeighbours(node);
            };
            break;
        }

        case CLOTH:
        {
            Grid4Neighbours neighbours_calc(num_nodes_, GetClothNumControlPointsX(*nh_));
            neighbour_fn = [neighbours_calc] (const ssize_t node)
            {
                return neighbours_calc.getNodeNeighbours(node);
            };
        }
        default:
            throw_arc_exception(std::invalid_argument, "Invalid deformable type; this should not be possible");
    }
    path_between_grippers_through_object_ = GetShortestPathBetweenGrippersThroughObject(
                robot_->getGrippersData(), GetObjectInitialConfiguration(*nh_), neighbour_fn);

    // Create the initial rubber band
    const double resampled_band_max_pointwise_dist = work_space_grid_.minStepDimension() / 2.0;
    const size_t upsampled_band_num_points = GetRRTBandMaxPoints(*ph_);

    const auto starting_band_points = GetPathBetweenGrippersThroughObject(
                world_state, path_between_grippers_through_object_);
    band_ = std::make_shared<RubberBand>(
                nh_,
                ph_,
                vis_,
                sdf_,
                work_space_grid_,
                starting_band_points,
                resampled_band_max_pointwise_dist,
                upsampled_band_num_points,
                max_band_length_);
}

void DataGeneration::runTests()
{
    const auto& transitions = transition_estimator_->transitions();

    ROS_INFO_STREAM("Visualizing " << transitions.size() << " transitions");
    for (size_t idx = 0; idx < transitions.size(); ++idx)
    {
        const TransitionEstimation::StateTransition& trans = transitions[idx];

        // Stored transition
        {
            auto bands = transition_estimator_->reduceMicrostepsToBands(
                        trans.microstep_state_history_,
                        path_between_grippers_through_object_);
            bands.insert(bands.begin(), trans.starting_state_.rubber_band_);

            // Add each band with a different color, ranging from
            // blue (early in the history) to red (late in the history)
            std::vector<std_msgs::ColorRGBA> colors;
            for (size_t band_idx = 0; band_idx < bands.size(); ++band_idx)
            {
                const float ratio = (float)(band_idx) / (float)(bands.size() - 1) ;
                const auto color = InterpolateColor(Visualizer::Blue(), Visualizer::Red(), ratio);
                colors.insert(colors.end(), bands[band_idx]->upsampleBand().size(), color);
            }

            const auto band_surface = RubberBand::AggregateBandPoints(bands);
            vis_->visualizePoints("band_surface", band_surface, colors, 1, 0.002);
            transition_estimator_->visualizeTransition(trans, 1, "transition_testing_");
        }

        // Test transition
        {
            AllGrippersSinglePose starting_gripper_poses(2);
            starting_gripper_poses[0] = trans.starting_state_.rope_node_transforms_.front();
            starting_gripper_poses[0].translation() = trans.starting_gripper_positions_.first;
            starting_gripper_poses[1] = trans.starting_state_.rope_node_transforms_.back();
            starting_gripper_poses[1].translation() = trans.starting_gripper_positions_.second;
            AllGrippersSinglePose target_gripper_poses = starting_gripper_poses;
            target_gripper_poses[0].translation() = trans.ending_gripper_positions_.first;
            target_gripper_poses[1].translation() = trans.ending_gripper_positions_.second;


            const std::vector<WorldState> test_result = robot_->testRobotMotionMicrosteps(
                        trans.starting_state_.rope_node_transforms_,
                        starting_gripper_poses,
                        target_gripper_poses,
                        (int)trans.microstep_state_history_.size() / 4);
            auto test_bands = transition_estimator_->reduceMicrostepsToBands(
                        test_result,
                        path_between_grippers_through_object_);
            test_bands.insert(test_bands.begin(), trans.starting_state_.rubber_band_);

            // Add each band with a different color, ranging from
            // cyan (early in the history) to magenta (late in the history)
            std::vector<std_msgs::ColorRGBA> colors;
            for (size_t band_idx = 0; band_idx < test_bands.size(); ++band_idx)
            {
                const float ratio = (float)(band_idx) / (float)(test_bands.size() - 1) ;
                const auto color = InterpolateColor(Visualizer::Cyan(), Visualizer::Magenta(), ratio);
                colors.insert(colors.end(), test_bands[band_idx]->upsampleBand().size(), color);
            }

            const auto band_surface = RubberBand::AggregateBandPoints(test_bands);
            vis_->visualizePoints("band_surface_test", band_surface, colors, 1, 0.002);
        }


        const auto delta =
                Sub(trans.ending_gripper_positions_,
                    trans.starting_gripper_positions_);
        std::cout << "Number of bands: " << trans.microstep_state_history_.size() + 1
                  << "  Gripper endpoint distances: " << delta.first.norm() << "  " << delta.second.norm()
                  << "  Net norm: " << std::sqrt(delta.first.squaredNorm() + delta.second.squaredNorm())
                  << "    Press any key to continue " << std::flush;
        GetChar();
        std::cout << std::endl;

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
