#include "smmap/learned_transitions.h"

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/path_utils.hpp>

using namespace smmap;
using namespace smmap_utilities;

constexpr char TransitionEstimation::MDP_PRE_STATE_NS[];
constexpr char TransitionEstimation::MDP_TESTING_STATE_NS[];
constexpr char TransitionEstimation::MDP_POST_STATE_NS[];

TransitionEstimation::TransitionEstimation(
        std::shared_ptr<ros::NodeHandle> nh,
        std::shared_ptr<ros::NodeHandle> ph,
        const DijkstrasCoverageTask::ConstPtr &task,
        const smmap_utilities::Visualizer::ConstPtr& vis)
    : nh_(nh)
    , ph_(ph)
    , task_(task)
    , vis_(vis)
//    , action_dist_threshold_(GetTransitionActionDistThreshold(*ph_))
//    , action_dist_scale_factor(GetTransitionActionDistScaleFactor(*ph_))
    , band_dist_threshold_(GetTransitionBandDistThreshold(*ph_))
    , band_dist_scale_factor_(GetTransitionBandDistScaleFactor(*ph_))
{}

// Assumes the vectors have already been appropriately discretized/resampled
bool TransitionEstimation::checkFirstOrderHomotopy(
        const EigenHelpers::VectorVector3d& b1,
        const EigenHelpers::VectorVector3d& b2) const
{
    // Checks if the straight line between elements of the two paths is collision free
    const auto straight_line_collision_check_fn = [&] (
            const ssize_t b1_ind,
            const ssize_t b2_ind)
    {
        const auto& b1_node = b1[b1_ind];
        const auto& b2_node = b2[b2_ind];

        const ssize_t num_steps = (ssize_t)std::ceil((b2_node - b1_node).norm() / (task_->work_space_grid_.minStepDimension() * 0.5));

        // Checking 0 and num_steps to catch the endpoints of each band
        for (ssize_t ind = 0; ind <= num_steps; ++ind)
        {
            const double ratio = (double)ind / (double)num_steps;
            const auto interpolated_point = EigenHelpers::Interpolate(b1_node, b2_node, ratio);
            if (task_->sdf_->GetImmutable3d(interpolated_point).first < 0.0)
            {
                return false;
            }
        }

        return true;
    };

    return arc_utilities::FirstOrderDeformation::CheckFirstOrderDeformation(
                    b1.size(),
                    b2.size(),
                    straight_line_collision_check_fn);
}

bool TransitionEstimation::checkFirstOrderHomotopy(
        const RubberBand& b1,
        const RubberBand& b2) const
{
    const auto b1_points = b1.resampleBand();
    const auto b2_points = b2.resampleBand();

    return checkFirstOrderHomotopy(b1_points, b2_points);
}

Maybe::Maybe<TransitionEstimation::StateTransition> TransitionEstimation::findMostRecentBadTransition(
        const std::vector<State, StateAllocator>& trajectory) const
{
    // We can only learn a transition if there are at least states
    if (trajectory.size() < 2)
    {
        ROS_WARN_STREAM_NAMED("mdp_transitions",
                              "Finding most recent bad transition. "
                              << "Only one state in the trajectory, "
                              << "returning no transition");
        return Maybe::Maybe<StateTransition>();
    }

    // First, check to make sure that the last state is in a different
    // first order homotopy class for the planned vs actual band
    if (checkFirstOrderHomotopy(
            *trajectory.back().rubber_band_,
            *trajectory.back().planned_rubber_band_))
    {
        ROS_WARN_STREAM_NAMED("mdp_transitions",
                              "Finding most recent bad transition. "
                              << "Last state in MDP does not have a homotopy mismatch, "
                              << "returning no transition");
        return Maybe::Maybe<StateTransition>();
    }


    // Search backwards through the list for the last time we transitioned
    // from a matching first order homotopy to not matching
    for (size_t idx = trajectory.size() - 1; idx > 0; --idx)
    {
        // If the first order homotopy check passes, then the actual rubber band
        // and the planned rubber band are in the same first order homotopy class
        if (checkFirstOrderHomotopy(
                *trajectory[idx - 1].rubber_band_,
                *trajectory[idx - 1].planned_rubber_band_))
        {
            const auto& start_state = trajectory[idx - 1];
            const auto& end_state = trajectory[idx];
            const GripperPositions starting_gripper_positions = start_state.planned_rubber_band_->getEndpoints();
            const GripperPositions ending_gripper_positions = end_state.planned_rubber_band_->getEndpoints();
            return Maybe::Maybe<StateTransition>({start_state, end_state, starting_gripper_positions, ending_gripper_positions});
        }
    }

    ROS_WARN_STREAM_NAMED("mdp_transitions",
                          "Finding most recent bad transition. "
                          << "No transition from homotopy match to non-match, "
                          << "returning no transition");
    return Maybe::Maybe<StateTransition>();
}

void TransitionEstimation::learnTransition(const StateTransition& transition)
{
    learned_transitions_.push_back(transition);
}

void TransitionEstimation::visualizeTransition(
        const StateTransition& transition,
        const int32_t id,
        const std::string& ns_prefix) const
{
//    task_->visualizeDeformableObject(ns_prefix + MDP_PRE_STATE_NS, transition.starting_state.deform_config_, Visualizer::Red(), id);
    transition.starting_state.rubber_band_->visualize(ns_prefix + MDP_PRE_STATE_NS, Visualizer::Yellow(), Visualizer::Yellow(), id + 1);
    transition.starting_state.planned_rubber_band_->visualize(ns_prefix + MDP_PRE_STATE_NS, Visualizer::Green(), Visualizer::Green(), id + 2);

//    task_->visualizeDeformableObject(ns_prefix + MDP_POST_STATE_NS, transition.ending_state.deform_config_, Visualizer::Red(0.4f), id);
    transition.ending_state.rubber_band_->visualize(ns_prefix + MDP_POST_STATE_NS, Visualizer::Yellow(0.4f), Visualizer::Yellow(0.4f), id + 1);
    transition.ending_state.planned_rubber_band_->visualize(ns_prefix + MDP_POST_STATE_NS, Visualizer::Green(0.4f), Visualizer::Green(0.4f), id + 2);
}

void TransitionEstimation::visualizeLearnedTransitions(const std::string& ns_prefix) const
{
    for (size_t idx = 0; idx < learned_transitions_.size(); ++idx)
    {
        visualizeTransition(learned_transitions_[idx], (int32_t)(3 * idx + 1), ns_prefix);
    }
}

void TransitionEstimation::clearVisualizations() const
{
    vis_->deleteObjects(MDP_PRE_STATE_NS, 0, (int32_t)(learned_transitions_.size() + 1) * 3);
    vis_->deleteObjects(MDP_POST_STATE_NS, 0, (int32_t)(learned_transitions_.size() + 1) * 3);
    vis_->deleteObjects(MDP_TESTING_STATE_NS, 0, (int32_t)(learned_transitions_.size() + 1) * 3);
}

Maybe::Maybe<double> TransitionEstimation::transitionUseful(
        const RubberBand::ConstPtr& test_band,
        const GripperPositions& test_ending_gripper_positions,
        const StateTransition& transition) const
{
    // First check if the start points of the grippers and the end points of the grippers are
    // within 1 work space grid cell. This tries to ensure that when propagating, we don't
    // "hop over" any obstacles
    {
        const double max_dist = task_->work_space_grid_.minStepDimension() * 8;
        const auto& test_starting_gripper_positions = test_band->getEndpoints();
        if ((transition.starting_gripper_positions_.first - test_starting_gripper_positions.first).norm() > max_dist)
        {
            return Maybe::Maybe<double>();
        }
        if ((transition.starting_gripper_positions_.second - test_starting_gripper_positions.second).norm() > max_dist)
        {
            return Maybe::Maybe<double>();
        }
        if ((transition.ending_gripper_positions_.first - test_ending_gripper_positions.first).norm() > max_dist)
        {
            return Maybe::Maybe<double>();
        }
        if ((transition.ending_gripper_positions_.second - test_ending_gripper_positions.second).norm() > max_dist)
        {
            return Maybe::Maybe<double>();
        }
    }

    const double actual_band_dist = test_band->distance(*transition.starting_state.rubber_band_);
    bool actual_band_match = false;
    if (actual_band_dist < band_dist_threshold_)
    {
        actual_band_match = checkFirstOrderHomotopy(*test_band, *transition.starting_state.rubber_band_);
    }

    const double planned_band_dist = test_band->distance(*transition.starting_state.planned_rubber_band_);
    bool planned_band_match = false;
    if (planned_band_dist < band_dist_threshold_)
    {
        planned_band_match = checkFirstOrderHomotopy(*test_band, *transition.starting_state.planned_rubber_band_);
    }

    double band_dist = std::numeric_limits<double>::infinity();
    // If only one matches, then return that distance
    if (actual_band_match && !planned_band_match)
    {
        band_dist = actual_band_dist;
    }
    else if (!actual_band_match && planned_band_match)
    {
        band_dist = planned_band_dist;
    }
    // If both bands match, then take the shorter distance;
    else if (actual_band_match && planned_band_match)
    {
        band_dist = std::min(actual_band_dist, planned_band_dist);
    }
    // Otherwise, return no useful transition
    else
    {
        return Maybe::Maybe<double>();
    }
    return band_dist_scale_factor_ * band_dist;
}

std::vector<std::pair<RubberBand::Ptr, double>> TransitionEstimation::applyLearnedTransitions(
        const RubberBand::ConstPtr& test_band,
        const GripperPositions& ending_gripper_positions) const
{
    std::vector<std::pair<RubberBand::Ptr, double>> possible_transitions;

    for (size_t idx = 0; idx < learned_transitions_.size(); ++idx)
    {
        const StateTransition& transition = learned_transitions_[idx];
        Maybe::Maybe<double> dist = transitionUseful(test_band, ending_gripper_positions, transition);
        if (dist.Valid())
        {
            possible_transitions.push_back({
                    applyTransition(ending_gripper_positions, transition),
                    confidence(dist.GetImmutable())});

            visualizeTransition(transition, (int32_t)(3 * idx + 1));
            test_band->visualize(MDP_TESTING_STATE_NS, Visualizer::Silver(), Visualizer::White(), (int32_t)idx + 1);
            possible_transitions.back().first->visualize(MDP_TESTING_STATE_NS, Visualizer::Silver(), Visualizer::White(), (int32_t)idx + 2);
        }
    }

    return possible_transitions;
}

RubberBand::Ptr TransitionEstimation::applyTransition(
        const GripperPositions& ending_gripper_positions,
        const StateTransition& transition) const
{
    auto resulting_band = std::make_shared<RubberBand>(*transition.ending_state.rubber_band_);

    const bool verbose = false;
    resulting_band->forwardPropagateRubberBandToEndpointTargets(
                ending_gripper_positions.first,
                ending_gripper_positions.second,
                verbose);

    return resulting_band;
}

double TransitionEstimation::confidence(const double dist) const
{
    return std::exp(-std::pow(dist, 2));
}
