#include "smmap/learned_transitions.h"

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/path_utils.hpp>

using namespace smmap;
using namespace smmap_utilities;

constexpr char TransitionEstimation::MDP_PRE_STATE_NS[];
constexpr char TransitionEstimation::MDP_ACTION_NS[];
constexpr char TransitionEstimation::MDP_POST_STATE_NS[];

TransitionEstimation::TransitionEstimation(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const DijkstrasCoverageTask::ConstPtr &task,
        const smmap_utilities::Visualizer::ConstPtr& vis)
    : nh_(nh)
    , ph_(ph)
    , task_(task)
    , vis_(vis)
    , action_dist_threshold_(GetTransitionActionDistThreshold(ph_))
    , action_dist_scale_factor(GetTransitionActionDistScaleFactor(ph_))
    , band_dist_threshold_(GetTransitionBandDistThreshold(ph_))
    , band_dist_scale_factor_(GetTransitionBandDistScaleFactor(ph_))
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

            const auto start_state_endpoints = start_state.planned_rubber_band_->getEndpoints();
            const auto end_state_endpoints = end_state.planned_rubber_band_->getEndpoints();

            const Action action = {
                end_state_endpoints.first - start_state_endpoints.first,
                end_state_endpoints.second - start_state_endpoints.second};

            return Maybe::Maybe<StateTransition>({start_state, end_state, action});
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

void TransitionEstimation::visualizeTransition(const StateTransition& transition, const int32_t id) const
{
    task_->visualizeDeformableObject(MDP_PRE_STATE_NS, transition.starting_state.deform_config_, Visualizer::Red(), id);
    transition.starting_state.rubber_band_->visualize(MDP_PRE_STATE_NS, Visualizer::Yellow(), Visualizer::Yellow(), id + 1);
    transition.starting_state.planned_rubber_band_->visualize(MDP_PRE_STATE_NS, Visualizer::Green(), Visualizer::Green(), id + 2);

    task_->visualizeDeformableObject(MDP_POST_STATE_NS, transition.ending_state.deform_config_, Visualizer::Red(0.4f), id);
    transition.ending_state.rubber_band_->visualize(MDP_POST_STATE_NS, Visualizer::Yellow(0.4f), Visualizer::Yellow(0.4f), id + 1);
    transition.ending_state.planned_rubber_band_->visualize(MDP_POST_STATE_NS, Visualizer::Green(0.4f), Visualizer::Green(0.4f), id + 2);
}

void TransitionEstimation::visualizeLearnedTransitions() const
{
    for (size_t idx = 0; idx < learned_transitions_.size(); ++idx)
    {
        visualizeTransition(learned_transitions_[idx], (int32_t)(3 * idx + 1));
    }
}


double TransitionEstimation::actionDistance(const Action& a1, const Action& a2) const
{
    return std::sqrt(
                (a1.first - a2.first).squaredNorm() +
                (a1.second - a2.second).squaredNorm());
}

Maybe::Maybe<double> TransitionEstimation::transitionUseful(
        const RubberBand& band,
        const Action& action,
        const StateTransition& transition) const
{
    const double action_dist = actionDistance(action, transition.action_);
    if (action_dist > action_dist_threshold_)
    {
        return Maybe::Maybe<double>();
    }

    const double actual_band_dist = band.distance(*transition.starting_state.rubber_band_);
    bool actual_band_match = false;
    if (actual_band_dist < band_dist_threshold_)
    {
        actual_band_match = checkFirstOrderHomotopy(band, *transition.starting_state.rubber_band_);
    }

    const double planned_band_dist = band.distance(*transition.starting_state.planned_rubber_band_);
    bool planned_band_match = false;
    if (planned_band_dist < band_dist_threshold_)
    {
        planned_band_match = checkFirstOrderHomotopy(band, *transition.starting_state.planned_rubber_band_);
    }

    // If only one matches, then return that distance
    if (actual_band_match && !planned_band_match)
    {
        return action_dist_scale_factor * action_dist + band_dist_scale_factor_ * actual_band_dist;
    }
    else if (!actual_band_match && planned_band_match)
    {
        return action_dist_scale_factor * action_dist + band_dist_scale_factor_ * planned_band_dist;
    }
    // If both bands match, then take the shorter distance;
    else if (actual_band_match && planned_band_match)
    {
        const double band_dist = std::min(actual_band_dist, planned_band_dist);
        return action_dist_scale_factor * action_dist + band_dist_scale_factor_ * band_dist;
    }
    // Otherwise, return no useful transition
    else
    {
        return Maybe::Maybe<double>();
    }
}

std::vector<std::pair<RubberBand::Ptr, double>> TransitionEstimation::applyLearnedTransitions(
        const RubberBand& band,
        const Action& action) const
{
    std::vector<std::pair<RubberBand::Ptr, double>> possible_transitions;

    for (const StateTransition& transition : learned_transitions_)
    {
        Maybe::Maybe<double> dist = transitionUseful(band, action, transition);
        if (dist.Valid())
        {
            possible_transitions.push_back({
                    applyTransition(band, action, transition),
                    confidence(dist.GetImmutable())});
        }
    }

    return possible_transitions;
}

RubberBand::Ptr TransitionEstimation::applyTransition(
        const RubberBand& band,
        const Action& action,
        const StateTransition& transition) const
{
    auto resulting_band = std::make_shared<RubberBand>(*transition.ending_state.rubber_band_);

    const bool verbose = false;
    const auto endpoints = band.getEndpoints();
    resulting_band->forwardPropagateRubberBandToEndpointTargets(
                endpoints.first + action.first,
                endpoints.second + action.second,
                verbose);

    return resulting_band;
}

double TransitionEstimation::confidence(const double dist) const
{
    return std::exp(-std::pow(dist, 4));
}
