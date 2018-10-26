#include "smmap/learned_transitions.h"

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/path_utils.hpp>

using namespace smmap;
using namespace smmap_utilities;

constexpr char MDP::MDP_PRE_STATE_NS[];
constexpr char MDP::MDP_ACTION_NS[];
constexpr char MDP::MDP_POST_STATE_NS[];

MDP::MDP(const DijkstrasCoverageTask::Ptr &task,
         const smmap_utilities::Visualizer::Ptr& vis)
    : task_(task)
    , vis_(vis)
{}

bool MDP::checkFirstOrderHomotopy(const EigenHelpers::VectorVector3d& b1, const EigenHelpers::VectorVector3d& b2) const
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

bool MDP::checkFirstOrderHomotopy(const RubberBand& b1, const RubberBand& b2) const
{
    const auto b1_points = b1.resampleBand(task_->work_space_grid_.minStepDimension() * 0.5);
    const auto b2_points = b2.resampleBand(task_->work_space_grid_.minStepDimension() * 0.5);

    return checkFirstOrderHomotopy(b1_points, b2_points);
}

Maybe::Maybe<MDP::StateTransition> MDP::findMostRecentBadTransition(
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
    for (size_t idx = trajectory.size() - 1; idx > 0; ++idx)
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

void MDP::learnTransition(const MDP::StateTransition& transition)
{
    learned_transitions_.push_back(transition);
}

void MDP::visualizeTransition(const MDP::StateTransition& transition, const int32_t id) const
{
    task_->visualizeDeformableObject(MDP_PRE_STATE_NS, transition.starting_state.deform_config_, Visualizer::Yellow(), id);
    transition.starting_state.rubber_band_->visualize(MDP_PRE_STATE_NS, Visualizer::Yellow(), Visualizer::Yellow(), id + 1);
    transition.starting_state.planned_rubber_band_->visualize(MDP_PRE_STATE_NS, Visualizer::Yellow(0.5), Visualizer::Yellow(0.5), id + 2);

    task_->visualizeDeformableObject(MDP_POST_STATE_NS, transition.ending_state.deform_config_, Visualizer::Red(), id);
    transition.ending_state.rubber_band_->visualize(MDP_POST_STATE_NS, Visualizer::Red(), Visualizer::Red(), id + 1);
    transition.ending_state.planned_rubber_band_->visualize(MDP_POST_STATE_NS, Visualizer::Red(0.5), Visualizer::Red(0.5), id + 2);
}

void MDP::visualizeLearnedTransitions() const
{
    for (size_t idx = 0; idx < learned_transitions_.size(); ++idx)
    {
        visualizeTransition(learned_transitions_[idx], (int32_t)(3 * idx + 1));
    }
}
