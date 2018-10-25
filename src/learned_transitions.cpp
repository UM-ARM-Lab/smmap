#include "smmap/learned_transitions.h"

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/path_utils.hpp>

using namespace smmap;

DijkstrasCoverageTask::Ptr MDP::task_ = nullptr;
bool MDP::initialized_ = false;


void MDP::Initialize(const DijkstrasCoverageTask::Ptr& task)
{
    task_ = task;
    initialized_ = true;
}

bool MDP::CheckFirstOrderHomotopy(const EigenHelpers::VectorVector3d& b1, const EigenHelpers::VectorVector3d& b2)
{
    assert(initialized_);

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

bool MDP::CheckFirstOrderHomotopy(const RubberBand& b1, const RubberBand& b2)
{
    assert(initialized_);

    const auto b1_points = b1.resampleBand(task_->work_space_grid_.minStepDimension() * 0.5);
    const auto b2_points = b2.resampleBand(task_->work_space_grid_.minStepDimension() * 0.5);

    return CheckFirstOrderHomotopy(b1_points, b2_points);
}

Maybe::Maybe<MDP::StateTransition> MDP::FindMostRecentBadTransition(
        const std::vector<State, StateAllocator>& trajectory)
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
    if (CheckFirstOrderHomotopy(
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
        if (CheckFirstOrderHomotopy(
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
