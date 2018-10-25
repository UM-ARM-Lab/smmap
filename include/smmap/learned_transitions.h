#ifndef LEARNED_TRANSITIONS_H
#define LEARNED_TRANSITIONS_H

#include <arc_utilities/maybe.hpp>
#include "smmap/trajectory.hpp"
#include "smmap/rubber_band.hpp"
#include "smmap/task_specification.h"

namespace smmap
{
    // TODO: Find a better name for this collection of functions
    class MDP
    {
    public:
        struct State
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            ObjectPointSet deform_config_;
            RubberBand::Ptr rubber_band_;
            RubberBand::Ptr planned_rubber_band_;
        };
        typedef Eigen::aligned_allocator<State> StateAllocator;
        typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> Action;

        struct StateTransition
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            State starting_state;
            State ending_state;
            Action band_endpoint_transitions_;
        };
        typedef Eigen::aligned_allocator<StateTransition> StateTransitionAllocator;

        static void Initialize(const DijkstrasCoverageTask::Ptr& task);

        static bool CheckFirstOrderHomotopy(const EigenHelpers::VectorVector3d& b1, const EigenHelpers::VectorVector3d& b2);
        static bool CheckFirstOrderHomotopy(const RubberBand& b1, const RubberBand& b2);

        static Maybe::Maybe<StateTransition> FindMostRecentBadTransition(
                const std::vector<State, StateAllocator>& trajectory);

    private:
        MDP() {}

        static DijkstrasCoverageTask::Ptr task_;
        static bool initialized_;
    };
}

#endif
