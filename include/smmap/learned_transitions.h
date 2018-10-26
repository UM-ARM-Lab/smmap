#ifndef LEARNED_TRANSITIONS_H
#define LEARNED_TRANSITIONS_H

#include <arc_utilities/maybe.hpp>
#include <smmap_utilities/visualization_tools.h>
#include "smmap/trajectory.hpp"
#include "smmap/rubber_band.hpp"
#include "smmap/task_specification.h"

namespace smmap
{
    // TODO: Find a better name for this collection of functions
    class MDP
    {
    public:
        typedef std::shared_ptr<MDP> Ptr;

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

        MDP(const DijkstrasCoverageTask::Ptr& task,
            const smmap_utilities::Visualizer::Ptr& vis);

        bool checkFirstOrderHomotopy(
                const EigenHelpers::VectorVector3d& b1,
                const EigenHelpers::VectorVector3d& b2) const;
        bool checkFirstOrderHomotopy(
                const RubberBand& b1,
                const RubberBand& b2) const;

        Maybe::Maybe<StateTransition> findMostRecentBadTransition(
                const std::vector<State, StateAllocator>& trajectory) const;

        void learnTransition(const StateTransition& transition);

        void visualizeTransition(const StateTransition& transition, const int32_t id = 1) const;
        void visualizeLearnedTransitions() const;

        // Topic names used for publishing visualization data
        static constexpr char MDP_PRE_STATE_NS[]    = "mdp_pre_state";
        static constexpr char MDP_ACTION_NS[]       = "mdp_action";
        static constexpr char MDP_POST_STATE_NS[]   = "mdp_post_state";

    private:

        const DijkstrasCoverageTask::Ptr task_;
        smmap_utilities::Visualizer::Ptr vis_;
        std::vector<MDP::StateTransition, MDP::StateTransitionAllocator> learned_transitions_;
    };
}

#endif
