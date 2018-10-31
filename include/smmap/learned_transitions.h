#ifndef LEARNED_TRANSITIONS_H
#define LEARNED_TRANSITIONS_H

#include <arc_utilities/maybe.hpp>
#include <smmap_utilities/visualization_tools.h>
#include "smmap/trajectory.hpp"
#include "smmap/rubber_band.hpp"
#include "smmap/task_specification.h"

namespace smmap
{
    class TransitionEstimation
    {
    public:
        typedef std::shared_ptr<TransitionEstimation> Ptr;
        typedef std::shared_ptr<const TransitionEstimation> ConstPtr;

        struct State
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            ObjectPointSet deform_config_;
            RubberBand::ConstPtr rubber_band_;
            RubberBand::ConstPtr planned_rubber_band_;
        };
        typedef Eigen::aligned_allocator<State> StateAllocator;

        typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> Action;

        struct StateTransition
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            State starting_state;
            State ending_state;
            Action action_;
        };
        typedef Eigen::aligned_allocator<StateTransition> StateTransitionAllocator;

        ////////////////////////////////////////////////////////////////////////
        // Constructor
        ////////////////////////////////////////////////////////////////////////

        TransitionEstimation(
                ros::NodeHandle& nh,
                ros::NodeHandle& ph,
                const DijkstrasCoverageTask::ConstPtr& task,
                const smmap_utilities::Visualizer::ConstPtr& vis);

        ////////////////////////////////////////////////////////////////////////
        // Helper functions - used externally and internally
        ////////////////////////////////////////////////////////////////////////

        // Assumes the vectors have already been appropriately discretized/resampled
        bool checkFirstOrderHomotopy(
                const EigenHelpers::VectorVector3d& b1,
                const EigenHelpers::VectorVector3d& b2) const;
        // Resamples the bands appropriately, then calls the above
        bool checkFirstOrderHomotopy(
                const RubberBand& b1,
                const RubberBand& b2) const;

        ////////////////////////////////////////////////////////////////////////
        // Learning and visualizing transitions
        ////////////////////////////////////////////////////////////////////////

        Maybe::Maybe<StateTransition> findMostRecentBadTransition(
                const std::vector<State, StateAllocator>& trajectory) const;

        void learnTransition(const StateTransition& transition);


        void visualizeTransition(
                const StateTransition& transition, const
                int32_t id = 1) const;

        void visualizeLearnedTransitions() const;

        // Topic names used for publishing visualization data
        static constexpr char MDP_PRE_STATE_NS[]    = "mdp_pre_state";
        static constexpr char MDP_ACTION_NS[]       = "mdp_action";
        static constexpr char MDP_POST_STATE_NS[]   = "mdp_post_state";

        ////////////////////////////////////////////////////////////////////////
        // Using transitions
        ////////////////////////////////////////////////////////////////////////

        double actionDistance(
                const Action& a1,
                const Action& a2) const;

        // If the transition could be applicable, then it returns the distance
        Maybe::Maybe<double> transitionUseful(
                const RubberBand& band,
                const Action& action,
                const StateTransition& transition) const;

        // Returns vector of potential outcomes of the action, and a relative
        // confidence from 0 (not likely) to 1 (input data exactly matched a stored transition)
        // in the possibility that the transition is possible.
        // I.e.; confidence 1 does not mean that the transition will happen, but rather
        // that it *could* happen.
        std::vector<std::pair<RubberBand::Ptr, double>> applyLearnedTransitions(
                const RubberBand& band,
                const Action& action) const;

        RubberBand::Ptr applyTransition(
                const RubberBand& band,
                const Action& action,
                const StateTransition& transition) const;

        double confidence(const double dist) const;

    private:

        ros::NodeHandle nh_;
        ros::NodeHandle ph_;

        const DijkstrasCoverageTask::ConstPtr task_;
        const smmap_utilities::Visualizer::ConstPtr vis_;
        std::vector<StateTransition, StateTransitionAllocator> learned_transitions_;

        const double action_dist_threshold_;
        const double action_dist_scale_factor;
        const double band_dist_threshold_;
        const double band_dist_scale_factor_;
    };
}

#endif
