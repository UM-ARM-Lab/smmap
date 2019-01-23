#ifndef LEARNED_TRANSITIONS_H
#define LEARNED_TRANSITIONS_H

#include <arc_utilities/maybe.hpp>
#include <smmap_utilities/visualization_tools.h>
#include "smmap/trajectory.hpp"
#include "smmap/quinlan_rubber_band.h"
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
            ObjectPointSet deform_config_;
            RubberBand::Ptr rubber_band_;
            RubberBand::Ptr planned_rubber_band_;
            EigenHelpers::VectorIsometry3d rope_node_transforms_;

            uint64_t serializeSelf(std::vector<uint8_t>& buffer) const;

            uint64_t deserializeIntoSelf(
                    const std::vector<uint8_t>& buffer,
                    const uint64_t current);

            static uint64_t Serialize(
                    const State& state,
                    std::vector<uint8_t>& buffer);

            static std::pair<State, uint64_t> Deserialize(
                    const std::vector<uint8_t>& buffer,
                    const uint64_t current,
                    const RubberBand& template_band);
        };

        typedef std::pair<Eigen::Vector3d, Eigen::Vector3d> GripperPositions;
        static uint64_t SerializeGrippers(
                const GripperPositions& grippers,
                std::vector<uint8_t>& buffer);
        static std::pair<GripperPositions, uint64_t> DeserializeGrippers(
                const std::vector<uint8_t>& buffer,
                const uint64_t current);

        struct StateTransition
        {
        public:
            State starting_state_;
            State ending_state_;
            // This is the target position of the grippers.
            // In practice this data is duplicated in the endpoints of the band,
            // but this is being kept to keep everything in the
            // "state, action, next state" framework
            GripperPositions starting_gripper_positions_;
            GripperPositions ending_gripper_positions_;
            std::vector<WorldState> microstep_state_history_;

            uint64_t serializeSelf(std::vector<uint8_t>& buffer) const;

            uint64_t deserializeIntoSelf(
                    const std::vector<uint8_t>& buffer,
                    const uint64_t current);

            static uint64_t Serialize(
                    const StateTransition& state_transition,
                    std::vector<uint8_t>& buffer);

            static std::pair<StateTransition, uint64_t> Deserialize(
                    const std::vector<uint8_t>& buffer,
                    const uint64_t current,
                    const RubberBand& template_band);

            std::string toString() const;
        };

        ////////////////////////////////////////////////////////////////////////
        // Constructor
        ////////////////////////////////////////////////////////////////////////

        TransitionEstimation(
                std::shared_ptr<ros::NodeHandle> nh,
                std::shared_ptr<ros::NodeHandle> ph,
                const DijkstrasCoverageTask::ConstPtr& task,
                const smmap_utilities::Visualizer::ConstPtr& vis,
                const RubberBand& template_band);

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
        // Learning
        ////////////////////////////////////////////////////////////////////////

        Maybe::Maybe<StateTransition> findMostRecentBadTransition(
                const std::vector<std::pair<State, std::vector<WorldState>>>& trajectory) const;

        void learnTransition(const StateTransition& transition);

        ////////////////////////////////////////////////////////////////////////
        // Using transitions
        ////////////////////////////////////////////////////////////////////////

        // If the transition could be applicable, then it returns the distance
        Maybe::Maybe<double> transitionUseful(
                const RubberBand::ConstPtr& test_band,
                const GripperPositions& test_ending_gripper_positions,
                const StateTransition& transition) const;

        // Returns vector of potential outcomes of the action, and a relative
        // confidence from 0 (not likely) to 1 (input data exactly matched a
        // stored transition) in the possibility that the transition is possible.
        // I.e.; confidence 1 does not mean that the transition will happen, but
        // rather that it *could* happen.
        std::vector<std::pair<RubberBand::Ptr, double>> applyLearnedTransitions(
                const RubberBand::ConstPtr& band,
                const GripperPositions& ending_gripper_positions) const;

        RubberBand::Ptr applyTransition(
                const GripperPositions& ending_gripper_positions,
                const StateTransition& transition) const;

        double confidence(const double dist) const;

        ////////////////////////////////////////////////////////////////////////
        // Visualizing transitions
        ////////////////////////////////////////////////////////////////////////

        void visualizeTransition(
                const StateTransition& transition,
                const int32_t id = 1,
                const std::string& ns_prefix = "") const;

        void visualizeLearnedTransitions(
                const std::string& ns_prefix = "all_") const;

        void clearVisualizations() const;

        // Topic names used for publishing visualization data
        static constexpr char MDP_PRE_STATE_NS[]        = "mdp_pre_state";
        static constexpr char MDP_TESTING_STATE_NS[]    = "mdp_testing_state";
        static constexpr char MDP_POST_STATE_NS[]       = "mdp_post_state";


    private:

        const std::shared_ptr<ros::NodeHandle> nh_;
        const std::shared_ptr<ros::NodeHandle> ph_;

        const DijkstrasCoverageTask::ConstPtr task_;
        const smmap_utilities::Visualizer::ConstPtr vis_;
        std::vector<StateTransition> learned_transitions_;

//        const double action_dist_threshold_;
//        const double action_dist_scale_factor;
        const double band_dist_threshold_;
        const double band_dist_scale_factor_;

        ////////////////////////////////////////////////////////////////////////
        // Saving and loading learned transitions
        ////////////////////////////////////////////////////////////////////////

        bool useStoredTransitions() const;
        void storeTransitions() const;
        void loadSavedTransitions();
        const RubberBand& template_band_;
    };

    std::ostream& operator<<(
            std::ostream& out,
            const TransitionEstimation::StateTransition& t);
}

#endif
