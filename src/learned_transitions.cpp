#include "smmap/learned_transitions.h"

#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/serialization_eigen.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/path_utils.hpp>

using namespace smmap;
using namespace smmap_utilities;

constexpr char TransitionEstimation::MDP_PRE_STATE_NS[];
constexpr char TransitionEstimation::MDP_TESTING_STATE_NS[];
constexpr char TransitionEstimation::MDP_POST_STATE_NS[];

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Internal types function helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////// State /////////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t TransitionEstimation::State::serializeSelf(
        std::vector<uint8_t>& buffer) const
{
    const uint64_t starting_size = buffer.size();
    arc_utilities::SerializeEigen(deform_config_, buffer);
    rubber_band_->serialize(buffer);
    planned_rubber_band_->serialize(buffer);
    arc_utilities::SerializeVector(rope_node_transforms_, buffer, arc_utilities::SerializeEigen<Eigen::Isometry3d>);
    return buffer.size() - starting_size;
}

uint64_t TransitionEstimation::State::deserializeIntoSelf(
        const std::vector<uint8_t>& buffer,
        const uint64_t current)
{
    uint64_t bytes_read = 0;

    const auto deform_config_deserialized =
            arc_utilities::DeserializeEigen<ObjectPointSet>(buffer, current + bytes_read);
    deform_config_ = deform_config_deserialized.first;
    bytes_read += deform_config_deserialized.second;
    bytes_read += rubber_band_->deserializeIntoSelf(buffer, current + bytes_read);
    bytes_read += planned_rubber_band_->deserializeIntoSelf(buffer, current + bytes_read);
    const auto rope_node_transforms_deserialized =
            arc_utilities::DeserializeVector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>(
                buffer, current + bytes_read, arc_utilities::DeserializeEigen<Eigen::Isometry3d>);
    rope_node_transforms_ = rope_node_transforms_deserialized.first;
    bytes_read += rope_node_transforms_deserialized.second;
    return bytes_read;
}

uint64_t TransitionEstimation::State::Serialize(
        const State& state,
        std::vector<uint8_t>& buffer)
{
    return state.serializeSelf(buffer);
}

std::pair<TransitionEstimation::State, uint64_t> TransitionEstimation::State::Deserialize(
        const std::vector<uint8_t>& buffer,
        const uint64_t current,
        const RubberBand& template_band)
{
    State state;
    state.planned_rubber_band_ = std::make_shared<RubberBand>(template_band);
    state.rubber_band_ = std::make_shared<RubberBand>(template_band);
    const auto bytes_read = state.deserializeIntoSelf(buffer, current);
    return {state, bytes_read};
}

bool TransitionEstimation::State::operator==(const State& other) const
{
    if ((deform_config_.array() != other.deform_config_.array()).any())
    {
        return false;
    }
    if (*rubber_band_ != *other.rubber_band_)
    {
        return false;
    }
    if (*planned_rubber_band_ != *other.planned_rubber_band_)
    {
        return false;
    }
    if (rope_node_transforms_.size() != other.rope_node_transforms_.size())
    {
        for (size_t idx = 0; idx < rope_node_transforms_.size(); ++idx)
        {
            if ((rope_node_transforms_[idx].matrix().array() != other.rope_node_transforms_[idx].matrix().array()).any())
            {
                return false;
            }
        }
    }
    return true;
}

bool TransitionEstimation::State::operator!=(const State& other) const
{
    return !(*this == other);
}

//////// Grippers //////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t TransitionEstimation::SerializeGrippers(
        const GripperPositions& grippers,
        std::vector<uint8_t>& buffer)
{
    // TODO: determine why I need these lambdas in order to avoid a linking error
    auto f1 = [] (const GripperPositions::first_type& vec, std::vector<uint8_t>& buf)
    {
        return arc_utilities::SerializeEigen(vec, buf);
    };
    auto f2 = [] (const GripperPositions::second_type& vec, std::vector<uint8_t>& buf)
    {
        return arc_utilities::SerializeEigen(vec, buf);
    };
    return arc_utilities::SerializePair<GripperPositions::first_type, GripperPositions::second_type>(
                grippers, buffer, f1, f2);
}

std::pair<TransitionEstimation::GripperPositions, uint64_t> TransitionEstimation::DeserializeGrippers(
        const std::vector<uint8_t>& buffer,
        const uint64_t current)
{
    return arc_utilities::DeserializePair<GripperPositions::first_type, GripperPositions::second_type>(
                buffer,
                current,
                arc_utilities::DeserializeEigen<GripperPositions::first_type>,
                arc_utilities::DeserializeEigen<GripperPositions::second_type>);
}

//////// StateTransition ///////////////////////////////////////////////////////////////////////////////////////////////

uint64_t TransitionEstimation::StateTransition::serializeSelf(
        std::vector<uint8_t>& buffer) const
{
    const uint64_t starting_bytes = buffer.size();
    starting_state_.serializeSelf(buffer);
    ending_state_.serializeSelf(buffer);
    SerializeGrippers(starting_gripper_positions_, buffer);
    SerializeGrippers(ending_gripper_positions_, buffer);
    arc_utilities::SerializeVector<WorldState>(microstep_state_history_, buffer, &WorldState::Serialize);
    const uint64_t bytes_written = buffer.size() - starting_bytes;

    // Verify no mistakes were made
    {
        const auto deserialized = Deserialize(buffer, starting_bytes, *starting_state_.rubber_band_);
        assert(deserialized.second == bytes_written);
        assert(deserialized.first.starting_gripper_positions_                                       == starting_gripper_positions_);
        assert(deserialized.first.starting_state_.deform_config_                                    == starting_state_.deform_config_);
        assert(deserialized.first.starting_state_.rubber_band_->getVectorRepresentation()           == starting_state_.rubber_band_->getVectorRepresentation());
        assert(deserialized.first.starting_state_.planned_rubber_band_->getVectorRepresentation()   == starting_state_.planned_rubber_band_->getVectorRepresentation());

        assert(deserialized.first.ending_gripper_positions_                                         == ending_gripper_positions_);
        assert(deserialized.first.ending_state_.deform_config_                                      == ending_state_.deform_config_);
        assert(deserialized.first.ending_state_.rubber_band_->getVectorRepresentation()             == ending_state_.rubber_band_->getVectorRepresentation());
        assert(deserialized.first.ending_state_.planned_rubber_band_->getVectorRepresentation()     == ending_state_.planned_rubber_band_->getVectorRepresentation());

        assert(deserialized.first.microstep_state_history_                                          == microstep_state_history_);
    }

    return bytes_written;
}

uint64_t TransitionEstimation::StateTransition::deserializeIntoSelf(
        const std::vector<uint8_t>& buffer,
        const uint64_t current)
{
    uint64_t bytes_read = 0;

    bytes_read += starting_state_.deserializeIntoSelf(buffer, current + bytes_read);
    bytes_read += ending_state_.deserializeIntoSelf(buffer, current + bytes_read);

    const auto staring_grippers_deserialized = DeserializeGrippers(buffer, current + bytes_read);
    starting_gripper_positions_ = staring_grippers_deserialized.first;
    bytes_read += staring_grippers_deserialized.second;

    const auto ending_grippers_deserialized = DeserializeGrippers(buffer, current + bytes_read);
    ending_gripper_positions_ = ending_grippers_deserialized.first;
    bytes_read += ending_grippers_deserialized.second;

    const auto microsteps_deserialized = arc_utilities::DeserializeVector<WorldState>(buffer, current + bytes_read, &WorldState::Deserialize);
    microstep_state_history_ = microsteps_deserialized.first;
    bytes_read += microsteps_deserialized.second;

    return bytes_read;
}

uint64_t TransitionEstimation::StateTransition::Serialize(
        const StateTransition& state_transition,
        std::vector<uint8_t>& buffer)
{
    return state_transition.serializeSelf(buffer);
}

std::pair<TransitionEstimation::StateTransition, uint64_t> TransitionEstimation::StateTransition::Deserialize(
        const std::vector<uint8_t>& buffer,
        const uint64_t current,
        const RubberBand& template_band)
{
    StateTransition transition;
    transition.starting_state_.rubber_band_ = std::make_shared<RubberBand>(template_band);
    transition.starting_state_.planned_rubber_band_ = std::make_shared<RubberBand>(template_band);
    transition.ending_state_.rubber_band_ = std::make_shared<RubberBand>(template_band);
    transition.ending_state_.planned_rubber_band_ = std::make_shared<RubberBand>(template_band);
    const auto bytes_read = transition.deserializeIntoSelf(buffer, current);
    return {transition, bytes_read};
}

bool TransitionEstimation::StateTransition::operator==(const StateTransition& other) const
{
    if (starting_state_ != other.starting_state_)
    {
        return false;
    }
    if (ending_state_ != other.ending_state_)
    {
        return false;
    }
    if ((starting_gripper_positions_.first.array() != other.starting_gripper_positions_.first.array()).any())
    {
        return false;
    }
    if ((starting_gripper_positions_.second.array() != other.starting_gripper_positions_.second.array()).any())
    {
        return false;
    }
    if ((ending_gripper_positions_.first.array() != other.ending_gripper_positions_.first.array()).any())
    {
        return false;
    }
    if ((ending_gripper_positions_.second.array() != other.ending_gripper_positions_.second.array()).any())
    {
        return false;
    }
    if (microstep_state_history_ != other.microstep_state_history_)
    {
        return false;
    }
    return true;
}

bool TransitionEstimation::StateTransition::operator!=(const StateTransition& other) const
{
    return !(*this == other);
}

std::string TransitionEstimation::StateTransition::toString() const
{
    std::stringstream ss;
    ss << "Ending state rope node transforms:\n"
       << PrettyPrint::PrettyPrint(ending_state_.rope_node_transforms_, true, "\n") << "\n"
       << "Microstep last state rope node transforms:\n"
       << PrettyPrint::PrettyPrint(microstep_state_history_.back().rope_node_transforms_, true, "\n") << "\n";

    return ss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main TransitionEstimation class
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TransitionEstimation::TransitionEstimation(
        std::shared_ptr<ros::NodeHandle> nh,
        std::shared_ptr<ros::NodeHandle> ph,
        const sdf_tools::SignedDistanceField::ConstPtr& sdf,
        const XYZGrid work_space_grid,
        const smmap_utilities::Visualizer::ConstPtr& vis,
        const RubberBand& template_band)
    : nh_(nh)
    , ph_(ph)
    , sdf_(sdf)
    , work_space_grid_(work_space_grid)
    , vis_(vis)
//    , action_dist_threshold_(GetTransitionActionDistThreshold(*ph_))
//    , action_dist_scale_factor(GetTransitionActionDistScaleFactor(*ph_))
    , band_dist_threshold_(GetTransitionBandDistThreshold(*ph_))
    , band_dist_scale_factor_(GetTransitionBandDistScaleFactor(*ph_))
    , template_band_(template_band)
{
    if (useStoredTransitions())
    {
        loadSavedTransitions();
    }
}


//////// Helper functions //////////////////////////////////////////////////////////////////////////////////////////////

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

        const ssize_t num_steps = (ssize_t)std::ceil((b2_node - b1_node).norm() / (work_space_grid_.minStepDimension() * 0.5));

        // Checking 0 and num_steps to catch the endpoints of each band
        for (ssize_t ind = 0; ind <= num_steps; ++ind)
        {
            const double ratio = (double)ind / (double)num_steps;
            const auto interpolated_point = EigenHelpers::Interpolate(b1_node, b2_node, ratio);
            if (sdf_->GetImmutable3d(interpolated_point).first < 0.0)
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

std::vector<RubberBand::Ptr> TransitionEstimation::reduceMicrostepsToBands(
        const std::vector<WorldState>& microsteps) const
{
    std::vector<RubberBand::Ptr> bands;
    bands.reserve(microsteps.size());
    for (size_t idx = 0; idx < microsteps.size(); ++idx)
    {
        if (microsteps[idx].all_grippers_single_pose_.size() == 0)
        {
            continue;
        }
        bands.push_back(std::make_shared<RubberBand>(template_band_));
        bands.back()->resetBand(microsteps[idx]);
    }

    return bands;
}

/////// Learning transitions ///////////////////////////////////////////////////////////////////////////////////////////

Maybe::Maybe<TransitionEstimation::StateTransition> TransitionEstimation::findMostRecentBadTransition(
        const std::vector<std::pair<TransitionEstimation::State, std::vector<WorldState>>>& trajectory) const
{
    // We can only learn a transition if there are at least states
    if (trajectory.size() < 2)
    {
        ROS_WARN_STREAM_NAMED("transitions",
                              "Finding most recent bad transition. "
                              << "Only one state in the trajectory, "
                              << "returning no transition");
        return Maybe::Maybe<StateTransition>();
    }

    // First, check to make sure that the last state is in a different
    // first order homotopy class for the planned vs actual band
    if (checkFirstOrderHomotopy(
            *trajectory.back().first.rubber_band_,
            *trajectory.back().first.planned_rubber_band_))
    {
        ROS_WARN_STREAM_NAMED("transitions",
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
                *trajectory[idx - 1].first.rubber_band_,
                *trajectory[idx - 1].first.planned_rubber_band_))
        {
            const auto& start_state = trajectory[idx - 1].first;
            const auto& end_state = trajectory[idx].first;
            const GripperPositions starting_gripper_positions = start_state.planned_rubber_band_->getEndpoints();
            const GripperPositions ending_gripper_positions = end_state.planned_rubber_band_->getEndpoints();
            StateTransition transition =
            {
                start_state,
                end_state,
                starting_gripper_positions,
                ending_gripper_positions,
                trajectory[idx].second
            };
            return Maybe::Maybe<StateTransition>(transition);
        }
    }

    ROS_WARN_STREAM_NAMED("transitions",
                          "Finding most recent bad transition. "
                          << "No transition from homotopy match to non-match, "
                          << "returning no transition");
    return Maybe::Maybe<StateTransition>();
}

void TransitionEstimation::learnTransition(const StateTransition& transition)
{
    std::cout << transition << std::endl;
    learned_transitions_.push_back(transition);
    storeTransitions();
}

//////// Using transitions /////////////////////////////////////////////////////////////////////////////////////////////

const std::vector<TransitionEstimation::StateTransition>& TransitionEstimation::transitions() const
{
    return learned_transitions_;
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
        static const double max_dist = work_space_grid_.minStepDimension() * std::sqrt(2.0);
        const auto& test_starting_gripper_positions = test_band->getEndpoints();

//        static double smallest_dist_starting_first = std::numeric_limits<double>::infinity();
//        static double smallest_dist_starting_second = std::numeric_limits<double>::infinity();
//        static double smallest_dist_ending_first = std::numeric_limits<double>::infinity();
//        static double smallest_dist_ending_second = std::numeric_limits<double>::infinity();

//        std::cout << "max dist: " << max_dist << std::endl;
//        smallest_dist_starting_first = std::min(smallest_dist_starting_first, (test_starting_gripper_positions.first - transition.starting_gripper_positions_.first).norm());
//        smallest_dist_starting_second = std::min(smallest_dist_starting_second, (test_starting_gripper_positions.second - transition.starting_gripper_positions_.second).norm());
//        smallest_dist_ending_first = std::min(smallest_dist_ending_first, (test_ending_gripper_positions.first - transition.ending_gripper_positions_.first).norm());
//        smallest_dist_ending_second = std::min(smallest_dist_ending_second, (test_ending_gripper_positions.second - transition.ending_gripper_positions_.second).norm());

//        std::cout << "smallest seen:\n"
//                  << "          " << smallest_dist_starting_first << std::endl
//                  << "          " << smallest_dist_starting_second << std::endl
//                  << "          " << smallest_dist_ending_first << std::endl
//                  << "          " << smallest_dist_ending_second << std::endl;

//        std::cout << "starting dist first:  " << (test_starting_gripper_positions.first - transition.starting_gripper_positions_.first).norm() << std::endl;
//        std::cout << "starting dist second: " << (test_starting_gripper_positions.second - transition.starting_gripper_positions_.second).norm() << std::endl;
//        std::cout << "ending   dist first:  " << (test_ending_gripper_positions.first - transition.ending_gripper_positions_.first).norm() << std::endl;
//        std::cout << "ending   dist second: " << (test_ending_gripper_positions.second - transition.ending_gripper_positions_.second).norm() << std::endl;

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

    const double actual_band_dist = test_band->distance(*transition.starting_state_.rubber_band_);
    bool actual_band_match = false;
    if (actual_band_dist < band_dist_threshold_)
    {
        actual_band_match = checkFirstOrderHomotopy(*test_band, *transition.starting_state_.rubber_band_);
    }
    std::cout << "Actual  band distance: " << actual_band_dist << " match: " << actual_band_match << std::endl;

    const double planned_band_dist = test_band->distance(*transition.starting_state_.planned_rubber_band_);
    bool planned_band_match = false;
    if (planned_band_dist < band_dist_threshold_)
    {
        planned_band_match = checkFirstOrderHomotopy(*test_band, *transition.starting_state_.planned_rubber_band_);
    }
    std::cout << "Planned band distance: " << planned_band_dist << " match: " << planned_band_match << std::endl;

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
            test_band->visualize(MDP_TESTING_STATE_NS + std::string("_pre"), Visualizer::White(), Visualizer::White(), (int32_t)idx + 1);
            possible_transitions.back().first->visualize(MDP_TESTING_STATE_NS + std::string("_post"), Visualizer::Silver(), Visualizer::Silver(), (int32_t)idx + 2);
            vis_->forcePublishNow(0.02);
        }
    }

    return possible_transitions;
}

RubberBand::Ptr TransitionEstimation::applyTransition(
        const GripperPositions& ending_gripper_positions,
        const StateTransition& transition) const
{
    auto resulting_band = std::make_shared<RubberBand>(*transition.ending_state_.rubber_band_);

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

//////// Visualization /////////////////////////////////////////////////////////////////////////////////////////////////

void TransitionEstimation::visualizeTransition(
        const StateTransition& transition,
        const int32_t id,
        const std::string& ns_prefix) const
{
//    visualizeDeformableObject(ns_prefix + MDP_PRE_STATE_NS, transition.starting_state.deform_config_, Visualizer::Red(), id);
    transition.starting_state_.rubber_band_->visualize(ns_prefix + MDP_PRE_STATE_NS, Visualizer::Yellow(), Visualizer::Yellow(), id + 1);
    transition.starting_state_.planned_rubber_band_->visualize(ns_prefix + MDP_PRE_STATE_NS, Visualizer::Green(), Visualizer::Green(), id + 2);

//    visualizeDeformableObject(ns_prefix + MDP_POST_STATE_NS, transition.ending_state.deform_config_, Visualizer::Red(0.4f), id);
    transition.ending_state_.rubber_band_->visualize(ns_prefix + MDP_POST_STATE_NS, Visualizer::Yellow(0.4f), Visualizer::Yellow(0.4f), id + 1);
    transition.ending_state_.planned_rubber_band_->visualize(ns_prefix + MDP_POST_STATE_NS, Visualizer::Green(0.4f), Visualizer::Green(0.4f), id + 2);
}

void TransitionEstimation::visualizeLearnedTransitions(
        const std::string& ns_prefix) const
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

//////// Saving and loading learned transitions ////////////////////////////////////////////////////////////////////////

bool TransitionEstimation::useStoredTransitions() const
{
    return ROSHelpers::GetParamRequired<bool>(*ph_, "transition_learning/use_stored_transitions", __func__).GetImmutable();
}

void TransitionEstimation::storeTransitions() const
{
    try
    {
        const auto log_folder = GetLogFolder(*nh_);
        arc_utilities::CreateDirectory(log_folder);
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(*ph_, "transition_learning/file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load file_name_prefix from parameter server");
        }

        const std::string file_name_suffix = arc_helpers::GetCurrentTimeAsString();
        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix + ".compressed";
        const std::string full_path = log_folder + file_name;
        ROS_INFO_STREAM_NAMED("transitions", "Saving learned_transitions to " << full_path);

        std::vector<uint8_t> buffer;
        arc_utilities::SerializeVector<StateTransition>(learned_transitions_, buffer, &StateTransition::Serialize);
        ZlibHelpers::CompressAndWriteToFile(buffer, full_path);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM_NAMED("transitions", "Failed to store transitions: " << e.what());
    }
}

void TransitionEstimation::loadSavedTransitions()
{
    try
    {
        const auto log_folder = GetLogFolder(*nh_);
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(*ph_, "transition_learning/file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load file_name_prefix from parameter server");
        }

        std::vector<std::string> suffixes_files_to_load;
        if (!ph_->getParam("transition_learning/file_name_suffixes_to_load", suffixes_files_to_load))
        {
            ROS_ERROR_STREAM_NAMED("transitions", "Cannot find "
                                   << ph_->getNamespace() << "/transition_learning"
                                   << "/file_name_suffixes_to_load on parameter server for "
                                   << __func__
                                   << ": Value must be on paramter sever");
            throw_arc_exception(std::runtime_error, "Unable to find parameter on server");
        }

        for (const auto& suffix : suffixes_files_to_load)
        {
            const auto full_path = log_folder + "/" + file_name_prefix.GetImmutable() + "__" + suffix + ".compressed";
            ROS_INFO_STREAM_NAMED("transitions", "Loading transitions from " << full_path);
            const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(full_path);
            const auto deserializer = [&] (const std::vector<uint8_t>& buf, const uint64_t current)
            {
                return StateTransition::Deserialize(buf, current, template_band_);
            };
            const auto transitions_deserialized = arc_utilities::DeserializeVector<StateTransition>(buffer, 0, deserializer);
            const auto& transitions = transitions_deserialized.first;
            learned_transitions_.insert(learned_transitions_.begin(), transitions.begin(), transitions.end());
        }
        ROS_INFO_STREAM_NAMED("transitions", "Loaded " << learned_transitions_.size() << " transitions from file");

//        for (const auto& t : learned_transitions_)
//        {
//            std::cout << t << std::endl;
//        }
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM_NAMED("transitions", "Error loading transitions from file: " << e.what());
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ostream operators
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& smmap::operator<<(std::ostream& out, const TransitionEstimation::StateTransition& t)
{
    return out << t.toString();
}
