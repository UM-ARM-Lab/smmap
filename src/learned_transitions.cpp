#include "smmap/learned_transitions.h"

#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/serialization_eigen.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/path_utils.hpp>
#include <deformable_manipulation_experiment_params/utility.hpp>

//#define TRANSITION_LEARNING_VERBOSE true
#define TRANSITION_LEARNING_VERBOSE false

using namespace smmap;
using namespace arc_utilities;
using namespace arc_helpers;
using namespace Eigen;
using namespace EigenHelpers;

constexpr char TransitionEstimation::MDP_PRE_STATE_NS[];
constexpr char TransitionEstimation::MDP_TESTING_STATE_NS[];
constexpr char TransitionEstimation::MDP_POST_STATE_NS[];

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Internal types function helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//////// State /////////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t TransitionEstimation::State::serialize(
        std::vector<uint8_t>& buffer) const
{
    const uint64_t starting_size = buffer.size();
    SerializeEigen(deform_config_, buffer);
    rubber_band_->serialize(buffer);
    planned_rubber_band_->serialize(buffer);
    SerializeVector(rope_node_transforms_, buffer, SerializeEigen<Isometry3d>);
    const auto bytes_written = buffer.size() - starting_size;
    // Verify no mistakes were made
    {
        const auto deserialized = Deserialize(buffer, starting_size, *rubber_band_);
        assert(deserialized.second == bytes_written);
        assert(deserialized.first == *this);
    }
    return bytes_written;
}

uint64_t TransitionEstimation::State::deserialize(
        const std::vector<uint8_t>& buffer,
        const uint64_t current)
{
    uint64_t bytes_read = 0;

    const auto deform_config_deserialized =
            DeserializeEigen<ObjectPointSet>(buffer, current + bytes_read);
    deform_config_ = deform_config_deserialized.first;
    bytes_read += deform_config_deserialized.second;
    bytes_read += rubber_band_->deserialize(buffer, current + bytes_read);
    bytes_read += planned_rubber_band_->deserialize(buffer, current + bytes_read);
    const auto rope_node_transforms_deserialized =
            DeserializeVector<Isometry3d, aligned_allocator<Isometry3d>>(
                buffer, current + bytes_read, DeserializeEigen<Isometry3d>);
    rope_node_transforms_ = rope_node_transforms_deserialized.first;
    bytes_read += rope_node_transforms_deserialized.second;
    return bytes_read;
}

uint64_t TransitionEstimation::State::Serialize(
        const State& state,
        std::vector<uint8_t>& buffer)
{
    return state.serialize(buffer);
}

std::pair<TransitionEstimation::State, uint64_t> TransitionEstimation::State::Deserialize(
        const std::vector<uint8_t>& buffer,
        const uint64_t current,
        const RubberBand& template_band)
{
    State state;
    state.planned_rubber_band_ = std::make_shared<RubberBand>(template_band);
    state.rubber_band_ = std::make_shared<RubberBand>(template_band);
    const auto bytes_read = state.deserialize(buffer, current);
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

//////// StateTransition ///////////////////////////////////////////////////////////////////////////////////////////////

uint64_t TransitionEstimation::StateTransition::serialize(
        std::vector<uint8_t>& buffer) const
{
    const uint64_t starting_bytes = buffer.size();
    starting_state_.serialize(buffer);
    ending_state_.serialize(buffer);
    DeserializePair3dPositions(starting_gripper_positions_, buffer);
    DeserializePair3dPositions(ending_gripper_positions_, buffer);
    SerializeVector<WorldState>(microstep_state_history_, buffer, &WorldState::Serialize);
    SerializeVector<RubberBand::Ptr>(microstep_band_history_, buffer, &RubberBand::Serialize);
    const uint64_t bytes_written = buffer.size() - starting_bytes;

    // Verify no mistakes were made
    {
        const auto deserialized = Deserialize(buffer, starting_bytes, *starting_state_.rubber_band_);
        assert(deserialized.second == bytes_written);
        assert(deserialized.first == *this);
    }

    return bytes_written;
}

uint64_t TransitionEstimation::StateTransition::deserialize(
        const std::vector<uint8_t>& buffer,
        const uint64_t current)
{
    uint64_t bytes_read = 0;

    bytes_read += starting_state_.deserialize(buffer, current + bytes_read);
    bytes_read += ending_state_.deserialize(buffer, current + bytes_read);

    const auto staring_grippers_deserialized = DeserializePair3dPositions(buffer, current + bytes_read);
    starting_gripper_positions_ = staring_grippers_deserialized.first;
    bytes_read += staring_grippers_deserialized.second;

    const auto ending_grippers_deserialized = DeserializePair3dPositions(buffer, current + bytes_read);
    ending_gripper_positions_ = ending_grippers_deserialized.first;
    bytes_read += ending_grippers_deserialized.second;

    const auto microsteps_deserialized = DeserializeVector<WorldState>(buffer, current + bytes_read, &WorldState::Deserialize);
    microstep_state_history_ = microsteps_deserialized.first;
    bytes_read += microsteps_deserialized.second;

    const auto band_deserializer = [&](const std::vector<uint8_t>& buffer_internal, const uint64_t current_internal)
    {
        return RubberBand::Deserialize(buffer_internal, current_internal, *starting_state_.rubber_band_);
    };
    const auto microsteps_bands_deserialized = DeserializeVector<RubberBand::Ptr>(buffer, current + bytes_read, band_deserializer);
    microstep_band_history_ = microsteps_bands_deserialized.first;
    bytes_read += microsteps_bands_deserialized.second;

    return bytes_read;
}

uint64_t TransitionEstimation::StateTransition::Serialize(
        const StateTransition& state_transition,
        std::vector<uint8_t>& buffer)
{
    return state_transition.serialize(buffer);
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
    const auto bytes_read = transition.deserialize(buffer, current);
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

//////// Other /////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Derived, typename OtherDerived>
typename internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type
InvariantTransform(const MatrixBase<Derived>& src, const MatrixBase<OtherDerived>& dst)
{
    typedef typename internal::umeyama_transform_matrix_type<Derived, OtherDerived>::type TransformationMatrixType;
    typedef typename internal::traits<TransformationMatrixType>::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;

    EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsComplex, NUMERIC_TYPE_MUST_BE_REAL)
    EIGEN_STATIC_ASSERT((internal::is_same<Scalar, typename internal::traits<OtherDerived>::Scalar>::value),
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)

    enum { Dimension = EIGEN_SIZE_MIN_PREFER_DYNAMIC(Derived::RowsAtCompileTime, OtherDerived::RowsAtCompileTime) };

    typedef Matrix<Scalar, Dimension, 1> VectorType;
    typedef Matrix<Scalar, Dimension, Dimension> MatrixType;
    typedef typename internal::plain_matrix_type_row_major<Derived>::type RowMajorMatrixType;

    const Index m = src.rows(); // dimension
    const Index n = src.cols(); // number of measurements

    // required for demeaning ...
    const RealScalar one_over_n = RealScalar(1) / static_cast<RealScalar>(n);

    // computation of mean
    const VectorType src_mean = src.rowwise().sum() * one_over_n;
    const VectorType dst_mean = dst.rowwise().sum() * one_over_n;

    // demeaning of src and dst points
    const RowMajorMatrixType src_demean = src.colwise() - src_mean;
    const RowMajorMatrixType dst_demean = dst.colwise() - dst_mean;

    // Compute the covariance between the source and destination
    const MatrixType sigma = one_over_n * dst_demean * src_demean.transpose();

    const JacobiSVD<MatrixType> svd(sigma, ComputeFullU | ComputeFullV);

    // Initialize the resulting transformation with an identity matrix...
    TransformationMatrixType Rt = TransformationMatrixType::Identity(m + 1, m + 1);

    // We want to allow mirroring the data, so ignore the possiblity of putting -1 in S
    Rt.topLeftCorner(m, m) = svd.matrixU() * svd.matrixV().transpose();
    Rt.topRightCorner(m, 1).noalias() = dst_mean - Rt.topLeftCorner(m, m) * src_mean;

    return Rt;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TransitionAdapationResult helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t TransitionEstimation::TransitionAdaptationResult::serialize(std::vector<uint8_t>& buffer) const
{
    const auto starting_bytes = buffer.size();

    default_next_band_->serialize(buffer);
    result_->serialize(buffer);
    SerializeEigen(target_points_to_match_, buffer);
    SerializeEigen(template_points_to_align_, buffer);
    SerializeEigen(invariant_transform_, buffer);
    SerializeEigen(template_planned_band_aligned_to_target_, buffer);
    SerializeEigen(next_band_points_to_smooth_, buffer);
    SerializeEigen(transformed_band_surface_points_, buffer);
    SerializeVector<RubberBand::Ptr>(tightened_transformed_bands_, buffer, &RubberBand::Serialize);
    SerializeEigen(tightened_transformed_bands_surface_, buffer);
    SerializeVector<bool>(foh_values_, buffer, &SerializeFixedSizePOD<bool>);
    SerializeFixedSizePOD(template_misalignment_dist_, buffer);
    SerializeFixedSizePOD(default_band_foh_result_, buffer);
    SerializeFixedSizePOD(default_band_dist_, buffer);
    SerializeFixedSizePOD(band_tighten_delta_, buffer);
    SerializeFixedSizePOD(num_foh_changes_, buffer);

    const auto bytes_written = buffer.size() - starting_bytes;
    const auto deserialized = Deserialize(buffer, starting_bytes, *default_next_band_);
    assert(bytes_written == deserialized.second);
    assert(*this == deserialized.first);
    return bytes_written;
}

uint64_t TransitionEstimation::TransitionAdaptationResult::Serialize(
        const TransitionAdaptationResult& adaptation_result,
        std::vector<uint8_t>& buffer)
{
    return adaptation_result.serialize(buffer);
}

std::pair<TransitionEstimation::TransitionAdaptationResult, uint64_t> TransitionEstimation::TransitionAdaptationResult::Deserialize(
        const std::vector<uint8_t>& buffer,
        const uint64_t current,
        const RubberBand& template_band)
{
    uint64_t pos = current;

    const auto default_next_band_deserialized = RubberBand::Deserialize(buffer, pos, template_band);
    pos += default_next_band_deserialized.second;

    const auto result_band_deserialized = RubberBand::Deserialize(buffer, pos, template_band);
    pos += result_band_deserialized.second;


    const auto target_points_to_match_deserialized = DeserializeEigen<ObjectPointSet>(buffer, pos);
    pos += target_points_to_match_deserialized.second;

    const auto template_points_to_align_deserialized = DeserializeEigen<ObjectPointSet>(buffer, pos);
    pos += template_points_to_align_deserialized.second;

    const auto invariant_transform_deserialized = DeserializeEigen<Isometry3d>(buffer, pos);
    pos += invariant_transform_deserialized.second;

    const auto template_planned_band_aligned_deserialized = DeserializeEigen<ObjectPointSet>(buffer, pos);
    pos += template_planned_band_aligned_deserialized.second;

    const auto next_band_points_to_smooth_deserialized = DeserializeEigen<ObjectPointSet>(buffer, pos);
    pos += next_band_points_to_smooth_deserialized.second;

    const auto transformed_band_surface_points_deserialized = DeserializeEigen<ObjectPointSet>(buffer, pos);
    pos += transformed_band_surface_points_deserialized.second;

    const auto band_deserializer = [&](const std::vector<uint8_t>& buffer_internal, const uint64_t current_internal)
    {
        return RubberBand::Deserialize(buffer_internal, current_internal, template_band);
    };
    const auto tightened_transformed_bands_deserialized = DeserializeVector<RubberBand::Ptr>(buffer, pos, band_deserializer);
    pos += tightened_transformed_bands_deserialized.second;

    const auto tightened_transformed_bands_surface_deserialized = DeserializeEigen<ObjectPointSet>(buffer, pos);
    pos += tightened_transformed_bands_surface_deserialized.second;

    const auto foh_changes_deserialized = DeserializeVector<bool>(buffer, pos, &DeserializeFixedSizePOD<bool>);
    pos += foh_changes_deserialized.second;


    const auto template_misalignment_dist_deserialized = DeserializeFixedSizePOD<double>(buffer, pos);
    pos += template_misalignment_dist_deserialized.second;

    const auto default_band_foh_result_deserialized = DeserializeFixedSizePOD<bool>(buffer, pos);
    pos += default_band_foh_result_deserialized.second;

    const auto default_band_dist_deserialized = DeserializeFixedSizePOD<double>(buffer, pos);
    pos += default_band_dist_deserialized.second;

    const auto band_tighten_delta_deserialized = DeserializeFixedSizePOD<double>(buffer, pos);
    pos += band_tighten_delta_deserialized.second;

    const auto num_foh_changes_deserialized = DeserializeFixedSizePOD<int>(buffer, pos);
    pos += num_foh_changes_deserialized.second;

    TransitionAdaptationResult record
    {
        default_next_band_deserialized.first,
        result_band_deserialized.first,

        target_points_to_match_deserialized.first,
        template_points_to_align_deserialized.first,
        invariant_transform_deserialized.first,
        template_planned_band_aligned_deserialized.first,
        next_band_points_to_smooth_deserialized.first,
        transformed_band_surface_points_deserialized.first,
        tightened_transformed_bands_deserialized.first,
        tightened_transformed_bands_surface_deserialized.first,
        foh_changes_deserialized.first,

        template_misalignment_dist_deserialized.first,
        default_band_foh_result_deserialized.first,
        default_band_dist_deserialized.first,
        band_tighten_delta_deserialized.first,
        num_foh_changes_deserialized.first
    };
    return {record, pos - current};
}

bool TransitionEstimation::TransitionAdaptationResult::operator==(const TransitionAdaptationResult& other) const
{
    if (*default_next_band_ != *other.default_next_band_)
    {
        return false;
    }
    if (*result_ != *other.result_)
    {
        return false;
    }
    if (invariant_transform_.matrix() != other.invariant_transform_.matrix())
    {
        return false;
    }
    if (template_planned_band_aligned_to_target_ != other.template_planned_band_aligned_to_target_)
    {
        return false;
    }
    if (next_band_points_to_smooth_ != other.next_band_points_to_smooth_)
    {
        return false;
    }
    if (transformed_band_surface_points_ != other.transformed_band_surface_points_)
    {
        return false;
    }
    if (tightened_transformed_bands_.size() != other.tightened_transformed_bands_.size())
    {
        return false;
    }
    for (size_t idx = 0; idx < tightened_transformed_bands_.size(); ++idx)
    {
        if (*tightened_transformed_bands_[idx] != *other.tightened_transformed_bands_[idx])
        {
            return false;
        }
    }
    if (foh_values_ != other.foh_values_)
    {
        return false;
    }
    if (template_misalignment_dist_ != other.template_misalignment_dist_)
    {
        return false;
    }
    if (default_band_foh_result_ != other.default_band_foh_result_)
    {
        return false;
    }
    if (default_band_dist_ != other.default_band_dist_)
    {
        return false;
    }
    if (num_foh_changes_ != other.num_foh_changes_)
    {
        return false;
    }
    return true;
}

bool TransitionEstimation::TransitionAdaptationResult::operator!=(const TransitionAdaptationResult& other) const
{
    return !(*this == other);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main TransitionEstimation class
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TransitionEstimation::TransitionEstimation(
        std::shared_ptr<ros::NodeHandle> nh,
        std::shared_ptr<ros::NodeHandle> ph,
        const sdf_tools::SignedDistanceField::ConstPtr& sdf,
        const XYZGrid work_space_grid,
        const Visualizer::Ptr& vis,
        const RubberBand& template_band)
    : nh_(nh)
    , ph_(ph)
    , sdf_(sdf)
    , work_space_grid_(work_space_grid)
    , vis_(vis)

    , default_propogation_confidence_(GetTransitionDefaultPropagationConfidence(*ph_))
    , default_band_dist_threshold_(GetTransitionDefaultBandDistThreshold(*ph_))
    , confidence_threshold_(GetTransitionConfidenceThreshold(*ph_))
    , template_misalignment_scale_factor_(GetTransitionTemplateMisalignmentScaleFactor(*ph_))
    , band_tighten_scale_factor_(GetTransitionTightenDeltaScaleFactor(*ph_))
    , homotopy_changes_scale_factor_(GetTransitionHomotopyChangesScaleFactor(*ph_))

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
        const VectorVector3d& b1,
        const VectorVector3d& b2) const
{
    static const double one_div_min_step_size = 1.0 / (work_space_grid_.minStepDimension() * 0.5);

    // Checks if the straight line between elements of the two paths is collision free
    const auto straight_line_collision_check_fn = [&] (
            const ssize_t b1_ind,
            const ssize_t b2_ind)
    {
        const auto& b1_node = b1[b1_ind];
        const auto& b2_node = b2[b2_ind];

        const size_t num_steps = (size_t)std::ceil((b2_node - b1_node).norm() * one_div_min_step_size);
        if (num_steps == 0)
        {
            return true;
        }

        // Checking 0 and num_steps to catch the endpoints of the line
        for (size_t ind = 0; ind <= num_steps; ++ind)
        {
            const double ratio = (double)ind / (double)num_steps;
            const Vector3d interpolated_point = Interpolate(b1_node, b2_node, ratio);
            if (sdf_->GetImmutable3d(interpolated_point).first < 0.0)
            {
                return false;
            }
        }

        return true;
    };

    return FirstOrderDeformation::CheckFirstOrderDeformation(
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
            // TODO: why is this here?
            continue;
        }
        bands.push_back(std::make_shared<RubberBand>(template_band_));
        if (!bands.back()->resetBand(microsteps[idx]))
        {
            ROS_ERROR_NAMED("transitions", "Unable to extract surface");
            PressAnyKeyToContinue("Unable to extract surface");
        }
    }

    return bands;
}

/////// Learning transitions ///////////////////////////////////////////////////////////////////////////////////////////

Maybe::Maybe<TransitionEstimation::StateTransition> TransitionEstimation::findMostRecentBadTransition(
        const std::vector<std::pair<TransitionEstimation::State, std::vector<WorldState>>>& trajectory,
        const bool visualize) const
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
                              << "Last state does not have a homotopy mismatch, "
                              << "returning no transition");
        return Maybe::Maybe<StateTransition>();
    }

    // Search backwards through the list for the last time we transitioned
    // from a matching first order homotopy to not matching
    for (size_t idx = trajectory.size() - 1; idx > 0; --idx)
    {
        const auto& start_state = trajectory[idx - 1].first;
        const auto& end_state = trajectory[idx].first;

        if (visualize)
        {
            start_state.planned_rubber_band_->visualize(
                        "TRANSITION_LEARNING_LOOKING_BACK_PLANNED",
                        Visualizer::Blue(),
                        Visualizer::Blue(),
                        1);

            start_state.rubber_band_->visualize(
                        "TRANSITION_LEARNING_LOOKING_BACK_EXECUTED",
                        Visualizer::Yellow(),
                        Visualizer::Yellow(),
                        1);

            end_state.planned_rubber_band_->visualize(
                        "TRANSITION_LEARNING_LOOKING_BACK_PLANNED",
                        Visualizer::Blue(),
                        Visualizer::Blue(),
                        2);

            end_state.rubber_band_->visualize(
                        "TRANSITION_LEARNING_LOOKING_BACK_EXECUTED",
                        Visualizer::Yellow(),
                        Visualizer::Yellow(),
                        2);

            RubberBand::VisualizeBandSurface(
                        vis_,
                        reduceMicrostepsToBands(trajectory[idx].second),
                        Visualizer::Green(),
                        Visualizer::Red(),
                        "TRANSITION_LEARNING_LOOKING_BACK_BAND_SURFACE",
                        1);
        }


        // If the first order homotopy check passes, then the actual rubber band
        // and the planned rubber band are in the same first order homotopy class
        if (checkFirstOrderHomotopy(*start_state.rubber_band_, *end_state.planned_rubber_band_))
        {
            StateTransition transition
            {
                start_state,
                end_state,
                start_state.planned_rubber_band_->getEndpoints(),
                end_state.planned_rubber_band_->getEndpoints(),
                trajectory[idx].second,
                reduceMicrostepsToBands(trajectory[idx].second)
            };
            return Maybe::Maybe<StateTransition>(transition);
        }

        if (visualize)
        {
            arc_helpers::Sleep(1.0);
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
    learned_transitions_.push_back(transition);
    storeTransitions();
}

std::vector<RubberBand> TransitionEstimation::extractBandSurface(const StateTransition& transition) const
{
    std::vector<RubberBand> band_surface;
    band_surface.reserve(transition.microstep_state_history_.size() + 1);
    band_surface.push_back(*transition.starting_state_.rubber_band_);
    for (const auto& state : transition.microstep_state_history_)
    {
        RubberBand temp_band(band_surface[0]);
        if (!temp_band.resetBand(state))
        {
            PressAnyKeyToContinue("creating band surface failed");
        }
        band_surface.push_back(temp_band);
    }
    return band_surface;
}

//////// Using transitions /////////////////////////////////////////////////////////////////////////////////////////////

const std::vector<TransitionEstimation::StateTransition>& TransitionEstimation::transitions() const
{
    return learned_transitions_;
}

// Blindy applies the transition without regard for applicability
TransitionEstimation::TransitionAdaptationResult TransitionEstimation::generateTransition(
        const StateTransition& stored_trans,
        const RubberBand& test_band_start,
        const PairGripperPositions& ending_gripper_positions) const
{
    auto default_next_band = std::make_shared<RubberBand>(test_band_start);
    default_next_band->forwardPropagate(ending_gripper_positions, false);

    // Extract the best transform based on the invariants in the system
    // (memorized data) into the target points (test data)
    #warning "Hard-coded div 4 to account for num simsteps per gripper command"
    const auto num_gripper_steps = stored_trans.microstep_state_history_.size() / 4 * 100;
    const ObjectPointSet warping_target_points = RubberBand::PointsFromBandAndGrippers(
                test_band_start,
                test_band_start.getEndpoints(),
                ending_gripper_positions,
                num_gripper_steps);
    const ObjectPointSet warping_template_points_planned = RubberBand::PointsFromBandAndGrippers(
                *stored_trans.starting_state_.planned_rubber_band_,
                stored_trans.starting_gripper_positions_,
                stored_trans.ending_gripper_positions_,
                num_gripper_steps);

    const Matrix3d xytransform = InvariantTransform(
                warping_template_points_planned.topRows<2>(),
                warping_target_points.topRows<2>());

    const double z_shift =
            warping_target_points.bottomRows<1>().mean() -
            warping_template_points_planned.bottomRows<1>().mean();

    Isometry3d transform = Isometry3d::Identity();
    transform.matrix().topLeftCorner<2, 2>() = xytransform.topLeftCorner<2, 2>();
    transform.matrix().topRightCorner<2, 1>() = xytransform.topRightCorner<2, 1>();
    transform.matrix()(2, 3) = z_shift;

    // Align the source (memorized) points to the target
    const ObjectPointSet template_planned_band_aligned_to_target = transform * warping_template_points_planned;
    const double template_misalignment_dist = (warping_target_points - template_planned_band_aligned_to_target).norm();

    // Create a new band based on the memorized ending state, once aligned.
    auto next_band = std::make_shared<RubberBand>(test_band_start);
    const auto next_band_transformed_points = TransformData(transform, stored_trans.ending_state_.rubber_band_->getVectorRepresentation());
    const auto next_band_points_to_smooth = RubberBand::PointsFromBandPointsAndGripperTargets(next_band_transformed_points, ending_gripper_positions, 1);
    if (!next_band->setPointsAndSmooth(next_band_points_to_smooth))
    {
        throw_arc_exception(std::runtime_error, "Unable to smooth band");
    }

    // Measure the difference between this result, and the default next step
    const bool default_band_foh_result = checkFirstOrderHomotopy(*default_next_band, *next_band);
    const auto default_band_dist_sq = default_next_band->distanceSq(*next_band);

    // Measure how much we changed the band when we re-tightened
    const double band_tighten_delta = next_band->distance(TransformData(transform, stored_trans.ending_state_.rubber_band_->upsampleBand()));

    // Map the entire memorized band surface to the new environment, and check that all bands are "mapable"
    const auto stored_bands = stored_trans.microstep_band_history_;
    std::vector<RubberBand::Ptr> tightened_transformed_bands_from_stored_bands(stored_bands.size(), nullptr);
    for (size_t band_surface_idx = 0; band_surface_idx < stored_bands.size(); ++band_surface_idx)
    {
        // Transform the stored band into the test band space
        const auto& stored_band = stored_bands[band_surface_idx];
        const auto transformed_band = TransformData(transform, stored_band->getVectorRepresentation());
        // Move the endpoints to the line along the test action vector
        assert(stored_bands.size() > 1);
        const double ratio = (double)band_surface_idx / (double)(stored_bands.size() - 1);
        const auto gripper_targets = Interpolate(test_band_start.getEndpoints(), ending_gripper_positions, ratio);
        const auto points_to_smooth = RubberBand::PointsFromBandPointsAndGripperTargets(transformed_band, gripper_targets, 1);

        auto band = std::make_shared<RubberBand>(*stored_band);
        if (!band->setPointsAndSmooth(points_to_smooth))
        {
            throw_arc_exception(std::runtime_error, "Unable to map band surface");
        }
        tightened_transformed_bands_from_stored_bands[band_surface_idx] = band;
    }

    std::vector<bool> foh_values;
    for (size_t idx = 0; idx < tightened_transformed_bands_from_stored_bands.size() - 1; ++idx)
    {
        RubberBand::Ptr b1 = tightened_transformed_bands_from_stored_bands[idx];
        RubberBand::Ptr b2 = tightened_transformed_bands_from_stored_bands[idx + 1];
        foh_values.push_back(checkFirstOrderHomotopy(*b1, *b2));
    }
    int num_foh_changes = 0;
    for (size_t idx = 0; idx < foh_values.size() - 1; ++idx)
    {
        if (foh_values[idx] != foh_values[idx + 1])
        {
            ++num_foh_changes;
        }
    }

    return TransitionAdaptationResult
    {
        default_next_band,
        next_band,

        warping_target_points,
        warping_template_points_planned,
        transform,
        template_planned_band_aligned_to_target,
        next_band_points_to_smooth,
        transform * RubberBand::AggregateBandPoints(stored_trans.microstep_band_history_),
        tightened_transformed_bands_from_stored_bands,
        RubberBand::AggregateBandPoints(tightened_transformed_bands_from_stored_bands),
        foh_values,

        template_misalignment_dist,
        default_band_foh_result,
        default_band_dist_sq,
        band_tighten_delta,
        num_foh_changes
    };
}

std::vector<std::pair<RubberBand::Ptr, double>> TransitionEstimation::estimateTransitions(
        const RubberBand& test_band_start,
        const PairGripperPositions& ending_gripper_positions,
        const bool verbose) const
{
    std::vector<std::pair<RubberBand::Ptr, double>> transitions;

    auto default_next_band = std::make_shared<RubberBand>(test_band_start);
    default_next_band->forwardPropagate(ending_gripper_positions, verbose);
    // Only add this transition if the band is not overstretched after moving
    if (default_next_band->isOverstretched())
    {
        ROS_INFO_COND_NAMED(TRANSITION_LEARNING_VERBOSE, "rrt.prop", "Stopped due to band overstretch");
    }
    else
    {
        transitions.push_back({default_next_band, default_propogation_confidence_});
    }

    for (size_t transition_idx = 0; transition_idx < learned_transitions_.size(); ++transition_idx)
    {
        const StateTransition& stored_trans = learned_transitions_[transition_idx];
//        const std::vector<RubberBand>& stored_bands = learned_band_surfaces_[transition_idx];

        // TODO: Update this to be in the new environment somehow
//        // If the stored band is in a different homotopy class, skip it
//        if (!checkFirstOrderHomotopy(test_band_start, *stored_trans.starting_state_.planned_rubber_band_))
//        {
//            continue;
//        }

        // Extract the best SE(3) transform that transforms the template points
        // (memorized data) into the target points (test data)
        const auto num_gripper_steps = stored_trans.microstep_state_history_.size() / 4;
        ObjectPointSet warping_template_points_planned = RubberBand::PointsFromBandAndGrippers(
                    *stored_trans.starting_state_.planned_rubber_band_,
                    stored_trans.starting_gripper_positions_,
                    stored_trans.ending_gripper_positions_,
                    num_gripper_steps);
        ObjectPointSet warping_target_points = RubberBand::PointsFromBandAndGrippers(
                    test_band_start,
                    test_band_start.getEndpoints(),
                    ending_gripper_positions,
                    num_gripper_steps);
        const Isometry3d transform = Isometry3d(umeyama(warping_template_points_planned, warping_target_points, true));

        const ObjectPointSet template_planned_band_aligned_to_target = transform * warping_template_points_planned;
        // Stop processing if we are already below the threshold
        const double template_misalignment_dist = (warping_target_points - template_planned_band_aligned_to_target).norm();
        double confidence = std::exp(-template_misalignment_scale_factor_ * template_misalignment_dist);
        if (confidence < confidence_threshold_)
        {
            continue;
        }

        // Visualization
        if (true)
        {
//            const ObjectPointSet stored_bands_planned_aligned_to_target = transform * RubberBand::AggregateBandPoints(stored_bands);

            vis_->visualizePoints("TEMPLATE_POINTS_PLANNED",
                                  warping_template_points_planned,
                                  Visualizer::Green(), 1, 0.002);

            vis_->visualizePoints("TARGET_POINTS",
                                  warping_target_points,
                                  Visualizer::Yellow(), 1, 0.002);

            vis_->visualizePoints("TEMPLATE_POINTS_PLANNED_ALIGNED",
                                  template_planned_band_aligned_to_target,
                                  Visualizer::Magenta(), 1, 0.002);

//            RubberBand::VisualizeBandSurface(vis_,
//                                             stored_bands,
//                                             Visualizer::Blue(),
//                                             Visualizer::Red(),
//                                             "stored_bands", 1);

//            RubberBand::VisualizeBandSurface(vis_,
//                                             stored_bands_planned_aligned_to_target,
//                                             stored_bands.size(),
//                                             Visualizer::Blue(),
//                                             Visualizer::Red(),
//                                             "stored_bands_planned_aligned", 1);
        }

        // Align ending state band, then tighten the resulting band
        auto next_band = std::make_shared<RubberBand>(test_band_start);
        {
            vis_->visualizePoints("MEMORIZED_NEXT_STATE",
                                  stored_trans.ending_state_.rubber_band_->upsampleBand(),
                                  Visualizer::Blue(), 1, 0.002);
            vis_->visualizePoints("TRANSFORMED_MEMORIZED_NEXT_STATE",
                                  TransformData(transform, stored_trans.ending_state_.rubber_band_->upsampleBand()),
                                  Visualizer::Cyan(), 1, 0.002);
            const auto transformed_points = TransformData(transform, stored_trans.ending_state_.rubber_band_->getVectorRepresentation());
            const auto points_to_smooth = RubberBand::PointsFromBandPointsAndGripperTargets(transformed_points, ending_gripper_positions, 1);
            if (!next_band->setPointsAndSmooth(points_to_smooth))
            {
                continue;
            }
        }
        // Stop processing this stored transition if the resulting band is overstretched
        if (next_band->isOverstretched())
        {
            continue;
        }

        // Stop processing if we are already below the threshold
        const double band_tighten_delta = next_band->distance(TransformData(transform, stored_trans.ending_state_.rubber_band_->upsampleBand()));
        confidence *= std::exp(-band_tighten_scale_factor_ * band_tighten_delta);
        if (confidence < confidence_threshold_)
        {
            continue;
        }

        // Visualization
        if (true)
        {
            default_next_band->visualize("DEFAULT_TRANSITION", Visualizer::Blue(), Visualizer::Blue(), 1);
            next_band->visualize("MEMORY_ADAPTED_TRANSITION", Visualizer::Red(), Visualizer::Red(), 1);
        }

        // Check if there is a "meaningful" difference between the default band, and this result
        // If there is not a meaningful difference, then stop processing this stored transition
        if (checkFirstOrderHomotopy(*default_next_band, *next_band))
        {
            const auto band_dist_sq = default_next_band->distanceSq(*next_band);
            if (std::sqrt(band_dist_sq) < default_band_dist_threshold_)
            {
//                    PressKeyToContinue("Stored transition is producing a result similar to the default");
                continue;
            }
            else
            {
//                    std::cout << "default to next dist: " << std::sqrt(band_dist_sq) << std::endl;
            }
        }
        else
        {
//            PressKeyToContinue("FO did not match ");
        }

        // Apply the best SE(3) transform to the memorized stored bands
        // Project the transformed points out of collision and retighten
//        std::vector<RubberBand> transformed_bands_from_stored_bands;
//        transformed_bands_from_stored_bands.reserve(stored_bands.size());
//        bool all_bands_mappable = true;
//        for (size_t band_surface_idx = 0; band_surface_idx < stored_bands.size(); ++band_surface_idx)
//        {
//            // Transform the stored band into the test band space
//            const auto& stored_band = stored_bands[band_surface_idx];
//            const auto transformed_band = TransformData(transform, stored_band.getVectorRepresentation());
//            // Move the endpoints to the line along the test action vector
//            assert(stored_bands.size() > 1);
//            const double ratio = (double)band_surface_idx / (double)(stored_bands.size() - 1);
//            const auto gripper_targets = Interpolate(test_band_start.getEndpoints(), ending_gripper_positions, ratio);
//            const auto points_to_smooth = RubberBand::PointsFromBandPointsAndGripperTargets(transformed_band, gripper_targets, 1);

//            RubberBand band(stored_band);
//            if (!band.setPointsAndSmooth(points_to_smooth))
//            {
//                all_bands_mappable = false;
//                break;
//            }
//            transformed_bands_from_stored_bands.push_back(band);
//        }
//        if (!all_bands_mappable)
//        {
//            continue;
//        }

//        if (true)
//        {
//            RubberBand::VisualizeBandSurface(vis_,
//                                             transformed_bands_from_stored_bands,
//                                             Visualizer::Seafoam(),
//                                             Visualizer::Orange(),
//                                             "stored_bands_planned_aligned_retightened", 1);
//        }

        PressAnyKeyToContinue("Made it to the end of the stored trans loop");
        transitions.push_back({next_band, confidence});
    }

    return transitions;
}

//////// Visualization /////////////////////////////////////////////////////////////////////////////////////////////////

void TransitionEstimation::visualizeTransition(
        const StateTransition& transition,
        const int32_t id,
        const std::string& ns_prefix) const
{
    VisualizeTransition(vis_, transition, id, ns_prefix);
}

void TransitionEstimation::VisualizeTransition(
        const Visualizer::Ptr& vis,
        const StateTransition& transition,
        const int32_t id,
        const std::string& ns_prefix)
{
    (void)vis;
    //    visualizeDeformableObject(ns_prefix + MDP_PRE_STATE_NS, transition.starting_state.deform_config_, Visualizer::Red(), id);
        transition.starting_state_.rubber_band_->visualize(ns_prefix + MDP_PRE_STATE_NS, Visualizer::Yellow(), Visualizer::Yellow(), id + 1);
        transition.starting_state_.planned_rubber_band_->visualize(ns_prefix + MDP_PRE_STATE_NS, Visualizer::Green(), Visualizer::Green(), id + 2);

    //    visualizeDeformableObject(ns_prefix + MDP_POST_STATE_NS, transition.ending_state.deform_config_, Visualizer::Red(0.4f), id);
        transition.ending_state_.rubber_band_->visualize(ns_prefix + MDP_POST_STATE_NS, Visualizer::Yellow(1.0f), Visualizer::Yellow(1.0f), id + 1);
        transition.ending_state_.planned_rubber_band_->visualize(ns_prefix + MDP_POST_STATE_NS, Visualizer::Green(1.0f), Visualizer::Green(1.0f), id + 2);
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
    return ROSHelpers::GetParamRequired<bool>(*ph_, "transition_estimation/use_stored_transitions", __func__).GetImmutable();
}

void TransitionEstimation::storeTransitions() const
{
    try
    {
        const auto log_folder = GetLogFolder(*nh_);
        CreateDirectory(log_folder);
        const auto file_name_prefix = ROSHelpers::GetParamRequired<std::string>(*ph_, "transition_estimation/file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load file_name_prefix from parameter server");
        }
        const std::string file_name_suffix = GetCurrentTimeAsString();
        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix + ".compressed";
        const std::string full_path = log_folder + file_name;
        ROS_INFO_STREAM_NAMED("transitions", "Saving learned_transitions to " << full_path);

        std::vector<uint8_t> buffer;
        SerializeVector<StateTransition>(learned_transitions_, buffer, &StateTransition::Serialize);
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
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(*ph_, "transition_estimation/file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load file_name_prefix from parameter server");
        }

        std::vector<std::string> suffixes_files_to_load;
        if (!ph_->getParam("transition_estimation/file_name_suffixes_to_load", suffixes_files_to_load))
        {
            ROS_ERROR_STREAM_NAMED("transitions", "Cannot find "
                                   << ph_->getNamespace() << "/transition_estimation"
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
            const auto transitions_deserialized = DeserializeVector<StateTransition>(buffer, 0, deserializer);
            const auto& transitions = transitions_deserialized.first;
            learned_transitions_.insert(learned_transitions_.begin(), transitions.begin(), transitions.end());
        }
        ROS_INFO_STREAM_NAMED("transitions", "Loaded " << learned_transitions_.size() << " transitions from file");

        learned_band_surfaces_.reserve(learned_transitions_.size());
        for (const auto& transition : learned_transitions_)
        {
            learned_band_surfaces_.push_back(extractBandSurface(transition));
        }
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
