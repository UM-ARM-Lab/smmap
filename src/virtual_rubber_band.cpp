#include "smmap/virtual_rubber_band.h"

#include <arc_utilities/shortcut_smoothing.hpp>

using namespace smmap;

const std::string VirtualRubberBand::BAND_POST_FORWARD_STEP_NS          = "band_post_forward_step";
const std::string VirtualRubberBand::BAND_POST_SUBDIVIDE_STEP_NS        = "band_post_subdivide_step";
const std::string VirtualRubberBand::BAND_POST_SHORTCUT_SMOOTHING_NS    = "band_post_shortcut_smoothing";
const std::string VirtualRubberBand::BAND_BADNESS_NS                    = "band_badness";

#pragma message "Magic number for smoothing distance here"
#define MIN_BAND_SMOOTHING_INDEX_DISTANCE (10)

#pragma message "Magic number - virtual rubber band constructor parameters"
VirtualRubberBand::VirtualRubberBand(
        const Eigen::Vector3d& start_point,
        const Eigen::Vector3d& end_point,
        const std::shared_ptr<DijkstrasCoverageTask>& task,
        const Visualizer& vis,
        std::mt19937_64& generator)
    : task_(task)
    , sdf_(task_->environment_sdf_)
    , vis_(vis)
    , max_integration_step_size_(sdf_.GetResolution() / 10.0)
    , max_distance_between_rubber_band_points_(task_->work_space_grid_.minStepDimension() / 2.0)
    , num_smoothing_ittrs_(200)
    , min_object_radius_(0.04)
    , max_total_band_distance_((end_point - start_point).norm() * task_->maxStretchFactor())
    , generator_(generator)
{
    band_ = {start_point, end_point};
    resampleBand(false);
}

VirtualRubberBand::VirtualRubberBand(
        const EigenHelpers::VectorVector3d starting_points,
        const double max_total_band_distance,
        const std::shared_ptr<DijkstrasCoverageTask>& task,
        const Visualizer& vis,
        std::mt19937_64& generator)
    : task_(task)
    , sdf_(task_->environment_sdf_)
    , vis_(vis)
    , max_integration_step_size_(sdf_.GetResolution() / 10.0)
    , max_distance_between_rubber_band_points_(task_->work_space_grid_.minStepDimension() / 2.0)
    , num_smoothing_ittrs_(200)
    , min_object_radius_(0.04)
    , max_total_band_distance_(max_total_band_distance)
    , band_(starting_points)
    , generator_(generator)
{
    resampleBand(false);
    shortcutSmoothBand(false);
}

smmap::VirtualRubberBand& VirtualRubberBand::operator=(const smmap::VirtualRubberBand& other)
{
    assert(task_ == other.task_);
    assert(&sdf_ == &(other.sdf_));
    assert(&vis_ == &(other.vis_));
    assert(&generator_ == &(other.generator_));

    assert(max_integration_step_size_ == other.max_integration_step_size_);
    assert(max_distance_between_rubber_band_points_ == other.max_distance_between_rubber_band_points_);
    assert(num_smoothing_ittrs_ == other.num_smoothing_ittrs_);
    assert(min_object_radius_ == other.min_object_radius_);
    assert(max_total_band_distance_ == other.max_total_band_distance_);

    band_ = other.band_;

    return *this;
}

void VirtualRubberBand::setPointsAndSmooth(const EigenHelpers::VectorVector3d& points)
{
    band_ = points;
    resampleBand(false);
    shortcutSmoothBand(false);
}

/**
 * @brief VirtualRubberBand::forwardSimulateVirtualRubberBandToEndpointTargets This function assumes that the endpoints
 * that are passed are collision free, and the path between them can be resampled without any issues. We may have to
 * project the resampled points out of collision, but it should be feasible to do so.
 * @param first_endpoint_target
 * @param second_endpoint_target
 * @param verbose
 * @return
 */
const EigenHelpers::VectorVector3d& VirtualRubberBand::forwardSimulateVirtualRubberBandToEndpointTargets(
        const Eigen::Vector3d first_endpoint_target,
        const Eigen::Vector3d second_endpoint_target,
        bool verbose)
{
    // Add the new endpoints, then let the resample and smooth process handle the propogation
    band_.insert(band_.begin(), first_endpoint_target);
    band_.push_back(second_endpoint_target);
    resampleBand(verbose);
    shortcutSmoothBand(verbose);
    return band_;
}

const EigenHelpers::VectorVector3d& VirtualRubberBand::getVectorRepresentation() const
{
    return band_;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> VirtualRubberBand::getEndpoints() const
{
    return std::make_pair(band_.front(), band_.back());
}

double VirtualRubberBand::maxSafeLength() const
{
    return max_total_band_distance_;
}

double VirtualRubberBand::totalLength() const
{
    return EigenHelpers::CalculateTotalDistance(band_);
}

bool VirtualRubberBand::isOverstretched() const
{
    return totalLength() > max_total_band_distance_;
}

void VirtualRubberBand::visualize(
        const std::string& marker_name,
        const std_msgs::ColorRGBA& safe_color,
        const std_msgs::ColorRGBA& overstretched_color,
        const int32_t id,
        const bool visualization_enabled) const
{
    if (visualization_enabled)
    {
        if (isOverstretched())
        {
            vis_.visualizeXYZTrajectory(marker_name, band_, overstretched_color, id);
        }
        else
        {
            vis_.visualizeXYZTrajectory(marker_name, band_, safe_color, id);
        }
    }
}

void VirtualRubberBand::resampleBand(const bool verbose)
{
    (void)verbose;

    const auto state_distance_fn = [] (const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
    {
        return (p1 - p2).norm();
    };
    const auto state_interpolation_fn = [&] (const Eigen::Vector3d& prev, const Eigen::Vector3d& curr, const double ratio)
    {
//        vis_.visualizePoints("band_test_resample_prev", {prev}, {Visualizer::Red()}, 1, 0.005);
//        vis_.visualizePoints("band_test_resample_curr", {curr}, {Visualizer::Blue()}, 1, 0.005);
        const auto pre_project = EigenHelpers::Interpolate(prev, curr, ratio);
//        vis_.visualizePoints("band_test_resample_inerp", {pre_project}, {Visualizer::Cyan()}, 1, 0.004);
        const auto projected = sdf_.ProjectOutOfCollision3d(pre_project);
//        vis_.visualizePoints("band_test_resample_result", {projected}, {Visualizer::Green()}, 1, 0.004);
        return projected;
    };

    // Continue to smooth with projected points until the result stabilizes
    auto resample_result = shortcut_smoothing::ResamplePath(
                band_, max_distance_between_rubber_band_points_, state_distance_fn, state_interpolation_fn);
    int resample_count = 0;

    // TODO: This can cause an infinite loop
    // https://github.com/UM-ARM-Lab/smmap/issues/6
    while (resample_result != band_)
    {
//        if (resample_count >= 1)
//        {
//            visualize("band_test_resample_pre_resample", Visualizer::Red(), Visualizer::Red(), 1, true);
//            std::cout << "Pre-resample: " << std::endl;
//            std::cout << PrettyPrint::PrettyPrint(band_, false, "\n") << std::endl << std::endl;

//            visualize("band_test_post_resample", Visualizer::Red(), Visualizer::Red(), 1, true);
//            std::cout << "Post-resample: " << std::endl;
//            std::cout << PrettyPrint::PrettyPrint(resample_result, false, "\n") << std::endl << std::endl;
//        }

        // Accept the previous resample as it was useful
        band_ = resample_result;
        resample_count++;

        if (resample_count == 10)
        {
            ROS_WARN("Rubber Band Smoothing is probably caught in an infinite loop while resampling");
        }

        resample_result = shortcut_smoothing::ResamplePath<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                    band_, max_distance_between_rubber_band_points_, state_distance_fn, state_interpolation_fn);
    }
//    std::cout << "resample count: " << resample_count << std::endl;

    // Double check the results; this is here to catch cases where the projection may be creating points that are too far apart
    for (size_t idx = 0; idx < band_.size() - 1; ++idx)
    {
        if ((band_[idx] - band_[idx + 1]).norm() > max_distance_between_rubber_band_points_)
        {
            visualize(BAND_BADNESS_NS, Visualizer::Red(), Visualizer::Red(), 1, true);

            std::cerr << "Problem at index " << idx << ": " << (band_[idx] - band_[idx + 1]).norm() - max_distance_between_rubber_band_points_ << " Post-resample distances:\n";
            for (size_t idx = 0; idx < band_.size() - 1; ++idx)
            {
                std::cout << (band_[idx] - band_[idx + 1]).norm() << " ";
            }
            std::cout << std::endl;

            #warning "Band resample needs refinement to avoid this crash"
            assert(false && "After resampling, the maximum distance between points is still violated. Please collect all log files from this execution and file a bug report.");
        }
    }
}

void VirtualRubberBand::shortcutSmoothBand(const bool verbose)
{
    const auto sdf_collision_fn = [&] (const Eigen::Vector3d& location)
    {
        return sdf_.EstimateDistance3d(location).first < 0.0;
    };
    for (int smoothing_ittr = 0; smoothing_ittr < num_smoothing_ittrs_; ++smoothing_ittr)
    {
        std::uniform_int_distribution<ssize_t> first_distribution(0, band_.size() - 1);
        const size_t first_ind = first_distribution(generator_);

        const ssize_t max_smooth_distance = std::max((ssize_t)MIN_BAND_SMOOTHING_INDEX_DISTANCE, (ssize_t)std::floor(sdf_.EstimateDistance3d(band_[first_ind]).first / max_distance_between_rubber_band_points_));
        std::uniform_int_distribution<ssize_t> second_distribution(-max_smooth_distance, max_smooth_distance);
        const ssize_t second_offset = second_distribution(generator_);
        const size_t second_ind = (size_t)arc_helpers::ClampValue<ssize_t>(first_ind + second_offset, 0, (ssize_t)band_.size() - 1);

        if (first_ind != second_ind)
        {
            band_ = shortcut_smoothing::InterpolateWithCollisionCheck(band_, first_ind, second_ind, max_distance_between_rubber_band_points_, sdf_collision_fn);
        }

        if (verbose)
        {
            vis_.visualizeXYZTrajectory(BAND_POST_SHORTCUT_SMOOTHING_NS, band_, Visualizer::Red(), smoothing_ittr);
        }
    }
}
