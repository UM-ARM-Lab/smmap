#include "smmap/virtual_rubber_band.h"

#include <arc_utilities/shortcut_smoothing.hpp>

using namespace smmap;

const std::string VirtualRubberBand::BAND_POST_FORWARD_STEP_NS          = "band_post_forward_step";
const std::string VirtualRubberBand::BAND_POST_SUBDIVIDE_STEP_NS        = "band_post_subdivide_step";
const std::string VirtualRubberBand::BAND_POST_SHORTCUT_SMOOTHING_NS    = "band_post_shortcut_smoothing";
const std::string VirtualRubberBand::BAND_BADNESS_NS                    = "band_badness";

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
    , max_total_band_distance_((end_point - start_point).norm() * task_->maxOverstretchFactor())
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


const EigenHelpers::VectorVector3d& VirtualRubberBand::forwardSimulateVirtualRubberBandToEndpointTargets(
        const Eigen::Vector3d first_endpoint_target,
        const Eigen::Vector3d second_endpoint_target,
        bool verbose)
{
    vis_.deleteObjects(BAND_POST_FORWARD_STEP_NS, 1, 30);
    vis_.deleteObjects(BAND_POST_SUBDIVIDE_STEP_NS, 1, 30);
    vis_.deleteObjects(BAND_POST_SHORTCUT_SMOOTHING_NS, 1, 500);

    int32_t first_step_failure = -1;

    double distance_to_first_target = (first_endpoint_target - band_.front()).norm();
    double distance_to_second_target = (second_endpoint_target - band_.back()).norm();
    int32_t integration_step_ind = 1;

    while (distance_to_first_target > 0.0 || distance_to_second_target > 0.0)
    {
        // Collect data for later use
        const std::vector<double> cummulative_distances_along_band = EigenHelpers::CalculateCumulativeDistances(band_);
        const double total_band_length = cummulative_distances_along_band.back();
        const Eigen::Vector3d first_endpoint_at_start_of_loop = band_.front();
        const Eigen::Vector3d second_endpoint_at_start_of_loop = band_.back();

        // Interpolate then project the endpoints
        {
            // First endpoint
//            {
                Eigen::Vector3d first_endpoint_interm_pos = first_endpoint_target;
                if (distance_to_first_target > max_integration_step_size_)
                {
                    first_endpoint_interm_pos = EigenHelpers::Interpolate(band_.front(), first_endpoint_target, max_integration_step_size_ / distance_to_first_target);
                }
                band_.front() = sdf_.ProjectOutOfCollision3d(first_endpoint_interm_pos);
//            }
            // Second endpoint
//            {
                Eigen::Vector3d second_endpoint_interm_pos = second_endpoint_target;
                if (distance_to_second_target > max_integration_step_size_)
                {
                    second_endpoint_interm_pos = EigenHelpers::Interpolate(band_.back(), second_endpoint_target, max_integration_step_size_ / distance_to_second_target);
                }
                band_.back() = sdf_.ProjectOutOfCollision3d(second_endpoint_interm_pos);
//            }


            // Ensure that we are making forward progress
            if ((distance_to_first_target  > 0.0 && (first_endpoint_target  - band_.front()).norm() >= distance_to_first_target) ||
                (distance_to_second_target > 0.0 && (second_endpoint_target - band_.back() ).norm() >= distance_to_second_target))
            {
                if (first_step_failure == -1)
                {
                    first_step_failure = integration_step_ind;
                }

                std::cerr << "Rubber band forward simulation stuck - endpoints not moving forward\n";

                std::cerr << "First endpoint target:       " << PrettyPrint::PrettyPrint(first_endpoint_target) << " SDF Distance: " << sdf_.Get3d(first_endpoint_target) << std::endl;
                std::cerr << "First endpoint at start:     " << PrettyPrint::PrettyPrint(first_endpoint_at_start_of_loop) << std::endl;
                std::cerr << "First target pre projection: " << PrettyPrint::PrettyPrint(first_endpoint_interm_pos) << std::endl;
                std::cerr << "First endpoint after delta:  " << PrettyPrint::PrettyPrint(band_.front()) << std::endl;
                std::cerr << "First distance at start: " << distance_to_first_target << std::endl;
                std::cerr << "First distance at end:   " << (first_endpoint_target  - band_.front()).norm() << std::endl;

                std::cerr << "Second endpoint target:       " << PrettyPrint::PrettyPrint(first_endpoint_target) << " SDF Distance: " << sdf_.Get3d(second_endpoint_target) << std::endl;
                std::cerr << "Second endpoint at start:     " << PrettyPrint::PrettyPrint(second_endpoint_at_start_of_loop) << std::endl;
                std::cerr << "Second target pre projection: " << PrettyPrint::PrettyPrint(second_endpoint_interm_pos) << std::endl;
                std::cerr << "Second endpoint after delta:  " << PrettyPrint::PrettyPrint(band_.back()) << std::endl;
                std::cerr << "Second distance at start: " << distance_to_second_target << std::endl;
                std::cerr << "Second distance at end:   " << (second_endpoint_target  - band_.back()).norm() << std::endl;

                if (first_step_failure != -1 && integration_step_ind > first_step_failure + 10)
                {
                    return band_;
                }
            }
        }

        // Determine how far each endpoint moved - used to decide the motion of the rest of the band
        const Eigen::Vector3d first_endpoint_delta = band_.front() - first_endpoint_at_start_of_loop;
        const Eigen::Vector3d second_endpoint_delta = band_.back() - second_endpoint_at_start_of_loop;

        // First move each node forward, projecting out of collision as needed
        for (size_t rubber_band_node_ind = 1; rubber_band_node_ind < band_.size() - 1; ++rubber_band_node_ind)
        {
            const Eigen::Vector3d& current_pos = band_[rubber_band_node_ind];
            const double distance_to_first_endpoint = cummulative_distances_along_band[rubber_band_node_ind];
            const double ratio = distance_to_first_endpoint / total_band_length;
            const Eigen::Vector3d delta = EigenHelpers::Interpolate(first_endpoint_delta, second_endpoint_delta, ratio);
            const Eigen::Vector3d new_pos_post_project = sdf_.ProjectOutOfCollision3d(current_pos + delta);
            band_[rubber_band_node_ind] = new_pos_post_project;
        }
        visualize(BAND_POST_FORWARD_STEP_NS, Visualizer::Green(), Visualizer::Green(), integration_step_ind, verbose);

        // Next subdivide the band if needed
        resampleBand(verbose);
        visualize(BAND_POST_SUBDIVIDE_STEP_NS, Visualizer::Magenta(), Visualizer::Magenta(), integration_step_ind, verbose);

        // Update distances for loop check
        distance_to_first_target = (first_endpoint_target - band_.front()).norm();
        distance_to_second_target = (second_endpoint_target - band_.back()).norm();
        ++integration_step_ind;
    }

    visualize(BAND_POST_SUBDIVIDE_STEP_NS, Visualizer::Magenta(), Visualizer::Magenta(), integration_step_ind, verbose);

    // Then finally shortcut smooth
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

    const std::function<double(const Eigen::Vector3d&, const Eigen::Vector3d&)> state_distance_fn = [] (const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
    {
        return (p1 - p2).norm();
    };
    const std::function<Eigen::Vector3d(const Eigen::Vector3d&, const Eigen::Vector3d&, const double)> state_interpolation_fn = [&] (const Eigen::Vector3d& prev, const Eigen::Vector3d& curr, const double ratio)
    {
        return sdf_.ProjectOutOfCollision3d(EigenHelpers::Interpolate(prev, curr, ratio));
    };

    // Continue to smooth with projected points until the result stabilizes
    EigenHelpers::VectorVector3d smoothing_result = shortcut_smoothing::ResamplePath(band_, max_distance_between_rubber_band_points_, state_distance_fn, state_interpolation_fn);
    while (smoothing_result != band_)
    {
        band_ = smoothing_result;
        smoothing_result = shortcut_smoothing::ResamplePath(band_, max_distance_between_rubber_band_points_, state_distance_fn, state_interpolation_fn);
    }

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

            assert(false && "After resampling, the maximum distance between points is still violated");
        }
    }
}

void VirtualRubberBand::shortcutSmoothBand(const bool verbose)
{
    const auto sdf_collision_fn = [&] (const Eigen::Vector3d& location)
    {
        return sdf_.Get3d(location) < 0.0;
    };
    for (int smoothing_ittr = 0; smoothing_ittr < num_smoothing_ittrs_; ++smoothing_ittr)
    {
        std::uniform_int_distribution<ssize_t> first_distribution(0, band_.size() - 1);
        const size_t first_ind = first_distribution(generator_);

        #warning "Magic number for smoothing distance here"
        const ssize_t max_smooth_distance = std::max((ssize_t)10, (ssize_t)std::floor(sdf_.Get3d(band_[first_ind]) / max_distance_between_rubber_band_points_));
        std::uniform_int_distribution<ssize_t> second_distribution(-max_smooth_distance, max_smooth_distance);
        const ssize_t second_offset = second_distribution(generator_);
        const size_t second_ind = (size_t)arc_helpers::ClampValue<ssize_t>(first_ind + second_offset, 0, (ssize_t)band_.size() - 1);

        if (first_ind != second_ind)
        {
            band_ = shortcut_smoothing::ShortcutSmooth(band_, first_ind, second_ind, max_distance_between_rubber_band_points_, sdf_collision_fn);
        }

        if (verbose)
        {
            vis_.visualizeXYZTrajectory(BAND_POST_SHORTCUT_SMOOTHING_NS, band_, Visualizer::Red(), smoothing_ittr);
        }
    }
}
