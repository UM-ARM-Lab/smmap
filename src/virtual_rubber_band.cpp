#include "smmap/virtual_rubber_band.h"

#include <arc_utilities/shortcut_smoothing.hpp>

using namespace smmap;

//static std::default_random_engine VirtualRubberBand::generator_(std::chrono::system_clock::now().time_since_epoch().count());
std::default_random_engine VirtualRubberBand::generator_;

#warning "Magic numbers in virtual band constructor"
VirtualRubberBand::VirtualRubberBand(
        const Eigen::Vector3d& start_point,
        const Eigen::Vector3d& end_point,
        const std::shared_ptr<DijkstrasCoverageTask>& task,
        const Visualizer& vis)
    : task_(task)
    , sdf_(task_->environment_sdf_)
    , vis_(vis)
    , max_integration_step_size_(sdf_.GetResolution() / 10.0)
    , max_distance_between_rubber_band_points_(task_->work_space_grid_.minStepDimension() / 2.0)
    , num_smoothing_ittrs_(200)
    , min_object_radius_(0.04)
    , max_total_band_distance_((end_point - start_point).norm() * task_->stretchingThreshold())
{
//    max_gripper_distance_ = EigenHelpers::CalculateTotalDistance(virtual_rubber_band) + (double)(GetClothNumDivsY(nh_) - 1) * dijkstras_task->stretchingThreshold();
//    ROS_INFO_STREAM_NAMED("planner", "  -----   Max gripper distance: " << max_gripper_distance_ << " Num rubber band nodes: " << virtual_rubber_band.size());

    assert(task_->deformable_type_ == DeformableType::CLOTH);

    const size_t num_divs = (size_t)std::ceil((end_point - start_point ).norm() / max_distance_between_rubber_band_points_);

    for (size_t i = 0; i <= num_divs; ++i)
    {
        band_.push_back(EigenHelpers::Interpolate(start_point, end_point, (double)i / (double)num_divs));
        assert(sdf_.Get3d(band_.back()) > 0.0);
    }
}

VirtualRubberBand::VirtualRubberBand(
        const EigenHelpers::VectorVector3d starting_points,
        const double max_total_band_distance,
        const std::shared_ptr<DijkstrasCoverageTask>& task,
        const Visualizer& vis)
    : task_(task)
    , sdf_(task_->environment_sdf_)
    , vis_(vis)
    , max_integration_step_size_(sdf_.GetResolution() / 10.0)
    , max_distance_between_rubber_band_points_(task_->work_space_grid_.minStepDimension() / 2.0)
    , num_smoothing_ittrs_(200)
    , min_object_radius_(0.04)
    , max_total_band_distance_(max_total_band_distance)
    , band_(starting_points)
{
    resampleBand(false);
    shortcutSmoothBand(false);
}


const EigenHelpers::VectorVector3d& VirtualRubberBand::forwardSimulateVirtualRubberBand(
        const Eigen::Vector3d first_endpoint_translation,
        const Eigen::Vector3d second_endpoint_translation,
        bool verbose)
{
    const double max_endpoint_delta = std::max(first_endpoint_translation.norm(), second_endpoint_translation.norm());
    const int num_integration_steps = (int)std::ceil(max_endpoint_delta / max_integration_step_size_);

    // Forward simulate the rubber band
    for (int integration_step_ind = 0; integration_step_ind < num_integration_steps; integration_step_ind++)
    {
        const std::vector<double> cummulative_distances_along_band = EigenHelpers::CalculateCumulativeDistances(band_);

        // First move each node forward, projecting out of collision as needed
        for (size_t rubber_band_node_ind = 0; rubber_band_node_ind < band_.size(); ++rubber_band_node_ind)
        {
            const Eigen::Vector3d& current_pos = band_[rubber_band_node_ind];
            const double distance_to_first_endpoint = cummulative_distances_along_band[rubber_band_node_ind];
            const double ratio = distance_to_first_endpoint / cummulative_distances_along_band.back();
            const Eigen::Vector3d delta = EigenHelpers::Interpolate(first_endpoint_translation, second_endpoint_translation, ratio) / (double)num_integration_steps;
            band_[rubber_band_node_ind] = sdf_.ProjectOutOfCollision3d(current_pos + delta);
        }
        visualize("post_forward_step", Visualizer::Green(), Visualizer::Green(), integration_step_ind, verbose);

        // Next subdivide the band if needed
        resampleBand(verbose);
        visualize("post_subdivide_step", Visualizer::Magenta(), Visualizer::Magenta(), integration_step_ind, verbose);
    }

    // Then finally shortcut smooth
    shortcutSmoothBand(verbose);

    return band_;
}

const EigenHelpers::VectorVector3d& VirtualRubberBand::getVectorRepresentation() const
{
    return band_;
}

bool VirtualRubberBand::isOverstretched() const
{
    return EigenHelpers::CalculateTotalDistance(band_) > max_total_band_distance_;
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

    band_ = shortcut_smoothing::ResamplePath(band_, max_distance_between_rubber_band_points_ / 1.5, state_distance_fn, state_interpolation_fn);

    // Double check the results; this is here to catch cases where the projection may be creating points that are too far apart
    for (size_t idx = 0; idx < band_.size() - 1; ++idx)
    {
        if ((band_[idx] - band_[idx + 1]).squaredNorm() > max_distance_between_rubber_band_points_ * max_distance_between_rubber_band_points_)
        {
            visualize("badness", Visualizer::Red(), Visualizer::Red(), 1, true);
            std::cout << PrettyPrint::PrettyPrint(EigenHelpers::CalculateCumulativeDistances(band_), true, " ");
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
        const size_t second_ind = (size_t)arc_helpers::ClampValue<ssize_t>(first_ind + second_distribution(generator_), 0, band_.size() - 1);

        if (first_ind != second_ind)// && (band_[first_ind] - band_[second_ind]).squaredNorm() < min_object_radius_ * min_object_radius_)
        {
            band_ = shortcut_smoothing::ShortcutSmooth(band_, first_ind, second_ind, max_distance_between_rubber_band_points_, sdf_collision_fn);
        }

        if (verbose)
        {
            vis_.visualizeXYZTrajectory("post_shortcut_smoothing", band_, Visualizer::Red(), smoothing_ittr);
        }
    }
}
