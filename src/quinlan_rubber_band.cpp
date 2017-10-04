#include "smmap/quinlan_rubber_band.h"

using namespace smmap;

QuinlanRubberBand::QuinlanRubberBand(
        const Eigen::Vector3d& start_point,
        const Eigen::Vector3d& end_point,
        const std::shared_ptr<DijkstrasCoverageTask>& task,
        const Visualizer& vis,
        std::mt19937_64& generator)
    : QuinlanRubberBand({start_point, end_point},
                        (end_point - start_point).norm() * task_->maxStretchFactor(),
                        task,
                        vis,
                        generator)
{}

QuinlanRubberBand::QuinlanRubberBand(
        const EigenHelpers::VectorVector3d starting_points,
        const double max_total_band_distance,
        const std::shared_ptr<DijkstrasCoverageTask>& task,
        const Visualizer& vis,
        std::mt19937_64& generator)
    : task_(task)
    , sdf_(task_->environment_sdf_)
    , vis_(vis)
    , band_(starting_points)
    , max_total_band_distance_(max_total_band_distance)
    , min_overlap_distance_(sdf_.GetResolution() * 0.01)
    , min_distance_to_obstacle_(min_overlap_distance_ * 2.0)
{
    interpolateBandPoints();
    (void)generator;
}

QuinlanRubberBand& QuinlanRubberBand::operator=(const QuinlanRubberBand& other)
{
    assert(task_ == other.task_);
    assert(&sdf_ == &(other.sdf_));
    assert(&vis_ == &(other.vis_));

    assert(max_total_band_distance_ == other.max_total_band_distance_);

    band_ = other.band_;

    return *this;
}

void QuinlanRubberBand::setPointsWithoutSmoothing(const EigenHelpers::VectorVector3d& points)
{
    band_ = points;
    interpolateBandPoints();
}

void QuinlanRubberBand::setPointsAndSmooth(const EigenHelpers::VectorVector3d& points)
{
    setPointsWithoutSmoothing(points);
    assert(false && "not implemented");
}

/**
 * @brief QuinlanRubberBand::forwardSimulateVirtualRubberBandToEndpointTargets This function assumes that the endpoints
 * that are passed are collision free, and the path between them can be resampled without any issues. We may have to
 * project the resampled points out of collision, but it should be feasible to do so without changing the "homotopy" class
 * @param first_endpoint_target
 * @param second_endpoint_target
 * @param verbose
 * @return
 */
const EigenHelpers::VectorVector3d& QuinlanRubberBand::forwardPropagateRubberBandToEndpointTargets(
        const Eigen::Vector3d first_endpoint_target,
        const Eigen::Vector3d second_endpoint_target,
        bool verbose)
{
    // Add the new endpoints, then let the interpolate and smooth process handle the propogation
    assert(sdf_.EstimateDistance3d(first_endpoint_target).first > min_distance_to_obstacle_);
    assert(sdf_.EstimateDistance3d(second_endpoint_target).first > min_distance_to_obstacle_);
    band_.insert(band_.begin(), first_endpoint_target);
    band_.push_back(second_endpoint_target);
    interpolateBandPoints();
    smoothBandPoints(verbose);
    return band_;
}

const EigenHelpers::VectorVector3d& QuinlanRubberBand::getVectorRepresentation() const
{
    return band_;
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> QuinlanRubberBand::getEndpoints() const
{
    return {band_.front(), band_.back()};
}

double QuinlanRubberBand::maxSafeLength() const
{
    return max_total_band_distance_;
}

double QuinlanRubberBand::totalLength() const
{
    return EigenHelpers::CalculateTotalDistance(band_);
}

bool QuinlanRubberBand::isOverstretched() const
{
    return totalLength() > max_total_band_distance_;
}

void QuinlanRubberBand::visualize(
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

bool QuinlanRubberBand::bandIsValid()
{
    if (band_.size() < 2)
    {
        return false;
    }

    for (size_t node_idx = 0; node_idx < band_.size() - 1; ++node_idx)
    {
        const auto& curr_node = band_[node_idx];
        const auto& next_node = band_[node_idx + 1];
        const double dist = (curr_node - next_node).norm();
        const double curr_bubble_size = sdf_.EstimateDistance3d(curr_node).first;
        const double next_bubble_size = sdf_.EstimateDistance3d(next_node).first;
        assert(curr_bubble_size >= min_distance_to_obstacle_);
        assert(next_bubble_size >= min_distance_to_obstacle_);
        if (dist > curr_bubble_size + next_bubble_size + min_overlap_distance_)
        {
            return false;
        }
    }

    return true;
}

/**
 * @brief QuinlanRubberBand::interpolateBandPoints
 * Re-interpolates the entire band, not to be used for just 1 segment
 */
void QuinlanRubberBand::interpolateBandPoints()
{
    assert(band_.size() >= 2);
    for (const auto& point: band_)
    {
        assert(sdf_.EstimateDistance3d(point).first >= min_distance_to_obstacle_);
    }

    // Make a copy of the band, then clear it for use as the resulting band
    const EigenHelpers::VectorVector3d starting_band = band_;
    band_.clear();

    band_.push_back(starting_band[0]);
    size_t prev_idx = 0;
    double prev_bubble_size = sdf_.EstimateDistance3d(starting_band[0]).first;
    while (prev_idx + 1 < starting_band.size())
    {
        const size_t next_idx = prev_idx + 1;
        const Eigen::Vector3d& next_node = starting_band[next_idx];
        const double next_bubble_size = sdf_.EstimateDistance3d(next_node).first;
        double distance_to_next_node = (band_.back() - starting_band[next_idx]).norm();

        // TODO: verify that this cannot get stuck in an infinite loop
        int outer_iteration_counter = 0;
        // Check if the bubbles for 2 adjacent nodes overlap with some minimum distance to spare
        while (distance_to_next_node > prev_bubble_size + next_bubble_size + min_overlap_distance_)
        {
            // TODO: verify that this cannot get stuck in an infinite loop
            int inner_iteration_counter = 0;
            // Find a position between band_.back() and next_node with sufficient bubble overlap with band_.back()
            double interpolation_ratio = 0.5;
            Eigen::Vector3d test_point = sdf_.ProjectOutOfCollisionToMinimumDistance3d(
                        EigenHelpers::Interpolate(band_.back(), next_node, interpolation_ratio),
                        min_distance_to_obstacle_);
            double test_point_bubble_size = sdf_.EstimateDistance3d(test_point).first;
            double distance_between_prev_and_target = (band_.back() - test_point).norm();
            while (distance_between_prev_and_target > prev_bubble_size + test_point_bubble_size + min_overlap_distance_)
            {
                interpolation_ratio *= 0.5;
                test_point = sdf_.ProjectOutOfCollisionToMinimumDistance3d(
                            EigenHelpers::Interpolate(band_.back(), next_node, interpolation_ratio),
                            min_distance_to_obstacle_);
                test_point_bubble_size = sdf_.EstimateDistance3d(test_point).first;
                distance_between_prev_and_target = (band_.back() - test_point).norm();
                ++inner_iteration_counter;
                ROS_WARN_COND_NAMED(inner_iteration_counter == 50, "rubber_band", "Rubber band interpolation inner loop counter at 50, probably stuck in an infinite loop");
            }
            // The bubbles now overlap sufficiently, so accept this point and record the new values
            band_.push_back(test_point);
            prev_bubble_size = test_point_bubble_size;
            ++outer_iteration_counter;
            ROS_WARN_COND_NAMED(outer_iteration_counter == 50, "rubber_band", "Rubber band interpolation outer loop counter at 50, probably stuck in an infinite loop");
        }

        band_.push_back(next_node);
        prev_idx = next_idx;
        prev_bubble_size = next_bubble_size;
    }

    assert(bandIsValid());
}

void QuinlanRubberBand::smoothBandPoints(const bool verbose)
{
    assert(bandIsValid());
    assert(false && "Not implemented");
}
