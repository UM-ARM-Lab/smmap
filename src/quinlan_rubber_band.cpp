#include "smmap/quinlan_rubber_band.h"

using namespace smmap;
using ColorBuilder = arc_helpers::RGBAColorBuilder<std_msgs::ColorRGBA>;

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
    , node_removal_overlap_factor_(1.2)
    , smoothing_iterations_(50)
{
    interpolateBandPoints();
    removeExtraBandPoints();
    (void)generator;
    const bool verbose = false;
    smoothBandPoints(verbose);
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
    removeExtraBandPoints();
}

void QuinlanRubberBand::setPointsAndSmooth(const EigenHelpers::VectorVector3d& points)
{
    band_ = points;
    interpolateBandPoints();
    const bool verbose = false;
    smoothBandPoints(verbose);
    removeExtraBandPoints();
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

void QuinlanRubberBand::visualizeWithBubbles(
        const std::string& marker_name,
        const std_msgs::ColorRGBA& safe_color,
        const std_msgs::ColorRGBA& overstretched_color,
        const int32_t id,
        const bool visualization_enabled) const
{
    if (visualization_enabled)
    {
        // Delete all markers, probably from just this publisher
        {
            visualization_msgs::Marker marker;
            marker.action = visualization_msgs::Marker::DELETEALL;
            vis_.publish(marker);
        }

        // Re-publish the new ones
        {
            visualize(marker_name, safe_color, overstretched_color, id, visualization_enabled);
            std::vector<double> bubble_sizes(band_.size());
            std::vector<std_msgs::ColorRGBA> colors(band_.size());
            for (size_t idx = 0; idx < band_.size(); ++idx)
            {
                bubble_sizes[idx] = sdf_.EstimateDistance3d(band_[idx]).first;
                colors[idx] = ColorBuilder::MakeFromFloatColors(
                            (float)idx / (float)(band_.size() - 1),
                            0.0f,
                            (float)(band_.size() - 1 - idx) / (float)(band_.size() - 1),
                            0.3f);
            }
            vis_.visualizeSpheres(marker_name,
                                  band_,
                                  bubble_sizes,
                                  colors,
                                  id + 1);
        }
    }
}

bool QuinlanRubberBand::sufficientOverlap(
        const double bubble_size_a,
        const double bubble_size_b,
        const double distance) const
{
    return bubble_size_a + bubble_size_b >= distance + min_overlap_distance_;
}

bool QuinlanRubberBand::bandIsValid() const
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
        if (!sufficientOverlap(curr_bubble_size, next_bubble_size, dist))
        {
            return false;
        }
    }

    return true;
}

bool QuinlanRubberBand::bandIsValidWithDebugging() const
{
    if (!bandIsValid())
    {
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);

        return bandIsValid();
    }
    return true;
}

void QuinlanRubberBand::interpolateBetweenPoints(
        EigenHelpers::VectorVector3d& point_buffer,
        const Eigen::Vector3d& target) const
{
    const double target_bubble_size = sdf_.EstimateDistance3d(target).first;

    // TODO: verify that this cannot get stuck in an infinite loop
    int outer_iteration_counter = 0;

    // Check if the bubbles for 2 adjacent nodes overlap with some minimum distance to spare
    double curr_bubble_size = sdf_.EstimateDistance3d(point_buffer.back()).first;
    double distance_to_end = (target - point_buffer.back()).norm();
    while (!sufficientOverlap(curr_bubble_size, target_bubble_size, distance_to_end))
    {
        const Eigen::Vector3d& curr = point_buffer.back();

        // Find a position between point_buffer.back() and next_node with sufficient bubble overlap
        // TODO: verify that this cannot get stuck in an infinite loop
        int inner_iteration_counter = 0;
        double interpolation_ratio = 0.5;
        Eigen::Vector3d test_point = sdf_.ProjectOutOfCollisionToMinimumDistance3d(
                    EigenHelpers::Interpolate(curr, target, interpolation_ratio),
                    min_distance_to_obstacle_);
        double test_point_bubble_size = sdf_.EstimateDistance3d(test_point).first;
        double distance_between_prev_and_target = (curr - test_point).norm();
        while (distance_between_prev_and_target > curr_bubble_size + test_point_bubble_size + min_overlap_distance_)
        {
            interpolation_ratio *= 0.5;
            test_point = sdf_.ProjectOutOfCollisionToMinimumDistance3d(
                        EigenHelpers::Interpolate(curr, target, interpolation_ratio),
                        min_distance_to_obstacle_);
            test_point_bubble_size = sdf_.EstimateDistance3d(test_point).first;
            distance_between_prev_and_target = (curr - test_point).norm();

            ++inner_iteration_counter;
            ROS_WARN_COND_NAMED(inner_iteration_counter == 50, "rubber_band", "Rubber band interpolation inner loop counter at 50, probably stuck in an infinite loop");
        }
        // The bubbles now overlap sufficiently, so accept this point and record the new values
        point_buffer.push_back(test_point);
        curr_bubble_size = test_point_bubble_size;
        distance_to_end = (target - point_buffer.back()).norm();

        ++outer_iteration_counter;
        ROS_WARN_COND_NAMED(outer_iteration_counter == 50, "rubber_band", "Rubber band interpolation outer loop counter at 50, probably stuck in an infinite loop");
    }
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

    EigenHelpers::VectorVector3d new_band(1, band_.front());
    for (size_t idx = 0; idx + 1 < band_.size(); ++idx)
    {
        const Eigen::Vector3d& next_node = band_[idx + 1];
        interpolateBetweenPoints(new_band, next_node);
        new_band.push_back(next_node);
    }

    band_ = new_band;

    assert(bandIsValidWithDebugging());
}

// TODO: Convert this into a single pass by using a "look forward" for overlap condition checker
void QuinlanRubberBand::removeExtraBandPoints()
{
    assert(bandIsValidWithDebugging());
    visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
    std::cout << "Before removing extra points, band size: " << band_.size() << std::endl;

    // Do a forward pass, then a backwards pass to help address issues of non-symmetry
    // Forward pass
    {
        // The start can't be removed, so push that point on immediately
        EigenHelpers::VectorVector3d forward_pass;
        forward_pass.reserve(band_.size());
        forward_pass.push_back(band_.front());

        for (size_t curr_idx = 1; curr_idx + 1 < band_.size(); ++curr_idx)
        {
            const auto& prev = forward_pass.back();
            const auto& curr = band_[curr_idx];
            const auto& next = band_[curr_idx + 1];

            const double prev_bubble_size = sdf_.EstimateDistance3d(prev).first;
            const double curr_bubble_size = sdf_.EstimateDistance3d(curr).first;
            const double next_bubble_size = sdf_.EstimateDistance3d(next).first;

            const double prev_curr_dist = (prev - curr).norm();
            const double curr_next_dist = (next - curr).norm();
            const double prev_next_dist = (prev - next).norm();

            const bool curr_bubble_is_wholey_contained_in_prev =
                    prev_bubble_size >= prev_curr_dist + curr_bubble_size + min_overlap_distance_;
            const bool prev_bubble_overlaps_curr_center_by_minimum =
                    prev_bubble_size >= prev_curr_dist * node_removal_overlap_factor_ + min_overlap_distance_;
            const bool next_bubble_overlaps_curr_center_by_minimum =
                    next_bubble_size >= curr_next_dist * node_removal_overlap_factor_ + min_overlap_distance_;

            // Only keep this point if there is not sufficient overlap for the neighbouring bubbles
            if (!(prev_bubble_overlaps_curr_center_by_minimum && next_bubble_overlaps_curr_center_by_minimum) &&
                !curr_bubble_is_wholey_contained_in_prev)
            {
                forward_pass.push_back(curr);
            }
        }
        forward_pass.push_back(band_.back());

        band_ = forward_pass;
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        std::cout << "After forward pass,           band size: " << band_.size() << std::endl;
        assert(bandIsValidWithDebugging());
    }

    // Backward pass
    {
        EigenHelpers::VectorVector3d backward_pass;
        backward_pass.reserve(band_.size());
        backward_pass.push_back(band_.back());

        for (size_t curr_idx = band_.size() - 2; curr_idx > 0; --curr_idx)
        {
            const auto& prev = backward_pass.back();
            const auto& curr = band_[curr_idx];
            const auto& next = band_[curr_idx - 1];

            const double prev_bubble_size = sdf_.EstimateDistance3d(prev).first;
            const double curr_bubble_size = sdf_.EstimateDistance3d(curr).first;
            const double next_bubble_size = sdf_.EstimateDistance3d(next).first;

            const double prev_curr_dist = (prev - curr).norm();
            const double curr_next_dist = (next - curr).norm();

            const bool curr_bubble_is_wholly_contained_in_prev =
                    prev_bubble_size >= prev_curr_dist + curr_bubble_size + min_overlap_distance_;
            const bool prev_bubble_overlaps_curr_center_by_minimum =
                    prev_bubble_size >= prev_curr_dist * node_removal_overlap_factor_ + min_overlap_distance_;
            const bool next_bubble_overlaps_curr_center_by_minimum =
                    next_bubble_size >= curr_next_dist * node_removal_overlap_factor_ + min_overlap_distance_;

            // Only keep this point if there is not sufficient overlap for the neighbouring bubbles
            if (!(prev_bubble_overlaps_curr_center_by_minimum && next_bubble_overlaps_curr_center_by_minimum) &&
                !curr_bubble_is_wholly_contained_in_prev)
            {
                backward_pass.push_back(curr);
            }
        }
        backward_pass.push_back(band_.front());

        band_ = backward_pass;
        std::reverse(band_.begin(), band_.end());
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        std::cout << "After backward pass,          band size: " << band_.size() << std::endl;
        assert(bandIsValidWithDebugging());
    }
}

void QuinlanRubberBand::smoothBandPoints(const bool verbose)
{
    assert(bandIsValidWithDebugging());
//    std::cout << PrettyPrint::PrettyPrint(band_, false, "\n") << std::endl << std::endl;

    for (size_t smoothing_iter = 0; smoothing_iter < smoothing_iterations_; ++smoothing_iter)
    {
        // The start doesn't move, so push that point on immediately
        EigenHelpers::VectorVector3d next_band;
        next_band.reserve(band_.size());
        next_band.push_back(band_.front());

        for (size_t curr_idx = 1; curr_idx + 1 < band_.size(); ++ curr_idx)
        {
            const auto& prev = next_band.back();
            const auto& curr = band_[curr_idx];
            const auto& next = band_[curr_idx + 1];

            const double prev_bubble_size = sdf_.EstimateDistance3d(prev).first;
            const double curr_bubble_size = sdf_.EstimateDistance3d(curr).first;
            const double next_bubble_size = sdf_.EstimateDistance3d(next).first;

            // Project the current point onto the line between prev and next
            const Eigen::Vector3d band_tangent_approx = next - prev;
            const Eigen::Vector3d curr_projected = prev + EigenHelpers::VectorProjection(band_tangent_approx, curr - prev);
            const Eigen::Vector3d delta = curr_projected - curr;
            assert(delta.dot(band_tangent_approx) < 1e-10);

            // Determine if the projection is within the bubble at the current point, and if not only move part way
            const double max_delta_norm = curr_bubble_size - min_distance_to_obstacle_;
            const Eigen::Vector3d curr_prime = delta.norm() <= max_delta_norm ? curr_projected : curr + max_delta_norm * delta.normalized();
            // Ensure that the resulting point is not in collision even with numerical rounding
            const Eigen::Vector3d projected_position = sdf_.ProjectOutOfCollisionToMinimumDistance3d(curr_prime, min_distance_to_obstacle_);

            // Check if the bubbles still overlap on each side
            const double prev_curr_dist = (prev - projected_position).norm();
            const double curr_next_dist = (next - projected_position).norm();

            const double projected_bubble_size = sdf_.EstimateDistance3d(projected_position).first;
            const bool prev_bubble_overlaps_curr = sufficientOverlap(prev_bubble_size, projected_bubble_size, prev_curr_dist);
            const bool next_bubble_overlaps_curr = sufficientOverlap(next_bubble_size, projected_bubble_size, curr_next_dist);

            if (!prev_bubble_overlaps_curr)
            {
                interpolateBetweenPoints(next_band, projected_position);
            }
            next_band.push_back(projected_position);
            if (!next_bubble_overlaps_curr)
            {
                interpolateBetweenPoints(next_band, next);
            }
        }

        // The end doesn't move, so push that point on at the end, then swap buffers
        next_band.push_back(band_.back());
        band_ = next_band;
        assert(bandIsValidWithDebugging());

        removeExtraBandPoints();
//        std::cout << PrettyPrint::PrettyPrint(band_, false, "\n") << std::endl << std::endl;
    }

    assert(bandIsValidWithDebugging());
}
