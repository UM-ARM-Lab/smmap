#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/serialization_eigen.hpp>
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
    : ph_("~/band")
    , task_(task)
    , sdf_(task_->environment_sdf_)
    , vis_(vis)
    , max_total_band_distance_(max_total_band_distance)
    , min_overlap_distance_(sdf_.GetResolution() * 0.05)
    , min_distance_to_obstacle_(min_overlap_distance_ * 2.0)
    , node_removal_overlap_factor_(1.2)
    , backtrack_threshold_(1e-6)
    , collision_margin_(sdf_.GetResolution() / std::sqrt(2.0))
    , smoothing_iterations_(50)
{
    (void)generator;
    setPointsAndSmooth(starting_points);
}

QuinlanRubberBand& QuinlanRubberBand::operator=(const QuinlanRubberBand& other)
{
    assert(task_ == other.task_);
    assert(&sdf_ == &(other.sdf_));
    assert(&vis_ == &(other.vis_));

    assert(max_total_band_distance_ == other.max_total_band_distance_);

    band_ = other.band_;



    if (useStoredBand())
    {
        loadStoredBand();
    }
    else
    {
        storeBand();
    }



    assert(bandIsValid());

    return *this;
}

void QuinlanRubberBand::setPointsWithoutSmoothing(const EigenHelpers::VectorVector3d& points)
{
    band_ = points;

    if (useStoredBand())
    {
        loadStoredBand();
    }
    else
    {
        storeBand();
    }

    for (auto& point: band_)
    {
        point = projectToValidBubble(point);
        assert(getBubbleSize(point) >= min_distance_to_obstacle_ &&
               "Every point in the band must be inside the valid region of the SDF");
    }
    interpolateBandPoints();
    removeExtraBandPoints();
}

void QuinlanRubberBand::setPointsAndSmooth(const EigenHelpers::VectorVector3d& points)
{
    setPointsWithoutSmoothing(points);
    const bool verbose = false;
    smoothBandPoints(verbose);
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
    // Ensure that the new points are both in bounds, and are at least min_distance_to_obstacle_ from anything
    assert(getBubbleSize(first_endpoint_target) >= min_distance_to_obstacle_);
    assert(getBubbleSize(second_endpoint_target) >= min_distance_to_obstacle_);
    // Add the new endpoints, then let the interpolate and smooth process handle the propogation
    band_.insert(band_.begin(), first_endpoint_target);
    band_.push_back(second_endpoint_target);



    if (useStoredBand())
    {
        loadStoredBand();
    }
    else
    {
        storeBand();
    }



    interpolateBandPoints();
    removeExtraBandPoints();
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
//            visualization_msgs::Marker marker;
//            marker.action = visualization_msgs::Marker::DELETEALL;
//            vis_.publish(marker);

//            vis_.deleteObjects(marker_name + "_bubbles", 1, 305);
        }

        // Re-publish the new ones
        {
            visualize(marker_name, safe_color, overstretched_color, id, visualization_enabled);
//            std::vector<double> bubble_sizes(band_.size());
//            std::vector<std_msgs::ColorRGBA> colors(band_.size());
//            for (size_t idx = 0; idx < band_.size(); ++idx)
//            {
//                bubble_sizes[idx] = sdf_.EstimateDistance3d(band_[idx]).first;
//                colors[idx] = ColorBuilder::MakeFromFloatColors(
//                            (float)idx / (float)(band_.size() - 1),
//                            0.0f,
//                            (float)(band_.size() - 1 - idx) / (float)(band_.size() - 1),
//                            0.3f);
//            }
//            vis_.visualizeSpheres(marker_name + "_bubbles", band_, bubble_sizes, colors, id);
        }
    }
}



Eigen::Vector3d QuinlanRubberBand::projectToValidBubble(const Eigen::Vector3d& location) const
{
    const auto post_collision_project = sdf_.ProjectOutOfCollisionToMinimumDistance3d(location, min_distance_to_obstacle_ + collision_margin_ * 1.00000000001);
    const auto post_boundary_project = sdf_.ProjectIntoValidVolumeToMinimumDistance3d(post_collision_project, min_distance_to_obstacle_);

    const auto distance_to_boundary = sdf_.DistanceToBoundary3d(post_boundary_project);
    const auto distance_to_obstacles = sdf_.EstimateDistance3d(post_boundary_project);

    if (distance_to_boundary.first < min_distance_to_obstacle_ ||
        distance_to_obstacles.first < min_distance_to_obstacle_)
    {
        constexpr int p = 20;
        std::cerr << std::setprecision(p) << "location:                                                                 " << location.transpose() << std::endl;

        const auto starting_distance_to_boundary = sdf_.DistanceToBoundary3d(location);
        const auto starting_distance_to_obstacles = sdf_.EstimateDistance3d(location);

        std::cerr << std::setprecision(p) << "Starting dist to obstacle: " << starting_distance_to_obstacles.first << std::endl;
        std::cerr << std::setprecision(p) << "Starting dist to boundary: " << starting_distance_to_boundary.first << std::endl;

        std::cerr << std::setprecision(p) << "Final dist to obstacle:    " << distance_to_obstacles.first << std::endl;
        std::cerr << std::setprecision(p) << "Final dist to boundary:    " << distance_to_boundary.first << std::endl;

        std::cerr << std::setprecision(p) << "collision - location:      " << (post_collision_project - location).norm() << std::endl;
        std::cerr << std::setprecision(p) << "boundary - collision:      " << (post_boundary_project - post_collision_project).norm() << std::endl;

        std::cerr << std::setprecision(p) << "Post boundary value:                                                      " << post_boundary_project.transpose() << std::endl;

        const auto post_collision_project = sdf_.ProjectOutOfCollisionToMinimumDistance3d(location, min_distance_to_obstacle_);
        const auto post_boundary_project = sdf_.ProjectIntoValidVolumeToMinimumDistance3d(post_collision_project, min_distance_to_obstacle_ + collision_margin_ * 1.00000000001);
    }


    assert(getBubbleSize(post_boundary_project) >= min_distance_to_obstacle_);
    return post_boundary_project;
}

double QuinlanRubberBand::getBubbleSize(const Eigen::Vector3d& location) const
{
    const Eigen::Vector4d loc_4d(location.x(), location.y(), location.z(), 1.0);
    const auto distance_to_boundary = sdf_.DistanceToBoundary4d(loc_4d);
//    const auto distance_to_obstacles = sdf_.EstimateDistance4d(loc_4d);
    const auto distance_to_obstacles = sdf_.GetSafe4d(loc_4d);
    assert(distance_to_boundary.second == distance_to_obstacles.second);
    const auto distance = std::min(distance_to_boundary.first - collision_margin_, (double)distance_to_obstacles.first);
//    std::cout << location.transpose() << " Estimate dist: " << distance_to_obstacles.first << " Boundary dist: " << distance_to_boundary.first << std::endl;
    assert(distance >= min_distance_to_obstacle_);
    return distance;
}

bool QuinlanRubberBand::sufficientOverlap(
        const double bubble_size_a,
        const double bubble_size_b,
        const double distance) const
{
    return (bubble_size_a + bubble_size_b) >= (distance + min_overlap_distance_);
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
        assert(sdf_.CheckInBounds3d(curr_node));
        const auto& next_node = band_[node_idx + 1];
        const double dist = (curr_node - next_node).norm();
        const double curr_bubble_size = getBubbleSize(curr_node);
        const double next_bubble_size = getBubbleSize(next_node);
        if (!sufficientOverlap(curr_bubble_size, next_bubble_size, dist))
        {
            std::cerr << "Problem between node " << node_idx << " and " << node_idx + 1 << std::endl
                      << "Curr bubble size: " << curr_bubble_size << std::endl
                      << "Next bubble size: " << next_bubble_size << std::endl
                      << "Curr + next:      " << curr_bubble_size + next_bubble_size << std::endl
                      << "Dist + min:       " << dist + min_overlap_distance_ << std::endl;
            return false;
        }
    }

    return true;
}

bool QuinlanRubberBand::bandIsValidWithVisualization() const
{
    if (!bandIsValid())
    {
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        vis_.visualizePoints("quinlan_band_test_points", band_, Visualizer::Green(), 1, 0.002);

        std::cout << "Num points: " << band_.size() << std::endl;
        std::cout << PrettyPrint::PrettyPrint(band_, false, "\n") << std::endl << std::endl;

        return bandIsValid();
    }
    return true;
}

void QuinlanRubberBand::interpolateBetweenPoints(
        EigenHelpers::VectorVector3d& point_buffer,
        const Eigen::Vector3d& target) const
{
    const double target_bubble_size = getBubbleSize(target);

    // TODO: verify that this cannot get stuck in an infinite loop
    int outer_iteration_counter = 0;

    // Check if the bubbles for 2 adjacent nodes overlap with some minimum distance to spare
    double curr_bubble_size = getBubbleSize(point_buffer.back());
    double distance_to_end = (target - point_buffer.back()).norm();
    while (!sufficientOverlap(curr_bubble_size, target_bubble_size, distance_to_end))
    {
        const Eigen::Vector3d& curr = point_buffer.back();

        // Find a position between point_buffer.back() and next_node with sufficient bubble overlap
        // TODO: verify that this cannot get stuck in an infinite loop
        int inner_iteration_counter = 0;
        double interpolation_ratio = 0.5;
        Eigen::Vector3d test_point = projectToValidBubble(EigenHelpers::Interpolate(curr, target, interpolation_ratio));
        double test_point_bubble_size = getBubbleSize(test_point);
        double distance_between_prev_and_test_point = (curr - test_point).norm();

        while (!sufficientOverlap(curr_bubble_size, test_point_bubble_size, distance_between_prev_and_test_point))
        {
            interpolation_ratio *= 0.5;
            test_point = projectToValidBubble(EigenHelpers::Interpolate(curr, target, interpolation_ratio));
            test_point_bubble_size = getBubbleSize(test_point);
            distance_between_prev_and_test_point = (curr - test_point).norm();

            ++inner_iteration_counter;
            ROS_WARN_COND_NAMED(inner_iteration_counter == 30, "rubber_band", "Rubber band interpolation inner loop counter at 30, probably stuck in an infinite loop");
            if (inner_iteration_counter == 30)
            {
                const Eigen::Vector3d min = sdf_.GetOriginTransform().translation();
                const Eigen::Vector3d max = min + Eigen::Vector3d(sdf_.GetXSize(), sdf_.GetYSize(), sdf_.GetZSize());

                std::cerr << "SDF limits: x, y, z\n"
                          << "Max:    " << max.transpose() << std::endl
                          << "Min:    " << min.transpose() << std::endl
                          << "Curr:   " << curr.transpose() << std::endl
                          << "Target: " << target.transpose() << std::endl
                          << "Test:   " << test_point.transpose() << std::endl;
            }
        }
        // The bubbles now overlap sufficiently, so accept this point and record the new values
        point_buffer.push_back(test_point);
        curr_bubble_size = test_point_bubble_size;
        distance_to_end = (target - point_buffer.back()).norm();

        ++outer_iteration_counter;
        ROS_WARN_COND_NAMED(outer_iteration_counter == 100, "rubber_band", "Rubber band interpolation outer loop counter at 100, probably stuck in an infinite loop");
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
        assert(getBubbleSize(point) >= min_distance_to_obstacle_);
    }

    EigenHelpers::VectorVector3d new_band(1, band_.front());
    for (size_t idx = 0; idx + 1 < band_.size(); ++idx)
    {
        const Eigen::Vector3d& next_node = band_[idx + 1];
        interpolateBetweenPoints(new_band, next_node);
        new_band.push_back(next_node);
    }

    band_ = new_band;

    assert(bandIsValidWithVisualization());
}

// TODO: Convert this into a single pass by using a "look forward" for overlap condition checker
void QuinlanRubberBand::removeExtraBandPoints()
{
    assert(bandIsValidWithVisualization());
    visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);

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

            const double prev_bubble_size = getBubbleSize(prev);
            const double curr_bubble_size = getBubbleSize(curr);
            const double next_bubble_size = getBubbleSize(next);

            const double prev_curr_dist = (prev - curr).norm();
            const double curr_next_dist = (next - curr).norm();

            const bool curr_bubble_is_wholey_contained_in_prev =
                    prev_bubble_size >= prev_curr_dist + curr_bubble_size + min_overlap_distance_;
            const bool prev_bubble_overlaps_curr_center_by_minimum =
                    prev_bubble_size >= prev_curr_dist * node_removal_overlap_factor_ + min_overlap_distance_;
            const bool next_bubble_overlaps_curr_center_by_minimum =
                    next_bubble_size >= curr_next_dist * node_removal_overlap_factor_ + min_overlap_distance_;

            const double angle_defined_by_points = EigenHelpers::AngleDefinedByPoints(prev, curr, next);
            assert(angle_defined_by_points >= 0.0);
            const bool band_backtracks = angle_defined_by_points < backtrack_threshold_;

//            if (band_backtracks)
//            {
//                std::cout << "Backtrack detected\n";

//                vis_.visualizePoints("prev", {prev}, Visualizer::Blue(), 1, 0.01);
//                vis_.visualizePoints("curr", {curr}, Visualizer::Magenta(), 1, 0.01);
//                vis_.visualizePoints("next", {next}, Visualizer::Red(), 1, 0.01);

//                std::cout << prev.transpose() << std::endl;
//                std::cout << curr.transpose() << std::endl;
//                std::cout << next.transpose() << std::endl;
//                std::cout << angle_defined_by_points << std::endl;
//                std::cout << std::endl;
//            }

            // Only keep this point if there is not sufficient overlap for the neighbouring bubbles
            if (!(prev_bubble_overlaps_curr_center_by_minimum && next_bubble_overlaps_curr_center_by_minimum) &&
                !curr_bubble_is_wholey_contained_in_prev &&
                !band_backtracks)
            {
                forward_pass.push_back(curr);
            }
        }
        forward_pass.push_back(band_.back());

        band_ = forward_pass;
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        assert(bandIsValidWithVisualization());
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

            const double prev_bubble_size = getBubbleSize(prev);
            const double curr_bubble_size = getBubbleSize(curr);
            const double next_bubble_size = getBubbleSize(next);

            const double prev_curr_dist = (prev - curr).norm();
            const double curr_next_dist = (next - curr).norm();

            const bool curr_bubble_is_wholly_contained_in_prev =
                    prev_bubble_size >= prev_curr_dist + curr_bubble_size + min_overlap_distance_;
            const bool prev_bubble_overlaps_curr_center_by_minimum =
                    prev_bubble_size >= prev_curr_dist * node_removal_overlap_factor_ + min_overlap_distance_;
            const bool next_bubble_overlaps_curr_center_by_minimum =
                    next_bubble_size >= curr_next_dist * node_removal_overlap_factor_ + min_overlap_distance_;

            const double angle_defined_by_points = EigenHelpers::AngleDefinedByPoints(prev, curr, next);
            assert(angle_defined_by_points >= 0.0);
            const bool band_backtracks = angle_defined_by_points < backtrack_threshold_;

//            if (band_backtracks)
//            {
//                std::cout << "Backtrack detected\n";

//                vis_.visualizePoints("prev", {prev}, Visualizer::Blue(), 1, 0.01);
//                vis_.visualizePoints("curr", {curr}, Visualizer::Magenta(), 1, 0.01);
//                vis_.visualizePoints("next", {next}, Visualizer::Red(), 1, 0.01);

//                std::cout << prev.transpose() << std::endl;
//                std::cout << curr.transpose() << std::endl;
//                std::cout << next.transpose() << std::endl;
//                std::cout << angle_defined_by_points << std::endl;
//                std::cout << std::endl;
//            }

            // Only keep this point if there is not sufficient overlap for the neighbouring bubbles
            if (!(prev_bubble_overlaps_curr_center_by_minimum && next_bubble_overlaps_curr_center_by_minimum) &&
                !curr_bubble_is_wholly_contained_in_prev &&
                !band_backtracks)
            {
                backward_pass.push_back(curr);
            }
        }
        backward_pass.push_back(band_.front());

        band_ = backward_pass;
        std::reverse(band_.begin(), band_.end());
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        assert(bandIsValidWithVisualization());
    }
}

void QuinlanRubberBand::smoothBandPoints(const bool verbose)
{
    assert(bandIsValidWithVisualization());

//    const auto initial_band = band_;

//    if (verbose)
//    {
//        std::cout << "Pre Point, bubble size, Angles:\n";
//        for (size_t idx = 0; idx < band_.size(); ++idx)
//        {
//            std::cout << std::setprecision(12) << band_[idx].transpose();
//            std::cout << std::setprecision(12) << "    " << getBubbleSize(band_[idx]);
//            if (idx > 0 && idx + 1 < band_.size())
//            {
//                std::cout << std::setprecision(12) << "    " << EigenHelpers::AngleDefinedByPoints(band_[idx - 1], band_[idx], band_[idx + 1]);
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }


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

            const double prev_bubble_size = getBubbleSize(prev);
            const double curr_bubble_size = getBubbleSize(curr);
            const double next_bubble_size = getBubbleSize(next);

            // Project the current point onto the line between prev and next
            const Eigen::Vector3d band_tangent_approx = next - prev;
            const Eigen::Vector3d curr_projected_onto_tangent = prev + EigenHelpers::VectorProjection(band_tangent_approx, curr - prev);
            const Eigen::Vector3d delta = curr_projected_onto_tangent - curr;
            assert(delta.dot(band_tangent_approx) < 1e-10);

            // Determine if the projection is within the bubble at the current point, and if not only move part way
            const double max_delta_norm = curr_bubble_size - min_distance_to_obstacle_;
            const Eigen::Vector3d curr_prime = delta.norm() <= max_delta_norm ? curr_projected_onto_tangent : curr + max_delta_norm * delta.normalized();


            if (!sdf_.CheckInBounds3d(curr_prime))
            {
                const Eigen::Vector3d min = sdf_.GetOriginTransform().translation();
                const Eigen::Vector3d max = min + Eigen::Vector3d(sdf_.GetXSize(), sdf_.GetYSize(), sdf_.GetZSize());

                std::cerr << "SDF limits: x, y, z\n"
                          << "Max:            " << max.transpose() << std::endl
                          << "Min:            " << min.transpose() << std::endl;
                std::cerr << "prev:           " << prev.transpose() << " bubble size: " << prev_bubble_size << std::endl
                          << "curr:           " << curr.transpose() << " bubble size: " << curr_bubble_size << std::endl
                          << "next:           " << next.transpose() << " bubble size: " << next_bubble_size << std::endl
                          << "tangent:        " << band_tangent_approx.transpose() << std::endl
                          << "curr_projected: " << curr_projected_onto_tangent.transpose() << std::endl
                          << "curr_prime:     " << curr_prime.transpose() << std::endl;

                const double prev_bubble_size = getBubbleSize(prev);
                const double curr_bubble_size = getBubbleSize(curr);
                const double next_bubble_size = getBubbleSize(next);

                std::cerr << std::endl << std::endl;
            }
            assert(sdf_.CheckInBounds3d(curr_prime));


            // Ensure that the resulting point is not in collision even with numerical rounding
            const Eigen::Vector3d curr_prime_projected_to_distance = projectToValidBubble(curr_prime);
            const double projected_bubble_size = getBubbleSize(curr_prime_projected_to_distance);

            // Check if the bubbles still overlap on each side
            const double prev_curr_dist = (prev - curr_prime_projected_to_distance).norm();
            const double curr_next_dist = (next - curr_prime_projected_to_distance).norm();

            const bool prev_bubble_overlaps_curr = sufficientOverlap(prev_bubble_size, projected_bubble_size, prev_curr_dist);
            const bool next_bubble_overlaps_curr = sufficientOverlap(next_bubble_size, projected_bubble_size, curr_next_dist);

            if (!prev_bubble_overlaps_curr)
            {
                interpolateBetweenPoints(next_band, curr_prime_projected_to_distance);
            }
            next_band.push_back(curr_prime_projected_to_distance);

//            std::cout << "Original bubble size: " << getBubbleSize(curr)
//                      << "     New bubble size: " << getBubbleSize(curr_prime_projected_to_distance)
//                      << "              Change: " << getBubbleSize(curr_prime_projected_to_distance) - getBubbleSize(curr) << std::endl;

//            if (getBubbleSize(curr_prime_projected_to_distance) - getBubbleSize(curr) < 0.0)
//            {
//                const auto curr_grid_indices = sdf_.LocationToGridIndex3d(curr);
//                const auto curr_cell_center = sdf_.GridIndexToLocation(curr_grid_indices[0], curr_grid_indices[1], curr_grid_indices[2]);

//                const auto new_grid_indices = sdf_.LocationToGridIndex3d(curr_prime_projected_to_distance);
//                const auto new_cell_center = sdf_.GridIndexToLocation(new_grid_indices[0], new_grid_indices[1], new_grid_indices[2]);

//                std::cout << std::setprecision(12)
//                          << "Original Position: " << curr.transpose() << std::endl
//                          << "New Position:      " << curr_prime_projected_to_distance.transpose() << std::endl
//                          << "Orig Cell center:  " << curr_cell_center[0] << " " << curr_cell_center[1] << " " << curr_cell_center[2] << std::endl
//                          << "New Cell center:   " << new_cell_center[0] << " " << new_cell_center[1] << " " << new_cell_center[2] << std::endl
//                          << "Original SDF Dist: " << sdf_.Get3d(curr) << std::endl
//                          << "New SDF Dist:      " << sdf_.Get3d(curr_prime_projected_to_distance) << std::endl
//                          << "Original SDF EstD: " << sdf_.EstimateDistance3d(curr).first << std::endl
//                          << "New SDF EstD:      " << sdf_.EstimateDistance3d(curr_prime_projected_to_distance).first << std::endl
//                          << "SDF Gradient:      " << PrettyPrint::PrettyPrint(sdf_.GetGradient(curr_cell_center[0], curr_cell_center[1], curr_cell_center[2], true), false, " ") << std::endl;
//            }

            if (!next_bubble_overlaps_curr)
            {
                interpolateBetweenPoints(next_band, next);
            }
        }
//        std::cout << std::endl;

        // The end doesn't move, so push that point on at the end, then swap buffers
        next_band.push_back(band_.back());
        band_ = next_band;
        assert(bandIsValidWithVisualization());

        removeExtraBandPoints();
//        std::cout << PrettyPrint::PrettyPrint(band_, false, "\n") << std::endl << std::endl;
    }

    //    if (verbose)
//    {
//        std::cout << PrettyPrint::PrettyPrint(band_, false, "\n") << std::endl << std::endl;
//        std::cout << "Post Point, bubble size, Angles:\n";
//        for (size_t idx = 0; idx < band_.size(); ++idx)
//        {
//            std::cout << std::setprecision(12) << band_[idx].transpose();
//            std::cout << std::setprecision(12) << "    " << getBubbleSize(band_[idx]);
//            if (idx > 0 && idx + 1 < band_.size())
//            {
//                std::cout << std::setprecision(12) << "    " << EigenHelpers::AngleDefinedByPoints(band_[idx - 1], band_[idx], band_[idx + 1]);
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }

    assert(bandIsValidWithVisualization());
}




void QuinlanRubberBand::storeBand() const
{
    return;
    try
    {
        const auto log_folder = ROSHelpers::GetParamRequired<std::string>(ph_, "log_folder", __func__);
        if (!log_folder.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load log_folder from parameter server");
        }
        arc_utilities::CreateDirectory(log_folder.GetImmutable());
        const auto file_name_prefix = ROSHelpers::GetParamRequired<std::string>(ph_, "band_file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load band_file_name_prefix from parameter server");
        }

        // Get a time string formated as YYYY-MM-DD__HH-MM-SS-milliseconds



        // https://stackoverflow.com/questions/24686846/get-current-time-in-milliseconds-or-hhmmssmmm-format
        using namespace std::chrono;

        // get current time
        const auto now = system_clock::now();

        // get number of milliseconds for the current second
        // (remainder after division into seconds)
        const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        // convert to std::time_t in order to convert to std::tm (broken time)
        const auto timer = system_clock::to_time_t(now);

        // convert to broken time
        std::tm bt = *std::localtime(&timer);

        std::ostringstream oss;
        oss << std::put_time(&bt, "%Y-%m-%d__%H-%M-%S");
        oss << '-' << std::setfill('0') << std::setw(3) << ms.count();
        const std::string file_name_suffix = oss.str();




        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix + ".compressed";
        const std::string full_path = log_folder.GetImmutable() + file_name;
        ROS_INFO_STREAM("Saving band to " << full_path);

        std::vector<uint8_t> buffer;
        arc_utilities::SerializeVector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                    band_, buffer, &arc_utilities::SerializeEigenVector3d);
        ZlibHelpers::CompressAndWriteToFile(buffer, full_path);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to store band: "  <<  e.what());
    }
}

void QuinlanRubberBand::loadStoredBand()
{
    try
    {
        const auto log_folder = ROSHelpers::GetParamRequired<std::string>(ph_, "log_folder", __func__);
        if (!log_folder.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load log_folder from parameter server");
        }
        const auto file_name_prefix = ROSHelpers::GetParamRequired<std::string>(ph_, "band_file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load band_file_name_prefix from parameter server");
        }
        const auto file_name_suffix = ROSHelpers::GetParamRequired<std::string>(ph_, "band_file_name_suffix_to_load", __func__);
        if (!file_name_suffix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load band_file_name_suffix_to_load from parameter server");
        }

        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix.GetImmutable() + ".compressed";
        const std::string full_path = log_folder.GetImmutable() + file_name;
        ROS_INFO_STREAM("Loading band from " << full_path);

        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(full_path);
        const auto deserialized_results = arc_utilities::DeserializeVector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                    buffer, 0, &arc_utilities::DeserializeEigenVector3d);
        band_ = deserialized_results.first;

    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to load stored band: "  <<  e.what());
    }
}

bool QuinlanRubberBand::useStoredBand() const
{
    return false;
    return ROSHelpers::GetParam<bool>(ph_, "use_stored_band", false);
}


