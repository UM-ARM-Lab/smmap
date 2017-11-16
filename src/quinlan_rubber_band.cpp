#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/serialization_eigen.hpp>
#include "smmap/quinlan_rubber_band.h"
#include <thread>

using namespace smmap;
using namespace smmap_utilities;
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
    , min_distance_to_obstacle_(sdf_.GetResolution() * 0.1)
    , node_removal_overlap_factor_(1.2)
    , backtrack_threshold_(0.1)
    , smoothing_iterations_(50)
{
    (void)generator;
    setPointsAndSmooth(starting_points);
    assert(bandIsValidWithVisualization());
}

QuinlanRubberBand& QuinlanRubberBand::operator=(const QuinlanRubberBand& other)
{
    assert(task_ == other.task_);
    assert(&sdf_ == &(other.sdf_));
    assert(&vis_ == &(other.vis_));

    assert(max_total_band_distance_ == other.max_total_band_distance_);

    band_ = other.band_;
#if ENABLE_BAND_LOAD_SAVE
    if (useStoredBand())
    {
        loadStoredBand();
    }
    else
    {
        storeBand();
    }
#endif

    assert(bandIsValidWithVisualization());
    return *this;
}

void QuinlanRubberBand::setPointsWithoutSmoothing(const EigenHelpers::VectorVector3d& points)
{
    band_ = points;

#if ENABLE_BAND_LOAD_SAVE
    if (useStoredBand())
    {
        loadStoredBand();
    }
    else
    {
        storeBand();
    }
#endif

    for (auto& point: band_)
    {
        point = projectToValidBubble(point);
    }
    interpolateBandPoints();
    const bool verbose = true;
    removeExtraBandPoints(verbose);
    assert(bandIsValidWithVisualization());
}

void QuinlanRubberBand::setPointsAndSmooth(const EigenHelpers::VectorVector3d& points)
{
    setPointsWithoutSmoothing(points);
    const bool verbose = true;
    smoothBandPoints(verbose);
    assert(bandIsValidWithVisualization());
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
    assert(bandIsValidWithVisualization());

    // Ensure that the new points are both in bounds, and are at least min_distance_to_obstacle_ from anything
    // Add the new endpoints, then let the interpolate and smooth process handle the propogation
    band_.insert(band_.begin(), projectToValidBubble(first_endpoint_target));
    band_.push_back(projectToValidBubble(second_endpoint_target));

#if ENABLE_BAND_LOAD_SAVE
    if (useStoredBand())
    {
        loadStoredBand();
    }
    else
    {
        storeBand();
    }
#endif

    verbose = true;
    interpolateBandPoints();
    removeExtraBandPoints(verbose);
    smoothBandPoints(verbose);

    assert(bandIsValidWithVisualization());
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
//        vis_.visualizePoints(marker_name + "_points", band_, Visualizer::Green(), 1, 0.002);
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
        visualize(marker_name, safe_color, overstretched_color, id, visualization_enabled);

        // Delete all sphere, markers, probably from just this publisher, and then republish
#if ENABLE_DEBUGGING
        {
            vis_.deleteObjects(marker_name + "_bubbles", 1, 305);
            std::this_thread::sleep_for(std::chrono::duration<double>(0.001));

            std::vector<double> bubble_sizes(band_.size());
            std::vector<std_msgs::ColorRGBA> colors(band_.size());
            for (size_t idx = 0; idx < band_.size(); ++idx)
            {
                bubble_sizes[idx] = getBubbleSize(band_[idx]);
                colors[idx] = ColorBuilder::MakeFromFloatColors(
                            (float)idx / (float)(band_.size() - 1),
                            0.0f,
                            (float)(band_.size() - 1 - idx) / (float)(band_.size() - 1),
                            0.3f);
            }
            vis_.visualizeSpheres(marker_name + "_bubbles", band_, colors, id, bubble_sizes);
            std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
        }
#endif
    }
}






Eigen::Vector3d QuinlanRubberBand::projectToValidBubble(const Eigen::Vector3d& location) const
{
    if (getBubbleSize(location) >= min_distance_to_obstacle_)
    {
        return location;
    }

    const auto post_collision_project = sdf_.ProjectOutOfCollisionToMinimumDistance3d(location, min_distance_to_obstacle_);
    const auto post_boundary_project = sdf_.ProjectIntoValidVolumeToMinimumDistance3d(post_collision_project, min_distance_to_obstacle_);

#if ENABLE_DEBUGGING
    std::cout << std::setprecision(12)
              << "location:       " << location.transpose() << std::endl
              << "post collision: " << post_collision_project.transpose() << std::endl
              << "post boundary:  " << post_boundary_project.transpose() << std::endl
              << "Resulting size:        " << getBubbleSize(post_boundary_project) << std::endl;



    if (getBubbleSize(post_boundary_project) < min_distance_to_obstacle_)
    {
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
            const auto post_boundary_project = sdf_.ProjectIntoValidVolumeToMinimumDistance3d(post_collision_project, min_distance_to_obstacle_);
        }
    }
    assert(getBubbleSize(post_boundary_project) >= min_distance_to_obstacle_);
#endif
    return post_boundary_project;
}

double QuinlanRubberBand::getBubbleSize(const Eigen::Vector3d& location) const
{
#if ENABLE_DEBUGGING
    vis_.visualizePoints("get_bubble_size_test_location", {location}, Visualizer::Orange(), 1, 0.005);
//    std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
#endif

    return sdf_.EstimateDistance3d(location).first;


    const Eigen::Vector4d loc_4d(location.x(), location.y(), location.z(), 1.0);
    const auto distance_to_boundary = sdf_.DistanceToBoundary4d(loc_4d);
    const auto distance_to_obstacles = sdf_.EstimateDistance4d(loc_4d);
    assert(distance_to_boundary.second == distance_to_obstacles.second);
    const auto distance = std::min(distance_to_boundary.first, distance_to_obstacles.first);
//    std::cout << std::setprecision(12) << std::setw(15) << location.transpose() << " Estimate dist: " << distance_to_obstacles.first << " Boundary dist: " << distance_to_boundary.first << " Reported distance: " << distance << std::endl;
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
#if ENABLE_DEBUGGING
    if (band_.size() < 2)
    {
        return false;
    }

    for (size_t node_idx = 0; node_idx < band_.size() - 1; ++node_idx)
    {
        const auto& curr = band_[node_idx];
        const auto& next = band_[node_idx + 1];
        const double curr_bubble_size = getBubbleSize(curr);
        const double next_bubble_size = getBubbleSize(next);
        const double dist = (curr - next).norm();
        if (!sdf_.CheckInBounds3d(curr) ||
            curr_bubble_size < min_distance_to_obstacle_ ||
            !sufficientOverlap(curr_bubble_size, next_bubble_size, dist))
        {
            std::cerr << "Problem between node " << node_idx << " and " << node_idx + 1 << std::endl
                      << "In bounds: " << sdf_.CheckInBounds3d(curr) << std::endl
                      << "Curr bubble size: " << curr_bubble_size << std::endl
                      << "Next bubble size: " << next_bubble_size << std::endl
                      << "Curr + next:      " << curr_bubble_size + next_bubble_size << std::endl
                      << "Dist + min:       " << dist + min_overlap_distance_ << std::endl;
            return false;
        }
    }
    if (getBubbleSize(band_.back()) < min_distance_to_obstacle_)
    {
        std::cerr << "Problem at last node: "
                  << "Bubble size:        " << getBubbleSize(band_.back()) << std::endl;
        return false;
    }
#endif
    return true;
}

bool QuinlanRubberBand::bandIsValidWithVisualization() const
{
#if ENABLE_DEBUGGING
    if (!bandIsValid())
    {
        visualizeWithBubbles("quinlan_band_something_is_invalid", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        printBandData();
        return bandIsValid();
    }
#endif
    return true;
}

/**
  * Interpolates bewteen the end of point_buffer and target, but does not add target to the buffer
  */
#if ENABLE_DEBUGGING
void QuinlanRubberBand::interpolateBetweenPoints(
        EigenHelpers::VectorVector3d& point_buffer,
        const Eigen::Vector3d& target)
#else
void QuinlanRubberBand::interpolateBetweenPoints(
        EigenHelpers::VectorVector3d& point_buffer,
        const Eigen::Vector3d& target) const
#endif
{
#if ENABLE_DEBUGGING
    const auto starting_band = band_;
    band_ = point_buffer;
    if (point_buffer.size() >= 2)
    {
        assert(bandIsValidWithVisualization());
    }
#endif

    const double target_bubble_size = getBubbleSize(target);

    // TODO: verify that this cannot get stuck in an infinite loop
    int outer_iteration_counter = 0;

    // Check if the bubbles for 2 adjacent nodes overlap with some minimum distance to spare
    double curr_bubble_size = getBubbleSize(point_buffer.back());

    double distance_to_end = (target - point_buffer.back()).norm();
    while (!sufficientOverlap(curr_bubble_size, target_bubble_size, distance_to_end))
    {
        const Eigen::Vector3d curr = point_buffer.back();

        // Find a position between point_buffer.back() and next_node with sufficient bubble overlap
        // TODO: verify that this cannot get stuck in an infinite loop
        int inner_iteration_counter = 0;
        double interpolation_ratio = 0.5;
        Eigen::Vector3d interpolated_point = EigenHelpers::Interpolate(curr, target, interpolation_ratio);
        Eigen::Vector3d test_point = projectToValidBubble(interpolated_point);
        double test_point_bubble_size = getBubbleSize(test_point);
        double distance_between_curr_and_test_point = (curr - test_point).norm();

#if ENABLE_DEBUGGING
        assert(sdf_.CheckInBounds3d(curr));
        assert(sdf_.CheckInBounds3d(target));
        assert(sdf_.CheckInBounds3d(interpolated_point));
        assert(sdf_.CheckInBounds3d(test_point));
#endif
        ROS_WARN_STREAM_COND_NAMED(outer_iteration_counter == 20, "rubber_band", "Rubber band interpolation outer loop counter at " << outer_iteration_counter << ", probably stuck in an infinite loop");
#if ENABLE_DEBUGGING
        if (outer_iteration_counter >= 20)
        {
            std::cout << "\n\n\n\n\n\n\nCurr:\n";
            getBubbleSize(curr);
            std::cout << "\nInterp:\n";
            const double interpolated_point_bubble_size = getBubbleSize(interpolated_point);
            std::cout << "\nTest:\n";
            getBubbleSize(test_point);
            std::cout << "\nTarget\n";
            getBubbleSize(target);
            std::cout << std::endl;

            std::cerr << std::setprecision(12)
                      << "Curr:   " << curr.transpose() << std::endl
                      << "Target: " << target.transpose() << std::endl
                      << "Interp: " << interpolated_point.transpose() << std::endl
                      << "Test:   " << test_point.transpose() << std::endl
                      << std::endl;
            vis_.visualizePoints( "interpolate_outer_debugging_curr_point",    {curr},                Visualizer::Red(1.0f),     1, 0.005);
            vis_.visualizeSpheres("interpolate_outer_debugging_curr_sphere",   {curr},                Visualizer::Red(0.2f),     2, curr_bubble_size);
            vis_.visualizePoints( "interpolate_outer_debugging_interp_point",  {interpolated_point},  Visualizer::Green(1.0f),   1, 0.005);
            vis_.visualizeSpheres("interpolate_outer_debugging_interp_sphere", {interpolated_point},  Visualizer::Green(0.2f),   2, interpolated_point_bubble_size);
            vis_.visualizePoints( "interpolate_outer_debugging_test_point",    {test_point},          Visualizer::Cyan(1.0f),    1, 0.005);
            vis_.visualizeSpheres("interpolate_outer_debugging_test_sphere",   {test_point},          Visualizer::Cyan(0.2f),    2, test_point_bubble_size);
            vis_.visualizePoints( "interpolate_outer_debugging_target_point",  {target},              Visualizer::Blue(1.0f),    1, 0.005);
            vis_.visualizeSpheres("interpolate_outer_debugging_target_sphere", {target},              Visualizer::Blue(0.2f),    2, target_bubble_size);

            vis_.deleteObjects("null", 1, 10);
        }
#endif

        while (!sufficientOverlap(curr_bubble_size, test_point_bubble_size, distance_between_curr_and_test_point))
        {
            ROS_WARN_STREAM_COND_NAMED(inner_iteration_counter == 5, "rubber_band", "Rubber band interpolation inner loop counter at " << inner_iteration_counter << ", probably stuck in an infinite loop");
#if ENABLE_DEBUGGING
            if (inner_iteration_counter >= 5)
            {
                std::cout << "\n\n\n\n\n\n\nCurr:\n";
                getBubbleSize(curr);
                std::cout << "\nInterp:\n";
                const double interpolated_point_bubble_size = getBubbleSize(interpolated_point);
                std::cout << "\nTest:\n";
                getBubbleSize(test_point);
                std::cout << "\nTarget\n";
                getBubbleSize(target);
                std::cout << std::endl;

                vis_.visualizePoints( "interpolate_inner_debugging_curr_point",    {curr},                Visualizer::Red(1.0f),     1, 0.005);
                vis_.visualizeSpheres("interpolate_inner_debugging_curr_sphere",   {curr},                Visualizer::Red(0.2f),     2, curr_bubble_size);
                vis_.visualizePoints( "interpolate_inner_debugging_interp_point",  {interpolated_point},  Visualizer::Green(1.0f),   1, 0.005);
                vis_.visualizeSpheres("interpolate_inner_debugging_interp_sphere", {interpolated_point},  Visualizer::Green(0.2f),   2, interpolated_point_bubble_size);
                vis_.visualizePoints( "interpolate_inner_debugging_test_point",    {test_point},          Visualizer::Cyan(1.0f),    1, 0.005);
                vis_.visualizeSpheres("interpolate_inner_debugging_test_sphere",   {test_point},          Visualizer::Cyan(0.2f),    2, test_point_bubble_size);
                vis_.visualizePoints( "interpolate_inner_debugging_target_point",  {target},              Visualizer::Blue(1.0f),    1, 0.005);
                vis_.visualizeSpheres("interpolate_inner_debugging_target_sphere", {target},              Visualizer::Blue(0.2f),    2, target_bubble_size);

                std::cerr << std::setprecision(12)
                          << "Curr:   " << curr.transpose() << std::endl
                          << "Interp: " << interpolated_point.transpose() << std::endl
                          << "Test:   " << test_point.transpose() << std::endl
                          << "Target: " << target.transpose() << std::endl
                          << "curr bubble size:   " << curr_bubble_size << std::endl
                          << "interp bubble size: " << interpolated_point_bubble_size << std::endl
                          << "test bubble size:   " << test_point_bubble_size << std::endl
                          << "target bubble size: " << target_bubble_size << std::endl
                          << "curr + test:        " << curr_bubble_size + test_point_bubble_size << std::endl
                          << "dist + min:         " << distance_between_curr_and_test_point + min_overlap_distance_ << std::endl
                          << std::endl;

                vis_.deleteObjects("null", 1, 10);
            }
#endif

            interpolation_ratio *= 0.5;
            interpolated_point = EigenHelpers::Interpolate(curr, target, interpolation_ratio);
            test_point = projectToValidBubble(interpolated_point);
            test_point_bubble_size = getBubbleSize(test_point);
            distance_between_curr_and_test_point = (curr - test_point).norm();

            ++inner_iteration_counter;
        }
        // The bubbles now overlap sufficiently, so accept this point and record the new values
        point_buffer.push_back(test_point);
        curr_bubble_size = test_point_bubble_size;
        distance_to_end = (target - test_point).norm();

#if ENABLE_DEBUGGING
        band_ = point_buffer;
        assert(bandIsValidWithVisualization());
#endif

        ++outer_iteration_counter;
    }

#if ENABLE_DEBUGGING
    band_ = point_buffer;
    band_.push_back(target);
    visualizeWithBubbles("end_of_interpolate_between_points", Visualizer::Black(), Visualizer::Cyan(), 1, true);
    assert(bandIsValidWithVisualization());
    band_ = starting_band;
#endif
}

/**
 * @brief QuinlanRubberBand::interpolateBandPoints
 * Re-interpolates the entire band, not to be used for just 1 segment
 */
void QuinlanRubberBand::interpolateBandPoints()
{
#if ENABLE_DEBUGGING
    assert(band_.size() >= 2);
    for (size_t idx = 0; idx < band_.size(); ++idx)
    {
        if (getBubbleSize(band_[idx]) < min_distance_to_obstacle_)
        {
            std::cerr << "idx: " << idx << " point: " << band_[idx].transpose() << " size: " << getBubbleSize(band_[idx]) << std::endl;
        }

        assert(getBubbleSize(band_[idx]) >= min_distance_to_obstacle_);
    }
#endif

    EigenHelpers::VectorVector3d new_band(1, band_.front());
    for (size_t idx = 0; idx + 1 < band_.size(); ++idx)
    {
#if ENABLE_DEBUGGING
        const auto next_node = band_[idx + 1];
#else
        const auto& next_node = band_[idx + 1];
#endif
        interpolateBetweenPoints(new_band, next_node);
        new_band.push_back(next_node);
#if ENABLE_DEBUGGING
        const auto temp = band_;
        band_ = new_band;
        assert(bandIsValidWithVisualization());
        band_ = temp;
#endif
    }

    band_ = new_band;

    assert(bandIsValidWithVisualization());
}

void QuinlanRubberBand::removeExtraBandPoints(const bool verbose)
{
    (void)verbose;
    assert(bandIsValidWithVisualization());

#if ENABLE_DEBUGGING
    if (verbose)
    {
        std::cout << "Start of removeExtraBandPoints\n";
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        printBandData();
    }
#endif

    // The start can't be removed, so push that point on immediately
    EigenHelpers::VectorVector3d forward_pass;
    forward_pass.reserve(band_.size());
    forward_pass.push_back(band_.front());
#if ENABLE_DEBUGGING
    vis_.visualizePoints( "remove_extra_test_points_kept_points",  {forward_pass.back()}, Visualizer::Cyan(1.0f), (int32_t)forward_pass.size(), 0.002);
    vis_.visualizeSpheres("remove_extra_test_points_kept_spheres", {forward_pass.back()}, Visualizer::Cyan(0.2f), (int32_t)forward_pass.size(), getBubbleSize(forward_pass.back()));
#endif

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
        const double prev_next_dist = (next - prev).norm();

        const bool curr_bubble_is_wholey_contained_in_prev =
                prev_bubble_size > prev_curr_dist + curr_bubble_size + min_overlap_distance_;

        const bool curr_bubble_is_wholey_contained_in_next =
                next_bubble_size > curr_next_dist + curr_bubble_size + min_overlap_distance_;

        // Discard this point if it is containted in either neighbouring bubble
        if (curr_bubble_is_wholey_contained_in_prev || curr_bubble_is_wholey_contained_in_next)
        {
            continue;
        }

        // Discard this point if prev overlaps next by enough
        if (sufficientOverlap(prev_bubble_size, next_bubble_size, prev_next_dist * node_removal_overlap_factor_))
        {
            continue;
        }

        // Discard this point if it is too close to the previous, or too close to the next
        if (prev.isApprox(curr) || next.isApprox(curr))
        {
            continue;
        }

        // Only keep points if they do not backtrack
        const double angle_defined_by_points = EigenHelpers::AngleDefinedByPoints(prev, curr, next);
        assert(angle_defined_by_points >= 0.0);
        const bool band_backtracks = angle_defined_by_points < backtrack_threshold_;

        if (band_backtracks)
        {
//                std::cout << "Backtrack detected\n";

//                vis_.visualizePoints("prev", {prev}, Visualizer::Blue(), 1, 0.01);
//                vis_.visualizePoints("curr", {curr}, Visualizer::Magenta(), 1, 0.01);
//                vis_.visualizePoints("next", {next}, Visualizer::Red(), 1, 0.01);

//                std::cout << prev.transpose() << std::endl;
//                std::cout << curr.transpose() << std::endl;
//                std::cout << next.transpose() << std::endl;
//                std::cout << angle_defined_by_points << std::endl;
//                std::cout << std::endl;
            continue;
        }

#if ENABLE_DEBUGGING
        if (verbose)
        {
            vis_.visualizePoints( "remove_extra_test_prev", {prev}, Visualizer::Red(1.0f),   1, 0.002);
            vis_.visualizeSpheres("remove_extra_test_prev", {prev}, Visualizer::Red(0.2f),   2, prev_bubble_size);
            vis_.visualizePoints( "remove_extra_test_curr", {curr}, Visualizer::Green(1.0f), 1, 0.002);
            vis_.visualizeSpheres("remove_extra_test_curr", {curr}, Visualizer::Green(0.2f), 2, curr_bubble_size);
            vis_.visualizePoints( "remove_extra_test_next", {next}, Visualizer::Blue(1.0f),  1, 0.002);
            vis_.visualizeSpheres("remove_extra_test_next", {next}, Visualizer::Blue(0.2f),  2, next_bubble_size);
            std::this_thread::sleep_for(std::chrono::duration<double>(0.01));



            std::cout << "prev bubble size: " << prev_bubble_size << std::endl;
            std::cout << "curr bubble size: " << curr_bubble_size << std::endl;
            std::cout << "next bubble size: " << next_bubble_size << std::endl;
            std::cout << "prev-curr dist:   " << prev_curr_dist << std::endl;
            std::cout << "curr-next dist:   " << curr_next_dist << std::endl;
            std::cout << "prev-next dist:   " << prev_next_dist << std::endl;
        }
#endif
        // If no item said we should delete this item, then keep it
        forward_pass.push_back(curr);
#if ENABLE_DEBUGGING
        vis_.visualizePoints( "remove_extra_test_points_kept_points",  {forward_pass.back()}, Visualizer::Cyan(1.0f), (int32_t)forward_pass.size(), 0.002);
        vis_.visualizeSpheres("remove_extra_test_points_kept_spheres", {forward_pass.back()}, Visualizer::Cyan(0.2f), (int32_t)forward_pass.size(), getBubbleSize(forward_pass.back()));
#endif
    }
    forward_pass.push_back(band_.back());
#if ENABLE_DEBUGGING
    vis_.visualizePoints( "remove_extra_test_points_kept_points",  {forward_pass.back()}, Visualizer::Cyan(1.0f), (int32_t)forward_pass.size(), 0.002);
    vis_.visualizeSpheres("remove_extra_test_points_kept_spheres", {forward_pass.back()}, Visualizer::Cyan(0.2f), (int32_t)forward_pass.size(), getBubbleSize(forward_pass.back()));
#endif

    band_ = forward_pass;
    visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
    assert(bandIsValidWithVisualization());
}

void QuinlanRubberBand::smoothBandPoints(const bool verbose)
{
    assert(bandIsValidWithVisualization());

    for (size_t smoothing_iter = 0; smoothing_iter < smoothing_iterations_; ++smoothing_iter)
    {
#if ENABLE_DEBUGGING
        if (verbose)
        {
            visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
            std::cerr << "Start of loop smoothBandPoints\n";
            printBandData();
        }
#endif

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

            // Only allow movement that points directly between next and prev
            const Eigen::Vector3d allowed_movement_direction = (next - curr).normalized() + (prev - curr).normalized();
            // If the allowed direction is numerically close to zero, then we are already in
            // nearly a straight line with our neighbours, so don't move
            // TODO: address magic number
            if (allowed_movement_direction.norm() < 1e-4)
            {
                next_band.push_back(curr);
                continue;
            }

            // The optimal point is directly between prev and next, so move as far that way as our bubble allows
            const Eigen::Vector3d midpoint = prev + (next - prev) / 2.0;
            const Eigen::Vector3d delta_raw = midpoint - curr;
            const Eigen::Vector3d delta = EigenHelpers::VectorProjection(allowed_movement_direction, delta_raw);
            // Determine if the projection is within the bubble at the current point, and if not only move part way
            // Only step at most half way there in order to try and avoid oscillations - to large of a step size is bad
            const double max_delta_norm = std::max(0.0, (curr_bubble_size - min_distance_to_obstacle_ * 1.00001) * 0.5);
            const bool curr_plus_delta_inside_bubble = delta.norm() <= max_delta_norm;
            const Eigen::Vector3d prime =  curr_plus_delta_inside_bubble ? Eigen::Vector3d(curr + delta) : Eigen::Vector3d(curr + max_delta_norm * delta.normalized());
            // Ensure that the resulting point is not in collision even with numerical rounding (and weirdness in the SDF)
            const Eigen::Vector3d projected = projectToValidBubble(prime);
            const double projected_bubble_size = getBubbleSize(projected);

#if ENABLE_DEBUGGING
            if (projected_bubble_size < min_distance_to_obstacle_)
            {
                const Eigen::Vector3d projected_testing = projectToValidBubble(prime);
                const double projected_bubble_size = getBubbleSize(projected_testing);
                std::cout << std::setprecision(12)
                          << "Projected: " << projected_testing.transpose() << std::endl
                          << "Proj bubble size: " << projected_bubble_size << std::endl;
            }
            const double prime_bubble_size = getBubbleSize(prime);

            if (verbose)
            {
                vis_.visualizePoints( "smoothing_test_prev",      {prev},      Visualizer::Red(1.0f),     1, 0.002);
                vis_.visualizeSpheres("smoothing_test_prev",      {prev},      Visualizer::Red(0.2f),     2, prev_bubble_size);
                vis_.visualizePoints( "smoothing_test_curr",      {curr},      Visualizer::Green(1.0f),   1, 0.002);
                vis_.visualizeSpheres("smoothing_test_curr",      {curr},      Visualizer::Green(0.2f),   2, curr_bubble_size);
                vis_.visualizePoints( "smoothing_test_prime",     {prime},     Visualizer::Cyan(1.0f),    1, 0.002);
                vis_.visualizeSpheres("smoothing_test_prime",     {prime},     Visualizer::Cyan(0.2f),    2, prime_bubble_size);
                vis_.visualizePoints( "smoothing_test_projected", {projected}, Visualizer::Magenta(1.0f), 1, 0.002);
                vis_.visualizeSpheres("smoothing_test_projected", {projected}, Visualizer::Magenta(0.2f), 2, projected_bubble_size);
                vis_.visualizePoints( "smoothing_test_next",      {next},      Visualizer::Blue(1.0f),    1, 0.002);
                vis_.visualizeSpheres("smoothing_test_next",      {next},      Visualizer::Blue(0.2f),    2, next_bubble_size);
                std::this_thread::sleep_for(std::chrono::duration<double>(0.01));

                std::cout << std::setprecision(12)
                          << "Prev:                         " << prev.transpose() << std::endl
                          << "Curr:                         " << curr.transpose() << std::endl
                          << "Next:                         " << next.transpose() << std::endl
                          << "Prev - Curr:                  " << (prev - curr).normalized().transpose() << std::endl
                          << "Next - Curr:                  " << (next - curr).normalized().transpose() << std::endl
                          << "Allowed movement dir:         " << allowed_movement_direction.transpose() << std::endl
                          << "Delta raw:                    " << delta_raw.transpose() << std::endl
                          << "Delta:                        " << delta.transpose() << std::endl
                          << "Allowed cross delta norm:     " << delta.cross(allowed_movement_direction).norm() << std::endl
                          << std::endl
                          << "prev bubble size:  " << prev_bubble_size << std::endl
                          << "curr bubble size:  " << curr_bubble_size << std::endl
                          << "prime bubble size: " << prime_bubble_size << std::endl
                          << "proj bubble size:  " << projected_bubble_size << std::endl
                          << "next bubble size:  " << next_bubble_size << std::endl
                          << std::endl;
            }
#endif

            // Check if the bubbles still overlap on each side
            const double prev_curr_dist = (prev - projected).norm();
            const double curr_next_dist = (next - projected).norm();
            const bool prev_bubble_overlaps_curr = sufficientOverlap(prev_bubble_size, projected_bubble_size, prev_curr_dist);
            const bool next_bubble_overlaps_curr = sufficientOverlap(next_bubble_size, projected_bubble_size, curr_next_dist);
            if (!prev_bubble_overlaps_curr)
            {
                interpolateBetweenPoints(next_band, projected);
            }
            next_band.push_back(projected);
            if (!next_bubble_overlaps_curr)
            {
                interpolateBetweenPoints(next_band, next);
            }
        }

        // The end doesn't move, so push that point on at the end, then swap buffers
        next_band.push_back(band_.back());

        // Shortcut the process if there has been no meaningful change in the band
        // TODO: remove magic number
        if (EigenHelpers::CloseEnough(band_, next_band, 1e-4))
        {
            return;
        }

        band_ = next_band;
#if ENABLE_DEBUGGING
        if (verbose)
        {
            std::cout << "\n\n\nEnd of smoothing loop, iteration: " << smoothing_iter << std::endl;
            visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
            printBandData();
        }
        assert(bandIsValidWithVisualization());
#endif

        removeExtraBandPoints(verbose);
    }

#if ENABLE_DEBUGGING
    if (verbose)
    {
        printBandData();
    }
    assert(bandIsValidWithVisualization());
#endif
}



void QuinlanRubberBand::printBandData() const
{
#if !ENABLE_DEBUGGING
    return;
#endif
    const Eigen::Vector3d min = sdf_.GetOriginTransform().translation();
    const Eigen::Vector3d max = min + Eigen::Vector3d(sdf_.GetXSize(), sdf_.GetYSize(), sdf_.GetZSize());
    std::cout << "SDF limits: x, y, z\n"
              << "Max:            " << max.transpose() << std::endl
              << "Min:            " << min.transpose() << std::endl
              << "Band limits: Min dist to obstacle: " << min_distance_to_obstacle_ << "   Min overlap distance: " << min_overlap_distance_ << std::endl;

    std::cout << "                         Point                    ,    bubble size   ,     overlap    ,   Angles:\n";

    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(band_.size(), 6) * NAN;

    for (size_t idx = 0; idx < band_.size(); ++idx)
    {
        data.block<1, 3>(idx, 0) = band_[idx].transpose();
        data(idx, 3) = getBubbleSize(band_[idx]);
        if (idx > 0)
        {
            const double prev_bubble_size = getBubbleSize(band_[idx - 1]);
            const double curr_bubble_size = getBubbleSize(band_[idx]);
            const double distance_between_prev_and_curr = (band_[idx] - band_[idx-1]).norm();
            data(idx, 4) = (prev_bubble_size + curr_bubble_size) - distance_between_prev_and_curr;
        }
        if (idx > 0 && idx + 1 < band_.size())
        {
            data(idx, 5) = EigenHelpers::AngleDefinedByPoints(band_[idx - 1], band_[idx], band_[idx + 1]);
        }
    }
    std::cout << std::setprecision(12) << data << std::endl;
    std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
}



void QuinlanRubberBand::storeBand() const
{
    try
    {
        const auto log_folder = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "log_folder", __func__);
        if (!log_folder.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load log_folder from parameter server");
        }
        arc_utilities::CreateDirectory(log_folder.GetImmutable());
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "band_file_name_prefix", __func__);
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
        ROS_DEBUG_STREAM("Saving band to " << full_path);

        std::vector<uint8_t> buffer;
        arc_utilities::SerializeVector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                    band_, buffer, &arc_utilities::SerializeEigenType<Eigen::Vector3d>);
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
//        const auto log_folder = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "log_folder", __func__);
        const auto log_folder = ROSHelpers::GetParamRequired<std::string>(ph_, "log_folder", __func__);
        if (!log_folder.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load log_folder from parameter server");
        }
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "band_file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load band_file_name_prefix from parameter server");
        }
        const auto file_name_suffix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "band_file_name_suffix_to_load", __func__);
        if (!file_name_suffix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load band_file_name_suffix_to_load from parameter server");
        }

        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix.GetImmutable() + ".compressed";
        const std::string full_path = log_folder.GetImmutable() + file_name;
        ROS_INFO_STREAM("Loading band from " << full_path);

        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(full_path);
        const auto deserialized_results = arc_utilities::DeserializeVector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                    buffer, 0, &arc_utilities::DeserializeEigenType<Eigen::Vector3d>);
        band_ = deserialized_results.first;

    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to load stored band: "  <<  e.what());
    }
}

bool QuinlanRubberBand::useStoredBand() const
{
    return ROSHelpers::GetParamDebugLog<bool>(ph_, "use_stored_band", false);
}
