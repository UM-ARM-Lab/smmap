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
        const Visualizer::Ptr vis,
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
        const Visualizer::Ptr vis,
        std::mt19937_64& generator)
    : ph_("/smmap_planner_node/band")
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
    assert(vis_.get() == (other.vis_.get()));

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
    visualize(band_, marker_name, safe_color, overstretched_color, id, visualization_enabled);
}

void QuinlanRubberBand::visualize(
        const EigenHelpers::VectorVector3d& test_band,
        const std::string& marker_name,
        const std_msgs::ColorRGBA& safe_color,
        const std_msgs::ColorRGBA& overstretched_color,
        const int32_t id,
        const bool visualization_enabled) const
{
    if (visualization_enabled)
    {
//        vis_->visualizePoints(marker_name + "_points", test_band, Visualizer::Green(), 1, 0.002);
        if (isOverstretched())
        {
            vis_->visualizeXYZTrajectory(marker_name, test_band, overstretched_color, id);
        }
        else
        {
            vis_->visualizeXYZTrajectory(marker_name, test_band, safe_color, id);
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
    visualizeWithBubbles(band_, marker_name, safe_color, overstretched_color, id, visualization_enabled);
}

void QuinlanRubberBand::visualizeWithBubbles(
        const EigenHelpers::VectorVector3d& test_band,
        const std::string& marker_name,
        const std_msgs::ColorRGBA& safe_color,
        const std_msgs::ColorRGBA& overstretched_color,
        const int32_t id,
        const bool visualization_enabled) const
{
    if (visualization_enabled)
    {
        visualize(test_band, marker_name, safe_color, overstretched_color, id, visualization_enabled);

        // Delete all sphere, markers, probably from just this publisher, and then republish
#if ENABLE_BAND_DEBUGGING
        {
            vis_->deleteObjects(marker_name + "_bubbles", 1, 305);
            std::this_thread::sleep_for(std::chrono::duration<double>(0.001));

            std::vector<double> bubble_sizes(test_band.size());
            std::vector<std_msgs::ColorRGBA> colors(test_band.size());
            for (size_t idx = 0; idx < test_band.size(); ++idx)
            {
                bubble_sizes[idx] = getBubbleSize(test_band[idx]);
                colors[idx] = ColorBuilder::MakeFromFloatColors(
                            (float)idx / (float)(test_band.size() - 1),
                            0.0f,
                            (float)(test_band.size() - 1 - idx) / (float)(test_band.size() - 1),
                            0.3f);
            }
            vis_->visualizeSpheres(marker_name + "_bubbles", test_band, colors, id, bubble_sizes);
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

#if ENABLE_BAND_DEBUGGING
    std::cout << "Projecting out of collision" << std::endl;
    vis_->visualizePoints("___point_to_project", {location}, Visualizer::Yellow(0.3f), 1, 0.05);
    vis_->visualizePoints("___point_to_project", {location}, Visualizer::Yellow(0.3f), 1, 0.05);
    vis_->visualizePoints("___point_to_project", {location}, Visualizer::Yellow(0.3f), 1, 0.05);
    arc_helpers::Sleep(0.01);
#endif
    const auto post_collision_project = sdf_.ProjectOutOfCollisionToMinimumDistance3d(location, min_distance_to_obstacle_);
#if ENABLE_BAND_DEBUGGING
    std::cout << "Projecting to valid volume" << std::endl;
#endif
    const auto post_boundary_project = sdf_.ProjectIntoValidVolumeToMinimumDistance3d(post_collision_project, min_distance_to_obstacle_);

#if ENABLE_BAND_DEBUGGING
    std::cout << std::setprecision(20)
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

            const auto starting_distance_to_obstacles = sdf_.EstimateDistance3d(location);
            const auto starting_distance_to_boundary = sdf_.DistanceToBoundary3d(location);

            std::cerr << std::setprecision(p) << "Starting dist to obstacle: " << PrettyPrint::PrettyPrint(starting_distance_to_obstacles, true, " ") << std::endl;
            std::cerr << std::setprecision(p) << "Starting dist to boundary: " << PrettyPrint::PrettyPrint(starting_distance_to_boundary, true, " ") << std::endl;

            std::cerr << std::setprecision(p) << "Final dist to obstacle:    " << PrettyPrint::PrettyPrint(distance_to_obstacles, true, " ") << std::endl;
            std::cerr << std::setprecision(p) << "Final dist to boundary:    " << PrettyPrint::PrettyPrint(distance_to_boundary, true, " ") << std::endl;

            std::cerr << std::setprecision(p) << "collision - location:      " << (post_collision_project - location).norm() << std::endl;
            std::cerr << std::setprecision(p) << "boundary - collision:      " << (post_boundary_project - post_collision_project).norm() << std::endl;

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
#if ENABLE_BAND_DEBUGGING
    vis_->visualizePoints("get_bubble_size_test_location", {location}, Visualizer::Orange(), 1, 0.005);
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
    return bandIsValid(band_);
}

bool QuinlanRubberBand::bandIsValid(const EigenHelpers::VectorVector3d& test_band) const
{
#if ENABLE_BAND_DEBUGGING
    if (test_band.size() < 2)
    {
        return false;
    }

    for (size_t node_idx = 0; node_idx < test_band.size() - 1; ++node_idx)
    {
        const auto& curr = test_band[node_idx];
        const auto& next = test_band[node_idx + 1];
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
    if (getBubbleSize(test_band.back()) < min_distance_to_obstacle_)
    {
        std::cerr << "Problem at last node: "
                  << "Bubble size:        " << getBubbleSize(test_band.back()) << std::endl;
        return false;
    }
#endif
    return true;
}

bool QuinlanRubberBand::bandIsValidWithVisualization() const
{
    return bandIsValidWithVisualization(band_);
}

bool QuinlanRubberBand::bandIsValidWithVisualization(const EigenHelpers::VectorVector3d& test_band) const
{
#if ENABLE_BAND_DEBUGGING
    if (!bandIsValid(test_band))
    {
        visualizeWithBubbles(test_band, "quinlan_band_something_is_invalid", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        printBandData(test_band);
        return bandIsValid(test_band);
    }
#endif
    return true;
}

/**
  * Interpolates bewteen the end of point_buffer and target, but does not add target to the buffer
  */

void QuinlanRubberBand::interpolateBetweenPoints(
        EigenHelpers::VectorVector3d& point_buffer,
        const Eigen::Vector3d& target) const
{
#if ENABLE_BAND_DEBUGGING
    if (point_buffer.size() >= 2)
    {
        assert(bandIsValidWithVisualization(point_buffer));
    }
    visualizeWithBubbles(point_buffer, "start_of_interpolateBetweenPoints", Visualizer::Blue(), Visualizer::Cyan(), 1, true);
    vis_->visualizePoints("start_of_interpolateBetweenPoints_target_point", {target}, Visualizer::Orange(), 1, 0.01);
    vis_->visualizePoints("start_of_interpolateBetweenPoints_target_point", {target}, Visualizer::Orange(), 1, 0.01);
    vis_->visualizePoints("start_of_interpolateBetweenPoints_target_point", {target}, Visualizer::Orange(), 1, 0.01);

    std::cout << PrettyPrint::PrettyPrint(point_buffer, true, "\n") << std::endl << "Start of interpolateBetweenPoints: " << std::flush;
    int tmp;
    std::cin >> tmp;
    visualizeWithBubbles(point_buffer, "start_of_interpolateBetweenPoints", Visualizer::Blue(), Visualizer::Cyan(), 1, true);
    vis_->visualizePoints("start_of_interpolateBetweenPoints_target_point", {target}, Visualizer::Orange(), 1, 0.01);
    vis_->visualizePoints("start_of_interpolateBetweenPoints_target_point", {target}, Visualizer::Orange(), 1, 0.01);
    vis_->visualizePoints("start_of_interpolateBetweenPoints_target_point", {target}, Visualizer::Orange(), 1, 0.01);
    arc_helpers::Sleep(0.1);
#endif


    const double target_bubble_size = getBubbleSize(target);

    // TODO: verify that this cannot get stuck in an infinite loop
    int outer_iteration_counter = 0;

    // Check if the bubbles for 2 adjacent nodes overlap with some minimum distance to spare
    double curr_bubble_size = getBubbleSize(point_buffer.back());

    double distance_to_end = (target - point_buffer.back()).norm();
    while (!sufficientOverlap(curr_bubble_size, target_bubble_size, distance_to_end))
    {
#if ENABLE_BAND_DEBUGGING
        std::cout << "Start of interpolateBetweenPoints outer loop" << std::endl;
#endif

        const Eigen::Vector3d curr = point_buffer.back();

        // Find a position between point_buffer.back() and next_node with sufficient bubble overlap
        // TODO: verify that this cannot get stuck in an infinite loop
        int inner_iteration_counter = 0;
        double interpolation_ratio = 0.5;

        Eigen::Vector3d interpolated_point = EigenHelpers::Interpolate(curr, target, interpolation_ratio);
        Eigen::Vector3d test_point = projectToValidBubble(interpolated_point);
        double test_point_bubble_size = getBubbleSize(test_point);
        double distance_between_curr_and_test_point = (curr - test_point).norm();

#if ENABLE_BAND_DEBUGGING
        assert(sdf_.CheckInBounds3d(curr));
        assert(sdf_.CheckInBounds3d(target));
        assert(sdf_.CheckInBounds3d(interpolated_point));
        assert(sdf_.CheckInBounds3d(test_point));
#endif
        ROS_WARN_STREAM_COND_NAMED(outer_iteration_counter == 20, "rubber_band", "Rubber band interpolation outer loop counter at " << outer_iteration_counter << ", probably stuck in an infinite loop");
#if ENABLE_BAND_DEBUGGING
        if (outer_iteration_counter >= 20)
        {
            int tmp;
            std::cout << "\n\n\n\n\n\n\nCurr: ";
            getBubbleSize(curr);
            std::cin >> tmp;
            std::cout << "\nInterp: ";
            const double interpolated_point_bubble_size = getBubbleSize(interpolated_point);
            std::cin >> tmp;
            std::cout << "\nTest: ";
            getBubbleSize(test_point);
            std::cin >> tmp;
            std::cout << "\nTarget: ";
            getBubbleSize(target);
            std::cin >> tmp;
            std::cout << std::endl;

            std::cerr << std::setprecision(12)
                      << "Curr:   " << curr.transpose() << std::endl
                      << "Target: " << target.transpose() << std::endl
                      << "Interp: " << interpolated_point.transpose() << std::endl
                      << "Test:   " << test_point.transpose() << std::endl
                      << std::endl;
            vis_->visualizePoints( "interpolate_outer_debugging_curr_point",    {curr},                Visualizer::Red(1.0f),     1, 0.005);
            vis_->visualizeSpheres("interpolate_outer_debugging_curr_sphere",   {curr},                Visualizer::Red(0.2f),     2, curr_bubble_size);
            vis_->visualizePoints( "interpolate_outer_debugging_interp_point",  {interpolated_point},  Visualizer::Green(1.0f),   1, 0.005);
            vis_->visualizeSpheres("interpolate_outer_debugging_interp_sphere", {interpolated_point},  Visualizer::Green(0.2f),   2, interpolated_point_bubble_size);
            vis_->visualizePoints( "interpolate_outer_debugging_test_point",    {test_point},          Visualizer::Cyan(1.0f),    1, 0.005);
            vis_->visualizeSpheres("interpolate_outer_debugging_test_sphere",   {test_point},          Visualizer::Cyan(0.2f),    2, test_point_bubble_size);
            vis_->visualizePoints( "interpolate_outer_debugging_target_point",  {target},              Visualizer::Blue(1.0f),    1, 0.005);
            vis_->visualizeSpheres("interpolate_outer_debugging_target_sphere", {target},              Visualizer::Blue(0.2f),    2, target_bubble_size);

            // delay some, doesn't actually do anything as nothing is on the "null" namespace
            vis_->deleteObjects("null", 1, 10);
            std::cin >> tmp;
        }
#endif

        while (!sufficientOverlap(curr_bubble_size, test_point_bubble_size, distance_between_curr_and_test_point))
        {
            ROS_WARN_STREAM_COND_NAMED(inner_iteration_counter == 5, "rubber_band", "Rubber band interpolation inner loop counter at " << inner_iteration_counter << ", probably stuck in an infinite loop");
#if ENABLE_BAND_DEBUGGING
            std::cout << "Start of interpolateBetweenPoints inner loop" << std::endl;
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

                vis_->visualizePoints( "interpolate_inner_debugging_curr_point",    {curr},                Visualizer::Red(1.0f),     1, 0.005);
                vis_->visualizeSpheres("interpolate_inner_debugging_curr_sphere",   {curr},                Visualizer::Red(0.2f),     2, curr_bubble_size);
                vis_->visualizePoints( "interpolate_inner_debugging_interp_point",  {interpolated_point},  Visualizer::Green(1.0f),   1, 0.005);
                vis_->visualizeSpheres("interpolate_inner_debugging_interp_sphere", {interpolated_point},  Visualizer::Green(0.2f),   2, interpolated_point_bubble_size);
                vis_->visualizePoints( "interpolate_inner_debugging_test_point",    {test_point},          Visualizer::Cyan(1.0f),    1, 0.005);
                vis_->visualizeSpheres("interpolate_inner_debugging_test_sphere",   {test_point},          Visualizer::Cyan(0.2f),    2, test_point_bubble_size);
                vis_->visualizePoints( "interpolate_inner_debugging_target_point",  {target},              Visualizer::Blue(1.0f),    1, 0.005);
                vis_->visualizeSpheres("interpolate_inner_debugging_target_sphere", {target},              Visualizer::Blue(0.2f),    2, target_bubble_size);

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

                // delay some, doesn't actually do anything as nothing is on the "null" namespace
                vis_->deleteObjects("null", 1, 10);
                std::cin >> tmp;
            }
#endif

            interpolation_ratio *= 0.5;
            interpolated_point = EigenHelpers::Interpolate(curr, target, interpolation_ratio);
            test_point = projectToValidBubble(interpolated_point);
            test_point_bubble_size = getBubbleSize(test_point);
            distance_between_curr_and_test_point = (curr - test_point).norm();

            ++inner_iteration_counter;

#if ENABLE_BAND_DEBUGGING
            std::cout << "End of interpolateBetweenPoints outer loop" << std::endl;
#endif
        }
        // The bubbles now overlap sufficiently, so accept this point and record the new values
        point_buffer.push_back(test_point);
        curr_bubble_size = test_point_bubble_size;
        distance_to_end = (target - test_point).norm();

#if ENABLE_BAND_DEBUGGING
        assert(bandIsValidWithVisualization(point_buffer));
        std::cout << "End of interpolateBetweenPoints outer loop" << std::endl;
#endif

        ++outer_iteration_counter;
    }

#if ENABLE_BAND_DEBUGGING
    point_buffer.push_back(target);
    visualizeWithBubbles(point_buffer, "end_of_interpolate_between_points", Visualizer::Black(), Visualizer::Cyan(), 1, true);
    assert(bandIsValidWithVisualization(point_buffer));
    point_buffer.erase(point_buffer.end() - 1);
    std::cout << "End of interpolateBetweenPoints" << std::endl;
#endif
}

/**
 * @brief QuinlanRubberBand::interpolateBandPoints
 * Re-interpolates the entire band, not to be used for just 1 segment
 */
void QuinlanRubberBand::interpolateBandPoints()
{
#if ENABLE_BAND_DEBUGGING
    assert(band_.size() >= 2);
    for (size_t idx = 0; idx < band_.size(); ++idx)
    {
        if (getBubbleSize(band_[idx]) < min_distance_to_obstacle_)
        {
            std::cerr << "idx: " << idx << " point: " << band_[idx].transpose() << " size: " << getBubbleSize(band_[idx]) << std::endl;
        }

        assert(getBubbleSize(band_[idx]) >= min_distance_to_obstacle_);
    }

    visualizeWithBubbles("start_of_interpolate_band_points", Visualizer::Blue(), Visualizer::Cyan(), 1, true);

    std::cout << PrettyPrint::PrettyPrint(band_, true, "\n") << std::endl << "Start of interpolateBandPoints: "  << std::flush;
//    int tmp;
//    std::cin >> tmp;

#endif

    EigenHelpers::VectorVector3d new_band(1, band_.front());
    for (size_t idx = 0; idx + 1 < band_.size(); ++idx)
    {
#if ENABLE_BAND_DEBUGGING
        std::cout << "Start of interpolateBandPointsLoop idx " << idx << std::endl;
#endif

        const auto& next_node = band_[idx + 1];
        interpolateBetweenPoints(new_band, next_node);
        new_band.push_back(next_node);
#if ENABLE_BAND_DEBUGGING
        assert(bandIsValidWithVisualization(new_band));
        std::cout << "End of interpolateBandPointsLoop idx " << idx << std::endl;
#endif
    }

    band_ = new_band;

    assert(bandIsValidWithVisualization());
}

void QuinlanRubberBand::removeExtraBandPoints(const bool verbose)
{
    assert(bandIsValidWithVisualization());

#if ENABLE_BAND_DEBUGGING
    std::cout << "Start of removeExtraBandPoints\n";
    visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
    printBandData(band_);
#endif

    // The start can't be removed, so push that point on immediately
    EigenHelpers::VectorVector3d forward_pass;
    forward_pass.reserve(band_.size());
    forward_pass.push_back(band_.front());
#if ENABLE_BAND_DEBUGGING
    vis_->visualizePoints( "remove_extra_test_points_kept_points",  {forward_pass.back()}, Visualizer::Cyan(1.0f), (int32_t)forward_pass.size(), 0.002);
    vis_->visualizeSpheres("remove_extra_test_points_kept_spheres", {forward_pass.back()}, Visualizer::Cyan(0.2f), (int32_t)forward_pass.size(), getBubbleSize(forward_pass.back()));
#endif

    for (size_t curr_idx = 1; curr_idx + 1 < band_.size(); ++curr_idx)
    {
#if ENABLE_BAND_DEBUGGING
        std::cout << "Start of removeExtraBandPoints loop, idx: " << curr_idx << std::endl;
#endif

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
#if ENABLE_BAND_DEBUGGING
            std::cout << "Removing point as it is wholey contained, idx: " << curr_idx << std::endl;
#endif
            continue;
        }

        // Discard this point if prev overlaps next by enough
        if (sufficientOverlap(prev_bubble_size, next_bubble_size, prev_next_dist * node_removal_overlap_factor_))
        {
#if ENABLE_BAND_DEBUGGING
            std::cout << "Removing point as it has enough overlap, idx: " << curr_idx << std::endl;
#endif
            continue;
        }

        // Discard this point if it is too close to the previous, or too close to the next
        if (prev.isApprox(curr) || next.isApprox(curr))
        {
#if ENABLE_BAND_DEBUGGING
            std::cout << "Removing point as it is really really really close to and adjacent one, idx: " << curr_idx << std::endl;
#endif
            continue;
        }

        // Only keep points if they do not backtrack
        const double angle_defined_by_points = EigenHelpers::AngleDefinedByPoints(prev, curr, next);
        assert(angle_defined_by_points >= 0.0);
        const bool band_backtracks = angle_defined_by_points < backtrack_threshold_;

        if (band_backtracks)
        {
#if ENABLE_BAND_DEBUGGING
            std::cout << "Removing point as it is a backtrack, idx: " << curr_idx << std::endl;
#endif
            continue;
        }

#if ENABLE_BAND_DEBUGGING
        vis_->visualizePoints( "remove_extra_test_prev", {prev}, Visualizer::Red(1.0f),   1, 0.002);
        vis_->visualizeSpheres("remove_extra_test_prev", {prev}, Visualizer::Red(0.2f),   2, prev_bubble_size);
        vis_->visualizePoints( "remove_extra_test_curr", {curr}, Visualizer::Green(1.0f), 1, 0.002);
        vis_->visualizeSpheres("remove_extra_test_curr", {curr}, Visualizer::Green(0.2f), 2, curr_bubble_size);
        vis_->visualizePoints( "remove_extra_test_next", {next}, Visualizer::Blue(1.0f),  1, 0.002);
        vis_->visualizeSpheres("remove_extra_test_next", {next}, Visualizer::Blue(0.2f),  2, next_bubble_size);
        std::this_thread::sleep_for(std::chrono::duration<double>(0.01));

        std::cout << "prev bubble size: " << prev_bubble_size << std::endl;
        std::cout << "curr bubble size: " << curr_bubble_size << std::endl;
        std::cout << "next bubble size: " << next_bubble_size << std::endl;
        std::cout << "prev-curr dist:   " << prev_curr_dist << std::endl;
        std::cout << "curr-next dist:   " << curr_next_dist << std::endl;
        std::cout << "prev-next dist:   " << prev_next_dist << std::endl;
#endif
        // If no item said we should delete this item, then keep it
        forward_pass.push_back(curr);
#if ENABLE_BAND_DEBUGGING
        vis_->visualizePoints( "remove_extra_test_points_kept_points",  {forward_pass.back()}, Visualizer::Cyan(1.0f), (int32_t)forward_pass.size(), 0.002);
        vis_->visualizeSpheres("remove_extra_test_points_kept_spheres", {forward_pass.back()}, Visualizer::Cyan(0.2f), (int32_t)forward_pass.size(), getBubbleSize(forward_pass.back()));

        std::cout << "End of removeExtraBandPoints loop"  << std::endl;
#endif
    }
    forward_pass.push_back(band_.back());
#if ENABLE_BAND_DEBUGGING
    vis_->visualizePoints( "remove_extra_test_points_kept_points",  {forward_pass.back()}, Visualizer::Cyan(1.0f), (int32_t)forward_pass.size(), 0.002);
    vis_->visualizeSpheres("remove_extra_test_points_kept_spheres", {forward_pass.back()}, Visualizer::Cyan(0.2f), (int32_t)forward_pass.size(), getBubbleSize(forward_pass.back()));
#endif

    band_ = forward_pass;
    visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, verbose);

#if ENABLE_BAND_DEBUGGING
    std::cout << "End of removeExtraBandPoints\n";
    visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
    printBandData(band_);
    assert(bandIsValidWithVisualization());
#endif
}

void QuinlanRubberBand::smoothBandPoints(const bool verbose)
{
    assert(bandIsValidWithVisualization());

    for (size_t smoothing_iter = 0; smoothing_iter < smoothing_iterations_; ++smoothing_iter)
    {
#if ENABLE_BAND_DEBUGGING
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        std::cerr << "Start of loop smoothBandPoints\n";
        printBandData(band_);
#endif

        // The start doesn't move, so push that point on immediately
        EigenHelpers::VectorVector3d next_band;
        next_band.reserve(band_.size());
        next_band.push_back(band_.front());

        for (size_t curr_idx = 1; curr_idx + 1 < band_.size(); ++ curr_idx)
        {
#if ENABLE_BAND_DEBUGGING
            std::cout << "Start of smoothBandPointsInnerLoop: band idx: " << curr_idx << std::endl;
#endif

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

#if ENABLE_BAND_DEBUGGING
            if (projected_bubble_size < min_distance_to_obstacle_)
            {
                const Eigen::Vector3d projected_testing = projectToValidBubble(prime);
                const double projected_bubble_size = getBubbleSize(projected_testing);
                std::cout << std::setprecision(12)
                          << "Projected: " << projected_testing.transpose() << std::endl
                          << "Proj bubble size: " << projected_bubble_size << std::endl;
            }
            const double prime_bubble_size = getBubbleSize(prime);

            vis_->visualizePoints( "smoothing_test_prev",      {prev},      Visualizer::Red(1.0f),     1, 0.002);
            vis_->visualizeSpheres("smoothing_test_prev",      {prev},      Visualizer::Red(0.2f),     2, prev_bubble_size);
            vis_->visualizePoints( "smoothing_test_curr",      {curr},      Visualizer::Green(1.0f),   1, 0.002);
            vis_->visualizeSpheres("smoothing_test_curr",      {curr},      Visualizer::Green(0.2f),   2, curr_bubble_size);
            vis_->visualizePoints( "smoothing_test_prime",     {prime},     Visualizer::Cyan(1.0f),    1, 0.002);
            vis_->visualizeSpheres("smoothing_test_prime",     {prime},     Visualizer::Cyan(0.2f),    2, prime_bubble_size);
            vis_->visualizePoints( "smoothing_test_projected", {projected}, Visualizer::Magenta(1.0f), 1, 0.002);
            vis_->visualizeSpheres("smoothing_test_projected", {projected}, Visualizer::Magenta(0.2f), 2, projected_bubble_size);
            vis_->visualizePoints( "smoothing_test_next",      {next},      Visualizer::Blue(1.0f),    1, 0.002);
            vis_->visualizeSpheres("smoothing_test_next",      {next},      Visualizer::Blue(0.2f),    2, next_bubble_size);
            arc_helpers::Sleep(0.01);

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

#if ENABLE_BAND_DEBUGGING
            std::cout << "End of smoothBandPointsInnerLoop: band idx: " << curr_idx << std::endl;
#endif
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
#if ENABLE_BAND_DEBUGGING
        std::cout << "\n\n\nEnd of smoothing loop, iteration: " << smoothing_iter << std::endl;
        visualizeWithBubbles("quinlan_band_test", Visualizer::Black(), Visualizer::Cyan(), 1, true);
        printBandData(band_);
        assert(bandIsValidWithVisualization());
#endif

        removeExtraBandPoints(verbose);
    }

#if ENABLE_BAND_DEBUGGING
    printBandData(band_);
    assert(bandIsValidWithVisualization());
#endif
}



void QuinlanRubberBand::printBandData(const EigenHelpers::VectorVector3d& test_band) const
{
#if !ENABLE_BAND_DEBUGGING
    return;
#endif
    const Eigen::Vector3d min = sdf_.GetOriginTransform().translation();
    const Eigen::Vector3d max = min + Eigen::Vector3d(sdf_.GetXSize(), sdf_.GetYSize(), sdf_.GetZSize());
    std::cout << "SDF limits: x, y, z\n"
              << "Max:            " << max.transpose() << std::endl
              << "Min:            " << min.transpose() << std::endl
              << "Band limits: Min dist to obstacle: " << min_distance_to_obstacle_ << "   Min overlap distance: " << min_overlap_distance_ << std::endl;

    std::cout << "                         Point                    ,    bubble size   ,     overlap    ,   Angles:\n";

    Eigen::MatrixXd data = Eigen::MatrixXd::Zero(test_band.size(), 6) * NAN;

    for (size_t idx = 0; idx < test_band.size(); ++idx)
    {
        data.block<1, 3>(idx, 0) = test_band[idx].transpose();
        data(idx, 3) = getBubbleSize(test_band[idx]);
        if (idx > 0)
        {
            const double prev_bubble_size = getBubbleSize(test_band[idx - 1]);
            const double curr_bubble_size = getBubbleSize(test_band[idx]);
            const double distance_between_prev_and_curr = (test_band[idx] - test_band[idx-1]).norm();
            data(idx, 4) = (prev_bubble_size + curr_bubble_size) - distance_between_prev_and_curr;
        }
        if (idx > 0 && idx + 1 < test_band.size())
        {
            data(idx, 5) = EigenHelpers::AngleDefinedByPoints(test_band[idx - 1], test_band[idx], test_band[idx + 1]);
        }
    }
    std::cout << std::setprecision(12) << data << std::endl;
    std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
}


uint64_t QuinlanRubberBand::serialize(std::vector<uint8_t>& buffer) const
{
    return arc_utilities::SerializeVector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                                                    band_, buffer, &arc_utilities::SerializeEigenType<Eigen::Vector3d>);
}

uint64_t QuinlanRubberBand::deserializeIntoSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
{
    const auto deserialized_results = arc_utilities::DeserializeVector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                buffer, current, &arc_utilities::DeserializeEigenType<Eigen::Vector3d>);
    band_ = deserialized_results.first;
    return deserialized_results.second;
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

        const std::string file_name_suffix = arc_helpers::GetCurrentTimeAsString();
        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix + ".compressed";
        const std::string full_path = log_folder.GetImmutable() + file_name;
        ROS_DEBUG_STREAM("Saving band to " << full_path);

        std::vector<uint8_t> buffer;
        serialize(buffer);
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
        deserializeIntoSelf(buffer, 0);
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to load stored band: "  <<  e.what());
    }

    visualizeWithBubbles("band_post_load", Visualizer::Blue(), Visualizer::Cyan(), 1, true);
}

bool QuinlanRubberBand::useStoredBand() const
{
    return ROSHelpers::GetParamRequired<bool>(ph_, "use_stored_band", __func__).GetImmutable();
}
