#include "smmap/prm_helper.h"

#include <functional>
#include <ros/ros.h>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/shortcut_smoothing.hpp>

using namespace smmap;

PRMHelper::PRMHelper(
        const sdf_tools::SignedDistanceField& environment_sdf,
        const Visualizer& vis,
        std::mt19937_64& generator,
        const Eigen::Vector3d planning_world_lower_limits,
        const Eigen::Vector3d planning_world_upper_limits,
        const bool visualization_enabled,
        const size_t num_neighbours_to_connect,
        const size_t num_samples_to_try,
        const double edge_validity_stepsize)
    : planning_world_lower_limits_(planning_world_lower_limits)
    , planning_world_upper_limits_(planning_world_upper_limits)
    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)
    , environment_sdf_(environment_sdf)
    , vis_(vis)
    , visualization_enabled_globally_(visualization_enabled)
    , roadmap_initialized_(false)
    , num_neighbours_to_connect_(num_neighbours_to_connect)
    , num_samples_to_try_(num_samples_to_try)
    , num_samples_done_(0)
    , edge_validity_stepsize_(edge_validity_stepsize)
{}

void PRMHelper::initializeRoadmap()
{
    if (roadmap_initialized_)
    {
        ROS_ERROR_NAMED("prm_helper", "initializeRoadmap called on an already initialized roadmap, ignoring");
        return;
    }

    ROS_INFO_NAMED("prm_helper", "Starting PRM initialization");

    const auto termination_check_fn = [&] ()
    {
        return num_samples_done_ >= num_samples_to_try_;
    };
    const bool distance_is_symmetric = true;
    roadmap_ = simple_prm_planner::SimpleGeometricPrmPlanner::BuildRoadMap<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                std::bind(&PRMHelper::sampleState, this),
                &PRMHelper::distance,
                std::bind(&PRMHelper::validState, this, std::placeholders::_1),
                std::bind(&PRMHelper::validEdge, this, std::placeholders::_1, std::placeholders::_2),
                termination_check_fn,
                num_neighbours_to_connect_,
                distance_is_symmetric);

    auto connected_components = roadmap_.GetConnectedComponentsUndirected();
    while (connected_components.second != 1)
    {
        ROS_WARN_STREAM_NAMED("prm_helper", "PRM is not connected, num components: " << connected_components.second);
        ROS_WARN_STREAM_NAMED("prm_helper", "Extending PRM by " << num_samples_to_try_ << " samples");
        simple_prm_planner::SimpleGeometricPrmPlanner::ExtendRoadMap<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                    roadmap_,
                    std::bind(&PRMHelper::sampleState, this),
                    &PRMHelper::distance,
                    std::bind(&PRMHelper::validState, this, std::placeholders::_1),
                    std::bind(&PRMHelper::validEdge, this, std::placeholders::_1, std::placeholders::_2),
                    termination_check_fn,
                    num_neighbours_to_connect_,
                    distance_is_symmetric);
        connected_components = roadmap_.GetConnectedComponentsUndirected();
    }

    ROS_INFO_NAMED("prm_helper", "PRM Initialized");

    roadmap_initialized_ = true;
}

void PRMHelper::visualize(const bool visualization_enabled)
{
    if (visualization_enabled_globally_ && visualization_enabled)
    {
        EigenHelpers::VectorVector3d node_points;
        EigenHelpers::VectorVector3d line_start_points;
        EigenHelpers::VectorVector3d line_end_points;

        const auto& nodes = roadmap_.GetNodesImmutable();
        node_points.reserve(nodes.size());
        for (const auto& node : nodes)
        {
            node_points.push_back(node.GetValueImmutable());

            // Only add the "out" edges for each node, the in edges will be added by the other node
            const auto& out_edges = node.GetOutEdgesImmutable();
            for (const auto& edge : out_edges)
            {
                line_start_points.push_back(node.GetValueImmutable());
                line_end_points.push_back(roadmap_.GetNodeImmutable(edge.GetToIndex()).GetValueImmutable());
            }
        }

        vis_.visualizePoints(PRM_NODES_NS, node_points, Visualizer::Green(), 1, 0.005);
        vis_.visualizeLines(PRM_EDGES_NS, line_start_points, line_end_points, Visualizer::Green(), 1, 0.001);
    }
}

EigenHelpers::VectorVector3d PRMHelper::getRandomPath(const Eigen::Vector3d& start, const Eigen::Vector3d& goal)
{
    const bool distance_is_symmetric = true;
    const auto plan_result = simple_prm_planner::SimpleGeometricPrmPlanner::QueryPathAndAddNodesSingleStartSingleGoalRandomWalk<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>(
                start,
                goal,
                generator_,
                roadmap_,
                std::bind(&PRMHelper::validEdge, this, std::placeholders::_1, std::placeholders::_2),
                &PRMHelper::distance,
                num_neighbours_to_connect_,
                distance_is_symmetric);
    return plan_result.first;
}

//////// Passed using std::bind to various SimpleGeometricPrmPlanner functions /////////////////////////////////////////

Eigen::Vector3d PRMHelper::sampleState()
{
    const double x = EigenHelpers::Interpolate(planning_world_lower_limits_.x(), planning_world_upper_limits_.x(), uniform_unit_distribution_(generator_));
    const double y = EigenHelpers::Interpolate(planning_world_lower_limits_.y(), planning_world_upper_limits_.y(), uniform_unit_distribution_(generator_));
    const double z = EigenHelpers::Interpolate(planning_world_lower_limits_.z(), planning_world_upper_limits_.z(), uniform_unit_distribution_(generator_));
    num_samples_done_++;
    if (num_samples_done_ % 100 == 0)
    {
        ROS_INFO_STREAM_NAMED("prm_helper", "  " << num_samples_done_ << "'th state sampled");
    }
    return Eigen::Vector3d(x, y, z);
}

double PRMHelper::distance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2)
{
    return (p1 - p2).norm();
}

bool PRMHelper::validState(const Eigen::Vector3d& p) const
{
    return environment_sdf_.Get3d(p) > 0.0;
}

bool PRMHelper::validEdge(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) const
{
    EigenHelpers::VectorVector3d path;
    path.push_back(p1);
    path.push_back(p2);
    const auto interpolation_fn = [] (const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const double ratio)
    {
        return EigenHelpers::Interpolate(p1, p2, ratio);
    };
    const auto resampled_path =
            shortcut_smoothing::ResamplePath(path, edge_validity_stepsize_, &PRMHelper::distance, interpolation_fn);

    for (const auto& point : resampled_path)
    {
        if (!validState(point))
        {
            return false;
        }
    }

    return true;
}
