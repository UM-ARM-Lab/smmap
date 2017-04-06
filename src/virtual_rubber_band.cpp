#include "smmap/virtual_rubber_band.h"

#include <arc_utilities/shortcut_smoothing.hpp>

using namespace smmap;

#warning "Magic numbers in virtual band constructor"
VirtualRubberBand::VirtualRubberBand(
        const WorldState &current_world_state,
        std::shared_ptr<DijkstrasCoverageTask> task,
        const Visualizer& vis)
    : task_(task)
    , sdf_(task_->environment_sdf_)
    , vis_(vis)
    , max_integration_step_size_(sdf_.GetResolution() / 10.0)
    , max_distance_between_rubber_band_points_(task_->work_space_grid_.minStepDimension() / 2.0)
    , num_smoothing_ittrs_(20000)
    , min_object_radius_(0.04)
    , max_total_band_distance_(0.0)
{
//    max_gripper_distance_ = EigenHelpers::CalculateTotalDistance(virtual_rubber_band) + (double)(GetClothNumDivsY(nh_) - 1) * dijkstras_task->stretchingThreshold();
//    ROS_INFO_STREAM_NAMED("planner", "  -----   Max gripper distance: " << max_gripper_distance_ << " Num rubber band nodes: " << virtual_rubber_band.size());

    assert(task_->deformable_type_ == DeformableType::CLOTH);

    const auto& start_point = current_world_state.all_grippers_single_pose_.at(0).translation();
    const auto& end_point = current_world_state.all_grippers_single_pose_.at(1).translation();

    const size_t num_divs = (size_t)std::ceil((end_point - start_point ).norm() / max_distance_between_rubber_band_points_);

    for (size_t i = 0; i <= num_divs; ++i)
    {
        band_.push_back(EigenHelpers::Interpolate(start_point, end_point, (double)i / (double)num_divs));
        assert(sdf_.Get3d(band_.back()) > 0.0);
    }
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
            const Eigen::Vector3d delta = EigenHelpers::Interpolate(first_endpoint_translation, second_endpoint_translation, distance_to_first_endpoint / cummulative_distances_along_band.back()) / (double)num_integration_steps;
            band_[rubber_band_node_ind] = projectOutOfObstacle(current_pos + delta);
        }
        if (verbose)
        {
            visualize("post_forward_step", Visualizer::Green(), integration_step_ind);
        }

        // Next subdivide the band if needed
        for (size_t rubber_band_node_ind = 1; rubber_band_node_ind < band_.size(); ++rubber_band_node_ind)
        {
            const Eigen::Vector3d& prev = band_[rubber_band_node_ind - 1];
            const Eigen::Vector3d& curr = band_[rubber_band_node_ind];
            const double dist_between_nodes = (prev - curr).norm();

            assert(dist_between_nodes < 2 * max_distance_between_rubber_band_points_);
            if (dist_between_nodes > max_distance_between_rubber_band_points_)
            {
                band_.insert(band_.begin() + rubber_band_node_ind, projectOutOfObstacle(EigenHelpers::Interpolate(prev, curr, 0.5)));
            }
        }
        if (verbose)
        {
            visualize("post_subdivide_step", Visualizer::Magenta(), integration_step_ind);
        }
    }


    // Shortcut smoothing
    const auto sdf_collision_fn = [&] (const Eigen::Vector3d& location) { return sdf_.Get3d(location) < 0.0; };
    for (int smoothing_ittr = 0; smoothing_ittr < num_smoothing_ittrs_; ++smoothing_ittr)
    {
        std::uniform_int_distribution<size_t> distribution(0, band_.size() - 1);

        const size_t first_ind = distribution(generator_);
        const size_t second_ind = distribution(generator_);

        if (first_ind != second_ind && (band_[first_ind] - band_[second_ind]).squaredNorm() < min_object_radius_ * min_object_radius_)
        {
            band_ = shortcut_smoothing::ShortcutSmooth(band_, first_ind, second_ind, max_distance_between_rubber_band_points_, sdf_collision_fn);
        }

        if (verbose)
        {
            vis_.visualizeXYZTrajectory("post_shortcut_smoothing", band_, Visualizer::Red(), smoothing_ittr);
        }
    }

    return band_;
}

const EigenHelpers::VectorVector3d& VirtualRubberBand::getVectorRepresentation() const
{
    return band_;
}

bool VirtualRubberBand::isOverstreched() const
{
    return EigenHelpers::CalculateTotalDistance(band_) > max_total_band_distance_;
}

void VirtualRubberBand::visualize(
        const std::string& marker_name,
        const std_msgs::ColorRGBA& color,
        const int32_t id) const
{
    vis_.visualizeXYZTrajectory(marker_name, band_, color, id);
}





Eigen::Vector3d VirtualRubberBand::projectOutOfObstacle(Eigen::Vector3d vec) const
{
    for (float sdf_dist = sdf_.Get3d(vec); sdf_dist < 0; sdf_dist = sdf_.Get3d(vec))
    {
        const bool enable_edge_gradients = true;
        const std::vector<double> gradient = sdf_.GetGradient3d(vec, enable_edge_gradients);
        const Eigen::Vector3d grad_eigen = EigenHelpers::StdVectorDoubleToEigenVector3d(gradient);

        assert(grad_eigen.norm() > sdf_.GetResolution() / 4.0); // Sanity check
        vec += grad_eigen.normalized() * sdf_.GetResolution() / 10.0;
    }

    return vec;
}
