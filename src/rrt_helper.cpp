#include "smmap/rrt_helper.h"

#include <thread>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/first_order_deformation.h>
#include <arc_utilities/simple_dtw.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/zlib_helpers.hpp>

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;
using namespace Eigen;

#warning "THIS IS STUPID! FIX THIS!"
static RRTRobotRepresentation robot_joint_limits_lower_;
static RRTRobotRepresentation robot_joint_limits_upper_;
static RRTRobotRepresentation robot_joint_weights_;

#define GRIPPER_TRANSLATION_IS_APPROX_DIST 0.001

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper function for assertion testing
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline bool gripperPositionsAreApproximatelyEqual(
        const RRTGrippersRepresentation& c1,
        const RRTGrippersRepresentation& c2)
{
    const auto& c1_first_gripper     = c1.first.translation();
    const auto& c1_second_gripper    = c1.second.translation();
    const auto& c2_first_gripper     = c2.first.translation();
    const auto& c2_second_gripper    = c2.second.translation();

    bool is_equal = true;
    is_equal &= c1_first_gripper.isApprox(c2_first_gripper, GRIPPER_TRANSLATION_IS_APPROX_DIST);
    is_equal &= c1_second_gripper.isApprox(c2_second_gripper, GRIPPER_TRANSLATION_IS_APPROX_DIST);
    return is_equal;
}

static inline bool robotConfigurationsAreApproximatelyEqual(
        const RRTRobotRepresentation& r1,
        const RRTRobotRepresentation& r2)
{
    return r1.isApprox(r2);
}

static inline bool bandEndpointsMatchGripperPositions(
        const RubberBand& band,
        const RRTGrippersRepresentation& grippers)
{
    RRTGrippersRepresentation test_representation = grippers;
    test_representation.first.translation() = band.getVectorRepresentation().front();
    test_representation.second.translation() = band.getVectorRepresentation().back();
    return gripperPositionsAreApproximatelyEqual(grippers, test_representation);
}

static inline bool maxGrippersDistanceViolated(
        const Vector3d& gripper_a_pos,
        const Vector3d& gripper_b_pos,
        const double max_dist)
{
    return (gripper_a_pos - gripper_b_pos).norm() > max_dist;
}

static inline bool maxGrippersDistanceViolated(
        const RRTGrippersRepresentation& grippers,
        const double max_dist)
{
    const auto& gripper_a_pos = grippers.first.translation();
    const auto& gripper_b_pos = grippers.second.translation();
    return (gripper_a_pos - gripper_b_pos).norm() > max_dist;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTConfig functions                      /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RRTNode::RRTNode()
    : parent_index_(-1)
    , child_indices_(0)
    , initialized_(false)
    , blacklisted_(false)
{}

RRTNode::RRTNode(
        const RRTGrippersRepresentation& grippers_poses,
        const RRTRobotRepresentation& robot_configuration,
        const RubberBand::Ptr& band)
    : grippers_poses_(grippers_poses)
    , robot_configuration_(robot_configuration)
    , band_(band)
    , parent_index_(-1)
    , child_indices_(0)
    , initialized_(true)
    , blacklisted_(false)
{}

RRTNode::RRTNode(
        const RRTGrippersRepresentation& grippers_poses,
        const RRTRobotRepresentation& robot_configuration,
        const RubberBand::Ptr& band,
        const int64_t parent_index,
        const std::vector<int64_t>& child_indices)
    : grippers_poses_(grippers_poses)
    , robot_configuration_(robot_configuration)
    , band_(band)
    , parent_index_(parent_index)
    , child_indices_(child_indices)
    , initialized_(true)
    , blacklisted_(false)
{}

RRTNode::RRTNode(
        const RRTGrippersRepresentation& grippers_poses,
        const RRTRobotRepresentation& robot_configuration,
        const RubberBand::Ptr& band,
        const int64_t parent_index)
    : grippers_poses_(grippers_poses)
    , robot_configuration_(robot_configuration)
    , band_(band)
    , parent_index_(parent_index)
    , child_indices_(0)
    , initialized_(true)
    , blacklisted_(false)
{}

bool RRTNode::isInitialized() const
{
    return initialized_;
}

const RRTGrippersRepresentation& RRTNode::getGrippers() const
{
    return grippers_poses_;
}

const RRTRobotRepresentation& RRTNode::getRobotConfiguration() const
{
    return robot_configuration_;
}

const RubberBand::Ptr& RRTNode::getBand() const
{
    return band_;
}


int64_t RRTNode::getParentIndex() const
{
    return parent_index_;
}

void RRTNode::setParentIndex(const int64_t parent_index)
{
    parent_index_ = parent_index;
}


const std::vector<int64_t>& RRTNode::getChildIndices() const
{
    return child_indices_;
}

void RRTNode::clearChildIndicies()
{
    child_indices_.clear();
}

void RRTNode::addChildIndex(const int64_t child_index)
{
    for (size_t idx = 0; idx < child_indices_.size(); idx++)
    {
        if (child_indices_[idx] == child_index)
        {
            return;
        }
    }
    child_indices_.push_back(child_index);
}

void RRTNode::removeChildIndex(const int64_t child_index)
{
    std::vector<int64_t> new_child_indices;
    for (size_t idx = 0; idx < child_indices_.size(); idx++)
    {
        if (child_indices_[idx] != child_index)
        {
            new_child_indices.push_back(child_indices_[idx]);
        }
    }
    child_indices_ = new_child_indices;
}


bool RRTNode::isBlacklisted() const
{
    return blacklisted_;
}

void RRTNode::blacklist()
{
    blacklisted_ = true;
}


// Returned distance is the Euclidian distance of two grippers posistions
double RRTNode::distance(const RRTNode& other) const
{
    return RRTNode::distance(*this, other);
}

double RRTNode::distance(const RRTNode& c1, const RRTNode& c2)
{
    assert(false && "Not updated in a while");
#ifdef DTW_DISTANCE
    const auto distance_fn = [] (const Vector3d p1, const Vector3d p2)
    {
        return (p1 - p2).norm();
    };
    return simple_dtw::ComputeDTWDistance(c1.getBand().getVectorRepresentation(), c2.getBand().getVectorRepresentation(), distance_fn);
#endif
    return RRTNode::distance(c1.getGrippers(), c2.getGrippers());
}

// Note that this does only translational distance
double RRTNode::distanceSquared(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2)
{
    const auto& c1_first_gripper     = c1.first.translation();
    const auto& c1_second_gripper    = c1.second.translation();
    const auto& c2_first_gripper     = c2.first.translation();
    const auto& c2_second_gripper    = c2.second.translation();
    return (c1_first_gripper - c2_first_gripper).squaredNorm() +
            (c1_second_gripper - c2_second_gripper).squaredNorm();
}

double RRTNode::distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2)
{
    return std::sqrt(distanceSquared(c1, c2));
}

double RRTNode::distanceSquared(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2)
{
    const auto delta = r1 - r2;
    return (delta.cwiseProduct(robot_joint_weights_)).squaredNorm();
}

double RRTNode::distance(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2)
{
    return std::sqrt(distanceSquared(r1, r2));
}

// Only calculates the distance travelled by the grippers, not the entire band
double RRTNode::grippersPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());
    double path_distance = 0;
    for (size_t idx = start_index; idx < end_index; ++idx)
    {
        path_distance += RRTNode::distance(path[idx].getGrippers(), path[idx + 1].getGrippers());
    }
    return path_distance;
}

double RRTNode::robotPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());
    double path_distance = 0;
    for (size_t idx = start_index; idx < end_index; ++idx)
    {
        path_distance += RRTNode::distance(path[idx].getRobotConfiguration(), path[idx + 1].getRobotConfiguration());
    }
    return path_distance;
}


std::string RRTNode::print() const
{
    std::stringstream out;
    out << parent_index_ << "    "
//        << PrettyPrint::PrettyPrint(grippers_position_, true, " ") << "    "
        << robot_configuration_.transpose();
    return out.str();
}

bool RRTNode::operator==(const RRTNode& other) const
{
    if (!gripperPositionsAreApproximatelyEqual(grippers_poses_, other.grippers_poses_))
    {
        return false;
    }

    if (!robotConfigurationsAreApproximatelyEqual(robot_configuration_, other.robot_configuration_))
    {
        return false;
    }

    const auto& this_band_as_vector = band_->getVectorRepresentation();
    const auto& other_band_as_vector = other.band_->getVectorRepresentation();
    if (this_band_as_vector.size() != other_band_as_vector.size())
    {
        return false;
    }

    for (size_t idx = 0; idx < this_band_as_vector.size(); ++idx)
    {
        if (!this_band_as_vector[idx].isApprox(other_band_as_vector[idx]))
        {
            return false;
        }
    }

    if (parent_index_ != other.parent_index_)
    {
        return false;
    }

    if (child_indices_ != other.child_indices_)
    {
        return false;
    }

    if (initialized_ != other.initialized_)
    {
        return false;
    }

    return true;
}

uint64_t RRTNode::serialize(std::vector<uint8_t>& buffer) const
{
    const uint64_t starting_bytes = buffer.size();

    arc_utilities::SerializePair<Isometry3d, Isometry3d>(grippers_poses_, buffer, &arc_utilities::SerializeEigen<Isometry3d>, &arc_utilities::SerializeEigen<Isometry3d>);
    arc_utilities::SerializeEigen<double, Dynamic, 1>(robot_configuration_, buffer);
    band_->serialize(buffer);
    arc_utilities::SerializeFixedSizePOD<int64_t>(parent_index_, buffer);
    arc_utilities::SerializeVector<int64_t>(child_indices_, buffer, arc_utilities::SerializeFixedSizePOD<int64_t>);
    arc_utilities::SerializeFixedSizePOD<uint8_t>((uint8_t)initialized_, buffer);
    arc_utilities::SerializeFixedSizePOD<uint8_t>((uint8_t)blacklisted_, buffer);

    const uint64_t ending_bytes = buffer.size();

    // Verify the result
    const auto deserialized = Deserialize(buffer, starting_bytes, *band_);
    assert(ending_bytes - starting_bytes == deserialized.second);
    assert(*this == deserialized.first);

    return ending_bytes - starting_bytes;
}

uint64_t RRTNode::Serialize(const RRTNode& config, std::vector<uint8_t>& buffer)
{
    return config.serialize(buffer);
}

std::pair<RRTNode, uint64_t> RRTNode::Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current, const RubberBand& starting_band)
{
    assert(current < buffer.size());
    uint64_t current_position = current;

    // Deserialize the grippers poses
    const auto grippers_poses_deserialized = arc_utilities::DeserializePair<Isometry3d, Isometry3d>(
                buffer, current_position, &arc_utilities::DeserializeEigen<Isometry3d>, &arc_utilities::DeserializeEigen<Isometry3d>);
    current_position += grippers_poses_deserialized.second;

    // Deserialize the robot configuration
    const auto robot_configuration_deserialized = arc_utilities::DeserializeEigen<VectorXd>(buffer, current_position);
    current_position += robot_configuration_deserialized.second;

    // Deserialize the rubber band
    auto band = std::make_shared<RubberBand>(starting_band);
    current_position += band->deserializeIntoSelf(buffer, current_position);

    // Deserialize the parent index
    const auto parent_index_deserialized = arc_utilities::DeserializeFixedSizePOD<int64_t>(buffer, current_position);
    current_position += parent_index_deserialized.second;

    // Deserialize the child indices
    const auto child_indices_deserialized = arc_utilities::DeserializeVector<int64_t>(buffer, current_position, &arc_utilities::DeserializeFixedSizePOD<int64_t>);
    current_position += child_indices_deserialized.second;

    // Deserialize the initialized
    const auto initialized_deserialized = arc_utilities::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
    current_position += initialized_deserialized.second;

    // Deserialize the initialized
    const auto blacklisted_deserialized = arc_utilities::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
    current_position += blacklisted_deserialized.second;

    // Build the resulting node
    RRTNode deserialized(
                grippers_poses_deserialized.first,
                robot_configuration_deserialized.first,
                band,
                parent_index_deserialized.first,
                child_indices_deserialized.first);
    deserialized.initialized_ = (bool)initialized_deserialized.first;
    deserialized.blacklisted_ = (bool)blacklisted_deserialized.first;

    const uint64_t bytes_read = current_position - current;
    return std::make_pair(deserialized, bytes_read);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTHelper functions                      /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RRTHelper::RRTHelper(
        // Robot/environment related parameters
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const RobotInterface::Ptr robot,
        const sdf_tools::SignedDistanceField::ConstPtr environment_sdf,
        const PRMHelper::Ptr& prm_helper,
        const std::shared_ptr<std::mt19937_64>& generator,
        // Planning algorithm parameters
        const bool using_cbirrt_style_projection,
        const size_t forward_tree_extend_iterations,
        const size_t backward_tree_extend_iterations,
        const size_t kd_tree_grow_threshold,
        const bool use_brute_force_nn,
        const double goal_bias,
        // Smoothing parameters
        const int64_t max_shortcut_index_distance,
        const uint32_t max_smoothing_iterations,
        const uint32_t max_failed_smoothing_iterations,
        // Task defined parameters
        const Eigen::Isometry3d& task_aligned_frame,
        const Eigen::Vector3d& task_aligned_lower_limits,
        const Eigen::Vector3d& task_aligned_upper_limits,
        const double max_gripper_step_size,
        const double max_robot_dof_step_size,
        const double min_robot_dof_step_size,
        const double max_gripper_rotation,
        const double goal_reach_radius,
        const double gripper_min_distance_to_obstacles,
        const double homotopy_distance_penalty,
        // Visualization
        const smmap_utilities::Visualizer::Ptr vis,
        const bool visualization_enabled)
    : nh_(nh)
    , ph_(ph.getNamespace() + "/rrt")
    , robot_(robot)
    , environment_sdf_(environment_sdf)

    , prm_helper_(prm_helper)
    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)

    , task_aligned_frame_transform_(task_aligned_frame)
    , task_aligned_frame_inverse_transform_(task_aligned_frame_transform_.inverse())
    , task_aligned_lower_limits_(task_aligned_lower_limits)
    , task_aligned_upper_limits_(task_aligned_upper_limits)
    , max_gripper_step_size_(max_gripper_step_size)
    , max_robot_dof_step_size_(max_robot_dof_step_size)
    , min_robot_dof_step_size_(min_robot_dof_step_size)
    , max_gripper_rotation_(max_gripper_rotation)
    , goal_bias_(goal_bias)
    , goal_reach_radius_(goal_reach_radius)
    , homotopy_distance_penalty_(homotopy_distance_penalty)
    , gripper_min_distance_to_obstacles_(gripper_min_distance_to_obstacles)

    , using_cbirrt_style_projection_(using_cbirrt_style_projection)
    , forward_tree_extend_iterations_(forward_tree_extend_iterations)
    , backward_tree_extend_iterations_(backward_tree_extend_iterations)
    , use_brute_force_nn_(use_brute_force_nn)
    , kd_tree_grow_threshold_(kd_tree_grow_threshold)

    , max_shortcut_index_distance_(max_shortcut_index_distance)
    , max_smoothing_iterations_(max_smoothing_iterations)
    , max_failed_smoothing_iterations_(max_failed_smoothing_iterations)
    , uniform_shortcut_smoothing_int_distribution_(1, 4)

    , forward_nn_raw_data_(0)
    , backward_nn_raw_data_(0)
    , forward_nn_index_(flann::KDTreeSingleIndexParams(10, true))
    , backward_nn_index_(flann::KDTreeSingleIndexParams(10, true))
    , forward_next_idx_to_add_to_nn_dataset_(0)
    , backward_next_idx_to_add_to_nn_dataset_(0)

    , total_sampling_time_(NAN)
    , total_nearest_neighbour_index_building_time_(NAN)
    , total_nearest_neighbour_index_searching_time_(NAN)
    , total_nearest_neighbour_linear_searching_time_(NAN)
    , total_nearest_neighbour_time_(NAN)
    , total_projection_time_(NAN)
    , total_collision_check_time_(NAN)
    , total_band_forward_propogation_time_(NAN)
    , total_first_order_vis_propogation_time_(NAN)
    , total_everything_included_forward_propogation_time_(NAN)

    , vis_(vis)
    , visualization_enabled_globally_(visualization_enabled)
    , gripper_a_forward_tree_color_(Visualizer::Magenta())
    , gripper_b_forward_tree_color_(Visualizer::Red())
    , gripper_a_backward_tree_color_(Visualizer::Yellow())
    , gripper_b_backward_tree_color_(Visualizer::Cyan())
    , band_tree_color_(Visualizer::Blue())
{
    assert(task_aligned_lower_limits_.x() <= task_aligned_upper_limits_.x());
    assert(task_aligned_lower_limits_.y() <= task_aligned_upper_limits_.y());
    assert(task_aligned_lower_limits_.z() <= task_aligned_upper_limits_.z());
    assert(max_gripper_step_size_ > 0.0);
    assert(goal_reach_radius_ > 0.0);
    assert(homotopy_distance_penalty_ >= 0.0);
    assert(max_shortcut_index_distance_ > 0);
    assert(gripper_min_distance_to_obstacles_ > 0.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for external RRT planning class
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t RRTHelper::rebuildNNIndex(NNIndexType& index, std::vector<float>& nn_raw_data, const std::vector<RRTNode, RRTAllocator>& tree, const size_t new_data_start_idx)
{
    // These pointers are used to check if we need to rebuild the whole tree because the data moved,
    // or if we can just add points
    const float* initial_data_pointer = nn_raw_data.data();
    nn_raw_data.resize(total_dof_ * tree.size());
    const float* final_data_pointer = nn_raw_data.data();

    for (size_t idx = new_data_start_idx; idx < tree.size(); ++idx)
    {
        const RRTRobotRepresentation& robot_config = tree[idx].getRobotConfiguration();
        const VectorXf robot_config_float = robot_config.cast<float>();
        memcpy(&nn_raw_data[total_dof_ * idx], robot_config_float.data(), total_dof_ * sizeof(float));
    }

    // If the tree has already been initialized, and the raw data did not move in memory,
    // then we can just add the new points
    if (new_data_start_idx != 0 && (initial_data_pointer == final_data_pointer))
    {
        flann::Matrix<float> data(&nn_raw_data[total_dof_ * new_data_start_idx], tree.size() - new_data_start_idx, total_dof_);
        index.addPoints(data);
    }
    // Otherwise rebuild the whole tree
    else
    {
        flann::Matrix<float> data(nn_raw_data.data(), tree.size(), total_dof_);
        index.buildIndex(data);
    }

    return tree.size();
}

std::pair<int64_t, double> getNearest(const RRTRobotRepresentation& robot_config, const NNIndexType& index)
{
    std::pair<int64_t, double> nearest(-1, std::numeric_limits<double>::infinity());

    VectorXf robot_config_float = robot_config.cast<float>();
    flann::Matrix<float> query(robot_config_float.data(), 1, robot_config.size());

    const size_t knn = 1;
    std::vector<std::vector<size_t>> indices(query.rows, std::vector<size_t>(knn, -1));
    std::vector<std::vector<float>> dists(query.rows, std::vector<float>(knn, INFINITY));

    const float eps = 0.0;
    flann::SearchParams params(flann::flann_checks_t::FLANN_CHECKS_UNLIMITED, eps);
    index.knnSearch(query, indices, dists, knn, params);
    nearest.first = indices[0][0];
    nearest.second = dists[0][0];

    return nearest;
}

template <bool planning_for_whole_robot>
std::pair<int64_t, double> getNearest(const RRTNode& config, const std::vector<RRTNode, RRTAllocator>& tree, const size_t start_idx)
{
    std::pair<int64_t, double> nearest(-1, std::numeric_limits<double>::infinity());

    for (size_t idx = start_idx; idx < tree.size(); idx++)
    {
        const RRTNode& test_node = tree[idx];
        double distance = 0.0;
//        if (test_node.isBlacklisted())
//        {
//            distance += RRTHelper::NN_BLACKLIST_DISTANCE;
//        }

        if (!planning_for_whole_robot)
        {
            distance += RRTNode::distanceSquared(test_node.getGrippers(), config.getGrippers());
        }
        else
        {
            distance += RRTNode::distanceSquared(test_node.getRobotConfiguration(), config.getRobotConfiguration());
        }

        if (nearest.second > distance)
        {
            nearest.first = (int64_t)idx;
            nearest.second = distance;
        }
    }

    return nearest;
}

int64_t RRTHelper::nearestNeighbour(
        const bool use_forward_tree,
        const RRTNode& config)
{
    Stopwatch stopwatch;

    arc_helpers::DoNotOptimize(config);
    int64_t nn_idx = nearestNeighbour_internal(use_forward_tree, config);
    arc_helpers::DoNotOptimize(nn_idx);

    const double nn_time = stopwatch(READ);
    total_nearest_neighbour_time_ += nn_time;

    return nn_idx;
}

int64_t RRTHelper::nearestNeighbour_internal(
        const bool use_forward_tree,
        const RRTNode& config)
{
    std::vector<RRTNode, RRTAllocator>* tree = nullptr;
    NNIndexType* nn_index = nullptr;
    std::vector<float>* nn_raw_data = nullptr;
    size_t* manual_search_start_idx = nullptr;

    if (use_forward_tree)
    {
        tree = &forward_tree_;
        nn_index = &forward_nn_index_;
        nn_raw_data = &forward_nn_raw_data_;
        manual_search_start_idx = &forward_next_idx_to_add_to_nn_dataset_;
    }
    else
    {
        tree = &backward_tree_;
        nn_index = &backward_nn_index_;
        nn_raw_data = &backward_nn_raw_data_;
        manual_search_start_idx = &backward_next_idx_to_add_to_nn_dataset_;
    }

    // Check if we should rebuild the NN Index
    if (!use_brute_force_nn_ &&
        *manual_search_start_idx + kd_tree_grow_threshold_ <= tree->size())
    {
        Stopwatch stopwatch;
        arc_helpers::DoNotOptimize(*manual_search_start_idx);
        *manual_search_start_idx = rebuildNNIndex(*nn_index, *nn_raw_data, *tree, *manual_search_start_idx);
        arc_helpers::DoNotOptimize(*manual_search_start_idx);
        const double index_building_time = stopwatch(READ);
        total_nearest_neighbour_index_building_time_ += index_building_time;
    }

    // If we have a FLANN index to search
    std::pair<int64_t, double> nearest(-1, std::numeric_limits<double>::infinity());
    if (!use_brute_force_nn_ && *manual_search_start_idx > 0)
    {
        Stopwatch stopwatch;
        arc_helpers::DoNotOptimize(config);
        nearest = getNearest(config.getRobotConfiguration(), *nn_index);
        arc_helpers::DoNotOptimize(nearest);
        const double index_searching_time = stopwatch(READ);
        total_nearest_neighbour_index_searching_time_ += index_searching_time;

        // Confirm no meaningful mistaktes made
//        const auto linear_nearest = getNearest(config, planning_for_whole_robot_, *tree, 0, *manual_search_start_idx);
//        if (linear_nearest.first != nearest.first)
//        {
//            const auto& flann_near = (*tree)[nearest.first];
//            const auto& linear_near = (*tree)[linear_nearest.first];

//            std::cout << "Mismatch between FLANN and Linear Search: " << " Forward tree? " << use_forward_tree << std::endl
//                      << "FLANN:  " << " Idx: " << nearest.first        << " Dist: " << RRTNode::distanceSquared(config.getRobotConfiguration(), flann_near.getRobotConfiguration()) << " Flann Dist:  " << nearest.second << std::endl
//                      << "Linear: " << " Idx: " << linear_nearest.first << " Dist: " << RRTNode::distanceSquared(config.getRobotConfiguration(), linear_near.getRobotConfiguration()) << std::endl
//                      << "Target: " << config.print() << std::endl
//                      << "FLANN:  " << flann_near.print() << std::endl
//                      << "Linear: " << linear_near.print() << std::endl
//                      << "Delta:  " << (flann_near.getRobotConfiguration().first - linear_near.getRobotConfiguration().first).transpose() << "  " << (flann_near.getRobotConfiguration().second - linear_near.getRobotConfiguration().second).transpose() << std::endl
//                      << "Delta dist: " << RRTNode::distanceSquared(flann_near.getRobotConfiguration(), linear_near.getRobotConfiguration()) << std::endl
//                      << std::endl;

//            char c;
//            std::cin >> c;
//        }
    }

    // If we have data that isn't in the FLANN index
    if (*manual_search_start_idx < tree->size())
    {
        Stopwatch stopwatch;
        arc_helpers::DoNotOptimize(*manual_search_start_idx);

        // Move the if statement from inside of the distance function to outside via templates
        std::pair<int64_t, double> linear_nearest(-1, std::numeric_limits<double>::infinity());
        if (planning_for_whole_robot_)
        {
            linear_nearest = getNearest<true>(config, *tree, *manual_search_start_idx);
        }
        else
        {
            linear_nearest = getNearest<false>(config, *tree, *manual_search_start_idx);
        }
        if (linear_nearest.second < nearest.second)
        {
            nearest = linear_nearest;
        }
        arc_helpers::DoNotOptimize(nearest);
        const double linear_searching_time = stopwatch(READ);
        total_nearest_neighbour_linear_searching_time_ += linear_searching_time;
    }

    const int64_t nn_idx = nearest.first;
    assert(nn_idx >= 0);
    return nn_idx;
}



RRTNode RRTHelper::configSampling()
{
    Stopwatch stopwatch;
    arc_helpers::DoNotOptimize(generator_);
    RRTNode sample;

    if (!planning_for_whole_robot_)
    {
        sample = RRTNode(
                    posPairSampling_internal(),
                    starting_robot_configuration_,
                    std::make_shared<RubberBand>(*starting_band_));
    }
    else
    {
        sample = RRTNode(
                    starting_grippers_poses_,
                    robotConfigPairSampling_internal(),
                    std::make_shared<RubberBand>(*starting_band_));
    }

    arc_helpers::DoNotOptimize(sample.getGrippers());
    const double sampling_time = stopwatch(READ);
    total_sampling_time_ += sampling_time;
    return sample;
}

//RRTNode RRTHelper::prmBasedSampling_internal()
//{
//    const RRTGrippersRepresentation rand_grippers_sample = posPairSampling_internal();
//    const bool goal_is_target_config = gripperPositionsAreApproximatelyEqual(grippers_goal_position_, rand_grippers_sample);

//    // If we've sampled the goal, then keep sampling until the result is not visible to the blacklist
//    EigenHelpers::VectorVector3d band_path;
//    do
//    {
//        band_path = prm_helper_->getRandomPath(rand_grippers_sample.first, rand_grippers_sample.second);
//    }
//    while (goal_is_target_config && isBandFirstOrderVisibileToBlacklist(band_path));

//    RubberBand::Ptr band = std::make_shared<RubberBand>(*starting_band_);
//    band->setPointsWithoutSmoothing(band_path);
//    band->visualize(PRMHelper::PRM_RANDOM_PATH_NS, Visualizer::Orange(), Visualizer::Orange(), 1, visualization_enabled_globally_);

//    return RRTNode(rand_grippers_sample, starting_robot_configuration_, band);
//}

RRTGrippersRepresentation RRTHelper::posPairSampling_internal()
{
    Vector3d gripper_a_pos, gripper_b_pos;

    do
    {
        const double x1 = EigenHelpers::Interpolate(task_aligned_lower_limits_.x(), task_aligned_upper_limits_.x(), uniform_unit_distribution_(*generator_));
        const double y1 = EigenHelpers::Interpolate(task_aligned_lower_limits_.y(), task_aligned_upper_limits_.y(), uniform_unit_distribution_(*generator_));
        const double z1 = EigenHelpers::Interpolate(task_aligned_lower_limits_.z(), task_aligned_upper_limits_.z(), uniform_unit_distribution_(*generator_));
        gripper_a_pos = Vector3d(x1, y1, z1);
    }
    while (environment_sdf_->EstimateDistance3d(task_aligned_frame_transform_ * gripper_a_pos).first < gripper_min_distance_to_obstacles_);

    // We want to only sample within a radius max_grippers_distance_, and within the world extents; to do so
    // uniformly, we sample from an axis aligned box limited by R and the world extents, rejecting samples that lie
    // outside a radius max_grippers_distance_
    const double x2_min = std::max(task_aligned_lower_limits_.x(), gripper_a_pos.x() - max_grippers_distance_);
    const double x2_max = std::min(task_aligned_upper_limits_.x(), gripper_a_pos.x() + max_grippers_distance_);
    const double y2_min = std::max(task_aligned_lower_limits_.y(), gripper_a_pos.y() - max_grippers_distance_);
    const double y2_max = std::min(task_aligned_upper_limits_.y(), gripper_a_pos.y() + max_grippers_distance_);
    const double z2_min = std::max(task_aligned_lower_limits_.z(), gripper_a_pos.z() - max_grippers_distance_);
    const double z2_max = std::min(task_aligned_upper_limits_.z(), gripper_a_pos.z() + max_grippers_distance_);

    bool valid = false;
    do
    {
        const double x2 = EigenHelpers::Interpolate(x2_min, x2_max, uniform_unit_distribution_(*generator_));
        const double y2 = EigenHelpers::Interpolate(y2_min, y2_max, uniform_unit_distribution_(*generator_));
        const double z2 = EigenHelpers::Interpolate(z2_min, z2_max, uniform_unit_distribution_(*generator_));
        gripper_b_pos = Vector3d(x2, y2, z2);
        valid = !maxGrippersDistanceViolated(gripper_a_pos, gripper_b_pos, max_grippers_distance_);
    }
    while (!valid || environment_sdf_->EstimateDistance3d(task_aligned_frame_transform_ * gripper_b_pos).first < gripper_min_distance_to_obstacles_);

    RRTGrippersRepresentation rand_sample = grippers_goal_poses_;
    rand_sample.first.translation() = task_aligned_frame_transform_ * gripper_a_pos;
    rand_sample.second.translation() = task_aligned_frame_transform_ * gripper_b_pos;

    return rand_sample;
}

RRTRobotRepresentation RRTHelper::robotConfigPairSampling_internal()
{
    RRTRobotRepresentation rand_sample(total_dof_);
    for (ssize_t idx = 0; idx < total_dof_; ++idx)
    {
        rand_sample(idx) = EigenHelpers::Interpolate(robot_joint_limits_lower_(idx), robot_joint_limits_upper_(idx), uniform_unit_distribution_(*generator_));
    }
    return rand_sample;
}



bool RRTHelper::goalReached(const RRTNode& node)
{
    // Check if the grippers are close enough to the goal position
    if (RRTNode::distance(node.getGrippers(), grippers_goal_poses_) > goal_reach_radius_)
    {
        return false;
    }

    // Check if the grippers have been rotated too far
    if (planning_for_whole_robot_)
    {
        const double gripper_a_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.first.rotation(), node.getGrippers().first.rotation());
        const double gripper_b_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.second.rotation(), node.getGrippers().second.rotation());
        if (gripper_a_rotation_dist > max_gripper_rotation_ || gripper_b_rotation_dist > max_gripper_rotation_)
        {
            return false;
        }

        if (visualization_enabled_globally_)
        {
            vis_->visualizeLineStrip(RRT_GOAL_TESTING_NS, node.getBand()->getVectorRepresentation(), Visualizer::White(), 1, 0.01);
        }
    }

    // Only accept paths that are different from those on the blacklist
    if (isBandFirstOrderVisibileToBlacklist(*node.getBand()))
    {
        return false;
    }

    return true;
}

const std::pair<bool, RRTRobotRepresentation> RRTHelper::projectToValidConfig(
        const RRTRobotRepresentation& configuration,
        const AllGrippersSinglePose& poses,
        const bool project_to_rotation_bound,
        const bool project_to_translation_bound) const
{
    AllGrippersSinglePose projected_poses = poses;

    // Check if we rotated the grippers too much
    if (project_to_rotation_bound)
    {
        const double gripper_a_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.first.rotation(), poses[0].rotation());
        const double gripper_b_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.second.rotation(), poses[1].rotation());
        if (gripper_a_rotation_dist > max_gripper_rotation_)
        {
            const Eigen::Quaterniond q_start(starting_grippers_poses_.first.rotation());
            const Eigen::Quaterniond q_end(poses[0].rotation());
            const Eigen::Quaterniond q_projected = EigenHelpers::Interpolate(q_start, q_end, max_gripper_rotation_ / gripper_a_rotation_dist);
            projected_poses[0].linear() = q_projected.toRotationMatrix();
//            projected_poses[0].rotation() = q_projected.toRotationMatrix();
        }
        if (gripper_b_rotation_dist > max_gripper_rotation_)
        {
            const Eigen::Quaterniond q_start(starting_grippers_poses_.second.rotation());
            const Eigen::Quaterniond q_end(poses[1].rotation());
            const Eigen::Quaterniond q_projected = EigenHelpers::Interpolate(q_start, q_end, max_gripper_rotation_ / gripper_b_rotation_dist);
            projected_poses[1].linear() = q_projected.toRotationMatrix();
//            projected_poses[1].rotation() = q_projected.toRotationMatrix();
        }
    }

    // Check if the grippers moved outside of the planning arena
    if (project_to_translation_bound)
    {
        Eigen::Vector3d task_aligned_gripper_a_constrained_position = task_aligned_frame_inverse_transform_ * poses[0].translation();
        Eigen::Vector3d task_aligned_gripper_b_constrained_position = task_aligned_frame_inverse_transform_ * poses[1].translation();

        task_aligned_gripper_a_constrained_position = task_aligned_gripper_a_constrained_position.cwiseMax(task_aligned_lower_limits_);
        task_aligned_gripper_a_constrained_position = task_aligned_gripper_a_constrained_position.cwiseMin(task_aligned_upper_limits_);

        task_aligned_gripper_b_constrained_position = task_aligned_gripper_b_constrained_position.cwiseMax(task_aligned_lower_limits_);
        task_aligned_gripper_b_constrained_position = task_aligned_gripper_b_constrained_position.cwiseMin(task_aligned_upper_limits_);

        projected_poses[0].translation() = task_aligned_frame_transform_ * task_aligned_gripper_a_constrained_position;
        projected_poses[1].translation() = task_aligned_frame_transform_ * task_aligned_gripper_b_constrained_position;
    }

    const std::pair<bool, RRTRobotRepresentation> projected_configuration_result =
            robot_->getGeneralIkSolution(configuration, projected_poses);

    return projected_configuration_result;
}

size_t RRTHelper::forwardPropogationFunction(
        std::vector<RRTNode, RRTAllocator>& tree_to_extend,
        const int64_t& nearest_neighbor_idx,
        const RRTNode& target,
        const bool extend_band,
        const size_t max_projected_new_states,
        const bool visualization_enabled_locally)
{
    arc_helpers::DoNotOptimize(target.getParentIndex());
    Stopwatch function_wide_stopwatch;
    Stopwatch stopwatch;

    const size_t nodes_at_start_of_propogation = tree_to_extend.size();
    const RRTNode nearest_neighbour = tree_to_extend[nearest_neighbor_idx];

    const bool rubber_band_verbose = false && visualization_enabled_globally_ && visualization_enabled_locally;
    if (false && visualization_enabled_globally_ && visualization_enabled_locally)
    {
        vis_->visualizeCubes(
                    RRT_FORWARD_PROP_START_NS,
                    {nearest_neighbour.getGrippers().first.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_a_forward_tree_color_,
                    1);
        vis_->visualizeCubes(
                    RRT_FORWARD_PROP_START_NS,
                    {nearest_neighbour.getGrippers().second.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_b_forward_tree_color_,
                    5);

        nearest_neighbour.getBand()->visualize(
                    RRT_FORWARD_PROP_START_NS,
                    Visualizer::Green(),
                    Visualizer::Green(),
                    10,
                    true);

        vis_->visualizeCubes(
                    RRT_SAMPLE_NS,
                    {target.getGrippers().first.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_a_forward_tree_color_,
                    1);
        vis_->visualizeCubes(
                    RRT_SAMPLE_NS,
                    {target.getGrippers().second.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_b_forward_tree_color_,
                    5);
    }

    const size_t visualization_frequency = 100;

    const RRTGrippersRepresentation& starting_grippers_poses = nearest_neighbour.getGrippers();
    const RRTRobotRepresentation& starting_robot_configuration = nearest_neighbour.getRobotConfiguration();

    // Extract the target gripper pose and corresponding robot configuration
    const RRTGrippersRepresentation& target_grippers_position = target.getGrippers();
    RRTRobotRepresentation target_robot_configuration = target.getRobotConfiguration();

    if (planning_for_whole_robot_ && !using_cbirrt_style_projection_)
    {
        // Allocate space for potential children
        const double total_distance = RRTNode::distance(nearest_neighbour.getRobotConfiguration(), target_robot_configuration);
        const uint32_t max_total_steps = (uint32_t)ceil(total_distance / max_robot_dof_step_size_);
        tree_to_extend.reserve(tree_to_extend.size() + max_total_steps);

        int64_t parent_idx = nearest_neighbor_idx;
        uint32_t step_index = 0;
        while (step_index < max_total_steps)
        {
            // We could be updating the child indices of this node later, so take the value by non-const reference
            RRTNode& prev_node = tree_to_extend[parent_idx];
            const RubberBand::Ptr& prev_band = prev_node.getBand();

            // Interpolate in joint space to find the translation of the grippers
            const double ratio = std::min(1.0, (double)(step_index + 1) * max_robot_dof_step_size_ / total_distance);
            const RRTRobotRepresentation next_robot_configuration =
                    EigenHelpers::Interpolate(starting_robot_configuration, target_robot_configuration, ratio);

            const AllGrippersSinglePose next_grippers_poses_vector = robot_->getGrippersPoses(next_robot_configuration);
            const RRTGrippersRepresentation next_grippers_poses(next_grippers_poses_vector[0], next_grippers_poses_vector[1]);
            // Check gripper position and rotation constraints
            {
                // Check if we rotated the grippers too much
                {
                    const double gripper_a_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.first.rotation(), next_grippers_poses.first.rotation());
                    const double gripper_b_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.second.rotation(), next_grippers_poses.second.rotation());
                    if (gripper_a_rotation_dist > max_gripper_rotation_ || gripper_b_rotation_dist > max_gripper_rotation_)
                    {
                        break;
                    }
                }

                // If the grippers move outside of the planning arena then stop
                {
                    const RRTGrippersRepresentation task_frame_next_grippers_poses(
                                task_aligned_frame_inverse_transform_ * next_grippers_poses.first,
                                task_aligned_frame_inverse_transform_ * next_grippers_poses.second);

                    if ((task_frame_next_grippers_poses.first.translation().array() > task_aligned_upper_limits_.array()).any() ||
                        (task_frame_next_grippers_poses.first.translation().array() < task_aligned_lower_limits_.array()).any() ||
                        (task_frame_next_grippers_poses.second.translation().array() > task_aligned_upper_limits_.array()).any() ||
                        (task_frame_next_grippers_poses.second.translation().array() < task_aligned_lower_limits_.array()).any())
                    {
                        break;
                    }
                }
            }

            // Collision checking
            {
                stopwatch(RESET);
                arc_helpers::DoNotOptimize(next_robot_configuration);
                const bool in_collision = robot_->checkRobotCollision(next_robot_configuration);
                arc_helpers::DoNotOptimize(next_robot_configuration);
                const double collision_check_time = stopwatch(READ);
                total_collision_check_time_ += collision_check_time;
                if (in_collision)
                {
                    break;
                }
            }

            RubberBand::Ptr next_band = std::make_shared<RubberBand>(*prev_band);
            if (extend_band)
            {
                //// Duplicated from pure gripper vesion below version below ////

                stopwatch(RESET);
                arc_helpers::DoNotOptimize(rubber_band_verbose);
                // Forward simulate the rubber band to test this transition
                next_band->forwardPropagateRubberBandToEndpointTargets(
                            next_grippers_poses.first.translation(),
                            next_grippers_poses.second.translation(),
                            rubber_band_verbose);
                arc_helpers::DoNotOptimize(next_band->getVectorRepresentation());
                const double band_forward_propogation_time = stopwatch(READ);
                total_band_forward_propogation_time_ += band_forward_propogation_time;

                // If we are still able to get to the next target position after retrying,
                // then return however far we were able to get
                if (!bandEndpointsMatchGripperPositions(*next_band, next_grippers_poses))
                {
                    break;
                }

                // If the rubber band becomes overstretched, then return however far we were able to get
                if (next_band->isOverstretched())
                {
                    break;
                }
            }
            else
            {
                if (maxGrippersDistanceViolated(next_grippers_poses, max_grippers_distance_))
                {
                    break;
                }
            }

            // The new configuation is valid, add it to the tree
            const RRTNode next_node(next_grippers_poses, next_robot_configuration, next_band, parent_idx);
            tree_to_extend.push_back(next_node);
            const int64_t new_node_idx = (int64_t)tree_to_extend.size() - 1;
            prev_node.addChildIndex(new_node_idx);

            parent_idx = new_node_idx;
            ++step_index;
        }
    }
    else if (planning_for_whole_robot_ && using_cbirrt_style_projection_)
    {
        int64_t parent_idx = nearest_neighbor_idx;
        // Only accept at most max_new_states new states, if there are more, then we're probably stuck in some sort of bad configuration for IK
        // TODO: distinguish between smoothing and exploration for this check
        uint32_t step_index = 0;
        while (step_index < max_projected_new_states)
        {
            stopwatch(RESET);
            arc_helpers::DoNotOptimize(parent_idx);

            // We could be updating the child indices of this node later, so take the value by non-const reference
            const RRTNode& prev_node = tree_to_extend[parent_idx];
            const RubberBand::Ptr& prev_band = prev_node.getBand();
            const RRTRobotRepresentation& prev_robot_config = prev_node.getRobotConfiguration();

            const double prev_distance = RRTNode::distance(prev_robot_config, target_robot_configuration);
            const double ratio = std::min(1.0, max_gripper_step_size_ / prev_distance);
            const RRTRobotRepresentation next_robot_configuration_pre_projection =
                    EigenHelpers::Interpolate(prev_robot_config, target_robot_configuration, ratio);
            const AllGrippersSinglePose next_grippers_poses_pre_projection =
                    robot_->getGrippersPoses(next_robot_configuration_pre_projection);

            // Project and check the projection result for failure
            const bool project_to_rotation_bound = true;
            const bool project_to_translation_bound = false;
            const auto next_robot_configuration_projection_result =
                    projectToValidConfig(
                        next_robot_configuration_pre_projection,
                        next_grippers_poses_pre_projection,
                        project_to_rotation_bound,
                        project_to_translation_bound);
            const RRTRobotRepresentation& next_robot_configuration = next_robot_configuration_projection_result.second;
            arc_helpers::DoNotOptimize(next_robot_configuration);
            const double projection_time = stopwatch(READ);
            total_projection_time_ += projection_time;

            // Check if the projection failed
            if (!next_robot_configuration_projection_result.first)
            {
                break;
            }

            // Check if we made any progress
            const double current_distance = RRTNode::distance(target_robot_configuration, next_robot_configuration);
            if (current_distance > prev_distance - min_robot_dof_step_size_ &&
                current_distance < prev_distance + min_robot_dof_step_size_)
            {
                break;
            }

            // Check the distance to the previous previous value as well (if it exists)
            if (step_index >= 1)
            {
                const RRTRobotRepresentation& ancient_robot_config = (tree_to_extend.end() - 2)->getRobotConfiguration();
                if (RRTNode::distance(ancient_robot_config, next_robot_configuration) < min_robot_dof_step_size_)
                {
                    break;
                }
            }

            const AllGrippersSinglePose next_grippers_poses_vector = robot_->getGrippersPoses(next_robot_configuration);
            const RRTGrippersRepresentation next_grippers_poses(next_grippers_poses_vector[0], next_grippers_poses_vector[1]);
            // Check gripper position and rotation constraints if we did not project to them
            {
                // Check if we rotated the grippers too much
                if (!project_to_rotation_bound)
                {
                    const double gripper_a_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.first.rotation(), next_grippers_poses.first.rotation());
                    const double gripper_b_rotation_dist = EigenHelpers::Distance(starting_grippers_poses_.second.rotation(), next_grippers_poses.second.rotation());
                    if (gripper_a_rotation_dist > max_gripper_rotation_ || gripper_b_rotation_dist > max_gripper_rotation_)
                    {
                        break;
                    }
                }

                // If the grippers move outside of the planning arena
                if (!project_to_translation_bound)
                {
                    const RRTGrippersRepresentation task_frame_next_grippers_poses(
                                task_aligned_frame_inverse_transform_ * next_grippers_poses.first,
                                task_aligned_frame_inverse_transform_ * next_grippers_poses.second);

                    if ((task_frame_next_grippers_poses.first.translation().array() > task_aligned_upper_limits_.array()).any() ||
                        (task_frame_next_grippers_poses.first.translation().array() < task_aligned_lower_limits_.array()).any() ||
                        (task_frame_next_grippers_poses.second.translation().array() > task_aligned_upper_limits_.array()).any() ||
                        (task_frame_next_grippers_poses.second.translation().array() < task_aligned_lower_limits_.array()).any())
                    {
                        break;
                    }
                }
            }

            // Colision checking
            {
                stopwatch(RESET);
                arc_helpers::DoNotOptimize(next_robot_configuration);
                const bool in_collision = robot_->checkRobotCollision(next_robot_configuration);
                arc_helpers::DoNotOptimize(next_robot_configuration);
                const double collision_check_time = stopwatch(READ);
                total_collision_check_time_ += collision_check_time;
                if (in_collision)
                {
                    break;
                }
            }

            RubberBand::Ptr next_band = std::make_shared<RubberBand>(*prev_band);
            if (extend_band)
            {
                //// Duplicated from pure gripper vesion below version below ////

                stopwatch(RESET);
                arc_helpers::DoNotOptimize(rubber_band_verbose);
                // Forward simulate the rubber band to test this transition
                next_band->forwardPropagateRubberBandToEndpointTargets(
                            next_grippers_poses.first.translation(),
                            next_grippers_poses.second.translation(),
                            rubber_band_verbose);
                arc_helpers::DoNotOptimize(next_band->getVectorRepresentation());
                const double band_forward_propogation_time = stopwatch(READ);
                total_band_forward_propogation_time_ += band_forward_propogation_time;

                // If we are still able to get to the next target position after retrying,
                // then return however far we were able to get
                if (!bandEndpointsMatchGripperPositions(*next_band, next_grippers_poses))
                {
                    break;
                }

                // If the rubber band becomes overstretched, then return however far we were able to get
                if (next_band->isOverstretched())
                {
                    break;
                }
            }
            else
            {
                if (maxGrippersDistanceViolated(next_grippers_poses, max_grippers_distance_))
                {
                    break;
                }
            }

            // The new configuation is valid, add it to the tree
            const RRTNode next_node(next_grippers_poses, next_robot_configuration, next_band, parent_idx);
            tree_to_extend.push_back(next_node);
            const int64_t new_node_idx = (int64_t)tree_to_extend.size() - 1;
            tree_to_extend[parent_idx].addChildIndex(new_node_idx);

            parent_idx = new_node_idx;
            ++step_index;
        }
    }
    else
    {
        // Allocate space for potential children
        const double total_distance = RRTNode::distance(nearest_neighbour.getGrippers(), target.getGrippers());
        const uint32_t max_total_steps = (uint32_t)ceil(total_distance / max_gripper_step_size_);
        tree_to_extend.reserve(tree_to_extend.size() + max_total_steps);

        int64_t parent_idx = nearest_neighbor_idx;
        uint32_t step_index = 0;
        while (step_index < max_total_steps)
        {
            // We could be updating the child indices of this node later, so take the value by non-const reference
            RRTNode& prev_node = tree_to_extend[parent_idx];
            const RubberBand::Ptr& prev_band = prev_node.getBand();
            const RRTRobotRepresentation& prev_robot_config = prev_node.getRobotConfiguration();

            const double ratio = std::min(1.0, (double)(step_index + 1) * max_gripper_step_size_ / total_distance);
            const Vector3d gripper_a_interpolated = EigenHelpers::Interpolate(Vector3d(starting_grippers_poses.first.translation()), Vector3d(target_grippers_position.first.translation()), ratio);
            const Vector3d gripper_b_interpolated = EigenHelpers::Interpolate(Vector3d(starting_grippers_poses.second.translation()), Vector3d(target_grippers_position.second.translation()), ratio);
            RRTGrippersRepresentation next_grippers_poses = starting_grippers_poses;
            next_grippers_poses.first.translation() = gripper_a_interpolated;
            next_grippers_poses.second.translation() = gripper_b_interpolated;
            const RRTRobotRepresentation next_robot_configuration = prev_robot_config;

            // Collision checking
            {
                stopwatch(RESET);
                arc_helpers::DoNotOptimize(gripper_a_interpolated);
                bool in_collision = (gripper_a_interpolated - gripper_b_interpolated).norm() < gripper_min_distance_to_obstacles_;
                arc_helpers::DoNotOptimize(in_collision);
                const double collision_check_time_pt1 = stopwatch(READ);
                total_collision_check_time_ += collision_check_time_pt1;
                // If the grippers collide with each other, then return however far we are able to get
                if (in_collision)
                {
                    break;
                }

                stopwatch(RESET);
                arc_helpers::DoNotOptimize(gripper_a_interpolated);
                // If the grippers enter collision with the environment, then return however far we were able to get
                in_collision = (environment_sdf_->EstimateDistance3d(gripper_a_interpolated).first < gripper_min_distance_to_obstacles_) ||
                               (environment_sdf_->EstimateDistance3d(gripper_b_interpolated).first < gripper_min_distance_to_obstacles_) ||
                               (environment_sdf_->DistanceToBoundary3d(gripper_a_interpolated).first < gripper_min_distance_to_obstacles_) ||
                               (environment_sdf_->DistanceToBoundary3d(gripper_b_interpolated).first < gripper_min_distance_to_obstacles_);
                arc_helpers::DoNotOptimize(in_collision);
                const double collision_check_time_pt2 = stopwatch(READ);
                total_collision_check_time_ += collision_check_time_pt2;
                if (in_collision)
                {
                    break;
                }
            }

            RubberBand::Ptr next_band = std::make_shared<RubberBand>(*prev_band);
            if (extend_band)
            {
                stopwatch(RESET);
                arc_helpers::DoNotOptimize(rubber_band_verbose);
                // Forward simulate the rubber band to test this transition
                next_band->forwardPropagateRubberBandToEndpointTargets(
                            next_grippers_poses.first.translation(),
                            next_grippers_poses.second.translation(),
                            rubber_band_verbose);
                arc_helpers::DoNotOptimize(next_band->getVectorRepresentation());
                const double band_forward_propogation_time = stopwatch(READ);
                total_band_forward_propogation_time_ += band_forward_propogation_time;

                // If we are still able to get to the next target position after retrying,
                // then return however far we were able to get
                if (!bandEndpointsMatchGripperPositions(*next_band, next_grippers_poses))
                {
                    break;
                }

                // If the rubber band becomes overstretched, then return however far we were able to get
                if (next_band->isOverstretched())
                {
                    break;
                }
            }
            else
            {
                if (maxGrippersDistanceViolated(next_grippers_poses, max_grippers_distance_))
                {
                    break;
                }
            }

            // The new configuation is valid, add it to the tree
            const RRTNode next_node(next_grippers_poses, next_robot_configuration, next_band, parent_idx);
            tree_to_extend.push_back(next_node);
            const int64_t new_node_idx = (int64_t)tree_to_extend.size() - 1;
            prev_node.addChildIndex(new_node_idx);

            parent_idx = new_node_idx;
            ++step_index;
        }
    }

    const size_t nodes_at_end_of_propogation = tree_to_extend.size();
    const size_t nodes_created = nodes_at_end_of_propogation - nodes_at_start_of_propogation;

    if (visualization_enabled_globally_ &&
        visualization_enabled_locally &&
        nodes_created > 0 &&
        tree_to_extend.size() % visualization_frequency == 0)
    {
        const auto starting_idx = (&tree_to_extend == &forward_tree_)
                    ? forward_tree_next_visualized_node_
                    : backward_tree_next_visualized_node_;

        const auto& tree_a_color = (&tree_to_extend == &forward_tree_)
                    ? gripper_a_forward_tree_color_
                    : gripper_a_backward_tree_color_;

        const auto& tree_b_color = (&tree_to_extend == &forward_tree_)
                    ? gripper_b_forward_tree_color_
                    : gripper_b_backward_tree_color_;

        const bool draw_band = false;
        visualizeTree(
                    tree_to_extend,
                    starting_idx,
                    RRT_TREE_GRIPPER_A_NS,
                    RRT_TREE_GRIPPER_B_NS,
                    RRT_TREE_BAND_NS,
                    tree_marker_id_,
                    tree_marker_id_,
                    1,
                    tree_a_color,
                    tree_b_color,
                    band_tree_color_,
                    draw_band);
        ++tree_marker_id_;

        if (&tree_to_extend == &forward_tree_)
        {
            forward_tree_next_visualized_node_ = tree_to_extend.size();
        }
        else
        {
            backward_tree_next_visualized_node_ = tree_to_extend.size();
        }
    }

    arc_helpers::DoNotOptimize(nodes_created);
    const double everything_included_forward_propogation_time = function_wide_stopwatch(READ);
    total_everything_included_forward_propogation_time_ += everything_included_forward_propogation_time;

    return nodes_created;
}




std::vector<RRTNode, RRTAllocator> RRTHelper::planningMainLoop()
{
    total_sampling_time_ = 0.0;
    total_nearest_neighbour_index_building_time_ = 0.0;
    total_nearest_neighbour_index_searching_time_ = 0.0;
    total_nearest_neighbour_linear_searching_time_ = 0.0;
    total_nearest_neighbour_time_ = 0.0;
    total_projection_time_ = 0.0;
    total_collision_check_time_ = 0.0;
    total_band_forward_propogation_time_ = 0.0;
    total_first_order_vis_propogation_time_ = 0.0;
    total_everything_included_forward_propogation_time_ = 0.0;

    size_t forward_random_samples_useful = 0;
    size_t forward_random_samples_useless = 0;
    size_t backward_random_samples_useful = 0;
    size_t backward_random_samples_useless = 0;
    size_t forward_connection_attempts_useful = 0;
    size_t forward_connection_attempts_useless = 0;
    size_t forward_connections_made = 0;

    path_found_ = false;
    int64_t goal_idx_in_forward_tree = -1;

    // Make sure we've been given a start and goal state
    assert(forward_tree_.size() > 0);
    assert(backward_tree_.size() > 0);
    // Make sure the tree is properly linked
    assert(CheckTreeLinkage(forward_tree_));
    assert(CheckTreeLinkage(backward_tree_));

    const std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
    forward_iteration_ = false;

    const bool fwd_prop_local_visualization_enabled = true;
    const size_t fwd_prop_max_steps = 32;

    // Plan
    std::cout << "Starting planning..." << std::endl;
    std::chrono::duration<double> time_ellapsed = std::chrono::steady_clock::now() - start_time;
    while (!path_found_ && time_ellapsed < time_limit_)
    {
        if (forward_iteration_)
        {
            for (size_t itr = 0; !path_found_ && itr < forward_tree_extend_iterations_ && time_ellapsed < time_limit_; ++itr)
            {
                //////////////// Extend (connect) the first tree towards a random target ////////////////
                const bool extend_band = true;

                // Sample a random target
                const RRTNode random_target = configSampling();
                // Get the nearest neighbor
                const int64_t forward_tree_nearest_neighbour_idx = nearestNeighbour(true, random_target);
                // Forward propagate towards the sampled target
                const size_t num_random_nodes_created =
                        forwardPropogationFunction(
                            forward_tree_,
                            forward_tree_nearest_neighbour_idx,
                            random_target,
                            extend_band,
                            fwd_prop_max_steps,
                            fwd_prop_local_visualization_enabled);

                // Record statistics for the randomly sampled extensions
                if (num_random_nodes_created != 0)
                {
                    ++forward_random_samples_useful;
                }
                else
                {
                    ++forward_random_samples_useless;
                }

                // Check if any of the new nodes reached the goal
                for (size_t idx = forward_tree_.size() - num_random_nodes_created; idx < forward_tree_.size(); ++idx)
                {
                    const RRTNode& test_node = forward_tree_[idx];
                    if (goalReached(test_node))
                    {
                        path_found_ = true;
                        goal_idx_in_forward_tree = idx;
                        break;
                    }
                }

                const bool sample_goal = uniform_unit_distribution_(*generator_) < goal_bias_;
                if (num_random_nodes_created > 0 && sample_goal)
                {
                    // Record the index of the last node in the new branch.
                    // This is either the last item in the tree, or the nearest neighbour itself
                    // Given the check for num_random_nodes_created > 0 above, this will always be the last item in the tree
                    const int64_t last_node_idx_in_forward_tree_branch = num_random_nodes_created > 0 ?
                                (int64_t)forward_tree_.size() - 1 : forward_tree_nearest_neighbour_idx;

                    const int64_t backward_tree_nearest_neighbour_idx =
                            nearestNeighbour(false, forward_tree_[last_node_idx_in_forward_tree_branch]);
                    const RRTNode& target_in_backward_tree = backward_tree_[backward_tree_nearest_neighbour_idx];

                    const size_t num_goal_directed_nodes_created =
                            forwardPropogationFunction(
                                forward_tree_,
                                last_node_idx_in_forward_tree_branch,
                                target_in_backward_tree,
                                extend_band,
                                fwd_prop_max_steps,
                                fwd_prop_local_visualization_enabled);

                    // Record statistics for the goal biased extension
                    if (num_goal_directed_nodes_created != 0)
                    {
                        ++forward_connection_attempts_useful;
                    }
                    else
                    {
                        ++forward_connection_attempts_useless;
                    }

                    //////////////// Check for a connection between the trees, extending the forward tree if possible ////////////////

                    if (num_goal_directed_nodes_created > 0)
                    {
                        const bool connection_made = planning_for_whole_robot_ ?
                                    robotConfigurationsAreApproximatelyEqual(forward_tree_.back().getRobotConfiguration(), target_in_backward_tree.getRobotConfiguration()) :
                                    gripperPositionsAreApproximatelyEqual(forward_tree_.back().getGrippers(), target_in_backward_tree.getGrippers());

                        if (connection_made)
                        {
                            assert(CheckTreeLinkage(forward_tree_));
                            assert(CheckTreeLinkage(backward_tree_));

                            // Record some statistics
                            ++forward_connections_made;

                            // March down the backward tree, propagating the band in the forward tree
                            int64_t forward_parent_idx = (int64_t)forward_tree_.size() - 1;
                            int64_t backward_next_idx = backward_tree_nearest_neighbour_idx;
                            while (backward_next_idx >= 0)
                            {
                                assert(backward_next_idx < (int64_t)backward_tree_.size());
                                RRTNode& backward_next_node = backward_tree_[backward_next_idx];
                                const RRTGrippersRepresentation& next_grippers_poses = backward_next_node.getGrippers();
                                const RRTRobotRepresentation& next_robot_configuration = backward_next_node.getRobotConfiguration();
                                const RubberBand::Ptr prev_band = forward_tree_[forward_parent_idx].getBand();
            //                    backward_next_node.blacklist();


                                RubberBand::Ptr next_band = std::make_shared<RubberBand>(*prev_band);
                                Stopwatch stopwatch;
                                const bool rubber_band_verbose = true;
                                arc_helpers::DoNotOptimize(rubber_band_verbose);
                                // Forward simulate the rubber band to test this transition
                                next_band->forwardPropagateRubberBandToEndpointTargets(
                                            next_grippers_poses.first.translation(),
                                            next_grippers_poses.second.translation(),
                                            rubber_band_verbose);
                                arc_helpers::DoNotOptimize(next_band->getVectorRepresentation());
                                const double band_forward_propogation_time = stopwatch(READ);
                                total_band_forward_propogation_time_ += band_forward_propogation_time;

                                // If we are still able to get to the next target position after retrying,
                                // then return however far we were able to get
                                if (!bandEndpointsMatchGripperPositions(*next_band, next_grippers_poses))
                                {
                                    break;
                                }

                                // If the rubber band becomes overstretched, then return however far we were able to get
                                if (next_band->isOverstretched())
                                {
                                    break;
                                }

                                next_band->visualize("bispace_connect_bands", Visualizer::Blue(), Visualizer::Blue(), (int)forward_parent_idx + 1, true);

                                // The new configuation is valid, add it to the forward tree
                                const RRTNode next_node(
                                            next_grippers_poses,
                                            next_robot_configuration,
                                            next_band,
                                            forward_parent_idx);
                                forward_tree_.push_back(next_node);
                                const int64_t new_node_idx = (int64_t)forward_tree_.size() - 1;
                                forward_tree_[forward_parent_idx].addChildIndex(new_node_idx);

                                if (goalReached(next_node))
                                {
                                    path_found_ = true;
                                    goal_idx_in_forward_tree = new_node_idx;
                                    break;
                                }

                                forward_parent_idx = new_node_idx;
                                backward_next_idx = backward_next_node.getParentIndex();
                            }
                        }
                    }
                }

                time_ellapsed = std::chrono::steady_clock::now() - start_time;
            }
        }
        else
        {
            for (size_t itr = 0; itr < backward_tree_extend_iterations_ && time_ellapsed < time_limit_; ++itr)
            {
                //////////////// Extend (connect) the backward tree towards a random target ////////////////
                const bool extend_band = false;

                // Sample a random target
                const RRTNode random_target = configSampling();
                // Get the nearest neighbor
                const int64_t backward_tree_nearest_neighbour_idx = nearestNeighbour(false, random_target);
                // Forward propagate towards the sampled target
                const size_t num_nodes_created =
                        forwardPropogationFunction(
                            backward_tree_,
                            backward_tree_nearest_neighbour_idx,
                            random_target,
                            extend_band,
                            fwd_prop_max_steps,
                            fwd_prop_local_visualization_enabled);

                // Record statistics for the randomly sampled extension
                if (num_nodes_created != 0)
                {
                    ++backward_random_samples_useful;
                }
                else
                {
                    ++backward_random_samples_useless;
                }

                time_ellapsed = std::chrono::steady_clock::now() - start_time;
            }
        }
        forward_iteration_ = !forward_iteration_;
    }

    std::cout << "Finished planning, for better or worse" << std::endl;

    assert(CheckTreeLinkage(forward_tree_));
    assert(CheckTreeLinkage(backward_tree_));

    std::vector<RRTNode, RRTAllocator> path;
    if (path_found_)
    {
        path = ExtractSolutionPath(forward_tree_, goal_idx_in_forward_tree);
    }

    const std::chrono::time_point<std::chrono::steady_clock> cur_time = std::chrono::steady_clock::now();
    const std::chrono::duration<double> planning_time(cur_time - start_time);

    planning_statistics_["planning_time0_sampling                                 "] = total_sampling_time_;
    planning_statistics_["planning_time1_1_nearest_neighbour_index_building       "] = total_nearest_neighbour_index_building_time_;
    planning_statistics_["planning_time1_2_nearest_neighbour_index_searching      "] = total_nearest_neighbour_index_searching_time_;
    planning_statistics_["planning_time1_3_nearest_neighbour_linear_searching     "] = total_nearest_neighbour_linear_searching_time_;
    planning_statistics_["planning_time1_nearest_neighbour                        "] = total_nearest_neighbour_time_;
    planning_statistics_["planning_time2_1_forward_propogation_projection         "] = total_projection_time_;
    planning_statistics_["planning_time2_2_forward_propogation_collision_check    "] = total_collision_check_time_;
    planning_statistics_["planning_time2_3_forward_propogation_band_sim           "] = total_band_forward_propogation_time_;
    planning_statistics_["planning_time2_4_forward_propogation_first_order_vis    "] = total_first_order_vis_propogation_time_;
    planning_statistics_["planning_time2_forward_propogation_everything_included  "] = total_everything_included_forward_propogation_time_;
    planning_statistics_["planning_time3_total                                    "] = planning_time.count();

    planning_statistics_["planning_size00_forward_random_samples_useless          "] = (double)forward_random_samples_useless;
    planning_statistics_["planning_size01_forward_random_samples_useful           "] = (double)forward_random_samples_useful;
    planning_statistics_["planning_size02_forward_states                          "] = (double)forward_tree_.size();

    planning_statistics_["planning_size03_backward_random_samples_useless         "] = (double)backward_random_samples_useless;
    planning_statistics_["planning_size04_backward_random_samples_useful          "] = (double)backward_random_samples_useful;
    planning_statistics_["planning_size05_backward_states                         "] = (double)backward_tree_.size();

    planning_statistics_["planning_size06_forward_connection_attempts_useless     "] = (double)forward_connection_attempts_useless;
    planning_statistics_["planning_size07_forward_connection_attempts_useful      "] = (double)forward_connection_attempts_useful;
    planning_statistics_["planning_size08_forward_connections_made                "] = (double)forward_connections_made;

    return path;
}

/*
 * Builds the helper functions needed by simple_rrt_planner and invokes the planner (and shortcut smoother)
 */
std::vector<RRTNode, RRTAllocator> RRTHelper::plan(
        const RRTNode& start,
        const RRTGrippersRepresentation& grippers_goal_poses,
        const std::chrono::duration<double>& time_limit,
        const bool planning_for_whole_robot)
{
    const auto estimated_tree_size = ROSHelpers::GetParam(ph_, "estimated_tree_size", 100000);

    // Extract robot information
    planning_for_whole_robot_ = planning_for_whole_robot;
    total_dof_ = start.getRobotConfiguration().size();
    robot_joint_limits_lower_ = robot_->getJointLowerLimits();
    robot_joint_limits_upper_ = robot_->getJointUpperLimits();
    robot_joint_weights_ = robot_->getJointWeights();

    // Extract start information
    starting_band_ = std::make_shared<RubberBand>(*start.getBand());
    starting_robot_configuration_ = start.getRobotConfiguration();
    starting_grippers_poses_ = start.getGrippers();

    // Setup the forward tree
    forward_tree_.clear();
    forward_tree_.reserve(estimated_tree_size);
    forward_tree_.push_back(start);

    // Extract goal/termination information
    max_grippers_distance_ = start.getBand()->maxSafeLength();
    time_limit_ = time_limit;

    // Setup the backward tree
    backward_tree_.clear();
    backward_tree_.reserve(estimated_tree_size);
    // If we're using the whole robot, we may need to tweak the goal config to be feasible
    if (planning_for_whole_robot_)
    {
        const auto goal_configurations = robot_->getCloseIkSolutions(
                    {grippers_goal_poses.first, grippers_goal_poses.second},
                    max_grippers_distance_);
        assert(goal_configurations.size() > 0);

        const auto grippers_goal_poses_updated_vec = robot_->getGrippersPoses(goal_configurations[0]);
        grippers_goal_poses_.first = grippers_goal_poses_updated_vec.at(0);
        grippers_goal_poses_.second = grippers_goal_poses_updated_vec.at(1);


        for (size_t idx = 0; idx < goal_configurations.size(); ++idx)
        {
            backward_tree_.push_back(
                        RRTNode(
                            grippers_goal_poses_,
                            goal_configurations[idx],
                            start.getBand()));
        }

    }
    else
    {
        grippers_goal_poses_ = grippers_goal_poses;
        backward_tree_.push_back(RRTNode(grippers_goal_poses_, start.getRobotConfiguration(), start.getBand()));
    }

    // Double check that the input goal location isn't immediately impossible
    const double first_gripper_dist_to_env = environment_sdf_->EstimateDistance3d(grippers_goal_poses_.first.translation()).first;
    const double second_gripper_dist_to_env = environment_sdf_->EstimateDistance3d(grippers_goal_poses_.second.translation()).first;
    if (first_gripper_dist_to_env < gripper_min_distance_to_obstacles_ ||
        second_gripper_dist_to_env < gripper_min_distance_to_obstacles_ ||
        (maxGrippersDistanceViolated(grippers_goal_poses_, max_grippers_distance_) > max_grippers_distance_))
    {
        const double dist_between_grippers = (grippers_goal_poses_.first.translation() - grippers_goal_poses_.second.translation()).norm();
        std::cerr << "Unfeasible goal location: " << grippers_goal_poses_.first.translation() << "  :  " << grippers_goal_poses_.second.translation() << std::endl;
        std::cerr << "Min gripper collision distance: " << gripper_min_distance_to_obstacles_ << " Current Distances: " << first_gripper_dist_to_env << " " << second_gripper_dist_to_env << std::endl;
        std::cerr << "Max allowable distance: " << max_grippers_distance_ << " Distance beteween goal grippers: " << dist_between_grippers << std::endl;
        assert(false && "Unfeasible goal location");
    }


    // Clear the forward tree flann data
    forward_nn_raw_data_.clear();
    forward_nn_raw_data_.reserve(total_dof_ * estimated_tree_size);
    forward_nn_index_ = NNIndexType(flann::KDTreeSingleIndexParams(), flann::L2_weighted<float>(robot_->getJointWeights()));
    forward_next_idx_to_add_to_nn_dataset_ = 0;

    // Clear the backward tree flann data
    backward_nn_raw_data_.clear();
    backward_nn_raw_data_.reserve(total_dof_ * estimated_tree_size);
    backward_nn_index_ = NNIndexType(flann::KDTreeSingleIndexParams(), flann::L2_weighted<float>(robot_->getJointWeights()));
    backward_next_idx_to_add_to_nn_dataset_ = 0;

    if (visualization_enabled_globally_)
    {
        tree_marker_id_ = 1;
        forward_tree_next_visualized_node_ = 0;
        backward_tree_next_visualized_node_ = 0;
        visualizeBlacklist();
    }

    ROS_INFO_NAMED("rrt", "Starting SimpleHybridRRTPlanner");
    std::vector<RRTNode, RRTAllocator> path;
    if (useStoredPath())
    {
        path = loadStoredPath();
    }

    // If we failed to retreive a path (or we didn't try) then plan a path
    if (path.size() == 0)
    {
        robot_->lockEnvironment();
        path = planningMainLoop();
        robot_->unlockEnvironment();
        std::cout << "RRT Helper Internal Statistics:\n" << PrettyPrint::PrettyPrint(planning_statistics_, false, "\n") << std::endl << std::endl;
        storePath(path);
    }

    std::cout << " !!!!!!!!!!!!! Smoothing currently disabled, returning unsmoothed path !!!!!" << std::endl;

    if (path.size() != 0)
    {
        if (visualization_enabled_globally_)
        {
            vis_->clearVisualizationsBullet();
            visualizePath(path);
        }
    }

    return path;






    // If we either retreived a path, or made a new one, visualize and do smoothing
    if (path.size() != 0)
    {
        if (visualization_enabled_globally_)
        {
            vis_->clearVisualizationsBullet();
            visualizePath(path);
        }

        ROS_INFO_NAMED("rrt", "Playing back unsmoothed path in OpenRAVE");
        robot_->testPathForCollision(ConvertRRTPathToRobotPath(path));

        ROS_INFO_NAMED("rrt", "Starting Shortcut Smoothing");
        robot_->lockEnvironment();
        const bool visualize_rrt_smoothing = true;
        const auto smoothed_path = rrtShortcutSmooth(path, visualize_rrt_smoothing);
        robot_->unlockEnvironment();
        storePath(smoothed_path);
        std::cout << "RRT Helper Internal Statistics:\n" << PrettyPrint::PrettyPrint(smoothing_statistics_, false, "\n") << std::endl << std::endl;

        ROS_INFO_NAMED("rrt", "Playing back smoothed path in OpenRAVE");
        robot_->testPathForCollision(ConvertRRTPathToRobotPath(smoothed_path));

        if (visualization_enabled_globally_)
        {
            vis_->deleteObjects(RRT_BLACKLISTED_GOAL_BANDS_NS, 1, 2);
        }

        return smoothed_path;
    }
    else
    {
        return path;
    }
}




/* Checks the planner tree to make sure the parent-child linkages are correct
 */
bool RRTHelper::CheckTreeLinkage(const std::vector<RRTNode, RRTAllocator>& tree)
{
    // Step through each state in the tree. Make sure that the linkage to the parent and child states are correct
    for (size_t current_index = 0; current_index < tree.size(); current_index++)
    {
        // For every state, make sure all the parent<->child linkages are valid
        const auto& current_node = tree[current_index];
        if (!current_node.isInitialized())
        {
            std::cerr << "Tree contains uninitialized node(s) " << current_index << std::endl;
            return false;
        }
        // Check the linkage to the parent state
        const int64_t parent_index = current_node.getParentIndex();
        if ((parent_index >= 0) && (parent_index < (int64_t)tree.size()))
        {
            if (parent_index != (int64_t)current_index)
            {
                const auto& parent_node = tree[parent_index];
                if (!parent_node.isInitialized())
                {
                    std::cerr << "Tree contains uninitialized node(s) " << parent_index << std::endl;
                    return false;
                }
                // Make sure the corresponding parent contains the current node in the list of child indices
                const std::vector<int64_t>& parent_child_indices = parent_node.getChildIndices();
                auto index_found = std::find(parent_child_indices.begin(), parent_child_indices.end(), (int64_t)current_index);
                if (index_found == parent_child_indices.end())
                {
                    std::cerr << "Parent state " << parent_index << " does not contain child index for current node " << current_index << std::endl;
                    return false;
                }
            }
            else
            {
                std::cerr << "Invalid parent index " << parent_index << " for state " << current_index << " [Indices can't be the same]" << std::endl;
                return false;
            }
        }
        else if (parent_index < -1)
        {
            std::cerr << "Invalid parent index " << parent_index << " for state " << current_index << std::endl;
            return false;
        }
        // Check the linkage to the child states
        const std::vector<int64_t>& current_child_indices = current_node.getChildIndices();
        for (size_t idx = 0; idx < current_child_indices.size(); idx++)
        {
            // Get the current child index
            const int64_t current_child_index = current_child_indices[idx];
            if ((current_child_index > 0) && (current_child_index < (int64_t)tree.size()))
            {
                if (current_child_index != (int64_t)current_index)
                {
                    const auto& child_state = tree[current_child_index];
                    if (!child_state.isInitialized())
                    {
                        std::cerr << "Tree contains uninitialized node(s) " << current_child_index << std::endl;
                        return false;
                    }
                    // Make sure the child node points to us as the parent index
                    const int64_t child_parent_index = child_state.getParentIndex();
                    if (child_parent_index != (int64_t)current_index)
                    {
                        std::cerr << "Parent index " << child_parent_index << " for current child state " << current_child_index << " does not match index " << current_index << " for current node " << std::endl;
                        return false;
                    }
                }
                else
                {
                    std::cerr << "Invalid child index " << current_child_index << " for state " << current_index << " [Indices can't be the same]" << std::endl;
                    return false;
                }
            }
            else
            {
                std::cerr << "Invalid child index " << current_child_index << " for state " << current_index << std::endl;
                return false;
            }
        }
    }
    return true;
}

std::vector<RRTNode, RRTAllocator> RRTHelper::ExtractSolutionPath(
        const std::vector<RRTNode, RRTAllocator>& tree,
        const int64_t goal_node_idx)
{
    std::vector<RRTNode, RRTAllocator> solution_path;
    int64_t next_index = goal_node_idx;
    while (next_index >= 0)
    {
        assert(next_index < (int64_t)tree.size());
        const auto& parent_node = tree[next_index];
        solution_path.push_back(parent_node);
        next_index = parent_node.getParentIndex();
    }
    // Put it in the right order
    std::reverse(solution_path.begin(), solution_path.end());
    return solution_path;
}

std::vector<VectorXd> RRTHelper::ConvertRRTPathToRobotPath(const std::vector<RRTNode, RRTAllocator>& path)
{
    std::vector<VectorXd> robot_config_path(path.size());
    for (size_t ind = 0; ind < path.size(); ++ind)
    {
        robot_config_path[ind] =  path[ind].getRobotConfiguration();
    }
    return robot_config_path;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper function for goal check
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void RRTHelper::addBandToBlacklist(const EigenHelpers::VectorVector3d& band)
{
    blacklisted_goal_rubber_bands_.push_back(band);
}

void RRTHelper::clearBlacklist()
{
    blacklisted_goal_rubber_bands_.clear();
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const EigenHelpers::VectorVector3d& test_band) const
{
    for (size_t idx = 0; idx < blacklisted_goal_rubber_bands_.size(); idx++)
    {
        const EigenHelpers::VectorVector3d& blacklisted_path = blacklisted_goal_rubber_bands_[idx];

        // Checks if the straight line between elements of the two paths is collision free
        const auto straight_line_collision_check_fn = [&] (
                const ssize_t blacklisted_path_ind,
                const ssize_t test_band_ind)
        {
            assert(0 <= blacklisted_path_ind && blacklisted_path_ind < (ssize_t)blacklisted_path.size());
            assert(0 <= test_band_ind && test_band_ind < (ssize_t)test_band.size());

            const Vector3d& first_node = blacklisted_path[blacklisted_path_ind];
            const Vector3d& second_node = test_band[test_band_ind];

//            if (visualize)
//            {
//                vis_->visualizeLineStrip("first_order_vis_check", {first_node, second_node}, Visualizer::White(), 2);
//            }

            const ssize_t num_steps = (ssize_t)std::ceil((second_node - first_node).norm() / environment_sdf_->GetResolution());

            // We don't need to check the endpoints as they are already checked as part of the rubber band process
            for (ssize_t ind = 1; ind < num_steps; ++ind)
            {
                const double ratio = (double)ind / (double)num_steps;
                const Vector3d interpolated_point = EigenHelpers::Interpolate(first_node, second_node, ratio);
                if (environment_sdf_->Get3d(interpolated_point) < 0.0)
                {
//                    if (visualize)
//                    {
//                        collision_points.push_back(Vector3d((double)blacklisted_path_ind, (double)test_band_ind, 0.0));
//                        vis.visualizePoints(
//                                    "first_order_vis_collision",
//                                    collision_points,
//                                    Visualizer::Red(),
//                                    1,
//                                    1.0);
//                        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
//                    }
                    return false;
                }
            }

            return true;
        };

        // If we've found a first order deformation, then we are similar to a blacklisted item
        if (FirstOrderDeformation::CheckFirstOrderDeformation(
                blacklisted_path.size(),
                test_band.size(),
                straight_line_collision_check_fn))
        {
            return true;
        }
    }

    return false;
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const RubberBand& test_band)
{
    Stopwatch stopwatch;
    auto vector_representation = test_band.getVectorRepresentation();

    arc_helpers::DoNotOptimize(vector_representation);
    const bool is_first_order_visible = isBandFirstOrderVisibileToBlacklist(vector_representation);
    arc_helpers::DoNotOptimize(is_first_order_visible);

    const double first_order_vis_time = stopwatch(READ);
    total_first_order_vis_propogation_time_ += first_order_vis_time;

    return is_first_order_visible;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for shortcut smoothing
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static EigenHelpers::VectorVector3d findFirstGripperWaypoints(
        const std::vector<RRTNode, RRTAllocator>& path,
        const size_t start_index,
        const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());

    // The start of the path is clearly the first 'kink'
    EigenHelpers::VectorVector3d gripper_path_kinks(1, path[start_index].getGrippers().first.translation());

    size_t last_kink = start_index;
    Vector3d last_kink_gripper_position = path[last_kink].getGrippers().first.translation();
    double path_distance = 0.0;

    // We don't include the last index because it is clearly the last 'kink'
    for (size_t idx = start_index; idx < end_index - 1; ++idx)
    {
        const Vector3d& current_gripper_position = path[idx].getGrippers().first.translation();
        const Vector3d& next_gripper_position    = path[idx + 1].getGrippers().first.translation();
        path_distance += (next_gripper_position - current_gripper_position).norm();
        const double straight_line_distance = (next_gripper_position - last_kink_gripper_position).norm();

        // If the straight line distance between the start and the next gripper does not match the path distance, then the current node is a kink
        if (!EigenHelpers::IsApprox(straight_line_distance, path_distance, 1e-6))
        {
            last_kink = idx;
            last_kink_gripper_position = path[last_kink].getGrippers().first.translation();
            path_distance = (next_gripper_position - current_gripper_position).norm();
            gripper_path_kinks.push_back(last_kink_gripper_position);
        }
    }
    gripper_path_kinks.push_back(path[end_index].getGrippers().first.translation());

    return gripper_path_kinks;
}

static EigenHelpers::VectorVector3d findSecondGripperWaypoints(
        const std::vector<RRTNode, RRTAllocator>& path,
        const size_t start_index,
        const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());

    // The start of the path is clearly the first 'kink'
    EigenHelpers::VectorVector3d gripper_path_kinks(1, path[start_index].getGrippers().second.translation());

    size_t last_kink = start_index;
    Vector3d last_kink_gripper_position = path[last_kink].getGrippers().second.translation();
    double path_distance = 0;

    // We don't include the last index because it is clearly the last 'kink'
    for (size_t idx = start_index; idx < end_index - 1; ++idx)
    {
        const Vector3d& current_gripper_position = path[idx].getGrippers().second.translation();
        const Vector3d& next_gripper_position    = path[idx + 1].getGrippers().second.translation();
        path_distance += (next_gripper_position - current_gripper_position).norm();
        const double straight_line_distance = (next_gripper_position - last_kink_gripper_position).norm();

        // If the straight line distance between the start and the next gripper does not match the path distance, then the current node is a kink
        if (!EigenHelpers::IsApprox(straight_line_distance, path_distance, 1e-6))
        {
            last_kink = idx;
            last_kink_gripper_position = current_gripper_position;
            path_distance = (next_gripper_position - current_gripper_position).norm();
            gripper_path_kinks.push_back(last_kink_gripper_position);
        }
    }
    gripper_path_kinks.push_back(path[end_index].getGrippers().second.translation());

    return gripper_path_kinks;
}

static EigenHelpers::VectorVector3d createOtherGripperWaypoints(
        const EigenHelpers::VectorVector3d& given_gripper_waypoints,
        const Vector3d& start_point,
        const Vector3d& end_point)
{
    const size_t num_waypoints = given_gripper_waypoints.size();
    assert(num_waypoints >= 2);

    EigenHelpers::VectorVector3d other_gripper_waypoints;
    other_gripper_waypoints.reserve(num_waypoints);
    other_gripper_waypoints.push_back(start_point);

    // We will need to "space out" the distance between start_point and end_point to match those of the given waypoints
    // Note that we've already inserted the first waypoint, and we'll insert the last manually as well
    const std::vector<double> cummulative_distances = EigenHelpers::CalculateCumulativeDistances(given_gripper_waypoints);
    for (size_t idx = 1; idx < num_waypoints - 1; ++idx)
    {
        const double ratio = cummulative_distances[idx] / cummulative_distances.back();
        const auto next_waypoint = EigenHelpers::Interpolate(start_point, end_point, ratio);
        other_gripper_waypoints.push_back(next_waypoint);
    }
    other_gripper_waypoints.push_back(end_point);

    assert(other_gripper_waypoints.size() == num_waypoints);
    return other_gripper_waypoints;
}

/**
 * @brief RRTHelper::forwardSimulateGrippersPath
 *   Forward simulates the rubber band starting the grippers at position path[start_index]
 *   and ending at the end of the path. Used by rrtShortcutSmooth.
 * @param rubber_band
 * @param path
 * @param start_index
 * @param end_index
 * @return A vector of RRTConfig of at most (end_index - start_index) elements; includes path[start_index].
 */
std::pair<bool, std::vector<RRTNode, RRTAllocator>> RRTHelper::forwardSimulateGrippersPath(
        const std::vector<RRTNode, RRTAllocator>& path,
        const size_t start_index,
        RubberBand rubber_band)
{
    Stopwatch function_wide_stopwatch;
    Stopwatch stopwatch;

    assert(start_index < path.size());

    // Verify that the endpoints of the rubber band match the start of the grippers path
    if (!bandEndpointsMatchGripperPositions(rubber_band, path[start_index].getGrippers()))
    {
        std::cerr << "Inside forwardSimulateGrippersPath\n";
        std::cerr << "initial rubber band endpoints:\n"
                  << PrettyPrint::PrettyPrint(rubber_band.getEndpoints()) << std::endl;

        std::cerr << "path gripper positions:\n"
                  << PrettyPrint::PrettyPrint(path[start_index].getGrippers()) << std::endl;

        assert(false && "Band endpoints do not match recorded gripper positions");
    }

    // Collect the results for use by the rrtShortcutSmooth function
    std::vector<RRTNode, RRTAllocator> resulting_path;
    // Put the start position on the path
    {
        resulting_path.reserve(path.size() - start_index);
        resulting_path.push_back(
                    RRTNode(path[start_index].getGrippers(),
                            path[start_index].getRobotConfiguration(),
                            std::make_shared<RubberBand>(rubber_band)));
    }

    // Advance the grippers, simulating the rubber band until we reach the end of the path, or the band is overstretched
    bool band_is_overstretched = rubber_band.isOverstretched();
    bool band_got_stuck = false;
    size_t path_idx = start_index + 1;
    const bool rubber_band_verbose = false && visualization_enabled_globally_;
    while (!band_is_overstretched && !band_got_stuck && path_idx < path.size())
    {
        // Forward simulate the band
        stopwatch(RESET);
        const auto& ending_grippers_pos = path[path_idx].getGrippers();
        rubber_band.forwardPropagateRubberBandToEndpointTargets(
                    ending_grippers_pos.first.translation(),
                    ending_grippers_pos.second.translation(),
                    rubber_band_verbose);
        const double forward_propogation_time = stopwatch(READ);
        total_band_forward_propogation_time_ += forward_propogation_time;

        // Store the band in the results
//        stopwatch(RESET);
//        const bool is_first_order_visible = isBandFirstOrderVisibileToBlacklist(rubber_band);
//        const double first_order_vis_time = stopwatch(READ);
//        total_first_order_vis_propogation_time_ += first_order_vis_time;
        resulting_path.push_back(RRTNode(
                                     path[path_idx].getGrippers(),
                                     path[path_idx].getRobotConfiguration(),
                                     std::make_shared<RubberBand>(rubber_band)));

        // Record if the band is overstretched
        band_is_overstretched = rubber_band.isOverstretched();
        band_got_stuck = !bandEndpointsMatchGripperPositions(rubber_band, path[path_idx].getGrippers());

        ++path_idx;
    }

    // If we the band is not overstretched, and the band did not get stuck,
    // then we reached the end of the path succesfully
    const bool success = !band_is_overstretched && !band_got_stuck;

    const double everything_included_forward_propogation_time = function_wide_stopwatch(READ);
    total_everything_included_forward_propogation_time_ += everything_included_forward_propogation_time;
    return std::make_pair(success, resulting_path);
}

std::vector<RRTNode, RRTAllocator> RRTHelper::rrtShortcutSmooth(
        std::vector<RRTNode, RRTAllocator> path,
        const bool visualization_enabled_locally)
{
    Stopwatch function_wide_stopwatch;

    uint32_t num_iterations = 0;
    uint32_t failed_iterations = 0;
    total_projection_time_ = 0.0;
    total_collision_check_time_ = 0.0;
    total_band_forward_propogation_time_ = 0.0;
    total_first_order_vis_propogation_time_ = 0.0;
    total_everything_included_forward_propogation_time_ = 0.0;

    if (visualization_enabled_globally_ && visualization_enabled_locally)
    {
        visualizePath(path);
    }

    // The main smoothing loop
    while (path.size() > 2 &&
           num_iterations < max_smoothing_iterations_ &&
           failed_iterations < max_failed_smoothing_iterations_)
    {
        ++num_iterations;

        if (visualization_enabled_globally_ && visualization_enabled_locally)
        {
            vis_->clearVisualizationsBullet();
        }

        ///////////////////// Determine which nodes to try to shortcut between /////////////////////////////////////////

        const int64_t base_index = (int64_t)std::uniform_int_distribution<size_t>(0, path.size() - 1)(*generator_);

        // Compute the offset index
        // We want to sample the start and goal slightly more frequently, so allow "overshoots" of endpoints for the offset
        const int64_t min_delta = std::max(-base_index - max_shortcut_index_distance_ / 10, -max_shortcut_index_distance_);
        const int64_t max_delta = std::min((int64_t)path.size() - base_index - 1 + max_shortcut_index_distance_ / 10, max_shortcut_index_distance_);
        const int64_t offset_delta = std::uniform_int_distribution<int64_t>(min_delta, max_delta)(*generator_);
        // Clamp to the boundaries of the current path
        const int64_t second_index = arc_helpers::ClampValue(base_index + offset_delta, (int64_t)0, (int64_t)path.size() - 1);

        // Get start and end indices to establish the direction of the shortcut
        const size_t smoothing_start_index = (size_t)std::min(base_index, second_index);
        const size_t smoothing_end_index = (size_t)std::max(base_index, second_index);

        const auto& smoothing_start_config = path[smoothing_start_index];
        const auto& smoothing_end_config = path[smoothing_end_index];

        ///////////////////// Determine if a shortcut is even possible /////////////////////////////////////////////////

        // We know start_index <= end_index, this essentially checks if start == end or start + 1 == end
        if (smoothing_start_index + 1 >= smoothing_end_index)
        {
            continue;
        }
        // Distance checks are determined once we know what type of smoothing we are attempting, first, second, or both

        ///////////////////// Attempte a shortcut //////////////////////////////////////////////////////////////////////

        // Create structures to hold the results which will get filled by each part of the if/else chain
        std::vector<RRTNode, RRTAllocator> smoothed_segment;
        smoothed_segment.reserve(256);
        std::pair<bool, std::vector<RRTNode, RRTAllocator>> end_of_smoothing_to_goal_results;

        const bool fwd_prop_local_visualization_enabled = false;
        const bool fwd_prop_extend_band = true;
        const size_t fwd_prop_max_steps = 256;

        if (planning_for_whole_robot_)
        {
            // Check if the edge possibly can be smoothed
            const double minimum_distance = RRTNode::distance(smoothing_start_config.getRobotConfiguration(), smoothing_end_config.getRobotConfiguration());
            const double path_distance = RRTNode::robotPathDistance(path, smoothing_start_index, smoothing_end_index);
            // Essentially this checks if there is a kink in the path
            if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
            {
//                std::cout << "No smoothing possible, continuing\n";
                ++failed_iterations;
                continue;
            }

            // Forward simulate the rubber band along the straight line between gripper/robot positions
            const int64_t start_idx = 0;
            smoothed_segment.push_back(smoothing_start_config);
            forwardPropogationFunction(smoothed_segment, start_idx, smoothing_end_config, fwd_prop_extend_band, fwd_prop_max_steps, fwd_prop_local_visualization_enabled);

            // Check if the rubber band gets overstretched while propogating the grippers/robot on the new path
            const auto& target_robot_configuration = smoothing_end_config.getRobotConfiguration();
            const auto& last_robot_configuration = smoothed_segment.back().getRobotConfiguration();

            if (!robotConfigurationsAreApproximatelyEqual(last_robot_configuration, target_robot_configuration))
            {
//                std::cout << "Shortcut failed, continuing"
//                          << "   Robot configuration equal? " << robotConfigurationsAreApproximatelyEqual(last_robot_configuration, target_robot_configuration)
//                          << "\n";
                ++failed_iterations;
                continue;
            }

            // We still need to check that the rubber band can still reach the goal correctly from this state,
            // so we'll forward propogate along the rest of the trajectory to check feasibility
            end_of_smoothing_to_goal_results = forwardSimulateGrippersPath(path, smoothing_end_index, *smoothed_segment.back().getBand());
        }
        else
        {
            // First determine which type of smoothing we are doing, both grippers, or a single gripper
            // On a 1 or a 2, smooth both grippers,
            // On a 3 smooth the first gripper only,
            // On a 4 smooth the second gripper only
            const int smoothing_type = uniform_shortcut_smoothing_int_distribution_(*generator_);

            if (smoothing_type == 1 || smoothing_type == 2)
            {
                // Check if the edge possibly can be smoothed
                const double minimum_distance = RRTNode::distance(smoothing_start_config.getGrippers(), smoothing_end_config.getGrippers());
                const double path_distance = RRTNode::grippersPathDistance(path, smoothing_start_index, smoothing_end_index);
                // Essentially this checks if there is a kink in the path
                if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
                {
                    continue;
                }

                // Forward simulate the rubber band along the straight line between gripper positions
                const int64_t start_idx = 0;
                smoothed_segment.push_back(smoothing_start_config);
                forwardPropogationFunction(smoothed_segment, start_idx, smoothing_end_config, fwd_prop_extend_band, fwd_prop_max_steps, fwd_prop_local_visualization_enabled);
            }
            else if (smoothing_type == 3 || smoothing_type == 4)
            {
                // Once we know the fixed waypoints, then we do smoothing between these waypoints
                const EigenHelpers::VectorVector3d current_waypoints_first_gripper =
                        findFirstGripperWaypoints(path, smoothing_start_index, smoothing_end_index);
                const EigenHelpers::VectorVector3d current_waypoints_second_gripper =
                        findSecondGripperWaypoints(path, smoothing_start_index, smoothing_end_index);

                EigenHelpers::VectorVector3d target_waypoints_first_gripper;
                EigenHelpers::VectorVector3d target_waypoints_second_gripper;
                // Smooth the first gripper
                if (smoothing_type == 3)
                {
                    const auto& first_gripper_start_pos = smoothing_start_config.getGrippers().first.translation();
                    const auto& first_gripper_end_pos = smoothing_end_config.getGrippers().first.translation();

                    // Check if there is room for improvement for the first gripper
                    const double minimum_distance = (first_gripper_end_pos - first_gripper_start_pos).norm();
                    const double path_distance = EigenHelpers::CalculateTotalDistance(current_waypoints_first_gripper);
                    // Essentially this checks if there is a kink in the path
                    if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
                    {
                        continue;
                    }

                    // Follow the waypoints of the second gripper
                    target_waypoints_second_gripper = current_waypoints_second_gripper;
                    // Create new waypoints in a stright line for the first gripper
                    target_waypoints_first_gripper = createOtherGripperWaypoints(
                                target_waypoints_second_gripper,
                                first_gripper_start_pos,
                                first_gripper_end_pos);
                }
                // Smooth the second gripper
                else
                {
                    const auto& second_gripper_start_pos = smoothing_start_config.getGrippers().second.translation();
                    const auto& second_gripper_end_pos = smoothing_end_config.getGrippers().second.translation();

                    // Check if there is room for improvement for the second gripper
                    const double minimum_distance = (second_gripper_end_pos - second_gripper_start_pos).norm();
                    const double path_distance = EigenHelpers::CalculateTotalDistance(current_waypoints_first_gripper);
                    // Essentially this checks if there is a kink in the path
                    if (EigenHelpers::IsApprox(path_distance, minimum_distance, 1e-6))
                    {
                        continue;
                    }

                    // Follow the waypoints of the first gripper
                    target_waypoints_first_gripper = current_waypoints_first_gripper;
                    // Create new waypoints in a stright line for the second gripper
                    target_waypoints_second_gripper = createOtherGripperWaypoints(
                                target_waypoints_first_gripper,
                                second_gripper_start_pos,
                                second_gripper_end_pos);
                }

                // Make a guess about the number of nodes we'll end up using
                assert(target_waypoints_first_gripper.size() == target_waypoints_second_gripper.size());
                const size_t num_waypoints = target_waypoints_first_gripper.size();

                // Now that we have the waypoints, start building the smoothed path, exiting early if we encouter an infeasible configuration
                smoothed_segment.push_back(smoothing_start_config);
                for (size_t waypoint_idx = 1; waypoint_idx < num_waypoints; ++waypoint_idx)
                {
                    RRTGrippersRepresentation target_poses = path.front().getGrippers();
                    target_poses.first.translation() = target_waypoints_first_gripper[waypoint_idx];
                    target_poses.second.translation() = target_waypoints_second_gripper[waypoint_idx];
                    const RRTNode forward_prop_target_config(
                                target_poses,
                                RRTRobotRepresentation(),
                                path.front().getBand());

                    const int64_t start_idx = (int64_t)smoothed_segment.size() - 1;
                    forwardPropogationFunction(smoothed_segment, start_idx, forward_prop_target_config, fwd_prop_extend_band, fwd_prop_max_steps, fwd_prop_local_visualization_enabled);
                }
            }
            else
            {
                assert(false && "Smoothing type was something other than [1, 4], this ougth to be impossible");
            }

            // Check if the rubber band gets overstretched while propogating the grippers on the new path
            const auto& target_gripper_position = smoothing_end_config.getGrippers();
            const auto& last_gripper_position = smoothed_segment.back().getGrippers();
            if (!gripperPositionsAreApproximatelyEqual(last_gripper_position, target_gripper_position))
            {
                ++failed_iterations;
                continue;
            }

            // We still need to check that the rubber band can still reach the goal correctly from this state,
            // so we'll forward propogate along the rest of the trajectory to check feasibility
            end_of_smoothing_to_goal_results = forwardSimulateGrippersPath(path, smoothing_end_index, *smoothed_segment.back().getBand());
        }

        const bool final_band_at_goal_success = end_of_smoothing_to_goal_results.first;
        const auto& end_of_smoothing_to_goal_path_ = end_of_smoothing_to_goal_results.second;

        // Check if the rubber band gets overstretched or ends up in a blacklisted first order
        // homotopy class while following the tail of the starting trajectory
        {
            const auto& final_node_of_smoothing = end_of_smoothing_to_goal_path_.back();
            const bool final_band_visible_to_blacklist = isBandFirstOrderVisibileToBlacklist(*final_node_of_smoothing.getBand());
            if (!final_band_at_goal_success || final_band_visible_to_blacklist)
            {
//                std::cout << "Shortcut failed, continuing "
//                          << "    Band at goal? " << final_band_at_goal_success
//                          << "    Band visible? " << final_band_visible_to_blacklist
//                          << "\n";
                ++failed_iterations;
                continue;
            }
        }

        ///////////////////// Smoothing success - Create the new smoothed path /////////////////////////////////////////
        {
//            std::cout << "Smoothing valid\n";

            // Allocate space for the total smoothed path
            std::vector<RRTNode, RRTAllocator> smoothed_path;
            smoothed_path.reserve((smoothing_start_index  + 1) + (smoothed_segment.size() - 1) + (end_of_smoothing_to_goal_path_.size() - 1));

            // Insert the starting unchanged part of the path
            smoothed_path.insert(smoothed_path.end(), path.begin(), path.begin() + smoothing_start_index + 1);

            // Insert the smoothed portion
            smoothed_path.insert(smoothed_path.end(), smoothed_segment.begin() + 1, smoothed_segment.end());

            // Insert the changed end of the path with the new rubber band - gripper/robot positions are identical
            smoothed_path.insert(smoothed_path.end(), end_of_smoothing_to_goal_path_.begin() + 1, end_of_smoothing_to_goal_path_.end());

            // Record the change and re-visualize
            path = smoothed_path;
        }

        if (visualization_enabled_globally_ && visualization_enabled_locally)
        {
            visualizePath(path);
        }
    }

    // Record the statistics and return the result
    const double smoothing_time = function_wide_stopwatch(READ);

    smoothing_statistics_["smoothing0_failed_iterations                            "] = (double)failed_iterations;
    smoothing_statistics_["smoothing1_iterations                                   "] = (double)num_iterations;
    smoothing_statistics_["smoothing2_forward_propogation_crrt_projection_time     "] = total_projection_time_;
    smoothing_statistics_["smoothing3_forward_propogation_collision_check_time     "] = total_collision_check_time_;
    smoothing_statistics_["smoothing4_forward_propogation_band_sim_time            "] = total_band_forward_propogation_time_;
    smoothing_statistics_["smoothing5_forward_propogation_first_order_vis_time     "] = total_first_order_vis_propogation_time_;
    smoothing_statistics_["smoothing6_forward_propogation_everything_included_time "] = total_everything_included_forward_propogation_time_;
    smoothing_statistics_["smoothing7_total_time                                   "] = smoothing_time;

    return path;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Visualization and other debugging tools
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Draws lines connecting all nodes in the tree, from start_idx through to the end of the vector
void RRTHelper::visualizeTree(
        const std::vector<RRTNode, RRTAllocator>& tree,
        const size_t start_idx,
        const std::string ns_a,
        const std::string ns_b,
        const std::string ns_band,
        const int id_a,
        const int id_b,
        const int id_band,
        const std_msgs::ColorRGBA& color_a,
        const std_msgs::ColorRGBA& color_b,
        const std_msgs::ColorRGBA& color_band,
        const bool draw_band) const
{
    if (visualization_enabled_globally_)
    {
        assert(start_idx < tree.size());

        EigenHelpers::VectorVector3d band_line_start_points;
        EigenHelpers::VectorVector3d band_line_end_points;

        EigenHelpers::VectorVector3d gripper_a_tree_start_points;
        EigenHelpers::VectorVector3d gripper_a_tree_end_points;
        EigenHelpers::VectorVector3d gripper_b_tree_start_points;
        EigenHelpers::VectorVector3d gripper_b_tree_end_points;

        gripper_a_tree_start_points.reserve(tree.size() - start_idx);
        gripper_b_tree_start_points.reserve(tree.size() - start_idx);
        gripper_a_tree_end_points.reserve(tree.size() - start_idx);
        gripper_b_tree_end_points.reserve(tree.size() - start_idx);

        for (size_t idx = start_idx; idx < tree.size(); ++idx)
        {
            const RRTNode& curr = tree[idx];

            if (draw_band)
            {
                const EigenHelpers::VectorVector3d& band_vec = curr.getBand()->getVectorRepresentation();
                for (size_t band_idx = 0; band_idx + 1 < band_vec.size(); ++band_idx)
                {
                    band_line_start_points.push_back(band_vec[band_idx]);
                    band_line_end_points.push_back(band_vec[band_idx + 1]);
                }
            }

            if (curr.getParentIndex() >= 0)
            {
                const RRTNode& parent = tree[curr.getParentIndex()];

                // Add edges from the parent to the current node
                gripper_a_tree_start_points.push_back(parent.getGrippers().first.translation());
                gripper_b_tree_start_points.push_back(parent.getGrippers().second.translation());

                gripper_a_tree_end_points.push_back(curr.getGrippers().first.translation());
                gripper_b_tree_end_points.push_back(curr.getGrippers().second.translation());
            }
        }

        vis_->visualizeLines(ns_a, gripper_a_tree_start_points, gripper_a_tree_end_points, color_a, id_a);
//        vis_->visualizeLines(ns_a, gripper_a_tree_start_points, gripper_a_tree_end_points, color_a, id_a);
//        vis_->visualizeLines(ns_a, gripper_a_tree_start_points, gripper_a_tree_end_points, color_a, id_a);
//        ros::spinOnce();
//        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
        vis_->visualizeLines(ns_b, gripper_b_tree_start_points, gripper_b_tree_end_points, color_b, id_b);
//        vis_->visualizeLines(ns_b, gripper_b_tree_start_points, gripper_b_tree_end_points, color_b, id_b);
//        vis_->visualizeLines(ns_b, gripper_b_tree_start_points, gripper_b_tree_end_points, color_b, id_b);
//        ros::spinOnce();
//        std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
        if (draw_band)
        {
            vis_->visualizeLines(ns_band, band_line_start_points, band_line_end_points, color_band, id_band);
//            vis_->visualizeLines(ns_band, band_line_start_points, band_line_end_points, color_band, id_band);
//            vis_->visualizeLines(ns_band, band_line_start_points, band_line_end_points, color_band, id_band);
//            ros::spinOnce();
//            std::this_thread::sleep_for(std::chrono::duration<double>(0.001));
        }
    }
}

void RRTHelper::visualizePath(const std::vector<RRTNode, RRTAllocator>& path) const
{
    if (visualization_enabled_globally_)
    {
        EigenHelpers::VectorVector3d gripper_a_cubes;
        EigenHelpers::VectorVector3d gripper_b_cubes;
        gripper_a_cubes.reserve(path.size());
        gripper_b_cubes.reserve(path.size());

        EigenHelpers::VectorVector3d line_start_points;
        EigenHelpers::VectorVector3d line_end_points;

        for (int32_t ind = 0; ind < (int32_t)path.size(); ++ind)
        {
            const RRTNode& config = path[ind];
            const RRTGrippersRepresentation& gripper_positions = config.getGrippers();
            const RubberBand::Ptr& rubber_band = config.getBand();

            gripper_a_cubes.push_back(gripper_positions.first.translation());
            gripper_b_cubes.push_back(gripper_positions.second.translation());

            const EigenHelpers::VectorVector3d band_vec = rubber_band->getVectorRepresentation();
            for (size_t band_idx = 0; band_idx + 1 < band_vec.size(); ++band_idx)
            {
                line_start_points.push_back(band_vec[band_idx]);
                line_end_points.push_back(band_vec[band_idx + 1]);
            }
        }

        vis_->visualizeCubes(RRT_SOLUTION_GRIPPER_A_NS, gripper_a_cubes, Vector3d(0.005, 0.005, 0.005), gripper_a_forward_tree_color_, 1);
        vis_->visualizeCubes(RRT_SOLUTION_GRIPPER_B_NS, gripper_b_cubes, Vector3d(0.005, 0.005, 0.005), gripper_b_forward_tree_color_, 1);
        vis_->visualizeLines(RRT_SOLUTION_RUBBER_BAND_NS, line_start_points, line_end_points, Visualizer::Yellow(), 1);
    }
}

void RRTHelper::visualizeBlacklist() const
{
    EigenHelpers::VectorVector3d line_start_points;
    EigenHelpers::VectorVector3d line_end_points;

    for (size_t idx = 0; idx < blacklisted_goal_rubber_bands_.size(); ++idx)
    {
        const auto& band = blacklisted_goal_rubber_bands_[idx];
        for (size_t band_idx = 1; band_idx < band.size(); ++band_idx)
        {
            line_start_points.push_back(band[band_idx - 1]);
            line_end_points.push_back(band[band_idx]);
        }
    }

    vis_->visualizeLines(RRT_BLACKLISTED_GOAL_BANDS_NS, line_start_points, line_end_points, Visualizer::Red(), 1, 0.01);
}

void RRTHelper::storePath(const std::vector<RRTNode, RRTAllocator>& path) const
{
    try
    {
        const auto log_folder = ROSHelpers::GetParamRequiredDebugLog<std::string>(nh_, "log_folder", __func__);
        if (!log_folder.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load log_folder from parameter server");
        }
        arc_utilities::CreateDirectory(log_folder.GetImmutable());
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "path_file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load path_file_name_prefix from parameter server");
        }

        const std::string file_name_suffix = arc_helpers::GetCurrentTimeAsString();
        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix + ".compressed";
        const std::string full_path = log_folder.GetImmutable() + file_name;
        ROS_DEBUG_STREAM("Saving path to " << full_path);

        std::vector<uint8_t> buffer;
        arc_utilities::SerializeVector<RRTNode, RRTAllocator>(path, buffer, &RRTNode::Serialize);
        ZlibHelpers::CompressAndWriteToFile(buffer, full_path);

        // Verify no mistakes were made
        {
            const auto deserializer = [&] (const std::vector<uint8_t>& buffer, const uint64_t current)
            {
                return RRTNode::Deserialize(buffer, current, *starting_band_);
            };

            const std::vector<RRTNode, RRTAllocator> retrieved_path =
                    arc_utilities::DeserializeVector<RRTNode, RRTAllocator>(buffer, 0, deserializer).first;

            assert(retrieved_path == path);
        }
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to store path: "  <<  e.what());
    }
}

std::vector<RRTNode, RRTAllocator> RRTHelper::loadStoredPath() const
{
    try
    {
        const auto log_folder = ROSHelpers::GetParamRequired<std::string>(nh_, "log_folder", __func__);
        if (!log_folder.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load log_folder from parameter server");
        }
        const auto file_name_prefix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "path_file_name_prefix", __func__);
        if (!file_name_prefix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load path_file_name_prefix from parameter server");
        }
        const auto file_name_suffix = ROSHelpers::GetParamRequiredDebugLog<std::string>(ph_, "path_file_name_suffix_to_load", __func__);
        if (!file_name_suffix.Valid())
        {
            throw_arc_exception(std::invalid_argument, "Unable to load path_file_name_suffix_to_load from parameter server");
        }

        const std::string file_name = file_name_prefix.GetImmutable() + "__" + file_name_suffix.GetImmutable() + ".compressed";
        const std::string full_path = log_folder.GetImmutable() + file_name;
        ROS_INFO_STREAM("Loading path from " << full_path);

        const auto deserializer = [&] (const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return RRTNode::Deserialize(buffer, current, *starting_band_);
        };

        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(full_path);
        const auto path_deserialized = arc_utilities::DeserializeVector<RRTNode, RRTAllocator>(buffer, 0, deserializer);
        return path_deserialized.first;
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM("Failed to load stored path: "  <<  e.what());
    }

    return std::vector<RRTNode, RRTAllocator>();
}

bool RRTHelper::useStoredPath() const
{
    return ROSHelpers::GetParamRequired<bool>(ph_, "use_stored_path", __func__).GetImmutable();
}
