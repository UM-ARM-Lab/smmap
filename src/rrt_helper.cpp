#include "smmap/rrt_helper.h"
#include "smmap/task_framework.h"

#include <thread>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/timing.hpp>
#include <arc_utilities/filesystem.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/path_utils.hpp>

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;
using namespace Eigen;
using namespace EigenHelpers;

#warning "!!!!!!!!! Magic number !!!!!!!!!!!"
#define GRIPPER_TRANSLATION_IS_APPROX_DIST 0.001
//#define SMMAP_RRT_VERBOSE true
#define SMMAP_RRT_VERBOSE false


RRTRobotRepresentation RRTDistance::joint_weights_;

constexpr char RRTHelper::RRT_BLACKLISTED_GOAL_BANDS_NS[];
constexpr char RRTHelper::RRT_GOAL_TESTING_NS[];

constexpr char RRTHelper::RRT_FORWARD_TREE_GRIPPER_A_NS[];
constexpr char RRTHelper::RRT_FORWARD_TREE_GRIPPER_B_NS[];
constexpr char RRTHelper::RRT_BACKWARD_TREE_GRIPPER_A_NS[];
constexpr char RRTHelper::RRT_BACKWARD_TREE_GRIPPER_B_NS[];
constexpr char RRTHelper::RRT_TREE_BAND_NS[];

constexpr char RRTHelper::RRT_SAMPLE_NS[];
constexpr char RRTHelper::RRT_FORWARD_PROP_START_NS[];

constexpr char RRTHelper::RRT_SOLUTION_GRIPPER_A_NS[];
constexpr char RRTHelper::RRT_SOLUTION_GRIPPER_B_NS[];
constexpr char RRTHelper::RRT_SOLUTION_RUBBER_BAND_NS[];

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
    return (gripper_a_pos - gripper_b_pos).squaredNorm() > (max_dist * max_dist);
}

static inline bool maxGrippersDistanceViolated(
        const RRTGrippersRepresentation& grippers,
        const double max_dist)
{
    const auto& gripper_a_pos = grippers.first.translation();
    const auto& gripper_b_pos = grippers.second.translation();
    return (gripper_a_pos - gripper_b_pos).squaredNorm() > (max_dist * max_dist);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTConfig functions                      /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

RRTNode::RRTNode()
    : cost_to_come_(std::numeric_limits<double>::infinity())
    , parent_index_(-1)
    , child_indices_(0)
    , initialized_(false)
{}

RRTNode::RRTNode(
        const RRTGrippersRepresentation& grippers_poses,
        const RRTRobotRepresentation& robot_configuration,
        const RubberBand::Ptr& band)
    : grippers_poses_(grippers_poses)
    , robot_configuration_(robot_configuration)
    , band_(band)
    , cost_to_come_(0.0)
    , parent_index_(-1)
    , child_indices_(0)
    , initialized_(true)
{}

RRTNode::RRTNode(
        const RRTGrippersRepresentation& grippers_poses,
        const RRTRobotRepresentation& robot_configuration,
        const RubberBand::Ptr& band,
        const double cost_to_come,
        const int64_t parent_index)
    : grippers_poses_(grippers_poses)
    , robot_configuration_(robot_configuration)
    , band_(band)
    , cost_to_come_(cost_to_come)
    , parent_index_(parent_index)
    , child_indices_(0)
    , other_tree_target_indices_blacklist_(0)
    , initialized_(true)
{}

RRTNode::RRTNode(
        const RRTGrippersRepresentation& grippers_poses,
        const RRTRobotRepresentation& robot_configuration,
        const RubberBand::Ptr& band,
        const double cost_to_come,
        const int64_t parent_index,
        const std::vector<int64_t>& child_indices,
        const std::vector<int64_t>& other_tree_target_indices_blacklist)
    : grippers_poses_(grippers_poses)
    , robot_configuration_(robot_configuration)
    , band_(band)
    , cost_to_come_(cost_to_come)
    , parent_index_(parent_index)
    , child_indices_(child_indices)
    , other_tree_target_indices_blacklist_(other_tree_target_indices_blacklist)
    , initialized_(true)
{}

bool RRTNode::isInitialized() const
{
    return initialized_;
}

const RRTGrippersRepresentation& RRTNode::grippers() const
{
    return grippers_poses_;
}

const RRTRobotRepresentation& RRTNode::robotConfiguration() const
{
    return robot_configuration_;
}

const RubberBand::Ptr& RRTNode::band() const
{
    return band_;
}

double RRTNode::costToCome() const
{
    return cost_to_come_;
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


const std::vector<int64_t>& RRTNode::getOtherTreeBlacklistIndices() const
{
    return other_tree_target_indices_blacklist_;
}

void RRTNode::clearOtherTreeBlacklistIndices()
{
    other_tree_target_indices_blacklist_.clear();
}

void RRTNode::addOtherTreeBlacklistIndex(const int64_t blacklist_index)
{
    for (size_t idx = 0; idx < other_tree_target_indices_blacklist_.size(); idx++)
    {
        if (other_tree_target_indices_blacklist_[idx] == blacklist_index)
        {
            return;
        }
    }
    other_tree_target_indices_blacklist_.push_back(blacklist_index);
}

void RRTNode::removeOtherTreeBlacklistIndex(const int64_t blacklist_index)
{
    std::vector<int64_t> new_blacklist_indices;
    for (size_t idx = 0; idx < child_indices_.size(); idx++)
    {
        if (other_tree_target_indices_blacklist_[idx] != blacklist_index)
        {
            new_blacklist_indices.push_back(other_tree_target_indices_blacklist_[idx]);
        }
    }
    other_tree_target_indices_blacklist_ = new_blacklist_indices;
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

    if (cost_to_come_ != other.cost_to_come_)
    {
        return false;
    }

    if (parent_index_ != other.parent_index_)
    {
        return false;
    }

    if (child_indices_ != other.child_indices_)
    {
        return false;
    }

    if (other_tree_target_indices_blacklist_ != other.other_tree_target_indices_blacklist_)
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
    arc_utilities::SerializeFixedSizePOD<double>(cost_to_come_, buffer);
    arc_utilities::SerializeFixedSizePOD<int64_t>(parent_index_, buffer);
    arc_utilities::SerializeVector<int64_t>(child_indices_, buffer, arc_utilities::SerializeFixedSizePOD<int64_t>);
    arc_utilities::SerializeVector<int64_t>(other_tree_target_indices_blacklist_, buffer, arc_utilities::SerializeFixedSizePOD<int64_t>);
    arc_utilities::SerializeFixedSizePOD<uint8_t>((uint8_t)initialized_, buffer);

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

    // Deserialize the cost to come
    const auto cost_to_come_deserialized = arc_utilities::DeserializeFixedSizePOD<double>(buffer, current_position);
    current_position += cost_to_come_deserialized.second;

    // Deserialize the parent index
    const auto parent_index_deserialized = arc_utilities::DeserializeFixedSizePOD<int64_t>(buffer, current_position);
    current_position += parent_index_deserialized.second;

    // Deserialize the child indices
    const auto child_indices_deserialized = arc_utilities::DeserializeVector<int64_t>(buffer, current_position, &arc_utilities::DeserializeFixedSizePOD<int64_t>);
    current_position += child_indices_deserialized.second;

    // Deserialize the blacklisted indices
    const auto blacklisted_indices_deserialized = arc_utilities::DeserializeVector<int64_t>(buffer, current_position, &arc_utilities::DeserializeFixedSizePOD<int64_t>);
    current_position += blacklisted_indices_deserialized.second;

    // Deserialize the initialized
    const auto initialized_deserialized = arc_utilities::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
    current_position += initialized_deserialized.second;

    // Build the resulting node
    RRTNode deserialized(
                grippers_poses_deserialized.first,
                robot_configuration_deserialized.first,
                band,
                cost_to_come_deserialized.first,
                parent_index_deserialized.first,
                child_indices_deserialized.first,
                blacklisted_indices_deserialized.first);
    deserialized.initialized_ = (bool)initialized_deserialized.first;

    const uint64_t bytes_read = current_position - current;
    return std::make_pair(deserialized, bytes_read);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////           RRTDistance functions                    /////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const RRTRobotRepresentation& RRTDistance::GetJointWeights()
{
    return joint_weights_;
}

void RRTDistance::SetJointWeights(const RRTRobotRepresentation joint_weights)
{
    joint_weights_ = joint_weights;
}

// Note that this does only translational distance
double RRTDistance::DistanceSquared(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2)
{
    const auto& c1_first_gripper     = c1.first.translation();
    const auto& c1_second_gripper    = c1.second.translation();
    const auto& c2_first_gripper     = c2.first.translation();
    const auto& c2_second_gripper    = c2.second.translation();
    return (c1_first_gripper - c2_first_gripper).squaredNorm() +
            (c1_second_gripper - c2_second_gripper).squaredNorm();
}

double RRTDistance::Distance(const RRTGrippersRepresentation& c1, const RRTGrippersRepresentation& c2)
{
    return std::sqrt(DistanceSquared(c1, c2));
}

double RRTDistance::DistanceSquared(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2)
{
    const auto delta = r1 - r2;
    return (delta.cwiseProduct(joint_weights_)).squaredNorm();
}

double RRTDistance::Distance(const RRTRobotRepresentation& r1, const RRTRobotRepresentation& r2)
{
    return std::sqrt(DistanceSquared(r1, r2));
}

// Only calculates the distance travelled by the grippers, not the entire band
double RRTDistance::GrippersPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());
    double path_distance = 0;
    for (size_t idx = start_index; idx < end_index; ++idx)
    {
        path_distance += Distance(path[idx].grippers(), path[idx + 1].grippers());
    }
    return path_distance;
}

double RRTDistance::RobotPathDistance(const std::vector<RRTNode, RRTAllocator>& path, const size_t start_index, const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());
    double path_distance = 0;
    for (size_t idx = start_index; idx < end_index; ++idx)
    {
        path_distance += Distance(path[idx].robotConfiguration(), path[idx + 1].robotConfiguration());
    }
    return path_distance;
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
        const bool planning_for_whole_robot,
        const sdf_tools::SignedDistanceField::ConstPtr environment_sdf,
        const XYZGrid& work_space_grid,
        const std::shared_ptr<std::mt19937_64>& generator,
        // Planning algorithm parameters
        const bool using_cbirrt_style_projection,
        const size_t forward_tree_extend_iterations,
        const size_t backward_tree_extend_iterations,
        const size_t kd_tree_grow_threshold,
        const bool use_brute_force_nn,
        const double goal_bias,
        const double best_near_radius,
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
        const double band_distance2_scaling_factor,
        const size_t band_max_points,
        // Visualization
        const smmap_utilities::Visualizer::Ptr vis,
        const bool visualization_enabled)
    : nh_(nh)
    , ph_(ph.getNamespace() + "/rrt")
    , robot_(robot)
    , planning_for_whole_robot_(planning_for_whole_robot)
    , sdf_(environment_sdf)
    , work_space_grid_(work_space_grid)

    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)

    , task_aligned_frame_transform_(task_aligned_frame)
    , task_aligned_frame_inverse_transform_(task_aligned_frame_transform_.inverse())
    , task_aligned_lower_limits_(task_aligned_lower_limits)
    , task_aligned_upper_limits_(task_aligned_upper_limits)

    , robot_joint_lower_limits_(robot_->getJointLowerLimits())
    , robot_joint_upper_limits_(robot_->getJointUpperLimits())
    , robot_joint_weights_(robot_->getJointWeights())
    , total_dof_(robot_joint_weights_.size())

    , max_gripper_step_size_(max_gripper_step_size)
    , max_robot_dof_step_size_(max_robot_dof_step_size)
    , min_robot_dof_step_size_(min_robot_dof_step_size)
    , max_gripper_rotation_(max_gripper_rotation)
    , goal_bias_(goal_bias)
    , goal_reach_radius_(goal_reach_radius)
    , gripper_min_distance_to_obstacles_(gripper_min_distance_to_obstacles)

    , band_distance2_scaling_factor_(band_distance2_scaling_factor)
    , band_max_points_(band_max_points)
    , band_max_dist2_(band_distance2_scaling_factor_ * (double)band_max_points_ * (task_aligned_upper_limits - task_aligned_lower_limits).squaredNorm())

    , using_cbirrt_style_projection_(using_cbirrt_style_projection)
    , forward_tree_extend_iterations_(forward_tree_extend_iterations)
    , backward_tree_extend_iterations_(backward_tree_extend_iterations)
    , use_brute_force_nn_(use_brute_force_nn)
    , kd_tree_grow_threshold_(kd_tree_grow_threshold)
    , best_near_radius2_(best_near_radius * best_near_radius)

    , max_shortcut_index_distance_(max_shortcut_index_distance)
    , max_smoothing_iterations_(max_smoothing_iterations)
    , max_failed_smoothing_iterations_(max_failed_smoothing_iterations)
    , uniform_shortcut_smoothing_int_distribution_(1, 4)

    , forward_nn_raw_data_(0)
    , backward_nn_raw_data_(0)
    , forward_nn_index_(nullptr)
    , backward_nn_index_(nullptr)
    , forward_next_idx_to_add_to_nn_dataset_(0)
    , backward_next_idx_to_add_to_nn_dataset_(0)

    , total_sampling_time_(NAN)
    , total_nearest_neighbour_index_building_time_(NAN)
    , total_nearest_neighbour_index_searching_time_(NAN)
    , total_nearest_neighbour_linear_searching_time_(NAN)
    , total_nearest_neighbour_radius_searching_time_(NAN)
    , total_nearest_neighbour_best_searching_time_(NAN)
    , total_nearest_neighbour_time_(NAN)
    , total_forward_kinematics_time_(NAN)
    , total_projection_time_(NAN)
    , total_collision_check_time_(NAN)
    , total_band_forward_propogation_time_(NAN)
    , total_first_order_vis_propogation_time_(NAN)
    , total_everything_included_forward_propogation_time_(NAN)
    , forward_random_samples_useful_(0)
    , forward_random_samples_useless_(0)
    , backward_random_samples_useful_(0)
    , backward_random_samples_useless_(0)
    , forward_connection_attempts_useful_(0)
    , forward_connection_attempts_useless_(0)
    , forward_connections_made_(0)
    , backward_connection_attempts_useful_(0)
    , backward_connection_attempts_useless_(0)
    , backward_connections_made_(0)
    , path_found_(false)
    , goal_idx_in_forward_tree_(-1)
    , start_time_()

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
    assert(max_shortcut_index_distance_ > 0);
    assert(gripper_min_distance_to_obstacles_ > 0.0);

    RRTDistance::SetJointWeights(robot_joint_weights_);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for external RRT planning class
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t RRTHelper::rebuildNNIndex(
        NNIndexType& index,
        std::vector<float>& nn_raw_data,
        const std::vector<RRTNode, RRTAllocator>& tree,
        const size_t new_data_start_idx)
{
    // These pointers are used to check if we need to rebuild the whole tree because the data moved,
    // or if we can just add points
    const float* initial_data_pointer = nn_raw_data.data();
    nn_raw_data.resize(total_dof_ * tree.size());
    const float* final_data_pointer = nn_raw_data.data();

    for (size_t idx = new_data_start_idx; idx < tree.size(); ++idx)
    {
        const RRTRobotRepresentation& robot_config = tree[idx].robotConfiguration();
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

std::pair<int64_t, double> getNearest(
        const RRTRobotRepresentation& robot_config,
        const NNIndexType& index)
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

std::pair<int64_t, double> getNearest(
        const RRTRobotRepresentation& robot_config,
        const std::vector<RRTNode, RRTAllocator>& tree,
        const size_t start_idx)
{
    std::pair<int64_t, double> nearest(-1, std::numeric_limits<double>::infinity());

    for (size_t idx = start_idx; idx < tree.size(); idx++)
    {
        const RRTNode& test_node = tree[idx];
        const double distance2 = RRTDistance::DistanceSquared(test_node.robotConfiguration(), robot_config);

        if (nearest.second > distance2)
        {
            nearest.first = (int64_t)idx;
            nearest.second = distance2;
        }
    }

    return nearest;
}

std::vector<std::pair<int64_t, double>> radiusSearch(
        const RRTRobotRepresentation& robot_config,
        const NNIndexType& index,
        const double radius2)
{
    std::vector<std::pair<int64_t, double>> near;

    VectorXf robot_config_float = robot_config.cast<float>();
    flann::Matrix<float> query(robot_config_float.data(), 1, robot_config.size());

    std::vector<std::vector<size_t>> indices(query.rows);
    std::vector<std::vector<float>> dists(query.rows);

    const float eps = 0.0;
    flann::SearchParams params(flann::flann_checks_t::FLANN_CHECKS_UNLIMITED, eps);
    index.radiusSearch(query, indices, dists, (float)radius2, params);

    assert(indices[0].size() == dists[0].size());
    near.reserve(indices[0].size());
    for (size_t idx = 0; idx < indices[0].size(); ++idx)
    {
        near.push_back({indices[0][idx], dists[0][idx]});
    }

    return near;
}

std::vector<std::pair<int64_t, double>> radiusSearch(
        const RRTRobotRepresentation& robot_config,
        const std::vector<RRTNode, RRTAllocator>& tree,
        const size_t start_idx,
        const double radius2)
{
    std::vector<std::pair<int64_t, double>> near;

    for (size_t idx = start_idx; idx < tree.size(); idx++)
    {
        const RRTNode& test_node = tree[idx];
        const double distance2 = RRTDistance::DistanceSquared(test_node.robotConfiguration(), robot_config);

        if (distance2 <= radius2)
        {
            near.push_back({idx, distance2});
        }
    }

    return near;
}

std::pair<int64_t, double> getNearestFullConfig(
        const RRTNode& config,
        const std::vector<RRTNode, RRTAllocator>& tree,
        const double band_distance2_scaling_factor_,
        const std::vector<std::pair<int64_t, double>>& radius_search_set_1,
        const std::vector<std::pair<int64_t, double>>& radius_search_set_2)
{
    assert(radius_search_set_1.size() + radius_search_set_2.size() > 0);

    std::pair<int64_t, double> nearest(-1, std::numeric_limits<double>::infinity());

    const VectorVector3d& config_band_path = config.band()->getVectorRepresentation();
    const VectorXd config_band_path_vec = VectorEigenVectorToEigenVectorX(config_band_path);

    // Search through the first set of potential nearest nodes
    for (const auto& item : radius_search_set_1)
    {
        const auto& test_band = tree[item.first].band();
        const auto test_band_path = test_band->upsampleBand(config_band_path.size());
        const auto test_band_path_vec = VectorEigenVectorToEigenVectorX(test_band_path);

        const double band_distance2 = (config_band_path_vec - test_band_path_vec).squaredNorm();
        const double total_distance2 = item.second + band_distance2_scaling_factor_ * band_distance2;
        if (total_distance2 < nearest.second)
        {
            nearest.first = item.first;
            nearest.second = total_distance2;
        }
    }

    // Search through the second set of potential nearest nodes
    for (const auto& item : radius_search_set_2)
    {
        const auto& test_band = tree[item.first].band();
        const auto test_band_path = test_band->upsampleBand(config_band_path.size());
        const auto test_band_path_vec = VectorEigenVectorToEigenVectorX(test_band_path);

        const double band_distance2 = (config_band_path_vec - test_band_path_vec).squaredNorm();
        const double total_distance2 = item.second + band_distance2_scaling_factor_ * band_distance2;
        if (total_distance2 < nearest.second)
        {
            nearest.first = item.first;
            nearest.second = total_distance2;
        }
    }

    return nearest;
}

std::pair<int64_t, double> getBestFullConfig(
        const std::vector<RRTNode, RRTAllocator>& tree,
        const std::vector<std::pair<int64_t, double>>& radius_search_set_1,
        const std::vector<std::pair<int64_t, double>>& radius_search_set_2)
{
    assert(radius_search_set_1.size() + radius_search_set_2.size() > 0);

    double min_cost = std::numeric_limits<double>::infinity();
    int64_t min_idx = -1;

    // Search through the first set of potential best nodes
    for (const auto& item : radius_search_set_1)
    {
        const auto test_cost = tree[item.first].costToCome();
        if (test_cost < min_cost)
        {
            min_idx = item.first;
            min_cost = test_cost;
        }
    }

    // Search through the second set of potential nearest nodes
    for (const auto& item : radius_search_set_2)
    {
        const auto test_cost = tree[item.first].costToCome();
        if (test_cost < min_cost)
        {
            min_idx = item.first;
            min_cost = test_cost;
        }
    }

    assert(min_idx >= 0);
    return {min_idx, min_cost};
}


int64_t RRTHelper::nearestNeighbour(
        const bool use_forward_tree,
        const RRTNode& config)
{
    Stopwatch stopwatch;

    arc_helpers::DoNotOptimize(config);
    int64_t nn_idx = -1;
    if (use_forward_tree)
    {
        nn_idx = nearestBestNeighbourFullSpace(config);
    }
    else
    {
        nn_idx = nearestNeighbourRobotSpace(use_forward_tree, config).first;
    }
    arc_helpers::DoNotOptimize(nn_idx);

    const double nn_time = stopwatch(READ);
    total_nearest_neighbour_time_ += nn_time;

    assert(nn_idx >= 0);
    return nn_idx;
}

std::pair<int64_t, double> RRTHelper::nearestNeighbourRobotSpace(
        const bool use_forward_tree,
        const RRTNode& config)
{
    std::vector<RRTNode, RRTAllocator>* tree = nullptr;
    std::shared_ptr<NNIndexType> nn_index = nullptr;
    std::vector<float>* nn_raw_data = nullptr;
    size_t* manual_search_start_idx = nullptr;

    if (use_forward_tree)
    {
//        ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Using forward tree");
        tree = &forward_tree_;
        nn_index = forward_nn_index_;
        nn_raw_data = &forward_nn_raw_data_;
        manual_search_start_idx = &forward_next_idx_to_add_to_nn_dataset_;
    }
    else
    {
//        ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Using backward tree");
        tree = &backward_tree_;
        nn_index = backward_nn_index_;
        nn_raw_data = &backward_nn_raw_data_;
        manual_search_start_idx = &backward_next_idx_to_add_to_nn_dataset_;
    }

    ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Querry:                    " << config.robotConfiguration().transpose());

    // Check if we should rebuild the NN Index
    if (!use_brute_force_nn_ &&
        *manual_search_start_idx + kd_tree_grow_threshold_ <= tree->size())
    {
        ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Rebuilding FLANN index; forward tree? " << use_forward_tree);
        ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "RRT tree size: " << tree->size());
        ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Initial manual search start idx: " << manual_search_start_idx);
        ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Initial FLANN index size: " << nn_index->size());

        Stopwatch stopwatch;
        arc_helpers::DoNotOptimize(*manual_search_start_idx);
        *manual_search_start_idx = rebuildNNIndex(*nn_index, *nn_raw_data, *tree, *manual_search_start_idx);
        arc_helpers::DoNotOptimize(*manual_search_start_idx);
        const double index_building_time = stopwatch(READ);
        total_nearest_neighbour_index_building_time_ += index_building_time;

        ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Final manual search start idx: " << manual_search_start_idx);
        ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.nn", "Final FLANN index size: " << nn_index->size());
    }

    // If we have a FLANN index to search
    std::pair<int64_t, double> nearest(-1, std::numeric_limits<double>::infinity());
    if (!use_brute_force_nn_ && *manual_search_start_idx > 0)
    {
        Stopwatch stopwatch;
        arc_helpers::DoNotOptimize(config);
        nearest = getNearest(config.robotConfiguration(), *nn_index);
        arc_helpers::DoNotOptimize(nearest);
        const double index_searching_time = stopwatch(READ);
        total_nearest_neighbour_index_searching_time_ += index_searching_time;
    }

    // If we have data that isn't in the FLANN index
    if (*manual_search_start_idx < tree->size())
    {
        Stopwatch stopwatch;
        arc_helpers::DoNotOptimize(*manual_search_start_idx);

        const std::pair<int64_t, double> linear_nearest = getNearest(config.robotConfiguration(), *tree, *manual_search_start_idx);
        if (linear_nearest.second < nearest.second)
        {
            nearest = linear_nearest;
        }
        arc_helpers::DoNotOptimize(nearest);
        const double linear_searching_time = stopwatch(READ);
        total_nearest_neighbour_linear_searching_time_ += linear_searching_time;
    }

    assert(nearest.first >= 0);
    return nearest;
}

int64_t RRTHelper::nearestBestNeighbourFullSpace(
        const RRTNode &config)
{
    const std::pair<int64_t, double> nearest_robot_space = nearestNeighbourRobotSpace(true, config);

    Stopwatch stopwatch;
    arc_helpers::DoNotOptimize(nearest_robot_space);

    // If we have a FLANN index to search
    std::vector<std::pair<int64_t, double>> flann_radius_result;
    if (!use_brute_force_nn_ && forward_next_idx_to_add_to_nn_dataset_ > 0)
    {
        flann_radius_result = radiusSearch(config.robotConfiguration(), *forward_nn_index_, nearest_robot_space.second + band_max_dist2_);
    }

    // If we have data that isn't in the FLANN index
    std::vector<std::pair<int64_t, double>> linear_radius_result;
    if (forward_next_idx_to_add_to_nn_dataset_ < forward_tree_.size())
    {
        linear_radius_result = radiusSearch(config.robotConfiguration(), forward_tree_, forward_next_idx_to_add_to_nn_dataset_, nearest_robot_space.second + band_max_dist2_);
    }

    // Search both sets of results for the nearest neighbour in the
    // full configuration space, including the band
    const std::pair<int64_t, double> nearest_full_space =
            getNearestFullConfig(config, forward_tree_, band_distance2_scaling_factor_, flann_radius_result, linear_radius_result);

    arc_helpers::DoNotOptimize(nearest_full_space);
    const double radius_searching_time = stopwatch(READ);
    total_nearest_neighbour_radius_searching_time_ += radius_searching_time;

    // Perform a "best" subsearch if needed
    if (nearest_full_space.second <= best_near_radius2_ + band_max_dist2_)
    {
        auto flann_best_near_radius_result = flann_radius_result;
        auto linear_best_near_radius_result = linear_radius_result;

        // If the radius search that we already did is too small, then do a new search
        //   Note that due to the dual layer buisness, we are bloating the radius by a small amount (max band distance).
        arc_helpers::DoNotOptimize(nearest_robot_space);
        stopwatch(RESET);
        if (nearest_robot_space.second < best_near_radius2_)
        {
            if (!use_brute_force_nn_ && forward_next_idx_to_add_to_nn_dataset_ > 0)
            {
                flann_best_near_radius_result = radiusSearch(config.robotConfiguration(), *forward_nn_index_, best_near_radius2_);
            }
            if (forward_next_idx_to_add_to_nn_dataset_ < forward_tree_.size())
            {
                linear_best_near_radius_result = radiusSearch(config.robotConfiguration(), forward_tree_, forward_next_idx_to_add_to_nn_dataset_, best_near_radius2_);
            }
        }

        const std::pair<int64_t, double> best_full_space = getBestFullConfig(forward_tree_, flann_best_near_radius_result, linear_best_near_radius_result);
        arc_helpers::DoNotOptimize(best_full_space);

        assert(best_full_space.first >= 0);
        return best_full_space.first;
    }
    else
    {
        assert(nearest_full_space.first >= 0);
        return nearest_full_space.first;
    }
}



RRTNode RRTHelper::configSampling(const bool sample_band)
{
    Stopwatch stopwatch;
    arc_helpers::DoNotOptimize(generator_);
    RRTNode sample;

    if (!planning_for_whole_robot_)
    {
        const auto gripper_poses = posPairSampling_internal();
        RRTRobotRepresentation robot_config(6);
        robot_config << gripper_poses.first.translation(), gripper_poses.second.translation();

        sample = RRTNode(
                    gripper_poses,
                    robot_config,
                    std::make_shared<RubberBand>(*starting_band_));
    }
    else
    {
        sample = RRTNode(
                    starting_grippers_poses_,
                    robotConfigPairSampling_internal(),
                    std::make_shared<RubberBand>(*starting_band_));
    }

    ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.sampling", "Random robot config: " << sample.robotConfiguration().transpose());

    if (sample_band)
    {
//        ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.sampling", "Sampling random band");
        sample.band()->overridePoints(bandSampling_internal());
    }


    arc_helpers::DoNotOptimize(sample);
    const double sampling_time = stopwatch(READ);
    total_sampling_time_ += sampling_time;
    return sample;
}

RRTGrippersRepresentation RRTHelper::posPairSampling_internal()
{
    Vector3d gripper_a_pos, gripper_b_pos;

    do
    {
        const double x1 = Interpolate(task_aligned_lower_limits_.x(), task_aligned_upper_limits_.x(), uniform_unit_distribution_(*generator_));
        const double y1 = Interpolate(task_aligned_lower_limits_.y(), task_aligned_upper_limits_.y(), uniform_unit_distribution_(*generator_));
        const double z1 = Interpolate(task_aligned_lower_limits_.z(), task_aligned_upper_limits_.z(), uniform_unit_distribution_(*generator_));
        gripper_a_pos = Vector3d(x1, y1, z1);
    }
    while (sdf_->EstimateDistance3d(task_aligned_frame_transform_ * gripper_a_pos).first < gripper_min_distance_to_obstacles_);

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
        const double x2 = Interpolate(x2_min, x2_max, uniform_unit_distribution_(*generator_));
        const double y2 = Interpolate(y2_min, y2_max, uniform_unit_distribution_(*generator_));
        const double z2 = Interpolate(z2_min, z2_max, uniform_unit_distribution_(*generator_));
        gripper_b_pos = Vector3d(x2, y2, z2);
        valid = !maxGrippersDistanceViolated(gripper_a_pos, gripper_b_pos, max_grippers_distance_);
    }
    while (!valid || sdf_->EstimateDistance3d(task_aligned_frame_transform_ * gripper_b_pos).first < gripper_min_distance_to_obstacles_);

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
        rand_sample(idx) = Interpolate(robot_joint_lower_limits_(idx), robot_joint_upper_limits_(idx), uniform_unit_distribution_(*generator_));
    }
    return rand_sample;
}

VectorVector3d RRTHelper::bandSampling_internal()
{
    VectorVector3d band_points;
    band_points.reserve(band_max_points_);

    for (size_t idx = 0; idx < band_max_points_; ++idx)
    {
        const double x = Interpolate(task_aligned_lower_limits_.x(), task_aligned_upper_limits_.x(), uniform_unit_distribution_(*generator_));
        const double y = Interpolate(task_aligned_lower_limits_.y(), task_aligned_upper_limits_.y(), uniform_unit_distribution_(*generator_));
        const double z = Interpolate(task_aligned_lower_limits_.z(), task_aligned_upper_limits_.z(), uniform_unit_distribution_(*generator_));
        band_points.push_back(task_aligned_frame_transform_ * Vector3d(x, y, z));
    }

    return band_points;
}


bool RRTHelper::goalReached(const RRTNode& node)
{
    // Check if the grippers are close enough to the goal position
    if (RRTDistance::Distance(node.grippers(), grippers_goal_poses_) > goal_reach_radius_)
    {
        return false;
    }

    // Check if the grippers have been rotated too far
    if (planning_for_whole_robot_)
    {
        const double gripper_a_rotation_dist = Distance(starting_grippers_poses_.first.rotation(), node.grippers().first.rotation());
        const double gripper_b_rotation_dist = Distance(starting_grippers_poses_.second.rotation(), node.grippers().second.rotation());
        if (gripper_a_rotation_dist > max_gripper_rotation_ || gripper_b_rotation_dist > max_gripper_rotation_)
        {
            return false;
        }

        if (visualization_enabled_globally_)
        {
            vis_->visualizeLineStrip(RRT_GOAL_TESTING_NS, node.band()->getVectorRepresentation(), Visualizer::White(), 1, 0.01);
        }
    }

    // Only accept paths that are different from those on the blacklist
    if (isBandFirstOrderVisibileToBlacklist(*node.band()))
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
        const double gripper_a_rotation_dist = Distance(starting_grippers_poses_.first.rotation(), poses[0].rotation());
        const double gripper_b_rotation_dist = Distance(starting_grippers_poses_.second.rotation(), poses[1].rotation());
        if (gripper_a_rotation_dist > max_gripper_rotation_)
        {
            const Eigen::Quaterniond q_start(starting_grippers_poses_.first.rotation());
            const Eigen::Quaterniond q_end(poses[0].rotation());
            const Eigen::Quaterniond q_projected = Interpolate(q_start, q_end, max_gripper_rotation_ / gripper_a_rotation_dist);
            projected_poses[0].linear() = q_projected.toRotationMatrix();
//            projected_poses[0].rotation() = q_projected.toRotationMatrix();
        }
        if (gripper_b_rotation_dist > max_gripper_rotation_)
        {
            const Eigen::Quaterniond q_start(starting_grippers_poses_.second.rotation());
            const Eigen::Quaterniond q_end(poses[1].rotation());
            const Eigen::Quaterniond q_projected = Interpolate(q_start, q_end, max_gripper_rotation_ / gripper_b_rotation_dist);
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
                    {nearest_neighbour.grippers().first.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_a_forward_tree_color_,
                    1);
        vis_->visualizeCubes(
                    RRT_FORWARD_PROP_START_NS,
                    {nearest_neighbour.grippers().second.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_b_forward_tree_color_,
                    5);

        nearest_neighbour.band()->visualize(
                    RRT_FORWARD_PROP_START_NS,
                    Visualizer::Green(),
                    Visualizer::Green(),
                    10,
                    true);

        vis_->visualizeCubes(
                    RRT_SAMPLE_NS,
                    {target.grippers().first.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_a_forward_tree_color_,
                    1);
        vis_->visualizeCubes(
                    RRT_SAMPLE_NS,
                    {target.grippers().second.translation()},
                    Vector3d(0.01, 0.01, 0.01),
                    gripper_b_forward_tree_color_,
                    5);
    }

    const size_t visualization_period = 100;
    const size_t force_vis_publish_period = 10;

    const RRTGrippersRepresentation& starting_grippers_poses = nearest_neighbour.grippers();
    const RRTRobotRepresentation& starting_robot_config = nearest_neighbour.robotConfiguration();

    // Extract the target gripper pose and corresponding robot configuration
//    const RRTGrippersRepresentation& target_grippers_position = target.grippers();
    const RRTRobotRepresentation target_robot_config = target.robotConfiguration();

    if (planning_for_whole_robot_ && !using_cbirrt_style_projection_)
    {
        // Allocate space for potential children
        const double total_distance = RRTDistance::Distance(nearest_neighbour.robotConfiguration(), target_robot_config);
        const uint32_t max_total_steps = (uint32_t)ceil(total_distance / max_robot_dof_step_size_);
        tree_to_extend.reserve(tree_to_extend.size() + max_total_steps);

        int64_t parent_idx = nearest_neighbor_idx;
        uint32_t step_index = 0;
        while (step_index < max_total_steps)
        {
            // We could be updating the child indices of this node later, so take the value by non-const reference
            RRTNode& prev_node = tree_to_extend[parent_idx];
            const RubberBand::Ptr& prev_band = prev_node.band();

            // Interpolate in joint space to find the translation of the grippers
            const double ratio = std::min(1.0, (double)(step_index + 1) * max_robot_dof_step_size_ / total_distance);
            const RRTRobotRepresentation next_robot_config = Interpolate(starting_robot_config, target_robot_config, ratio);

            stopwatch(RESET);
            arc_helpers::DoNotOptimize(next_robot_config);
            robot_->setActiveDOFValues(next_robot_config);
            const AllGrippersSinglePose next_grippers_poses_vector = robot_->getGrippersPosesFunctionPointer();
            const RRTGrippersRepresentation next_grippers_poses(next_grippers_poses_vector[0], next_grippers_poses_vector[1]);
            arc_helpers::DoNotOptimize(next_grippers_poses);
            const double forward_kinematics_time = stopwatch(READ);
            total_forward_kinematics_time_ += forward_kinematics_time;

            // Check gripper position and rotation constraints
            {
                // Check if we rotated the grippers too much
                {
                    const double gripper_a_rotation_dist = Distance(starting_grippers_poses_.first.rotation(), next_grippers_poses.first.rotation());
                    const double gripper_b_rotation_dist = Distance(starting_grippers_poses_.second.rotation(), next_grippers_poses.second.rotation());
                    if (gripper_a_rotation_dist > max_gripper_rotation_ || gripper_b_rotation_dist > max_gripper_rotation_)
                    {
                        ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.prop", "Stopped due to excess gripper rotation");
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
                        ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.prop", "Stopped due to moving outside of planning arena");
                        break;
                    }
                }
            }

            // Collision checking
            {
                stopwatch(RESET);
                const bool in_collision = robot_->checkRobotCollision();
                const double collision_check_time = stopwatch(READ);
                total_collision_check_time_ += collision_check_time;
                if (in_collision)
                {
                    ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.prop", "Stopped due to collision");
                    break;
                }
            }

            RubberBand::Ptr next_band = std::make_shared<RubberBand>(*prev_band);
            if (extend_band)
            {
                //// Duplicated from pure gripper vesion below version below ////

                stopwatch(RESET);
                arc_helpers::DoNotOptimize(next_band);
                // Forward simulate the rubber band to test this transition
                next_band->forwardPropagateRubberBandToEndpointTargets(
                            next_grippers_poses.first.translation(),
                            next_grippers_poses.second.translation(),
                            rubber_band_verbose);
                arc_helpers::DoNotOptimize(next_band);
                const double band_forward_propogation_time = stopwatch(READ);
                total_band_forward_propogation_time_ += band_forward_propogation_time;

                // If we are still able to get to the next target position after retrying,
                // then return however far we were able to get
                if (!bandEndpointsMatchGripperPositions(*next_band, next_grippers_poses))
                {
                    ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.prop", "Stopped due to band endpoints not matching");
                    break;
                }

                // If the rubber band becomes overstretched, then return however far we were able to get
                if (next_band->isOverstretched())
                {
                    ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.prop", "Stopped due to band overstretch");
                    break;
                }
            }
            else
            {
                if (maxGrippersDistanceViolated(next_grippers_poses, max_grippers_distance_))
                {
                    ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.prop", "Stopped due to grippers moving too far apart");
                    break;
                }
            }

            // The new configuation is valid, add it to the tree
            const RRTRobotRepresentation& prev_robot_config = prev_node.robotConfiguration();
            const double additional_cost = RRTDistance::Distance(prev_robot_config, next_robot_config);
            const double next_cost_to_come = prev_node.costToCome() + additional_cost;

            const RRTNode next_node(next_grippers_poses, next_robot_config, next_band, next_cost_to_come, parent_idx);
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
        tree_to_extend.reserve(tree_to_extend.size() + max_projected_new_states);
        uint32_t step_index = 0;
        while (step_index < max_projected_new_states)
        {
            stopwatch(RESET);
            arc_helpers::DoNotOptimize(parent_idx);

            RRTNode& prev_node = tree_to_extend[parent_idx];
            const RubberBand::Ptr& prev_band = prev_node.band();
            const RRTRobotRepresentation& prev_robot_config = prev_node.robotConfiguration();

            const double prev_distance = RRTDistance::Distance(prev_robot_config, target_robot_config);
            const double ratio = std::min(1.0, max_gripper_step_size_ / prev_distance);
            const RRTRobotRepresentation next_robot_config_pre_projection = Interpolate(prev_robot_config, target_robot_config, ratio);

            stopwatch(RESET);
            arc_helpers::DoNotOptimize(next_robot_config_pre_projection);
            robot_->setActiveDOFValues(next_robot_config_pre_projection);
            const AllGrippersSinglePose next_grippers_poses_pre_projection = robot_->getGrippersPosesFunctionPointer();
            arc_helpers::DoNotOptimize(next_grippers_poses_pre_projection);
            const double forward_kinematics_time_part1 = stopwatch(READ);
            total_forward_kinematics_time_ += forward_kinematics_time_part1;


            // Project and check the projection result for failure
            const bool project_to_rotation_bound = true;
            const bool project_to_translation_bound = true;
            const auto next_robot_config_projection_result =
                    projectToValidConfig(
                        next_robot_config_pre_projection,
                        next_grippers_poses_pre_projection,
                        project_to_rotation_bound,
                        project_to_translation_bound);
            const RRTRobotRepresentation& next_robot_config = next_robot_config_projection_result.second;
            arc_helpers::DoNotOptimize(next_robot_config);
            const double projection_time = stopwatch(READ);
            total_projection_time_ += projection_time;

            // Check if the projection failed
            if (!next_robot_config_projection_result.first)
            {
                break;
            }

            // Check if we made any progress
            const double current_distance = RRTDistance::Distance(target_robot_config, next_robot_config);
            if (current_distance > prev_distance - min_robot_dof_step_size_ &&
                current_distance < prev_distance + min_robot_dof_step_size_)
            {
                break;
            }

            // Check the distance to the previous previous value as well (if it exists)
            if (step_index >= 1)
            {
                const RRTRobotRepresentation& ancient_robot_config = (tree_to_extend.end() - 2)->robotConfiguration();
                if (RRTDistance::Distance(ancient_robot_config, next_robot_config) < min_robot_dof_step_size_)
                {
                    break;
                }
            }

            stopwatch(RESET);
            arc_helpers::DoNotOptimize(next_robot_config);
            robot_->setActiveDOFValues(next_robot_config);
            const AllGrippersSinglePose next_grippers_poses_vector = robot_->getGrippersPosesFunctionPointer();
            const RRTGrippersRepresentation next_grippers_poses(next_grippers_poses_vector[0], next_grippers_poses_vector[1]);
            arc_helpers::DoNotOptimize(next_grippers_poses);
            const double forward_kinematics_time_part2 = stopwatch(READ);
            total_forward_kinematics_time_ += forward_kinematics_time_part2;

            // Check gripper position and rotation constraints if we did not project to them
            {
                // Check if we rotated the grippers too much
                if (!project_to_rotation_bound)
                {
                    const double gripper_a_rotation_dist = Distance(starting_grippers_poses_.first.rotation(), next_grippers_poses.first.rotation());
                    const double gripper_b_rotation_dist = Distance(starting_grippers_poses_.second.rotation(), next_grippers_poses.second.rotation());
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
                const bool in_collision = robot_->checkRobotCollision();
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
            const double additional_cost = RRTDistance::Distance(prev_robot_config, next_robot_config);
            const double next_cost_to_come = prev_node.costToCome() + additional_cost;

            const RRTNode next_node(next_grippers_poses, next_robot_config, next_band, next_cost_to_come, parent_idx);
            tree_to_extend.push_back(next_node);
            const int64_t new_node_idx = (int64_t)tree_to_extend.size() - 1;
            prev_node.addChildIndex(new_node_idx);

            parent_idx = new_node_idx;
            ++step_index;
        }
    }
    else
    {
        // Allocate space for potential children
        const double total_distance = RRTDistance::Distance(nearest_neighbour.grippers(), target.grippers());
        const uint32_t max_total_steps = (uint32_t)ceil(total_distance / max_gripper_step_size_);
        tree_to_extend.reserve(tree_to_extend.size() + max_total_steps);

        int64_t parent_idx = nearest_neighbor_idx;
        uint32_t step_index = 0;
        while (step_index < max_total_steps)
        {
            // We could be updating the child indices of this node later, so take the value by non-const reference
            RRTNode& prev_node = tree_to_extend[parent_idx];
            const RubberBand::Ptr& prev_band = prev_node.band();

            const double ratio = std::min(1.0, (double)(step_index + 1) * max_gripper_step_size_ / total_distance);
            const RRTRobotRepresentation next_robot_config =
                    Interpolate(starting_robot_config, target_robot_config, ratio);
            RRTGrippersRepresentation next_grippers_poses = starting_grippers_poses;
            next_grippers_poses.first.translation() = next_robot_config.head<3>();
            next_grippers_poses.second.translation() = next_robot_config.tail<3>();

            // Collision checking
            {
                // If the grippers collide with each other, then return however far we are able to get
                {
                    stopwatch(RESET);
                    arc_helpers::DoNotOptimize(next_grippers_poses);
                    const bool in_collision = (next_grippers_poses.first.translation() - next_grippers_poses.second.translation()).norm() < gripper_min_distance_to_obstacles_;
                    arc_helpers::DoNotOptimize(in_collision);
                    const double collision_check_time_pt1 = stopwatch(READ);
                    total_collision_check_time_ += collision_check_time_pt1;
                    if (in_collision)
                    {
                        break;
                    }
                }

                // If the grippers enter collision with the environment, then return however far we were able to get
                {
                    stopwatch(RESET);
                    arc_helpers::DoNotOptimize(next_grippers_poses);
                    const bool in_collision =
                            (sdf_->EstimateDistance3d(next_grippers_poses.first.translation()).first < gripper_min_distance_to_obstacles_) ||
                            (sdf_->EstimateDistance3d(next_grippers_poses.second.translation()).first < gripper_min_distance_to_obstacles_) ||
                            (sdf_->DistanceToBoundary3d(next_grippers_poses.first.translation()).first < gripper_min_distance_to_obstacles_) ||
                            (sdf_->DistanceToBoundary3d(next_grippers_poses.second.translation()).first < gripper_min_distance_to_obstacles_);
                    arc_helpers::DoNotOptimize(in_collision);
                    const double collision_check_time_pt2 = stopwatch(READ);
                    total_collision_check_time_ += collision_check_time_pt2;
                    if (in_collision)
                    {
                        break;
                    }
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
            const RRTRobotRepresentation& prev_robot_config = prev_node.robotConfiguration();
            const double additional_cost = RRTDistance::Distance(prev_robot_config, next_robot_config);
            const double next_cost_to_come = prev_node.costToCome() + additional_cost;

            const RRTNode next_node(next_grippers_poses, next_robot_config, next_band, next_cost_to_come, parent_idx);
            tree_to_extend.push_back(next_node);
            const int64_t new_node_idx = (int64_t)tree_to_extend.size() - 1;
            prev_node.addChildIndex(new_node_idx);

            parent_idx = new_node_idx;
            ++step_index;
        }
    }

    const size_t nodes_at_end_of_propogation = tree_to_extend.size();
    const size_t nodes_created = nodes_at_end_of_propogation - nodes_at_start_of_propogation;

    bool visualize = false;
    if (visualization_enabled_globally_ &&
        visualization_enabled_locally &&
        nodes_created > 0)
    {
        if (&tree_to_extend == &forward_tree_)
        {
            if (tree_to_extend.size() - forward_tree_next_visualized_node_ >= visualization_period)
            {
                visualize = true;
            }
        }
        else
        {
            if (tree_to_extend.size() - backward_tree_next_visualized_node_ >= visualization_period)
            {
                visualize = true;
            }
        }
    }

    if (visualize)
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

        const auto& tree_a_ns = (&tree_to_extend == &forward_tree_)
                ? RRT_FORWARD_TREE_GRIPPER_A_NS
                : RRT_BACKWARD_TREE_GRIPPER_A_NS;

        const auto& tree_b_ns = (&tree_to_extend == &forward_tree_)
                ? RRT_FORWARD_TREE_GRIPPER_B_NS
                : RRT_BACKWARD_TREE_GRIPPER_B_NS;

        const bool draw_band = extend_band;
        visualizeTree(
                    tree_to_extend,
                    starting_idx,
                    tree_a_ns,
                    tree_b_ns,
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

        if (tree_marker_id_ % force_vis_publish_period == 0)
        {
            vis_->forcePublishNow();
            vis_->purgeMarkerList();
        }
    }

    arc_helpers::DoNotOptimize(nodes_created);
    const double everything_included_forward_propogation_time = function_wide_stopwatch(READ);
    total_everything_included_forward_propogation_time_ += everything_included_forward_propogation_time;

    return nodes_created;
}



size_t RRTHelper::connectForwardTree(const RRTNode& target, const bool is_random)
{
    const bool extend_band = true;
    const bool nn_forward_tree = true;
    const size_t fwd_prop_max_steps = 32; // Only relevant if using constrained projection
    const bool fwd_prop_local_visualization_enabled = true;

    // Get the nearest neighbor
    forward_tree_nearest_neighbour_idx_ = nearestNeighbour(nn_forward_tree, target);
    // Forward propagate towards the sampled target
    const size_t num_random_nodes_created =
            forwardPropogationFunction(
                forward_tree_,
                forward_tree_nearest_neighbour_idx_,
                target,
                extend_band,
                fwd_prop_max_steps,
                fwd_prop_local_visualization_enabled);

    ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Nodes created: " << num_random_nodes_created << " Tree size: " << forward_tree_.size());

    if (SMMAP_RRT_VERBOSE)
    {
        for (size_t idx = forward_tree_.size() - num_random_nodes_created; idx < forward_tree_.size(); ++idx)
        {
            const RRTNode& node = forward_tree_[idx];
            ROS_INFO_STREAM_NAMED("rrt", "Node idx: " << idx << " Parent: " << node.getParentIndex() << " Config: " << node.robotConfiguration().transpose());
        }
    }

    // Record statistics for the randomly sampled extensions
    if (num_random_nodes_created != 0)
    {
        if (is_random)
        {
            ++forward_random_samples_useful_;
        }
        else
        {
            ++forward_connection_attempts_useful_;
        }
    }
    else
    {
        if (is_random)
        {
            ++forward_random_samples_useless_;
        }
        else
        {
            ++forward_connection_attempts_useless_;
        }
    }

    // Check if any of the new nodes reached the goal
    for (size_t idx = forward_tree_.size() - num_random_nodes_created; idx < forward_tree_.size(); ++idx)
    {
        const RRTNode& test_node = forward_tree_[idx];
        if (goalReached(test_node))
        {
            path_found_ = true;
            goal_idx_in_forward_tree_ = idx;
            ROS_INFO_NAMED("rrt", "Goal found durring 'connect' operation");
        }
    }

    return num_random_nodes_created;
}

size_t RRTHelper::connectBackwardTree(const RRTNode& target, const bool is_random)
{
    //////////////// Extend (connect) the backward tree towards a random target ////////////////
    const bool extend_band = false;
    const bool nn_forward_tree = false;
    const size_t fwd_prop_max_steps = 32; // Only relevant if using constrained projection
    const bool fwd_prop_local_visualization_enabled = true;

    // Get the nearest neighbor
    const int64_t backward_tree_nearest_neighbour_idx = nearestNeighbour(nn_forward_tree, target);
    // Forward propagate towards the sampled target
    const size_t num_nodes_created =
            forwardPropogationFunction(
                backward_tree_,
                backward_tree_nearest_neighbour_idx,
                target,
                extend_band,
                fwd_prop_max_steps,
                fwd_prop_local_visualization_enabled);

    ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Nodes created: " << num_nodes_created
                                                      << " Tree size: " << backward_tree_.size());

    for (size_t idx = backward_tree_.size() - num_nodes_created; idx < backward_tree_.size(); ++idx)
    {
        const RRTNode& node = backward_tree_[idx];
        ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Node idx: " << idx << " Parent: " << node.getParentIndex() << " Config: " << node.robotConfiguration().transpose());
    }

    // Record statistics for the randomly sampled extension
    if (num_nodes_created != 0)
    {
        if (is_random)
        {
            ++backward_random_samples_useful_;
        }
        else
        {
            ++backward_connection_attempts_useful_;
        }
    }
    else
    {
        if (is_random)
        {
            ++backward_random_samples_useless_;
        }
        else
        {
            ++backward_connection_attempts_useless_;
        }
    }

    return num_nodes_created;
}

size_t RRTHelper::connectForwardTreeToBackwardTreeBispace(const int64_t last_node_idx_in_forward_tree_branch)
{
    const bool extend_band = true;
    const bool nn_forward_tree = false;
    const size_t fwd_prop_max_steps = 32; // Only relevant if using constrained projection
    const bool fwd_prop_local_visualization_enabled = true;

    // Record the index of the last node in the new branch.
    // This is either the last item in the tree, or the nearest neighbour itself
    RRTNode& last_node_in_forward_tree = forward_tree_[last_node_idx_in_forward_tree_branch];

    const int64_t backward_tree_nearest_neighbour_idx = nearestNeighbour(nn_forward_tree, last_node_in_forward_tree);

    const auto& other_tree_blacklist = last_node_in_forward_tree.getOtherTreeBlacklistIndices();
    if (std::find(other_tree_blacklist.begin(), other_tree_blacklist.end(), backward_tree_nearest_neighbour_idx) != other_tree_blacklist.end())
    {
        // We've already expanded towards this node, so don't do so again
        return 0;
    }
    else
    {
        last_node_in_forward_tree.addOtherTreeBlacklistIndex(backward_tree_nearest_neighbour_idx);
    }

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
        ++forward_connection_attempts_useful_;
    }
    else
    {
        ++forward_connection_attempts_useless_;
    }

    // Check if any of the new nodes reached the goal
    for (size_t idx = forward_tree_.size() - num_goal_directed_nodes_created; idx < forward_tree_.size(); ++idx)
    {
        const RRTNode& test_node = forward_tree_[idx];
        if (goalReached(test_node))
        {
            path_found_ = true;
            goal_idx_in_forward_tree_ = idx;
            ROS_INFO_STREAM_NAMED("rrt", "Goal found via connection to backward tree: backward target idx: " << backward_tree_nearest_neighbour_idx);
        }
    }

    return num_goal_directed_nodes_created;
}

void RRTHelper::followBackwardTree(
        const size_t forward_tree_node_idx,
        const size_t backward_tree_node_idx)
{
    // March down the backward tree, propagating the band in the forward tree
    int64_t forward_parent_idx = forward_tree_node_idx;
    int64_t backward_next_idx = backward_tree_node_idx;
    while (!path_found_ && backward_next_idx >= 0)
    {
        const RRTNode& prev_node = forward_tree_[forward_parent_idx];
        const RubberBand::Ptr prev_band = prev_node.band();

        assert(backward_next_idx < (int64_t)backward_tree_.size());
        RRTNode& backward_next_node = backward_tree_[backward_next_idx];
        const RRTGrippersRepresentation& next_grippers_poses = backward_next_node.grippers();
        const RRTRobotRepresentation& next_robot_config = backward_next_node.robotConfiguration();

        RubberBand::Ptr next_band = std::make_shared<RubberBand>(*prev_band);
        Stopwatch stopwatch;
        const bool rubber_band_verbose = false;
        arc_helpers::DoNotOptimize(next_band);
        // Forward simulate the rubber band to test this transition
        next_band->forwardPropagateRubberBandToEndpointTargets(
                    next_grippers_poses.first.translation(),
                    next_grippers_poses.second.translation(),
                    rubber_band_verbose);
        arc_helpers::DoNotOptimize(next_band);
        const double band_forward_propogation_time = stopwatch(READ);
        total_band_forward_propogation_time_ += band_forward_propogation_time;

        // If we are still able to get to the next target position after retrying,
        // then return however far we were able to get
        if (!bandEndpointsMatchGripperPositions(*next_band, next_grippers_poses))
        {
            ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.bispace_follow", "Stopped due to band endpoints not matching");
            break;
        }

        // If the rubber band becomes overstretched, then return however far we were able to get
        if (next_band->isOverstretched())
        {
            ROS_INFO_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt.bispace_follow", "Stopped due to band overstretch");
            break;
        }

        if (visualization_enabled_globally_)
        {
            next_band->visualize("rrt_following_backward_tree_bands", Visualizer::Blue(), Visualizer::Blue(), (int32_t)backward_next_idx + 1, true);
        }

        // The new configuation is valid, add it to the forward tree
        const RRTRobotRepresentation& prev_robot_config = prev_node.robotConfiguration();
        const double additional_cost = RRTDistance::Distance(prev_robot_config, next_robot_config);
        const double next_cost_to_come = prev_node.costToCome() + additional_cost;

        const RRTNode next_node(
                    next_grippers_poses,
                    next_robot_config,
                    next_band,
                    next_cost_to_come,
                    forward_parent_idx);
        forward_tree_.push_back(next_node);
        const int64_t new_node_idx = (int64_t)forward_tree_.size() - 1;
        // Note that we do not use prev_node here as the push_back may have caused a reallocation
        forward_tree_[forward_parent_idx].addChildIndex(new_node_idx);

        forward_parent_idx = new_node_idx;
        backward_next_idx = backward_next_node.getParentIndex();

        if (goalReached(next_node))
        {
            path_found_ = true;
            goal_idx_in_forward_tree_ = new_node_idx;
            ROS_INFO_NAMED("rrt", "Goal found via following backward tree");
        }
    }
}



void RRTHelper::planningMainLoopBispace()
{
    // Make sure we've been given a start and goal state
    assert(forward_tree_.size() > 0);
    assert(backward_tree_.size() > 0);
    // Make sure the tree is properly linked
    assert(CheckTreeLinkage(forward_tree_));
    assert(CheckTreeLinkage(backward_tree_));

    // Plan
    ROS_INFO_NAMED("rrt", "Using bispace tree");
    std::chrono::duration<double> time_ellapsed = std::chrono::steady_clock::now() - start_time_;
    bool forward_iteration = true;
    while (!path_found_ && time_ellapsed < time_limit_)
    {
        if (forward_iteration)
        {
            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Starting forward iteration. Tree size: " << forward_tree_.size());
            for (size_t itr = 0; !path_found_ && itr < forward_tree_extend_iterations_ && time_ellapsed < time_limit_; ++itr)
            {
                ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Inner iteration # " << itr);

                //////////////// Extend (connect) the first tree towards a random target ////////////////
                const bool sample_band = true;
                const RRTNode random_target = configSampling(sample_band);
                const size_t num_random_nodes_created = connectForwardTree(random_target, true);
                if (path_found_)
                {
                    ROS_INFO_NAMED("rrt", "Goal found via random exploration");
                    break;
                }

                //////////////// Attempt to connect to the backward tree ////////////////////////////////
                const bool sample_goal = uniform_unit_distribution_(*generator_) < goal_bias_;
                if (sample_goal)// && num_random_nodes_created > 0)
                {
                    const int64_t last_node_idx_in_forward_tree_branch = num_random_nodes_created > 0 ?
                                (int64_t)forward_tree_.size() - 1 : forward_tree_nearest_neighbour_idx_;
                    const size_t num_goal_directed_nodes_created = connectForwardTreeToBackwardTreeBispace(last_node_idx_in_forward_tree_branch);
                    if (path_found_)
                    {
                        break;
                    }

                    //////////////// Check for a connection between the trees, extending the forward tree if possible ////////////////
                    if (num_goal_directed_nodes_created > 0)
                    {
                        const bool connection_made =
                                    robotConfigurationsAreApproximatelyEqual(
                                    forward_tree_.back().robotConfiguration(),
                                    backward_tree_.back().robotConfiguration());

                        if (connection_made)
                        {
                            // Record some statistics
                            ++forward_connections_made_;

                            // March down the backward tree, propagating the band in the forward tree
                            int64_t forward_parent_idx = (int64_t)forward_tree_.size() - 1;
                            int64_t backward_next_idx = (int64_t)backward_tree_.size() - 1;
                            followBackwardTree(forward_parent_idx, backward_next_idx);
                            if (path_found_)
                            {
                                break;
                            }
                        }
                    }
                }

                time_ellapsed = std::chrono::steady_clock::now() - start_time_;
            }

            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Ending forward iteration. Tree size: " << forward_tree_.size());
        }
        else
        {
            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Starting backward iteration. Tree size: " << backward_tree_.size());

            for (size_t itr = 0; itr < backward_tree_extend_iterations_ && time_ellapsed < time_limit_; ++itr)
            {
                ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Inner iteration # " << itr);
                const bool sample_band = false;
                const RRTNode random_target = configSampling(sample_band);
                connectBackwardTree(random_target, true);
                time_ellapsed = std::chrono::steady_clock::now() - start_time_;
            }

            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Ending backward iteration. Tree size: " << backward_tree_.size());
        }
//        forward_iteration = !forward_iteration;
    }
}





void RRTHelper::planningMainLoopBidirectional()
{
    // Make sure we've been given a start and goal state
    assert(forward_tree_.size() > 0);
    assert(backward_tree_.size() > 0);
    // Make sure the tree is properly linked
    assert(CheckTreeLinkage(forward_tree_));
    assert(CheckTreeLinkage(backward_tree_));

    // Plan
    ROS_INFO_NAMED("rrt", "Using bidirectional tree");
    std::chrono::duration<double> time_ellapsed = std::chrono::steady_clock::now() - start_time_;
    bool forward_iteration = true;
    while (!path_found_ && time_ellapsed < time_limit_)
    {
        size_t num_random_nodes_created = 0;
        size_t num_goal_directed_nodes_created = 0;

        if (forward_iteration)
        {
            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Starting forward iteration. Tree size: " << forward_tree_.size());

            //////////////// Extend (connect) the first tree towards a random target ////////////////
            const bool sample_band = true;
            const RRTNode random_target = configSampling(sample_band);
            num_random_nodes_created = connectForwardTree(random_target, true);
            if (path_found_)
            {
                ROS_INFO_NAMED("rrt", "Goal found via random exploration");
                break;
            }

            //////////////// Attempt to connect the backward tree to the last node created in the forward tree
            if (num_random_nodes_created > 0)
            {
//                const size_t forward_target_idx = (int64_t)std::uniform_int_distribution<size_t>(0, forward_tree_.size() - 1)(*generator_);
//                num_goal_directed_nodes_created = connectBackwardTree(forward_tree_[forward_target_idx], false);
//                num_goal_directed_nodes_created = connectBackwardTree(forward_tree_.back(), false);
                num_goal_directed_nodes_created = connectForwardTreeToBackwardTreeBispace(forward_tree_.size() - 1);
            }

            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Ending forward iteration. Tree size: " << forward_tree_.size());
        }
        else
        {
            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Starting backward iteration. Tree size: " << backward_tree_.size());

            const bool sample_band = false;
            const RRTNode random_target = configSampling(sample_band);
            num_random_nodes_created = connectBackwardTree(random_target, true);

            //////////////// Attempt to connect the forward tree to the last node created in the backward tree
            if (num_random_nodes_created > 0)
            {
                RRTNode target = backward_tree_.back();
                Stopwatch stopwatch;
                arc_helpers::DoNotOptimize(target);
                target.band()->overridePoints(bandSampling_internal());
                arc_helpers::DoNotOptimize(target);
                const double sampling_time = stopwatch(READ);
                total_sampling_time_ += sampling_time;

                num_goal_directed_nodes_created = connectForwardTree(target, false);
                if (path_found_)
                {
                    ROS_INFO_NAMED("rrt", "Goal found via connection to backward tree");
                    break;
                }
            }

            ROS_INFO_STREAM_COND_NAMED(SMMAP_RRT_VERBOSE, "rrt", "Ending backward iteration. Tree size: " << backward_tree_.size());
        }

        //////////////// Attempt to connect follow the backward tree if a connection was made
        if (num_random_nodes_created > 0 && num_goal_directed_nodes_created > 0)
        {
            const bool connection_made =
                        robotConfigurationsAreApproximatelyEqual(
                        forward_tree_.back().robotConfiguration(),
                        backward_tree_.back().robotConfiguration());

            if (connection_made)
            {
                // Record some statistics
                if (forward_iteration)
                {
                    ROS_INFO_COND_NAMED(true, "rrt", "Connection made between forward and backward tree: forward target was random, backward target was the last forward node");
                    ++backward_connections_made_;
                }
                else
                {
                    ROS_INFO_COND_NAMED(true, "rrt", "Connection made between forward and backward tree: backward target was random, forward target was the last backward node");
                    ++forward_connections_made_;
                }

                // March down the backward tree, propagating the band in the forward tree
                int64_t forward_parent_idx = (int64_t)forward_tree_.size() - 1;
                int64_t backward_next_idx = (int64_t)backward_tree_.size() - 1;

                visualizeBothTrees();
                followBackwardTree(forward_parent_idx, backward_next_idx);
                std::cout << "Waiting on char input" << std::endl;
                arc_helpers::Sleep(1.0);
                std::getchar();

                if (path_found_)
                {
                    ROS_INFO_NAMED("rrt", "Path found by following the backward tree");
                    break;
                }
            }
        }

        ROS_INFO_COND(SMMAP_RRT_VERBOSE, "");
        forward_iteration = !forward_iteration;
        time_ellapsed = std::chrono::steady_clock::now() - start_time_;
    }
}












/*
 * Builds the helper functions needed by simple_rrt_planner and invokes the planner (and shortcut smoother)
 */
std::vector<RRTNode, RRTAllocator> RRTHelper::plan(
        const RRTNode& start,
        const RRTGrippersRepresentation& grippers_goal_poses,
        const std::chrono::duration<double>& time_limit)
{
    const auto estimated_tree_size = ROSHelpers::GetParam(ph_, "estimated_tree_size", 100000);

    // Extract start information
    starting_band_ = std::make_shared<RubberBand>(*start.band());
    starting_robot_configuration_ = start.robotConfiguration();
    starting_grippers_poses_ = start.grippers();

    // Setup the forward tree
    forward_tree_.clear();
    forward_tree_.reserve(estimated_tree_size);
    forward_tree_.push_back(start);

    // Extract goal/termination information
    max_grippers_distance_ = starting_band_->maxSafeLength();
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

        robot_->setActiveDOFValues(goal_configurations[0]);
        const auto grippers_goal_poses_updated_vec = robot_->getGrippersPosesFunctionPointer();
        grippers_goal_poses_.first = grippers_goal_poses_updated_vec.at(0);
        grippers_goal_poses_.second = grippers_goal_poses_updated_vec.at(1);

        for (size_t idx = 0; idx < goal_configurations.size(); ++idx)
        {
            backward_tree_.push_back(
                        RRTNode(
                            grippers_goal_poses_,
                            goal_configurations[idx],
                            start.band()));
        }
    }
    else
    {
        grippers_goal_poses_ = grippers_goal_poses;
        RRTRobotRepresentation goal_configuration(6);
        goal_configuration << grippers_goal_poses_.first.translation(), grippers_goal_poses_.second.translation();
        backward_tree_.push_back(RRTNode(grippers_goal_poses_, goal_configuration, start.band()));
    }

    // Double check that the input goal location isn't immediately impossible
    const double first_gripper_dist_to_env = sdf_->EstimateDistance3d(grippers_goal_poses_.first.translation()).first;
    const double second_gripper_dist_to_env = sdf_->EstimateDistance3d(grippers_goal_poses_.second.translation()).first;
    if (first_gripper_dist_to_env < gripper_min_distance_to_obstacles_ ||
        second_gripper_dist_to_env < gripper_min_distance_to_obstacles_ ||
        (maxGrippersDistanceViolated(grippers_goal_poses_, max_grippers_distance_) > max_grippers_distance_))
    {
        const double dist_between_grippers = (grippers_goal_poses_.first.translation() - grippers_goal_poses_.second.translation()).norm();
        std::cerr << "Unfeasible goal location: " << grippers_goal_poses_.first.translation().transpose() << "  :  " << grippers_goal_poses_.second.translation().transpose() << std::endl;
        std::cerr << "Min gripper collision distance: " << gripper_min_distance_to_obstacles_ << " Current Distances: " << first_gripper_dist_to_env << " " << second_gripper_dist_to_env << std::endl;
        std::cerr << "Max allowable distance: " << max_grippers_distance_ << " Distance beteween goal grippers: " << dist_between_grippers << std::endl;

        vis_->visualizeGrippers("weird_gripper_goals", {grippers_goal_poses_.first, grippers_goal_poses_.second}, Visualizer::Red(), 1);
        std::getchar();

        assert(false && "Unfeasible goal location");
    }

    // Clear the forward tree flann data
    forward_nn_raw_data_.clear();
    forward_nn_raw_data_.reserve(total_dof_ * estimated_tree_size);
    forward_nn_index_ = std::make_shared<NNIndexType>(flann::KDTreeSingleIndexParams(), flann::L2_weighted<float>(robot_->getJointWeights()));
    forward_next_idx_to_add_to_nn_dataset_ = 0;

    // Clear the backward tree flann data
    backward_nn_raw_data_.clear();
    backward_nn_raw_data_.reserve(total_dof_ * estimated_tree_size);
    backward_nn_index_ = std::make_shared<NNIndexType>(flann::KDTreeSingleIndexParams(), flann::L2_weighted<float>(robot_->getJointWeights()));
    backward_next_idx_to_add_to_nn_dataset_ = 0;

    if (visualization_enabled_globally_)
    {
        tree_marker_id_ = 1;
        forward_tree_next_visualized_node_ = 0;
        backward_tree_next_visualized_node_ = 0;
        visualizeBlacklist();
    }

    // Clear statistics
    total_sampling_time_ = 0.0;
    total_nearest_neighbour_index_building_time_ = 0.0;
    total_nearest_neighbour_index_searching_time_ = 0.0;
    total_nearest_neighbour_linear_searching_time_ = 0.0;
    total_nearest_neighbour_radius_searching_time_ = 0.0;
    total_nearest_neighbour_best_searching_time_ = 0.0;
    total_nearest_neighbour_time_ = 0.0;
    total_forward_kinematics_time_ = 0.0;
    total_projection_time_ = 0.0;
    total_collision_check_time_ = 0.0;
    total_band_forward_propogation_time_ = 0.0;
    total_first_order_vis_propogation_time_ = 0.0;
    total_everything_included_forward_propogation_time_ = 0.0;

    forward_random_samples_useful_ = 0;
    forward_random_samples_useless_ = 0;
    backward_random_samples_useful_ = 0;
    backward_random_samples_useless_ = 0;
    forward_connection_attempts_useful_ = 0;
    forward_connection_attempts_useless_ = 0;
    forward_connections_made_ = 0;
    backward_connection_attempts_useful_ = 0;
    backward_connection_attempts_useless_ = 0;
    backward_connections_made_ = 0;

    path_found_ = false;
    goal_idx_in_forward_tree_ = -1;

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
        start_time_ = std::chrono::steady_clock::now();
        ROS_INFO_NAMED("rrt", "Starting planning...");
        planningMainLoopBispace();
//        planningMainLoopBidirectional();
        const std::chrono::time_point<std::chrono::steady_clock> end_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> planning_time(end_time - start_time_);
        robot_->unlockEnvironment();

        ROS_INFO_STREAM_NAMED("rrt", "Finished planning, for better or worse. Path found? " << path_found_);

        planning_statistics_["planning_time0_sampling                                 "] = total_sampling_time_;
        planning_statistics_["planning_time1_1_nearest_neighbour_index_building       "] = total_nearest_neighbour_index_building_time_;
        planning_statistics_["planning_time1_2_nearest_neighbour_index_searching      "] = total_nearest_neighbour_index_searching_time_;
        planning_statistics_["planning_time1_3_nearest_neighbour_linear_searching     "] = total_nearest_neighbour_linear_searching_time_;
        planning_statistics_["planning_time1_4_nearest_neighbour_radius_searching     "] = total_nearest_neighbour_radius_searching_time_;
        planning_statistics_["planning_time1_5_nearest_neighbour_best_searching       "] = total_nearest_neighbour_best_searching_time_;
        planning_statistics_["planning_time1_nearest_neighbour                        "] = total_nearest_neighbour_time_;
        planning_statistics_["planning_time2_1_forward_propogation_fk                 "] = total_forward_kinematics_time_;
        planning_statistics_["planning_time2_2_forward_propogation_projection         "] = total_projection_time_;
        planning_statistics_["planning_time2_3_forward_propogation_collision_check    "] = total_collision_check_time_;
        planning_statistics_["planning_time2_4_forward_propogation_band_sim           "] = total_band_forward_propogation_time_;
        planning_statistics_["planning_time2_5_forward_propogation_first_order_vis    "] = total_first_order_vis_propogation_time_;
        planning_statistics_["planning_time2_forward_propogation_everything_included  "] = total_everything_included_forward_propogation_time_;
        planning_statistics_["planning_time3_total                                    "] = planning_time.count();

        planning_statistics_["planning_size00_forward_random_samples_useless          "] = (double)forward_random_samples_useless_;
        planning_statistics_["planning_size01_forward_random_samples_useful           "] = (double)forward_random_samples_useful_;
        planning_statistics_["planning_size02_forward_states                          "] = (double)forward_tree_.size();

        planning_statistics_["planning_size03_backward_random_samples_useless         "] = (double)backward_random_samples_useless_;
        planning_statistics_["planning_size04_backward_random_samples_useful          "] = (double)backward_random_samples_useful_;
        planning_statistics_["planning_size05_backward_states                         "] = (double)backward_tree_.size();

        planning_statistics_["planning_size06_forward_connection_attempts_useless     "] = (double)forward_connection_attempts_useless_;
        planning_statistics_["planning_size07_forward_connection_attempts_useful      "] = (double)forward_connection_attempts_useful_;
        planning_statistics_["planning_size08_forward_connections_made                "] = (double)forward_connections_made_;

//        planning_statistics_["planning_size09_backward_connection_attempts_useless    "] = (double)backward_connection_attempts_useless_;
//        planning_statistics_["planning_size10_backward_connection_attempts_useful     "] = (double)backward_connection_attempts_useful_;
//        planning_statistics_["planning_size11_backward_connections_made               "] = (double)backward_connections_made_;

        ROS_INFO_STREAM_NAMED("rrt", "RRT Helper Planning Statistics:\n" << PrettyPrint::PrettyPrint(planning_statistics_, false, "\n") << std::endl);

        if (visualization_enabled_globally_)
        {
            ROS_INFO_NAMED("rrt", "Visualizing tree.");
            visualizeBothTrees();
            visualizeBlacklist();
            visualizePath(path);
            vis_->forcePublishNow(0.5);
        }

        if (path_found_)
        {
            assert(goal_idx_in_forward_tree_ >= 0 && goal_idx_in_forward_tree_ < (int64_t)forward_tree_.size());
            ROS_INFO_NAMED("rrt", "Extracting solution path");
            path = ExtractSolutionPath(forward_tree_, goal_idx_in_forward_tree_);
            storePath(path);
        }
    }

    // If we either retreived a path, or made a new one, visualize and do smoothing
    if (path.size() != 0)
    {
        if (visualization_enabled_globally_)
        {
            deleteTreeVisualizations();
            visualizeBlacklist();
            visualizePath(path);
            vis_->forcePublishNow(0.05);
        }

//        ROS_INFO_NAMED("rrt", "Playing back unsmoothed path in OpenRAVE");
//        robot_->testPathForCollision(ConvertRRTPathToRobotPath(path));

        ROS_INFO_NAMED("rrt", "Starting Shortcut Smoothing");
        robot_->lockEnvironment();
        const bool visualize_rrt_smoothing = visualization_enabled_globally_ && true;
        path = rrtShortcutSmooth(path, visualize_rrt_smoothing);
        robot_->unlockEnvironment();
        storePath(path);
        std::cout << "RRT Helper Smoothing Statistics:\n" << PrettyPrint::PrettyPrint(smoothing_statistics_, false, "\n") << std::endl << std::endl;

//        ROS_INFO_NAMED("rrt", "Playing back smoothed path in OpenRAVE");
//        robot_->testPathForCollision(ConvertRRTPathToRobotPath(smoothed_path));

        if (visualization_enabled_globally_)
        {
            visualizeBlacklist();
            visualizePath(path);
            vis_->forcePublishNow(0.05);
        }

    }

    return path;
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
        robot_config_path[ind] =  path[ind].robotConfiguration();
    }
    return robot_config_path;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper function for goal check
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void RRTHelper::addBandToBlacklist(const VectorVector3d& band)
{
    blacklisted_goal_rubber_bands_.push_back(band);
}

void RRTHelper::clearBlacklist()
{
    blacklisted_goal_rubber_bands_.clear();
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const VectorVector3d& test_band) const
{
    for (size_t idx = 0; idx < blacklisted_goal_rubber_bands_.size(); idx++)
    {
        const VectorVector3d& blacklisted_path = blacklisted_goal_rubber_bands_[idx];
        if (MDP::CheckFirstOrderHomotopy(test_band, blacklisted_path))
        {
            return true;
        }
    }

    return false;
}

bool RRTHelper::isBandFirstOrderVisibileToBlacklist(const RubberBand& test_band)
{
    Stopwatch stopwatch;
    const auto vector_representation = test_band.resampleBand(work_space_grid_.minStepDimension() / 2.0);
    const bool is_first_order_visible = isBandFirstOrderVisibileToBlacklist(vector_representation);
    const double first_order_vis_time = stopwatch(READ);
    total_first_order_vis_propogation_time_ += first_order_vis_time;

    return is_first_order_visible;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for shortcut smoothing
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static VectorVector3d findFirstGripperWaypoints(
        const std::vector<RRTNode, RRTAllocator>& path,
        const size_t start_index,
        const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());

    // The start of the path is clearly the first 'kink'
    VectorVector3d gripper_path_kinks(1, path[start_index].grippers().first.translation());

    size_t last_kink = start_index;
    Vector3d last_kink_gripper_position = path[last_kink].grippers().first.translation();
    double path_distance = 0.0;

    // We don't include the last index because it is clearly the last 'kink'
    for (size_t idx = start_index; idx < end_index - 1; ++idx)
    {
        const Vector3d& current_gripper_position = path[idx].grippers().first.translation();
        const Vector3d& next_gripper_position    = path[idx + 1].grippers().first.translation();
        path_distance += (next_gripper_position - current_gripper_position).norm();
        const double straight_line_distance = (next_gripper_position - last_kink_gripper_position).norm();

        // If the straight line distance between the start and the next gripper does not match the path distance, then the current node is a kink
        if (!IsApprox(straight_line_distance, path_distance, 1e-6))
        {
            last_kink = idx;
            last_kink_gripper_position = path[last_kink].grippers().first.translation();
            path_distance = (next_gripper_position - current_gripper_position).norm();
            gripper_path_kinks.push_back(last_kink_gripper_position);
        }
    }
    gripper_path_kinks.push_back(path[end_index].grippers().first.translation());

    return gripper_path_kinks;
}

static VectorVector3d findSecondGripperWaypoints(
        const std::vector<RRTNode, RRTAllocator>& path,
        const size_t start_index,
        const size_t end_index)
{
    assert(start_index < end_index);
    assert(end_index < path.size());

    // The start of the path is clearly the first 'kink'
    VectorVector3d gripper_path_kinks(1, path[start_index].grippers().second.translation());

    size_t last_kink = start_index;
    Vector3d last_kink_gripper_position = path[last_kink].grippers().second.translation();
    double path_distance = 0;

    // We don't include the last index because it is clearly the last 'kink'
    for (size_t idx = start_index; idx < end_index - 1; ++idx)
    {
        const Vector3d& current_gripper_position = path[idx].grippers().second.translation();
        const Vector3d& next_gripper_position    = path[idx + 1].grippers().second.translation();
        path_distance += (next_gripper_position - current_gripper_position).norm();
        const double straight_line_distance = (next_gripper_position - last_kink_gripper_position).norm();

        // If the straight line distance between the start and the next gripper does not match the path distance, then the current node is a kink
        if (!IsApprox(straight_line_distance, path_distance, 1e-6))
        {
            last_kink = idx;
            last_kink_gripper_position = current_gripper_position;
            path_distance = (next_gripper_position - current_gripper_position).norm();
            gripper_path_kinks.push_back(last_kink_gripper_position);
        }
    }
    gripper_path_kinks.push_back(path[end_index].grippers().second.translation());

    return gripper_path_kinks;
}

static VectorVector3d createOtherGripperWaypoints(
        const VectorVector3d& given_gripper_waypoints,
        const Vector3d& start_point,
        const Vector3d& end_point)
{
    const size_t num_waypoints = given_gripper_waypoints.size();
    assert(num_waypoints >= 2);

    VectorVector3d other_gripper_waypoints;
    other_gripper_waypoints.reserve(num_waypoints);
    other_gripper_waypoints.push_back(start_point);

    // We will need to "space out" the distance between start_point and end_point to match those of the given waypoints
    // Note that we've already inserted the first waypoint, and we'll insert the last manually as well
    const std::vector<double> cummulative_distances = CalculateCumulativeDistances(given_gripper_waypoints);
    for (size_t idx = 1; idx < num_waypoints - 1; ++idx)
    {
        const double ratio = cummulative_distances[idx] / cummulative_distances.back();
        const auto next_waypoint = Interpolate(start_point, end_point, ratio);
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
    if (!bandEndpointsMatchGripperPositions(rubber_band, path[start_index].grippers()))
    {
        std::cerr << "Inside forwardSimulateGrippersPath\n";
        std::cerr << "initial rubber band endpoints:\n"
                  << PrettyPrint::PrettyPrint(rubber_band.getEndpoints()) << std::endl;

        std::cerr << "path gripper positions:\n"
                  << PrettyPrint::PrettyPrint(path[start_index].grippers()) << std::endl;

        assert(false && "Band endpoints do not match recorded gripper positions");
    }

    // Collect the results for use by the rrtShortcutSmooth function
    std::vector<RRTNode, RRTAllocator> resulting_path;
    // Put the start position on the path
    {
        resulting_path.reserve(path.size() - start_index);
        resulting_path.push_back(
                    RRTNode(path[start_index].grippers(),
                            path[start_index].robotConfiguration(),
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
        const auto& ending_grippers_pos = path[path_idx].grippers();
        rubber_band.forwardPropagateRubberBandToEndpointTargets(
                    ending_grippers_pos.first.translation(),
                    ending_grippers_pos.second.translation(),
                    rubber_band_verbose);
        const double forward_propogation_time = stopwatch(READ);
        total_band_forward_propogation_time_ += forward_propogation_time;

        // Store the band in the results
        resulting_path.push_back(RRTNode(
                                     path[path_idx].grippers(),
                                     path[path_idx].robotConfiguration(),
                                     std::make_shared<RubberBand>(rubber_band)));

        // Record if the band is overstretched
        band_is_overstretched = rubber_band.isOverstretched();
        band_got_stuck = !bandEndpointsMatchGripperPositions(rubber_band, path[path_idx].grippers());

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
    total_forward_kinematics_time_ = 0.0;
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
            const double minimum_distance = RRTDistance::Distance(smoothing_start_config.robotConfiguration(), smoothing_end_config.robotConfiguration());
            const double path_distance = RRTDistance::RobotPathDistance(path, smoothing_start_index, smoothing_end_index);
            // Essentially this checks if there is a kink in the path
            if (IsApprox(path_distance, minimum_distance, 1e-6))
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
            const auto& target_robot_configuration = smoothing_end_config.robotConfiguration();
            const auto& last_robot_configuration = smoothed_segment.back().robotConfiguration();

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
            end_of_smoothing_to_goal_results = forwardSimulateGrippersPath(path, smoothing_end_index, *smoothed_segment.back().band());
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
                const double minimum_distance = RRTDistance::Distance(smoothing_start_config.grippers(), smoothing_end_config.grippers());
                const double path_distance = RRTDistance::GrippersPathDistance(path, smoothing_start_index, smoothing_end_index);
                // Essentially this checks if there is a kink in the path
                if (IsApprox(path_distance, minimum_distance, 1e-6))
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
                const VectorVector3d current_waypoints_first_gripper =
                        findFirstGripperWaypoints(path, smoothing_start_index, smoothing_end_index);
                const VectorVector3d current_waypoints_second_gripper =
                        findSecondGripperWaypoints(path, smoothing_start_index, smoothing_end_index);

                VectorVector3d target_waypoints_first_gripper;
                VectorVector3d target_waypoints_second_gripper;
                // Smooth the first gripper
                if (smoothing_type == 3)
                {
                    const auto& first_gripper_start_pos = smoothing_start_config.grippers().first.translation();
                    const auto& first_gripper_end_pos = smoothing_end_config.grippers().first.translation();

                    // Check if there is room for improvement for the first gripper
                    const double minimum_distance = (first_gripper_end_pos - first_gripper_start_pos).norm();
                    const double path_distance = CalculateTotalDistance(current_waypoints_first_gripper);
                    // Essentially this checks if there is a kink in the path
                    if (IsApprox(path_distance, minimum_distance, 1e-6))
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
                    const auto& second_gripper_start_pos = smoothing_start_config.grippers().second.translation();
                    const auto& second_gripper_end_pos = smoothing_end_config.grippers().second.translation();

                    // Check if there is room for improvement for the second gripper
                    const double minimum_distance = (second_gripper_end_pos - second_gripper_start_pos).norm();
                    const double path_distance = CalculateTotalDistance(current_waypoints_first_gripper);
                    // Essentially this checks if there is a kink in the path
                    if (IsApprox(path_distance, minimum_distance, 1e-6))
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
                    RRTGrippersRepresentation target_poses = path.front().grippers();
                    target_poses.first.translation() = target_waypoints_first_gripper[waypoint_idx];
                    target_poses.second.translation() = target_waypoints_second_gripper[waypoint_idx];
                    RRTRobotRepresentation target_config(6);
                    target_config << target_waypoints_first_gripper[waypoint_idx], target_waypoints_second_gripper[waypoint_idx];
                    const RRTNode forward_prop_target_config(
                                target_poses,
                                target_config,
                                path.front().band());

                    const int64_t start_idx = (int64_t)smoothed_segment.size() - 1;
                    forwardPropogationFunction(smoothed_segment, start_idx, forward_prop_target_config, fwd_prop_extend_band, fwd_prop_max_steps, fwd_prop_local_visualization_enabled);
                }
            }
            else
            {
                assert(false && "Smoothing type was something other than [1, 4], this ougth to be impossible");
            }

            // Check if the rubber band gets overstretched while propogating the grippers on the new path
            const auto& target_gripper_position = smoothing_end_config.grippers();
            const auto& last_gripper_position = smoothed_segment.back().grippers();
            if (!gripperPositionsAreApproximatelyEqual(last_gripper_position, target_gripper_position))
            {
                ++failed_iterations;
                continue;
            }

            // We still need to check that the rubber band can still reach the goal correctly from this state,
            // so we'll forward propogate along the rest of the trajectory to check feasibility
            end_of_smoothing_to_goal_results = forwardSimulateGrippersPath(path, smoothing_end_index, *smoothed_segment.back().band());
        }

        const bool final_band_at_goal_success = end_of_smoothing_to_goal_results.first;
        const auto& end_of_smoothing_to_goal_path_ = end_of_smoothing_to_goal_results.second;

        // Check if the rubber band gets overstretched or ends up in a blacklisted first order
        // homotopy class while following the tail of the starting trajectory
        {
            const auto& final_node_of_smoothing = end_of_smoothing_to_goal_path_.back();
            const bool final_band_visible_to_blacklist = isBandFirstOrderVisibileToBlacklist(*final_node_of_smoothing.band());
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
    smoothing_statistics_["smoothing2_forward_propogation_fk_time                  "] = total_forward_kinematics_time_;
    smoothing_statistics_["smoothing3_forward_propogation_crrt_projection_time     "] = total_projection_time_;
    smoothing_statistics_["smoothing4_forward_propogation_collision_check_time     "] = total_collision_check_time_;
    smoothing_statistics_["smoothing5_forward_propogation_band_sim_time            "] = total_band_forward_propogation_time_;
    smoothing_statistics_["smoothing6_forward_propogation_first_order_vis_time     "] = total_first_order_vis_propogation_time_;
    smoothing_statistics_["smoothing7_forward_propogation_everything_included_time "] = total_everything_included_forward_propogation_time_;
    smoothing_statistics_["smoothing8_total_time                                   "] = smoothing_time;

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

        VectorVector3d band_line_start_points;
        VectorVector3d band_line_end_points;

        VectorVector3d gripper_a_tree_start_points;
        VectorVector3d gripper_a_tree_end_points;
        VectorVector3d gripper_b_tree_start_points;
        VectorVector3d gripper_b_tree_end_points;

        gripper_a_tree_start_points.reserve(tree.size() - start_idx);
        gripper_b_tree_start_points.reserve(tree.size() - start_idx);
        gripper_a_tree_end_points.reserve(tree.size() - start_idx);
        gripper_b_tree_end_points.reserve(tree.size() - start_idx);

        for (size_t idx = start_idx; idx < tree.size(); ++idx)
        {
            const RRTNode& curr = tree[idx];

            if (draw_band)
            {
                const VectorVector3d& band_vec = curr.band()->getVectorRepresentation();
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
                gripper_a_tree_start_points.push_back(parent.grippers().first.translation());
                gripper_b_tree_start_points.push_back(parent.grippers().second.translation());

                gripper_a_tree_end_points.push_back(curr.grippers().first.translation());
                gripper_b_tree_end_points.push_back(curr.grippers().second.translation());
            }
        }

        vis_->visualizeLines(ns_a, gripper_a_tree_start_points, gripper_a_tree_end_points, color_a, id_a);
        vis_->visualizeLines(ns_b, gripper_b_tree_start_points, gripper_b_tree_end_points, color_b, id_b);
        if (draw_band)
        {
            vis_->visualizeLines(ns_band, band_line_start_points, band_line_end_points, color_band, id_band);
        }
    }
}

void RRTHelper::visualizeBothTrees() const
{
    deleteTreeVisualizations();

    const bool draw_band = false;

    visualizeTree(
                forward_tree_,
                0,
                RRT_FORWARD_TREE_GRIPPER_A_NS,
                RRT_FORWARD_TREE_GRIPPER_B_NS,
                RRT_TREE_BAND_NS,
                1,
                1,
                1,
                gripper_a_forward_tree_color_,
                gripper_b_forward_tree_color_,
                band_tree_color_,
                draw_band);

    visualizeTree(
                backward_tree_,
                0,
                RRT_BACKWARD_TREE_GRIPPER_A_NS,
                RRT_BACKWARD_TREE_GRIPPER_B_NS,
                RRT_TREE_BAND_NS,
                2,
                2,
                2,
                gripper_a_backward_tree_color_,
                gripper_b_backward_tree_color_,
                band_tree_color_,
                draw_band);

    vis_->forcePublishNow();
}

void RRTHelper::deleteTreeVisualizations() const
{
    vis_->purgeMarkerList();
    visualization_msgs::Marker marker;
    marker.action = visualization_msgs::Marker::DELETEALL;
    marker.header.frame_id = "world_origin";
    marker.header.stamp = ros::Time::now();
    vis_->publish(marker);
    vis_->forcePublishNow(0.01);
    vis_->purgeMarkerList();

    vis_->visualizeCubes(TaskFramework::CLUSTERING_RESULTS_POST_PROJECT_NS, {grippers_goal_poses_.first.translation()}, Vector3d::Ones() * work_space_grid_.minStepDimension(), gripper_a_forward_tree_color_, 1);
    vis_->visualizeCubes(TaskFramework::CLUSTERING_RESULTS_POST_PROJECT_NS, {grippers_goal_poses_.second.translation()}, Vector3d::Ones() * work_space_grid_.minStepDimension(), gripper_b_forward_tree_color_, 5);
}

void RRTHelper::visualizePath(const std::vector<RRTNode, RRTAllocator>& path) const
{
    VectorVector3d gripper_a_cubes;
    VectorVector3d gripper_b_cubes;
    gripper_a_cubes.reserve(path.size());
    gripper_b_cubes.reserve(path.size());

    VectorVector3d line_start_points;
    VectorVector3d line_end_points;

    for (int32_t ind = 0; ind < (int32_t)path.size(); ++ind)
    {
        const RRTNode& config = path[ind];
        const RRTGrippersRepresentation& gripper_positions = config.grippers();
        const RubberBand::Ptr& rubber_band = config.band();

        gripper_a_cubes.push_back(gripper_positions.first.translation());
        gripper_b_cubes.push_back(gripper_positions.second.translation());

        const VectorVector3d band_vec = rubber_band->getVectorRepresentation();
        for (size_t band_idx = 0; band_idx + 1 < band_vec.size(); ++band_idx)
        {
            line_start_points.push_back(band_vec[band_idx]);
            line_end_points.push_back(band_vec[band_idx + 1]);
        }
    }

    vis_->visualizeCubes(RRT_SOLUTION_GRIPPER_A_NS, gripper_a_cubes, Vector3d(0.005, 0.005, 0.005), gripper_a_forward_tree_color_, 1);
    vis_->visualizeCubes(RRT_SOLUTION_GRIPPER_B_NS, gripper_b_cubes, Vector3d(0.005, 0.005, 0.005), gripper_b_forward_tree_color_, 1);
//        vis_->visualizeLines(RRT_SOLUTION_RUBBER_BAND_NS, line_start_points, line_end_points, Visualizer::Yellow(), 1);
}

void RRTHelper::visualizeBlacklist() const
{
    VectorVector3d line_start_points;
    VectorVector3d line_end_points;

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

void RRTHelper::storePath(const std::vector<RRTNode, RRTAllocator>& path, std::string file_path) const
{
    try
    {
        if (file_path.empty())
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
            file_path = log_folder.GetImmutable() + file_name;
        }
        ROS_DEBUG_STREAM_NAMED("rrt", "Saving path to " << file_path);

        std::vector<uint8_t> buffer;
        arc_utilities::SerializeVector<RRTNode, RRTAllocator>(path, buffer, &RRTNode::Serialize);
        ZlibHelpers::CompressAndWriteToFile(buffer, file_path);

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
        ROS_ERROR_STREAM_NAMED("rrt", "Failed to store path: "  <<  e.what());
    }
}

std::vector<RRTNode, RRTAllocator> RRTHelper::loadStoredPath(std::string file_path) const
{
    try
    {
        if (file_path.empty())
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
            file_path = log_folder.GetImmutable() + file_name;
        }
        ROS_INFO_STREAM_NAMED("rrt", "Loading path from " << file_path);

        const auto deserializer = [&] (const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return RRTNode::Deserialize(buffer, current, *starting_band_);
        };

        const auto buffer = ZlibHelpers::LoadFromFileAndDecompress(file_path);
        const auto path_deserialized = arc_utilities::DeserializeVector<RRTNode, RRTAllocator>(buffer, 0, deserializer);
        return path_deserialized.first;
    }
    catch (const std::exception& e)
    {
        ROS_ERROR_STREAM_NAMED("rrt", "Failed to load stored path: "  <<  e.what());
    }

    return std::vector<RRTNode, RRTAllocator>();
}

bool RRTHelper::useStoredPath() const
{
    return ROSHelpers::GetParamRequired<bool>(ph_, "use_stored_path", __func__).GetImmutable();
}
