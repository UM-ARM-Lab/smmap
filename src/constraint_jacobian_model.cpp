#include "smmap/constraint_jacobian_model.h"

#include <cmath>
#include <arc_utilities/arc_exceptions.hpp>
#include <arc_utilities/timing.hpp>

using namespace smmap;
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Static member initialization
////////////////////////////////////////////////////////////////////////////////

std::atomic_bool ConstraintJacobianModel::static_data_initialized_(false);
Eigen::MatrixXd ConstraintJacobianModel::object_initial_node_distance_;
Eigen::MatrixXd ConstraintJacobianModel::gripper_influence_per_node_;
//Eigen::VectorXd ConstraintJacobianModel::sum_of_object_node_to_grippers_distances_;
ssize_t ConstraintJacobianModel::num_nodes_;

////////////////////////////////////////////////////////////////////////////////
// Static helpers
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief ConstraintJacobianModel::SetInitialObjectConfiguration This function
 *          is not thread safe.
 * @param object_initial_configuration
 */
void ConstraintJacobianModel::SetInitialObjectConfiguration(
        const ObjectPointSet& object_initial_configuration)
{
    if (!grippers_data_initialized_.load())
    {
        throw_arc_exception(std::runtime_error, "You must call DeformableModel::SetGrippersData before setting the static data for ConstraintJacobianModel");
    }
    const ssize_t num_grippers = (ssize_t)(grippers_data_.size());

    num_nodes_ = object_initial_configuration.cols();
    object_initial_node_distance_ = EigenHelpers::CalculateDistanceMatrix(object_initial_configuration);

    // First, collect the distances for each gripper <-> node pair
    MatrixXd gripper_to_node_min_distances(num_grippers, num_nodes_);
    for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
    {
        for (ssize_t node_ind = 0; node_ind < num_nodes_; node_ind++)
        {
            const std::pair<double, long> nearest_node_on_gripper =
                    GetMinimumDistanceIndexToGripper(
                        grippers_data_[(size_t)gripper_ind].node_indices_,
                        node_ind, object_initial_node_distance_);

            gripper_to_node_min_distances(gripper_ind, node_ind) = nearest_node_on_gripper.first;
        }
    }

    // object_node_to_grippers_distances_ is indexed first by gripper, second by node i.e. (gripper_ind, node_ind)
    gripper_influence_per_node_.resize(num_grippers, num_nodes_);
    // Then, calculate relative control authority
    // Last, normalize
    for (ssize_t node_ind = 0; node_ind < num_nodes_; node_ind++)
    {
        const double min_dist = gripper_to_node_min_distances.col(node_ind).minCoeff();

        for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
        {
            const double gripper_dist = gripper_to_node_min_distances(gripper_ind, node_ind);
            if (gripper_dist == 0.0)
            {
                assert(min_dist == 0.0);
                gripper_influence_per_node_(gripper_ind, node_ind) = 1.0;
            }
            else
            {
                gripper_influence_per_node_(gripper_ind, node_ind) = min_dist / gripper_dist;
            }
        }
        const double normalizer = gripper_influence_per_node_.col(node_ind).sum();
        gripper_influence_per_node_.col(node_ind) /= normalizer;
    }

    static_data_initialized_.store(true);
}

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////

ConstraintJacobianModel::ConstraintJacobianModel(
        const double translation_dir_deformability,
        const double translation_dis_deformability,
        const double rotation_deformability,
//        const double translational_deformablity,
        const sdf_tools::SignedDistanceField& environment_sdf)
    : translation_dir_deformability_(translation_dir_deformability)
    , translation_dis_deformability_(translation_dis_deformability)
    , rotation_deformability_(rotation_deformability)
//    , translational_deformability_(translational_deformablity)
    , environment_sdf_(environment_sdf)
    , obstacle_threshold_(0.0)
{
    if (!static_data_initialized_.load())
    {
        throw_arc_exception(std::runtime_error, "You must call SetInitialObjectConfiguration before constructing a ConstraintJacobianModel");
    }
    if (translation_dir_deformability < 0)
    {
        throw_arc_exception(std::invalid_argument, "translation_deformability must be >= 0");
    }
    if (translation_dis_deformability < 0)
    {
        throw_arc_exception(std::invalid_argument, "translation_deformability must be >= 0");
    }
    if (rotation_deformability < 0)
    {
        throw_arc_exception(std::invalid_argument, "rotation_deformability must be >= 0");
    }
}

////////////////////////////////////////////////////////////////////////////////
// Virtual function overrides
////////////////////////////////////////////////////////////////////////////////

void ConstraintJacobianModel::updateModel_impl(const WorldState& previous, const WorldState& next)
{
    // This model doesn't do any updates, so tell the compiler that it's okay
    // that these values are unused.
    (void)previous;
    (void)next;
}

ObjectPointSet ConstraintJacobianModel::getObjectDelta_impl(
        const WorldState& world_state,
        const AllGrippersSinglePoseDelta& grippers_pose_delta) const
{
    const MatrixXd J = computeGrippersToDeformableObjectJacobian(world_state, grippers_pose_delta);
    const ObjectPointSet& current_configuration = world_state.object_configuration_;

    const Eigen::VectorXd grippers_delta =
            EigenHelpers::VectorEigenVectorToEigenVectorX(grippers_pose_delta);

    // Move the object based on the movement of each gripper
    MatrixXd object_delta = J * grippers_delta;

    #pragma omp parallel for
    for (ssize_t node_ind = 0; node_ind < num_nodes_; node_ind++)
    {
        const auto node = current_configuration.col(node_ind);
        // Do nothing if we are not in collision
        if (environment_sdf_.EstimateDistance4dLegacy(Eigen::Vector4d(node.x(), node.y(), node.z(), 1.0)).first > obstacle_threshold_)
        {
            continue;
        }
        else
        {
            const Vector3d& node_p_dot = object_delta.block<3, 1>(node_ind * 3, 0);
            std::vector<double> sur_n = environment_sdf_.GetGradient3d(node);
            if (sur_n.size() > 1)
            {
                const Vector3d surface_normal = Vector3d::Map(sur_n.data(), sur_n.size()).normalized();

                // if node is moving outward from obstacle, unmask.
                const double dot_result = node_p_dot.dot(surface_normal);
                if (dot_result < 0.0)
                {
                    const auto projected_node_p_dot = node_p_dot - dot_result * surface_normal;
                    object_delta.block<3, 1>(node_ind * 3, 0) = projected_node_p_dot;
                }
            }

        }
    }

    // Resize delta to a 3xn vector
    object_delta.resizeLike(world_state.object_configuration_);
    return object_delta;
}

////////////////////////////////////////////////////////////////////////////////
// Jacobian and Mask matrix computation
////////////////////////////////////////////////////////////////////////////////


/**
 * @brief ConstraintJacobianModel::computeGrippersToDeformableObjectJacobian_impl
 * Computes a Jacobian that converts gripper velocities in the individual
 * gripper frames into object velocities in the world frame
 * @param input_data
 * @return
 */

// Mask Version

Eigen::MatrixXd ConstraintJacobianModel::computeGrippersToDeformableObjectJacobian(
        const WorldState& world_state,
        const AllGrippersSinglePoseDelta &grippers_pose_delta) const
{
    const AllGrippersSinglePose& grippers_current_poses = world_state.all_grippers_single_pose_;
    const ObjectPointSet& current_configuration = world_state.object_configuration_;

    const kinematics::VectorIsometry3d grippers_next_poses = kinematics::applyTwist(
                grippers_current_poses,
                grippers_pose_delta);

    const ssize_t num_grippers = (ssize_t)grippers_current_poses.size();
    const ssize_t num_Jcols = num_grippers * 6;
    const ssize_t num_Jrows = num_nodes_ * 3;

    MatrixXd J(num_Jrows, num_Jcols);

//    Eigen::RowVectorXd disLinear_terms(num_nodes_);
//    Eigen::RowVectorXd dirPropotional_terms(num_nodes_);
//    Eigen::RowVectorXd transDimOld_terms(num_nodes_);

    // for each gripper
    for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
    {
        // Get all the data we need for a given gripper
        const Matrix3d& gripper_rot = grippers_current_poses[(size_t)gripper_ind].rotation();

        // P dot of the node on object, grasped gripper
        // Due to the assumption of free-flying grippers, I simply take it as the xyz motion of grippers
        // In the future, it should be the translational motion of end effector.
        const Vector3d& gripper_translation = grippers_next_poses[gripper_ind].translation()
                - grippers_current_poses[gripper_ind].translation();

        for (ssize_t node_ind = 0; node_ind < num_nodes_; node_ind++)
        {
            const std::pair<double, long> nearest_node_on_gripper =
                    GetMinimumDistanceIndexToGripper(
                        grippers_data_[(size_t)gripper_ind].node_indices_,
                        node_ind, object_initial_node_distance_);
            const double geodesic_distance_to_gripper = nearest_node_on_gripper.first;
            const long nearest_node_on_gripper_ind = nearest_node_on_gripper.second;

            const Vector3d node_to_gripper =
                    current_configuration.col(nearest_node_on_gripper_ind) -
                    current_configuration.col(node_ind);
            const double euclidean_dist_to_gripper = node_to_gripper.norm();

            const Matrix3d& J_trans = gripper_rot;


            double cos_angle = 0.0;
            if (node_to_gripper.norm() > 1e-6 && gripper_translation.norm() > 1e-6)
            {
                cos_angle = node_to_gripper.normalized().dot(gripper_translation.normalized());
            }
            else
            {
                cos_angle = 1.0;
            }
            const double direction_based_rigidity = std::exp(translation_dir_deformability_ * (cos_angle - 1.0) * geodesic_distance_to_gripper);//std::min(cos_angle, 0.0));
//            const double distance_based_rigidity = std::exp(-translational_deformability_ * geodesic_distance_to_gripper);
//            const double lambda = (1.0 - cos_angle) / 2.0;
//            const double combined_translational = lambda * distance_based_rigidity + (1.0 - lambda) * direction_based_rigidity;

            const double distance_ratio_rigidity = disLinearModel(euclidean_dist_to_gripper, geodesic_distance_to_gripper);

            const double rigidity_translation =
                    gripper_influence_per_node_(gripper_ind, node_ind) *
                    distance_ratio_rigidity *
//                    combined_translational;
                    direction_based_rigidity;

            J.block<3, 3>(node_ind * 3, gripper_ind * 6) = rigidity_translation * J_trans;




//            disLinear_terms(node_ind) = distance_ratio_rigidity;
//            dirPropotional_terms(node_ind) = direction_based_rigidity;
//            transDimOld_terms(node_ind) = distance_based_rigidity;




            // Calculate the cross product between the grippers x, y, and z axes
            // and the vector from the gripper to the node, for rotation utilization

            const Vector3d gripper_to_node = -node_to_gripper;

            Matrix3d J_rot;
            J_rot.col(0) = gripper_rot.col(0).cross(gripper_to_node);
            J_rot.col(1) = gripper_rot.col(1).cross(gripper_to_node);
            J_rot.col(2) = gripper_rot.col(2).cross(gripper_to_node);

            const double rigidity_rotation =
                    gripper_influence_per_node_(gripper_ind, node_ind) *
                    std::exp(-rotation_deformability_ * geodesic_distance_to_gripper);

            J.block<3, 3>(node_ind * 3, gripper_ind * 6 + 3) = rigidity_rotation * J_rot;
        }
    }

//    std::cout << "dis linear:   " << disLinear_terms.head<8>() << std::endl;
//    std::cout << "dir proporp:  " << dirPropotional_terms.head<8>() << std::endl;
//    std::cout << "old dis term: " << transDimOld_terms.head<8>() << std::endl;
//    std::cout << std::endl;

    return J;
}

/**
 * @brief ConstraintJacobianModel::computeObjectVelocityMask
 * @param current_configuration
 * @param desired_object_velocity
 * @return
 */
Eigen::MatrixXd ConstraintJacobianModel::computeObjectVelocityMask(
        const ObjectPointSet &current_configuration,
        const MatrixXd &object_p_dot) const
{
    assert(false && "This is not used right now");

    const ssize_t num_lines = num_nodes_ * 3;
    MatrixXd M(num_lines, num_lines);
    M.setIdentity(num_lines,num_lines);
    const Matrix3d I3 = Matrix3d::Identity();

    for (ssize_t node_ind = 0; node_ind < num_nodes_; node_ind++)
    {
        // if is far from obstacle
        if (environment_sdf_.EstimateDistance3d(current_configuration.col(node_ind)).first > obstacle_threshold_)
        {
            continue;
        }
        // if is close to obstacle
        else
        {
            const Vector3d node_p_dot = object_p_dot.block<3, 1>(node_ind * 3, 0);

            // sur_n pointing out from obstacle
            std::vector<double> sur_n = environment_sdf_.GetGradient3d(current_configuration.col(node_ind));

            if (sur_n.size() > 1)
            {
                const Vector3d surface_normal = Vector3d::Map(sur_n.data(),sur_n.size()).normalized();
                // if node is moving outward from obstacle, unmask.
                const double dot_result = node_p_dot.dot(surface_normal);
                if (dot_result < 0.0)
                {
                    const Matrix<double, 1, 3> surface_normal_inv = surface_normal.adjoint();
                    M.block<3, 3>(node_ind * 3, node_ind * 3) = I3 - surface_normal*surface_normal_inv;
                }
            }
        }
    }

    return M;
}

////////////////////////////////////////////////////////////////////////////////
// Candidate function to model rigidity weighting for translation Jacobian
////////////////////////////////////////////////////////////////////////////////

// TODO: consider replacing this "dot" with (1 - cos(theta)) and rescaling translation_dir_deformablility_
// Candidate function to count vector effect, return 3x3 matrix = diag{exp[beta(.,.)]}
double ConstraintJacobianModel::dirPropotionalModel(const Vector3d node_to_gripper, const Vector3d gripper_translation) const
{
    double dot = 0.0;
    if (node_to_gripper.norm() > 1e-6 && gripper_translation.norm() > 1e-6)
    {
        dot = node_to_gripper.normalized().dot(gripper_translation.normalized());
        dot = std::min(dot, 0.0);
    }
    else
    {
        dot = 0.0;
    }

    return std::exp(translation_dir_deformability_ * dot);
}


double ConstraintJacobianModel::disLinearModel(const double dist_to_gripper, const double dist_rest) const
{
    double ratio;
    if (std::fabs(dist_rest) < 0.00001)
    {
        ratio = 1;
    }
    else
    {
        ratio = dist_to_gripper / dist_rest;
    }

    return std::pow(ratio, translation_dis_deformability_);
}
