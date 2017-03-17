#include "smmap/constraint_jacobian_model.h"
//#include "smmap/grippers.hpp"

#include <cmath>
#include <arc_utilities/arc_exceptions.hpp>

using namespace smmap;
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Static member initialization
////////////////////////////////////////////////////////////////////////////////

std::atomic_bool ConstraintJacobianModel::static_data_initialized_(false);
Eigen::MatrixXd ConstraintJacobianModel::object_initial_node_distance_;
long ConstraintJacobianModel::num_nodes_;

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
    num_nodes_ = object_initial_configuration.cols();
    object_initial_node_distance_ = CalculateDistanceMatrix(object_initial_configuration);
    static_data_initialized_.store(true);
}

////////////////////////////////////////////////////////////////////////////////
// Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////


ConstraintJacobianModel::ConstraintJacobianModel(
        const double translation_dir_deformability,
        const double translation_dis_deformability,
        const double rotation_deformability,
        const sdf_tools::SignedDistanceField environment_sdf,
        const bool optimize)
    : ConstraintJacobianModel(
         translation_dir_deformability,
         translation_dis_deformability,
         rotation_deformability,
         environment_sdf,
         simpleFn,
         simpleFn,
         optimize)
{}

ConstraintJacobianModel::ConstraintJacobianModel(
        const double translation_dir_deformability,
        const double translation_dis_deformability,
        const double rotation_deformability,
        const sdf_tools::SignedDistanceField environment_sdf,
        RigidityFnType trans_dir_fn,
        RigidityFnType trans_dis_fn,
        const bool optimize)
    : JacobianModel(optimize)
    , translation_dir_deformability_(translation_dir_deformability)
    , translation_dis_deformability_(translation_dis_deformability)
    , rotation_deformability_(rotation_deformability)
    , trans_dir_type_(trans_dir_fn)
    , trans_dis_type_(trans_dis_fn)
    , environment_sdf_(environment_sdf)
    , obstacle_threshold_(2.0)
{
    // Set obstacle distance threshold, to be modified later
    // Should check with Dale, whether it counts as #grids
//    obstacle_threshold_ = 2.0;

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

/**
 * @brief ConstraintJacobianModel::computeGrippersToDeformableObjectJacobian_impl
 * Computes a Jacobian that converts gripper velocities in the individual
 * gripper frames into object velocities in the world frame
 * @param input_data
 * @return
 */
Eigen::MatrixXd ConstraintJacobianModel::computeGrippersToDeformableObjectJacobian_impl(
        const DeformableModelInputData &input_data) const
{
    const WorldState& world_state = input_data.world_initial_state_;
    const AllGrippersSinglePose& grippers_pose = world_state.all_grippers_single_pose_;
    const ObjectPointSet& current_configuration = world_state.object_configuration_;

    const ssize_t num_grippers = (ssize_t)grippers_pose.size();
    const ssize_t num_Jcols = num_grippers * 6;
    const ssize_t num_Jrows = num_nodes_ * 3;

    MatrixXd J(num_Jrows, num_Jcols);

    // Retrieve the desired object velocity (p_dot)
    const VectorXd& object_p_dot = input_data.task_desired_object_delta_fn_(world_state).delta;

    // for each gripper
    for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
    {
        // Get all the data we need for a given gripper
        const Matrix3d& gripper_rot = grippers_pose[(size_t)gripper_ind].rotation();

        for (ssize_t node_ind = 0; node_ind < num_nodes_; node_ind++)
        {

            const std::pair<double, long> nearest_node_on_gripper =
                    getMinimumDistanceIndexToGripper(
                        grippers_data_[(size_t)gripper_ind].node_indices,
                        node_ind, object_initial_node_distance_);

            // Calculate the cross product between the grippers x, y, and z axes
            // and the vector from the gripper to the node, for rotation utilization
            const Vector3d gripper_to_node =
                    current_configuration.col(node_ind) -
                    grippers_pose[(size_t)gripper_ind].translation();

            // TODO: get dist_rest, get node velocity
            // Dist_real_vec is vector form, it is for translation rigidity utilization
            // Dist_real is the distance between two nodes on objects
            // Gripper_to_node is the radius of rotation, one node may not on object
            const Vector3d dist_real_vec =
                    current_configuration.col(nearest_node_on_gripper.second)-
                    current_configuration.col(node_ind) ; // in .hpp file ;
            const double dist_real = dist_real_vec.norm();

            // P dot of the node on object, grasped gripper
            const Vector3d& node_v = object_p_dot.segment<3>(nearest_node_on_gripper.second*3); // planner

            const Matrix3d& J_trans = gripper_rot;

            // Translation rigidity depends on both gamma(scalar) function and beta(vector) function
            // Gamma inputs are real distance between two nodes and distance at rest
            // Beta inputs are distance vector, node velocity
            const Matrix3d rigidity_translation =
                    disLinearModel(dist_real, nearest_node_on_gripper.first)
                    *dirPropotionalModel(dist_real_vec, node_v);

            J.block<3, 3>(node_ind * 3, gripper_ind * 6) =
                    rigidity_translation * J_trans;

            //J.block<3, 3>(node_ind * 3, gripper_ind * 6) =
            //        std::exp(-translation_deformability_ * dist_to_gripper) * J_trans;


            Matrix3d J_rot;
            J_rot.col(0) = gripper_rot.col(0).cross(gripper_to_node);
            J_rot.col(1) = gripper_rot.col(1).cross(gripper_to_node);
            J_rot.col(2) = gripper_rot.col(2).cross(gripper_to_node);

            J.block<3, 3>(node_ind * 3, gripper_ind * 6 + 3) =
                    std::exp(-rotation_deformability_ * nearest_node_on_gripper.first) * J_rot;

        }
    }

    return J;
}

////////////////////////////////////////////////////////////////////////////////
// Candidate function to model rigidity weighting for translation Jacobian
////////////////////////////////////////////////////////////////////////////////

// Candidate function to count vector effect, return 3x3 matrix = diag{exp[beta(.,.)]}
Eigen::Matrix3d ConstraintJacobianModel::dirPropotionalModel(const Vector3d node_to_gripper, const Vector3d node_v) const
{
    Matrix3d beta_rigidity=MatrixXd::Zero(3,3);
    Vector3d dot_product = node_to_gripper.cwiseProduct(node_v);
    beta_rigidity(1,1) = std::exp(dot_product(1)-std::fabs(dot_product(1)));
    beta_rigidity(2,2) = std::exp(dot_product(2)-std::fabs(dot_product(2)));
    beta_rigidity(3,3) = std::exp(dot_product(3)-std::fabs(dot_product(3)));

    return beta_rigidity;
}


double ConstraintJacobianModel::disLinearModel(const double dist_to_gripper, const double dist_rest) const
{
    double ration = dist_to_gripper/dist_rest;

    return std::pow(ration,translation_dis_deformability_);
}


/**
 * @brief ConstraintJacobianModel::computeObjectVelocityMask
 * @param current_configuration
 * @param desired_object_velocity
 * @return
 */
// q_dot = pinv(J)*Mask*P_dot

Eigen::MatrixXd ConstraintJacobianModel::computeObjectVelocityMask(
        const ObjectPointSet &current_configuration,
        const VectorXd object_p_dot)
{
    const ssize_t num_lines = num_nodes_ * 3;
    MatrixXd M(num_lines, num_lines);
    M.setIdentity(num_lines,num_lines);
    Matrix3d I3 = Matrix3d::Identity(3,3);

    for (ssize_t node_ind = 0; node_ind < num_nodes_; node_ind++)
    {
        // if is far from obstacle
        if (environment_sdf_.Get3d(current_configuration.col(node_ind))>obstacle_threshold_)
        {
            continue;
        }
        // if is close to obstacle
        else
        {
            const Vector3d node_p_dot = object_p_dot.segment<3>(node_ind);
            std::vector<double> sur_n
                    = environment_sdf_.GetGradient3d(current_configuration.col(node_ind));

            Vector3d surface_normal= Vector3d::Map(sur_n.data(),sur_n.size());
            // if node is moving outward from obstacle, unmask.
            if (node_p_dot.dot(surface_normal)>=0)
            {
                continue;
            }
            // Check with Dale, whether the vector is normalized.
            const Matrix<double, 1, 3> surface_normal_inv
                    = EigenHelpers::Pinv(surface_normal, EigenHelpers::SuggestedRcond());
            M.block<3,3>(node_ind*3,node_ind*3)=I3-surface_normal*surface_normal_inv;
        }

    }

    return M;

}




