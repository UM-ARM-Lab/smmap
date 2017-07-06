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
        const sdf_tools::SignedDistanceField& environment_sdf)
    : ConstraintJacobianModel(
         translation_dir_deformability,
         translation_dis_deformability,
         rotation_deformability,
         environment_sdf,
         simpleFn,
         simpleFn)
{}

ConstraintJacobianModel::ConstraintJacobianModel(
        const double translation_dir_deformability,
        const double translation_dis_deformability,
        const double rotation_deformability,
        const sdf_tools::SignedDistanceField& environment_sdf,
        const RigidityFnType trans_dir_fn,
        const RigidityFnType trans_dis_fn)
    : translation_dir_deformability_(translation_dir_deformability)
    , translation_dis_deformability_(translation_dis_deformability)
    , rotation_deformability_(rotation_deformability)
    , trans_dir_type_(trans_dir_fn)
    , trans_dis_type_(trans_dis_fn)
    , environment_sdf_(environment_sdf)
    , obstacle_threshold_(0.03)
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

ObjectPointSet ConstraintJacobianModel::getObjectDelta_impl(
        const DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& grippers_pose_delta) const
{
    const MatrixXd J = computeGrippersToDeformableObjectJacobian(input_data);

    MatrixXd delta = MatrixXd::Zero(input_data.world_current_state_.object_configuration_.cols() * 3, 1);

    // Move the object based on the movement of each gripper
    for (size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++)
    {
        // Assume that our Jacobian is correct, and predict where we will end up
        delta += J.block(0, 6 * (ssize_t)gripper_ind, J.rows(), 6) * grippers_pose_delta[gripper_ind];
    }

    // This delta is a stacked vector
    Eigen::MatrixXd delta_with_mask = computeObjectVelocityMask(input_data.world_current_state_.object_configuration_, delta) * delta;

    // this delta is a 3xn vector
    delta_with_mask.resizeLike(input_data.world_current_state_.object_configuration_);
    return delta_with_mask;
}

std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> ConstraintJacobianModel::getSuggestedGrippersCommand_impl(
        const DeformableModelInputData& input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale) const
{
    assert(false && "Not written");
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
/*
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
                    GetMinimumDistanceIndexToGripper(
                        grippers_data_[(size_t)gripper_ind].node_indices_,
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
*/

// Mask Version

Eigen::MatrixXd ConstraintJacobianModel::computeGrippersToDeformableObjectJacobian(
        const DeformableModelInputData &input_data) const
{
    const WorldState& world_state = input_data.world_current_state_;
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
                    GetMinimumDistanceIndexToGripper(
                        grippers_data_[(size_t)gripper_ind].node_indices_,
                        node_ind, object_initial_node_distance_);

            // Calculate the cross product between the grippers x, y, and z axes
            // and the vector from the gripper to the node, for rotation utilization

        //            const Vector3d gripper_to_node =
    //                current_configuration.col(node_ind) -
    //                grippers_pose[(size_t)gripper_ind].translation();

            const Vector3d gripper_to_node =
                    current_configuration.col(node_ind) -
                    current_configuration.col(nearest_node_on_gripper.second);

            /*
            // TODO: get dist_rest, get node velocity
            // Dist_real_vec is vector form, it is for translation rigidity utilization
            // Dist_real is the distance between two nodes on objects
            // Gripper_to_node is the radius of rotation, one node may not on object
            */
            const Vector3d dist_real_vec =
                    current_configuration.col(nearest_node_on_gripper.second)-
                    current_configuration.col(node_ind) ; // in .hpp file ;
            const double dist_real = dist_real_vec.norm();

            // P dot of the node on object, grasped gripper
            const Vector3d& node_v = object_p_dot.segment<3>(nearest_node_on_gripper.second*3); // planner

            // P dot of the node on object, at this node
//            const Vector3d& target_p_dot = object_p_dot.segment<3>(node_ind*3); // planner


            // Mask from obstacle constrain:
            /*
            const Vector3d& target_node_v = object_p_dot.segment<3>(node_ind);
            Matrix3d M3 = Matrix3d::Identity(3,3);

            if (environment_sdf_.Get3d(current_configuration.col(node_ind))< obstacle_threshold_)
            {
                std::vector<double> sur_n
                        = environment_sdf_.GetGradient3d(current_configuration.col(node_ind));
                if(sur_n.size()>1)
                {
                    Vector3d surface_normal= Vector3d::Map(sur_n.data(),sur_n.size());
                    surface_normal = surface_normal/surface_normal.norm();
                    // if is toward obstacle
                    if (target_node_v.dot(surface_normal)<0)
//                    {}
    //                {  M3 = M3-surface_normal*surface_normal.adjoint(); }
                    {   M3 = M3-surface_normal*(EigenHelpers::Pinv(surface_normal,EigenHelpers::SuggestedRcond())); }
                }
            }
            */
            // End of Mask

            const Matrix3d& J_trans = gripper_rot;

            /*
            // Translation rigidity depends on both gamma(scalar) function and beta(vector) function
            // Gamma inputs are real distance between two nodes and distance at rest
            // Beta inputs are distance vector, node velocity
            */
            const Matrix3d rigidity_translation =
                    disLinearModel(dist_real, nearest_node_on_gripper.first)*
                    dirPropotionalModel(dist_real_vec, node_v);
//            const Matrix3d rigidity_translation =
//                    disLinearModel(dist_real, nearest_node_on_gripper.first)*
//                    dirPropotionalModel(dist_real_vec, target_p_dot);

            // *M3
            J.block<3, 3>(node_ind * 3, gripper_ind * 6) =
                     rigidity_translation *J_trans;

            //J.block<3, 3>(node_ind * 3, gripper_ind * 6) =
            //        std::exp(-translation_deformability_ * dist_to_gripper) * J_trans;


            Matrix3d J_rot;
            J_rot.col(0) = gripper_rot.col(0).cross(gripper_to_node);
            J_rot.col(1) = gripper_rot.col(1).cross(gripper_to_node);
            J_rot.col(2) = gripper_rot.col(2).cross(gripper_to_node);

            // *M3
            J.block<3, 3>(node_ind * 3, gripper_ind * 6 + 3) =
                    std::exp(-rotation_deformability_ * nearest_node_on_gripper.first) * J_rot;
        }
    }
    return J;
//    return Mask*J;
}


// Backup for dislinearModel
/*
double ConstraintJacobianModel::disLinearModel(const double dist_to_gripper, const double dist_rest) const
{
    double ration;
    if (std::fabs(dist_rest)<0.00001)
    {   ration = 1;}
    else
    {   ration = dist_to_gripper/dist_rest;}
    return std::pow(ration,translation_dis_deformability_);
}
*/


/**
 * @brief ConstraintJacobianModel::computeObjectVelocityMask
 * @param current_configuration
 * @param desired_object_velocity
 * @return
 */
// q_dot = pinv(J)*pinv(Mask)*P_dot
// Back up Mask

Eigen::MatrixXd ConstraintJacobianModel::computeObjectVelocityMask(
        const ObjectPointSet &current_configuration,
        const MatrixXd &object_p_dot) const
{
    const ssize_t num_lines = num_nodes_ * 3;
    MatrixXd M(num_lines, num_lines);
    M.setIdentity(num_lines,num_lines);
    Matrix3d I3 = Matrix3d::Identity(3,3);
//    MatrixXd v_mask = MatrixXd::Zero(num_lines,1);


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
            const Vector3d node_p_dot = object_p_dot.block<3,1>(node_ind*3,0);

            // sur_n pointing out from obstacle
            std::vector<double> sur_n
                    = environment_sdf_.GetGradient3d(current_configuration.col(node_ind));

            if(sur_n.size()>1)
            {
                Vector3d surface_normal= Vector3d::Map(sur_n.data(),sur_n.size());

                double surface_vector_norm = std::sqrt(std::pow(surface_normal(0),2)+std::pow(surface_normal(1),2)+std::pow(surface_normal(2),2));

                surface_normal = surface_normal/surface_vector_norm;
                // if node is moving outward from obstacle, unmask.
                double dot_result = 100*node_p_dot(0)*surface_normal(0)+100*node_p_dot(1)*surface_normal(1)+100*node_p_dot(2)*surface_normal(2);
                if (dot_result<0.0)
                {
//                    const Matrix<double, 1, 3> surface_normal_inv
//                            = EigenHelpers::Pinv(surface_normal, EigenHelpers::SuggestedRcond());
                    const Matrix<double, 1, 3> surface_normal_inv = surface_normal.adjoint();
                    M.block<3,3>(node_ind*3,node_ind*3) = I3-surface_normal*surface_normal_inv;
                }
            }
        }

    }


    return M;
}


// backup for mask


// This version of Null-space projection use position vector
/*
Eigen::MatrixXd ConstraintJacobianModel::computeObjectVelocityMask(
        const ObjectPointSet &current_configuration,
        const AllGrippersSinglePose& grippers_pose) const
{
    const ssize_t num_lines = num_nodes_ * 3;
    MatrixXd M(num_lines, num_lines);
    M.setIdentity(num_lines,num_lines);
    Matrix3d I3 = Matrix3d::Identity(3,3);
//    MatrixXd v_mask = MatrixXd::Zero(num_lines,1);

    const ssize_t num_grippers = (ssize_t)grippers_pose.size();

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
            ssize_t nearest_g_ind=0;
            double min_g_dis=0;

            // BEGINE: Gripper position vector version
            // Search for the gripper with min geodesic
            // for each gripper
            for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
            {
                const std::pair<double, long> nearest_node_on_gripper =
                        getMinimumDistanceIndexToGripper(
                            grippers_data_[(size_t)gripper_ind].node_indices,
                            node_ind, object_initial_node_distance_);
                if(min_g_dis>nearest_node_on_gripper.first)
                {
                    min_g_dis = nearest_node_on_gripper.first;
                    nearest_g_ind = nearest_node_on_gripper.second;
                }

            }
            // node_p_dot is actually the vector from the node to the grasped node,
            // its name here just for convinience.
            const Vector3d node_p_dot =  grippers_pose[(size_t)nearest_g_ind].translation() -
                    current_configuration.col(node_ind);
            // END of gripper position vector version

            // Node p Dot Version for Mask
//            const Vector3d node_p_dot = object_p_dot.segment<3>(node_ind);

            std::vector<double> sur_n
                    = environment_sdf_.GetGradient3d(current_configuration.col(node_ind));

            if(sur_n.size()>1)
            {
                Vector3d surface_normal= Vector3d::Map(sur_n.data(),sur_n.size());
                surface_normal = surface_normal/surface_normal.norm();
                // if node is moving outward from obstacle, unmask.
                if (node_p_dot.dot(surface_normal)<0)
                {
//                    v_mask.segment<3>(3*node_ind) = node_p_dot-node_p_dot.dot(surface_normal)*surface_normal;
                    const Matrix<double, 1, 3> surface_normal_inv
                            = EigenHelpers::Pinv(surface_normal, EigenHelpers::SuggestedRcond());
                    M.block<3,3>(node_ind*3,node_ind*3) = I3-surface_normal*surface_normal_inv;
                }
            }
            // Check with Dale, whether the vector is normalized.
//            const Matrix<double, 1, 3> surface_normal_inv
//                    = EigenHelpers::Pinv(surface_normal, EigenHelpers::SuggestedRcond());
//            M.block<3,3>(node_ind*3,node_ind*3) = I3-surface_normal*surface_normal_inv;
        }

    }
    return M;
}
*/


////////////////////////////////////////////////////////////////////////////////
// Candidate function to model rigidity weighting for translation Jacobian
////////////////////////////////////////////////////////////////////////////////

// Candidate function to count vector effect, return 3x3 matrix = diag{exp[beta(.,.)]}
Eigen::Matrix3d ConstraintJacobianModel::dirPropotionalModel(const Vector3d node_to_gripper, const Vector3d node_v) const
{
    Matrix3d beta_rigidity = Matrix3d::Zero();
    double dot1, dot2, dot3;
    if (node_to_gripper.norm() > 0.0001 && node_v.norm() > 0)
    {
        double dot_Value = node_to_gripper.normalized().dot(node_v);
        dot_Value = dot_Value - std::fabs(dot_Value);
        dot1 = dot_Value;
        dot2 = dot_Value;
        dot3 = dot_Value;
        /*
        dot1 = (dot_product(0)-std::fabs(dot_product(0)))/(node_to_gripper.norm()*node_v.norm());
        dot2 = (dot_product(1)-std::fabs(dot_product(1)))/(node_to_gripper.norm()*node_v.norm());
        dot3 = (dot_product(2)-std::fabs(dot_product(2)))/(node_to_gripper.norm()*node_v.norm());
        */
    }
    else if (node_v.norm() < 0.0001)
    {
        double dot_Value = 0;
        dot1 = dot_Value;
        dot2 = dot_Value;
        dot3 = dot_Value;
    }
    else
    {
        dot1 = 0;
        dot2 = 0;
        dot3 = 0;
    }

    beta_rigidity(0,0) = std::exp(translation_dir_deformability_ * dot1);
    beta_rigidity(1,1) = std::exp(translation_dir_deformability_ * dot2);
    beta_rigidity(2,2) = std::exp(translation_dir_deformability_ * dot3);
/*
    beta_rigidity(0,0) = std::exp(translation_dir_deformability_*(dot_product(0)-std::fabs(dot_product(0))));
    beta_rigidity(1,1) = std::exp(translation_dir_deformability_*(dot_product(1)-std::fabs(dot_product(1))));
    beta_rigidity(2,2) = std::exp(translation_dir_deformability_*(dot_product(2)-std::fabs(dot_product(2))));
*/

    /*
    beta_rigidity.row(0) = beta_rigidity.row(0)*std::exp(translation_dir_deformability_*dot1);
    beta_rigidity.row(1) = beta_rigidity.row(1)*std::exp(translation_dir_deformability_*dot2);
    beta_rigidity.row(2) = beta_rigidity.row(2)*std::exp(translation_dir_deformability_*dot3);

    beta_rigidity(1,1) = std::exp(dot_product(1)-std::fabs(dot_product(1)));
    beta_rigidity(2,2) = std::exp(dot_product(2)-std::fabs(dot_product(2)));
    beta_rigidity(3,3) = std::exp(dot_product(3)-std::fabs(dot_product(3)));
    */
    return beta_rigidity;
}


double ConstraintJacobianModel::disLinearModel(const double dist_to_gripper, const double dist_rest) const
{
    double ration;
    if (std::fabs(dist_rest) < 0.00001)
    {
        ration = 1;
    }
    else
    {
        ration = dist_to_gripper / dist_rest;
    }

//    return exp(-translation_dis_deformability_*(1-ration));

    return std::pow(ration,translation_dis_deformability_);

}
