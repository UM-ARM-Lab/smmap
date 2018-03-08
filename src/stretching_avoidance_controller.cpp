#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <smmap_utilities/nomad_solvers.h>
#include <kinematics_toolbox/kinematics.h>
#include <omp.h>

#include "smmap/stretching_avoidance_controller.h"
#include "smmap/ros_communication_helpers.hpp"
#include "smmap/grippers.hpp"

// Needed due to rounding problems
#define GRIPPER_COLLISION_REPULSION_MARGIN 0.0001
#define STRETCHING_COSINE_THRESHOLD_MARGIN 0.001

#define BARRIER_INNER_LOOP_CONVERGENCE_LIMIT 1e-6
#define BARRIER_CONVERGENCE_LIMIT 1e-4
#define BARRIER_UPDATE_RATE 10.0
#define BARRIER_VIOLATED_LARGE_COST 1e3

using namespace smmap;
using namespace smmap_utilities;




inline std::string print(const AllGrippersSinglePoseDelta& delta)
{
    assert(delta.size() == 2);

    ostringstream strm;
    strm << "0th: " << delta[0].transpose() << "     1st: " << delta[1].transpose();

    return strm.str();
}



// We want to constrain all vectors "r" to lie within a specified angle of the cone direction.
// I.e. cone_direction.transpose() * r / norm(r) >= cos(angle)
// or cone_direction.transpose() * r_normalized >= min_normalized_dot_product
// It is assumed that cone_direction is already normalized
// Returns the normal vectors that point out of a pyramid approximation of the cone
EigenHelpers::VectorVector3d convertConeToPyramid(const Eigen::Vector3d& cone_direction, const double min_normalized_dot_product)
{
    assert(false && "This function is not currently used");
    // First find vectors perpendicular to the cone direction
    Eigen::FullPivLU<Eigen::Vector3d> lu_decomp(cone_direction.transpose());
    const auto basis = lu_decomp.kernel();
//    const Eigen::Vector3d p1 = basis.col(0);
//    const Eigen::Vector3d p2 = basis.col(1);
//    const Eigen::Vector3d p3 = -p1;
//    const Eigen::Vector3d p4 = -p2;

//    EigenHelpers::VectorVector3d normals(4);
//    normals[0] = p1.cross(p2).normalized();
//    normals[1] = p2.cross(p3).normalized();
//    normals[2] = p3.cross(p4).normalized();
//    normals[3] = p4.cross(p1).normalized();

//    return normals;
}








StretchingAvoidanceController::StretchingAvoidanceController(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        const DeformableModel::Ptr& deformable_model,
        const RobotInterface::Ptr& robot,
        const sdf_tools::SignedDistanceField& sdf,
        std::mt19937_64& generator,
        const smmap_utilities::Visualizer::Ptr& vis,
        const StretchingAvoidanceControllerSolverType gripper_controller_type,
        const int max_count)
    : DeformableController(robot)
    , gripper_collision_checker_(nh)
    , grippers_data_(robot->getGrippersData())
    , environment_sdf_(sdf)
    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)
    , vis_(vis)
    , gripper_controller_type_(gripper_controller_type)
    , deformable_type_(GetDeformableType(nh))
    , task_type_(GetTaskType(nh))
    , model_(deformable_model)
    , max_node_distance_(GetMaxStretchFactor(ph) * EigenHelpers::CalculateDistanceMatrix(GetObjectInitialConfiguration(nh)))
    , max_node_squared_distance_(max_node_distance_.cwiseProduct(max_node_distance_))
    , distance_to_obstacle_threshold_(GetRobotGripperRadius())
    , stretching_cosine_threshold_(GetStretchingCosineThreshold(ph))
    , max_count_(max_count)
    , sample_count_(-1)
    , fix_step_(GetUseFixedGripperDeltaSize(ph))
    , over_stretch_(false)
    , log_file_path_(GetLogFolder(nh))
{}

////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////

DeformableController::OutputData StretchingAvoidanceController::getGripperMotion_impl(const InputData& input_data)
{
    switch (gripper_controller_type_)
    {
        case StretchingAvoidanceControllerSolverType::RANDOM_SAMPLING:
            return solvedByRandomSampling(input_data);
            break;

        case StretchingAvoidanceControllerSolverType::NOMAD_OPTIMIZATION:
            return solvedByNomad(input_data);
            break;

        case StretchingAvoidanceControllerSolverType::GRADIENT_DESCENT:
            return solvedByGradientDescent(input_data);
            break;

        default:
            assert(false && "This code should be un-reachable");
            break;
    };
}

/////////////////////////////////////////////////////////////////////////////////
// Private optimization function
/////////////////////////////////////////////////////////////////////////////////

//#define USE_MULTITHREADED_EVALUATION_FOR_SAMPLING_CONTROLLER 1

DeformableController::OutputData StretchingAvoidanceController::solvedByRandomSampling(const InputData& input_data)
{
    assert(false && "Not updated to use new constraints etc. Verify that this whole function is doing what we want, for both simulation and live robot");

    const WorldState& current_world_state = input_data.world_current_state_;
    const ssize_t num_grippers = (ssize_t)(current_world_state.all_grippers_single_pose_.size());
//    const ssize_t num_nodes = current_world_state.object_configuration_.cols();
    assert(num_grippers == 2 && "This function is only intended to be used with 2 grippers");

    const Eigen::VectorXd& desired_object_p_dot = input_data.desired_object_motion_.delta;
    const Eigen::VectorXd& desired_p_dot_weight = input_data.desired_object_motion_.weight;

    const Eigen::MatrixXd node_squared_distance =
            EigenHelpers::CalculateSquaredDistanceMatrix(current_world_state.object_configuration_);

    // Check object current stretching status
    // Checking the stretching status for current object configuration for once
    over_stretch_ = ((max_node_squared_distance_ - node_squared_distance).array() < 0.0).any();

    if (input_data.robot_jacobian_valid_)
    {
        assert(false && "Not implemented");
    }
    else
    {
        const double max_step_size = robot_->max_gripper_velocity_norm_ * robot_->dt_;

        std::vector<std::pair<AllGrippersSinglePoseDelta, double>> per_thread_optimal_command(
        #ifdef USE_MULTITHREADED_EVALUATION_FOR_SAMPLING_CONTROLLER
                    arc_helpers::GetNumOMPThreads(),
        #else
                    1,
        #endif
                    std::make_pair(AllGrippersSinglePoseDelta(), std::numeric_limits<double>::infinity()));

        // Check object current stretching status
        // Checking the stretching status for current object configuration for once
        over_stretch_ = ((max_node_squared_distance_ - node_squared_distance).array() < 0.0).any();

        #ifdef USE_MULTITHREADED_EVALUATION_FOR_SAMPLING_CONTROLLER
            #pragma omp parallel for
        #endif
        for (int64_t ind_count = 0; ind_count < max_count_; ind_count++)
        {
            AllGrippersSinglePoseDelta grippers_motion_sample = allGripperPoseDeltaSampler(num_grippers, max_step_size);

            #ifdef USE_MULTITHREADED_EVALUATION_FOR_SAMPLING_CONTROLLER
                const size_t thread_num = (size_t)omp_get_thread_num();
            #else
                const size_t thread_num = 0;
            #endif

            // Use constraint_violation checker for gripper collosion
            // Constraint violation checking here
            const bool collision_violation = gripperCollisionCheckResult(
                        current_world_state.all_grippers_single_pose_,
                        grippers_motion_sample);

            const bool stretching_violation = stretchingDetection(
                        input_data,
                        grippers_motion_sample);

            // If no constraint violation
            if (!collision_violation && !stretching_violation)
            {
                std::pair<AllGrippersSinglePoseDelta, double>& current_thread_optimal = per_thread_optimal_command[thread_num];

                // get predicted object motion
                ObjectPointSet predicted_object_p_dot = model_->getObjectDelta(
                            input_data.world_current_state_,
                            grippers_motion_sample);

                const double sample_error = errorOfControlByPrediction(predicted_object_p_dot,
                                                                 desired_object_p_dot,
                                                                 desired_p_dot_weight);

                // Compare if the sample grippers motion is better than the best to now
                if (sample_error < current_thread_optimal.second)
                {
                    current_thread_optimal.first = grippers_motion_sample;
                    current_thread_optimal.second = sample_error;
                }
            }
        }

        // Aggreate the results from each thread into a single best command
        double best_error = std::numeric_limits<double>::infinity();
        AllGrippersSinglePoseDelta optimal_gripper_motion;
        for (size_t thread_idx = 0; thread_idx < per_thread_optimal_command.size(); thread_idx++)
        {
            if (per_thread_optimal_command[thread_idx].second < best_error)
            {
                optimal_gripper_motion = per_thread_optimal_command[thread_idx].first;
                best_error = per_thread_optimal_command[thread_idx].second;
            }
        }

        if (sample_count_ >= 0)
        {
            sample_count_++;
            if (sample_count_ >= num_grippers)
            {
                sample_count_ = 0;
            }
        }
        if ((optimal_gripper_motion.size() == 0))
        {
            ROS_WARN("No valid samples generated, setting motion to zero.");
            const kinematics::Vector6d no_movement = kinematics::Vector6d::Zero();
            optimal_gripper_motion = AllGrippersSinglePoseDelta(num_grippers, no_movement);
        }

        const ObjectPointSet object_motion = model_->getObjectDelta(
                    input_data.world_current_state_,
                    optimal_gripper_motion);

        const OutputData suggested_robot_command(
                    optimal_gripper_motion,
                    object_motion,
                    Eigen::VectorXd());

        return suggested_robot_command;
    }
}

//#undef USE_MULTITHREADED_EVALUATION_FOR_SAMPLING_CONTROLLER

DeformableController::OutputData StretchingAvoidanceController::solvedByNomad(const InputData& input_data)
{
    assert(false && "Not updated to use new constraints etc. Verify that this whole function is doing what we want, for both simulation and live robot");

    const WorldState& current_world_state = input_data.world_current_state_;
    const ssize_t num_grippers = (ssize_t)(current_world_state.all_grippers_single_pose_.size());
    assert(num_grippers == 2 && "This function is only intended to be used with 2 grippers");

    const Eigen::VectorXd& desired_object_p_dot = input_data.desired_object_motion_.delta;
    const Eigen::VectorXd& desired_p_dot_weight = input_data.desired_object_motion_.weight;

    const Eigen::MatrixXd node_squared_distance =
            EigenHelpers::CalculateSquaredDistanceMatrix(current_world_state.object_configuration_);

    // Check object current stretching status
    // Checking the stretching status for current object configuration for once
    over_stretch_ = ((max_node_squared_distance_ - node_squared_distance).array() < 0.0).any();

    if (input_data.robot_jacobian_valid_)
    {
        std::cerr << "Using direct robot joint velocity version of optimization" << std::endl;

        const ssize_t num_dof = input_data.world_current_state_.robot_configuration_.size();

        // Determine the search space for NOMAD, at least in terms of the decision variables only
        const double max_step_size = robot_->max_dof_velocity_norm_ * robot_->dt_;
        const Eigen::VectorXd distance_to_lower_joint_limits =
                input_data.robot_->joint_lower_limits_ - input_data.world_current_state_.robot_configuration_;
        const Eigen::VectorXd min_joint_delta =
                distance_to_lower_joint_limits.unaryExpr([&max_step_size] (const double x) {return std::max(x, -max_step_size);});

        const Eigen::VectorXd distance_to_upper_joint_limits =
                input_data.robot_->joint_upper_limits_ - input_data.world_current_state_.robot_configuration_;
        const Eigen::VectorXd max_joint_delta =
                distance_to_upper_joint_limits.unaryExpr([&max_step_size] (const double x) {return std::min(x, max_step_size);});

        // Return value of objective function, cost = norm(p_dot_desired - p_dot_test)
        const std::function<double(const Eigen::VectorXd&)> eval_error_cost_fn = [&] (
                const Eigen::VectorXd& test_robot_motion)
        {
            const Eigen::VectorXd grippers_motion_as_single_vector =
                    input_data.robot_jacobian_ * test_robot_motion;

            if (grippers_motion_as_single_vector.size() != num_grippers * 6)
            {
                assert(false && "num of grippers not match");
            }

            AllGrippersSinglePoseDelta test_gripper_motion(num_grippers);
            for (ssize_t ind = 0; ind < num_grippers; ++ind)
            {
                test_gripper_motion[ind] = grippers_motion_as_single_vector.segment<6>(ind * 6);
            }

            const ObjectPointSet predicted_object_p_dot = model_->getObjectDelta(
                        input_data.world_current_state_,
                        test_gripper_motion);

            return errorOfControlByPrediction(predicted_object_p_dot,
                                              desired_object_p_dot,
                                              desired_p_dot_weight);
        };

        // Note that NOMAD wants all constraints in the form c(x) <= 0
        // Return the min distance of the points of interest to the obstacles, minus the required clearance
        const std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> poi_collision_data =
                robot_->getPointsOfInterestCollisionData(input_data.world_current_state_.robot_configuration_);
        const double required_obstacle_clearance = input_data.robot_->min_controller_distance_to_obstacles_;
        const std::function<double(const Eigen::VectorXd&)> collision_constraint_fn = [&] (
                const Eigen::VectorXd& test_robot_motion)
        {
            double min_poi_distance = std::numeric_limits<double>::max();
            for (size_t poi_ind = 0; poi_ind < poi_collision_data.size(); ++poi_ind)
            {
                const CollisionData& collision_data = poi_collision_data[poi_ind].first;
                const Eigen::Matrix3Xd& poi_jacobian = poi_collision_data[poi_ind].second;
                const double initial_distance = poi_collision_data[poi_ind].first.distance_to_obstacle_;

                const double current_distance = initial_distance +
                        collision_data.obstacle_surface_normal_.transpose() * poi_jacobian * test_robot_motion;

                min_poi_distance = std::min(min_poi_distance, current_distance);
            }

            return required_obstacle_clearance - min_poi_distance;
        };

        // Note that NOMAD wants all constraints in the form c(x) <= 0
        // Return the sum of cos (an indicator of direction) gripper motion to stretching vector
        const std::function<double(const Eigen::VectorXd&)> stretching_constraint_fn = [&] (
                const Eigen::VectorXd& test_robot_motion)
        {
            const Eigen::VectorXd grippers_motion_as_single_vector =
                    input_data.robot_jacobian_ * test_robot_motion;

            if (grippers_motion_as_single_vector.size() != num_grippers * 6)
            {
                assert(false && "num of grippers not match");
            }

            AllGrippersSinglePoseDelta test_gripper_motion(num_grippers);
            for (ssize_t ind = 0; ind < num_grippers; ++ind)
            {
                test_gripper_motion[ind] = grippers_motion_as_single_vector.segment<6>(ind * 6);
            }
            return stretchingFunctionEvaluation(input_data, test_gripper_motion);
        };

        // Prevents the robot from moving too quickly
        const std::function<double(const Eigen::VectorXd&)> robot_motion_constraint_fn = [&] (
                const Eigen::VectorXd& test_robot_motion)
        {
            return test_robot_motion.norm() - max_step_size;
        };

        std::cerr << "Invoking NOMAD wrapper" << std::endl;

        const Eigen::VectorXd optimal_robot_motion =
                smmap_utilities::minFunctionPointerDirectRobotDOF(
                    log_file_path_,
                    fix_step_,
                    max_count_,
                    num_dof,
                    min_joint_delta,
                    max_joint_delta,
                    generator_,
                    uniform_unit_distribution_,
                    eval_error_cost_fn,
                    collision_constraint_fn,
                    stretching_constraint_fn,
                    robot_motion_constraint_fn);

        const Eigen::VectorXd grippers_motion_as_single_vector =
                input_data.robot_jacobian_ * optimal_robot_motion;

        if (grippers_motion_as_single_vector.size() != num_grippers * 6)
        {
            assert(false && "num of grippers not match");
        }

        AllGrippersSinglePoseDelta optimal_gripper_motion(num_grippers);
        for (ssize_t ind = 0; ind < num_grippers; ++ind)
        {
            optimal_gripper_motion[ind] = grippers_motion_as_single_vector.segment<6>(ind * 6);
        }

        const ObjectPointSet object_motion = model_->getObjectDelta(
                    input_data.world_current_state_,
                    optimal_gripper_motion);

        const OutputData suggested_robot_command(
                    optimal_gripper_motion,
                    object_motion,
                    optimal_robot_motion);

        return suggested_robot_command;
    }
    else
    {
        std::cerr << "Using pure tan(SE3) velocity version of optimization" << std::endl;

        const double max_step_size = robot_->max_gripper_velocity_norm_ * robot_->dt_;

        // Return value of objective function, cost = norm(p_dot_desired - p_dot_test)
        const std::function<double(const AllGrippersSinglePoseDelta&)> eval_error_cost_fn = [&] (
                const AllGrippersSinglePoseDelta& test_gripper_motion)
        {
            const ObjectPointSet predicted_object_p_dot = model_->getObjectDelta(
                        input_data.world_current_state_,
                        test_gripper_motion);

            return errorOfControlByPrediction(predicted_object_p_dot,
                                              desired_object_p_dot,
                                              desired_p_dot_weight);
        };

        // Return the min distance of gripper to obstacle, minus the gripper radius
        const double required_obstacle_clearance = GetRobotGripperRadius();
        const std::function<double(const AllGrippersSinglePoseDelta&)> collision_constraint_fn = [&] (
                const AllGrippersSinglePoseDelta& test_gripper_motion)
        {
            const double min_dis_to_obstacle = gripperCollisionCheckHelper(
                        current_world_state.all_grippers_single_pose_,
                        test_gripper_motion);

            return required_obstacle_clearance - min_dis_to_obstacle;
        };

        // Return the sum of cos (an indicator of direction) gripper motion to stretching vector
        const std::function<double(const AllGrippersSinglePoseDelta&)> stretching_constraint_fn = [&] (
                const AllGrippersSinglePoseDelta& test_gripper_motion)
        {
            return stretchingFunctionEvaluation(input_data, test_gripper_motion);
        };

        // Prevents the grippers from moving too quickly
        const std::function<double(const AllGrippersSinglePoseDelta&)> gripper_motion_constraint_fn = [&] (
                const AllGrippersSinglePoseDelta& test_gripper_motion)
        {
            double max_value = 0.0;
            for (size_t gripper_ind = 0; gripper_ind < test_gripper_motion.size(); gripper_ind += 6)
            {
                const double velocity_norm = GripperVelocity6dNorm(test_gripper_motion[gripper_ind]);
                if (velocity_norm > max_value)
                {
                    max_value = velocity_norm;
                }
            }
            return max_value - max_step_size;
        };

        std::cerr << "Invoking NOMAD wrapper" << std::endl;

        const AllGrippersSinglePoseDelta optimal_gripper_motion =
                smmap_utilities::minFunctionPointerSE3Delta(
                    log_file_path_,
                    fix_step_,
                    max_count_,
                    num_grippers,
                    max_step_size,
                    generator_,
                    uniform_unit_distribution_,
                    eval_error_cost_fn,
                    collision_constraint_fn,
                    stretching_constraint_fn,
                    gripper_motion_constraint_fn);

        const ObjectPointSet object_motion = model_->getObjectDelta(
                    input_data.world_current_state_,
                    optimal_gripper_motion);

        const OutputData suggested_robot_command(
                    optimal_gripper_motion,
                    object_motion,
                    Eigen::VectorXd());

        return suggested_robot_command;
    }
}













inline Eigen::VectorXd projectToMaxDeltaConstraints(
        const Eigen::VectorXd& delta,
        const double max_step_size,
        const Eigen::VectorXd& min_joint_delta,
        const Eigen::VectorXd& max_joint_delta)
{
    auto max_norm_constrainted = delta;

    // Project to max norm
    const double starting_norm = delta.norm();
    if (starting_norm > max_step_size)
    {
        max_norm_constrainted *= (max_step_size / starting_norm);
    }

    // Project to joint limits
    auto min_delta_constrained = max_norm_constrainted.cwiseMax(min_joint_delta);
    auto max_delta_constrained = min_delta_constrained.cwiseMin(max_joint_delta);

    return max_delta_constrained;
}



// Calculates left side of: dmin - d - normal' * J * dq <= 0
//   In the form of:               b -           M * dq <= 0
inline double collisionConstraintFunction(
        const double b,
        const Eigen::RowVectorXd& M,
        const Eigen::VectorXd& dq)
{
    return b - M * dq;
}



inline double barrier(const double u)
{
    return -std::log(-u);
}

inline double barrier(const double t, const double u)
{
    return -(1.0/t) * std::log(-u);
}

Eigen::VectorXd evaluateJointLimitsMaxDeltaConstraintsBarrier(
        const Eigen::VectorXd& dq,
        const Eigen::VectorXd& dq_min,
        const Eigen::VectorXd& dq_max,
        const double max_step_size,
        const double t)
{
    assert(dq.size() == dq_min.size());
    assert(dq.size() == dq_max.size());
    const auto u_min = dq_min - dq;
    const auto u_max = dq - dq_max;

    Eigen::VectorXd results(dq.size() * 2 + 1);
    for (ssize_t ind = 0; ind < dq.size(); ++ind)
    {
        results(ind) = barrier(t, u_min(ind));
        results(dq.size() + ind) = barrier(t, u_max(ind));
    }
    results(dq.size() * 2) = barrier(t, dq.squaredNorm() - max_step_size * max_step_size);
    return results;
}

Eigen::VectorXd evaluateCollisionConstraintsBarrier(
        const std::vector<double>& b_offsets,
        const std::vector<Eigen::RowVectorXd>& M_matrices,
        const Eigen::VectorXd& dq,
        const double t)
{
    assert(b_offsets.size() == M_matrices.size());
    Eigen::VectorXd results(b_offsets.size());
    for (size_t ind = 0; ind < b_offsets.size(); ++ind)
    {
        const double u = collisionConstraintFunction(b_offsets[ind], M_matrices[ind], dq);
        results[ind] = barrier(t, u);
    }
    return results;
}

template <typename T>
Eigen::VectorXd evaluateStretchingConstraintsBarrier(
        const AllGrippersSinglePoseDelta& gripper_motion,
        const T& stretching_evaluation_fn,
        const double t)
{
    Eigen::VectorXd result(1);
    const double stretching_constraint_val = stretching_evaluation_fn(gripper_motion);
    std::cout << "stretching_constraint_val: " << stretching_constraint_val << std::endl;
    result(0) = barrier(t, stretching_constraint_val);
    return result;
}




inline AllGrippersSinglePoseDelta stepInDirection(
        const AllGrippersSinglePoseDelta& start,
        const AllGrippersSinglePoseDelta& direction,
        const double step_size)
{
    assert(start.size() == direction.size());

    AllGrippersSinglePoseDelta result(start.size());
    for (size_t ind = 0; ind < start.size(); ++ind)
    {
        result[ind] = start[ind] + step_size * direction[ind];
    }
    return result;
}

inline AllGrippersSinglePoseDelta stepInDirection(
        const AllGrippersSinglePoseDelta& start,
        const Eigen::VectorXd& direction,
        const double step_size)
{
    AllGrippersSinglePoseDelta dir_as_eigen = EigenHelpers::EigenVectorXToVectorEigenVector<double, 6>(direction);
    return stepInDirection(start, dir_as_eigen, step_size);
}

inline AllGrippersSinglePoseDelta stepInDirection(
        const AllGrippersSinglePoseDelta& start,
        const std::vector<double>& direction,
        const double step_size)
{
    AllGrippersSinglePoseDelta dir_as_eigen = EigenHelpers::StdVectorXToVectorEigenVector<double, 6>(direction);
    return stepInDirection(start, dir_as_eigen, step_size);
}

// Note that this assumes that the projectino is needed, and is already on the boundary of the collision constraint
inline kinematics::Vector6d projectToCornerOfCollisionAndMaxDeltaConstraints(
        const kinematics::Vector6d & delta,
        const Eigen::Matrix<double, 3, 6>& collision_jacobian,
        const double max_step_size)
{
    const Eigen::Matrix<double, 6, 3> collision_jacobian_inv = EigenHelpers::Pinv(collision_jacobian, EigenHelpers::SuggestedRcond());
    const auto nullspace_projector = Eigen::Matrix<double, 6, 6>::Identity() - collision_jacobian_inv * collision_jacobian;

    const auto unnormalized_direction_along_line = nullspace_projector * delta;
    const double direction_norm = GripperVelocity6dNorm(unnormalized_direction_along_line);
    assert(direction_norm > 0.0);
    const auto direction_along_line = unnormalized_direction_along_line / direction_norm;
    const auto& point_on_line = delta;

    // Math taken from here: https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
    // They have an error in the simplification, but definition of a, b, and c is correct
//    const double a = 1.0;
    const auto b = 2.0 * GripperVelocityDotProduct(direction_along_line, point_on_line);
    const auto c = GripperVelocity6dSquaredNorm(point_on_line) - max_step_size * max_step_size;
    const auto discriminant = std::max(0.0, b * b - 4 * c);

//    std::cout << "b: " << b << std::endl;
//    std::cout << "c: " << c << std::endl;
//    std::cout << "d: " << discriminant << std::endl;

    const auto distance_along_line_plus = (-b + std::sqrt(discriminant)) / 2.0;
    const auto distance_along_line_minus = (-b - std::sqrt(discriminant)) / 2.0;

//    std::cout << "minus: " << distance_along_line_minus << std::endl;
//    std::cout << "plus: " << distance_along_line_plus << std::endl;
//    std::cout << "b*b - 4*c: " << b * b - 4 * c << std::endl;

    // Accept the motion that is smaller (i.e. closer to the start point)
    if (std::abs(distance_along_line_plus) < std::abs(distance_along_line_minus))
    {
        return delta + distance_along_line_plus * direction_along_line;
    }
    else
    {
        return delta + distance_along_line_minus * direction_along_line;
    }
}

inline kinematics::Vector6d projectToCollisionAndMaxDeltaConstraints(
        const Eigen::Isometry3d& starting_pose,
        const kinematics::Vector6d& delta,
        const CollisionData& collision_data,
        const double min_distance,
        const double max_step_size)
{
    const Eigen::Matrix<double, 3, 6> J_collision =
            ComputeGripperMotionToPointMotionJacobian(
                collision_data.nearest_point_to_obstacle_, starting_pose);

    // Gripper DOF direction to increase distance from the obstacle
    const auto J_distance = collision_data.obstacle_surface_normal_.transpose() * J_collision;
    const auto J_distance_pinv = EigenHelpers::Pinv(J_distance, EigenHelpers::SuggestedRcond());

    // First check if the intersection of valid motion is empty; if it is empty;
    // take the least violating motion (max speed, directly away from collision) as the result of projection
    {
        // If the distance is already far enough away, then we are already satisfying the constraint with zero movement,
        // thus the valid motion region cannot be empty
        const double current_distance = collision_data.distance_to_obstacle_;
        const double current_distance_change_needed = std::max(0.0, min_distance + GRIPPER_COLLISION_REPULSION_MARGIN - current_distance);
        const auto min_gripper_delta_to_satisfy_collision_constraint = J_distance_pinv * current_distance_change_needed;
        const double norm = GripperVelocity6dNorm(min_gripper_delta_to_satisfy_collision_constraint);
        if (norm > max_step_size)
        {
            return min_gripper_delta_to_satisfy_collision_constraint * (max_step_size / norm);
        }
    }

    // Second option, project to the delta ball first.
    // If, after projecting, we are violating the collision constraint,
    // then we need to find the "corner" between the circle and the half plane
    {
        const double delta_norm = GripperVelocity6dNorm(delta);
        if (delta_norm > max_step_size)
        {
            const auto max_motion_constrained_delta = delta * (max_step_size / delta_norm);
            const double displacement_towards_obstacle =  J_distance * max_motion_constrained_delta;
            const double current_distance = collision_data.distance_to_obstacle_ + displacement_towards_obstacle;
            if (current_distance >= min_distance)
            {
                return max_motion_constrained_delta;
            }
            else
            {
                const double current_distance_change_needed = min_distance + GRIPPER_COLLISION_REPULSION_MARGIN - current_distance;
                const auto collision_constrained_delta_update = J_distance_pinv * current_distance_change_needed;
                const auto collision_constrained_delta = max_motion_constrained_delta + collision_constrained_delta_update;
                const auto collision_constrained_displacement = J_distance * collision_constrained_delta;

                assert(collision_constrained_displacement.size() == 1);
                assert(collision_data.distance_to_obstacle_ + collision_constrained_displacement(0) >= min_distance);

                return projectToCornerOfCollisionAndMaxDeltaConstraints(
                            collision_constrained_delta,
                            J_collision,
                            max_step_size);
            }
        }
    }

    // We did not need to project to the delta ball, so project to the collision constraint.
    // If, after projecting, we are violating the max delta constraint
    // then we need to find the "corner" between the circle and the half plane
    {
        const double displacement_towards_obstacle =  J_distance * delta;
        const double current_distance = collision_data.distance_to_obstacle_ + displacement_towards_obstacle;
        if (current_distance < min_distance)
        {
            const double current_distance_change_needed = min_distance + GRIPPER_COLLISION_REPULSION_MARGIN - current_distance;
            const auto collision_constrained_delta = J_distance_pinv * current_distance_change_needed + delta;
            const auto collision_constrained_displacement = J_distance * collision_constrained_delta;

            assert(collision_constrained_displacement.size() == 1);
            assert(collision_data.distance_to_obstacle_ + collision_constrained_displacement(0) >= min_distance);

            const double delta_norm = GripperVelocity6dNorm(collision_constrained_delta);
            if (delta_norm <= max_step_size)
            {
                return collision_constrained_delta;
            }
            else
            {
                return projectToCornerOfCollisionAndMaxDeltaConstraints(
                            collision_constrained_delta,
                            J_collision,
                            max_step_size);
            }
        }
    }

    // If no constraints are violated, then return the original value unchanged
    return delta;
}

inline AllGrippersSinglePoseDelta projectToCollisionAndMaxDeltaConstraints(
        const AllGrippersSinglePose& starting_poses,
        const AllGrippersSinglePoseDelta& delta,
        const std::vector<CollisionData>& collision_data,
        const double min_distance,
        const double max_step_size)
{
    assert(delta.size() == starting_poses.size());
    assert(delta.size() == collision_data.size());

    AllGrippersSinglePoseDelta result(delta.size());
    for (size_t ind = 0; ind < delta.size(); ++ind)
    {
        result[ind] = projectToCollisionAndMaxDeltaConstraints(starting_poses[ind], delta[ind], collision_data[ind], min_distance, max_step_size);
    }
    return result;
}








DeformableController::OutputData StretchingAvoidanceController::solvedByGradientDescent(const InputData& input_data)
{
    // Unpack the input data into its constituent parts
    const auto& world_state = input_data.world_current_state_;
    const auto& object_config = world_state.object_configuration_;
    const auto& grippers_poses = world_state.all_grippers_single_pose_;
    const auto& collision_data = world_state.gripper_collision_data_;
    const ssize_t num_grippers = (ssize_t)(grippers_poses.size());
    assert(num_grippers == 2 && "This function is only intended to be used with 2 grippers");

    const Eigen::VectorXd& desired_object_p_dot = input_data.desired_object_motion_.delta;
    const Eigen::VectorXd& desired_p_dot_weight = input_data.desired_object_motion_.weight;

    // Check object current stretching status
    const Eigen::MatrixXd node_squared_distance = EigenHelpers::CalculateSquaredDistanceMatrix(object_config);
    over_stretch_ = ((max_node_squared_distance_ - node_squared_distance).array() < 0.0).any();

    if (input_data.robot_jacobian_valid_)
    {
        assert(false && "Not implemented");
    }
    else
    {
        //////////// First, find an initial feasible point //////////////////////
        const double max_individual_gripper_step_size = robot_->max_gripper_velocity_norm_ * robot_->dt_;
        const auto stretching_constraint_data = stretchingCorrectionVectorsAndPoints(input_data);

        std::cout << "Initializing variables and functions for optimization use" << std::endl;

        //////////// Construct the collision constraint functions ///////////////
        const auto J_collision_g0 = ComputeGripperMotionToPointMotionJacobian(collision_data[0].nearest_point_to_obstacle_, grippers_poses[0]);
        const auto J_distance_g0 = collision_data[0].obstacle_surface_normal_.transpose() * J_collision_g0;
        // Returns true if the constraint is satisfied
        const auto collision_constraint_fn_g0 = [&] (const kinematics::Vector6d& gripper_delta)
        {
            return robot_->min_controller_distance_to_obstacles_ - collision_data[0].distance_to_obstacle_ - J_distance_g0 * gripper_delta;
        };

        const auto J_collision_g1 = ComputeGripperMotionToPointMotionJacobian(collision_data[1].nearest_point_to_obstacle_, grippers_poses[1]);
        const auto J_distance_g1 = collision_data[0].obstacle_surface_normal_.transpose() * J_collision_g1;
        // Returns true if the constraint is satisfied
        const auto collision_constraint_fn_g1 = [&] (const kinematics::Vector6d& gripper_delta)
        {
            return robot_->min_controller_distance_to_obstacles_ - collision_data[1].distance_to_obstacle_ - J_distance_g1 * gripper_delta;
        };

        /////////// Construct the stretching constraint functions ///////////////
        // These functions return a large negative if the constraint is not active
        // to indicate that the constraint is as satisfied as it can possibly be
        const auto stretching_constraint_fn_g0 = [&] (const kinematics::Vector6d& gripper_delta)
        {
            const auto& stretching_reduction_vector              = stretching_constraint_data[0].first;
            const auto& vector_from_gripper_to_translation_point = stretching_constraint_data[0].second;

            if (!over_stretch_)
            {
                return -1000.0;
            }
            else
            {
                auto r_dot = gripper_delta.head<3>() + gripper_delta.tail<3>().cross(vector_from_gripper_to_translation_point);
                auto r_dot_norm = r_dot.norm();
                if (r_dot_norm < 1e-6)
                {
                    return stretching_cosine_threshold_ - 1.0;
                }
                else
                {
                    const auto cos_angle = stretching_reduction_vector.dot(r_dot) / r_dot_norm;
                    return  stretching_cosine_threshold_ - cos_angle;
                }
            }
        };
        const auto stretching_constraint_fn_g1 = [&] (const kinematics::Vector6d& gripper_delta)
        {
            const auto& stretching_reduction_vector              = stretching_constraint_data[1].first;
            const auto& vector_from_gripper_to_translation_point = stretching_constraint_data[1].second;

            if (!over_stretch_)
            {
                return -1000.0;
            }
            else
            {
                const auto r_dot = gripper_delta.head<3>() + gripper_delta.tail<3>().cross(vector_from_gripper_to_translation_point);
                const auto r_dot_norm = r_dot.norm();
                if (r_dot_norm < 1e-6)
                {
                    return stretching_cosine_threshold_ - 1.0;
                }
                else
                {
                    const auto cos_angle = stretching_reduction_vector.dot(r_dot) / r_dot_norm;
                    return  stretching_cosine_threshold_ - cos_angle;
                }
            }
        };

        /////////// Construct the max vel constraint function //////////////////
        const auto individual_max_vel_constraint_fn = [&] (const kinematics::Vector6d& gripper_delta)
        {
            return GripperVelocity6dSquaredNorm(gripper_delta) - max_individual_gripper_step_size * max_individual_gripper_step_size;
        };

        /////////// Construct the augmented objective function /////////////////
        const auto constraint_barrier_fn = [&] (const AllGrippersSinglePoseDelta& gripper_motion)
        {
//            Eigen::VectorXd raw_constraints(6);

//            raw_constraints(0) = collision_constraint_fn_g0(gripper_motion[0]);
//            raw_constraints(1) = collision_constraint_fn_g1(gripper_motion[1]);
//            raw_constraints(2) = stretching_constraint_fn_g0(gripper_motion[0]);
//            raw_constraints(3) = stretching_constraint_fn_g1(gripper_motion[1]);

//            // Note that we are artificially downweighting the max vel constraints to allow us to move much closer to the max velocity
//            raw_constraints(4) = individual_max_vel_constraint_fn(gripper_motion[0]);
//            raw_constraints(5) = individual_max_vel_constraint_fn(gripper_motion[1]);

//            std::cout << "Raw constraints inside barrier calc: " << raw_constraints.transpose() << std::endl;

            const auto collision_constraint_g0_barrier_val = barrier(collision_constraint_fn_g0(gripper_motion[0]));
            const auto collision_constraint_g1_barrier_val = barrier(collision_constraint_fn_g1(gripper_motion[1]));
            const auto stretching_constraint_g0_barrier_val = barrier(stretching_constraint_fn_g0(gripper_motion[0]));
            const auto stretching_constraint_g1_barrier_val = barrier(stretching_constraint_fn_g1(gripper_motion[1]));

            // Note that we are artificially downweighting the max vel constraints to allow us to move much closer to the max velocity
            const auto max_delta_g0_barrier_val = barrier(individual_max_vel_constraint_fn(gripper_motion[0])) / 50.0;
            const auto max_delta_g1_barrier_val = barrier(individual_max_vel_constraint_fn(gripper_motion[1])) / 50.0;

            Eigen::Matrix<double, 6, 1> result;
            result(0) = collision_constraint_g0_barrier_val;
            result(1) = collision_constraint_g1_barrier_val;
            result(2) = stretching_constraint_g0_barrier_val;
            result(3) = stretching_constraint_g1_barrier_val;
            result(4) = max_delta_g0_barrier_val;
            result(5) = max_delta_g1_barrier_val;

//            std::cout << "Barrier results inside barrier calc: " << result.transpose() << std::endl;

            return result;
        };
        const auto objective_fn = [&] (const ObjectPointSet& delta)
        {
            return this->errorOfControlByPrediction(delta, desired_object_p_dot, desired_p_dot_weight);
        };

        std::cout << "Generating valid initial feasible point" << std::endl;
        const AllGrippersSinglePoseDelta feasible_starting_gripper_motion =
        {
            getConstraintAwareGripperDeltaSample(grippers_poses[0], collision_data[0], max_individual_gripper_step_size, stretching_constraint_data[0].first, stretching_constraint_data[0].second),
            getConstraintAwareGripperDeltaSample(grippers_poses[1], collision_data[1], max_individual_gripper_step_size, stretching_constraint_data[1].first, stretching_constraint_data[1].second)
        };

        /////////// Gradient descent variables and function pointers ////////////
        const double differencing_step_size = max_individual_gripper_step_size / 100.0;
        const double initial_gradient_step_size = max_individual_gripper_step_size / 2.0;
        double barrier_t = 40.0;
        auto gripper_motion = feasible_starting_gripper_motion;
        auto object_delta = model_->getObjectDelta(world_state, gripper_motion);
        auto constraint_error = constraint_barrier_fn(gripper_motion);
        auto objective_error = objective_fn(object_delta);
        auto combined_error = barrier_t * objective_error + constraint_error.sum();

        // Make sure that our starting point is in fact feasible
        if ((constraint_error.array().isNaN().any()))
        {
            std::cout << "!!!!!!!!!! Invalid starting point!!!!!!!!!\n";
            std::cout << "Starting gripper motion:  " << print(gripper_motion) << std::endl;

            Eigen::VectorXd raw_constraints(6);

            raw_constraints(0) = collision_constraint_fn_g0(gripper_motion[0]);
            raw_constraints(1) = collision_constraint_fn_g1(gripper_motion[1]);
            raw_constraints(2) = stretching_constraint_fn_g0(gripper_motion[0]);
            raw_constraints(3) = stretching_constraint_fn_g1(gripper_motion[1]);

            // Note that we are artificially downweighting the max vel constraints to allow us to move much closer to the max velocity
            raw_constraints(4) = individual_max_vel_constraint_fn(gripper_motion[0]);
            raw_constraints(5) = individual_max_vel_constraint_fn(gripper_motion[1]);

            std::cout << "Raw constraints:   " << raw_constraints.transpose() << std::endl;
            std::cout << "Barrier functions: " << constraint_error.transpose() << std::endl;
            constraint_barrier_fn(gripper_motion);
            assert(false && "Invalid starting point for barrier function");
        }
        else
        {
            std::cout << "Starting gripper motion: " << print(gripper_motion) << std::endl;
            std::cout << "Barrier functions: " << constraint_error.transpose() << std::endl;
        }


        do
        {
            std::cout << "Start of outer loop, objective val = " << combined_error << std::endl;

            bool inner_loop_converged = false;
            do
            {
                std::cout << "\tStart of inner loop, objective val = " << combined_error << std::endl;

//                std::cout << "Finding objective gradient" << std::endl;
                Eigen::VectorXd objective_numerical_gradient(num_grippers * 6);
                for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; ++gripper_ind)
                {
                    for (size_t single_gripper_dir_ind = 0; single_gripper_dir_ind < 6; ++single_gripper_dir_ind)
                    {
                        const auto test_input_ind = gripper_ind * 6 + single_gripper_dir_ind;

                        AllGrippersSinglePoseDelta local_test_motion = gripper_motion;
                        local_test_motion[gripper_ind](single_gripper_dir_ind) += differencing_step_size;

                        // Ensure that the test motion gives us a valid gradient
                        auto test_constraint_error = constraint_barrier_fn(local_test_motion);
                        for (ssize_t ind = 0; ind < test_constraint_error.size(); ++ind)
                        {
                            if (std::isnan(test_constraint_error(ind)))
                            {
                                test_constraint_error(ind) = BARRIER_VIOLATED_LARGE_COST;
                                std::cout << "\t\tGradient calc: barrier violated for test_ind: " << test_input_ind << " constraint ind: " << ind << std::endl;
                            }
                        }

                        const auto predicted_object_delta = model_->getObjectDelta(world_state, local_test_motion);
                        const auto test_objective_error = objective_fn(predicted_object_delta);
                        const auto test_combined_error = barrier_t * test_objective_error + test_constraint_error.sum();
                        objective_numerical_gradient(test_input_ind) = test_combined_error - combined_error;
                    }
                }

                // Normalize the gradient as it is just giving us a direction to move, not a distance to move
                if (objective_numerical_gradient.norm() > 1e-6)
                {
                    objective_numerical_gradient.normalize();
                }
                else
                {
                    std::cout << "\tObjective gradient is effectively flat, exiting inner loop" << std::endl;
                    inner_loop_converged = true;
                    continue;
                }

//                std::cout << "\t\tGradient: " << objective_numerical_gradient.transpose() << std::endl;

                // Find a downhill step
//                std::cout << "\t\tFinding a downhill step" << std::endl;
                auto next_gripper_motion = gripper_motion;
                auto next_object_delta = object_delta;

                int downhill_attempt_ind = 0;
                auto objective_gradient_step_size = -initial_gradient_step_size;
                auto next_constraint_error = constraint_error;
                auto next_objective_error = objective_error;
                auto next_combined_error = combined_error;
                do
                {
                    next_gripper_motion = stepInDirection(gripper_motion, objective_numerical_gradient, objective_gradient_step_size);
                    next_constraint_error = constraint_barrier_fn(next_gripper_motion);
                    // If we violate a constraint, then reduce the stepsize and try again
                    if (next_constraint_error.array().isNaN().any())
                    {
                        ++downhill_attempt_ind;
                        objective_gradient_step_size *= 0.7;
                        continue;
                    }

                    next_object_delta = model_->getObjectDelta(world_state, next_gripper_motion);
                    next_objective_error = objective_fn(next_object_delta);
                    next_combined_error = barrier_t * next_objective_error + next_constraint_error.sum();

                    ++downhill_attempt_ind;
                    objective_gradient_step_size *= 0.7;
                }
                while (next_combined_error > combined_error && downhill_attempt_ind < 10);

                // Check convergence
                inner_loop_converged = (combined_error - next_combined_error) < std::abs(combined_error) * barrier_t * BARRIER_INNER_LOOP_CONVERGENCE_LIMIT;

                // If we made progress, record the update
                if (next_combined_error < combined_error)
                {
                    gripper_motion = next_gripper_motion;
                    object_delta = next_object_delta;
                    constraint_error = next_constraint_error;
                    objective_error = next_objective_error;
                    combined_error = next_combined_error;
                    std::cout << "\t\tAccepting motion update: " << print (gripper_motion) << std::endl;
                }
                else
                {
                    std::cout << "\t\tUnable to find downhill motion" << std::endl;
                }

                std::cout << "\tEnd of inner loop,objective val =   " << combined_error << std::endl;
            }
            while (!inner_loop_converged);

            std::cout << "End of outer loop, objective val =   " << combined_error << std::endl;

            barrier_t *= BARRIER_UPDATE_RATE;
            combined_error = barrier_t * objective_error + constraint_error.sum();

            std::cout << "Updated barrier t, objective val =   " << combined_error << " Barrier t: " << barrier_t << std::endl;
        }
        while ((double)(constraint_error.size()) / barrier_t >= BARRIER_CONVERGENCE_LIMIT);

        return OutputData(gripper_motion, object_delta, Eigen::VectorXd());
    }
}






#warning "Magic numbers all throughout this function"
DeformableController::OutputData StretchingAvoidanceController::solvedByGradientDescentOld(const InputData& input_data)
{
    const WorldState& current_world_state = input_data.world_current_state_;
    const ssize_t num_grippers = (ssize_t)(current_world_state.all_grippers_single_pose_.size());
    assert(num_grippers == 2 && "This function is only intended to be used with 2 grippers");

    const Eigen::VectorXd& desired_object_p_dot = input_data.desired_object_motion_.delta;
    const Eigen::VectorXd& desired_p_dot_weight = input_data.desired_object_motion_.weight;

    const Eigen::MatrixXd node_squared_distance =
            EigenHelpers::CalculateSquaredDistanceMatrix(current_world_state.object_configuration_);

    // Check object current stretching status
    // Checking the stretching status for current object configuration for once
    over_stretch_ = ((max_node_squared_distance_ - node_squared_distance).array() < 0.0).any();

    if (input_data.robot_jacobian_valid_)
    {
        // Basic robot data
        const ssize_t num_dof = input_data.world_current_state_.robot_configuration_.size();
        const auto joint_motion_to_gripper_motion_fn = [&](const Eigen::VectorXd& robot_motion)
        {
            const Eigen::VectorXd grippers_motion_as_single_vector = input_data.robot_jacobian_ * robot_motion;

            if (grippers_motion_as_single_vector.size() != num_grippers * 6)
            {
                assert(false && "num of grippers not match");
            }

            AllGrippersSinglePoseDelta gripper_motion(num_grippers);
            for (ssize_t ind = 0; ind < num_grippers; ++ind)
            {
                gripper_motion[ind] = grippers_motion_as_single_vector.segment<6>(ind * 6);
            }

            return gripper_motion;
        };

        // Determine the search space for Gradient Descent, at least in terms of the decision variables only
        const double max_step_size = robot_->max_dof_velocity_norm_ * robot_->dt_;

        // Lower limit
        const Eigen::VectorXd distance_to_lower_joint_limits = input_data.robot_->joint_lower_limits_ - input_data.world_current_state_.robot_configuration_;
        const Eigen::VectorXd min_joint_delta = distance_to_lower_joint_limits.unaryExpr([&max_step_size] (const double x) {return std::max(x, -max_step_size);});

        // Upper limit
        const Eigen::VectorXd distance_to_upper_joint_limits = input_data.robot_->joint_upper_limits_ - input_data.world_current_state_.robot_configuration_;
        const Eigen::VectorXd max_joint_delta = distance_to_upper_joint_limits.unaryExpr([&max_step_size] (const double x) {return std::min(x, max_step_size);});

        // Collect the collision data needed
        const std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> poi_collision_data =
                robot_->getPointsOfInterestCollisionData(input_data.world_current_state_.robot_configuration_);
        const double required_obstacle_clearance = input_data.robot_->min_controller_distance_to_obstacles_;
        // Calculate the matrics needed once for the projection
        std::vector<Eigen::RowVectorXd> M_matrices(poi_collision_data.size());
        std::vector<double> b_offsets(poi_collision_data.size());
        for (size_t ind = 0; ind < poi_collision_data.size(); ++ind)
        {
            M_matrices[ind] = poi_collision_data[ind].first.obstacle_surface_normal_.transpose() * poi_collision_data[ind].second;
            b_offsets[ind] = required_obstacle_clearance - poi_collision_data[ind].first.distance_to_obstacle_;
        }

        // Gradient descent variables and function pointers
        const double differencing_step_size = max_step_size / 100.0;
        const double initial_gradient_step_size = max_step_size / 10.0;
        double barrier_t = 40.0;
        const auto stretching_constraint_fn = [&] (const AllGrippersSinglePoseDelta& gripper_motion)
        {
//            std::cout << "Stretching constraint fn: motion size: " << gripper_motion.size() << std::endl;
//            std::cout << print(gripper_motion) << std::endl;
            return stretchingFunctionEvaluation(input_data, gripper_motion);
        };
        const auto constraint_barrier_fn = [&] (const Eigen::VectorXd& robot_motion, const AllGrippersSinglePoseDelta& gripper_motion)
        {
//            std::cout << "Joint limits" << std::endl;
            auto joint_limits_max_delta_val = evaluateJointLimitsMaxDeltaConstraintsBarrier(robot_motion, min_joint_delta, max_joint_delta, max_step_size, barrier_t);
//            std::cout << "collision constraints" << std::endl;
            auto collision_constraints_val = evaluateCollisionConstraintsBarrier(b_offsets, M_matrices, robot_motion, barrier_t);
//            std::cout << "stretching constraints" << std::endl;
            auto stretching_constraint_val = evaluateStretchingConstraintsBarrier(gripper_motion, stretching_constraint_fn, barrier_t);
//            std::cout << "Combining" << std::endl;
            Eigen::VectorXd result(joint_limits_max_delta_val.size() + collision_constraints_val.size() + stretching_constraint_val.size());
            result.head(joint_limits_max_delta_val.size()) = joint_limits_max_delta_val;
            result.segment(joint_limits_max_delta_val.size(), collision_constraints_val.size()) = collision_constraints_val;
            result.tail(stretching_constraint_val.size()) = stretching_constraint_val;
            return result;
        };
        const auto objective_fn = [&] (const ObjectPointSet& delta)
        {
            return this->errorOfControlByPrediction(delta, desired_object_p_dot, desired_p_dot_weight);
        };

        std::cout << "Setting start of loop values" << std::endl;

        // Variables updated each loop iteration
        Eigen::VectorXd robot_motion = Eigen::VectorXd::Zero(num_dof);
        AllGrippersSinglePoseDelta gripper_motion = joint_motion_to_gripper_motion_fn(robot_motion);
        ObjectPointSet current_object_delta = model_->getObjectDelta(current_world_state, gripper_motion);
        double current_objective_error = objective_fn(current_object_delta);
        auto current_constraint_error = constraint_barrier_fn(robot_motion, gripper_motion);
        double current_combined_error = current_objective_error + current_constraint_error.sum();
        bool converged = false;

        std::cout << "Starting gradient descent" << std::endl;

        while (!converged)
        {
            std::cout << "Gradient descent start of loop error: " << current_combined_error << std::endl;

            Eigen::VectorXd error_numerical_gradient(num_dof);
            for (ssize_t dof_ind = 0; dof_ind < num_dof; ++dof_ind)
            {
                Eigen::VectorXd local_test_robot_motion = robot_motion;
                local_test_robot_motion(dof_ind) += differencing_step_size;
                const auto local_test_gripper_motion = joint_motion_to_gripper_motion_fn(local_test_robot_motion);

                const auto predicted_object_delta = model_->getObjectDelta(current_world_state, local_test_gripper_motion);
                const double new_objective_error = objective_fn(predicted_object_delta);
                const auto new_constraint_error = constraint_barrier_fn(local_test_robot_motion, local_test_gripper_motion);
                const auto new_combined_error = new_objective_error + new_constraint_error.sum();
                error_numerical_gradient(dof_ind) = (new_combined_error - current_combined_error) / differencing_step_size;

                std::cout << new_objective_error << " " << new_constraint_error.transpose() << std::endl;
            }

            std::cout << error_numerical_gradient << std::endl;
            assert(false);

            // Find a downhill step that does not violate the collision, max velocity, and joint limit constraints etc.
            auto next_robot_motion = robot_motion;
            auto next_gripper_motion = gripper_motion;
            auto next_object_delta = current_object_delta;

            int downhill_attempt_ind = 0;
            double error_gradient_step_size = -initial_gradient_step_size;
            double next_objective_error = current_objective_error;
            auto next_constraint_error = current_constraint_error;
            double next_combined_error = current_combined_error;
            do
            {
                std::cout << "\nFinding downhill step start of loop" << std::endl;

                next_robot_motion = robot_motion + error_gradient_step_size * error_numerical_gradient;
                next_gripper_motion = joint_motion_to_gripper_motion_fn(next_robot_motion);

                next_constraint_error = constraint_barrier_fn(next_robot_motion, next_gripper_motion);
                // If we have gone off the edge of the constraint, then decrease the stepsize and try again
                if (next_constraint_error.array().isNaN().any())
                {
                    ++downhill_attempt_ind;
                    error_gradient_step_size *= 0.7;
                    continue;
                }

                next_object_delta = model_->getObjectDelta(current_world_state, next_gripper_motion);
                next_objective_error = errorOfControlByPrediction(next_object_delta, desired_object_p_dot, desired_p_dot_weight);
                next_combined_error = next_objective_error + next_constraint_error.sum();

                std::cout << "Finding downhill end of loop" << std::endl;
                std::cout << "Errors: " << next_objective_error << " " << next_constraint_error.transpose() << std::endl;
                std::cout << "Sum: " << next_combined_error << std::endl;

                ++downhill_attempt_ind;
                error_gradient_step_size *= 0.7;
            }
            while (next_combined_error > current_combined_error && downhill_attempt_ind < 100);

            // If we could not find a downhill step, then exit early
            if (next_constraint_error.array().isNaN().any())
            {
                std::cout << "Unable to find downhill step, stopping" << std::endl;
                converged = true;
                break;
            }


            // If progress was made, record it
            if (next_combined_error <= current_combined_error)
            {
                robot_motion = next_robot_motion;
                gripper_motion = next_gripper_motion;
                current_object_delta = next_object_delta;
                current_objective_error = next_objective_error;
                current_constraint_error = next_constraint_error;
                current_combined_error = next_combined_error;
            }

            if ((current_combined_error - next_combined_error) < std::abs(current_combined_error) * 0.00001)
            {
                converged = true;
                std::cout << "Error has stabilized or increased, stopping: " << current_combined_error << std::endl;
            }


            std::cout << "Downhill step found: " << next_robot_motion.transpose() << std::endl;




/*
            // Follow the stretching constraint gradient until it is satisfied,
            // obeying the collision and max velocity constraints as well
            double stretching_constraint_gradient_step_size = -initial_gradient_step_size / 1000.0;
            auto stretching_constraint_correction_robot_motion = error_minimization_downhill_step;
            auto stretching_constraint_correction_gripper_motion = joint_motion_to_gripper_motion_fn(robot_motion);
            double stretching_constraint_loop_value = stretchingFunctionEvaluation(input_data, stretching_constraint_correction_gripper_motion);
            int constraint_satisfaction_ind = 0;

            std::cout << "Start of stretching constraint satisfaction" << std::endl;
            std::cout << "Stretching constraint value: " << stretching_constraint_loop_value << std::endl;

            // Calculate a gradient to follow out of the constraint violation
            Eigen::VectorXd stretching_constraint_numerical_gradient(num_dof);
            for (ssize_t dof_ind = 0; dof_ind < num_dof; ++dof_ind)
            {
                Eigen::VectorXd local_test_robot_motion = robot_motion;
                local_test_robot_motion(dof_ind) += differencing_step_size;
                const auto local_test_gripper_motion = joint_motion_to_gripper_motion_fn(local_test_robot_motion);
                const double test_constraint_value = stretchingFunctionEvaluation(input_data, local_test_gripper_motion);
                stretching_constraint_numerical_gradient(dof_ind) = (test_constraint_value - stretching_constraint_loop_value) / differencing_step_size;
            }

            while (stretching_constraint_loop_value > 0.0 && constraint_satisfaction_ind < 100)
            {
                std::cout << "Stretching constraint value: " << stretching_constraint_loop_value << std::endl;

                stretching_constraint_correction_robot_motion = error_minimization_downhill_step + stretching_constraint_gradient_step_size * stretching_constraint_numerical_gradient;
                stretching_constraint_correction_robot_motion = projectToMaxDeltaConstraints(
                            stretching_constraint_correction_robot_motion,
                            max_step_size,
                            min_joint_delta,
                            max_joint_delta);

                stretching_constraint_correction_gripper_motion = joint_motion_to_gripper_motion_fn(stretching_constraint_correction_robot_motion);
                stretching_constraint_loop_value = stretchingFunctionEvaluation(input_data, stretching_constraint_correction_gripper_motion);

                stretching_constraint_gradient_step_size -= initial_gradient_step_size / 1000.0;
                ++constraint_satisfaction_ind;
            }

            std::cout << "Stretching constraint resolution finished: " << stretching_constraint_correction_robot_motion.transpose() << std::endl;
            std::cout << "Stretching constraint value: " << stretching_constraint_loop_value << std::endl;

            const auto& post_stretching_robot_motion = stretching_constraint_correction_robot_motion;
            const auto post_stretching_gripper_motion = joint_motion_to_gripper_motion_fn(post_stretching_robot_motion);
            const auto post_stretching_predicted_object_delta = model_->getObjectDelta(current_world_state, post_stretching_gripper_motion);
            const double post_stretching_objective_error = errorOfControlByPrediction(post_stretching_predicted_object_delta, desired_object_p_dot, desired_p_dot_weight);
            const auto post_stretching_constraint_error = constraint_barrier_fn(robot_motion);
            const double post_stretching_combined_error = post_stretching_objective_error + post_stretching_constraint_error.sum();

            // If we could not make progress, then return whatever the last valid movement we had was
            if (stretching_constraint_loop_value > 0.0)
            {
                converged = true;
                std::cout << "Unable to resolve stretching constraint, stopping" << std::endl;
            }
            // Otherwise, accept the update
            else
            {
                converged = (current_combined_error - post_stretching_combined_error) < std::abs(current_combined_error) * 0.000001;

                robot_motion = post_stretching_robot_motion;
                gripper_motion = post_stretching_gripper_motion;
                current_object_delta = post_stretching_predicted_object_delta;
                current_objective_error = post_stretching_objective_error;
                current_constraint_error = post_stretching_constraint_error;
                current_combined_error = post_stretching_combined_error;

                if (converged)
                {
                    std::cout << "Error has stabilized or increased, stopping: " << current_combined_error << std::endl;
                }
            }
*/



        }

        return OutputData(gripper_motion, current_object_delta, robot_motion);
    }
    else
    {
        const double max_individual_gripper_step_size = robot_->max_gripper_velocity_norm_ * robot_->dt_;
        const double differencing_step_size = max_individual_gripper_step_size / 100.0;
        const double initial_gradient_step_size = max_individual_gripper_step_size / 10.0;

        bool converged = false;
        AllGrippersSinglePoseDelta robot_motion(num_grippers, kinematics::Vector6d::Zero());
        ObjectPointSet current_object_delta = model_->getObjectDelta(current_world_state, robot_motion);
        double current_error = errorOfControlByPrediction(current_object_delta, desired_object_p_dot, desired_p_dot_weight);

        while (!converged)
        {
            Eigen::VectorXd error_numerical_gradient(num_grippers * 6);
            for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; ++gripper_ind)
            {
                for (size_t single_gripper_dir_ind = 0; single_gripper_dir_ind < 6; ++single_gripper_dir_ind)
                {
                    const auto test_input_ind = gripper_ind * 6 + single_gripper_dir_ind;

                    AllGrippersSinglePoseDelta local_test_motion = robot_motion;
                    local_test_motion[gripper_ind](single_gripper_dir_ind) += differencing_step_size;

                    const auto predicted_object_delta = model_->getObjectDelta(current_world_state, local_test_motion);
                    const double new_error = errorOfControlByPrediction(predicted_object_delta, desired_object_p_dot, desired_p_dot_weight);
                    error_numerical_gradient(test_input_ind) = (new_error - current_error) / differencing_step_size;
                }
            }

            // Take a step downhill, doing a line search to find a reasonable motion
            // This accounts for the collision and max velocity constraints, but not the stretching constraint
            auto error_minimization_downhill_step = robot_motion;
            {
                int downhill_attempt_ind = 0;
                double error_gradient_step_size = -initial_gradient_step_size;
                double next_downhill_error = current_error;
                do
                {
                    error_minimization_downhill_step = stepInDirection(robot_motion, error_numerical_gradient, error_gradient_step_size);
                    error_minimization_downhill_step = projectToCollisionAndMaxDeltaConstraints(
                                current_world_state.all_grippers_single_pose_,
                                error_minimization_downhill_step,
                                input_data.world_current_state_.gripper_collision_data_,
                                input_data.robot_->min_controller_distance_to_obstacles_,
                                max_individual_gripper_step_size);

                    const auto predicted_object_delta = model_->getObjectDelta(current_world_state, error_minimization_downhill_step);
                    next_downhill_error = errorOfControlByPrediction(predicted_object_delta, desired_object_p_dot, desired_p_dot_weight);

                    ++downhill_attempt_ind;
                    error_gradient_step_size *= 0.7;
                }
                while (next_downhill_error > current_error && downhill_attempt_ind < 10);
            }

            // Follow the constraint gradient until it is satisfied,
            // obeying the collision and max velocity constraints as well
            double constraint_gradient_step_size = -initial_gradient_step_size / 100.0;
            auto stretching_constraint_correction_result = error_minimization_downhill_step;
            double loop_constraint_value = stretchingFunctionEvaluation(input_data, stretching_constraint_correction_result);
            int constraint_satisfaction_ind = 0;
            while (loop_constraint_value > 0.0 && constraint_satisfaction_ind < 10)
            {
                // Calculate a gradient to follow out of the constraint violation
                Eigen::VectorXd stretching_constraint_numerical_gradient(num_grippers * 6);
                for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; ++gripper_ind)
                {
                    for (size_t single_gripper_dir_ind = 0; single_gripper_dir_ind < 6; ++single_gripper_dir_ind)
                    {
                        const auto test_input_ind = gripper_ind * 6 + single_gripper_dir_ind;

                        AllGrippersSinglePoseDelta local_test_motion = stretching_constraint_correction_result;
                        local_test_motion[gripper_ind](single_gripper_dir_ind) += differencing_step_size;
                        const double test_constraint_value = stretchingFunctionEvaluation(input_data, local_test_motion);
                        stretching_constraint_numerical_gradient(test_input_ind) = (test_constraint_value - loop_constraint_value) / differencing_step_size;
                    }
                }

                stretching_constraint_correction_result = stepInDirection(stretching_constraint_correction_result, stretching_constraint_numerical_gradient, constraint_gradient_step_size);
                stretching_constraint_correction_result = projectToCollisionAndMaxDeltaConstraints(
                                                current_world_state.all_grippers_single_pose_,
                                                stretching_constraint_correction_result,
                                                input_data.world_current_state_.gripper_collision_data_,
                                                input_data.robot_->min_controller_distance_to_obstacles_,
                                                max_individual_gripper_step_size);


                loop_constraint_value = stretchingFunctionEvaluation(input_data, stretching_constraint_correction_result);

                constraint_gradient_step_size -= initial_gradient_step_size / 100.0;
                ++constraint_satisfaction_ind;
            }

            const auto loop_final_robot_motion = stretching_constraint_correction_result;
            const auto loop_final_predicted_object_delta = model_->getObjectDelta(current_world_state, loop_final_robot_motion);
            const double loop_final_downhill_error = errorOfControlByPrediction(loop_final_predicted_object_delta, desired_object_p_dot, desired_p_dot_weight);

            // If we could not make progress, then return whatever the last valid movement we had was
            if (loop_constraint_value > 0.0 || loop_final_downhill_error > current_error)
            {
                converged = true;
            }
            // Otherwise, accept the update
            else
            {
                converged = (current_error - loop_final_downhill_error) < std::abs(current_error) * 0.000001;

                robot_motion = loop_final_robot_motion;
                current_object_delta = loop_final_predicted_object_delta;
                current_error = loop_final_downhill_error;
            }
        }

        return OutputData(robot_motion, current_object_delta, Eigen::VectorXd());
    }
}

//////////////////////////////////////////////////////////////////////////////////
// Helper functions
//////////////////////////////////////////////////////////////////////////////////

kinematics::Vector6d StretchingAvoidanceController::singleGripperPoseDeltaSampler(
        const double max_delta)
{
    const double x_trans = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double y_trans = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double z_trans = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));

    const double x_rot = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double y_rot = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double z_rot = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));

    kinematics::Vector6d random_sample;

    double raw_norm = std::sqrt(std::pow(x_trans, 2) + std::pow(y_trans, 2) + std::pow(z_trans, 2));

    if (raw_norm > 0.000001 && (fix_step_ || raw_norm > max_delta))
    {
        random_sample(0) = x_trans / raw_norm * max_delta;
        random_sample(1) = y_trans / raw_norm * max_delta;
        random_sample(2) = z_trans / raw_norm * max_delta;
    }
    else
    {
        random_sample(0) = x_trans;
        random_sample(1) = y_trans;
        random_sample(2) = z_trans;
    }

    random_sample(3) = x_rot;
    random_sample(4) = y_rot;
    random_sample(5) = z_rot;

    return ClampGripperPoseDeltas(random_sample, max_delta);
}

AllGrippersSinglePoseDelta StretchingAvoidanceController::allGripperPoseDeltaSampler(
        const ssize_t num_grippers,
        const double max_delta)
{
    AllGrippersSinglePoseDelta grippers_motion_sample(num_grippers, kinematics::Vector6d::Zero());

    // if sample_count_ < 0, return all-sampled motion, otherwise, return one-for-each-time sample
    if (sample_count_ < 0)
    {
        for (ssize_t ind = 0; ind < num_grippers; ind++)
        {
            grippers_motion_sample[ind] = singleGripperPoseDeltaSampler(max_delta);
        }
    }
    else if (sample_count_ < num_grippers)
    {
        grippers_motion_sample[sample_count_] = singleGripperPoseDeltaSampler(max_delta);
    }
    else
    {
        assert(false && "This code should not be reachable");
    }
    return grippers_motion_sample;
}


kinematics::Vector6d StretchingAvoidanceController::getConstraintAwareGripperDeltaSample(
                    const Eigen::Isometry3d& gripper_pose,
                    const CollisionData& collision_data,
                    const double max_delta,
                    const Eigen::Vector3d& stretching_reduction_vector,
                    const Eigen::Vector3d& vector_from_gripper_to_translation_point)
{
    const auto J_collision = ComputeGripperMotionToPointMotionJacobian(collision_data.nearest_point_to_obstacle_, gripper_pose);
    const auto J_distance = collision_data.obstacle_surface_normal_.transpose() * J_collision;
    // Returns true if the constraint is satisfied
    const auto collision_constraint_fn = [&] (const kinematics::Vector6d& gripper_delta)
    {
        return robot_->min_controller_distance_to_obstacles_ + GRIPPER_COLLISION_REPULSION_MARGIN
                - collision_data.distance_to_obstacle_ - J_distance * gripper_delta;
    };

    const auto stretching_constraint_fn = [&] (const kinematics::Vector6d& gripper_delta)
    {
        if (!over_stretch_)
        {
            return -1000.0;
        }
        else
        {
            const auto r_dot = gripper_delta.head<3>() + gripper_delta.tail<3>().cross(vector_from_gripper_to_translation_point);
            const auto r_dot_norm = r_dot.norm();
            if (r_dot_norm < 1e-6)
            {
                return stretching_cosine_threshold_ - 1.0;
            }
            else
            {
                const auto cos_angle = stretching_reduction_vector.dot(r_dot) / r_dot_norm;
                return  stretching_cosine_threshold_ - cos_angle;
            }
        }
    };

    kinematics::Vector6d sample = singleGripperPoseDeltaSampler(max_delta * 0.8);
    bool collision_satisfied = (collision_constraint_fn(sample) < 0.0);
    bool stretching_satisified = (stretching_constraint_fn(sample) < 0.0);
    bool valid_sample =  collision_satisfied && stretching_satisified;
    while (!valid_sample)
    {
        sample = singleGripperPoseDeltaSampler(max_delta * 0.8);
        collision_satisfied = (collision_constraint_fn(sample) < 0.0);
        stretching_satisified = (stretching_constraint_fn(sample) < 0.0);
        valid_sample = collision_satisfied && stretching_satisified;
    }

    assert(collision_satisfied);
    assert(stretching_satisified);

    return sample;
}






double StretchingAvoidanceController::errorOfControlByPrediction(
        const ObjectPointSet predicted_object_p_dot,
        const Eigen::VectorXd& desired_object_p_dot,
        const Eigen::VectorXd& desired_p_dot_weight) const
{
    const Eigen::Map<const Eigen::VectorXd> prediction_as_vector(predicted_object_p_dot.data(), desired_object_p_dot.size());
    const auto individual_error = (prediction_as_vector- desired_object_p_dot).cwiseAbs2();
    return std::sqrt(individual_error.dot(desired_p_dot_weight));

//    double sum_of_error = 0;

//    for (ssize_t node_ind = 0; node_ind < num_nodes; node_ind++)
//    {
//        Eigen::Vector3d node_predicted_p_dot = predicted_object_p_dot.col(node_ind);
//        Eigen::Vector3d node_desired_p_dot = desired_object_p_dot.segment<3>(node_ind * 3);

//        // Only none_zero desired p dot is considered.
//        if (desired_p_dot_weight(node_ind * 3) > 0)
//        {
//            double node_p_dot_error = (node_predicted_p_dot - node_desired_p_dot).norm();
//            sum_of_error += node_p_dot_error * desired_p_dot_weight(node_ind *3);
//        }
//    }

//    return sum_of_error;
}


void StretchingAvoidanceController::visualize_stretching_vector(
        const ObjectPointSet& object_configuration)
{
    switch (deformable_type_)
    {
        case ROPE:
        {
            visualize_rope_stretching_vector(object_configuration);
            break;
        }
        case CLOTH:
        {
            visualize_cloth_stretching_vector(object_configuration);
            break;
        }
        default:
        {
            assert(false && "visualize stretching vector of neither cloth nor rope");
            break;
        }
    }
}

void StretchingAvoidanceController::visualize_rope_stretching_vector(
        const ObjectPointSet& object_configuration)
{
    const ssize_t num_nodes = object_configuration.cols();
    const ssize_t start_node = 1;
    const ssize_t end_node = num_nodes - 2;

    Eigen::Vector3d first_correction_vector =
            (object_configuration.block<3, 1>(0, start_node + 1)
                - object_configuration.block<3, 1>(0, start_node));
    first_correction_vector = first_correction_vector/first_correction_vector.norm();

    Eigen::Vector3d second_correction_vector =
            (object_configuration.block<3, 1>(0, end_node - 1)
                - object_configuration.block<3, 1>(0, end_node));
    second_correction_vector = second_correction_vector/second_correction_vector.norm();

    EigenHelpers::VectorVector3d line_starts;
    EigenHelpers::VectorVector3d line_ends;
    line_starts.push_back(object_configuration.block<3,1>(0, 0));
    line_starts.push_back(object_configuration.block<3,1>(0, num_nodes-1));
    line_ends.push_back(line_starts[0] + 0.5 * first_correction_vector);
    line_ends.push_back(line_starts[1] + 0.5 * second_correction_vector);

    vis_->visualizeLines("gripper overstretch motion",
                        line_starts,
                        line_ends,
                        Visualizer::Orange());
}

void StretchingAvoidanceController::visualize_cloth_stretching_vector(
        const ObjectPointSet& object_configuration)
{
    // Assume knowing there are two grippers.
    assert(grippers_data_.size()==2 || "grippers size is not 2, stretching vector visualization not developed");

    const StretchingVectorInfo& first_stretching_vector_info = grippers_data_[0].stretching_vector_info_;
    const std::vector<long>& first_from_nodes = first_stretching_vector_info.from_nodes_;
    const std::vector<long>& first_to_nodes = first_stretching_vector_info.to_nodes_;
    const std::vector<double>& first_contribution = first_stretching_vector_info.node_contribution_;

    const StretchingVectorInfo& second_stretching_vector_info = grippers_data_[1].stretching_vector_info_;
    const std::vector<long>& second_from_nodes = second_stretching_vector_info.from_nodes_;
    const std::vector<long>& second_to_nodes = second_stretching_vector_info.to_nodes_;
    const std::vector<double>& second_contribution = second_stretching_vector_info.node_contribution_;

    Eigen::Vector3d first_correction_vector = Eigen::MatrixXd::Zero(3,1);
    for (size_t stretching_ind = 0; stretching_ind < first_from_nodes.size(); stretching_ind++)
    {
        first_correction_vector +=
                first_contribution[stretching_ind] *
                (object_configuration.block<3, 1>(0, first_to_nodes[stretching_ind])
                 - object_configuration.block<3, 1>(0, first_from_nodes[stretching_ind]));
    }
    Eigen::Vector3d second_correction_vector = Eigen::MatrixXd::Zero(3,1);
    for (size_t stretching_ind = 0; stretching_ind < second_from_nodes.size(); stretching_ind++)
    {
        second_correction_vector +=
                second_contribution[stretching_ind] *
                (object_configuration.block<3, 1>(0, second_to_nodes[stretching_ind])
                 - object_configuration.block<3, 1>(0, second_from_nodes[stretching_ind]));
    }

    EigenHelpers::VectorVector3d line_starts;
    EigenHelpers::VectorVector3d line_ends;
    line_starts.push_back(object_configuration.block<3,1>(0, first_from_nodes[0]));
    line_starts.push_back(object_configuration.block<3,1>(0, second_from_nodes[0]));
    line_ends.push_back(line_starts[0] + 10 * first_correction_vector);
    line_ends.push_back(line_starts[1] + 10 * second_correction_vector);

    vis_->visualizeLines("gripper overstretch motion",
                        line_starts,
                        line_ends,
                        Visualizer::Orange());
}

void StretchingAvoidanceController::visualize_gripper_motion(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& gripper_motion)
{
    const auto grippers_test_poses = kinematics::applyTwist(current_gripper_pose, gripper_motion);
    EigenHelpers::VectorVector3d line_starts;
    EigenHelpers::VectorVector3d line_ends;

    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
    {
        line_starts.push_back(current_gripper_pose[gripper_ind].translation());
        line_ends.push_back(current_gripper_pose[gripper_ind].translation() + 100 * (grippers_test_poses[gripper_ind].translation() - current_gripper_pose[gripper_ind].translation()));
    }

    vis_->visualizeLines("gripper motion",
                         line_starts,
                         line_ends,
                         Visualizer::Olive());
}

double StretchingAvoidanceController::gripperCollisionCheckHelper(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& test_gripper_motion) const
{
    assert(false && "This function is not used for the gradient descent method");

    const auto grippers_test_poses = kinematics::applyTwist(current_gripper_pose, test_gripper_motion);

    double min_collision_distance = std::numeric_limits<double>::infinity();

    for (size_t gripper_idx = 0; gripper_idx < grippers_test_poses.size(); ++gripper_idx)
    {
        const auto tmp = grippers_test_poses[gripper_idx].translation();
        const Eigen::Vector4d test_point(tmp.x(), tmp.y(), tmp.z(), 1.0);
        const auto collision_result = environment_sdf_.EstimateDistance4dLegacy(test_point);
        if (collision_result.first < min_collision_distance)
        {
            min_collision_distance = collision_result.first;
        }
    }

    return min_collision_distance;
}

bool StretchingAvoidanceController::gripperCollisionCheckResult(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& test_gripper_motion) const
{
    const double min_dis_to_obstacle = gripperCollisionCheckHelper(current_gripper_pose, test_gripper_motion);
    const bool collision_violation = (min_dis_to_obstacle < distance_to_obstacle_threshold_);
    return collision_violation;
}


// Returns true if the constraint is violated
bool StretchingAvoidanceController::stretchingDetection(
        const InputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    return (stretchingFunctionEvaluation(input_data, test_gripper_motion) > 0.0);
}

// Note that NOMAD wants all constraints in the form c(x) <= 0
// If the calc'd value is larger than the threshold, then the gripper motion is sufficently pointed
// in the direction needed to reduce/not cause more stretching
double StretchingAvoidanceController::stretchingFunctionEvaluation(
        const InputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    assert(false && "This function is not used for the gradient descent method");
    if (test_gripper_motion.size() != 2)
    {
        assert(false && "num of grippers not match");
    }

    switch (deformable_type_)
    {
        case ROPE:
        {
            return stretching_cosine_threshold_ - ropeTwoGripperStretchingHelper(input_data, test_gripper_motion);
        }
        case CLOTH:
        {
            return stretching_cosine_threshold_ - clothTwoGripperStretchingHelper(input_data, test_gripper_motion);
        }
        default:
        {
            assert(false && "deformable_type is neither rope nor cloth");
            return 0.0;
        }
    }
}

double StretchingAvoidanceController::ropeTwoGripperStretchingHelper(
        const InputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    assert(false && "This function is not used for the gradient descent method" && "Verify the math in this implementation");

    double stretching_sum = 0.0;
    double stretching_cos = 1.0; // return a value > stretching_cos_threshold

    if (over_stretch_)
    {
        const ObjectPointSet& object_configuration = input_data.world_current_state_.object_configuration_;
        const AllGrippersSinglePose& current_gripper_pose = input_data.world_current_state_.all_grippers_single_pose_;
        const ssize_t num_nodes = object_configuration.cols();

        const ssize_t start_node = 0;
        const ssize_t end_node = num_nodes - 1;

        Eigen::Vector3d first_correction_vector =
                (object_configuration.block<3, 1>(0, start_node + 1)
                    - object_configuration.block<3, 1>(0, start_node));
        first_correction_vector = first_correction_vector/first_correction_vector.norm();

        Eigen::Vector3d second_correction_vector =
                (object_configuration.block<3, 1>(0, end_node - 1)
                    - object_configuration.block<3, 1>(0, end_node));
        second_correction_vector = second_correction_vector/second_correction_vector.norm();

        EigenHelpers::VectorVector3d stretching_correction_vector;
        stretching_correction_vector.push_back(first_correction_vector);
        stretching_correction_vector.push_back(second_correction_vector);

        const auto grippers_test_poses = kinematics::applyTwist(current_gripper_pose, test_gripper_motion);
        double sum_resulting_motion_norm = 0.0;

        switch (gripper_controller_type_)
        {
            case StretchingAvoidanceControllerSolverType::RANDOM_SAMPLING:
            {
                if (sample_count_ > -1)
                {
                    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
                    {
                        Eigen::Vector3d resulting_gripper_motion = grippers_test_poses[gripper_ind].translation()
                                - current_gripper_pose[gripper_ind].translation();
                       stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector[gripper_ind]);
                        sum_resulting_motion_norm += resulting_gripper_motion.norm();
                    }
                    if (sum_resulting_motion_norm != 0.0)
                    {
                        stretching_cos = stretching_sum / sum_resulting_motion_norm;
                    }
                }
                else
                {
                    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
                    {
                        Eigen::Vector3d resulting_gripper_motion = grippers_test_poses[gripper_ind].translation()
                                - current_gripper_pose[gripper_ind].translation();
                        stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector[gripper_ind])
                                / resulting_gripper_motion.norm();
                    }
                    stretching_cos = stretching_sum / (double)(current_gripper_pose.size());
               }
                break;
            }
            case StretchingAvoidanceControllerSolverType::NOMAD_OPTIMIZATION:
            {
                for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
                {
                    Eigen::Vector3d resulting_gripper_motion = grippers_test_poses[gripper_ind].translation()
                            - current_gripper_pose[gripper_ind].translation();
                    stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector[gripper_ind])
                            / resulting_gripper_motion.norm();
                }
                stretching_cos = stretching_sum / (double)(current_gripper_pose.size());
                break;
            }
            default:
            {
                assert(false && "not valid controller solving type");
                break;
            }
        }

    }
    return stretching_cos;

}

double StretchingAvoidanceController::clothTwoGripperStretchingHelper(
        const InputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    assert(false && "This function is not used for the gradient descent method");

    // Assume knowing there are two grippers.
    assert(grippers_data_.size() == 2 || "grippers size is not 2, stretching vector visualization not developed");

    std::cout << print(test_gripper_motion) << std::endl;

    const ObjectPointSet& object_configuration = input_data.world_current_state_.object_configuration_;
    const AllGrippersSinglePose& current_grippers_pose = input_data.world_current_state_.all_grippers_single_pose_;
    const AllGrippersSinglePose grippers_test_poses = kinematics::applyTwist(current_grippers_pose, test_gripper_motion);
    const Eigen::Isometry3d& gripper0_current_pose = current_grippers_pose[0];
    const Eigen::Isometry3d& gripper1_current_pose = current_grippers_pose[1];
    const Eigen::Isometry3d& gripper0_test_pose = grippers_test_poses[0];
    const Eigen::Isometry3d& gripper1_test_pose = grippers_test_poses[1];

    // If the object is not overstretched already, then the constraint is not active, indicited by a value of 1.0
    // 1.0 indicates the least stretching possible, with both grippers moving in the direction needed to reduce stretching
    if (!over_stretch_)
    {
        return 1.0;
    }

    const StretchingVectorInfo& first_stretching_vector_info = grippers_data_[0].stretching_vector_info_;
    const std::vector<long>& first_from_nodes = first_stretching_vector_info.from_nodes_;
    const std::vector<long>& first_to_nodes = first_stretching_vector_info.to_nodes_;
    const std::vector<double>& first_contribution = first_stretching_vector_info.node_contribution_;

    const StretchingVectorInfo& second_stretching_vector_info = grippers_data_[1].stretching_vector_info_;
    const std::vector<long>& second_from_nodes = second_stretching_vector_info.from_nodes_;
    const std::vector<long>& second_to_nodes = second_stretching_vector_info.to_nodes_;
    const std::vector<double>& second_contribution = second_stretching_vector_info.node_contribution_;

    Eigen::Vector3d point_on_first_gripper_before_motion = Eigen::MatrixXd::Zero(3,1);
    Eigen::Vector3d point_on_second_gripper_before_motion = Eigen::MatrixXd::Zero(3,1);
    Eigen::Vector3d first_correction_vector = Eigen::MatrixXd::Zero(3,1);
    Eigen::Vector3d second_correction_vector = Eigen::MatrixXd::Zero(3,1);

    for (size_t stretching_ind = 0; stretching_ind < first_from_nodes.size(); stretching_ind++)
    {
        const auto from_node = object_configuration.block<3, 1>(0, first_from_nodes[stretching_ind]);
        const auto to_node = object_configuration.block<3, 1>(0, first_to_nodes[stretching_ind]);
        const auto node_delta = to_node - from_node;
        first_correction_vector += first_contribution[stretching_ind] * node_delta;
        point_on_first_gripper_before_motion += first_contribution[stretching_ind] * from_node;
    }

    for (size_t stretching_ind = 0; stretching_ind < second_from_nodes.size(); stretching_ind++)
    {
        const auto from_node = object_configuration.block<3, 1>(0, second_from_nodes[stretching_ind]);
        const auto to_node = object_configuration.block<3, 1>(0, second_to_nodes[stretching_ind]);
        const auto node_delta = to_node - from_node;
        second_correction_vector += second_contribution[stretching_ind] * node_delta;
        point_on_second_gripper_before_motion += second_contribution[stretching_ind] * from_node;

    }

    // Normalize the vectors to get direction only; will be zero if the input norm is 0.
    EigenHelpers::VectorVector3d per_gripper_stretching_correction_vector(2, Eigen::Vector3d::Zero());
    {
        const auto first_vector_norm = first_correction_vector.norm();
        if (first_vector_norm > 1e-6)
        {
            per_gripper_stretching_correction_vector[0] = first_correction_vector / first_vector_norm;
        }

        const auto second_vector_norm = second_correction_vector.norm();
        if (second_vector_norm > 1e-6)
        {
            per_gripper_stretching_correction_vector[1] = second_correction_vector / second_vector_norm;
        }
    }

    // Transform the input points based on the motion of each gripper
    const Eigen::Vector3d first_point_in_first_gripper_frame   = gripper0_current_pose.inverse() * point_on_first_gripper_before_motion;
    const Eigen::Vector3d second_point_in_second_gripper_frame = gripper1_current_pose.inverse() * point_on_second_gripper_before_motion;
    const Eigen::Vector3d point_on_first_gripper_after_motion  = gripper0_test_pose * first_point_in_first_gripper_frame;
    const Eigen::Vector3d point_on_second_gripper_after_motion = gripper1_test_pose * second_point_in_second_gripper_frame;

    std::vector<Eigen::Vector3d> point_motion_vector(2, Eigen::Vector3d::Zero());
    {
        const auto first_point_motion = point_on_first_gripper_after_motion - point_on_first_gripper_before_motion;
        const auto first_vector_norm = first_point_motion.norm();
        if (first_vector_norm > 1e-6)
        {
            point_motion_vector[0] = first_point_motion / first_vector_norm;
        }

        const auto second_point_motion = point_on_second_gripper_after_motion - point_on_second_gripper_before_motion;
        const auto second_vector_norm = second_point_motion.norm();
        if (second_vector_norm > 1e-6)
        {
            point_motion_vector[1] = second_point_motion / second_vector_norm;
        }
    }

    double stretching_cos = 0.0;
    // sample_count_ > -1 means only sample one gripper each time
    if ((sample_count_ > -1) && (gripper_controller_type_ == StretchingAvoidanceControllerSolverType::RANDOM_SAMPLING))
    {
        assert(sample_count_ < (int)(per_gripper_stretching_correction_vector.size()));
        const Eigen::Vector3d& point_movement_direction = point_motion_vector[sample_count_];
        stretching_cos = point_movement_direction.dot(per_gripper_stretching_correction_vector[sample_count_]);
    }
    else
    {
        stretching_cos = 0.0;
        for (size_t gripper_ind = 0; gripper_ind < current_grippers_pose.size(); gripper_ind++)
        {
            std::cout << "point direction: " << point_motion_vector[gripper_ind].transpose() << "    stretching_correction_vec: " << per_gripper_stretching_correction_vector[gripper_ind].transpose() << std::endl;

            const Eigen::Vector3d& point_movement_direction = point_motion_vector[gripper_ind];
            stretching_cos += point_movement_direction.dot(per_gripper_stretching_correction_vector[gripper_ind]);
        }
        stretching_cos /= (double)(current_grippers_pose.size());

        std::cout << "result: " << stretching_cos << std::endl;
    }

    if (std::isnan(stretching_cos))
    {
        std::cout << "Test gripper motion: " << print(test_gripper_motion) << std::endl;
        std::cout << std::endl;

        std::cout << "Point on first gripper:        " << point_on_first_gripper_before_motion.transpose() << std::endl;
        std::cout << "Point on second gripper:       " << point_on_second_gripper_before_motion.transpose() << std::endl;
        std::cout << "First correction vector:       " << first_correction_vector.normalized().transpose() << std::endl;
        std::cout << "Second correction vector:      " << second_correction_vector.normalized().transpose() << std::endl;
        std::cout << "Point on first gripper after:  " << point_on_first_gripper_after_motion.transpose() << std::endl;
        std::cout << "Point on second gripper after: " << point_on_second_gripper_after_motion.transpose() << std::endl;

        std::cout << "trans0: " << (grippers_test_poses[0].translation() - current_grippers_pose[0].translation()).transpose() << std::endl;
        std::cout << "trans1: " << (grippers_test_poses[1].translation() - current_grippers_pose[1].translation()).transpose() << std::endl;

        assert(!std::isnan(stretching_cos));
    }

    return stretching_cos;
}




std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> StretchingAvoidanceController::stretchingCorrectionVectorsAndPoints(const InputData& input_data) const
{
    switch (deformable_type_)
    {
        case ROPE:
        {
            return ropeTwoGrippersStretchingCorrectionVectorsAndPoints(input_data);
        }
        case CLOTH:
        {
            return clothTwoGrippersStretchingCorrectionVectorsAndPoints(input_data);
        }
        default:
        {
            assert(false && "deformable_type is neither rope nor cloth");
        }
    }
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> StretchingAvoidanceController::ropeTwoGrippersStretchingCorrectionVectorsAndPoints(const InputData& input_data) const
{
    assert(grippers_data_.size() == 2 || "grippers size is not 2, stretching vectors not defined");

    const ObjectPointSet& object_config = input_data.world_current_state_.object_configuration_;
    const Eigen::Isometry3d& first_gripper_pose = input_data.world_current_state_.all_grippers_single_pose_[0];
    const Eigen::Isometry3d& second_gripper_pose = input_data.world_current_state_.all_grippers_single_pose_[1];

    const ssize_t num_nodes = object_config.cols();
    const ssize_t start_node = 0;
    const ssize_t end_node = num_nodes - 1;

    const auto first_correction_vector = object_config.col(start_node + 1) - object_config.col(start_node);
    const auto second_correction_vector = object_config.col(end_node - 1) - object_config.col(end_node);

    // Zero initialize the vectors, we will update momentarily
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> result(2, {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});

    // Normalize the vectors to get direction only; will be zero if the input norm is 0.
    // Store the resulting vector in the return value
    const auto first_vector_norm = first_correction_vector.norm();
    if (first_vector_norm > 1e-6)
    {
        result[0].first = first_correction_vector / first_vector_norm;
    }

    const auto second_vector_norm = second_correction_vector.norm();
    if (second_vector_norm > 1e-6)
    {
        result[1].first = second_correction_vector / second_vector_norm;
    }

    // Rotate the vectors into gripper frame
    result[0].first = first_gripper_pose.linear().transpose() * result[0].first;
    result[1].first = second_gripper_pose.linear().transpose() * result[1].first;

    return result;
}

// Note that the returned vectors and points are in gripper frame
std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> StretchingAvoidanceController::clothTwoGrippersStretchingCorrectionVectorsAndPoints(const InputData& input_data) const
{
    // Assume knowing there are two grippers.
    assert(grippers_data_.size() == 2 || "grippers size is not 2, stretching vectors not defined");

    const ObjectPointSet& object_config = input_data.world_current_state_.object_configuration_;
    const Eigen::Isometry3d& first_gripper_pose = input_data.world_current_state_.all_grippers_single_pose_[0];
    const Eigen::Isometry3d& second_gripper_pose = input_data.world_current_state_.all_grippers_single_pose_[1];

    const StretchingVectorInfo& first_stretching_vector_info = grippers_data_[0].stretching_vector_info_;
    const std::vector<long>& first_from_nodes = first_stretching_vector_info.from_nodes_;
    const std::vector<long>& first_to_nodes = first_stretching_vector_info.to_nodes_;
    const std::vector<double>& first_contribution = first_stretching_vector_info.node_contribution_;

    const StretchingVectorInfo& second_stretching_vector_info = grippers_data_[1].stretching_vector_info_;
    const std::vector<long>& second_from_nodes = second_stretching_vector_info.from_nodes_;
    const std::vector<long>& second_to_nodes = second_stretching_vector_info.to_nodes_;
    const std::vector<double>& second_contribution = second_stretching_vector_info.node_contribution_;

    Eigen::Vector3d point_on_first_gripper_before_motion = Eigen::MatrixXd::Zero(3,1);
    Eigen::Vector3d point_on_second_gripper_before_motion = Eigen::MatrixXd::Zero(3,1);
    Eigen::Vector3d first_correction_vector = Eigen::MatrixXd::Zero(3,1);
    Eigen::Vector3d second_correction_vector = Eigen::MatrixXd::Zero(3,1);

    for (size_t stretching_ind = 0; stretching_ind < first_from_nodes.size(); stretching_ind++)
    {
        const auto from_node = object_config.col(first_from_nodes[stretching_ind]);
        const auto to_node = object_config.col(first_to_nodes[stretching_ind]);
        const auto node_delta = to_node - from_node;
        first_correction_vector += first_contribution[stretching_ind] * node_delta;
        point_on_first_gripper_before_motion += first_contribution[stretching_ind] * from_node;
    }

    for (size_t stretching_ind = 0; stretching_ind < second_from_nodes.size(); stretching_ind++)
    {
        const auto from_node = object_config.col(second_from_nodes[stretching_ind]);
        const auto to_node = object_config.col(second_to_nodes[stretching_ind]);
        const auto node_delta = to_node - from_node;
        second_correction_vector += second_contribution[stretching_ind] * node_delta;
        point_on_second_gripper_before_motion += second_contribution[stretching_ind] * from_node;
    }

    // Zero initialize the vectors, we will update momentarily
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> result(2, {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});

    // Normalize the vectors to get direction only; will be zero if the input norm is 0.
    // Store the resulting vector in the return value
    const auto first_vector_norm = first_correction_vector.norm();
    if (first_vector_norm > 1e-6)
    {
        result[0].first = first_correction_vector / first_vector_norm;
    }
    result[0].second = point_on_first_gripper_before_motion;

    const auto second_vector_norm = second_correction_vector.norm();
    if (second_vector_norm > 1e-6)
    {
        result[1].first = second_correction_vector / second_vector_norm;
    }
    result[1].second = point_on_second_gripper_before_motion;

    // Rotate the vectors into gripper frame
    result[0].first = first_gripper_pose.linear().transpose() * result[0].first;
    result[1].first = second_gripper_pose.linear().transpose() * result[1].first;

    // Transform the points into the gripper frame
    result[0].second = first_gripper_pose.inverse() * result[0].second;
    result[1].second = second_gripper_pose.inverse() * result[1].second;

    return result;
}
