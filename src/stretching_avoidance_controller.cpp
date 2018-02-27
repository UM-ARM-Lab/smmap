#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <smmap_utilities/nomad_solvers.h>
#include <kinematics_toolbox/kinematics.h>
#include <omp.h>

#include "smmap/stretching_avoidance_controller.h"
#include "smmap/ros_communication_helpers.hpp"
#include "smmap/grippers.hpp"

using namespace smmap;
using namespace smmap_utilities;

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
    , robot_min_distance_to_obstacles_(GetControllerMinDistanceToObstacles(nh))
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
    , fix_step_(GetGrippersMotionSampleSize(ph))
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

        // Return the min distance of the points of interest to the obstacles, minus the required clearance
        const std::vector<std::pair<CollisionData, Eigen::Matrix3Xd>> poi_collision_data =
                robot_->getPointsOfInterestCollisionData(input_data.world_current_state_.robot_configuration_);
        const double required_obstacle_clearance = robot_min_distance_to_obstacles_;
        const std::function<double(const Eigen::VectorXd&)> collision_constraint_fn = [&] (
                const Eigen::VectorXd& test_robot_motion)
        {
            double min_poi_distance = std::numeric_limits<double>::max();
            for (size_t poi_ind = 0; poi_ind < poi_collision_data.size(); ++poi_ind)
            {
                const CollisionData& collision_data = poi_collision_data[poi_ind].first;
                const Eigen::MatrixXd& poi_jacobian = poi_collision_data[poi_ind].second;
                const double initial_distance = poi_collision_data[poi_ind].first.distance_to_obstacle_;

                const double current_distance = initial_distance -
                        collision_data.obstacle_surface_normal_.transpose() * poi_jacobian * test_robot_motion;

                min_poi_distance = std::min(min_poi_distance, current_distance);
            }

            return required_obstacle_clearance - min_poi_distance;
        };

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


inline std::string print(const AllGrippersSinglePoseDelta& delta)
{
    assert(delta.size() == 2);

    ostringstream strm;
    strm << "0th: " << delta[0].transpose() << "     1st: " << delta[1].transpose();

    return strm.str();
}


inline AllGrippersSinglePoseDelta stepInDirection(const AllGrippersSinglePoseDelta& start, const AllGrippersSinglePoseDelta& direction, const double step_size)
{
    assert(start.size() == direction.size());

    AllGrippersSinglePoseDelta result(start.size());
    for (size_t ind = 0; ind < start.size(); ++ind)
    {
        result[ind] = start[ind] + step_size * direction[ind];
    }
    return result;
}

inline AllGrippersSinglePoseDelta stepInDirection(const AllGrippersSinglePoseDelta& start, const Eigen::VectorXd& direction, const double step_size)
{
    AllGrippersSinglePoseDelta dir_as_eigen = EigenHelpers::EigenVectorXToVectorEigenVector<double, 6>(direction);
    return stepInDirection(start, dir_as_eigen, step_size);
}

inline AllGrippersSinglePoseDelta stepInDirection(const AllGrippersSinglePoseDelta& start, const std::vector<double>& direction, const double step_size)
{
    AllGrippersSinglePoseDelta dir_as_eigen = EigenHelpers::StdVectorXToVectorEigenVector<double, 6>(direction);
    return stepInDirection(start, dir_as_eigen, step_size);
}

inline AllGrippersSinglePoseDelta projectToMaxDeltaConstraint(const AllGrippersSinglePoseDelta& start, const double max_step_size)
{
    // Each gripper is handled independently
    AllGrippersSinglePoseDelta result(start.size());
    for (size_t ind = 0; ind < start.size(); ++ind)
    {
        const double norm = GripperVelocity6dNorm(start[ind]);
        if (norm <= max_step_size)
        {
            result[ind] = start[ind];
        }
        else
        {
            result[ind] = start[ind] * (max_step_size / norm);
        }
    }
    return result;
}

DeformableController::OutputData StretchingAvoidanceController::solvedByGradientDescent(const InputData& input_data)
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
        assert(false && "Not implemented");
    }
    else
    {
        const double max_step_size = robot_->max_gripper_velocity_norm_ * robot_->dt_;
        const double differencing_step_size = max_step_size / 100.0;
        const double initial_gradient_step_size = max_step_size / 10.0;

        bool converged = false;
        AllGrippersSinglePoseDelta robot_motion(num_grippers, kinematics::Vector6d::Zero());
        ObjectPointSet current_object_delta = model_->getObjectDelta(current_world_state, robot_motion);
        double current_error = errorOfControlByPrediction(current_object_delta, desired_object_p_dot, desired_p_dot_weight);
        double current_stretching_constraint = stretchingFunctionEvaluation(input_data, robot_motion);

        while (!converged)
        {
            std::cout << "Start of loop motion:  " << print(robot_motion) << std::endl;
            std::cout << "Start of loop error:   " << current_error << std::endl;
            std::cout << "Start of stretching:   " << current_stretching_constraint << std::endl;

            std::vector<AllGrippersSinglePoseDelta> test_robot_motion(num_grippers * 6, robot_motion);
            std::vector<ObjectPointSet> predicted_object_movement(num_grippers * 6);
            Eigen::VectorXd error_numerical_gradient(num_grippers * 6);
            Eigen::VectorXd stretching_constraint_numerical_gradient(num_grippers * 6);

            for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; ++gripper_ind)
            {
                for (size_t single_gripper_dir_ind = 0; single_gripper_dir_ind < 6; ++single_gripper_dir_ind)
                {
                    const auto test_input_ind = gripper_ind * 6 + single_gripper_dir_ind;

                    AllGrippersSinglePoseDelta& local_test_motion = test_robot_motion[test_input_ind];
                    local_test_motion[gripper_ind](single_gripper_dir_ind) += differencing_step_size;

                    predicted_object_movement[test_input_ind] = model_->getObjectDelta(current_world_state, local_test_motion);

                    error_numerical_gradient(test_input_ind) =
                            (errorOfControlByPrediction(
                                 predicted_object_movement[test_input_ind],
                                 desired_object_p_dot,
                                 desired_p_dot_weight) - current_error)
                            / differencing_step_size;

                    stretching_constraint_numerical_gradient(test_input_ind) =
                            (stretchingFunctionEvaluation(
                                 input_data,
                                 local_test_motion) - current_stretching_constraint)
                            / differencing_step_size;
                }
            }

            std::cout << "Error Gradient:      " << error_numerical_gradient.transpose() << std::endl;
            std::cout << "Stretching Gradient: " << stretching_constraint_numerical_gradient.transpose() << std::endl;

            double error_gradient_step_size = -initial_gradient_step_size;
            auto error_downhill_step = stepInDirection(robot_motion, error_numerical_gradient, error_gradient_step_size);
            auto max_movement_projected_step = projectToMaxDeltaConstraint(error_downhill_step, max_step_size);
            auto predicted_object_delta = model_->getObjectDelta(current_world_state, max_movement_projected_step);
            double next_downhill_error = errorOfControlByPrediction(predicted_object_delta, desired_object_p_dot, desired_p_dot_weight);

            int downhill_attempt_ind = 0;
            while (next_downhill_error > current_error && downhill_attempt_ind < 10)
            {
                ++downhill_attempt_ind;
                error_gradient_step_size *= 0.7;
                error_downhill_step = stepInDirection(robot_motion, error_numerical_gradient, error_gradient_step_size);
                max_movement_projected_step = projectToMaxDeltaConstraint(error_downhill_step, max_step_size);
                predicted_object_delta = model_->getObjectDelta(current_world_state, max_movement_projected_step);
                next_downhill_error = errorOfControlByPrediction(predicted_object_delta, desired_object_p_dot, desired_p_dot_weight);
            }

            std::cout << "Post downhill motion:  " << print(max_movement_projected_step) << std::endl;

            double constraint_gradient_step_size = -initial_gradient_step_size / 10.0;
            auto stretching_constraint_correction_result = max_movement_projected_step;
            double new_constraint_value = stretchingFunctionEvaluation(input_data, stretching_constraint_correction_result);

            int constraint_satisfaction_ind = 0;
            while (new_constraint_value > 0.0 && constraint_satisfaction_ind < 10)
            {
                ++constraint_satisfaction_ind;

                std::cout << "Stretching  Pre:  " << print(stretching_constraint_correction_result) << std::endl;

                stretching_constraint_correction_result =
                        projectToMaxDeltaConstraint(
                            stepInDirection(stretching_constraint_correction_result, stretching_constraint_numerical_gradient, constraint_gradient_step_size),
                            max_step_size);

                std::cout << "Stretching Post: " << print(stretching_constraint_correction_result) << std::endl;

                new_constraint_value = stretchingFunctionEvaluation(input_data, stretching_constraint_correction_result);

                constraint_gradient_step_size -= initial_gradient_step_size / 10.0;
            }

            std::cout << "Post stretching tweak: " << print(stretching_constraint_correction_result) << std::endl;

            const auto loop_final_robot_motion = stretching_constraint_correction_result;
            const auto loop_final_predicted_object_delta = model_->getObjectDelta(current_world_state, loop_final_robot_motion);
            const double loop_final_downhill_error = errorOfControlByPrediction(loop_final_predicted_object_delta, desired_object_p_dot, desired_p_dot_weight);
            const double loop_final_constraint_value = new_constraint_value;

            // If we could not make progress, then return whatever the last valid movement we had was
            if (new_constraint_value > 0.0 || loop_final_downhill_error > current_error)
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
                current_stretching_constraint = loop_final_constraint_value;
            }
        }

        return OutputData(robot_motion, current_object_delta, Eigen::VectorXd());;
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

double StretchingAvoidanceController::stretchingFunctionEvaluation(
        const InputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
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
    // Assume knowing there are two grippers.
    assert(grippers_data_.size() == 2 || "grippers size is not 2, stretching vector visualization not developed");

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
    const EigenHelpers::VectorVector3d per_gripper_stretching_correction_vector =
        {first_correction_vector.normalized(), second_correction_vector.normalized()};

    // Transform the input points based on the motion of each gripper
    const Eigen::Vector3d first_point_in_first_gripper_frame   = gripper0_current_pose.inverse() * point_on_first_gripper_before_motion;
    const Eigen::Vector3d second_point_in_second_gripper_frame = gripper1_current_pose.inverse() * point_on_second_gripper_before_motion;
    const Eigen::Vector3d point_on_first_gripper_after_motion  = gripper0_test_pose * first_point_in_first_gripper_frame;
    const Eigen::Vector3d point_on_second_gripper_after_motion = gripper1_test_pose * second_point_in_second_gripper_frame;

    const std::vector<Eigen::Vector3d> point_motion =
    {
        point_on_first_gripper_after_motion - point_on_first_gripper_before_motion,
        point_on_second_gripper_after_motion - point_on_second_gripper_before_motion
    };

    double stretching_cos = 0.0;
    // sample_count_ > -1 means only sample one gripper each time
    if ((sample_count_ > -1) && (gripper_controller_type_ == StretchingAvoidanceControllerSolverType::RANDOM_SAMPLING))
    {
        assert(sample_count_ < (int)(per_gripper_stretching_correction_vector.size()));
        const Eigen::Vector3d point_movement_direction = point_motion[sample_count_].normalized();
        stretching_cos = point_movement_direction.dot(per_gripper_stretching_correction_vector[sample_count_]);
    }
    else
    {
        stretching_cos = 0.0;
        for (size_t gripper_ind = 0; gripper_ind < current_grippers_pose.size(); gripper_ind++)
        {
            const Eigen::Vector3d point_movement_direction = point_motion[gripper_ind].normalized();
            stretching_cos += point_movement_direction.dot(per_gripper_stretching_correction_vector[gripper_ind]);
        }
        stretching_cos /= (double)(current_grippers_pose.size());
    }

    if (std::isnan(stretching_cos))
    {
        std::cout << "Point on first gripper:        " << point_on_first_gripper_before_motion.transpose() << std::endl;
        std::cout << "Point on second gripper:       " << point_on_second_gripper_before_motion.transpose() << std::endl;
        std::cout << "First correction vector:       " << first_correction_vector.transpose() << std::endl;
        std::cout << "Second correction vector:      " << second_correction_vector.transpose() << std::endl;
        std::cout << "Point on first gripper after:  " << point_on_first_gripper_after_motion.transpose() << std::endl;
        std::cout << "Point on second gripper after: " << point_on_second_gripper_after_motion.transpose() << std::endl;

        std::cout << (grippers_test_poses[0].translation() - current_grippers_pose[1].translation()).normalized().transpose() << std::endl;
        std::cout << (grippers_test_poses[1].translation() - current_grippers_pose[2].translation()).normalized().transpose() << std::endl;

        assert(!std::isnan(stretching_cos));
    }

    return stretching_cos;
}
