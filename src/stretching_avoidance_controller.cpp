#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <omp.h>

#include "smmap/stretching_avoidance_controller.h"
#include "smmap/ros_communication_helpers.hpp"

#include "smmap/grippers.hpp"
#include <kinematics_toolbox/kinematics.h>

using namespace smmap;

StretchingAvoidanceController::StretchingAvoidanceController(
        ros::NodeHandle& nh,
        ros::NodeHandle& ph,
        RobotInterface& robot,
        const sdf_tools::SignedDistanceField& sdf,
        std::mt19937_64& generator,
        Visualizer& vis,
        GripperControllerType gripper_controller_type,
        const DeformableModel::Ptr& deformable_model,
        const int64_t max_count,
        const double distance_to_obstacle_threshold)
    : object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh)))
    , gripper_collision_checker_(nh)
    , grippers_data_(robot.getGrippersData())
    , enviroment_sdf_(sdf)
    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)
    , vis_(vis)
    , gripper_controller_type_(gripper_controller_type)
    , deformable_type_(GetDeformableType(nh))
    , task_type_(GetTaskType(nh))
    , model_(deformable_model)
    , distance_to_obstacle_threshold_(GetRobotGripperRadius())
    , max_stretch_factor_(GetMaxStretchFactor(ph))
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

std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> StretchingAvoidanceController::getGripperMotion_impl(
        const DeformableModel::DeformableModelInputData& input_data,
        const double max_gripper_velocity)
{
    switch (gripper_controller_type_)
    {
        case GripperControllerType::RANDOM_SAMPLING:
            return solvedByRandomSampling(
                        input_data,
                        max_gripper_velocity);
            break;

        case GripperControllerType::NOMAD_OPTIMIZATION:
            return solvedByNomad(
                        input_data,
                        max_gripper_velocity);
            break;

        default:
            assert(false && "This code should be un-reachable");
            break;
    };
}


/////////////////////////////////////////////////////////////////////////////////
// Private optimization function
/////////////////////////////////////////////////////////////////////////////////

std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> StretchingAvoidanceController::solvedByRandomSampling(
        const DeformableModel::DeformableModelInputData& input_data,
        const double max_gripper_velocity)
{
    const double max_step_size = max_gripper_velocity * input_data.dt_;
    const WorldState& current_world_state = input_data.world_current_state_;

    const Eigen::VectorXd& desired_object_p_dot =
            input_data.desired_object_motion_.delta;
          //  input_data.task_desired_object_delta_fn_(current_world_state).delta;
    const Eigen::VectorXd& desired_p_dot_weight =
            input_data.desired_object_motion_.weight;
          //  input_data.task_desired_object_delta_fn_(current_world_state).weight;

    const ssize_t num_grippers = current_world_state.all_grippers_single_pose_.size();
    const ssize_t num_nodes = current_world_state.object_configuration_.cols();

    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(current_world_state.object_configuration_);

    std::vector<std::pair<AllGrippersSinglePoseDelta, double>> per_thread_optimal_command(
//                arc_helpers::GetNumOMPThreads(),
                1,
                std::make_pair(AllGrippersSinglePoseDelta(), std::numeric_limits<double>::infinity()));

    // Checking the stretching status for current object configuration for once
    over_stretch_ = false;

    for (ssize_t first_node = 0; first_node < num_nodes; ++first_node)
    {
        for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
        {
            const double max_distance = max_stretch_factor_ * object_initial_node_distance_(first_node, second_node);
	    if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
	    {
	        over_stretch_ = true;
	        break;
	    }
	}
        if (over_stretch_)
        {
            break;
	}
    }

//    #pragma omp parallel for
    for (int64_t ind_count = 0; ind_count < max_count_; ind_count++)
    {
        AllGrippersSinglePoseDelta grippers_motion_sample = allGripperPoseDeltaSampler(num_grippers, max_step_size);

//        #if defined(_OPENMP)
//        const size_t thread_num = (size_t)omp_get_thread_num();
//        #else
        const size_t thread_num = 0;
//        #endif

        // Use constraint_violation checker for gripper collosion
        // Constraint violation checking here
        const bool collision_violation = gripperCollisionCheckResult(
                    current_world_state.all_grippers_single_pose_,
                    grippers_motion_sample);

        bool stretching_violation = stretchingDetection(
                    input_data,
                    grippers_motion_sample);

        // If no constraint violation
        if (!collision_violation && !stretching_violation)
        {
            std::pair<AllGrippersSinglePoseDelta, double>& current_thread_optimal = per_thread_optimal_command[thread_num];

            // get predicted object motion
            ObjectPointSet predicted_object_p_dot = model_->getObjectDelta(
                        input_data,
                        grippers_motion_sample);

            double sample_error = errorOfControlByPrediction(predicted_object_p_dot,
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
    AllGrippersSinglePoseDelta optimal_gripper_command;
    for (size_t thread_idx = 0; thread_idx < per_thread_optimal_command.size(); thread_idx++)
    {
        if (per_thread_optimal_command[thread_idx].second < best_error)
        {
            optimal_gripper_command = per_thread_optimal_command[thread_idx].first;
            best_error = per_thread_optimal_command[thread_idx].second;
        }
    }

    std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> suggested_grippers_command(
                optimal_gripper_command,
                ObjectPointSet::Zero(3, num_nodes));

    if(sample_count_ >= 0)
    {
        sample_count_++;
        if(sample_count_ >= num_grippers)
        {
            sample_count_=0;
        }
    }
    if(!(suggested_grippers_command.first.size() > 0))
    {
        suggested_grippers_command.first = setAllGripperPoseDeltaZero(num_grippers);
    }

    suggested_grippers_command.second = model_->getObjectDelta(
                input_data,
                suggested_grippers_command.first);

    return suggested_grippers_command;
}


std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> StretchingAvoidanceController::solvedByNomad(
        const DeformableModel::DeformableModelInputData &input_data,
        const double max_gripper_velocity)
{
    const double max_step_size = max_gripper_velocity * input_data.dt_;
    const WorldState& current_world_state = input_data.world_current_state_;

    const Eigen::VectorXd& desired_object_p_dot =
            input_data.desired_object_motion_.delta;
          //  input_data.task_desired_object_delta_fn_(current_world_state).delta;
    const Eigen::VectorXd& desired_p_dot_weight =
            input_data.desired_object_motion_.weight;
          //  input_data.task_desired_object_delta_fn_(current_world_state).weight;

    const ssize_t num_grippers = current_world_state.all_grippers_single_pose_.size();
    const ssize_t num_nodes = current_world_state.object_configuration_.cols();

    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(current_world_state.object_configuration_);

    AllGrippersSinglePoseDelta optimal_gripper_command;

    // Check object current stretching status
    // Checking the stretching status for current object configuration for once
    over_stretch_ = false;

    for (ssize_t first_node = 0; first_node < num_nodes; ++first_node)
        {
            for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
            {
                const double max_distance = max_stretch_factor_ * object_initial_node_distance_(first_node, second_node);
                if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
                {
                    over_stretch_ = true;
                    break;
                }
            }
            if (over_stretch_)
            {
                break;
            }
        }

    // Return value of objective function, cost = norm(p_dot_desired - p_dot_test)
    const std::function<double(const AllGrippersSinglePoseDelta&)> eval_error_cost_fn = [&] (
            const AllGrippersSinglePoseDelta& test_gripper_motion)
    {
        ObjectPointSet predicted_object_p_dot = model_->getObjectDelta(
                    input_data,
                    test_gripper_motion);

        return errorOfControlByPrediction(predicted_object_p_dot,
                                          desired_object_p_dot,
                                          desired_p_dot_weight);
    };

    // Return the min distance of gripper to obstacle
    const std::function<double(const AllGrippersSinglePoseDelta&)> collision_constraint_fn = [&] (
            const AllGrippersSinglePoseDelta& test_gripper_motion)
    {
        const double min_dis_to_obstacle = gripperCollisionCheckHelper(
                    current_world_state.all_grippers_single_pose_,
                    test_gripper_motion);
        return min_dis_to_obstacle;
    };

    // Return the sum of cos (an indicator of direction) gripper motion to stretching vector
    const std::function<double(const AllGrippersSinglePoseDelta&)> stretching_constraint_fn = [&] (
            const AllGrippersSinglePoseDelta& test_gripper_motion)
    {
        if(test_gripper_motion.size()!=2 || test_gripper_motion.size()!=num_grippers)
        {
            assert(false && "num of grippers not match");
        }
        switch (deformable_type_)
        {
            case ROPE:
            {
                return ropeTwoGripperStretchingHelper(
                            input_data,
                            test_gripper_motion);
            }
            case CLOTH:
            {
                return clothTwoGripperStretchingHelper(
                            input_data,
                            test_gripper_motion);
            }
            default:
            {
                assert(false && "deformable_type is neither rope nor cloth");
                return 0.0;
            }
        }
    };

    const std::function<double(const AllGrippersSinglePoseDelta&)> gripper_motion_constraint_fn = [&] (
            const AllGrippersSinglePoseDelta& test_gripper_motion)
    {
        double max_value = 0.0;
        for (ssize_t gripper_ind = 0; gripper_ind < test_gripper_motion.size(); gripper_ind += 6)
        {
            const double velocity_norm = GripperVelocity6dNorm(test_gripper_motion.at(gripper_ind));
            if (velocity_norm > max_value)
            {
                max_value = velocity_norm;
            }
        }
        return max_value;

    };

    ofstream out(log_file_path_.c_str(), ios::out);
    // NOMAD::Display out ( std::cout );
    out.precision ( NOMAD::DISPLAY_PRECISION_STD );

    try
    {
        // NOMAD initializations:
        NOMAD::begin(0, nullptr);

        // parameters creation:
        NOMAD::Parameters p ( out );
//        NOMAD::Parameters p;
        p.set_DIMENSION (6 * num_grippers);             // number of variables

        vector<NOMAD::bb_output_type> bbot (4); // definition of
        bbot[0] = NOMAD::OBJ;                   // output types
        // TODO: might need to decide which kind of constraint to use
        bbot[1] = NOMAD::PB;
        bbot[2] = NOMAD::PB;
        bbot[3] = NOMAD::PB;

        if (fix_step_)
        {
            bbot.push_back(NOMAD::EB);
        }

        p.set_BB_OUTPUT_TYPE(bbot);

        const int x_dim = 6 * num_grippers;
        const int size_of_initial_batch = 5;
        // Set a list of initial points
        /*
        {
            NOMAD::Point x0 = NOMAD::Point(x_dim, 0.0);
            NOMAD::Point x1 = NOMAD::Point(x_dim, 0.0);
            NOMAD::Point x2 = NOMAD::Point(x_dim, 0.0);
            NOMAD::Point x3 = NOMAD::Point(x_dim, 0.0);
            NOMAD::Point x4 = NOMAD::Point(x_dim, 0.0);
            NOMAD::Point x5 = NOMAD::Point(x_dim, 0.0);
            NOMAD::Point x6 = NOMAD::Point(x_dim, 0.0);
            NOMAD::Point x7 = NOMAD::Point(x_dim, 0.0);

            x1.reset(x_dim, max_step_size / 6.0);
            x2.set_coord(0, max_step_size / 3.0);
            x3.set_coord(x_dim / 2, max_step_size / 3.0);
            x4.set_coord(1, max_step_size/3.0);
            x5.set_coord(x_dim / 2 + 1, max_step_size/3.0);
            x6.set_coord(2, max_step_size/3.0);
            x7.set_coord(x_dim / 2 + 2, max_step_size/3.0);

            p.set_X0 (x0);  // starting point
            p.set_X0 (x1);  // starting point
            p.set_X0 (-x1);  // starting point
            p.set_X0(-x2-x3);
            p.set_X0(-x4 - x5);
            p.set_X0(-x6 - x7);
            p.set_X0(x2 + x3);
            p.set_X0(x4 + x5);
            p.set_X0(-x6 - x7);
            p.set_X0(x0 + x2 + x4 + x6);
            p.set_X0(x0 + x1 + x3 + x5);
            p.set_X0(x0 - x2 - x4 - x6);
            p.set_X0(x0 - x1 - x3 - x5);

        }
        */
        {
            for (int sample_ind = 0; sample_ind < size_of_initial_batch; sample_ind++)
            {
                NOMAD::Point x0 = NOMAD::Point(x_dim, 0.0);
                for (int coord_ind = 0; coord_ind < x_dim; coord_ind++)
                {
                    x0.set_coord(coord_ind, EigenHelpers::Interpolate(-max_step_size, max_step_size, uniform_unit_distribution_(generator_)));
                }
                p.set_X0(x0);
            }
        }


        p.set_LOWER_BOUND(NOMAD::Point(6 * num_grippers, -max_step_size)); // all var. >= -6
        p.set_UPPER_BOUND(NOMAD::Point(6 * num_grippers, max_step_size)); // all var. >= -6

        p.set_MAX_BB_EVAL (max_count_);     // the algorithm terminates after
                                     // 100 black-box evaluations
        p.set_DISPLAY_DEGREE(2);
        //p.set_SGTELIB_MODEL_DISPLAY("");
        p.set_SOLUTION_FILE("sol.txt");

        // parameters validation:
        p.check();

        const double gripper_radius = 0.023;
        // custom evaluator creation:
        GripperMotionNomadEvaluator ev(p,
                                       num_grippers,
                                       gripper_radius,
                                       stretching_cosine_threshold_,
                                       max_step_size,
                                       eval_error_cost_fn,
                                       collision_constraint_fn,
                                       stretching_constraint_fn,
                                       gripper_motion_constraint_fn,
                                       fix_step_);

        // algorithm creation and execution:
        NOMAD::Mads mads ( p , &ev );
        mads.run();

        const NOMAD::Eval_Point* best_x = mads.get_best_feasible();

        optimal_gripper_command = ev.evalPointToGripperPoseDelta(*best_x);
    }
    catch (exception& e)
    {
        cerr << "\nNOMAD has been interrupted (" << e.what() << ")\n\n";
    }

    NOMAD::Slave::stop_slaves ( out );
    NOMAD::end();

    std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> suggested_grippers_command(
                optimal_gripper_command,
                ObjectPointSet::Zero(3, num_nodes));

    if(!(suggested_grippers_command.first.size() > 0))
    {
        suggested_grippers_command.first = setAllGripperPoseDeltaZero(num_grippers);
    }

    suggested_grippers_command.second = model_->getObjectDelta(
                input_data,
                suggested_grippers_command.first);

    return suggested_grippers_command;

}


//////////////////////////////////////////////////////////////////////////////////
// Helper function
//////////////////////////////////////////////////////////////////////////////////

kinematics::Vector6d StretchingAvoidanceController::singleGripperPoseDeltaSampler(const double max_delta)
{
    const double x_trans = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double y_trans = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double z_trans = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));

    const double x_rot = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double y_rot = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));
    const double z_rot = EigenHelpers::Interpolate(-max_delta, max_delta, uniform_unit_distribution_(generator_));

    kinematics::Vector6d random_sample;

    double raw_norm = std::sqrt(std::pow(x_trans,2) + std::pow(y_trans,2) + std::pow(z_trans,2));

    if ( raw_norm > 0.000001 && (fix_step_ || raw_norm > max_delta))
    {
        random_sample(0) = x_trans/raw_norm * max_delta;
        random_sample(1) = y_trans/raw_norm * max_delta;
        random_sample(2) = z_trans/raw_norm * max_delta;
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
    AllGrippersSinglePoseDelta grippers_motion_sample;

    // if sample_count_ < 0, return all-sampled motion, otherwise, return one-for-each-time sample
    if (sample_count_ < 0)
    {
        for (ssize_t ind_gripper = 0; ind_gripper < num_grippers; ind_gripper++)
        {
            // Eigen::Isometry3d single_gripper_motion_sample = EigenHelpers::ExpTwist(singleGripperPoseDeltaSampler(), 1.0);
            grippers_motion_sample.push_back(singleGripperPoseDeltaSampler(max_delta));
        }
        return grippers_motion_sample;
    }
    else if (sample_count_ < num_grippers)
    {
        for (ssize_t ind_gripper = 0; ind_gripper < num_grippers; ind_gripper++)
        {
            if (ind_gripper == sample_count_)
            {
                grippers_motion_sample.push_back(singleGripperPoseDeltaSampler(max_delta));
            }
            else
            {
                kinematics::Vector6d no_sample = Eigen::MatrixXd::Zero(6,1);
                grippers_motion_sample.push_back(no_sample);
            }
        }

        return grippers_motion_sample;
    }
    else
    {
        assert(false && "This code should not be reachable");
    }
}

AllGrippersSinglePoseDelta StretchingAvoidanceController::setAllGripperPoseDeltaZero(const ssize_t num_grippers)
{
    const kinematics::Vector6d no_movement = kinematics::Vector6d::Zero();
    const AllGrippersSinglePoseDelta grippers_motion_sample(num_grippers, no_movement);
    return grippers_motion_sample;
}

double StretchingAvoidanceController::errorOfControlByPrediction(
        const ObjectPointSet predicted_object_p_dot,
        const Eigen::VectorXd& desired_object_p_dot,
        const Eigen::VectorXd& desired_p_dot_weight) const
{
    ssize_t num_nodes = predicted_object_p_dot.cols();
    double sum_of_error = 0;
    const double zero_thrshold = 0.000001;

    for (ssize_t node_ind = 0; node_ind < num_nodes; node_ind++)
    {
        Eigen::Vector3d node_predicted_p_dot = predicted_object_p_dot.col(node_ind);
        Eigen::Vector3d node_desired_p_dot = desired_object_p_dot.segment<3>(node_ind*3);

        // Only none_zero desired p dot is considered.
        if(desired_p_dot_weight(node_ind * 3) > 0)
        {
            double node_p_dot_error = (node_predicted_p_dot - node_desired_p_dot).norm();
            sum_of_error += node_p_dot_error * desired_p_dot_weight(node_ind *3);
        }
    }

    return sum_of_error;
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
    line_ends.push_back(line_starts.at(0) + 0.5 * first_correction_vector);
    line_ends.push_back(line_starts.at(1) + 0.5 * second_correction_vector);

    vis_.visualizeLines("gripper overstretch motion",
                        line_starts,
                        line_ends,
                        Visualizer::Orange());
}

void StretchingAvoidanceController::visualize_cloth_stretching_vector(
        const ObjectPointSet& object_configuration)
{
    // Assume knowing there are two grippers.
    assert(grippers_data_.size()==2 || "grippers size is not 2, stretching vector visualization not developed");

    const StretchingVectorInfo& first_stretching_vector_info = grippers_data_.at(0).stretching_vector_info_;
    const std::vector<long>& first_from_nodes = first_stretching_vector_info.from_nodes_;
    const std::vector<long>& first_to_nodes = first_stretching_vector_info.to_nodes_;
    const std::vector<double>& first_contribution = first_stretching_vector_info.node_contribution_;

    const StretchingVectorInfo& second_stretching_vector_info = grippers_data_.at(1).stretching_vector_info_;
    const std::vector<long>& second_from_nodes = second_stretching_vector_info.from_nodes_;
    const std::vector<long>& second_to_nodes = second_stretching_vector_info.to_nodes_;
    const std::vector<double>& second_contribution = second_stretching_vector_info.node_contribution_;

    Eigen::Vector3d first_correction_vector = Eigen::MatrixXd::Zero(3,1);
    for (int stretching_ind = 0; stretching_ind < first_from_nodes.size(); stretching_ind++)
    {
        first_correction_vector +=
                first_contribution.at(stretching_ind) *
                (object_configuration.block<3, 1>(0, first_to_nodes.at(stretching_ind))
                 - object_configuration.block<3, 1>(0, first_from_nodes.at(stretching_ind)));
    }
    Eigen::Vector3d second_correction_vector = Eigen::MatrixXd::Zero(3,1);
    for (int stretching_ind = 0; stretching_ind < second_from_nodes.size(); stretching_ind++)
    {
        second_correction_vector +=
                second_contribution.at(stretching_ind) *
                (object_configuration.block<3, 1>(0, second_to_nodes.at(stretching_ind))
                 - object_configuration.block<3, 1>(0, second_from_nodes.at(stretching_ind)));
    }

    EigenHelpers::VectorVector3d line_starts;
    EigenHelpers::VectorVector3d line_ends;
    line_starts.push_back(object_configuration.block<3,1>(0, first_from_nodes.at(0)));
    line_starts.push_back(object_configuration.block<3,1>(0, second_from_nodes.at(0)));
    line_ends.push_back(line_starts.at(0) + 10 * first_correction_vector);
    line_ends.push_back(line_starts.at(1) + 10 * second_correction_vector);

    vis_.visualizeLines("gripper overstretch motion",
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

    vis_.visualizeLines("gripper motion",
                        line_starts,
                        line_ends,
                        Visualizer::Olive());
}

const double StretchingAvoidanceController::gripperCollisionCheckHelper(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    const auto grippers_test_poses = kinematics::applyTwist(current_gripper_pose, test_gripper_motion);

    double min_collision_distance = std::numeric_limits<double>::infinity();

    for (size_t gripper_idx = 0; gripper_idx < grippers_test_poses.size(); ++gripper_idx)
    {
        const auto collision_result = enviroment_sdf_.EstimateDistance3d(grippers_test_poses[gripper_idx].translation());
        if (collision_result.first < min_collision_distance)
        {
            min_collision_distance = collision_result.first;
        }
    }

    return min_collision_distance;
}

bool StretchingAvoidanceController::gripperCollisionCheckResult(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    bool collision_violation = false;

    const double min_dis_to_obstacle = gripperCollisionCheckHelper(current_gripper_pose,
                                                                   test_gripper_motion);
    if (min_dis_to_obstacle < distance_to_obstacle_threshold_)
    {
        collision_violation = true;
    }

    return collision_violation;
}


bool StretchingAvoidanceController::stretchingDetection(
        const DeformableModel::DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    switch (deformable_type_)
    {
        case ROPE:
        {
            return ropeTwoGrippersStretchingDetection(input_data, test_gripper_motion);
        }

        case CLOTH:
        {
            return clothTwoGrippersStretchingDetection(input_data, test_gripper_motion);
            break;
        }
        default:
        {
            assert(false && "stretching detection for neither cloth nor rope");
            return false;            
        }
    }
    return false;
}

double StretchingAvoidanceController::ropeTwoGripperStretchingHelper(
        const DeformableModel::DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    double stretching_sum = 0.0;
    double stretching_cos = 1.0; // return a value > stretching_cos_threshold

    if(over_stretch_)
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
            case GripperControllerType::RANDOM_SAMPLING:
            {
                if(sample_count_ > -1)
                {
                    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
                    {
                        Eigen::Vector3d resulting_gripper_motion = grippers_test_poses.at(gripper_ind).translation()
                                - current_gripper_pose.at(gripper_ind).translation();
                       stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector.at(gripper_ind));
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
                        Eigen::Vector3d resulting_gripper_motion = grippers_test_poses.at(gripper_ind).translation()
                                - current_gripper_pose.at(gripper_ind).translation();
                        stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector.at(gripper_ind))
                                / resulting_gripper_motion.norm();
                    }
                    stretching_cos = stretching_sum / current_gripper_pose.size();
               }
                break;
            }
            case GripperControllerType::NOMAD_OPTIMIZATION:
            {
                for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
                {
                    Eigen::Vector3d resulting_gripper_motion = grippers_test_poses.at(gripper_ind).translation()
                            - current_gripper_pose.at(gripper_ind).translation();
                    stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector.at(gripper_ind))
                            / resulting_gripper_motion.norm();
                }
                stretching_cos = stretching_sum / current_gripper_pose.size();
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

bool StretchingAvoidanceController::ropeTwoGrippersStretchingDetection(
        const DeformableModel::DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{

    bool motion_induced_streching = false;

    if(over_stretch_)
    {
        double streching_sum = ropeTwoGripperStretchingHelper(
                    input_data,
                    test_gripper_motion);

        if (streching_sum <= stretching_cosine_threshold_)
        {
            motion_induced_streching = true;
        }
    }

    return motion_induced_streching;

}

double StretchingAvoidanceController::clothTwoGripperStretchingHelper(
        const DeformableModel::DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    const ObjectPointSet& object_configuration = input_data.world_current_state_.object_configuration_;
    const AllGrippersSinglePose& current_gripper_pose = input_data.world_current_state_.all_grippers_single_pose_;
    double stretching_sum = 0.0;
    double stretching_cos = 1.0;

    if(!over_stretch_)
    {
        return stretching_cos;
    }

    // Assume knowing there are two grippers.
    assert(grippers_data_.size()==2 || "grippers size is not 2, stretching vector visualization not developed");

    const StretchingVectorInfo& first_stretching_vector_info = grippers_data_.at(0).stretching_vector_info_;
    const std::vector<long>& first_from_nodes = first_stretching_vector_info.from_nodes_;
    const std::vector<long>& first_to_nodes = first_stretching_vector_info.to_nodes_;
    const std::vector<double>& first_contribution = first_stretching_vector_info.node_contribution_;

    const StretchingVectorInfo& second_stretching_vector_info = grippers_data_.at(1).stretching_vector_info_;
    const std::vector<long>& second_from_nodes = second_stretching_vector_info.from_nodes_;
    const std::vector<long>& second_to_nodes = second_stretching_vector_info.to_nodes_;
    const std::vector<double>& second_contribution = second_stretching_vector_info.node_contribution_;

    Eigen::Vector3d point_on_first_gripper = Eigen::MatrixXd::Zero(3,1);
    Eigen::Vector3d point_on_second_gripper = Eigen::MatrixXd::Zero(3,1);

    Eigen::Vector3d first_correction_vector = Eigen::MatrixXd::Zero(3,1);
    for (int stretching_ind = 0; stretching_ind < first_from_nodes.size(); stretching_ind++)
    {
        first_correction_vector +=
                first_contribution.at(stretching_ind) *
                (object_configuration.block<3, 1>(0, first_to_nodes.at(stretching_ind))
                 - object_configuration.block<3, 1>(0, first_from_nodes.at(stretching_ind)));

        point_on_first_gripper +=
                first_contribution.at(stretching_ind) *
                object_configuration.block<3, 1>(0, first_from_nodes.at(stretching_ind));
    }
    if(first_correction_vector.norm() > 0)
    {
        first_correction_vector = first_correction_vector / first_correction_vector.norm();
    }
    point_on_first_gripper = point_on_first_gripper - current_gripper_pose.at(0).translation();

    Eigen::Vector3d second_correction_vector = Eigen::MatrixXd::Zero(3,1);
    for (int stretching_ind = 0; stretching_ind < second_from_nodes.size(); stretching_ind++)
    {
        second_correction_vector +=
                second_contribution.at(stretching_ind) *
                (object_configuration.block<3, 1>(0, second_to_nodes.at(stretching_ind))
                 - object_configuration.block<3, 1>(0, second_from_nodes.at(stretching_ind)));

        point_on_second_gripper +=
                second_contribution.at(stretching_ind) *
                object_configuration.block<3, 1>(0, second_from_nodes.at(stretching_ind));

    }
    if(second_correction_vector.norm() > 0)
    {
        second_correction_vector = second_correction_vector / second_correction_vector.norm();
    }
    point_on_second_gripper = point_on_second_gripper - current_gripper_pose.at(1).translation();

    const auto grippers_test_poses = kinematics::applyTwist(current_gripper_pose, test_gripper_motion);

    // Get track the point on edge of the gripper; stretching offset by geometric shape
    Eigen::Vector3d point_in_gripper_tm_first = current_gripper_pose.at(0).linear().inverse()
            * point_on_first_gripper;

    Eigen::Vector3d point_in_gripper_tm_second = current_gripper_pose.at(1).linear().inverse()
            * point_on_second_gripper;

    Eigen::Vector3d point_on_first_gripper_after = grippers_test_poses.at(0).linear()
            * point_in_gripper_tm_first;

    Eigen::Vector3d point_on_second_gripper_after = grippers_test_poses.at(1).linear()
            * point_in_gripper_tm_second;

    std::vector<Eigen::Vector3d> points_moving;
    points_moving.push_back((grippers_test_poses.at(0).translation()
                             + point_on_first_gripper_after)
                             - (current_gripper_pose.at(0).translation()
                                +point_on_first_gripper));
    points_moving.push_back((grippers_test_poses.at(1).translation()
                             + point_on_second_gripper_after)
                             - (current_gripper_pose.at(1).translation()
                                +point_on_second_gripper));

    double sum_resulting_motion_norm = 0.0;

    EigenHelpers::VectorVector3d stretching_correction_vector;
    stretching_correction_vector.push_back(first_correction_vector);
    stretching_correction_vector.push_back(second_correction_vector);

    // sample_count_ > -1 means only sample one gripper each time
    if((sample_count_ > -1) && (gripper_controller_type_ == GripperControllerType::RANDOM_SAMPLING))
    {
        Eigen::Vector3d resulting_gripper_motion = points_moving.at(sample_count_);
        stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector.at(sample_count_));

        sum_resulting_motion_norm += resulting_gripper_motion.norm();
        if(sum_resulting_motion_norm > 0.000000001)
        {
            stretching_cos = stretching_sum / sum_resulting_motion_norm;
        }
    }
    else
    {
        for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
        {
            Eigen::Vector3d resulting_gripper_motion = grippers_test_poses.at(gripper_ind).translation()
                    - current_gripper_pose.at(gripper_ind).translation();
            stretching_sum += resulting_gripper_motion.dot(stretching_correction_vector.at(gripper_ind))
                    / resulting_gripper_motion.norm();
        }
        stretching_cos = stretching_sum / current_gripper_pose.size();
    }

    return stretching_cos;

}

bool StretchingAvoidanceController::clothTwoGrippersStretchingDetection(
        const DeformableModel::DeformableModelInputData& input_data,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    assert(grippers_data_.size()==2 || "grippers size is not 2, stretching vector visualization not developed");
    bool motion_induced_streching = false;

    if(over_stretch_)
    {
        double stretching_sum = clothTwoGripperStretchingHelper(input_data,
                                                               test_gripper_motion);
        if(stretching_sum <= stretching_cosine_threshold_)
        {
            motion_induced_streching = true;
        }
    }
    return motion_induced_streching;
}
