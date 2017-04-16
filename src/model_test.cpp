#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/diminishing_rigidity_model.h"
#include "smmap/adaptive_jacobian_model.h"
#include "smmap/least_squares_jacobian_model.h"
#include "smmap/constraint_jacobian_model.h"

#include "smmap/model_test.h"

using namespace smmap;
using namespace EigenHelpersConversions;

modelTest::modelTest(RobotInterface& robot,
           Visualizer& vis,
           const TestSpecification::Ptr& test_specification)
    : nh_()
    , ph_("~")
    , robot_(robot)
    , vis_(vis)
    , test_specification_(test_specification)
    , logging_fn_(createLoggingFunction())
    , test_logging_fn_(createTestLogFunction())
    , planner_(robot_, vis_, test_specification_, logging_fn_, test_logging_fn_)
{
    initializeLogging();
}

void modelTest::execute()
{
    const int planning_horizion = GetPlanningHorizon(ph_);

    // Run the planner at whatever rate we've been given
    ROS_INFO_STREAM_NAMED("task", "Running our planner with a horizion of " << planning_horizion);
    WorldState world_feedback = robot_.start();
    const double start_time = world_feedback.sim_time_;

    initializeModelSet(world_feedback);

    // Track last world state
    WorldState last_world_state = world_feedback;
    ObjectDeltaAndWeight model_delta_p = test_specification_->calculateDesiredDirection(last_world_state);

    while (robot_.ok())
    {
        const WorldState current_world_state = world_feedback;
        const ObjectPointSet real_delta_p = current_world_state.object_configuration_-last_world_state.object_configuration_;



        last_world_state = current_world_state;

        // TODO: implement error function for test_specification
        // It should be the dynamics difference between real delta p and model delta p.
        const double current_error = test_specification_->calculateError(real_delta_p, model_delta_p);
        ROS_INFO_STREAM_NAMED("task", "Planner/Task sim time " << current_world_state.sim_time_ << "\t Error: " << current_error);

        ///// Mengyao's Test Log Here
        test_logging_fn_(current_world_state,real_delta_p,model_delta_p);

        // TODO: implement constraint violation function
        planner_.detectFutureConstraintViolations(last_world_state);

        // TODO: Revise the sendNextCommand function to update the world state;
        // in test_planner class; log data should also be updated
        // model_delta_p calculated inside this function should be campared with the real_delta_p above
        // The input should be q_delta
        world_feedback = planner_.sendNextCommand(last_world_state);

        // it is not the right way!!!!!! do it with model in test_planner
        model_delta_p = test_specification_->calculateDesiredDirection(last_world_state);

        if (unlikely(world_feedback.sim_time_ - start_time >= test_specification_->maxTime()));
        {
            robot_.shutdown();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal initialization helpers
////////////////////////////////////////////////////////////////////////////////

void modelTest::initializeModelSet(const WorldState& initial_world_state)
{
    // Initialze each model type with the shared data
    DeformableModel::SetGrippersData(robot_.getGrippersData());
    // TODO: fix this interface so that I'm not passing a null ptr here
    DeformableModel::SetCallbackFunctions(gripper_collision_check_fn_);
    DiminishingRigidityModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));
    ConstraintJacobianModel::SetInitialObjectConfiguration(GetObjectInitialConfiguration(nh_));

    const bool optimization_enabled = GetOptimizationEnabled(ph_);

    // Create some models and add them to the model set
    double translational_deformability, rotational_deformability;
    if (ph_.getParam("translational_deformability", translational_deformability) &&
             ph_.getParam("rotational_deformability", rotational_deformability))
    {
        ROS_INFO_STREAM_NAMED("task", "Overriding deformability values to "
                               << translational_deformability << " "
                               << rotational_deformability);

        planner_.addModel(std::make_shared<DiminishingRigidityModel>(
                              translational_deformability,
                              rotational_deformability,
                              optimization_enabled));
    }
    else if (GetUseMultiModel(ph_))
    {
        ////////////////////////////////////////////////////////////////////////
        // Diminishing rigidity models
        ////////////////////////////////////////////////////////////////////////

        const double deform_min = 0.0;
        const double deform_max = 25.0;
        const double deform_step = 4.0;

        for (double trans_deform = deform_min; trans_deform < deform_max; trans_deform += deform_step)
        {
            for (double rot_deform = deform_min; rot_deform < deform_max; rot_deform += deform_step)
            {
                planner_.addModel(std::make_shared<DiminishingRigidityModel>(
                                      trans_deform,
                                      rot_deform,
                                      optimization_enabled));
            }
        }
        ROS_INFO_STREAM_NAMED("task", "Num diminishing rigidity models: "
                               << std::floor((deform_max - deform_min) / deform_step));

        ////////////////////////////////////////////////////////////////////////
        // Adaptive jacobian models
        ////////////////////////////////////////////////////////////////////////

        const double learning_rate_min = 1e-10;
        const double learning_rate_max = 1.1e0;
        const double learning_rate_step = 10.0;

        const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
        {
            return test_specification_->calculateDesiredDirection(world_state);
        };

        const DeformableModel::DeformableModelInputData input_data(task_desired_direction_fn, initial_world_state, robot_.dt_);
        for (double learning_rate = learning_rate_min; learning_rate < learning_rate_max; learning_rate *= learning_rate_step)
        {
                planner_.addModel(std::make_shared<AdaptiveJacobianModel>(
                                      DiminishingRigidityModel(test_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(input_data),
                                      learning_rate,
                                      optimization_enabled));
        }
        ROS_INFO_STREAM_NAMED("task", "Num adaptive Jacobian models: "
                               << std::floor(std::log(learning_rate_max / learning_rate_min) / std::log(learning_rate_step)));

        ////////////////////////////////////////////////////////////////////////
        // Single manually tuned model
        ////////////////////////////////////////////////////////////////////////

//        planner_.addModel(std::make_shared<DiminishingRigidityModel>(
//                              task_specification_->defaultDeformability(),
//                              GetOptimizationEnabled(nh_)));
    }
    else if (GetUseAdaptiveModel(ph_))
    {
        const TaskDesiredObjectDeltaFunctionType task_desired_direction_fn = [&] (const WorldState& world_state)
        {
            // TODO : to be implemented, the function should generate the planned delta_p
            // In my model, the effective portion of delta_p is actually just the grasping points
            // Thus, the function should set all node_v zero, except for the nodes on the end-effector of gripper
            // Such node should have the same velocity as gripper endeffector.
            return test_specification_->calculateDesiredDirection(world_state);
        };

        // TODO: fix the task_specification_->defaultDeformablility();
        const DeformableModel::DeformableModelInputData input_data(task_desired_direction_fn, initial_world_state, robot_.dt_);
        planner_.addModel(std::make_shared<AdaptiveJacobianModel>(
                              DiminishingRigidityModel(test_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(input_data),
                              GetAdaptiveModelLearningRate(ph_),
                              optimization_enabled));
    }
    // Mengyao's model here
    else if (GetUseConstraintModel(ph_))
    {
        const double translation_dir_deformability=0.1;
        const double translation_dis_deformability=1.0;
        const double rotation_deformability=0.1;
        // Douoble check this usage
        const sdf_tools::SignedDistanceField environment_sdf(GetEnvironmentSDF(nh_));


        planner_.addModel(std::make_shared<ConstraintJacobianModel>(
                              translation_dir_deformability,
                              translation_dis_deformability,
                              rotation_deformability,
                              environment_sdf,
                              optimization_enabled));

    }
    // Mengyao's model above
    else
    {
        // TODO: fix the task_specification_->defaultDeformablility();
        ROS_INFO_STREAM_NAMED("task", "Using default deformability value of "
                               << test_specification_->defaultDeformability());

        // TODO: fix the task_specification_->defaultDeformablility();
        planner_.addModel(std::make_shared<DiminishingRigidityModel>(
                              test_specification_->defaultDeformability(),
                              optimization_enabled));

//        planner_.addModel(std::make_shared<LeastSquaresJacobianModel>(
//                              DiminishingRigidityModel(task_specification_->defaultDeformability(), false).getGrippersToObjectJacobian(robot_.getGrippersPose(), GetObjectInitialConfiguration(nh_)),
//                              2));
    }

    planner_.createBandits();
}

// TODO: To add the log for new defined error, data
void modelTest::initializeLogging()
{
    // Enable logging if it is requested
    logging_enabled_ = GetLoggingEnabled(nh_);

    if (logging_enabled_)
    {
        const std::string log_folder = GetLogFolder(nh_);

        ROS_INFO_STREAM_NAMED("planner", "Logging to " << log_folder);

        loggers.insert(std::make_pair<std::string, Log::Log>(
                            "time_bandit",
                            Log::Log(log_folder + "time_bandit.txt", false)));

//        loggers.insert(std::make_pair<std::string, Log::Log>(
//                            "error",
//                            Log::Log(log_folder + "error.txt", false)));

        loggers.insert(std::make_pair<std::string, Log::Log>(
                            "utility_mean",
                            Log::Log(log_folder + "utility_mean.txt", false)));

        loggers.insert(std::make_pair<std::string, Log::Log>(
                            "utility_covariance",
                            Log::Log(log_folder + "utility_covariance.txt", false)));

        loggers.insert(std::make_pair<std::string, Log::Log>(
                            "model_chosen",
                            Log::Log(log_folder + "model_chosen.txt", false)));

        loggers.insert(std::make_pair<std::string, Log::Log>(
                            "rewards_for_all_models",
                            Log::Log(log_folder + "rewards_for_all_models.txt", false)));

        loggers.insert(std::make_pair<std::string, Log::Log>(
                            "correlation_scale_factor",
                            Log::Log(log_folder + "correlation_scale_factor.txt", false)));

        // NEWLY ADD FOR TEST LOG
        test_loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "time_test",
                            Log::Log(log_folder + "time_test.txt", false)));

        test_loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "error_realtime",
                            Log::Log(log_folder + "error_realtime.txt", false)));

        test_loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "real_dp",
                            Log::Log(log_folder + "real_dp.txt", false)));
        test_loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "model_dp",
                            Log::Log(log_folder + "model_dp.txt", false)));

    }
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

void modelTest::logData(
        const WorldState& current_world_state,
        const Eigen::VectorXd& model_utility_mean,
        const Eigen::MatrixXd& model_utility_covariance,
        const ssize_t model_used,
        const std::vector<double>& rewards_for_all_models,
        const double correlation_strength_factor)
{
    if (logging_enabled_)
    {
        const Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
                    " ", " ", "", "");

        LOG(loggers.at("time_bandit"),
             current_world_state.sim_time_);

        // TODO, This function should return normed error
//        LOG(loggers.at("error"),
//             test_specification_->calculateError(current_world_state.object_configuration_));

        LOG(loggers.at("utility_mean"),
             model_utility_mean.format(single_line));

        LOG(loggers.at("utility_covariance"),
             model_utility_covariance.format(single_line));

        LOG(loggers.at("model_chosen"),
             model_used);

        LOG(loggers.at("rewards_for_all_models"),
            PrettyPrint::PrettyPrint(rewards_for_all_models, false, " "));

        LOG(loggers.at("correlation_scale_factor"),
            correlation_strength_factor);
    }
}

///////////////////// Log Data for Mengyao's Test ///////////////////

void modelTest::testLogData(
        const WorldState& current_world_state,
        const ObjectPointSet &real_delta_p,
        ObjectDeltaAndWeight &model_delta_p)
{
    if (logging_enabled_)
    {
        const Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
                    " ", " ", "", "");

        LOG(test_loggers_.at("time_test"),
             current_world_state.sim_time_);

        // TODO, This function should return normed error
        LOG(test_loggers_.at("error_realtime"),
             test_specification_->calculateError(real_delta_p, model_delta_p));

        LOG(test_loggers_.at("real_dp"),
             real_delta_p.format(single_line));

        LOG(test_loggers_.at("model_dp"),
             model_delta_p.delta.format(single_line));
    }
}


////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////

GripperCollisionCheckFunctionType modelTest::createGripperCollisionCheckFunction()
{
    return std::bind(&RobotInterface::checkGripperCollision,
                     &robot_,
                     std::placeholders::_1);
}

LoggingFunctionType modelTest::createLoggingFunction()
{
    return std::bind(&modelTest::logData,
                     this,
                     std::placeholders::_1,
                     std::placeholders::_2,
                     std::placeholders::_3,
                     std::placeholders::_4,
                     std::placeholders::_5,
                     std::placeholders::_6);
//                     std::placeholders::_7,
//                     std::placeholders::_8);
//                     std::placeholders::_9);
}

///////////////////// Log Function for Mengyao's Test ///////////////////

TestLoggingFunctionType modelTest::createTestLogFunction()
{
    return std::bind(&modelTest::testLogData,
                     this,
                     std::placeholders::_1,
                     std::placeholders::_2,
                     std::placeholders::_3);
//                     std::placeholders::_4,
//                     std::placeholders::_5);
//                     std::placeholders::_6);
}

