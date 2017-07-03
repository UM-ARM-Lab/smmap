#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/diminishing_rigidity_model.h"
#include "smmap/adaptive_jacobian_model.h"
#include "smmap/least_squares_jacobian_model.h"
#include "smmap/constraint_jacobian_model.h"

#include "smmap/task.h"

using namespace smmap;
using namespace EigenHelpersConversions;

Task::Task(RobotInterface& robot,
           Visualizer& vis,
           const TaskSpecification::Ptr& task_specification)
    : nh_()
    , ph_("~")
    , robot_(robot)
    , vis_(vis)
    , task_specification_(task_specification)
    , logging_fn_(createLoggingFunction())
    , planner_(robot_, vis_, task_specification_, logging_fn_)
{
    initializeLogging();
}

void Task::execute()
{
    // Run the planner at whatever rate we've been given
    WorldState world_feedback = robot_.start();
    const double start_time = world_feedback.sim_time_;

    initializeModelSet(world_feedback);

    while (robot_.ok())
    {
        const WorldState world_state = world_feedback;
        world_feedback = planner_.sendNextCommand(world_state);

        if (unlikely(world_feedback.sim_time_ - start_time >= task_specification_->maxTime()
                     || task_specification_->taskDone(world_feedback)))
        {
            ROS_INFO_NAMED("task", "------------------------------- End of Task -------------------------------------------");
            const double current_error = task_specification_->calculateError(world_state);
            ROS_INFO_STREAM_NAMED("task", "   Planner/Task sim time " << world_state.sim_time_ << "\t Error: " << current_error);

            vis_.deleteObjects(Planner::PROJECTED_GRIPPER_NS,            1, (int32_t)(4 * GetNumLookaheadSteps(ph_)) + 10);
            vis_.deleteObjects(Planner::PROJECTED_BAND_NS,               1, (int32_t)GetNumLookaheadSteps(ph_) + 10);
            vis_.deleteObjects(Planner::PROJECTED_POINT_PATH_NS,         1, 2);
            vis_.deleteObjects(Planner::PROJECTED_POINT_PATH_LINES_NS,   1, 2);

            if (world_feedback.sim_time_ - start_time >= task_specification_->maxTime())
            {
                ROS_INFO("Terminating task as time has run out");
            }
            if (task_specification_->taskDone(world_feedback))
            {
                ROS_INFO("Terminating task as the task has been completed");
            }
            robot_.shutdown();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Internal initialization helpers
////////////////////////////////////////////////////////////////////////////////

void Task::initializeModelSet(const WorldState& initial_world_state)
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
            return task_specification_->calculateDesiredDirection(world_state);
        };

        const DeformableModel::DeformableModelInputData input_data(task_desired_direction_fn, initial_world_state, robot_.dt_);
        for (double learning_rate = learning_rate_min; learning_rate < learning_rate_max; learning_rate *= learning_rate_step)
        {
                planner_.addModel(std::make_shared<AdaptiveJacobianModel>(
                                      DiminishingRigidityModel(task_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(input_data),
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
            return task_specification_->calculateDesiredDirection(world_state);
        };

        const DeformableModel::DeformableModelInputData input_data(task_desired_direction_fn, initial_world_state, robot_.dt_);
        planner_.addModel(std::make_shared<AdaptiveJacobianModel>(
                              DiminishingRigidityModel(task_specification_->defaultDeformability(), false).computeGrippersToDeformableObjectJacobian(input_data),
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
        ROS_INFO_STREAM_NAMED("task", "Using default deformability value of "
                               << task_specification_->defaultDeformability());

        planner_.addModel(std::make_shared<DiminishingRigidityModel>(
                              task_specification_->defaultDeformability(),
                              optimization_enabled));

//        planner_.addModel(std::make_shared<LeastSquaresJacobianModel>(
//                              DiminishingRigidityModel(task_specification_->defaultDeformability(), false).getGrippersToObjectJacobian(robot_.getGrippersPose(), GetObjectInitialConfiguration(nh_)),
//                              2));
    }

    planner_.createBandits();
}

void Task::initializeLogging()
{
    // Enable logging if it is requested
    logging_enabled_ = GetLoggingEnabled(nh_);

    if (logging_enabled_)
    {
        const std::string log_folder = GetLogFolder(nh_);

        ROS_INFO_STREAM_NAMED("planner", "Logging to " << log_folder);

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "time",
                            Log::Log(log_folder + "time.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "error",
                            Log::Log(log_folder + "error.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "utility_mean",
                            Log::Log(log_folder + "utility_mean.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "utility_covariance",
                            Log::Log(log_folder + "utility_covariance.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "model_chosen",
                            Log::Log(log_folder + "model_chosen.txt", false)));

        loggers_.insert(std::make_pair<std::string, Log::Log>(
                            "rewards_for_all_models",
                            Log::Log(log_folder + "rewards_for_all_models.txt", false)));

//        loggers.insert(std::make_pair<std::string, Log::Log>(
//                            "correlation_scale_factor",
//                            Log::Log(log_folder + "correlation_scale_factor.txt", false)));
    }
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

void Task::logData(
        const WorldState& current_world_state,
        const Eigen::VectorXd& model_utility_mean,
        const Eigen::MatrixXd& model_utility_covariance,
        const ssize_t model_used,
        const std::vector<double>& rewards_for_all_models)
{
    if (logging_enabled_)
    {
        const static Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
                    " ", " ", "", "");

        LOG(loggers_.at("time"),
             current_world_state.sim_time_);

        LOG(loggers_.at("error"),
             task_specification_->calculateError(current_world_state));

        LOG(loggers_.at("utility_mean"),
             model_utility_mean.format(single_line));

        LOG(loggers_.at("utility_covariance"),
             model_utility_covariance.format(single_line));

        LOG(loggers_.at("model_chosen"),
             model_used);

        LOG(loggers_.at("rewards_for_all_models"),
            PrettyPrint::PrettyPrint(rewards_for_all_models, false, " "));
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////

GripperCollisionCheckFunctionType Task::createGripperCollisionCheckFunction()
{
    return std::bind(&RobotInterface::checkGripperCollision,
                     &robot_,
                     std::placeholders::_1);
}

LoggingFunctionType Task::createLoggingFunction()
{
    return std::bind(&Task::logData,
                     this,
                     std::placeholders::_1,
                     std::placeholders::_2,
                     std::placeholders::_3,
                     std::placeholders::_4,
                     std::placeholders::_5);
}
