#include <mutex>
#include <boost/filesystem.hpp>
#include <smmap_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/diminishing_rigidity_model.h"
#include "smmap/adaptive_jacobian_model.h"
#include "smmap/least_squares_jacobian_model.h"

#include "smmap/task.h"

using namespace smmap;
using namespace EigenHelpersConversions;

Task::Task( RobotInterface& robot,
            Visualizer& vis,
            TaskSpecification::Ptr task_specification )
    : nh_()
    , ph_( "~" )
    , robot_( robot )
    , vis_( vis )
    , task_specification_( task_specification )
    , error_fn_( createErrorFunction() )
    , gripper_collision_check_fn_( createGripperCollisionCheckFunction() )
    , task_object_delta_projection_fn_( createTaskObjectDeltaProjectionFunction() )
    , execute_trajectory_fn_( createExecuteGripperTrajectoryFunction() )
    , logging_fn_( createLoggingFunction() )
    , planner_( error_fn_, execute_trajectory_fn_, logging_fn_, vis_, GetRobotControlPeriod( nh_ ) )
{
    initializeModelSet();
    initializeLogging();
}


void Task::execute()
{
    const int planning_horizion = GetPlanningHorizon( ph_ );

    // Run the planner at whatever rate we've been given
    ROS_INFO_STREAM_NAMED( "task", "Running our planner with a horizion of " << planning_horizion );
    std::vector< WorldState > world_feedback = robot_.start();
    const double start_time = world_feedback.back().sim_time_;

    while ( robot_.ok() )
    {
        const WorldState current_world_state = world_feedback.back();

        std::pair< Eigen::VectorXd, Eigen::VectorXd > first_step_desired_motion;
        std::pair< Eigen::VectorXd, Eigen::VectorXd > first_step_error_correction;
        std::pair< Eigen::VectorXd, Eigen::VectorXd > first_step_stretching_correction;
        std::atomic_bool first_step_calculated( false );
        std::mutex first_step_mtx;

        // Update our function callbacks for the models
        TaskDesiredObjectDeltaFunctionType caching_task_desired_object_delta_fn =
                [&]( const WorldState& state )
        {
            if ( state.sim_time_ == current_world_state.sim_time_ )
            {
                if ( first_step_calculated.load() )
                {
                    return first_step_desired_motion;
                }
                else
                {
                    std::lock_guard< std::mutex > lock( first_step_mtx );
                    if ( first_step_calculated.load() )
                    {
                        return first_step_desired_motion;
                    }
                    else
                    {
                        first_step_error_correction =
                                task_specification_->calculateObjectErrorCorrectionDelta( state );

                        first_step_stretching_correction =
                                task_specification_->calculateStretchingCorrectionDelta( state );

                        first_step_desired_motion =
                                task_specification_->combineErrorCorrectionAndStretchingCorrection(
                                    first_step_error_correction, first_step_stretching_correction );

                        first_step_calculated.store( true );
                        return first_step_desired_motion;
                    }
                }
            }
            else
            {
                return task_specification_->calculateObjectErrorCorrectionDelta( state );
            }
        };

        DeformableModel::SetCallbackFunctions( gripper_collision_check_fn_,
                                               caching_task_desired_object_delta_fn,
                                               task_object_delta_projection_fn_ );

        world_feedback = planner_.sendNextTrajectory(
                    current_world_state,
                    caching_task_desired_object_delta_fn,
                    planning_horizion,
                    RobotInterface::MAX_GRIPPER_VELOCITY,
                    task_specification_->getCollisionScalingFactor() );

        if ( task_specification_->maxTime() < world_feedback.back().sim_time_ - start_time )
        {
            robot_.shutdown();
        }
    }
}

////////////////////////////////////////////////////////////////////
// Internal initialization helpers
////////////////////////////////////////////////////////////////////

void Task::initializeModelSet()
{
    // Initialze each model type with the shared data
    DeformableModel::SetGrippersData( robot_.getGrippersData() );
    // TODO: fix this interface so that I'm not passing a null ptr here
    DeformableModel::SetCallbackFunctions( gripper_collision_check_fn_,
                                           TaskDesiredObjectDeltaFunctionType( nullptr ),
                                           task_object_delta_projection_fn_ );
    DiminishingRigidityModel::SetInitialObjectConfiguration( GetObjectInitialConfiguration( nh_) );

    // Create some models and add them to the model set
    double translational_deformability, rotational_deformability;
    if ( ph_.getParam( "translational_deformability", translational_deformability ) &&
             ph_.getParam( "rotational_deformability", rotational_deformability ) )
    {
        ROS_INFO_STREAM_NAMED( "task", "Overriding deformability values to "
                               << translational_deformability << " "
                               << rotational_deformability );

        planner_.addModel( std::make_shared< DiminishingRigidityModel >(
                                 DiminishingRigidityModel(
                                     translational_deformability,
                                     rotational_deformability ) ) );
    }
    else if ( GetUseMultiModel( ph_ ) )
    {
        ////////////////////////////////////////////////////////////////////////
        // Diminishing rigidity models
        ////////////////////////////////////////////////////////////////////////

        const double deform_min = 0.0;
        const double deform_max = 25.0;
        const double deform_step = 2.0;

        for ( double trans_deform = deform_min; trans_deform < deform_max; trans_deform += deform_step )
        {
            double rot_deform = trans_deform;
//            for ( double rot_deform = deform_min; rot_deform < deform_max; rot_deform += deform_step )
//            {
                planner_.addModel( std::make_shared< DiminishingRigidityModel >(
                                         DiminishingRigidityModel(
                                             trans_deform,
                                             rot_deform ) ) );
//            }
        }
        ROS_INFO_STREAM_NAMED( "task", "Num diminishing rigidity models: "
                               << std::floor( ( deform_max - deform_min ) / deform_step ) );

        ////////////////////////////////////////////////////////////////////////
        // Adaptive jacobian models
        ////////////////////////////////////////////////////////////////////////

        const double learning_rate_min = 1e-10;
        const double learning_rate_max = 1.1e0;
        const double learning_rate_step = 10.0;
        for ( double learning_rate = learning_rate_min; learning_rate < learning_rate_max; learning_rate *= learning_rate_step )
        {
                planner_.addModel( std::make_shared< AdaptiveJacobianModel >(
                                         AdaptiveJacobianModel(
                                             DiminishingRigidityModel(
                                                 task_specification_->getDeformability() )
                                             .getGrippersToObjectJacobian(
                                                 robot_.getGrippersPose(),
                                                 GetObjectInitialConfiguration( nh_) ),
                                             learning_rate ) ) );
        }
        ROS_INFO_STREAM_NAMED( "task", "Num adaptive Jacobian models: "
                               << std::floor( std::log( learning_rate_max / learning_rate_min ) / std::log( learning_rate_step ) ) );
    }
    else if ( GetUseAdaptiveModel( ph_ ) )
    {
                planner_.addModel( std::make_shared< AdaptiveJacobianModel >(
                                         AdaptiveJacobianModel(
                                             DiminishingRigidityModel(
                                                 task_specification_->getDeformability() )
                                             .getGrippersToObjectJacobian(
                                                 robot_.getGrippersPose(),
                                                 GetObjectInitialConfiguration( nh_) ),
                                             GetAdaptiveModelLearningRate( ph_ ) ) ) );
    }
    else
    {
        ROS_INFO_STREAM_NAMED( "task", "Using default deformability value of "
                               << task_specification_->getDeformability() );

        planner_.addModel( std::make_shared< DiminishingRigidityModel >(
                                 DiminishingRigidityModel(
                                     task_specification_->getDeformability() ) ) );

//        model_set_.addModel( std::make_shared< LeastSquaresJacobianModel >(
//                                 LeastSquaresJacobianModel(
//                                     DiminishingRigidityModel(
//                                         task_specification_->getDeformability() )
//                                     .getGrippersToObjectJacobian(
//                                         robot_.getGrippersPose(),
//                                         GetObjectInitialConfiguration( nh_) ),
//                                     2 ) ) );

    }

    planner_.createBandits();
}

void Task::initializeLogging()
{
    // Enable logging if it is requested
    logging_enabled_ = GetLoggingEnabled( nh_ );

    if ( logging_enabled_ )
    {
        std::string log_folder = GetLogFolder( nh_ );

        // If it hasn't been opened, assume that it is because the
        // directory doesn't exist.
        boost::filesystem::path p( log_folder );
        if ( !boost::filesystem::is_directory( p ) )
        {
            std::cerr << "\x1b[33;1m" << log_folder << " does not exist! Creating ... ";

            // NOTE: create_directories should be able to return true in this case
            // however due to a bug related to a trailing '/' this is not currently
            // the case in my version of boost
            // https://svn.boost.org/trac/boost/ticket/7258
            boost::filesystem::create_directories( p );
            if ( boost::filesystem::is_directory( p ) )
//            if ( boost::filesystem::create_directories( p ) )
            {
                std::cerr << "Succeeded!\x1b[37m\n";
            }
            else
            {
                std::cerr << "\x1b[31;1mFailed!\x1b[37m\n";
            }
        }

        ROS_INFO_STREAM_NAMED( "planner", "Logging to " << log_folder );

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "time",
                            Log::Log( log_folder + "time.txt", false ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "error",
                            Log::Log( log_folder + "error.txt", false ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "utility_mean",
                            Log::Log( log_folder + "utility_mean.txt", false ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "utility_covariance",
                            Log::Log( log_folder + "utility_covariance.txt", false ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "model_chosen",
                            Log::Log( log_folder + "model_chosen.txt", false ) ) ) ;
    }
}

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

void Task::logData(
        const WorldState& current_world_state,
        const Eigen::VectorXd& model_utility_mean,
        const Eigen::MatrixXd& model_utility_covariance,
        const ssize_t model_used )
{
    if ( logging_enabled_ )
    {
        const Eigen::IOFormat single_line(
                    Eigen::StreamPrecision,
                    Eigen::DontAlignCols,
                    " ", " ", "", "" );

        LOG( loggers.at( "time" ),
             current_world_state.sim_time_ );

        LOG( loggers.at( "error" ),
             task_specification_->calculateError( current_world_state.object_configuration_ ) );

        LOG( loggers.at( "utility_mean" ),
             model_utility_mean.format( single_line ) );

        LOG( loggers.at( "utility_covariance" ),
             model_utility_covariance.format( single_line ) );

        LOG( loggers.at( "model_chosen" ),
             model_used );
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////

ErrorFunctionType Task::createErrorFunction()
{
    return std::bind( &TaskSpecification::calculateError,
                      task_specification_,
                      std::placeholders::_1 );
}

GripperCollisionCheckFunctionType Task::createGripperCollisionCheckFunction()
{
    return std::bind( &RobotInterface::checkGripperCollision,
                      &robot_,
                      std::placeholders::_1 );
}

TaskExecuteGripperTrajectoryFunctionType Task::createExecuteGripperTrajectoryFunction()
{
    return std::bind( &RobotInterface::sendGripperTrajectory,
                      &robot_,
                      std::placeholders::_1 );
}

LoggingFunctionType Task::createLoggingFunction()
{
    return std::bind( &Task::logData,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4 );
}


TaskObjectDeltaProjectionFunctionType Task::createTaskObjectDeltaProjectionFunction()
{
    return std::bind( &TaskSpecification::projectObjectDelta,
                      task_specification_,
                      std::placeholders::_1,
                      std::placeholders::_2 );
}
