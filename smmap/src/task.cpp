#include <boost/filesystem.hpp>
#include <mutex>

#include "smmap/task.h"
#include "smmap/ros_params.hpp"
#include "smmap/ros_communication_helpers.hpp"
#include "smmap/diminishing_rigidity_model.h"

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
    , model_prediction_fn_( createModelPredictionFunction() )
    , model_suggested_grippers_traj_fn_( createModelSuggestedGrippersTrajFunction() )
    , get_model_utility_fn_( createGetModelUtilityFunction() )
    , update_model_utility_fn_( createUpdateModelUtilityFunction() )
    , gripper_collision_check_fn_( createGripperCollisionCheckFunction() )
    , task_desired_object_delta_fn_( createTaskDesiredObjectDeltaFunction() )
    , model_set_( update_model_utility_fn_ )
    , planner_( error_fn_, model_prediction_fn_, model_suggested_grippers_traj_fn_, get_model_utility_fn_, vis_ )

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

    while ( robot_.ok() )
    {
        const auto& current_world_state = world_feedback.back();

        std::pair< Eigen::VectorXd, Eigen::VectorXd > first_step_desired_motion;
        std::atomic_bool first_step_desired_motion_calculated( false );
        std::mutex first_step_desired_motion_mtx;

        // Update our function callbacks for the models
        TaskDesiredObjectDeltaFunctionType caching_task_desired_object_delta_fn =
                [&]( const WorldState& state )
        {
            if ( state.sim_time_ == current_world_state.sim_time_ )
            {
                if ( first_step_desired_motion_calculated.load() )
                {
                    return first_step_desired_motion;
                }
                else
                {
                    std::lock_guard< std::mutex > lock( first_step_desired_motion_mtx );
                    if ( first_step_desired_motion_calculated.load() )
                    {
                        return first_step_desired_motion;
                    }
                    else
                    {
                        first_step_desired_motion = task_specification_->calculateObjectDesiredDelta( state );
                        first_step_desired_motion_calculated.store( true );
                        return first_step_desired_motion;
                    }
                }
            }
            else
            {
                return task_specification_->calculateObjectDesiredDelta( state );
            }
        };
        DeformableModel::SetCallbackFunctions( gripper_collision_check_fn_,
                                               caching_task_desired_object_delta_fn );

        AllGrippersPoseTrajectory next_trajectory = planner_.getNextTrajectory(
                    current_world_state,
                    planning_horizion,
                    RobotInterface::DT,
                    RobotInterface::MAX_GRIPPER_VELOCITY,
                    task_specification_->getCollisionScalingFactor(),
                    task_specification_->getStretchingScalingThreshold() );

        assert( first_step_desired_motion_calculated.load() );

        // delete the starting pose which should match the current pose
//        next_trajectory.erase( next_trajectory.begin() );
        ROS_INFO_NAMED( "task", "Sending 'best' trajectory" );
        world_feedback = robot_.sendGripperTrajectory( next_trajectory );

        // TODO: this is already being calculated the the planner
        ROS_INFO_NAMED( "task", "Updating models" );
        model_set_.updateModels( world_feedback, first_step_desired_motion.second );

        // Log stuff if so desired
        {
            // TODO: only works with 1 model
            LOG_COND( loggers.at( "time" ), logging_enabled_,
                      world_feedback.back().sim_time_ );

            LOG_COND( loggers.at( "error"), logging_enabled_,
                      task_specification_->calculateError( world_feedback.back().object_configuration_ ) );

            LOG_COND( loggers.at( "utility"), logging_enabled_,
                      model_set_.getModelUtility()[0] );
        }

        if ( task_specification_->maxTime() < world_feedback.back().sim_time_ )
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
    DeformableModel::SetCallbackFunctions( gripper_collision_check_fn_,
                                           task_desired_object_delta_fn_ );

    DiminishingRigidityModel::SetInitialObjectConfiguration( GetObjectInitialConfiguration( nh_) );

    // Create some models and add them to the model set
    double translational_deformability, rotational_deformability;
    if ( ph_.getParam( "translational_deformability", translational_deformability ) &&
             ph_.getParam( "rotational_deformability", rotational_deformability ) )
    {
        ROS_INFO_STREAM_NAMED( "task", "Overriding deformability values to "
                               << translational_deformability << " "
                               << rotational_deformability );

        model_set_.addModel( std::make_shared< DiminishingRigidityModel >(
                                 DiminishingRigidityModel(
                                     translational_deformability,
                                     rotational_deformability ) ) );
    }
    else if ( GetUseMultiModel( ph_ ) )
    {
        // TODO: replace this maic number
        #warning "Magic number here needs to be moved"
        const size_t num_models_per_parameter = 20;

        ROS_INFO_STREAM_NAMED( "task", "Creating " << num_models_per_parameter
                               << " models per parameter " );

        const double deform_step = 0.5;
        double deform_min = std::max( 0., task_specification_->getDeformability() - 5 );
        // This round is here to force the values to 10-20 for cloth, 5-15 for rope
        deform_min = 5.0 * std::round( deform_min / 5.0 );
        const double deform_max = deform_min + (double)num_models_per_parameter * deform_step;

        for ( double trans_deform = deform_min; trans_deform < deform_max; trans_deform += deform_step )
        {
            for ( double rot_deform = deform_min; rot_deform < deform_max; rot_deform += deform_step )
            {
                model_set_.addModel( std::make_shared< DiminishingRigidityModel >(
                                         DiminishingRigidityModel(
                                             trans_deform,
                                             rot_deform ) ) );
            }
        }
    }
    else
    {
        ROS_INFO_STREAM_NAMED( "task", "Using default deformability value of "
                               << task_specification_->getDeformability() );

        model_set_.addModel( std::make_shared< DiminishingRigidityModel >(
                                 DiminishingRigidityModel(
                                     task_specification_->getDeformability() ) ) );
    }
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
                            "utility",
                            Log::Log( log_folder + "utility.txt", false ) ) ) ;
/*
//        loggers.insert( std::make_pair< std::string, Log::Log > (
//                            "model_chosen",
//                            Log::Log( log_folder + "model_chosen.txt", false ) ) ) ;

//        loggers.insert( std::make_pair< std::string, Log::Log >(
//                            "object_current_configuration",
//                            Log::Log( log_folder + "object_current_configuration.txt", false ) ) );

//        loggers.insert( std::make_pair< std::string, Log::Log > (
//                            "suggested_grippers_delta",
//                            Log::Log( log_folder + "suggested_grippers_delta.txt", false ) ) ) ;

//        loggers.insert( std::make_pair< std::string, Log::Log > (
//                            "object_predicted_configuration",
//                            Log::Log( log_folder + "object_predicted_configuration.txt", false ) ) ) ;
*/
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

ModelPredictionFunctionType Task::createModelPredictionFunction()
{
    return std::bind( &ModelSet::getPredictions,
                      &model_set_,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4 );
}

ModelSuggestedGrippersTrajFunctionType Task::createModelSuggestedGrippersTrajFunction()
{
    return std::bind( &ModelSet::getSuggestedGrippersTrajectories,
                      &model_set_,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4,
                      std::placeholders::_5,
                      std::placeholders::_6 );
}

GetModelUtilityFunctionType Task::createGetModelUtilityFunction()
{
    return std::bind( &ModelSet::getModelUtility,
                      &model_set_ );
}

UpdateModelUtilityFunctionType Task::createUpdateModelUtilityFunction()
{
    return std::bind( &Planner::UpdateUtility,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4 );
}

GripperCollisionCheckFunctionType Task::createGripperCollisionCheckFunction()
{
    return std::bind( &RobotInterface::checkGripperCollision,
                      &robot_,
                      std::placeholders::_1 );
}

TaskDesiredObjectDeltaFunctionType Task::createTaskDesiredObjectDeltaFunction()
{
    return std::bind( &TaskSpecification::calculateObjectDesiredDelta,
                      task_specification_,
                      std::placeholders::_1 );
}
