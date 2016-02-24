#include <arc_utilities/eigen_helpers_conversions.hpp>

#include <boost/filesystem.hpp>

#include "smmap/task.h"
 #include "smmap/ros_params.hpp"
#include "smmap/ros_communication_helpers.hpp"
#include "smmap/diminishing_rigidity_model.h"

using namespace smmap;
using namespace EigenHelpersConversions;

Task::Task( ros::NodeHandle& nh )
    : nh_( nh )
    , ph_( "~" )
    , vis_( nh )
    , task_specification_( TaskSpecification::MakeTaskSpecification( nh ) )
    , grippers_data_( GetGrippersData( nh_) )
    , error_fn_( createErrorFunction() )
    , model_prediction_fn_( createModelPredictionFunction() )
    , model_suggested_grippers_traj_fn_( createModelSuggestedGrippersTrajFunction() )
    , get_model_utility_fn_( createGetModelUtilityFunction() )
    , update_model_utility_fn_( createUpdateModelUtilityFunction() )
    , gripper_collision_check_fn_( createGripperCollisionCheckFunction() )
    , task_desired_object_delta_fn_( createTaskDesiredObjectDeltaFunction() )
    , model_set_( update_model_utility_fn_ )

{
    initializeModelSet();
    initializeLogging();
}

void Task::execute()
{}

////////////////////////////////////////////////////////////////////
// Internal initialization helpers
////////////////////////////////////////////////////////////////////

void Task::initializeModelSet()
{
    // Initialze each model type with the shared data
    DeformableModel::SetGrippersData( grippers_data_ );
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
        assert(false);
    }
    else
    {
        assert(false);
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

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "model_chosen",
                            Log::Log( log_folder + "model_chosen.txt", false ) ) ) ;

//        loggers.insert( std::make_pair< std::string, Log::Log >(
//                            "object_current_configuration",
//                            Log::Log( log_folder + "object_current_configuration.txt", false ) ) );

//        loggers.insert( std::make_pair< std::string, Log::Log > (
//                            "suggested_grippers_delta",
//                            Log::Log( log_folder + "suggested_grippers_delta.txt", false ) ) ) ;

//        loggers.insert( std::make_pair< std::string, Log::Log > (
//                            "object_predicted_configuration",
//                            Log::Log( log_folder + "object_predicted_configuration.txt", false ) ) ) ;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////

ErrorFunctionType Task::createErrorFunction()
{
}

ModelPredictionFunctionType Task::createModelPredictionFunction()
{
    return std::bind( &ModelSet::getPredictions, &model_set_,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4 );
}

ModelSuggestedGrippersTrajFunctionType Task::createModelSuggestedGrippersTrajFunction()
{
}

GetModelUtilityFunctionType Task::createGetModelUtilityFunction()
{
    return std::bind( &ModelSet::getModelUtility, &model_set_ );
}

UpdateModelUtilityFunctionType Task::createUpdateModelUtilityFunction()
{
}

GripperCollisionCheckFunctionType Task::createGripperCollisionCheckFunction()
{
}

TaskDesiredObjectDeltaFunctionType Task::createTaskDesiredObjectDeltaFunction()
{
}
