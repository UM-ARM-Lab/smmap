#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <ros/callback_queue.h>

#include "smmap/task.h"
#include "smmap/ros_params.hpp"
#include "smmap/ros_communication_helpers.hpp"
#include "smmap/diminishing_rigidity_model.h"

using namespace smmap;
using namespace EigenHelpersConversions;

Task::Task( ros::NodeHandle& nh )
    : nh_( nh )
    , ph_( "~" )
    , gripper_collision_checker_( nh_ )
    , vis_( nh_ )
    , cmd_grippers_traj_client_( nh_, GetCommandGripperTrajTopic( nh_ ), false )
    , task_specification_( TaskSpecification::MakeTaskSpecification( nh_ ) )
    , grippers_data_( GetGrippersData( nh_) )
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
    // TODO: remove this hardcoded spin rate
    // TODO: make this std, not boost
    boost::thread spin_thread( boost::bind( &Task::spin, 1000 ) );

    // TODO: make a "RobotBridge" class (or similar)
    ROS_INFO_NAMED( "task", "Waiting for the robot gripper action server to be available" );
    cmd_grippers_traj_client_.waitForServer();

    const size_t planning_horizion = GetPlanningHorizon( ph_ );

    // Run the planner at whatever rate we've been given
    ROS_INFO_STREAM_NAMED( "task", "Running our planner with a horizion of " << planning_horizion );
    std::vector< WorldState > world_feedback;

    ROS_INFO_NAMED( "planner", "Kickstarting the planner with a no-op" );
    world_feedback = sendGripperTrajectory( noOpTrajectoryGoal( 2 ) );

    while ( ros::ok() )
    {
        // TODO: more magic numbers
        #warning "More magic numbers"
        // TODO: this is already being calculated the the planner
        auto task_desired_motion_weights = task_specification_->calculateObjectDesiredDelta( world_feedback.back() ).second;
        AllGrippersPoseTrajectory next_trajectory = planner_.getNextTrajectory(
                    world_feedback.back(),
                    planning_horizion,
                    0.01, // dt
                    0.05/20.0/0.01, // max velocity
                    task_specification_->getCollisionScalingFactor(),
                    // TODO: why is this throwing a conversion warning?
                    task_specification_->getStretchingScalingThreshold() );

        // delete the starting pose which should match the current pose
//        next_trajectory.erase( next_trajectory.begin() );
        ROS_INFO_NAMED( "task", "Sending 'best' trajectory" );
        world_feedback = sendGripperTrajectory( toRosGoal( next_trajectory ) );

        ROS_INFO_NAMED( "task", "Updating models" );
        model_set_.updateModels( world_feedback, task_desired_motion_weights );

        // TODO: only works with 1 model
        LOG_COND( loggers.at( "time" ), logging_enabled_,
                  world_feedback.back().sim_time_ );

        LOG_COND( loggers.at( "error"), logging_enabled_,
                  task_specification_->calculateError( world_feedback.back().object_configuration_ ) );

        LOG_COND( loggers.at( "utility"), logging_enabled_,
                  model_set_.getModelUtility()[0] );

        if ( task_specification_->maxTime() < world_feedback.back().sim_time_ )
        {
            ros::shutdown();
        }
    }

    ROS_INFO_NAMED( "task", "Terminating" );
    spin_thread.join();
}

////////////////////////////////////////////////////////////////////
// ROS objects and helpers
////////////////////////////////////////////////////////////////////

void Task::spin( double loop_rate )
{
    ROS_INFO_NAMED( "task" , "Starting feedback spinner" );
    while ( ros::ok() )
    {
        ros::getGlobalCallbackQueue()->callAvailable( ros::WallDuration( loop_rate ) );
    }
}

std::vector< WorldState > Task::sendGripperTrajectory(
        const smmap_msgs::CmdGrippersTrajectoryGoal& goal )
{
    std::vector< WorldState > feedback;

    cmd_grippers_traj_client_.sendGoalAndWait( goal );
    if ( cmd_grippers_traj_client_.getState() == actionlib::SimpleClientGoalState::SUCCEEDED )
    {
        feedback = ParseGripperActionResult( cmd_grippers_traj_client_.getResult() );
    }
    else
    {
        ROS_FATAL_NAMED( "planner", "Sending a goal to the robot failed" );
    }

    return feedback;
}

smmap_msgs::CmdGrippersTrajectoryGoal Task::noOpTrajectoryGoal( size_t num_no_op )
{
    smmap_msgs::CmdGrippersTrajectoryGoal goal;
    goal.gripper_names = GetGripperNames( grippers_data_ );

    smmap_msgs::VectorPose grippers_pose;
    grippers_pose.pose.resize( grippers_data_.size() );
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
    {
        ros::ServiceClient gripper_pose_client =
            nh_.serviceClient< smmap_msgs::GetGripperPose >( GetGripperPoseTopic( nh_ ) );
        gripper_pose_client.waitForExistence();

        smmap_msgs::GetGripperPose pose_srv_data;
        pose_srv_data.request.name = grippers_data_[gripper_ind].name;
        if ( !gripper_pose_client.call( pose_srv_data ) )
        {
            ROS_FATAL_STREAM_NAMED( "task", "Unabled to retrieve gripper pose: " << grippers_data_[gripper_ind].name );
        }

        grippers_pose.pose[gripper_ind] = pose_srv_data.response.pose;
    }

    goal.trajectory.resize( num_no_op, grippers_pose );

    return goal;
}

smmap_msgs::CmdGrippersTrajectoryGoal Task::toRosGoal(
        const AllGrippersPoseTrajectory& trajectory )
{
    smmap_msgs::CmdGrippersTrajectoryGoal goal;
    goal.gripper_names = GetGripperNames( grippers_data_ );

    goal.trajectory.resize( trajectory.size() );
    for ( size_t time_ind = 0; time_ind < trajectory.size(); time_ind++ )
    {
        goal.trajectory[time_ind].pose = VectorAffine3dToVectorGeometryPose( trajectory[time_ind] );
    }

    return goal;
}

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
    return std::bind( &Task::updateUtility,
                      this,
                      std::placeholders::_1,
                      std::placeholders::_2,
                      std::placeholders::_3,
                      std::placeholders::_4 );
}

GripperCollisionCheckFunctionType Task::createGripperCollisionCheckFunction()
{
    return std::bind( &GripperCollisionChecker::gripperCollisionCheck,
                      &gripper_collision_checker_,
                      std::placeholders::_1 );
}

TaskDesiredObjectDeltaFunctionType Task::createTaskDesiredObjectDeltaFunction()
{
    return std::bind( &TaskSpecification::calculateObjectDesiredDelta,
                      task_specification_,
                      std::placeholders::_1 );
}

double Task::updateUtility( const double old_utility,
                            const WorldState& world_state,
                            const ObjectPointSet& prediction,
                            const Eigen::VectorXd& weights ) const
{
    const double distance = distanceWeighted(
                world_state.object_configuration_,
                prediction,
                weights );
    // TODO: use dt here somewhere, plus the number of nodes, etc.
    const double new_utility = 1.0/(1.0 + std::sqrt( std::sqrt( distance ) ) );
    #warning "Another magic number here"
    return anneal( old_utility, new_utility, 0.1 );
}
