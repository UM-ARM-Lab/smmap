#include "smmap/planner.h"
#include "smmap/ros_params.hpp"
#include "smmap/trajectory.hpp"

#include <algorithm>
#include <assert.h>
#include <chrono>

#include <actionlib/client/simple_action_client.h>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <ros/callback_queue.h>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/pretty_print.hpp>
//#include <sensor_msgs/image_encodings.h>

using namespace smmap;
using namespace EigenHelpersConversions;

// TODO: move this from here, this is terrible
const Eigen::IOFormat Planner::eigen_io_one_line_( Eigen::FullPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", ""  );

Planner::Planner( ros::NodeHandle& nh )
    : visualize_gripper_translation_( false )
    , nh_( nh )
    , ph_( "~" )
    , it_( nh )
{
    // Initialize our task object with whatever it needs
    initializeTask();

    // Get the data we need to create our model set
    getGrippersData();

    double deformability;
    if ( ph_.getParam( "deformability", deformability ) )
    {
        model_set_ = std::unique_ptr< ModelSet >(
                new ModelSet( grippers_data_, getObjectInitialConfiguration(), *task_, deformability ) );
    }
    else
    {
        model_set_ = std::unique_ptr< ModelSet >(
                new ModelSet( grippers_data_, getObjectInitialConfiguration(), *task_ ) );
    }


    // Publish visualization request markers
    visualization_marker_pub_ =
            nh_.advertise< visualization_msgs::Marker >( GetVisualizationMarkerTopic( nh_ ), 10 );

    visualization_marker_array_pub_ =
            nh_.advertise< visualization_msgs::MarkerArray >( GetVisualizationMarkerArrayTopic( nh_ ), 10 );


    // Publish a our confidence values
    confidence_pub_ = nh_.advertise< smmap_msgs::ConfidenceStamped >( GetConfidenceTopic( nh_ ), 10 );
    confidence_image_pub_ = it_.advertise( GetConfidenceImageTopic( nh_ ), 10 );


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

        loggers.insert( std::make_pair< std::string, Log::Log >(
                            "object_current_configuration",
                            Log::Log( log_folder + "object_current_configuration.txt" ) ) );

        loggers.insert( std::make_pair< std::string, Log::Log >(
                            "object_desired_configuration",
                            Log::Log( log_folder + "object_desired_configuration.txt" ) ) );

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "object_predicted_configuration",
                            Log::Log( log_folder + "object_predicted_configuration.txt" ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "error",
                            Log::Log( log_folder + "error.txt" ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "time",
                            Log::Log( log_folder + "time.txt" ) ) ) ;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main function that makes things happen
////////////////////////////////////////////////////////////////////////////////

void Planner::run(const size_t num_traj_cmds_per_loop , const double dt )
{
    // TODO: remove this hardcoded spin rate
    boost::thread spin_thread( boost::bind( &Planner::spin, 1000 ) );

    actionlib::SimpleActionClient< smmap_msgs::CmdGrippersTrajectoryAction >
            cmd_grippers_traj_client( nh_, GetCommandGripperTrajTopic( nh_ ), false );
    ROS_INFO_NAMED( "planner" , "Waiting for the robot gripper action server to be avaiable" );
    cmd_grippers_traj_client.waitForServer();

    ROS_INFO_NAMED( "planner", "Getting the planner ready to go" );
    smmap_msgs::CmdGrippersTrajectoryGoal cmd_grippers_traj_goal = noOpTrajectoryGoal();

    // Run the planner at whatever rate we've been given
    ROS_INFO_NAMED( "planner" , "Running our planner" );
    while ( ros::ok() )
    {
        // get the best trajectory given the current data
        const std::vector< AllGrippersSinglePose > best_grippers_traj = replan( num_traj_cmds_per_loop, dt );

        // convert the trajectory into a ROS message
//        cmd_traj_req.trajectories.resize( best_grippers_traj.size() );
//        for ( size_t gripper_ind = 0; gripper_ind < cmd_traj_req.trajectories.size(); gripper_ind++ )
//        {
//            cmd_traj_req.trajectories[gripper_ind].pose =
//                    VectorAffine3dToVectorGeometryPose( best_grippers_traj[gripper_ind] );
//        }

        ROS_INFO_NAMED( "planner" , "Sending 'best' trajectory" );
//        cmd_gripper_traj_client_.call( cmd_traj_req, cmd_traj_res );
    }

    ROS_INFO_NAMED( "planner" , "Terminating" );
    spin_thread.join();
}

////////////////////////////////////////////////////////////////////////////////
// Internal helpers for the run() function
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Planner::replan
 * @param num_traj_cmds_per_loop
 * @return
 */
std::vector< AllGrippersSinglePose > Planner::replan( size_t num_traj_cmds_per_loop, double dt )
{
    std::vector< WorldFeedback > world_feedback;
    updateModels( world_feedback );

    // here we find the desired configuration of the object given the current config
    ObjectPointSet object_desired_config = task_->findObjectDesiredConfiguration( world_feedback.back().object_configuration_ );

    // Querry each model for it's best trajectory
    std::vector< std::pair< std::vector< AllGrippersSinglePose >, double > > suggested_trajectories =
            model_set_->getDesiredGrippersTrajectories(
                world_feedback.back(),
                object_desired_config,
                MAX_GRIPPER_VELOCITY * dt,
                num_traj_cmds_per_loop );

    size_t min_weighted_cost_ind = 0;
    double min_weighted_cost = std::numeric_limits< double >::infinity();

    ROS_INFO_NAMED( "planner" , "Finding 'best' trajectory" );
    for ( size_t traj_ind = 0; traj_ind < suggested_trajectories.size(); traj_ind++ )
    {
        double weighted_cost = suggested_trajectories[traj_ind].second; //*
                //( suggested_trajectories[traj_ind].first.back() - object_desired_config ).norm();
        if ( weighted_cost < min_weighted_cost )
        {
            min_weighted_cost = weighted_cost;
            min_weighted_cost_ind = traj_ind;
        }
    }

    // Send the desired "best" gripper translation to the visualizer to plot
    if ( visualize_gripper_translation_ )
    {
        for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
        {
            std_msgs::ColorRGBA color;
            color.r = 1;
            color.g = 1;
            color.b = 1;
            color.a = 1;
            visualizeTranslation( grippers_data_[gripper_ind].name,
                                  world_feedback.back().all_grippers_single_pose_[gripper_ind],
                                  suggested_trajectories[min_weighted_cost_ind].first.back()[gripper_ind],
                                  color );
        }
    }

    // TODO: deal with multiple predictions, which one is the best?
//    VectorObjectTrajectory model_predictions = model_set_->makePredictions( world_feedback.back(), suggested_trajectories[min_weighted_cost_ind].first );

//    task_->visualizePredictions( model_predictions, min_weighted_cost_ind );

//    LOG_COND( loggers.at( "object_predicted_configuration" ) , logging_enabled_,
//              (model_predictions[min_weighted_cost_ind].back()).format( eigen_io_one_line_ ) );

    return suggested_trajectories[min_weighted_cost_ind].first;
}

smmap_msgs::CmdGrippersTrajectoryGoal Planner::noOpTrajectoryGoal()
{

}

/**
 * @brief Planner::updateModels
 * @param object_trajectory
 * @param grippers_trajectory
 */
void Planner::updateModels( const std::vector<WorldFeedback>& feedback )
{
    ROS_INFO_NAMED( "planner" , "Updating models" );

    if ( feedback.size() >= 2 )
    {
        model_set_->updateModels( feedback );

        ROS_INFO_NAMED( "planner", "Evaluating confidence" );
        smmap_msgs::ConfidenceStamped double_msg;
        double_msg.confidence = model_set_->getModelConfidence();
        double_msg.header.stamp = ros::Time::now();
        confidence_pub_.publish( double_msg );

/*        cv::Mat image( 1, model_confidence.size(), CV_8UC3 );

        const double min_conf = *std::min_element( double_msg.confidence.begin(), double_msg.confidence.end() );
        const double max_conf = *std::max_element( double_msg.confidence.begin(), double_msg.confidence.end() );

        for ( size_t ind = 0; ind < double_msg.confidence.size(); ind++ )
        {
            image.at< cv::Vec3b >( 0, ind )[0] = (double_msg.confidence[ind] - min_conf) / ( max_conf - min_conf) * 255.0;
            image.at< cv::Vec3b >( 0, ind )[1] = 0;
            image.at< cv::Vec3b >( 0, ind )[2] = 0;
        }

        sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, image).toImageMsg();
        img_msg->header.stamp = ros::Time::now();
        confidence_image_pub_.publish( img_msg );
        */
    }
}


////////////////////////////////////////////////////////////////////////////////
// Task specific functionality
////////////////////////////////////////////////////////////////////////////////

void Planner::initializeTask()
{
    TaskType task_type = GetTaskType( nh_ );
    DeformableType deformable_type = GetDeformableType( nh_ );

    if ( deformable_type == DeformableType::ROPE && task_type == TaskType::COVERAGE )
    {
        task_.reset( new RopeCoverage( nh_ ) );
    }
    else if ( deformable_type == DeformableType::CLOTH && task_type == TaskType::COLAB_FOLDING )
    {
        task_.reset( new ClothColabFolding( nh_ ) );
    }
    else if ( deformable_type == DeformableType::CLOTH && task_type == TaskType::COVERAGE )
    {
        task_.reset( new ClothTableCoverage( nh_ ) );
    }
    else
    {
        assert( false && "THIS PAIR OF DEFORMALBE AND TASK IS NOT YET IMPLEMENTED" );
    }

    // TODO: the rest
}

void Planner::visualizeObjectDelta( const std::string& marker_name,
                                    const ObjectPointSet& current,
                                    const ObjectPointSet& desired )
{
    visualization_msgs::Marker marker;
    std_msgs::ColorRGBA color;

    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points.reserve( (size_t)current.cols() * 2 );
    marker.colors.reserve( (size_t)current.cols() * 2 );
    for ( long col = 0; col < current.cols(); col++ )
    {
        color.r = 0;//(1.0 + std::cos( 2*M_PI*(double)col/15.0 )) / 3;
        color.g = 1;//(1.0 + std::cos( 2*M_PI*(double)(col+5)/15.0 )) / 3;
        color.b = 0;//(1.0 + std::cos( 2*M_PI*double(col+10)/15.0 )) / 3;
        color.a = 1;

        marker.points.push_back( EigenVector3dToGeometryPoint( current.block< 3, 1 >( 0, col ) ) );
        marker.points.push_back( EigenVector3dToGeometryPoint( desired.block< 3, 1 >( 0, col ) ) );
        marker.colors.push_back( color );
        marker.colors.push_back( color );
    }

    visualization_marker_pub_.publish( marker );
}

void Planner::visualizeTranslation( const std::string& marker_name,
                                    const geometry_msgs::Point& start,
                                    const geometry_msgs::Point& end,
                                    const std_msgs::ColorRGBA& color )
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points.push_back( start );
    marker.points.push_back( end );
    marker.colors.push_back( color );
    marker.colors.push_back( color );

    visualization_marker_pub_.publish( marker );
}

void Planner::visualizeTranslation( const std::string& marker_name,
                                    const Eigen::Vector3d& start,
                                    const Eigen::Vector3d& end,
                                    const std_msgs::ColorRGBA& color )
{
    visualizeTranslation( marker_name,
                          EigenVector3dToGeometryPoint( start ),
                          EigenVector3dToGeometryPoint( end ),
                          color );
}

void Planner::visualizeTranslation( const std::string& marker_name,
                                    const Eigen::Affine3d &start,
                                    const Eigen::Affine3d &end,
                                    const std_msgs::ColorRGBA& color )
{
    visualizeTranslation( marker_name,
                          start.translation(),
                          end.translation(),
                          color );
}

void Planner::visualizeLines( const std::string& marker_name,
                              const EigenHelpers::VectorVector3d& start,
                              const EigenHelpers::VectorVector3d& end,
                              const std_msgs::ColorRGBA& color )
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;

    for ( size_t ind = 0; ind < start.size(); ind++ )
    {
        marker.points.push_back( EigenVector3dToGeometryPoint( start[ind] ) );
        marker.points.push_back( EigenVector3dToGeometryPoint( end[ind] ) );
        marker.colors.push_back( color );
        marker.colors.push_back( color );
    }

    visualization_marker_pub_.publish( marker );
}

////////////////////////////////////////////////////////////////////////////////
// ROS Callbacks
////////////////////////////////////////////////////////////////////////////////

/*
void Planner::simulatorFbkCallback( const smmap_msgs::SimulatorFeedback& fbk )
{
    // TODO: confirm that this locking is correct
    boost::recursive_mutex::scoped_lock lock( input_mtx_ );

    // TODO: if this data arrived out of order, do something smart
    object_trajectory_.push_back( VectorGeometryPointToEigenMatrix3Xd( fbk.object_configuration ) );

    for ( size_t gripper_ind = 0; gripper_ind < fbk.gripper_names.size(); gripper_ind++ )
    {
        grippers_trajectory_[gripper_ind].push_back(
                GeometryPoseToEigenAffine3d( fbk.gripper_poses[gripper_ind] ) );
        grippers_data_[gripper_ind].pose = grippers_trajectory_[gripper_ind].back();

        // Collect the collision data
        grippers_data_[gripper_ind].nearest_point_on_gripper =
                GeometryPointToEigenVector3d( fbk.gripper_nearest_point_to_obstacle[gripper_ind] );
        grippers_data_[gripper_ind].obstacle_surface_normal =
                GeometryVector3ToEigenVector3d( fbk.obstacle_surface_normal[gripper_ind] );

        grippers_data_[gripper_ind].distance_to_obstacle = fbk.gripper_distance_to_obstacle[gripper_ind];
    }

    sim_time_ = fbk.sim_time;
}
*/

////////////////////////////////////////////////////////////////////////////////
// ROS Objects and Helpers
////////////////////////////////////////////////////////////////////////////////

void Planner::spin( double loop_rate )
{
    ROS_INFO_NAMED( "planner" , "Starting feedback spinner" );
    while ( ros::ok() )
    {
        ros::getGlobalCallbackQueue()->callAvailable( ros::WallDuration( loop_rate ) );
    }
}

void Planner::getGrippersData()
{
    ROS_INFO_NAMED( "planner" , "Getting grippers data" );

    // Service client to get the names of each gripper
    ros::ServiceClient gripper_names_client =
        nh_.serviceClient< smmap_msgs::GetGripperNames >( GetGripperNamesTopic( nh_ ) );
    gripper_names_client.waitForExistence();

    smmap_msgs::GetGripperNames names_srv_data;
    if ( !gripper_names_client.call( names_srv_data ) )
    {
        ROS_FATAL_NAMED( "planner", "Unabled to retrieve gripper names." );
    }
    std::vector< std::string > gripper_names = names_srv_data.response.names;

    // Service client to get the attached nodes and transform for each gripper
    ros::ServiceClient gripper_node_indices_client =
        nh_.serviceClient< smmap_msgs::GetGripperAttachedNodeIndices >( GetGripperAttachedNodeIndicesTopic( nh_ ) );
    gripper_node_indices_client.waitForExistence();

    grippers_data_.reserve( gripper_names.size() );
    for ( size_t gripper_ind = 0; gripper_ind < gripper_names.size(); gripper_ind++ )
    {
        smmap_msgs::GetGripperAttachedNodeIndices node_indices_srv_data;
        node_indices_srv_data.request.name = gripper_names[gripper_ind];

        if ( gripper_node_indices_client.call( node_indices_srv_data ) )
        {
            grippers_data_.push_back( GripperData( gripper_names[gripper_ind],
                                      VectorAnytypeToVectorLong( node_indices_srv_data.response.indices ) ) );
        }
        else
        {
            ROS_ERROR_STREAM_NAMED( "planner", "Unable to retrieve node indices for gripper: " << gripper_names[gripper_ind] );
        }
    }
}

ObjectPointSet Planner::getObjectInitialConfiguration()
{
    ROS_INFO_NAMED( "planner" , "Getting object initial configuration" );

    // Get the initial configuration of the object
    ros::ServiceClient object_initial_configuration_client =
        nh_.serviceClient< smmap_msgs::GetPointSet >( GetObjectInitialConfigurationTopic( nh_ ) );

    object_initial_configuration_client.waitForExistence();

    smmap_msgs::GetPointSet srv_data;
    object_initial_configuration_client.call( srv_data );

    ROS_INFO_NAMED( "planner" , "Number of points on object: %zu", srv_data.response.points.size() );

    return VectorGeometryPointToEigenMatrix3Xd( srv_data.response.points );
}

void Planner::getObjectPlanningStartConfiguration()
{
    ROS_INFO_NAMED( "planner" , "Getting object planning start configuration" );

    // Get the initial configuration of the object
    ros::ServiceClient object_planning_start_configuration_client =
        nh_.serviceClient< smmap_msgs::GetPointSet >( GetObjectCurrentConfigurationTopic( nh_ ) );

    object_planning_start_configuration_client.waitForExistence();

    smmap_msgs::GetPointSet srv_data;
    object_planning_start_configuration_client.call( srv_data );
}
