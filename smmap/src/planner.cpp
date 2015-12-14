#include "smmap/planner.h"
#include "smmap/ros_params.h"
#include "smmap/trajectory.h"

#include <algorithm>
#include <assert.h>
#include <chrono>

#include <boost/thread.hpp>
#include <ros/callback_queue.h>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/pretty_print.hpp>
//#include <sensor_msgs/image_encodings.h>

using namespace smmap;
using namespace EigenHelpersConversions;

// TODO: move this from here, this is terrible
const Eigen::IOFormat Planner::eigen_io_one_line_( Eigen::FullPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", ""  );

Planner::Planner(ros::NodeHandle& nh )
    : visualize_object_desired_config_( false )
    , visualize_object_predicted_config_( false )
    , visualize_gripper_translation_( false )
//    , visualize_correspondances_( false )
    , nh_( nh )
    , it_( nh_ )
    , sim_time_( 0 )
{
    // Subscribe to feedback channels
    simulator_fbk_sub_ = nh_.subscribe(
            GetSimulatorFeedbackTopic( nh_ ), 20, &Planner::simulatorFbkCallback, this );

    // Initialize our task object with whatever it needs
    initializeTask();

    // Get the data we need to create our model set
    getGrippersData();
    getObjectInitialConfiguration();

    model_set_ = std::unique_ptr< ModelSet >(
            new ModelSet( grippers_data_, object_initial_configuration_ ) );


    // Connect to the robot's gripper command service
    cmd_gripper_traj_client_ =
            nh_.serviceClient< smmap_msgs::CmdGrippersTrajectory >( GetCommandGripperTrajTopic( nh_ ), true );
    cmd_gripper_traj_client_.waitForExistence();


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

        loggers.insert( std::make_pair< std::string, Log::Log >(
                            "object_current_configuration",
                            Log::Log( log_folder + "object_current_configuration.txt" ) ) );

        loggers.insert( std::make_pair< std::string, Log::Log >(
                            "object_desired_configuration",
                            Log::Log( log_folder + "object_desired_configuration.txt" ) ) );

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "object_predicted_configuration",
                            Log::Log( log_folder + "object_predicted_configuration.txt" ) ) ) ;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main function that makes things happen
////////////////////////////////////////////////////////////////////////////////

void Planner::run( const size_t num_traj_cmds_per_loop )
{
    // TODO: remove this hardcoded spin rate
    boost::thread spin_thread( boost::bind( &Planner::spin, 1000 ) );
    // TODO: This is lame. There needs to be a better way.
    // Wait for ROS to finish making topic connections
    usleep( 500000 );

    // Initialize the trajectory command message
    ROS_INFO_NAMED( "planner" , "Initializing gripper command message" );
    smmap_msgs::CmdGrippersTrajectory::Request cmd_traj_req;
    smmap_msgs::CmdGrippersTrajectory::Response cmd_traj_res;

    // Get the initial best traj
    const AllGrippersTrajectory best_grippers_traj = replan( num_traj_cmds_per_loop );

    // fill the request structure
    cmd_traj_req.trajectories.resize( grippers_data_.size() );
    for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
    {
        cmd_traj_req.gripper_names.push_back( grippers_data_[gripper_ind].name );
        cmd_traj_req.trajectories[gripper_ind].pose = VectorAffine3dToVectorGeometryPose( best_grippers_traj[gripper_ind] );
    }
    cmd_gripper_traj_client_.call( cmd_traj_req, cmd_traj_res );

    // Run the planner at whatever rate we've been given
    ROS_INFO_NAMED( "planner" , "Running our planner" );
    while ( ros::ok() )
    {
        // TODO: remove this 'continue' crap
        // TODO: why is this a recursive lock again?
        // Wait for data from the simulator
        boost::recursive_mutex::scoped_lock lock ( input_mtx_ );
        if ( object_trajectory_.size() < num_traj_cmds_per_loop + 1 )
        {
            lock.unlock();
            usleep( 500 );
            continue;
        }

        // get the best trajectory given the current data
        const AllGrippersTrajectory best_grippers_traj = replan( num_traj_cmds_per_loop );
        lock.unlock();

        // convert the trajectory into a ROS message
        cmd_traj_req.trajectories.resize( best_grippers_traj.size() );
        for ( size_t gripper_ind = 0; gripper_ind < cmd_traj_req.trajectories.size(); gripper_ind++ )
        {
            cmd_traj_req.trajectories[gripper_ind].pose =
                    VectorAffine3dToVectorGeometryPose( best_grippers_traj[gripper_ind] );
        }

        ROS_INFO_NAMED( "planner" , "Sending 'best' trajectory" );
        cmd_gripper_traj_client_.call( cmd_traj_req, cmd_traj_res );
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
AllGrippersTrajectory Planner::replan( size_t num_traj_cmds_per_loop )
{
     boost::recursive_mutex::scoped_lock lock( input_mtx_ );

    // Update the models with whatever feedback we have
    std::pair< ObjectTrajectory, AllGrippersTrajectory > fbk = readSimulatorFeedbackBuffer();
    updateModels( fbk.first, fbk.second );
    ObjectPointSet object_current_config = fbk.first.back();

    lock.unlock();

    // here we find the desired configuration of the object given the current config
    ObjectPointSet object_desired_config = task_->findObjectDesiredConfiguration( object_current_config );

    // Send the desired config to the visualizer to plot
    if ( visualize_object_desired_config_ )
    {
        std_msgs::ColorRGBA object_desired_config_color;
        object_desired_config_color.r = 0;
        object_desired_config_color.g = 0;
        object_desired_config_color.b = 1;
        object_desired_config_color.a = 1;
        // TODO: make this work for non-ropes
        visualizeRopeObject( "object_desired_config", object_desired_config, object_desired_config_color );
        visualizeObjectDelta( "object_delta", fbk.first.back(), object_desired_config );
    }

    // Querry each model for it's best trajectory
    std::vector< std::pair< AllGrippersTrajectory, double > > suggested_trajectories =
            model_set_->getDesiredGrippersTrajectories(
                object_current_config,
                object_desired_config,
                grippers_data_,
                MAX_GRIPPER_STEP_SIZE,
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
                                  fbk.second[gripper_ind].back(),
                                  suggested_trajectories[min_weighted_cost_ind].first[gripper_ind].back(),
                                  color );
        }
    }

    // TODO: deal with multiple predictions, which one is the best?
    VectorObjectTrajectory model_predictions = model_set_->makePredictions( suggested_trajectories[min_weighted_cost_ind].first, fbk.first.back() );
    // Send the predicted object config to the visualizer to plot
    if ( visualize_object_predicted_config_ )
    {
        std_msgs::ColorRGBA model_prediction_color;
        model_prediction_color.r = 1;
        model_prediction_color.g = 0;
        model_prediction_color.b = 0;
        model_prediction_color.a = 1;
        // TODO: make this work for non-ropes
        visualizeRopeObject( "rope_predicted_config", model_predictions[min_weighted_cost_ind].back(), model_prediction_color );
    }

    LOG_COND( loggers.at( "object_predicted_configuration" ) , logging_enabled_,
              (model_predictions[min_weighted_cost_ind].back()).format( eigen_io_one_line_ ) );

    return suggested_trajectories[min_weighted_cost_ind].first;
}

/**
 * @brief Planner::readSimulatorFeedbackBuffer
 * @return
 */
std::pair< ObjectTrajectory, AllGrippersTrajectory > Planner::readSimulatorFeedbackBuffer()
{
    // TODO: inherit this lock from the callee
    //boost::mutex::scoped_lock lock( input_mtx_ );

    std::pair< ObjectTrajectory, AllGrippersTrajectory > fbk;

    // record the data we have
    fbk.first = object_trajectory_;
    fbk.second = grippers_trajectory_;

    // reset our buffers
    object_trajectory_.clear();
    //object_trajectory_.push_back( fbk.first[0] );

    for ( size_t gripper_ind = 0; gripper_ind < grippers_trajectory_.size(); gripper_ind++ )
    {
        grippers_trajectory_[gripper_ind].clear();
//        grippers_trajectory_[gripper_ind].push_back( fbk.second[gripper_ind][0] );
    }

    return fbk;
}

/**
 * @brief Planner::updateModels
 * @param object_trajectory
 * @param grippers_trajectory
 */
void Planner::updateModels( const ObjectTrajectory& object_trajectory,
                            const AllGrippersTrajectory& grippers_trajectory )
{
    // TODO: recursive lock here
    ROS_INFO_NAMED( "planner" , "Updating models" );

    assert( grippers_trajectory.size() >= 1 );
    assert( object_trajectory.size() == grippers_trajectory[0].size() );

    if ( object_trajectory.size() >= 2 )
    {
        model_set_->updateModels( grippers_data_, grippers_trajectory, object_trajectory );

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
    else
    {
        assert( false && "THIS PAIR OF DEFORMALBE AND TASK IS NOT YET IMPLEMENTED" );
    }

    // TODO: the rest
}

void Planner::visualizeRopeObject( const std::string& marker_name,
                                   const ObjectPointSet& rope,
                                   const std_msgs::ColorRGBA& color )
{
    visualization_msgs::Marker marker;

    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.ns = marker_name;
    marker.id = 0;
    marker.scale.x = 0.1;
    marker.points = EigenMatrix3XdToVectorGeometryPoint( rope );
    marker.colors = std::vector< std_msgs::ColorRGBA >( (size_t)rope.cols(), color );
    visualization_marker_pub_.publish( marker );

    marker.type = visualization_msgs::Marker::SPHERE;
    marker.id = 1;
    marker.scale.x = 0.01;
    visualization_marker_pub_.publish( marker );
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

void Planner::visualizeTranslation(const std::string& marker_name,
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

void Planner::simulatorFbkCallback(
        const smmap_msgs::SimulatorFbkStamped& fbk )
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
//    std::cout << "Post convert:\n" << PrettyPrint::PrettyPrint( grippers_trajectory_[0][ grippers_trajectory_[0].size() -1 ] ) << std::endl ;
}

////////////////////////////////////////////////////////////////////////////////
// ROS Objects and Helpers
////////////////////////////////////////////////////////////////////////////////

void Planner::spin( double loop_rate )
{
    ros::NodeHandle ph("~");
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
    gripper_names_client.call( names_srv_data );
    std::vector< std::string > gripper_names = names_srv_data.response.names;

    // Service client to get the attached nodes and transform for each gripper
    ros::ServiceClient gripper_node_indices_client =
        nh_.serviceClient< smmap_msgs::GetGripperAttachedNodeIndices >( GetGripperAttachedNodeIndicesTopic( nh_ ) );
    gripper_node_indices_client.waitForExistence();

    ros::ServiceClient gripper_pose_client =
        nh_.serviceClient< smmap_msgs::GetGripperPose >( GetGripperPoseTopic( nh_ ) );
    gripper_pose_client.waitForExistence();

    // Now actually collect the data for each gripper from the simulator
    grippers_trajectory_.resize( gripper_names.size() );
    for ( size_t gripper_ind = 0; gripper_ind < gripper_names.size(); gripper_ind++ )
    {
        smmap_msgs::GetGripperAttachedNodeIndices node_srv_data;
        node_srv_data.request.name = gripper_names[gripper_ind];
        gripper_node_indices_client.call( node_srv_data );

        smmap_msgs::GetGripperPose pose_srv_data;
        pose_srv_data.request.name = gripper_names[gripper_ind];
        gripper_pose_client.call( pose_srv_data );

        std::vector< long > node_indices( node_srv_data.response.indices.size() );
        memcpy( &(node_indices.front()), &(node_srv_data.response.indices.front()), node_srv_data.response.indices.size() );

        grippers_data_.push_back(
                    GripperData( GeometryPoseToEigenAffine3d( pose_srv_data.response.pose ),
                                 node_indices, gripper_names[gripper_ind] ) );

        grippers_trajectory_[gripper_ind].push_back(
                GeometryPoseToEigenAffine3d( pose_srv_data.response.pose ) );

        ROS_INFO_NAMED( "planner" , "Gripper #%zu: %s", gripper_ind, PrettyPrint::PrettyPrint( grippers_data_[gripper_ind] ).c_str() );
    }
}

void Planner::getObjectInitialConfiguration()
{
    ROS_INFO_NAMED( "planner" , "Getting object initial configuration" );

    // Get the initial configuration of the object
    ros::ServiceClient object_initial_configuration_client =
        nh_.serviceClient< smmap_msgs::GetPointSet >( GetObjectInitialConfigurationTopic( nh_ ) );

    object_initial_configuration_client.waitForExistence();

    smmap_msgs::GetPointSet srv_data;
    object_initial_configuration_client.call( srv_data );
    object_initial_configuration_ =
        VectorGeometryPointToEigenMatrix3Xd( srv_data.response.points );

    object_trajectory_.clear();
    object_trajectory_.push_back( object_initial_configuration_ );

    ROS_INFO_NAMED( "planner" , "Number of points on object: %zu", srv_data.response.points.size() );
}
