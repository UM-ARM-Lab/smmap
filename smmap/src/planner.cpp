#include "smmap/planner.h"
#include "smmap/ros_params.hpp"
#include "smmap/trajectory.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <assert.h>

#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

#include <Eigen/QR>

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
    , cmd_grippers_traj_client_( nh_, GetCommandGripperTrajTopic( nh_ ), false )
{
    // Initialize our task object with whatever it needs
    initializeTask();

    // Get the data we need to create our model set
    getGrippersData();

    double translational_deformability, rotational_deformability;
    if ( ph_.getParam( "translational_deformability", translational_deformability ) &&
         ph_.getParam( "rotational_deformability", rotational_deformability ) )
    {
        model_set_ = std::unique_ptr< ModelSet >(
                    new ModelSet( grippers_data_, getObjectInitialConfiguration(), *task_,
                                  translational_deformability, rotational_deformability ) );
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

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "time",
                            Log::Log( log_folder + "time.txt", false ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log >(
                            "object_current_configuration",
                            Log::Log( log_folder + "object_current_configuration.txt", false ) ) );

        loggers.insert( std::make_pair< std::string, Log::Log >(
                            "object_desired_configuration",
                            Log::Log( log_folder + "object_desired_configuration.txt", false ) ) );

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "error",
                            Log::Log( log_folder + "error.txt", false ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "suggested_grippers_delta",
                            Log::Log( log_folder + "suggested_grippers_delta.txt", false ) ) ) ;

        loggers.insert( std::make_pair< std::string, Log::Log > (
                            "object_predicted_configuration",
                            Log::Log( log_folder + "object_predicted_configuration.txt", false ) ) ) ;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Main function that makes things happen
////////////////////////////////////////////////////////////////////////////////

void Planner::run( const size_t num_traj_cmds_per_loop , const double dt )
{
    // TODO: remove this hardcoded spin rate
    boost::thread spin_thread( boost::bind( &Planner::spin, 1000 ) );

    ROS_INFO_NAMED( "planner" , "Waiting for the robot gripper action server to be available" );
    cmd_grippers_traj_client_.waitForServer();

    // Objects used for simulator/robot IO
    std::vector< WorldFeedback > world_feedback;

    ROS_INFO_NAMED( "planner", "Kickstarting the planner with a no-op" );
    world_feedback = sendGripperTrajectory( noOpTrajectoryGoal( 2 ) );

    // Run the planner at whatever rate we've been given
    ROS_INFO_NAMED( "planner" , "Running our planner" );
    while ( ros::ok() )
    {
        // get the best trajectory given the current data
        const std::vector< AllGrippersSinglePose > best_grippers_traj = replan(
                    world_feedback, num_traj_cmds_per_loop, dt );

        ROS_INFO_NAMED( "planner" , "Sending 'best' trajectory" );
        world_feedback = sendGripperTrajectory( toRosGoal( best_grippers_traj ) );

        if ( task_->maxTime() < world_feedback.back().sim_time_ )
        {
            ros::shutdown();
        }
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
std::vector< AllGrippersSinglePose > Planner::replan(
        std::vector< WorldFeedback >& world_feedback,
        size_t num_traj_cmds_per_loop, double dt )
{
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

    ROS_INFO_NAMED( "planner" , "Finding 'best' trajectory suggestion for a seed" );
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

    std::vector< AllGrippersSinglePose > best_trajectory =
            optimizeTrajectoryDirectShooting(
                world_feedback.back(),
                suggested_trajectories[min_weighted_cost_ind].first,
                dt );

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
                                  best_trajectory.back()[gripper_ind],
                                  color );
        }
    }

    // TODO: deal with multiple predictions, which one is the best?
    VectorObjectTrajectory model_predictions = model_set_->makePredictions(
                world_feedback.back(),
                best_trajectory,
                CalculateGrippersVelocities( best_trajectory, dt ),
                dt );

    // TODO: make this logging work for multi-step trajectories
    if ( logging_enabled_ )
    {
        assert( num_traj_cmds_per_loop == 1 );
    }

    LOG_COND( loggers.at( "time" ), logging_enabled_,
              world_feedback.back().sim_time_ );

    LOG_COND( loggers.at( "object_current_configuration" ), logging_enabled_,
              world_feedback.back().object_configuration_.format( eigen_io_one_line_ ) );

    LOG_COND( loggers.at( "object_desired_configuration" ), logging_enabled_,
              object_desired_config.format( eigen_io_one_line_ ) );

    LOG_COND( loggers.at( "error"), logging_enabled_,
              task_->calculateError( world_feedback.back().object_configuration_ ) );

    LOG_COND( loggers.at( "suggested_grippers_delta"), logging_enabled_,
              PrintDeltaOneLine( suggested_trajectories[min_weighted_cost_ind].first ) );

    LOG_COND( loggers.at( "object_predicted_configuration" ), logging_enabled_,
              (model_predictions[min_weighted_cost_ind].back()).format( eigen_io_one_line_ ) );

    return suggested_trajectories[min_weighted_cost_ind].first;
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


ObjectTrajectory Planner::combineModelPredictions(
        const VectorObjectTrajectory& model_predictions ) const
{
    assert( model_predictions.size() > 0 );
    assert( model_predictions[0].size() > 0 );

    const std::vector< double >& model_confidences =
            model_set_->getModelConfidence();
    double total_weight = std::accumulate( model_confidences.begin(), model_confidences.end(), 0. );

    ObjectTrajectory weighted_average_trajectory( model_predictions[0].size(),
            ObjectPointSet::Zero( 3, model_predictions[0][0].cols() ) );

    // Itterate through each model prediction
    for ( size_t model_ind = 0; model_ind < model_predictions.size(); model_ind++ )
    {
        // For each model, itterate through time, weighing by the model confidence
        for ( size_t time_ind = 0; time_ind < model_predictions[model_ind].size(); time_ind++ )
        {
            weighted_average_trajectory[time_ind] +=
                    model_predictions[model_ind][time_ind] * model_confidences[model_ind] / total_weight;
        }
    }

    return weighted_average_trajectory;
}

ObjectPointSet Planner::combineModelPredictionsLastTimestep(
        const VectorObjectTrajectory& model_predictions ) const
{
    assert( model_predictions.size() > 0 );
    size_t traj_length = model_predictions[0].size();
    assert( traj_length > 0 );

    const std::vector< double >& model_confidences =
            model_set_->getModelConfidence();
    double total_weight = std::accumulate( model_confidences.begin(), model_confidences.end(), 0. );

    ObjectPointSet weighted_average_configuration = ObjectPointSet::Zero( 3, model_predictions[0][0].cols() );

    // Itterate through each model prediction
    for ( size_t model_ind = 0; model_ind < model_predictions.size(); model_ind++ )
    {
        assert( model_predictions[model_ind].size() == traj_length );

        weighted_average_configuration +=
                model_predictions[model_ind][traj_length-1] * model_confidences[model_ind] / total_weight;
    }

    return weighted_average_configuration;
}

std::pair< Eigen::VectorXd, Eigen::MatrixXd > Planner::combineModelDerivitives(
        const std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >& model_derivitives ) const
{
    assert( model_derivitives.size() > 0 );

    const std::vector< double >& model_confidences =
            model_set_->getModelConfidence();
    double total_weight = std::accumulate( model_confidences.begin(), model_confidences.end(), 0. );

    std::pair< Eigen::VectorXd, Eigen::MatrixXd > weighted_average_derivitive(
            Eigen::VectorXd::Zero( model_derivitives[0].first.size() ),
            Eigen::MatrixXd::Zero( model_derivitives[0].second.rows(), model_derivitives[0].second.cols() ) );

    // Itterate through each model derivitive
    for ( size_t model_ind = 0; model_ind < model_derivitives.size(); model_ind++ )
    {
        weighted_average_derivitive.first +=
                model_derivitives[model_ind].first * model_confidences[model_ind] / total_weight;

        weighted_average_derivitive.second +=
                model_derivitives[model_ind].second * model_confidences[model_ind] / total_weight;
    }

    return weighted_average_derivitive;
}


std::vector< AllGrippersSinglePose > Planner::optimizeTrajectoryDirectShooting(
        const WorldFeedback& current_world_configuration,
        std::vector< AllGrippersSinglePose > grippers_trajectory,
        double dt ) const
{
    ROS_INFO_NAMED( "planner" , "Using direct shooting to optimize the trajectory" );

    std::cout << "Pre optimizing grippers last pose:\n"
              << PrettyPrint::PrettyPrint( grippers_trajectory.back(), true, "\n " ) << std::endl;

    // TODO: move these magic numbers elsewhere
    #warning "Magic numbers here need to be moved elsewhere"
    const int MAX_ITTR = 10;
    const double LEARNING_RATE = 0.1;

    double objective_delta = std::numeric_limits< double >::infinity();

    std::vector< AllGrippersSingleVelocity > grippers_velocities =
            CalculateGrippersVelocities( grippers_trajectory, dt );

    std::function< double( const ObjectPointSet& ) > objective_function =
            std::bind( &Task::calculateError, task_.get(), std::placeholders::_1  );

    // TODO: replace this with something that just averages the last positions
    double objective_value = objective_function(
                combineModelPredictionsLastTimestep(
                     model_set_->makePredictions(
                         current_world_configuration,
                         grippers_trajectory,
                         grippers_velocities,
                         dt ) ) );

    int ittr = 0;
    do
    {
        ROS_INFO_STREAM_NAMED( "planner" , "  Direct shooting itteration " << ittr << ". Current objective value " << objective_value );
        // Find the first and second derivitives of the objective function with
        // respect to the gripper velocities
        //
        // Note that these derivitives are layed out as
        //     Eigen::VectorXd( num_grippers * 6 * num_timesteps ),
        //     Eigen::MatrixXd( num_grippers * 6 * num_timesteps, num_grippers * 6 * num_timesteps ) );
        //
        std::pair< Eigen::VectorXd, Eigen::MatrixXd > derivitives =
                combineModelDerivitives(
                    model_set_->getObjectiveFunctionDerivitives(
                        current_world_configuration,
                        grippers_trajectory,
                        grippers_velocities,
                        dt,
                        objective_function ) );

        // Update the gripper velocities based on a Newton style gradient descent
//        Eigen::VectorXd velocity_update = derivitives.second.colPivHouseholderQr().solve( -derivitives.first );
        Eigen::VectorXd velocity_update = -derivitives.first ;
        if ( velocity_update.norm() > MAX_GRIPPER_VELOCITY )
        {
            velocity_update *= MAX_GRIPPER_VELOCITY / velocity_update.norm();
        }

        for ( size_t time_ind = 0; time_ind < grippers_velocities.size(); time_ind++ )
        {
            for ( size_t gripper_ind = 0; gripper_ind < grippers_data_.size(); gripper_ind++ )
            {
                grippers_velocities[time_ind][gripper_ind] += LEARNING_RATE *
                        velocity_update.segment< 6 >(
                            (long)( time_ind * grippers_data_.size() * 6 +
                                    gripper_ind * 6 ) );

                if ( GripperVelocity6dNorm( grippers_velocities[time_ind][gripper_ind] ) > MAX_GRIPPER_VELOCITY )
                {
                    grippers_velocities[time_ind][gripper_ind] *=
                            MAX_GRIPPER_VELOCITY / GripperVelocity6dNorm( grippers_velocities[time_ind][gripper_ind] );
                }
            }
        }

        // Update the trajectory of the grippers based on the new
        grippers_trajectory = CalculateGrippersTrajectory(
                    grippers_trajectory[0],
                    grippers_velocities,
                    dt );

        // Calculate the new value of the objective function at the updated velocity
        // locations
        double new_objective_value = objective_function(
                    combineModelPredictionsLastTimestep(
                        model_set_->makePredictions(
                            current_world_configuration,
                            grippers_trajectory,
                            grippers_velocities,
                            dt ) ) );

        objective_delta = std::abs( objective_value - new_objective_value );
        objective_value = new_objective_value;

        ittr++;
    }
    while ( ittr < MAX_ITTR  && objective_delta > objective_value * 1e-6 );

    std::cout << "Post optimizing grippers last pose:\n"
              << PrettyPrint::PrettyPrint( grippers_trajectory.back(), true, "\n " ) << std::endl;

    return grippers_trajectory;
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

std::vector< WorldFeedback > Planner::sendGripperTrajectory(
        const smmap_msgs::CmdGrippersTrajectoryGoal& goal )
{
    std::vector< WorldFeedback > feedback;

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

smmap_msgs::CmdGrippersTrajectoryGoal Planner::noOpTrajectoryGoal( size_t num_no_op )
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
            ROS_FATAL_STREAM_NAMED( "planner", "Unabled to retrieve gripper pose: " << grippers_data_[gripper_ind].name );
        }

        grippers_pose.pose[gripper_ind] = pose_srv_data.response.pose;
    }

    goal.trajectory.resize( num_no_op, grippers_pose );

    return goal;
}

smmap_msgs::CmdGrippersTrajectoryGoal Planner::toRosGoal(
        const std::vector<AllGrippersSinglePose>& trajectory )
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
