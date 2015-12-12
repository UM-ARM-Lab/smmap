#ifndef smmap_planner_h
#define smmap_planner_h

#include <boost/thread/recursive_mutex.hpp>

#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <smmap_msgs/messages.h>

#include <arc_utilities/log.hpp>

#include "smmap/model_set.h"
#include "smmap/task.h"

namespace smmap
{
    class Planner
    {
        public:
            Planner(ros::NodeHandle& nh );

            ////////////////////////////////////////////////////////////////////
            // Main function that makes things happen
            ////////////////////////////////////////////////////////////////////

            void run( const size_t num_traj_cmds_per_loop = 1 );

        private:

            ////////////////////////////////////////////////////////////////////
            // Magic numbers
            ////////////////////////////////////////////////////////////////////

            static constexpr double MAX_GRIPPER_STEP_SIZE = 0.05/20.0;

            ////////////////////////////////////////////////////////////////////
            // Loggering functionality
            ////////////////////////////////////////////////////////////////////

            // TODO: move this from here, this is terrible
            static const Eigen::IOFormat eigen_io_one_line_;

            bool logging_enabled_;
            std::map< std::string, Log::Log > loggers;

            ////////////////////////////////////////////////////////////////////
            // Visualization flags
            ////////////////////////////////////////////////////////////////////

            bool visualize_object_desired_config_;
            bool visualize_object_predicted_config_;
            bool visualize_gripper_translation_;
            bool visualize_correspondances_;

            ////////////////////////////////////////////////////////////////////
            // Task parameters
            ////////////////////////////////////////////////////////////////////

            std::unique_ptr< Task > task_;
            void initializeTask();

            std::unique_ptr< ModelSet > model_set_;

            VectorGrippersData grippers_data_;
            ObjectPointSet object_initial_configuration_;

            ////////////////////////////////////////////////////////////////////
            // Internal helpers for the run() function
            ////////////////////////////////////////////////////////////////////

            AllGrippersTrajectory replan( size_t num_traj_cmds_per_loop );

            std::pair<ObjectTrajectory, AllGrippersTrajectory> readSimulatorFeedbackBuffer();
            void updateModels( const ObjectTrajectory& object_trajectory,
                               const AllGrippersTrajectory& grippers_trajectory );


            ////////////////////////////////////////////////////////////////////
            // Task specific functionality
            ////////////////////////////////////////////////////////////////////

            void visualizeRopeObject( const std::string& marker_name,
                                      const ObjectPointSet& rope,
                                      const std_msgs::ColorRGBA& color );

            void visualizeObjectDelta( const std::string& marker_name,
                                       const ObjectPointSet& current,
                                       const ObjectPointSet& desired );

            void visualizeTranslation( const std::string& marker_name,
                                       const geometry_msgs::Point& start,
                                       const geometry_msgs::Point& end,
                                       const std_msgs::ColorRGBA& color );

            void visualizeTranslation( const std::string& marker_name,
                                       const Eigen::Vector3d& start,
                                       const Eigen::Vector3d& end,
                                       const std_msgs::ColorRGBA& color );

            void visualizeTranslation( const std::string& marker_name,
                                       const Eigen::Affine3d& start,
                                       const Eigen::Affine3d& end,
                                       const std_msgs::ColorRGBA& color );

            void visualizeLines( const std::string& marker_name,
                                 const EigenHelpers::VectorVector3d& start,
                                 const EigenHelpers::VectorVector3d& end,
                                 const std_msgs::ColorRGBA& color );

            ////////////////////////////////////////////////////////////////////
            // ROS Callbacks
            ////////////////////////////////////////////////////////////////////

            // TODO: when moving to a real robot, create a node that deals with
            // synchronization problems, and rename this function
            void simulatorFbkCallback( const smmap_msgs::SimulatorFbkStamped& fbk );

            ////////////////////////////////////////////////////////////////////
            // ROS Objects and Helpers
            ////////////////////////////////////////////////////////////////////

            // Our internal version of ros::spin()
            static void spin( double loop_rate );
            void getGrippersData();
            void getObjectInitialConfiguration();

            ros::NodeHandle nh_;

            ros::Publisher confidence_pub_;
            image_transport::ImageTransport it_;
            image_transport::Publisher confidence_image_pub_;
            ros::ServiceClient cmd_gripper_traj_client_;

            ros::Publisher visualization_marker_pub_;
            ros::Publisher visualization_marker_array_pub_;

            // global input mutex
            boost::recursive_mutex input_mtx_;

            ros::Subscriber simulator_fbk_sub_;
            ObjectTrajectory object_trajectory_;
            AllGrippersTrajectory grippers_trajectory_;
            double sim_time_;
    };
}

#endif // planner_h
