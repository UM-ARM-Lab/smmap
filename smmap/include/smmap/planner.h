#ifndef smmap_planner_h
#define smmap_planner_h

#include <boost/thread/recursive_mutex.hpp>

#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <arc_utilities/maybe.hpp>

#include <visualization_msgs/MarkerArray.h>

#include <smmap_msgs/messages.h>

#include "smmap/model_set.h"
#include "smmap/task.h"

namespace smmap
{
    class Planner
    {
        public:
            Planner(ros::NodeHandle& nh,
                     TaskType task = TaskType::COVERAGE );

            ////////////////////////////////////////////////////////////////////
            // Main function that makes things happen
            ////////////////////////////////////////////////////////////////////

            void run( const size_t num_traj_cmds_per_loop = 1 );

        private:
            // TODO: Use this
            TaskType task_;
            std::unique_ptr< ModelSet > model_set_;
            // Stores a "gripper name", {gripper_node_indices} pair for each gripper
            VectorGrippersData gripper_data_;
            ObjectPointSet object_initial_configuration_;
            ObjectPointSet cover_points_;

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

            ObjectPointSet findObjectDesiredConfiguration( const ObjectPointSet& current );

            void visualizeRopeObject( const std::string& marker_name,
                                      const ObjectPointSet& rope,
                                      const std_msgs::ColorRGBA& color );

            void visualizeObjectDelta(const std::string& marker_name,
                                       const ObjectPointSet& current,
                                       const ObjectPointSet& desired );

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
            void getCoverPoints();

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
    };
}

#endif // planner_h
