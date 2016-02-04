#ifndef SMMAP_PLANNER_H
#define SMMAP_PLANNER_H

#include <boost/thread/recursive_mutex.hpp>

#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <std_msgs/ColorRGBA.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/MarkerArray.h>
#include <smmap_msgs/messages.h>

#include <arc_utilities/log.hpp>

#include "smmap/model_set.h"
#include "smmap/task.hpp"

namespace smmap
{
    class Planner
    {
        public:
            Planner( ros::NodeHandle& nh );

            ////////////////////////////////////////////////////////////////////
            // Main function that makes things happen
            ////////////////////////////////////////////////////////////////////

            void run( const double dt );

        private:

            ////////////////////////////////////////////////////////////////////
            // Magic numbers
            ////////////////////////////////////////////////////////////////////

            static constexpr double MAX_GRIPPER_VELOCITY = 0.05/20.0/0.01;

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

            bool visualize_gripper_translation_;

            ////////////////////////////////////////////////////////////////////
            // Task parameters
            ////////////////////////////////////////////////////////////////////

            std::unique_ptr< Task > task_;
            void initializeTask();

            std::unique_ptr< ModelSet > model_set_;

            std::vector< GripperData > grippers_data_;
            void getGrippersData();

            ////////////////////////////////////////////////////////////////////
            // Internal helpers for the run() function
            ////////////////////////////////////////////////////////////////////

            std::vector< AllGrippersSinglePose > replan(
                    const std::vector<WorldFeedback>& world_feedback,
                    const size_t planning_horizion,
                    const double dt );

            //std::pair<ObjectTrajectory, AllGrippersTrajectory> readSimulatorFeedbackBuffer();
            void updateModels( const std::vector< WorldFeedback >& feedback );

            ObjectTrajectory combineModelPredictions(
                    const VectorObjectTrajectory& model_predictions ) const;

            ObjectPointSet combineModelPredictionsLastTimestep(
                    const VectorObjectTrajectory& model_predictions ) const;

            Eigen::VectorXd combineModelDerivitives(
                    const std::vector< Eigen::VectorXd >& model_derivitives ) const;

            std::pair< Eigen::VectorXd, Eigen::MatrixXd > combineModelDerivitives(
                    const std::vector< std::pair< Eigen::VectorXd, Eigen::MatrixXd > >& model_derivitives ) const;

            std::vector< AllGrippersSinglePose > optimizeTrajectoryDirectShooting(const WorldFeedback& current_world_configuration,
                    std::vector<AllGrippersSinglePose> grippers_trajectory,
                    double dt ) const;

            ////////////////////////////////////////////////////////////////////
            // Task specific functionality
            ////////////////////////////////////////////////////////////////////

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


            ////////////////////////////////////////////////////////////////////
            // ROS Objects and Helpers
            ////////////////////////////////////////////////////////////////////

            // Our internal version of ros::spin()
            static void spin( double loop_rate );
            std::vector< std::string > getGripperNames();
            ObjectPointSet getObjectInitialConfiguration();
            std::vector< WorldFeedback > sendGripperTrajectory(
                    const smmap_msgs::CmdGrippersTrajectoryGoal& goal );

            smmap_msgs::CmdGrippersTrajectoryGoal noOpTrajectoryGoal( size_t num_no_op );
            smmap_msgs::CmdGrippersTrajectoryGoal toRosGoal(
                    const std::vector< AllGrippersSinglePose >& trajectory );

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;

            ros::Publisher confidence_pub_;
            image_transport::ImageTransport it_;
            image_transport::Publisher confidence_image_pub_;

            ros::Publisher visualization_marker_pub_;
            ros::Publisher visualization_marker_array_pub_;

            actionlib::SimpleActionClient< smmap_msgs::CmdGrippersTrajectoryAction > cmd_grippers_traj_client_;
    };
}

#endif // SMMAP_PLANNER_H
