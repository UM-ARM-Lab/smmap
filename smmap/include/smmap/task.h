#ifndef task_h
#define task_h

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <smmap_msgs/messages.h>

#include "smmap/point_reflector.hpp"
#include "smmap/ros_params.h"
#include "smmap/task_enums.h"
#include "smmap/trajectory.h"

namespace smmap
{
    class Task
    {
        public:
            typedef std::shared_ptr< Task > Ptr;

            Task( ros::NodeHandle& nh )
                : visualize_prediction_( true )
            {
                // Publish visualization request markers
                visualization_marker_pub_ =
                        nh.advertise< visualization_msgs::Marker >( GetVisualizationMarkerTopic( nh ), 10 );

                visualization_marker_array_pub_ =
                        nh.advertise< visualization_msgs::MarkerArray >( GetVisualizationMarkerArrayTopic( nh ), 10 );
            }

            virtual void visualizePredictions( const VectorObjectTrajectory& model_predictions, size_t best_traj )
            {
                (void)model_predictions;
                (void)best_traj;
            }

            ObjectPointSet findObjectDesiredConfiguration( const ObjectPointSet& current_configuration )
            {
                return doFindObjectDesiredConfiguration( current_configuration );
            }

            double calculateError( const ObjectPointSet& current_configuration )
            {
                return doCalculateError( current_configuration );
            }

            virtual double getDeformability() const = 0;                // k
            virtual double getCollisionScalingFactor() const = 0;       // beta (or k2)
            virtual double getStretchingScalingThreshold() const = 0;   // lambda
            virtual bool getUseRotation() const = 0;

        // TODO: this is now a bastardized abstract class. This is bad.
        protected:
            ros::Publisher visualization_marker_pub_;
            ros::Publisher visualization_marker_array_pub_;

            bool visualize_prediction_;

        private:
            virtual ObjectPointSet doFindObjectDesiredConfiguration( const ObjectPointSet& current_configuration ) = 0;
            virtual double doCalculateError( const ObjectPointSet& current_configuration ) = 0;
    };

    class RopeCoverage : public Task
    {
        public:
            RopeCoverage( ros::NodeHandle& nh )
                : Task( nh )
            {
                ROS_INFO_NAMED( "rope_coverage_task" , "Getting cover points" );

                // Get the initial configuration of the object
                ros::ServiceClient cover_points_client =
                    nh.serviceClient< smmap_msgs::GetPointSet >( GetCoverPointsTopic( nh ) );

                cover_points_client.waitForExistence();

                smmap_msgs::GetPointSet srv_data;
                cover_points_client.call( srv_data );
                cover_points_ =
                    EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd( srv_data.response.points );

                ROS_INFO_NAMED( "rope_coverage_task" , "Number of cover points: %zu", srv_data.response.points.size() );
            }

            virtual void visualizePredictions( const VectorObjectTrajectory& model_predictions, size_t best_traj )
            {
                std_msgs::ColorRGBA color;
                color.r = 1;
                color.g = 1;
                color.b = 0;
                color.a = 1;

                visualizeRope( model_predictions[best_traj].back(), color, "rope_predicted" );
            }

            void visualizeRope( const ObjectPointSet& rope, const std_msgs::ColorRGBA& color, const std::string& name )
            {
                visualization_msgs::Marker marker;

                marker.type = visualization_msgs::Marker::LINE_STRIP;
                marker.ns = name;
                marker.id = 0;
                marker.scale.x = 0.1;
                marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint( rope );
                marker.colors = std::vector< std_msgs::ColorRGBA >( (size_t)rope.cols(), color );
                visualization_marker_pub_.publish( marker );

                marker.type = visualization_msgs::Marker::SPHERE;
                marker.id = 1;
                marker.scale.x = 0.01;
                visualization_marker_pub_.publish( marker );
            }

            double getDeformability() const
            {
                return 0.5*20; // k
            }

            double getCollisionScalingFactor() const
            {
                return  10*20; // beta
            }

            double getStretchingScalingThreshold() const
            {
                return 0.1/20; // lambda
            }

            bool getUseRotation() const
            {
                return true;
            }

        private:
            ObjectPointSet doFindObjectDesiredConfiguration( const ObjectPointSet& current_configuration )
            {
                ROS_INFO_NAMED( "rope_coverage_task" , "Finding 'best' configuration" );

                // point should be the same size
                assert( current_configuration.rows() == cover_points_.rows() );

                ObjectPointSet desired_configuration = current_configuration;

                // We'll need to track how many cover points are mapping to a given object point
                // in order to do the averaging.
//                std::vector< int > num_mapped( (size_t)current_configuration.cols(), 0 );

                // for every cover point, find the nearest deformable object point
                for ( int cover_ind = 0; cover_ind < cover_points_.cols(); cover_ind++ )
                {
                    Eigen::Vector3d cover_point = cover_points_.block< 3, 1 >( 0, cover_ind );

                    ObjectPointSet diff = ( cover_point * Eigen::MatrixXd::Ones( 1, current_configuration.cols() ) ) - current_configuration;

                    Eigen::RowVectorXd dist_sq = diff.array().square().colwise().sum();

                    // find the closest deformable point
                    int min_ind = -1;
                    double min_dist = std::numeric_limits< double >::infinity();
                    for ( int object_ind = 0; object_ind < dist_sq.cols(); object_ind++ )
                    {
                        if ( dist_sq( object_ind ) < min_dist )
                        {
                            min_ind = object_ind;
                            min_dist = dist_sq( object_ind );
                        }
                    }

                    assert( min_ind >= 0 );
    //                // If this is the first time we've found this as the closest, just use it
    //                num_mapped[(size_t)min_ind]++;
    //                if ( num_mapped[(size_t)min_ind] == 1 )
    //                {
    //                    desired_configuration.block< 3, 1 >( 0, min_ind ) = cover_points_.block< 3, 1 >( 0, cover_ind );
    //                }
    //                // Otherwise average it
    //                else
    //                {
    //                    // Averaging method:
    //                    // http://jvminside.blogspot.com/2010/01/incremental-average-calculation.html
    //                    desired_configuration.block< 3, 1 >( 0, min_ind ) =
    //                            desired_configuration.block< 3, 1 >( 0, min_ind ) +virtual void visualizePredictions( const VectorObjectTrajectory& model_predictions )
    //                            (cover_points_.block< 3, 1 >( 0, cover_ind ) - desired_configuration.block< 3, 1 >( 0, min_ind ))
    //                            / num_mapped[(size_t)min_ind];
    //                }

                    if ( min_dist >= 0.01 )
                    {
                        desired_configuration.block< 3, 1 >( 0, min_ind ) = desired_configuration.block< 3, 1 >( 0, min_ind ) + diff.block< 3, 1 >( 0, min_ind );
                    }
                }

                return desired_configuration;
            }

            double doCalculateError( const ObjectPointSet &current )
            {
                //TODO: do
                (void)current;
                return -1;
            }

            ObjectPointSet cover_points_;
    };

    class ClothColabFolding : public Task
    {
        public:
            ClothColabFolding( ros::NodeHandle& nh )
                : Task( nh )
            {
                visualize_prediction_ = false;

                ROS_INFO_NAMED( "cloth_colab_folding_task" , "Getting object initial configuration" );

                // Get the initial configuration of the object
                ros::ServiceClient object_initial_configuration_client =
                    nh.serviceClient< smmap_msgs::GetPointSet >( GetObjectInitialConfigurationTopic( nh ) );

                object_initial_configuration_client.waitForExistence();

                smmap_msgs::GetPointSet initial_point_set_data;
                object_initial_configuration_client.call( initial_point_set_data );
                ObjectPointSet object_initial_configuration =
                    EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd( initial_point_set_data.response.points );

                ROS_INFO_NAMED( "cloth_colab_folding_task" , "Getting mirror line" );

                // Get the initial configuration of the object
                ros::ServiceClient mirror_line_client =
                    nh.serviceClient< smmap_msgs::GetMirrorLine >( GetMirrorLineTopic( nh ) );

                mirror_line_client.waitForExistence();

                smmap_msgs::GetMirrorLine mirror_line_data;
                mirror_line_client.call( mirror_line_data );

                point_reflector_ = PointReflector( mirror_line_data.response.mid_x,
                                                   mirror_line_data.response.min_y,
                                                   mirror_line_data.response.max_y );

                ROS_INFO_NAMED( "cloth_colab_folding_task", "Finding point correspondences" );

                for ( long node_ind = 0; node_ind < object_initial_configuration.cols(); node_ind++ )
                {
                    // for every node on one side of the mirror line, find the closest match on the other side
                    // Note that nodes that have an x value > than mid_x are on the manual gripper side
                    if ( object_initial_configuration( 0, node_ind ) > mirror_line_data.response.mid_x )
                    {
                        long mirror_ind = closestPointInSet( object_initial_configuration,
                                point_reflector_.reflect( object_initial_configuration.block< 3, 1 >( 0, node_ind ) ) );

                        mirror_map_[ node_ind ] = mirror_ind;
                    }
                }
            }

            virtual void visualizePredictions( const VectorObjectTrajectory& model_predictions, size_t best_traj )
            {
                if ( visualize_prediction_ )
                {
                    std_msgs::ColorRGBA color;
                    color.r = 1;
                    color.g = 1;
                    color.b = 0;
                    color.a = 1;

                    visualizeCloth( model_predictions[best_traj].back(), color, "cloth_predicted" );
                }
            }

            void visualizeCloth( const ObjectPointSet& cloth, const std_msgs::ColorRGBA color, const std::string& name  )
            {
                std::vector< std_msgs::ColorRGBA > colors( (size_t)cloth.cols(), color );

                visualizeCloth( cloth, colors, name );
            }

            void visualizeCloth( const ObjectPointSet& cloth, std::vector< std_msgs::ColorRGBA > colors, const std::string& name  )
            {
                visualization_msgs::Marker marker;

                marker.type = visualization_msgs::Marker::POINTS;
                marker.ns = name;
                marker.id = 0;
                marker.scale.x = 0.002;
                marker.scale.y = 0.002;
                marker.points = EigenHelpersConversions::EigenMatrix3XdToVectorGeometryPoint( cloth );
                marker.colors = colors;

                visualization_marker_pub_.publish( marker );
            }

            double getDeformability() const
            {
                return 0.7*20; // k
            }

            double getCollisionScalingFactor() const
            {
                return  100*20; // beta
            }

            double getStretchingScalingThreshold() const
            {
                return 0.1*20; // lambda
            }

            bool getUseRotation() const
            {
                return true;
            }

        private:
            ObjectPointSet doFindObjectDesiredConfiguration( const ObjectPointSet& current_configuration )
            {
                ObjectPointSet desired_configuration = current_configuration;

                ObjectPointSet robot_cloth_points_desired( 3, (long)mirror_map_.size() );
                ObjectPointSet robot_cloth_points_current( 3, (long)mirror_map_.size() );
                std::vector< std_msgs::ColorRGBA > robot_cloth_points_current_colors( mirror_map_.size() );

                long robot_cloth_points_ind = 0;
                for ( std::map< long, long >::iterator ittr = mirror_map_.begin(); ittr != mirror_map_.end(); ittr++ )
                {
                    desired_configuration.block< 3, 1 >( 0, ittr->second ) =
                            point_reflector_.reflect( current_configuration.block< 3, 1 >( 0, ittr->first ) );

                    robot_cloth_points_desired.block< 3, 1 >( 0, robot_cloth_points_ind ) = desired_configuration.block< 3, 1 >( 0, ittr->second );
                    robot_cloth_points_current.block< 3, 1 >( 0, robot_cloth_points_ind ) = current_configuration.block< 3, 1 >( 0, ittr->second );

                    std_msgs::ColorRGBA color;

                    color.r = 0;
                    color.g = (float)( robot_cloth_points_desired.block< 3, 1 >( 0, robot_cloth_points_ind )
                               - robot_cloth_points_current.block< 3, 1 >( 0, robot_cloth_points_ind ) ).norm() * 20;
                    color.b = 0;
                    color.a = 1;

                    robot_cloth_points_current_colors[(size_t)robot_cloth_points_ind] = color;

                    robot_cloth_points_ind++;
                }

                std_msgs::ColorRGBA color;

                color.r = 1;
                color.g = 0;
                color.b = 0;
                color.a = 1;

                visualizeCloth( robot_cloth_points_desired, color, "cloth_desired" );
                visualizeCloth( robot_cloth_points_current, robot_cloth_points_current_colors, "cloth_current" );

                return desired_configuration;
            }

            double doCalculateError( const ObjectPointSet &current )
            {
                //TODO: do
                (void)current;
                return -1;
            }

            std::map< long, long > mirror_map_;
            PointReflector point_reflector_;
    };
}

#endif // task_h
