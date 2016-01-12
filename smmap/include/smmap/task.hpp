#ifndef task_hpp
#define task_hpp

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
            {
                // Publish visualization request markers
                visualization_marker_pub_ =
                        nh.advertise< visualization_msgs::Marker >( GetVisualizationMarkerTopic( nh ), 10 );

                visualization_marker_array_pub_ =
                        nh.advertise< visualization_msgs::MarkerArray >( GetVisualizationMarkerArrayTopic( nh ), 10 );
            }

            virtual void visualizePredictions( const VectorObjectTrajectory& model_predictions, size_t best_traj ) const
            {
                (void)model_predictions;
                (void)best_traj;
            }

            virtual ObjectPointSet findObjectDesiredConfiguration( const ObjectPointSet& current_configuration ) const = 0;

            virtual double calculateError( const ObjectPointSet& current_configuration ) const
            {
                const ObjectPointSet desired_configuration = findObjectDesiredConfiguration( current_configuration );

                return (desired_configuration - current_configuration).norm();
            }

            virtual ObjectPointSet getObjectErrorGradient( ObjectPointSet current_configuration ) const
            {
                const double current_error = calculateError( current_configuration );
                const double delta = 0.01;
                ObjectPointSet gradient( 3, current_configuration.cols() );

                for ( long ind = 0; ind < 3 * current_configuration.cols() ; ind++ )
                {
                    current_configuration.data()[ind] += delta;
                    gradient.data()[ind] = ( calculateError( current_configuration ) - current_error ) / delta;
                    current_configuration.data()[ind] -= delta;
                }

                return gradient;
            }

            virtual double getDeformability() const = 0;                // k
            virtual double getCollisionScalingFactor() const = 0;       // beta (or k2)
            virtual double getStretchingScalingThreshold() const = 0;   // lambda
            virtual bool getUseRotation() const = 0;

        protected:
            mutable ros::Publisher visualization_marker_pub_;
            mutable ros::Publisher visualization_marker_array_pub_;
    };

    class RopeCoverage : public Task
    {
        public:
            RopeCoverage( ros::NodeHandle& nh )
                : Task( nh )
                , cover_points_( getCoverPointsHelper( nh ) )
            {}

            virtual void visualizePredictions( const VectorObjectTrajectory& model_predictions, size_t best_traj ) const
            {
                std_msgs::ColorRGBA color;
                color.r = 1;
                color.g = 1;
                color.b = 0;
                color.a = 1;

                visualizeRope( model_predictions[best_traj].back(), color, "rope_predicted" );
            }

            void visualizeRope( const ObjectPointSet& rope, const std_msgs::ColorRGBA& color, const std::string& name ) const
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

            ObjectPointSet findObjectDesiredConfiguration( const ObjectPointSet& current_configuration ) const
            {
                ROS_INFO_NAMED( "rope_coverage_task" , "Finding 'best' configuration" );

                // point should be the same size
                assert( current_configuration.rows() == cover_points_.rows() );

                ObjectPointSet desired_configuration = current_configuration;

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

                    if ( min_dist >= 0.01 )
                    {
                        desired_configuration.block< 3, 1 >( 0, min_ind ) = desired_configuration.block< 3, 1 >( 0, min_ind ) + diff.block< 3, 1 >( 0, min_ind );
                    }
                }

                return desired_configuration;
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
            /// Stores the points that we are trying to cover with the rope
            const ObjectPointSet cover_points_;
            /// Constructor helper that allows cover_points_ to be const
            ObjectPointSet getCoverPointsHelper( ros::NodeHandle& nh )
            {
                ROS_INFO_NAMED( "rope_coverage_task" , "Getting cover points" );

                // Get the initial configuration of the object
                ros::ServiceClient cover_points_client =
                    nh.serviceClient< smmap_msgs::GetPointSet >( GetCoverPointsTopic( nh ) );

                cover_points_client.waitForExistence();

                smmap_msgs::GetPointSet srv_data;
                cover_points_client.call( srv_data );
                ObjectPointSet cover_points =
                    EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd( srv_data.response.points );

                ROS_INFO_NAMED( "rope_coverage_task" , "Number of cover points: %zu", srv_data.response.points.size() );

                return cover_points;
            }
    };

    class ClothColabFolding : public Task
    {
        public:
            ClothColabFolding( ros::NodeHandle& nh )
                : Task( nh )
                , point_reflector_( createPointReflector( nh ) )
                , mirror_map_( createMirrorMap( nh, point_reflector_ ) )
            {}

            virtual void visualizePredictions( const VectorObjectTrajectory& model_predictions, size_t best_traj ) const
            {
                std_msgs::ColorRGBA color;
                color.r = 1;
                color.g = 1;
                color.b = 0;
                color.a = 1;

                visualizeCloth( model_predictions[best_traj].back(), color, "cloth_predicted" );
            }

            void visualizeCloth( const ObjectPointSet& cloth, const std_msgs::ColorRGBA color, const std::string& name ) const
            {
                std::vector< std_msgs::ColorRGBA > colors( (size_t)cloth.cols(), color );

                visualizeCloth( cloth, colors, name );
            }

            void visualizeCloth( const ObjectPointSet& cloth, std::vector< std_msgs::ColorRGBA > colors, const std::string& name ) const
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

            ObjectPointSet findObjectDesiredConfiguration( const ObjectPointSet& current_configuration ) const
            {
                ObjectPointSet desired_configuration = current_configuration;

                ObjectPointSet robot_cloth_points_desired( 3, (long)mirror_map_.size() );
                ObjectPointSet robot_cloth_points_current( 3, (long)mirror_map_.size() );
                std::vector< std_msgs::ColorRGBA > robot_cloth_points_current_colors( mirror_map_.size() );

                long robot_cloth_points_ind = 0;
                for ( std::map< long, long >::const_iterator ittr = mirror_map_.begin(); ittr != mirror_map_.end(); ittr++ )
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
            const PointReflector point_reflector_;
            PointReflector createPointReflector( ros::NodeHandle& nh )
            {
                ROS_INFO_NAMED( "cloth_colab_folding_task" , "Getting mirror line" );

                // Get the initial configuration of the object
                ros::ServiceClient mirror_line_client =
                    nh.serviceClient< smmap_msgs::GetMirrorLine >( GetMirrorLineTopic( nh ) );

                mirror_line_client.waitForExistence();

                smmap_msgs::GetMirrorLine mirror_line_data;
                mirror_line_client.call( mirror_line_data );

                return PointReflector( mirror_line_data.response.mid_x,
                                       mirror_line_data.response.min_y,
                                       mirror_line_data.response.max_y );
            }

            const std::map< long, long > mirror_map_;
            static std::map< long, long > createMirrorMap( ros::NodeHandle& nh, const PointReflector& point_reflector )
            {
                ROS_INFO_NAMED( "cloth_colab_folding_task" , "Getting object initial configuration" );

                // Get the initial configuration of the object
                ros::ServiceClient object_initial_configuration_client =
                    nh.serviceClient< smmap_msgs::GetPointSet >( GetObjectInitialConfigurationTopic( nh ) );

                object_initial_configuration_client.waitForExistence();

                smmap_msgs::GetPointSet initial_point_set_data;
                object_initial_configuration_client.call( initial_point_set_data );
                ObjectPointSet object_initial_configuration =
                    EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd( initial_point_set_data.response.points );

                ROS_INFO_NAMED( "cloth_colab_folding_task", "Finding point correspondences" );

                std::map< long, long > mirror_map;
                for ( long node_ind = 0; node_ind < object_initial_configuration.cols(); node_ind++ )
                {
                    // for every node on one side of the mirror line, find the closest match on the other side
                    // Note that nodes that have an x value > than mid_x are on the manual gripper side
                    if ( object_initial_configuration( 0, node_ind ) > point_reflector.get_mid_x() )
                    {
                        long mirror_ind = closestPointInSet( object_initial_configuration,
                                point_reflector.reflect( object_initial_configuration.block< 3, 1 >( 0, node_ind ) ) );

                        mirror_map[ node_ind ] = mirror_ind;
                    }
                }

                return mirror_map;
            }
    };
}

#endif // task_h
