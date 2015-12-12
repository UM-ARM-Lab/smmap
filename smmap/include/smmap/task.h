#ifndef task_h
#define task_h

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <ros/ros.h>
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

            ObjectPointSet findObjectDesiredConfiguration( const ObjectPointSet& current_configuration )
            {
                return doFindObjectDesiredConfiguration( current_configuration );
            }

            double calculateError( const ObjectPointSet& current_configuration )
            {
                return doCalculateError( current_configuration );
            }

        private:
            virtual ObjectPointSet doFindObjectDesiredConfiguration( const ObjectPointSet& current_configuration ) = 0;
            virtual double doCalculateError( const ObjectPointSet& current_configuration ) = 0;

    };

    class RopeCoverage : public Task
    {
        public:
            RopeCoverage( ros::NodeHandle& nh )
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
    //                            desired_configuration.block< 3, 1 >( 0, min_ind ) +
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
                return -1;
            }

            ObjectPointSet cover_points_;
    };

    class ClothColabFolding : public Task
    {
        public:
            ClothColabFolding( ros::NodeHandle& nh )
            {
                ROS_INFO_NAMED( "cloth_colab_folding_task" , "Getting mirror line" );

                // Get the initial configuration of the object
                ros::ServiceClient mirror_line_client =
                    nh.serviceClient< smmap_msgs::GetMirrorLine >( GetMirrorLineTopic( nh ) );

                mirror_line_client.waitForExistence();

                smmap_msgs::GetMirrorLine srv_data;
                mirror_line_client.call( srv_data );

                point_reflector_ = PointReflector( srv_data.response.mid_x,
                                                   srv_data.response.min_y,
                                                   srv_data.response.max_y );
            }

        private:
            ObjectPointSet doFindObjectDesiredConfiguration( const ObjectPointSet& current_configuration )
            {

            }

            double doCalculateError( const ObjectPointSet &current )
            {
                return -1;
            }

            PointReflector point_reflector_;

    };
}

#endif // task_h
