#ifndef TASK_SPECIFICATION_IMPLEMENTIONS_HPP
#define TASK_SPECIFICATION_IMPLEMENTIONS_HPP

#include "smmap/task_specification.h"
#include "smmap/point_reflector.hpp"
#include "smmap/ros_communication_helpers.hpp"

namespace smmap
{
    /**
     * @brief The RopeCoverage class
     */
    class RopeCoverage : public TaskSpecification
    {
        public:
            RopeCoverage( ros::NodeHandle& nh )
                : cover_points_( GetCoverPoints( nh ) )
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 0.5*20; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return  10*20; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.1/20; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 15.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color ) const
            {
                vis.visualizeRope( marker_name, object_configuration, color );
            }

            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration ) const
            {
                Eigen::VectorXd error( cover_points_.cols() );

                // for every cover point, find the nearest deformable object point
                #pragma omp parallel for
                for ( long cover_ind = 0; cover_ind < cover_points_.cols(); cover_ind++ )
                {
                    const Eigen::Vector3d& cover_point = cover_points_.block< 3, 1 >( 0, cover_ind );

                    // find the closest deformable object point
                    double min_dist_squared = std::numeric_limits< double >::infinity();
                    for ( long rope_ind = 0; rope_ind < current_configuration.cols(); rope_ind++ )
                    {
                        const Eigen::Vector3d& rope_point = current_configuration.block< 3, 1 >( 0, rope_ind );
                        const double new_dist_squared = ( cover_point - rope_point ).squaredNorm();
                        min_dist_squared = std::min( new_dist_squared, min_dist_squared );
                    }

                    if ( std::sqrt( min_dist_squared ) >= 0.2/20. )
                    {
                        error( cover_ind ) = std::sqrt( min_dist_squared );
                    }
                    else
                    {
                        error( cover_ind ) = 0;
                    }
                }

                return error.sum();
            }

            virtual std::pair< Eigen::VectorXd, Eigen::VectorXd > calculateObjectDesiredDelta_impl(
                    const WorldState& world_state ) const
            {
                ROS_INFO_NAMED( "rope_coverage_task" , "Finding 'best' object delta" );
                const ObjectPointSet& object_configuration = world_state.object_configuration_;

                std::pair< Eigen::VectorXd, Eigen::VectorXd > desired_rope_delta =
                        std::make_pair( Eigen::VectorXd::Zero( object_configuration.cols() * 3 ),
                                        Eigen::VectorXd::Zero( object_configuration.cols() * 3 ) );

                // for every cover point, find the nearest deformable object point
                for ( long cover_ind = 0; cover_ind < cover_points_.cols(); cover_ind++ )
                {
                    const Eigen::Vector3d& cover_point = cover_points_.block< 3, 1 >( 0, cover_ind );

                    // find the closest deformable object point
                    long min_ind = -1;
                    double min_dist_squared = std::numeric_limits< double >::infinity();
                    // Note that this cannot be done in parallel (without locks) due to the desired_velocity object
                    for ( long rope_ind = 0; rope_ind < object_configuration.cols(); rope_ind++ )
                    {
                        const Eigen::Vector3d& rope_point = object_configuration.block< 3, 1 >( 0, rope_ind );
                        const double new_dist_squared = ( cover_point - rope_point ).squaredNorm();
                        if ( new_dist_squared < min_dist_squared )
                        {
                            min_dist_squared = new_dist_squared;
                            min_ind = rope_ind;
                        }
                    }

                    if ( std::sqrt( min_dist_squared ) >= 0.2/20. )
                    {
                        desired_rope_delta.first.segment< 3 >( min_ind * 3 ) =
                                desired_rope_delta.first.segment< 3 >( min_ind * 3 )
                                + ( cover_point - object_configuration.block< 3, 1 >( 0, min_ind ) );

                        desired_rope_delta.second( min_ind * 3 ) += 1.0;
                        desired_rope_delta.second( min_ind * 3 + 1 ) += 1.0;
                        desired_rope_delta.second( min_ind * 3 + 2 ) += 1.0;
                    }
                }

                return desired_rope_delta;
            }

        private:
            /// Stores the points that we are trying to cover with the rope
            const ObjectPointSet cover_points_;
    };

    /**
     * @brief The ClothTableCoverage class
     */
    class ClothTableCoverage : public TaskSpecification
    {
        public:
            ClothTableCoverage( ros::NodeHandle& nh )
                : cover_points_( GetCoverPoints( nh ) )
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 0.7*20; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return  100*20; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.1/20; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 4.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color ) const
            {
                vis.visualizeCloth( marker_name, object_configuration, color );
            }

            virtual double calculateError_impl(
                    const ObjectPointSet &current_configuration ) const
            {
                // for every cover point, find the nearest deformable object point
                Eigen::VectorXd error( cover_points_.cols() );
                #pragma omp parallel for
                for ( long cover_ind = 0; cover_ind < cover_points_.cols(); cover_ind++ )
                {
                    const Eigen::Vector3d& cover_point = cover_points_.block< 3, 1 >( 0, cover_ind );

                    double min_dist_squared = std::numeric_limits< double >::infinity();
                    for ( long cloth_ind = 0; cloth_ind < current_configuration.cols(); cloth_ind++ )
                    {
                        const Eigen::Vector3d& cloth_point = current_configuration.block< 3, 1 >( 0, cloth_ind );
                        const double new_dist_squared = ( cover_point - cloth_point ).squaredNorm();
                        min_dist_squared = std::min( new_dist_squared, min_dist_squared );
                    }

                    if ( std::sqrt( min_dist_squared ) > 0.04/20.0 )
                    {
                        error( cover_ind ) = std::sqrt( min_dist_squared );
                    }
                    else
                    {
                        error( cover_ind ) = 0;
                    }
                }

                return error.sum();
            }

            virtual std::pair< Eigen::VectorXd, Eigen::VectorXd > calculateObjectDesiredDelta_impl(
                    const WorldState& world_state ) const
            {
                ROS_INFO_NAMED( "cloth_table_coverage" , "Finding 'best' cloth delta" );

                const ObjectPointSet& object_configuration = world_state.object_configuration_;

                std::pair< Eigen::VectorXd, Eigen::VectorXd > desired_cloth_delta =
                        std::make_pair( Eigen::VectorXd::Zero( object_configuration.cols() * 3 ),
                                        Eigen::VectorXd::Zero( object_configuration.cols() * 3 ) );

                // for every cover point, find the nearest deformable object point
                for ( long cover_ind = 0; cover_ind < cover_points_.cols(); cover_ind++ )
                {
                    const Eigen::Vector3d& cover_point = cover_points_.block< 3, 1 >( 0, cover_ind );

                    // find the closest deformable object point
                    long min_ind = -1;
                    double min_dist_squared = std::numeric_limits< double >::infinity();
                    for ( long cloth_ind = 0; cloth_ind < object_configuration.cols(); cloth_ind++ )
                    {
                        const Eigen::Vector3d& cloth_point = object_configuration.block< 3, 1 >( 0, cloth_ind );
                        const double new_dist_squared = ( cover_point - cloth_point ).squaredNorm();
                        if ( new_dist_squared < min_dist_squared )
                        {
                            min_dist_squared = new_dist_squared;
                            min_ind = cloth_ind;
                        }
                    }

                    if ( std::sqrt( min_dist_squared ) > 0.04/20.0 )
                    {
                        desired_cloth_delta.first.segment< 3 >( min_ind * 3 ) =
                                desired_cloth_delta.first.segment< 3 >( min_ind * 3 )
                                + ( cover_point - object_configuration.block< 3, 1 >( 0, min_ind ) );

                        desired_cloth_delta.second( min_ind * 3 ) += 1.0;
                        desired_cloth_delta.second( min_ind * 3 + 1 ) += 1.0;
                        desired_cloth_delta.second( min_ind * 3 + 2 ) += 1.0;
                    }
                }

                return desired_cloth_delta;
            }

        private:
            /// Stores the points that we are trying to cover with the cloth
            const ObjectPointSet cover_points_;
    };

    /**
     * @brief The ClothColabFolding class
     */
    class ClothColabFolding : public TaskSpecification
    {
        public:
            ClothColabFolding( ros::NodeHandle& nh )
                : point_reflector_( createPointReflector( nh ) )
                , mirror_map_( createMirrorMap( nh, point_reflector_ ) )
            {}

        private:
            virtual double getDeformability_impl() const
            {
                return 0.7*20; // k
            }

            virtual double getCollisionScalingFactor_impl() const
            {
                return  100*20; // beta
            }

            virtual double getStretchingScalingThreshold_impl() const
            {
                return 0.1/20; // lambda
            }

            virtual double maxTime_impl() const
            {
                return 6.0;
            }

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color ) const
            {
                vis.visualizeCloth( marker_name, object_configuration, color );
            }

            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration ) const
            {
                double error = 0;

                for ( std::map< long, long >::const_iterator ittr = mirror_map_.begin(); ittr != mirror_map_.end(); ittr++ )
                {
                    error += ( current_configuration.block< 3, 1 >( 0, ittr->second ) -
                               point_reflector_.reflect( current_configuration.block< 3, 1 >( 0, ittr->first ) ) ).norm();
                }

                return error;
            }

            virtual std::pair< Eigen::VectorXd, Eigen::VectorXd > calculateObjectDesiredDelta_impl(
                    const WorldState& world_state ) const
            {
                ROS_INFO_NAMED( "cloth_colab_folding" , "Finding 'best' cloth delta" );
                const ObjectPointSet& object_configuration = world_state.object_configuration_;

                std::pair< Eigen::VectorXd, Eigen::VectorXd > desired_cloth_delta =
                        std::make_pair( Eigen::VectorXd::Zero( object_configuration.cols() * 3 ),
                                        Eigen::VectorXd::Ones( object_configuration.cols() * 3 ) );

                long robot_cloth_points_ind = 0;
                for ( std::map< long, long >::const_iterator ittr = mirror_map_.begin();
                      ittr != mirror_map_.end(); ittr++, robot_cloth_points_ind++ )
                {
                    desired_cloth_delta.first.segment< 3 >( ittr->second * 3) =
                            point_reflector_.reflect( object_configuration.block< 3, 1 >( 0, ittr->first ) )
                            - object_configuration.block< 3, 1 >( 0, ittr->second );
                }

                return desired_cloth_delta;
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
            static std::map< long, long > createMirrorMap(
                    ros::NodeHandle& nh,
                    const PointReflector& point_reflector )
            {
                ObjectPointSet object_initial_configuration =
                    GetObjectInitialConfiguration( nh );

                ROS_INFO_NAMED( "cloth_colab_folding", "Finding point correspondences" );

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

#endif // TASK_SPECIFICATION_IMPLEMENTIONS_HPP
