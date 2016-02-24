#ifndef TASK_H
#define TASK_H

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <ros/ros.h>
#include <smmap_msgs/messages.h>

#include "smmap/task_enums.h"
#include "smmap/task_function_pointer_types.h"
#include "smmap/point_reflector.hpp"
#include "smmap/model_set.h"
#include "smmap/visualization_tools.h"

/*
Eigen::VectorXd DiminishingRigidityModel::computeStretchingCorrection(
        const ObjectPointSet& object_current_configuration ) const
{
    Eigen::VectorXd stretching_correction = Eigen::VectorXd::Zero( object_current_configuration.cols() * 3 );

    Eigen::MatrixXd node_distance_delta =
            distanceMatrix( object_current_configuration )
            - object_initial_node_distance_;

    for ( long first_node = 0; first_node < node_distance_delta.rows(); first_node++)
    {
        for ( long second_node = first_node + 1; second_node < node_distance_delta.cols(); second_node++)
        {
            if ( node_distance_delta( first_node, second_node ) > stretching_correction_threshold_ )
            {
                // The correction vector points from the first node to the second node,
                // and is half the length of the "extra" distance
                Eigen::Vector3d correction_vector = 0.5 * node_distance_delta( first_node, second_node )
                        * ( object_current_configuration.block< 3, 1 >( 0, second_node )
                            - object_current_configuration.block< 3, 1 >( 0, first_node ) );

                stretching_correction.segment< 3 >( 3 * first_node ) += correction_vector;
                stretching_correction.segment< 3 >( 3 * second_node ) -= correction_vector;
            }
        }
    }

    return stretching_correction;
}
*/

namespace smmap
{
    class Task
    {
        public:
            Task( ros::NodeHandle& nh );
            void execute();

        private:
            ////////////////////////////////////////////////////////////////////
            //
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            Visualizer vis_;

            const TaskType task_type_;
            const DeformableType deformable_type_;

            ////////////////////////////////////////////////////////////////////
            // Function pointers that are created in the construtor that are
            // then passed on to the models or the planner
            ////////////////////////////////////////////////////////////////////

            const ErrorFunctionType error_fn_;
            const ModelPredictionFunctionType model_prediction_fn_;
            const ModelSuggestedGrippersTrajFunctionType model_suggested_grippers_traj_fn_;
            const ModelGetUtilityFunctionType model_get_utility_fn_;
            const ModelUpdateUtilityFunctionType model_utility_update_fn_;
            const GripperCollisionCheckFunctionType gripper_collision_check_fn_;
            const TaskDesiredObjectDeltaFunctionType task_desired_object_delta_fn_;

            ////////////////////////////////////////////////////////////////////
            // Functions that are used to initialize function pointers in the
            // constructor. These all require that task_type_ and
            // deformable_type_ have been set already
            ////////////////////////////////////////////////////////////////////

            ErrorFunctionType createErrorFunction();
            ModelPredictionFunctionType createModelPredictionFunction();
            ModelSuggestedGrippersTrajFunctionType createModelSuggestedGrippersTrajFunction();
            ModelGetUtilityFunctionType createModelGetUtilityFunction();
            ModelUpdateUtilityFunctionType createModelUtilityUpdateFunction();
            GripperCollisionCheckFunctionType createGripperCollisionCheckFunction();
            TaskDesiredObjectDeltaFunctionType createTaskDesiredObjectDeltaFunction();

            ////////////////////////////////////////////////////////////////////
            // Task specific functions that get bound as needed
            ////////////////////////////////////////////////////////////////////

            double calculateRopeCoverageError( const ObjectPointSet& current_configuration ) const;
            double calculateClothCoverageError( const ObjectPointSet& current_configuration ) const;
            double calculateClothColabFoldingError( const ObjectPointSet& current_configuration ) const;

            std::pair< Eigen::VectorXd, Eigen::MatrixXd > calculateRopeCoverageDesiredDelta(
                    const WorldState& world_state );
            std::pair< Eigen::VectorXd, Eigen::MatrixXd > calculateClothCoverageDesiredDelta(
                    const WorldState& world_state );
            std::pair< Eigen::VectorXd, Eigen::MatrixXd > calculateClothColabFoldingDesiredDelta(
                    const WorldState& world_state );

            ////////////////////////////////////////////////////////////////////
            // ModelSet management
            ////////////////////////////////////////////////////////////////////

            ModelSet model_set_;
    };
}

/*
namespace blarg
{
    class Task_old
    {
        public:
            // TODO: This is closer to a way of finding a error gradient
            // for some of the methods. Fix this naming problem/usage problem.
            virtual Eigen::VectorXd calculateObjectDesiredVelocity(
                    const ObjectPointSet& current_configuration ) const = 0;

            double calculateError(
                    const ObjectPointSet& current_configuration ) const
            {
                return calculateError_impl( current_configuration );
            }


//            ObjectPointSet getObjectErrorGradient(
//                    ObjectPointSet current_configuration ) const
//            {
//                return getObjectErrorGradient_impl( current_configuration );
//            }


            virtual double getDeformability() const = 0;                // k
            virtual double getCollisionScalingFactor() const = 0;       // beta (or k2)
            virtual double getStretchingScalingThreshold() const = 0;   // lambda
            virtual double maxTime() const = 0;                         // max simulation time when scripting things

            void visualizePredictions(
                    const VectorObjectTrajectory& model_predictions,
                    size_t best_traj ) const
            {
                visualizePredictions_impl( model_predictions, best_traj );
            }

        protected:
            mutable Visualizer vis_;

        private:
            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration ) const = 0;

            virtual void visualizePredictions_impl(
                    const VectorObjectTrajectory& model_predictions,
                    size_t best_traj ) const = 0;
    };

    class RopeCoverage : public Task
    {
        public:
            RopeCoverage( ros::NodeHandle& nh )
                : Task( nh )
                , cover_points_( getCoverPointsHelper( nh ) )
            {}

            virtual Eigen::VectorXd calculateObjectDesiredVelocity(
                    const ObjectPointSet& current_configuration ) const
            {
                ROS_INFO_NAMED( "rope_coverage_task" , "Finding 'best' configuration" );

                Eigen::VectorXd desired_velocity =
                        Eigen::VectorXd::Zero( current_configuration.cols() * 3 );

                // for every cover point, find the nearest deformable object point
                for ( long cover_ind = 0; cover_ind < cover_points_.cols(); cover_ind++ )
                {
                    const Eigen::Vector3d& cover_point = cover_points_.block< 3, 1 >( 0, cover_ind );

                    // find the closest deformable object point
                    long min_ind = -1;
                    double min_dist_squared = std::numeric_limits< double >::infinity();
                    // Note that this cannot be done in parallel (without locks) due to the desired_velocity object
                    for ( long rope_ind = 0; rope_ind < current_configuration.cols(); rope_ind++ )
                    {
                        const Eigen::Vector3d& rope_point = current_configuration.block< 3, 1 >( 0, rope_ind );
                        const double new_dist_squared = ( cover_point - rope_point ).squaredNorm();
                        if ( new_dist_squared < min_dist_squared )
                        {
                            min_dist_squared = new_dist_squared;
                            min_ind = rope_ind;
                        }
                    }

                    if ( std::sqrt( min_dist_squared ) >= 0.2/20. )
                    {
                        desired_velocity.segment< 3 >( min_ind * 3 ) =
                                desired_velocity.segment< 3 >( min_ind * 3 )
                                + ( cover_point - current_configuration.block< 3, 1 >( 0, min_ind ) );
                    }
                }

                return desired_velocity;
            }

            virtual double getDeformability() const
            {
                return 0.5*20; // k
            }

            virtual double getCollisionScalingFactor() const
            {
                return  10*20; // beta
            }

            virtual double getStretchingScalingThreshold() const
            {
                return 0.1/20; // lambda
            }

            virtual bool getUseRotation() const
            {
                return true;
            }

            virtual double maxTime() const
            {
                return 15.0;
            }

        private:
            virtual void visualizePredictions_impl(
                    const VectorObjectTrajectory& model_predictions,
                    size_t best_traj ) const
            {
                std_msgs::ColorRGBA color;
                color.r = 1;
                color.g = 1;
                color.b = 0;
                color.a = 1;

                vis_.visualizeRope( model_predictions[best_traj].back(), color, "rope_predicted" );
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

            virtual Eigen::VectorXd calculateObjectDesiredVelocity(
                    const ObjectPointSet& current_configuration ) const
            {
                ROS_INFO_NAMED( "cloth_colab_folding_task" , "Finding 'best' configuration" );

                Eigen::VectorXd desired_velocity =
                        Eigen::VectorXd::Zero( current_configuration.cols() * 3 );

                ObjectPointSet robot_cloth_points_desired = ObjectPointSet::Zero( 3, (long)mirror_map_.size() );
//                ObjectPointSet robot_cloth_points_current( 3, (long)mirror_map_.size() );
//                std::vector< std_msgs::ColorRGBA > robot_cloth_points_current_colors( mirror_map_.size() );

                std_msgs::ColorRGBA red;

                red.r = 1;
                red.g = 0;
                red.b = 0;
                red.a = 1;

                long robot_cloth_points_ind = 0;
                for ( std::map< long, long >::const_iterator ittr = mirror_map_.begin();
                      ittr != mirror_map_.end(); ittr++, robot_cloth_points_ind++ )
                {
                    desired_velocity.segment< 3 >( ittr->second * 3) =
                            point_reflector_.reflect( current_configuration.block< 3, 1 >( 0, ittr->first ) )
                            - current_configuration.block< 3, 1 >( 0, ittr->second );

                    robot_cloth_points_desired.block< 3, 1 >( 0, robot_cloth_points_ind ) =
                            point_reflector_.reflect( current_configuration.block< 3, 1 >( 0, ittr->first ) );
//                    robot_cloth_points_current.block< 3, 1 >( 0, robot_cloth_points_ind ) = current_configuration.block< 3, 1 >( 0, ittr->second );

//                    std_msgs::ColorRGBA color;

//                    color.r = 0;
//                    color.g = (float)( robot_cloth_points_desired.block< 3, 1 >( 0, robot_cloth_points_ind )
//                               - robot_cloth_points_current.block< 3, 1 >( 0, robot_cloth_points_ind ) ).norm() * 20;
//                    color.b = 0;
//                    color.a = 1;

//                    robot_cloth_points_current_colors[(size_t)robot_cloth_points_ind] = color;
                }

//                visualizeCloth( robot_cloth_points_desired, red, "cloth_desired" );
//                visualizeCloth( robot_cloth_points_current, robot_cloth_points_current_colors, "cloth_current" );

                return desired_velocity;
            }

            virtual double getDeformability() const
            {
                return 0.7*20; // k
            }

            virtual double getCollisionScalingFactor() const
            {
                return  100*20; // beta
            }

            virtual double getStretchingScalingThreshold() const
            {
                return 0.1/20; // lambda
            }

            virtual bool getUseRotation() const
            {
                return true;
            }

            virtual double maxTime() const
            {
                return 6.0;
            }

        private:
            virtual void visualizePredictions_impl(
                    const VectorObjectTrajectory& model_predictions,
                    size_t best_traj ) const
            {
                std_msgs::ColorRGBA color;
                color.r = 1;
                color.g = 1;
                color.b = 0;
                color.a = 1;

                vis_.visualizeCloth( model_predictions[best_traj].back(), color, "cloth_predicted" );
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

    class ClothTableCoverage : public Task
    {
        public:
            ClothTableCoverage( ros::NodeHandle& nh )
                : Task( nh )
                , cover_points_( getCoverPointsHelper( nh ) )
            {

            }

            virtual Eigen::VectorXd calculateObjectDesiredVelocity(
                    const ObjectPointSet &current_configuration ) const
            {
                ROS_INFO_NAMED( "cloth_table_coverage_task" , "Finding 'best' configuration" );

                Eigen::VectorXd desired_velocity =
                        Eigen::VectorXd::Zero( current_configuration.cols() * 3 );

                std_msgs::ColorRGBA red;
                {
                    red.r = 1;
                    red.g = 0;
                    red.b = 0;
                    red.a = 1;
                }

                // for every cover point, find the nearest deformable object point
                for ( long cover_ind = 0; cover_ind < cover_points_.cols(); cover_ind++ )
                {
                    const Eigen::Vector3d& cover_point = cover_points_.block< 3, 1 >( 0, cover_ind );

                    // find the closest deformable object point
                    long min_ind = -1;
                    double min_dist_squared = std::numeric_limits< double >::infinity();
                    for ( long cloth_ind = 0; cloth_ind < current_configuration.cols(); cloth_ind++ )
                    {
                        const Eigen::Vector3d& cloth_point = current_configuration.block< 3, 1 >( 0, cloth_ind );
                        const double new_dist_squared = ( cover_point - cloth_point ).squaredNorm();
                        if ( new_dist_squared < min_dist_squared )
                        {
                            min_dist_squared = new_dist_squared;
                            min_ind = cloth_ind;
                        }
                    }

                    if ( std::sqrt( min_dist_squared ) > 0.04/20.0 )
                    {
                        desired_velocity.segment< 3 >( min_ind * 3 ) =
                                desired_velocity.segment< 3 >( min_ind * 3 )
                                + ( cover_point - current_configuration.block< 3, 1 >( 0, min_ind ) );
                    }

                }

                Eigen::MatrixXd cloth_delta = desired_velocity;
                cloth_delta.resizeLike( current_configuration );

//                visualizeCloth( current_configuration + cloth_delta, red, "desired_delta" );

                return desired_velocity;
            }

            virtual double getDeformability() const
            {
                return 0.7*20; // k
            }

            virtual double getCollisionScalingFactor() const
            {
                return  100*20; // beta
            }

            virtual double getStretchingScalingThreshold() const
            {
                return 0.1/20; // lambda
            }

            virtual bool getUseRotation() const
            {
                return true;
            }

            virtual double maxTime() const
            {
                return 4.0;
            }

        private:
            virtual void visualizePredictions_impl(
                    const VectorObjectTrajectory& model_predictions,
                    size_t best_traj ) const
            {
                std_msgs::ColorRGBA color;
                color.r = 1;
                color.g = 1;
                color.b = 0;
                color.a = 1;

                vis_.visualizeCloth( model_predictions[best_traj].back(), color, "cloth_predicted" );
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

        private:
            /// Stores the points that we are trying to cover with the cloth
            const ObjectPointSet cover_points_;
            /// Constructor helper that allows cover_points_ to be const
            ObjectPointSet getCoverPointsHelper( ros::NodeHandle& nh )
            {
                ROS_INFO_NAMED( "cloth_table_coverage_task" , "Getting cover points" );

                // Get the initial configuration of the object
                ros::ServiceClient cover_points_client =
                    nh.serviceClient< smmap_msgs::GetPointSet >( GetCoverPointsTopic( nh ) );

                cover_points_client.waitForExistence();

                smmap_msgs::GetPointSet srv_data;
                cover_points_client.call( srv_data );
                ObjectPointSet cover_points =
                    EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd( srv_data.response.points );

                ROS_INFO_NAMED( "cloth_table_coverage_task" , "Number of cover points: %zu", srv_data.response.points.size() );

                return cover_points;
            }
    };
}
*/

#endif // TASK_H
