#ifndef COLLISION_HELPERS_HPP
#define COLLISION_HELPERS_HPP

#include <arc_utilities/eigen_helpers.hpp>
#include "smmap/gripper_helpers.hpp"

namespace smmap
{
    /// Stores the result of a collision check from bullet (or similar)
    struct CollisionData
    {
        public:
            CollisionData( const Eigen::Vector3d& nearest_point_to_obstacle,
                           const Eigen::Vector3d& obstacle_surface_normal,
                           const double distance_to_obstacle )
                : nearest_point_to_obstacle_( nearest_point_to_obstacle )
                , obstacle_surface_normal_( obstacle_surface_normal )
                , distance_to_obstacle_( distance_to_obstacle )
            {}

            CollisionData( Eigen::Vector3d&& nearest_point_to_obstacle,
                           Eigen::Vector3d&& obstacle_surface_normal,
                           const double distance_to_obstacle )
                : nearest_point_to_obstacle_( nearest_point_to_obstacle )
                , obstacle_surface_normal_( obstacle_surface_normal )
                , distance_to_obstacle_( distance_to_obstacle )
            {}

            Eigen::Vector3d nearest_point_to_obstacle_;
            Eigen::Vector3d obstacle_surface_normal_;
            double distance_to_obstacle_;
    };

    /// Stores the result of a collision avoidance calculation for a single gripper
    struct CollisionAvoidanceResult
    {
        public:
            CollisionAvoidanceResult()
                : nullspace_projector( Eigen::Matrix< double, 6, 6 >::Identity() )
                , velocity( Eigen::Matrix< double, 6, 1 >::Zero() )
                , distance( std::numeric_limits< double >::infinity() )
            {}
            Eigen::Matrix< double, 6, 6 > nullspace_projector;
            Eigen::Matrix< double, 6, 1 > velocity;
            double distance;
    };

    /**
     * @brief ComputeCollisionToGripperJacobian
     * @param point_on_gripper
     * @param gripper_pose
     * @return
     */
    inline Eigen::Matrix< double, 3, 6 > ComputeCollisionToGripperJacobian(
            const Eigen::Vector3d& point_on_gripper,
            const Eigen::Affine3d& gripper_pose )
    {

        Eigen::Matrix< double, 3, 6 > J_collision;
        const Eigen::Matrix3d gripper_rot = gripper_pose.rotation();

        // Translation - if I move the gripper along its x/y/z-axis, what happens to the given point?
        J_collision.block< 3, 3 >( 0, 0 ) = gripper_rot;

        const Eigen::Vector3d gripper_to_point_in_collision =
                point_on_gripper - gripper_pose.translation();

        // If I rotate the gripper about its x/y/z-axis, what happens to the point in question?
        J_collision.block< 3, 1 >( 0, 3 ) = gripper_rot.block< 3, 1 >( 0, 0 ).cross( gripper_to_point_in_collision );
        J_collision.block< 3, 1 >( 0, 4 ) = gripper_rot.block< 3, 1 >( 0, 1 ).cross( gripper_to_point_in_collision );
        J_collision.block< 3, 1 >( 0, 5 ) = gripper_rot.block< 3, 1 >( 0, 2 ).cross( gripper_to_point_in_collision );

        return J_collision;
    }

    /**
     * @brief ComputeGripperObjectAvoidance
     * @param collision_data
     * @param gripper_pose
     * @param max_step_size
     * @return
     */
    inline CollisionAvoidanceResult ComputeGripperObjectAvoidance(
            const CollisionData& collision_data,
            const Eigen::Affine3d& gripper_pose,
            double max_step_size )
    {
        CollisionAvoidanceResult collision_avoidance_result;

        collision_avoidance_result.distance = collision_data.distance_to_obstacle_;

        // If we have a collision to avoid, then find the vector
        if ( !std::isinf( collision_data.distance_to_obstacle_ ) )
        {
            // Create the collision Jacobian
            const Eigen::Matrix< double, 3, 6 > J_collision =
                    ComputeCollisionToGripperJacobian(
                        collision_data.nearest_point_to_obstacle_, gripper_pose );
            const Eigen::Matrix< double, 6, 3 > J_collision_inv =
                    EigenHelpers::Pinv( J_collision, EigenHelpers::SuggestedRcond() );

            // Create the collision avoidance vector to follow
            const Eigen::Vector3d& avoid_collision_delta = collision_data.obstacle_surface_normal_;

            collision_avoidance_result.velocity =  J_collision_inv * avoid_collision_delta;
            collision_avoidance_result.velocity *=
                    max_step_size / GripperVelocity6dNorm( collision_avoidance_result.velocity );

            collision_avoidance_result.nullspace_projector =
                    Eigen::Matrix< double, 6, 6 >::Identity() - J_collision_inv * J_collision;
        }
        // Otherwise, leave the collision result as the default "no collision" state
        else {}

        return collision_avoidance_result;
    }

    inline std::vector< CollisionAvoidanceResult > ComputeGripperObjectAvoidance(
            const std::vector< CollisionData >& collision_data,
            const EigenHelpers::VectorAffine3d& gripper_pose,
            double max_step_size )
    {
        std::vector< CollisionAvoidanceResult > collision_avoidance_results;
        collision_avoidance_results.reserve( collision_data.size() );

        for ( size_t gripper_ind = 0; gripper_ind < gripper_pose.size(); gripper_ind++ )
        {
            collision_avoidance_results.push_back(
                        ComputeGripperObjectAvoidance(
                                    collision_data[gripper_ind],
                                    gripper_pose[gripper_ind],
                                    max_step_size ) );
        }

        return collision_avoidance_results;
    }
}

#endif // COLLISION_HELPERS_HPP
