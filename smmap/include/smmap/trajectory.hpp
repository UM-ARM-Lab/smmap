#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <kinematics_toolbox/kinematics.h>
#include <smmap_msgs/messages.h>

#include "smmap/collision_helpers.hpp"

namespace smmap
{
    // Typedefs to help deal with Deformable Object and Gripper tracking
    typedef Eigen::Matrix3Xd ObjectPointSet;
    typedef std::vector< ObjectPointSet > ObjectTrajectory;
    typedef std::vector< ObjectTrajectory > VectorObjectTrajectory;

    typedef EigenHelpers::VectorAffine3d AllGrippersSinglePose;
    typedef std::vector< AllGrippersSinglePose > AllGrippersPoseTrajectory;

    typedef kinematics::VectorVector6d AllGrippersSingleVelocity;
    typedef AllGrippersSingleVelocity AllGrippersSinglePoseDelta;
    typedef std::vector< AllGrippersSingleVelocity > AllGrippersVelocityTrajectory;
    typedef std::vector< AllGrippersSinglePoseDelta > AllGrippersPoseDeltaTrajectory;

    /// World state structure for a single time step
    struct WorldState
    {
        ObjectPointSet object_configuration_;
        AllGrippersSinglePose all_grippers_single_pose_;
        std::vector< CollisionData > gripper_collision_data_;
        double sim_time_;
    };

    /**
     * @brief computeNextFeedback
     * @param next_feedback_ros
     * @return
     */
    inline WorldState ConvertToEigenFeedback(
            const smmap_msgs::SimulatorFeedback& feedback_ros )
    {
        WorldState feedback_eigen;

        feedback_eigen.object_configuration_ =
                EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd(
                    feedback_ros.object_configuration );

        feedback_eigen.all_grippers_single_pose_ =
                EigenHelpersConversions::VectorGeometryPoseToVectorAffine3d(
                    feedback_ros.gripper_poses );

        // TODO: modify the feedback to change this data ordering
        size_t num_grippers = feedback_ros.gripper_poses.size();
        feedback_eigen.gripper_collision_data_.reserve( num_grippers );
        for ( size_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++ )
        {
            feedback_eigen.gripper_collision_data_.push_back(
                        CollisionData(
                            EigenHelpersConversions::GeometryPointToEigenVector3d(
                                feedback_ros.gripper_nearest_point_to_obstacle[gripper_ind] ),
                            EigenHelpersConversions::GeometryVector3ToEigenVector3d(
                                feedback_ros.obstacle_surface_normal[gripper_ind] ),
                            feedback_ros.gripper_distance_to_obstacle[gripper_ind] ) );
        }

        feedback_eigen.sim_time_ = feedback_ros.sim_time;

        return feedback_eigen;
    }

    /**
     * @brief getGripperTrajectories
     * @param feedback
     * @return
     */
    inline AllGrippersPoseTrajectory GetGripperTrajectories(
            const std::vector< WorldState >& feedback )
    {
        AllGrippersPoseTrajectory grippers_trajectories( feedback.size() );

        for ( size_t time_ind = 0; time_ind < feedback.size(); time_ind++ )
        {
            grippers_trajectories[time_ind] =
                    feedback[time_ind].all_grippers_single_pose_;
        }

        return grippers_trajectories;
    }

    /**
     * @brief CalculateGrippersPoseDeltas
     * @param grippers_trajectory
     * @return
     */
    inline AllGrippersPoseDeltaTrajectory CalculateGrippersPoseDeltas(
            const AllGrippersPoseTrajectory& grippers_trajectory )
    {
        assert( grippers_trajectory.size() > 1 );
        const size_t num_grippers = grippers_trajectory[0].size();

        AllGrippersPoseDeltaTrajectory grippers_pose_delta_traj(
                    grippers_trajectory.size() - 1,
                    AllGrippersSingleVelocity( num_grippers ) );

        for ( size_t time_ind = 0; time_ind < grippers_pose_delta_traj.size(); time_ind++ )
        {
            for ( size_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++ )
            {
                grippers_pose_delta_traj[time_ind][gripper_ind] =
                        kinematics::calculateError(
                            grippers_trajectory[time_ind][gripper_ind],
                            grippers_trajectory[time_ind + 1][gripper_ind] );
            }
        }

        return grippers_pose_delta_traj;
    }

    /**
     * @brief CalculateGrippersTrajectory
     * @param grippers_initial_pose
     * @param grippers_velocities
     * @param dt
     * @return
     */
    inline AllGrippersPoseTrajectory CalculateGrippersTrajectory(
            const AllGrippersSinglePose& grippers_initial_pose,
            const AllGrippersVelocityTrajectory& grippers_velocities,
            const double dt )
    {
        const size_t num_grippers = grippers_initial_pose.size();

        AllGrippersPoseTrajectory grippers_pose_trajectory(
                    grippers_velocities.size() + 1,
                    AllGrippersSinglePose( num_grippers ) );

        grippers_pose_trajectory[0] = grippers_initial_pose;

        for ( size_t time_ind = 0; time_ind < grippers_velocities.size(); time_ind++ )
        {
            for ( size_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind ++ )
            {
                grippers_pose_trajectory[time_ind+1][gripper_ind] =
                        grippers_pose_trajectory[time_ind][gripper_ind] *
                        kinematics::expTwistAffine3d( grippers_velocities[time_ind][gripper_ind], dt );
            }
        }

        return grippers_pose_trajectory;
    }

    /**
     * @brief Calculates the distance bewtween two deformable object
     * configurations using an L2 norm.
     *
     * @param set1 The first object configuration
     * @param set2 The second object configuration
     *
     * @return The L2 distanace between object configurations
     */
    inline double distance( const ObjectPointSet& set1,
                            const ObjectPointSet& set2 )
    {
        return ( set1 - set2 ).norm();
    }

    /**
     * @brief Calculates the squared distance bewtween two deformable object
     * configurations using an L2 norm.
     *
     * @param set1 The first object configuration
     * @param set2 The second object configuration
     *
     * @return The squared L2 distanace between object configurations
     */
    inline double distanceSquared( const ObjectPointSet& set1,
                                   const ObjectPointSet& set2 )
    {
        return ( set1 - set2 ).squaredNorm();
    }

    /**
     * @brief Calculates the RMS distance between two object trajectories using
     * an L2 norm for each time step.
     *
     * @param traj1 The first object trajectory
     * @param traj2 The second object trajectory
     *
     * @return The RMS distance between two object trajectories
     */
    inline double distanceRMS( const ObjectTrajectory& traj1,
                               const ObjectTrajectory& traj2 )
    {
        assert( traj1.size() == traj2.size() );
        double dist_squared = 0;
        for ( size_t ind = 0; ind < traj1.size(); ind++ )
        {
            dist_squared += distanceSquared( traj1[ind], traj2[ind]);
        }

        return std::sqrt( dist_squared / (double)traj1.size() );
    }

    /**
     * @brief Calculates the RMS distance between two object trajectories using
     * an L2 norm for each time step.
     *
     * @param traj1 The first object trajectory
     * @param traj2 The second object trajectory
     *
     * @return The RMS distance between two object trajectories
     */
    inline double distanceRMS( const std::vector< WorldState >& traj1,
                               const ObjectTrajectory& traj2 )
    {
        assert( traj1.size() == traj2.size() );
        double dist_squared = 0;
        for ( size_t ind = 0; ind < traj1.size(); ind++ )
        {
            dist_squared += distanceSquared( traj1[ind].object_configuration_,
                                             traj2[ind] );
        }

        return std::sqrt( dist_squared / (double)traj1.size() );
    }

    /**
     * @brief Computes the distance between each node in the given object
     *
     * @param obj The object to compute distances on
     *
     * @return The distances between each pair of nodes
     */
    inline Eigen::MatrixXd distanceMatrix( const ObjectPointSet& obj )
    {
        const long num_nodes = obj.cols();
        Eigen::MatrixXd dist( num_nodes, num_nodes );

        #pragma omp parallel for
        for ( long i = 0; i < num_nodes; i++ )
        {
            for ( long j = i; j < num_nodes; j++ )
            {
                dist( i, j ) =
                    ( obj.block< 3, 1>( 0, i ) - obj.block< 3, 1>( 0, j ) ).norm();
                dist( j, i ) = dist( i, j );
            }
        }

        return dist;
    }

    // TODO: vectorize this
    // TODO: use this for the coverage task error functions?
    inline long closestPointInSet( const ObjectPointSet& obj,
                                   const Eigen::Vector3d& point )
    {
        assert ( obj.cols() > 0 );
        long min_ind = 0;
        double min_dist = (obj.block< 3, 1 >( 0, 0 ) - point).norm();

        for ( long ind = 1; ind < obj.cols(); ind++ )
        {
            double dist = (obj.block< 3, 1 >( 0, ind ) - point).norm();
            if ( dist < min_dist )
            {
                min_ind = ind;
                min_dist = dist;
            }
        }

        return min_ind;
    }

    // TODO: vectorize this
    // TODO: use this for the coverage task error functions?
    inline long closestPointInSet( const ObjectPointSet& obj,
                                   Eigen::Vector3d&& point )
    {
        assert ( obj.cols() > 0 );
        long min_ind = 0;
        double min_dist = (obj.block< 3, 1 >( 0, 0 ) - point).norm();

        for ( long ind = 1; ind < obj.cols(); ind++ )
        {
            double dist = (obj.block< 3, 1 >( 0, ind ) - point).norm();
            if ( dist < min_dist )
            {
                min_ind = ind;
                min_dist = dist;
            }
        }

        return min_ind;
    }
}

#endif // TRAJECTORY_HPP
