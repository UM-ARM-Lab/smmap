#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <kinematics_toolbox/kinematics.h>
#include <smmap_msgs/messages.h>

namespace smmap
{
    // Typedefs to help deal with Deformable Object and Gripper tracking
    typedef Eigen::Matrix3Xd ObjectPointSet;
    typedef std::vector< ObjectPointSet > ObjectTrajectory;
    typedef std::vector< ObjectTrajectory > VectorObjectTrajectory;

    typedef EigenHelpers::VectorAffine3d AllGrippersSinglePose;
    typedef kinematics::VectorVector6d AllGrippersSingleVelocity;

    /// Feedback structure for a single time step
    struct WorldFeedback
    {
        ObjectPointSet object_configuration_;
        AllGrippersSinglePose all_grippers_single_pose_;
        kinematics::VectorVector3d all_grippers_nearest_point_to_obstacle_;
        kinematics::VectorVector3d all_grippers_obstacle_surface_normal_;
        std::vector< double > all_grippers_distance_to_obstacle_;
        double sim_time_;

        // NOTE: that this velocity is calculated as the difference between the
        // previous state and the state encoded by this structure. Thus the
        // first entry in an array of std::vector< WorldFeedback > will not have
        // a valid velocity entry.
        Eigen::Matrix3Xd object_velocity_;
        AllGrippersSingleVelocity all_grippers_single_velocity_;
    };

    /**
     * @brief computeNextFeedback
     * @param next_feedback_ros
     * @return
     */
    inline WorldFeedback computeNextFeedback(
            const smmap_msgs::SimulatorFeedback& next_feedback_ros )
    {
        WorldFeedback next_feedback_eigen;

        next_feedback_eigen.object_configuration_ =
                EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd(
                    next_feedback_ros.object_configuration );

        next_feedback_eigen.all_grippers_single_pose_ =
                EigenHelpersConversions::VectorGeometryPoseToVectorAffine3d(
                    next_feedback_ros.gripper_poses );

        next_feedback_eigen.all_grippers_nearest_point_to_obstacle_ =
                EigenHelpersConversions::VectorGeometryPointToEigenVector3d(
                    next_feedback_ros.gripper_nearest_point_to_obstacle );

        next_feedback_eigen.all_grippers_obstacle_surface_normal_ =
                EigenHelpersConversions::VectorGeometryVector3ToEigenVector3d(
                    next_feedback_ros.obstacle_surface_normal );

        next_feedback_eigen.all_grippers_distance_to_obstacle_ =
                next_feedback_ros.gripper_distance_to_obstacle;

        next_feedback_eigen.sim_time_ = next_feedback_ros.sim_time;

        return next_feedback_eigen;
    }

    /**
     * @brief computeNextFeedback
     * @param prev_feedback_ros
     * @param next_feedback_ros
     * @return
     */
    inline WorldFeedback computeNextFeedback(
            const WorldFeedback& prev_feedback_ros,
            const smmap_msgs::SimulatorFeedback& next_feedback_ros )
    {
        WorldFeedback next_feedback_eigen = computeNextFeedback( next_feedback_ros );

        // Calculate object and gripper velocities based on the previous timestep
        const double time_delta =
                next_feedback_ros.sim_time - prev_feedback_ros.sim_time_;

        const ObjectPointSet object_delta =
                next_feedback_eigen.object_configuration_
                - prev_feedback_ros.object_configuration_;
        next_feedback_eigen.object_velocity_ = object_delta / time_delta;

        next_feedback_eigen.all_grippers_single_velocity_.resize(
                    next_feedback_eigen.all_grippers_single_pose_.size() );

        for ( size_t gripper_ind = 0;
              gripper_ind < next_feedback_eigen.all_grippers_single_pose_.size();
              gripper_ind++ )
        {
            const Eigen::Affine3d curr_pose =
                    next_feedback_eigen.all_grippers_single_pose_[gripper_ind];
            const Eigen::Affine3d prev_pose =
                    prev_feedback_ros.all_grippers_single_pose_[gripper_ind];

            next_feedback_eigen.all_grippers_single_velocity_[gripper_ind] =
                    kinematics::calculateVelocity( prev_pose, curr_pose, time_delta );
        }

        return next_feedback_eigen;
    }

    /**
     * @brief parseGripperActionResult
     * @param result
     * @return
     */
    inline std::vector< WorldFeedback > parseGripperActionResult(
            const smmap_msgs::CmdGrippersTrajectoryResultConstPtr& result )
    {
        assert( result->sim_state_trajectory.size() > 0 );
        std::vector< WorldFeedback > world_feedback( result->sim_state_trajectory.size() );

        world_feedback[0] = computeNextFeedback( result->sim_state_trajectory[0] );
        for ( size_t time_ind = 1; time_ind < result->sim_state_trajectory.size(); time_ind++ )
        {
            world_feedback[time_ind] = computeNextFeedback(
                    world_feedback[time_ind-1],
                    result->sim_state_trajectory[time_ind] );
        }

        return world_feedback;
    }

    /**
     * @brief getGripperTrajectories
     * @param feedback
     * @return
     */
    inline std::vector< AllGrippersSinglePose > getGripperTrajectories(
            const std::vector< WorldFeedback >& feedback )
    {
        std::vector< AllGrippersSinglePose >
                grippers_trajectories( feedback.size() );

        for ( size_t time_ind = 0; time_ind < feedback.size(); time_ind++ )
        {
            grippers_trajectories[time_ind] =
                    feedback[time_ind].all_grippers_single_pose_;
        }

        return grippers_trajectories;
    }

    /**
     * @brief getGripperVelocities
     * @param feedback
     * @return
     */
    inline std::vector< AllGrippersSingleVelocity > getGripperVelocities(
            const std::vector< WorldFeedback >& feedback )
    {
        std::vector< AllGrippersSingleVelocity >
                grippers_velocities( feedback.size() - 1 );

        for ( size_t time_ind = 1; time_ind < feedback.size(); time_ind++ )
        {
            grippers_velocities[time_ind-1] =
                    feedback[time_ind].all_grippers_single_velocity_;
        }

        return grippers_velocities;
    }

    /**
     * @brief calculateGrippersVelocities
     * @param grippers_trajectory
     * @param dt
     * @return
     */
    inline std::vector< AllGrippersSingleVelocity > calculateGrippersVelocities(
            const std::vector< AllGrippersSinglePose >& grippers_trajectory,
            const double dt )
    {
        assert( grippers_trajectory.size() > 1 );
        const size_t num_grippers = grippers_trajectory[0].size();

        std::vector< AllGrippersSingleVelocity > grippers_velocities(
                    grippers_trajectory.size() - 1,
                    AllGrippersSingleVelocity( num_grippers ) );

        for ( size_t time_ind = 0; time_ind < grippers_velocities.size(); time_ind++ )
        {
            for ( size_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++ )
            {
                grippers_velocities[time_ind][gripper_ind] =
                        kinematics::calculateVelocity(
                            grippers_trajectory[time_ind][gripper_ind],
                            grippers_trajectory[time_ind + 1][gripper_ind],
                            dt );
            }
        }

        return grippers_velocities;
    }


    inline std::string PrintDeltaOneLine( std::vector< AllGrippersSinglePose > gripper_traj )
    {
        const Eigen::IOFormat eigen_io_one_line_( Eigen::FullPrecision, Eigen::DontAlignCols, " ", " ", "", "", "", ""  );

        std::stringstream out;

        AllGrippersSinglePose start = gripper_traj.front();
        AllGrippersSinglePose end = gripper_traj.back();

        assert( start.size() == end.size() );

        for ( size_t gripper_ind = 0; gripper_ind < start.size(); gripper_ind++ )
        {
            out << kinematics::calculateVelocity( start[gripper_ind], end[gripper_ind], 1 ).format( eigen_io_one_line_ ) << " ";
        }

        return out.str();
    }


    /// Stores the result of a collision avoidance calculation for a single gripper
    struct CollisionAvoidanceResult
    {
        CollisionAvoidanceResult( long cols_per_gripper )
            : nullspace_projector( Eigen::MatrixXd::Identity(
                                       cols_per_gripper, cols_per_gripper ) )
            , velocity( Eigen::VectorXd::Zero( cols_per_gripper ) )
            , distance( std::numeric_limits< double >::infinity() )
        {}
        Eigen::MatrixXd nullspace_projector;
        Eigen::VectorXd velocity;
        double distance;

        CollisionAvoidanceResult() = delete;
    };


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
    inline double distanceRMS( const std::vector< WorldFeedback >& traj1,
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
