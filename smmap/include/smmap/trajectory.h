#ifndef trajectory_h
#define trajectory_h

#include <arc_utilities/eigen_helpers.hpp>

namespace smmap
{
    // Typedefs to help deal with Deformable Object and Gripper tracking
    typedef Eigen::Matrix3Xd ObjectPointSet;
    typedef std::vector< ObjectPointSet, Eigen::aligned_allocator< Eigen::Matrix3Xd > > ObjectTrajectory;
    typedef std::vector< ObjectTrajectory, Eigen::aligned_allocator< ObjectTrajectory > > VectorObjectTrajectory;

    typedef EigenHelpers::VectorAffine3d SingleGripperTrajectory;
    typedef std::vector< SingleGripperTrajectory, Eigen::aligned_allocator< SingleGripperTrajectory > > AllGrippersTrajectory;

    /**
     * @brief Given a set of grippers and the trajectory that each follows,
     * returns the last (most recent) pose of each gripper.
     *
     * @param grippers_trajectory The trajectory for each grippe
     *
     * @return The last pose of each gripper
     */
    inline EigenHelpers::VectorAffine3d getLastGrippersPose( const AllGrippersTrajectory& grippers_trajectory )
    {
        EigenHelpers::VectorAffine3d last_poses( grippers_trajectory.size() );
        for ( size_t gripper_ind = 0; gripper_ind < grippers_trajectory.size(); gripper_ind++ )
        {
            last_poses[gripper_ind] = grippers_trajectory[gripper_ind].back();
        }
        return last_poses;
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
    inline double distance( const ObjectPointSet& set1, const ObjectPointSet& set2 )
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
    inline double distanceSquared( const ObjectPointSet& set1, const ObjectPointSet& set2 )
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
    inline double distanceRMS( const ObjectTrajectory& traj1, const ObjectTrajectory& traj2 )
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
}

#endif // trajectory_h
