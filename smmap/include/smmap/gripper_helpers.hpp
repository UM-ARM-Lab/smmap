#ifndef GRIPPER_HELPERS_HPP
#define GRIPPER_HELPERS_HPP

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <kinematics_toolbox/kinematics.h>
#include <limits>
#include <memory>

namespace smmap
{
    struct GripperData
    {
        GripperData( const std::string& name, const std::vector< long >& node_indices )
            : name( name )
            , node_indices( node_indices )
        {}

        /// The name associated with this gripper
        std::string name;

        /// Vector of the indices of the nodes that are grasped by the gripper
        std::vector< long > node_indices;

        /**
         * @brief operator <<
         * @param out The stream to output the data too
         * @param data The gripper data to output
         * @return
         */
        friend std::ostream& operator<< ( std::ostream& out, const GripperData& data )
        {
            out << data.name << " Node Indices: " << PrettyPrint::PrettyPrint( data.node_indices );
            return out;
        }
    };

    template< typename T >
    inline std::vector< long > VectorAnytypeToVectorLong( const std::vector< T >& vector_anytype )
    {
        std::vector< long > vector_signed( vector_anytype.size() );
        for ( size_t ind = 0; ind < vector_anytype.size(); ind++ )
        {
            vector_signed[ind] = (long)(vector_anytype[ind]);
        }
        return vector_signed;
    }

    inline std::vector< std::string > GetGripperNames( const std::vector< GripperData > grippers_data )
    {
        std::vector< std::string > names( grippers_data.size() );

        for ( size_t gripper_ind = 0; gripper_ind < grippers_data.size(); gripper_ind++ )
        {
            names[gripper_ind] = grippers_data[gripper_ind].name;
        }

        return names;
    }

    inline double GripperVelocity6dNormSquared( const kinematics::Vector6d& velocity )
    {
        kinematics::Vector6d weight;
        weight << 1.0, 1.0, 1.0, 1.0/20.0, 1.0/20.0, 1.0/20.0;

        return ( weight.cwiseProduct( velocity ) ).squaredNorm();
    }

    inline double GripperVelocity6dNorm( const kinematics::Vector6d& velocity )
    {
        return std::sqrt( GripperVelocity6dNormSquared( velocity ) );
    }

    inline double VectorGripperVelocity6dNorm( const Eigen::VectorXd& velocities )
    {
        assert( velocities.size() % 6 == 0 );

        double norm = 0;
        for ( long gripper_ind = 0; gripper_ind < velocities.size(); gripper_ind += 6 )
        {
            norm += GripperVelocity6dNormSquared( velocities.segment< 6 >( gripper_ind ) );
        }
        return std::sqrt( norm );
    }

    inline void ClampGripperVelocities( Eigen::VectorXd& velocities, const double max_vel )
    {
        assert( velocities.size() % 6 == 0 );

        for ( long gripper_ind = 0; gripper_ind < velocities.size(); gripper_ind += 6 )
        {
            const double velocity_norm = GripperVelocity6dNorm( velocities.segment< 6 >( gripper_ind ) );
            if ( velocity_norm > max_vel )
            {
                velocities.segment< 6 >( gripper_ind ) *= max_vel / velocity_norm;
            }
        }

    }

    /**
     * @brief getMinimumDistanceToGripper
     * @param gripper_indices The indices of the nodes that the gripper is in contact with
     * @param node_index The index of the node that we want to get the distance to
     * @param object_initial_node_distance The matrix of distances between nodes
     * @return The index of the closest node grasped by the gripper and the distance to that node
     */
    inline std::pair< long, double > getMinimumDistanceToGripper(
            const std::vector< long >& gripper_indices, long node_index,
            const Eigen::MatrixXd& object_initial_node_distance )
    {
        double min_dist = std::numeric_limits< double >::infinity();
        long min_ind = -1;

        for ( long ind: gripper_indices )
        {
            if ( object_initial_node_distance( ind, node_index ) < min_dist )
            {
                min_dist = object_initial_node_distance( ind, node_index );
                min_ind = ind;
            }
        }

        return std::pair< long, double>( min_ind, min_dist );
    }
}

#endif // GRIPPER_HELPERS_HPP
