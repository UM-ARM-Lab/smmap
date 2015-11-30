#ifndef gripper_helpers_h
#define gripper_helpers_h

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <limits>
#include <memory>

namespace smmap
{
    struct GripperData
    {
        GripperData( const Eigen::Affine3d& pose, const std::vector< long >& node_indices, const std::string& name )
            : pose( pose )
            , nearest_point_on_gripper ( Eigen::Vector3d::Zero() )
            , nearest_point_on_obstacle ( Eigen::Vector3d::Zero() )
            , distance_to_obstacle( std::numeric_limits< double >::infinity() )
            , node_indices( node_indices )
            , name( name )
        {}

        friend std::ostream& operator<< ( std::ostream& out, const GripperData& data )
        {
            out << data.name << " Num Indices: " << PrettyPrint::PrettyPrint( data.node_indices )
                << " " << PrettyPrint::PrettyPrint( data.pose );
            return out;
        }

        Eigen::Affine3d pose;
        Eigen::Vector3d nearest_point_on_gripper;  // closest point on the gripper to the obstacle
        Eigen::Vector3d nearest_point_on_obstacle; // closest point on the obstacle to the gripper
        double distance_to_obstacle;

        std::vector< long > node_indices;
        std::string name;
    };
    typedef std::vector< GripperData, Eigen::aligned_allocator<Eigen::Affine3d> > VectorGrippersData;


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

#endif // gripper_helpers_h
