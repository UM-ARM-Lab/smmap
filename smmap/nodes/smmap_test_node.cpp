#include "smmap/planner.h"

using namespace smmap;

int main( int argc, char* argv[] )
{
    // Read in all ROS parameters
    ros::init( argc, argv, "smmap_planner", ros::init_options::NoSigintHandler );

    ros::NodeHandle nh;

    const size_t num_traj_cmds_per_loop = 10;
    const double dt = 0.01;

    Planner planner( nh );
    planner.run( num_traj_cmds_per_loop, dt );

    return 0;
}
