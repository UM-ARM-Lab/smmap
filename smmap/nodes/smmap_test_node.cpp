#include "smmap/task.h"

using namespace smmap;

int main( int argc, char* argv[] )
{
    // Read in all ROS parameters
    ros::init( argc, argv, "smmap_planner", ros::init_options::NoSigintHandler );

    ros::NodeHandle nh;

    Task task( nh );
    task.execute();

    return 0;
}
