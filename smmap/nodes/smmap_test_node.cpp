#include "smmap/planner.h"
#include "smmap/diminishing_rigidity_model.h"

using namespace smmap;

int main( int argc, char* argv[] )
{
    // Read in all ROS parameters
    ros::init( argc, argv, "smmap_planner", ros::init_options::NoSigintHandler );

    ros::NodeHandle nh;
    Visualizer vis( nh );

    const double dt = 0.01;

    Planner planner( nullptr, nullptr, nullptr, nullptr, vis );
    planner.getNextTrajectory( WorldState(), 1, dt );

    DiminishingRigidityModel model( 1 );

    return 0;
}
