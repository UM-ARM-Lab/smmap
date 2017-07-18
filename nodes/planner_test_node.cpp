#include "smmap/planner.h"

using namespace smmap;

int main(int argc, char* argv[])
{
    // Read in all ROS parameters
    ros::init(argc, argv, "smmap_planner_node", ros::init_options::NoSigintHandler);

    ros::NodeHandle nh;
    ros::NodeHandle ph("~");

    RobotInterface robot(nh);
    Visualizer vis(nh, ph);

    TaskSpecification::Ptr task_specification = TaskSpecification::MakeTaskSpecification(nh, ph);
    Planner planner(robot, vis, task_specification);
    planner.execute();

    return 0;
}
