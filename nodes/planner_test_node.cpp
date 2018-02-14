#include "smmap/planner.h"

int main(int argc, char* argv[])
{
    // Read in all ROS parameters
    ros::init(argc, argv, "smmap_planner_node", ros::init_options::NoSigintHandler);

    ros::NodeHandle nh;
    ros::NodeHandle ph("~");

    smmap::RobotInterface robot(nh);
    smmap_utilities::Visualizer vis(nh, ph);

    smmap::TaskSpecification::Ptr task_specification = smmap::TaskSpecification::MakeTaskSpecification(nh, ph, vis);
    smmap::Planner planner(nh, ph, robot, vis, task_specification);
    planner.execute();

    return 0;
}
