#include "smmap/planner.h"

int main(int argc, char* argv[])
{
    // Read in all ROS parameters
    ros::init(argc, argv, "smmap_planner_node", ros::init_options::NoSigintHandler);

    ros::NodeHandle nh;
    ros::NodeHandle ph("~");

    ROS_INFO("Creating utility objects");
    smmap::RobotInterface::Ptr robot = std::make_shared<smmap::RobotInterface>(nh);
    robot->setCallbackFunctions(
                nullptr,
                nullptr,
                nullptr);
    smmap_utilities::Visualizer::Ptr vis = std::make_shared<smmap_utilities::Visualizer>(nh, ph);
    smmap::TaskSpecification::Ptr task_specification(smmap::TaskSpecification::MakeTaskSpecification(nh, ph, vis));

    ROS_INFO("Creating and executing planner");
    smmap::Planner planner(nh, ph, robot, vis, task_specification);
    planner.execute();

    ROS_INFO("Disposing planner...");

    return 0;
}
