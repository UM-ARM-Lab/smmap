#include "smmap/model_test.h"

using namespace smmap;

int main(int argc, char* argv[])
{
    // Read in all ROS parameters
    ros::init(argc, argv, "smmap_planner_node", ros::init_options::NoSigintHandler);

    ros::NodeHandle nh;
    ros::NodeHandle ph("~");

    RobotInterface robot(nh);
    Visualizer vis(nh, ph);

    TestSpecification::Ptr test_specification = TestSpecification::MakeTestSpecification(nh, ph);
    ModelTest test(robot, vis, test_specification);
    test.execute();

    return 0;
}
