#include <Python.h>
#include "smmap/task.h"
#include "smmap/model_test.h"

using namespace smmap;

int main(int argc, char* argv[])
{
    // Read in all ROS parameters
    ros::init(argc, argv, "smmap_planner_node", ros::init_options::NoSigintHandler);

    ros::NodeHandle nh;

    RobotInterface robot(nh);
    Visualizer vis(nh);

    TestSpecification::Ptr test_specification = TestSpecification::MakeTaskSpecification(nh);
    modelTest test(robot, vis, test_specification);
    test.execute();

    ///// The Previous Task Version
    /**
    TaskSpecification::Ptr task_specification = TaskSpecification::MakeTaskSpecification(nh);
    Task task(robot, vis, task_specification);
    task.execute();
    */

    return 0;
}
