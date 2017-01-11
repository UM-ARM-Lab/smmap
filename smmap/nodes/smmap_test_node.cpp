#include <Python.h>
#include "smmap/task.h"
#include "smmap/cvxopt_solvers.h"

using namespace smmap;

int main(int argc, char* argv[])
{
#warning "This is manually disabled and ought to be fixed instead"
//    CVXOptSolvers::Initialize();

    // Read in all ROS parameters
    ros::init(argc, argv, "smmap_planner_node", ros::init_options::NoSigintHandler);

    ros::NodeHandle nh;

    RobotInterface robot(nh);
    Visualizer vis(nh);
    TaskSpecification::Ptr task_specification = TaskSpecification::MakeTaskSpecification(nh);
    Task task(robot, vis, task_specification);
    task.execute();

    CVXOptSolvers::Finalize();
    return 0;
}
