#include <smmap_utilities/visualization_tools.h>
#include "smmap/transition_learning_data_generation.h"
#include "smmap/robot_interface.h"

std::vector<Eigen::VectorXd> getJointInfo()
{
    const Eigen::VectorXd lower_limits = Eigen::VectorXd::Constant(6, -std::numeric_limits<double>::max());
    const Eigen::VectorXd upper_limits = Eigen::VectorXd::Constant(6, std::numeric_limits<double>::max());
    const Eigen::VectorXd weights = Eigen::VectorXd::Constant(6, 1.0);
    return {lower_limits, upper_limits, weights};
}

int main(int argc, char* argv[])
{
    using namespace smmap;
    using namespace smmap_utilities;

    // Read in all ROS parameters
    ros::init(argc, argv, "transition_learning_data_generation_node", ros::init_options::NoSigintHandler);

    auto nh = std::make_shared<ros::NodeHandle>();
    auto ph = std::make_shared<ros::NodeHandle>("~");

    ROS_INFO("Creating utility objects");
    auto robot = std::make_shared<RobotInterface>(nh, ph);
    robot->setCallbackFunctions(
                nullptr,
                nullptr,
                nullptr,
                &getJointInfo,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr);
    auto vis = std::make_shared<Visualizer>(nh, ph, true);
//    auto task = std::dynamic_pointer_cast<DijkstrasCoverageTask>(
//                TaskSpecification::MakeTaskSpecification(nh, ph, vis));
//    assert(task != nullptr);
    auto data_generator = DataGeneration(nh, ph, robot, vis);

    return EXIT_SUCCESS;
}
