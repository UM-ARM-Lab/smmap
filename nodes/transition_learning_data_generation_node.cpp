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
    auto transition_tester = TransitionTesting(nh, ph, robot, vis);
    const bool generate_new_data = ROSHelpers::GetParam<bool>(*ph, "generate_new_data", false);
    transition_tester.runTests(generate_new_data);

    return EXIT_SUCCESS;
}
