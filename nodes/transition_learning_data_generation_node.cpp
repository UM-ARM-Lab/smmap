#include <smmap_utilities/visualization_tools.h>
#include "smmap/transition_learning_data_generation.h"
#include "smmap/robot_interface.h"

namespace dmm = deformable_manipulation_msgs;

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
    const bool generate_test_data = ROSHelpers::GetParam<bool>(*ph, "generate_test_data", false);
    const bool generate_transition_approximations = ROSHelpers::GetParam<bool>(*ph, "generate_transitions", false);
    transition_tester.runTests(generate_test_data, generate_transition_approximations);

    // Set the source to make sense, if this file exists
    dmm::TransitionTestingVisualizationRequest req;
    req.data = "cannonical_straight_test/unmodified__test_results.compressed";
    dmm::TransitionTestingVisualizationResponse res;
    transition_tester.setSourceCallback(req, res);
    ROS_INFO("Waiting for visualization requests...");
    ros::spin();

    return EXIT_SUCCESS;
}
