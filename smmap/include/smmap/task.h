#ifndef task_h
#define task_h

#include <string>
#include <arc_utilities/ros_helpers.hpp>

namespace smmap
{
    inline std::string CommandGripperTrajTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "command_gripper_traj_topic", "command_gripper_traj" );
    }

    inline std::string SimulatorFeedbackTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "simulator_feedback_topic", "simulator_feedback" );
    }

    inline std::string GetCoverPointsTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "get_cover_points_topic", "get_cover_points" );
    }

    inline std::string GetGripperNamesTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "get_gripper_names_topic", "get_gripper_names" );
    }

    inline std::string GetGripperAttachedNodeIndicesTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "get_gripper_attached_node_indices", "get_gripper_attached_node_indices" );
    }

    inline std::string GetGripperPoseTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "get_gripper_pose_topic", "get_gripper_pose" );
    }

    inline std::string GetObjectInitialConfigurationTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "get_object_initial_configuration_topic", "get_object_initial_configuration" );
    }

    inline std::string VisualizationMarkerTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "visualization_marker_topic", "visualization_marker" );
    }

    inline std::string VisualizationMarkerArrayTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "visualization_marker_array_topic", "visualization_marker_array" );
    }

    inline std::string ConfidenceTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "confidence_topic", "confidence" );
    }

    inline std::string ConfidenceImageTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >(nh, "confidence_image_topic", "confidence_image" );
    }

    enum DeformableType
    {
        ROPE,
        CLOTH
    };

    enum TaskType
    {
        COVERAGE,
        COLAB_FOLDING
    };


//    class task
//    {
//    };
}

#endif // task_h
