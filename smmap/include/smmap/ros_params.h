#ifndef ROS_PARAMS_H
#define ROS_PARAMS_H

#include <string>
#include <arc_utilities/ros_helpers.hpp>
#include <exception>

#include "smmap/task_enums.h"

namespace smmap
{
    ////////////////////////////////////////////////////////////////////////////
    // Task and Deformable Type parameters
    ////////////////////////////////////////////////////////////////////////////

    inline DeformableType GetDeformableType( ros::NodeHandle& nh )
    {
        std::string deformable_type = ROSHelpers::GetParam< std::string >( nh, "deformable_type", "rope" );

        if ( deformable_type.compare( "rope" ) == 0 )
        {
            return DeformableType::ROPE;
        }
        else if ( deformable_type.compare( "cloth" ) == 0 )
        {
            return DeformableType::CLOTH;
        }
        else
        {
            throw new std::invalid_argument( "Unknown deformable type: " + deformable_type );
        }
    }

    inline TaskType GetTaskType( ros::NodeHandle& nh )
    {
        std::string task_type = ROSHelpers::GetParam< std::string >(nh, "task_type", "coverage" );

        if ( task_type.compare( "coverage" ) == 0 )
        {
            return TaskType::COVERAGE;
        }
        else if ( task_type.compare( "colab_folding" ) == 0 )
        {
            return TaskType::COLAB_FOLDING;
        }
        else
        {
            throw new std::invalid_argument( "Unknown task type: " + task_type );
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Logging functionality
    ////////////////////////////////////////////////////////////////////////////

    inline bool GetLoggingEnabled( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam( nh, "logging_enabled", false );
    }

    inline std::string GetLogFolder( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "log_folder", "/tmp/" );
    }

    ////////////////////////////////////////////////////////////////////////////
    // ROS Topic settings
    ////////////////////////////////////////////////////////////////////////////

    inline std::string GetCommandGripperTrajTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "command_gripper_traj_topic", "command_gripper_traj" );
    }

    inline std::string GetSimulatorFeedbackTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "simulator_feedback_topic", "simulator_feedback" );
    }

    inline std::string GetCoverPointsTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "get_cover_points_topic", "get_cover_points" );
    }

    inline std::string GetMirrorLineTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "get_mirror_line_topic", "get_mirror_line" );
    }

    inline std::string GetGripperNamesTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "get_gripper_names_topic", "get_gripper_names" );
    }

    inline std::string GetGripperAttachedNodeIndicesTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "get_gripper_attached_node_indices", "get_gripper_attached_node_indices" );
    }

    inline std::string GetGripperPoseTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "get_gripper_pose_topic", "get_gripper_pose" );
    }

    inline std::string GetObjectInitialConfigurationTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "get_object_initial_configuration_topic", "get_object_initial_configuration" );
    }

    inline std::string GetObjectCurrentConfigurationTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "get_object_current_configuration_topic", "get_object_current_configuration" );
    }

    inline std::string GetVisualizationMarkerTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "visualization_marker_topic", "visualization_marker" );
    }

    inline std::string GetVisualizationMarkerArrayTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "visualization_marker_array_topic", "visualization_marker_array" );
    }

    inline std::string GetConfidenceTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "confidence_topic", "confidence" );
    }

    inline std::string GetConfidenceImageTopic( ros::NodeHandle& nh )
    {
        return ROSHelpers::GetParam< std::string >( nh, "confidence_image_topic", "confidence_image" );
    }
}

#endif // ROS_PARAMS_H
