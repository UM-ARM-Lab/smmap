#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <kinematics_toolbox/kinematics.h>
#include <deformable_manipulation_msgs/messages.h>

#include "smmap/grippers.hpp"

namespace smmap
{
    typedef Eigen::Matrix3Xd ObjectPointSet;
    typedef std::vector<ObjectPointSet> ObjectTrajectory;
    typedef std::vector<ObjectTrajectory> VectorObjectTrajectory;

    struct ObjectDeltaAndWeight
    {
        public:
            ObjectDeltaAndWeight()
            {}

            ObjectDeltaAndWeight(ssize_t num_elems)
                : delta(Eigen::VectorXd::Zero(num_elems))
                , weight(Eigen::VectorXd::Zero(num_elems))
            {}

            Eigen::VectorXd delta;
            Eigen::VectorXd weight;
    };

    /// World state structure for a single time step
    struct WorldState
    {
        ObjectPointSet object_configuration_;
        AllGrippersSinglePose all_grippers_single_pose_;
        Eigen::VectorXd robot_configuration_;
        bool robot_configuration_valid_;
        std::vector<CollisionData> gripper_collision_data_;
        double sim_time_;
    };

    /**
     * @brief ConvertToEigenFeedback
     * @param feedback_ros
     * @return
     */
    inline WorldState ConvertToEigenFeedback(
            const deformable_manipulation_msgs::WorldState& feedback_ros)
    {
        WorldState feedback_eigen;

        feedback_eigen.object_configuration_ =
                EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd(feedback_ros.object_configuration);

        feedback_eigen.all_grippers_single_pose_ =
                EigenHelpersConversions::VectorGeometryPoseToVectorIsometry3d(feedback_ros.gripper_poses);

        feedback_eigen.robot_configuration_ =
                EigenHelpersConversions::StdVectorToEigenVectorX(feedback_ros.robot_configuration);
        feedback_eigen.robot_configuration_valid_ = feedback_ros.robot_configuration_valid;

        size_t num_grippers = feedback_ros.gripper_poses.size();
        feedback_eigen.gripper_collision_data_.reserve(num_grippers);
        for (size_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
        {
            feedback_eigen.gripper_collision_data_.push_back(
                        CollisionData(
                            EigenHelpersConversions::GeometryPointToEigenVector3d(
                                feedback_ros.gripper_nearest_point_to_obstacle[gripper_ind]),
                            EigenHelpersConversions::GeometryVector3ToEigenVector3d(
                                feedback_ros.obstacle_surface_normal[gripper_ind]),
                            feedback_ros.gripper_distance_to_obstacle[gripper_ind]));
        }

        feedback_eigen.sim_time_ = feedback_ros.sim_time;

        return feedback_eigen;
    }

    /**
     * @brief getGripperTrajectories
     * @param feedback
     * @return
     */
    inline AllGrippersPoseTrajectory GetGripperTrajectories(
            const std::vector<WorldState>& feedback)
    {
        AllGrippersPoseTrajectory grippers_trajectories(feedback.size());

        for (size_t time_ind = 0; time_ind < feedback.size(); time_ind++)
        {
            grippers_trajectories[time_ind] =
                    feedback[time_ind].all_grippers_single_pose_;
        }

        return grippers_trajectories;
    }

    inline Eigen::VectorXd CalculateObjectDeltaAsVector(
            const ObjectPointSet& start,
            const ObjectPointSet& end)
    {
        Eigen::MatrixXd diff = end - start;
        diff.resize(diff.rows() * diff.cols(), 1);
        return diff;
    }

    inline ObjectPointSet AddObjectDelta(
            ObjectPointSet start,
            const Eigen::VectorXd& delta)
    {
        assert(delta.rows() == start.cols() * 3);

        for (ssize_t point_ind = 0; point_ind < start.cols(); point_ind++)
        {
            start.col(point_ind) = start.col(point_ind) + delta.segment<3>(point_ind * 3);
        }
        return start;
    }   
}

#endif // TRAJECTORY_HPP
