#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/serialization_eigen.hpp>
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

    struct DesiredDirection
    {
        public:
            ObjectDeltaAndWeight error_correction_;
            ObjectDeltaAndWeight stretching_correction_;
            ObjectDeltaAndWeight combined_correction_;
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

        inline uint64_t serialize(std::vector<uint8_t>& buffer) const
        {
            const size_t starting_bytes = buffer.size();
            arc_utilities::SerializeEigen(object_configuration_, buffer);
            SerializeAllGrippersSinglePose(all_grippers_single_pose_, buffer);
            arc_utilities::SerializeEigen(robot_configuration_, buffer);
            arc_utilities::SerializeFixedSizePOD(robot_configuration_valid_, buffer);
            SerializeCollisionDataVector(gripper_collision_data_, buffer);
            arc_utilities::SerializeFixedSizePOD(sim_time_, buffer);
            const size_t ending_bytes = buffer.size();
            return ending_bytes - starting_bytes;
        }

        static inline std::pair<WorldState, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            uint64_t bytes_read = 0;
            WorldState result;

            const auto deserialized_object = arc_utilities::DeserializeEigen<ObjectPointSet>(buffer, current + bytes_read);
            result.object_configuration_ = deserialized_object.first;
            bytes_read += deserialized_object.second;

            const auto deserialized_grippers = DeserializeAllGrippersSinglePose(buffer, current + bytes_read);
            result.all_grippers_single_pose_ = deserialized_grippers.first;
            bytes_read += deserialized_grippers.second;

            const auto deserialized_robot_config = arc_utilities::DeserializeEigen<Eigen::VectorXd>(buffer, current + bytes_read);
            result.robot_configuration_ = deserialized_robot_config.first;
            bytes_read += deserialized_robot_config.second;

            const auto deserialized_robot_config_valid = arc_utilities::DeserializeFixedSizePOD<bool>(buffer, current + bytes_read);
            result.robot_configuration_valid_ = deserialized_robot_config_valid.first;
            bytes_read += deserialized_robot_config_valid.second;

            const auto deserialized_collision_data = DeserializeCollisionDataVector(buffer, current + bytes_read);
            result.gripper_collision_data_ = deserialized_collision_data.first;
            bytes_read += deserialized_collision_data.second;

            const auto deserialized_sim_time = arc_utilities::DeserializeFixedSizePOD<double>(buffer, current + bytes_read);
            result.sim_time_ = deserialized_sim_time.first;
            bytes_read += deserialized_sim_time.second;

            return {result, bytes_read};
        }

        bool operator==(const WorldState& other) const
        {
            if (object_configuration_.cwiseNotEqual(other.object_configuration_).any())
            {
                return false;
            }

            if (all_grippers_single_pose_.size() != other.all_grippers_single_pose_.size())
            {
                return false;
            }

            for (size_t idx = 0; idx < all_grippers_single_pose_.size(); ++ idx)
            {
                if (all_grippers_single_pose_[idx].matrix().cwiseNotEqual(other.all_grippers_single_pose_[idx].matrix()).any())
                {
                    return false;
                }
            }

            if (robot_configuration_.cwiseNotEqual(other.robot_configuration_).any())
            {
                return false;
            }

            if (robot_configuration_valid_ != other.robot_configuration_valid_)
            {
                return false;
            }

            if (gripper_collision_data_.size() != other.gripper_collision_data_.size())
            {
                return false;
            }

            for (size_t idx = 0; idx < gripper_collision_data_.size(); ++idx)
            {
                if (gripper_collision_data_[idx] != other.gripper_collision_data_[idx])
                {
                    return false;
                }
            }

            if (sim_time_ != other.sim_time_)
            {
                return false;
            }

            return true;
        }

        bool operator!=(const WorldState& other) const
        {
            return !(*this == other);
        }
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
                EigenHelpers::StdVectorToEigenVectorX(feedback_ros.robot_configuration);
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
