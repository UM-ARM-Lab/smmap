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

    // structure for wrench  --- Added by Mengyao
    struct Wrench
    {
        public:
            Wrench()
            {}

            Wrench(std::pair<Eigen::Vector3d, Eigen::Vector3d> wrench)
                : force(wrench.first)
                , torque(wrench.second)
            {}

            Wrench(const Wrench& wrench)
                : force(wrench.force)
                , torque(wrench.torque)
            {}

            Eigen::Vector3d force;
            Eigen::Vector3d torque;
    };

    struct ObjectWrench
    {
        public:
            ObjectWrench()
            { }

            ObjectWrench(std::vector<Wrench> wrench_vector)
            {
                object_force.clear();
                object_torque.clear();
                for (size_t node_ind = 0; node_ind < wrench_vector.size(); node_ind++)
                {
                    object_force.push_back(wrench_vector.at(node_ind).force);
                    object_torque.push_back(wrench_vector.at(node_ind).torque);
                }
            }

            void SetObjectWrench(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> wrench_pair_vector)
            {
                object_force.clear();
                object_torque.clear();
                for (size_t node_ind = 0; node_ind < wrench_pair_vector.size(); node_ind++)
                {
                    object_force.push_back(wrench_pair_vector.at(node_ind).first);
                    object_torque.push_back(wrench_pair_vector.at(node_ind).second);
                }

            }

            const std::vector<Eigen::Vector3d> MagnifiedForce(double scale) const
            {
                std::vector<Eigen::Vector3d> magnified_force;
                for (size_t node_ind = 0; node_ind < object_force.size(); node_ind++)
                {
                    magnified_force.push_back(object_force.at(node_ind) * scale);
                }
                return magnified_force;
            }
            const std::vector<Eigen::Vector3d> MagnifiedTorque(double scale) const
            {
                std::vector<Eigen::Vector3d> magnified_torque;
                for (size_t node_ind = 0; node_ind < object_force.size(); node_ind++)
                {
                    magnified_torque.push_back(object_torque.at(node_ind) * scale);
                }
                return magnified_torque;
            }

            const std::pair<Eigen::Vector3d, Eigen::Vector3d> GetRopeEndsForce() const
            {
                std::pair<Eigen::Vector3d, Eigen::Vector3d> ends_force;
                if (object_force.size() > 0)
                {
                    ends_force = std::make_pair(object_force.at(0), object_force.at(object_force.size()-1));

                }
                else
                {
                    std::cout << "size of force data < 1"
                              << std::endl;
                }
                return ends_force;
            }

            std::vector<Eigen::Vector3d> object_force;
            std::vector<Eigen::Vector3d> object_torque;
    };

    struct SingleGripperWrench
    {
        public:
            SingleGripperWrench()
            {}

            SingleGripperWrench(const Wrench& top_data, const Wrench& bottom_data)
                : top_clamp(top_data)
                , bottom_clamp(bottom_data)
            {}

            SingleGripperWrench(const SingleGripperWrench& single_gripper_wrench)
                : top_clamp(single_gripper_wrench.top_clamp)
                , bottom_clamp(single_gripper_wrench.bottom_clamp)
            {}

            Wrench top_clamp;
            Wrench bottom_clamp;
    };
    typedef std::vector<SingleGripperWrench> AllGrippersWrench;


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

        // Force and torque data --- Added by Mengyao
        ObjectWrench object_wrench_;

        AllGrippersSinglePose all_grippers_single_pose_;
        std::vector<CollisionData> gripper_collision_data_;

        // Force and torque data --- Added by Mengyao
        AllGrippersWrench gripper_wrench_;

        double sim_time_;
    };

    /**
     * @brief computeNextFeedback
     * @param next_feedback_ros
     * @return
     */
    inline WorldState ConvertToEigenFeedback(
            const deformable_manipulation_msgs::SimulatorFeedback& feedback_ros)
    {
        WorldState feedback_eigen;

        feedback_eigen.object_configuration_ =
                EigenHelpersConversions::VectorGeometryPointToEigenMatrix3Xd(
                    feedback_ros.object_configuration);

        // Read wrench information --- Added by Mengyao
        feedback_eigen.object_wrench_.object_force.clear();
        feedback_eigen.object_wrench_.object_torque.clear();

        feedback_eigen.object_wrench_.SetObjectWrench(
                    EigenHelpersConversions::GeometryWrenchToEigenPairVector(
                        feedback_ros.object_wrenches));

        feedback_eigen.all_grippers_single_pose_ =
                EigenHelpersConversions::VectorGeometryPoseToVectorAffine3d(
                    feedback_ros.gripper_poses);

        // TODO: modify the feedback to change this data ordering
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
            // Read wrench information --- Added by Mengyao
            feedback_eigen.gripper_wrench_.push_back(
                        SingleGripperWrench(
                            Wrench(EigenHelpersConversions::GeometryWrenchToEigenPair(
                                       feedback_ros.gripper_wrenches[gripper_ind * 2])),
                            Wrench(EigenHelpersConversions::GeometryWrenchToEigenPair(
                                       feedback_ros.gripper_wrenches[gripper_ind * 2 + 1]))));

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

    /**
     * @brief Computes the squared distance between each node in the given object
     *
     * @param obj The object to compute distances on
     *
     * @return The distances between each pair of nodes
     */
    // TODO: This is in the wrong spot
    inline Eigen::MatrixXd CalculateSquaredDistanceMatrix(const ObjectPointSet& obj)
    {
        assert (obj.cols() > 0);
        const ssize_t num_nodes = obj.cols();

        Eigen::MatrixXd squared_dist(num_nodes, num_nodes);
        #pragma omp parallel for
        for (ssize_t i = 0; i < num_nodes; i++)
        {
            for (ssize_t j = i; j < num_nodes; j++)
            {
                const double sq_dist = (obj.col(i) - obj.col(j)).squaredNorm();
                squared_dist(i, j) = sq_dist;
                squared_dist(j, i) = sq_dist;
            }
        }

        return squared_dist;
    }

    /**
     * @brief Computes the distance between each node in the given object
     *
     * @param obj The object to compute distances on
     *
     * @return The distances between each pair of nodes
     */
    // TODO: This is in the wrong spot
    inline Eigen::MatrixXd CalculateDistanceMatrix(const ObjectPointSet& obj)
    {
        return CalculateSquaredDistanceMatrix(obj).cwiseSqrt();
    }

    // TODO: This is in the wrong spot
    inline Eigen::VectorXd CalculateSquaredDistanceToSet(const ObjectPointSet& obj, const Eigen::Vector3d& point)
    {
        return (obj.colwise() - point).colwise().squaredNorm();
    }

    // TODO: This is in the wrong spot
    inline Eigen::VectorXd CalculateDistanceToSet(const ObjectPointSet& obj, const Eigen::Vector3d& point)
    {
        return CalculateSquaredDistanceToSet(obj, point).cwiseSqrt();
    }

    // TODO: use this for the coverage task error functions?
    inline ssize_t ClosestPointInSet(const ObjectPointSet& obj, const Eigen::Vector3d& point)
    {
        assert (obj.cols() > 0);
        ssize_t min_ind = 0;
        const Eigen::VectorXd squared_dist = CalculateSquaredDistanceToSet(obj, point);
        squared_dist.minCoeff(&min_ind);
        return min_ind;
    }
}

#endif // TRAJECTORY_HPP
