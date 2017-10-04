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

        // Force and torque data --- Added by Mengyao
        //ObjectWrench object_wrench_;

        AllGrippersSinglePose all_grippers_single_pose_;
        std::vector<CollisionData> gripper_collision_data_;

        // Force and torque data --- Added by Mengyao
        //AllGrippersWrench gripper_wrench_;

        double sim_time_;
    };

    // Helper structure to convert between x ind, y ind, and node ind for cloth;
    // TODO: Should I put this struct here or else where?    --- Added by Mengyao
    struct NodeXYInd
    {
        NodeXYInd()
        {}

        NodeXYInd(ssize_t x_num, ssize_t y_num)
            : num_x_steps_(x_num)
            , num_y_steps_(y_num)
            , num_nodes_(x_num * y_num)
        {}

        void SetNodeXYInd(ssize_t x_num, ssize_t y_num)
        {
            num_x_steps_ = x_num;
            num_y_steps_ = y_num;
            num_nodes_ = x_num * y_num;
        }

        ssize_t GetNodeInd(ssize_t x_ind, ssize_t y_ind)
        {
            assert(NodeInBound(x_ind, y_ind)||"xy_ind out of bound");
            return y_ind * num_x_steps_ + x_ind;
        }

        // first element is x ind, second is y ind
        std::pair<ssize_t, ssize_t> GetXYInd(ssize_t node_ind)
        {
            assert(NodeInBound(node_ind)||"node_ind out of bound");
            std::pair<ssize_t, ssize_t> xy_ind;
            xy_ind.second = node_ind / num_x_steps_;
            xy_ind.first = node_ind - xy_ind.second * num_x_steps_;
            return xy_ind;
        }

        // the size of vector is always 8, value is -1 if out of bound.
        /* Layout :
         *   3  2  1
         *   4  X  0
         *   5  6  7
        */
        std::vector<ssize_t> Neighbor8Ind(ssize_t node_ind)
        {
            const int num_neighbor = 8;
            std::vector<ssize_t> nearest_8_neighbor(num_neighbor, -1);

            if (NodeInBound(node_ind + 1))
            { nearest_8_neighbor.at(0) = node_ind + 1;}

            if (NodeInBound(node_ind + 1 + num_x_steps_))
            { nearest_8_neighbor.at(1) = node_ind + 1 + num_x_steps_;}

            if (NodeInBound(node_ind + num_x_steps_))
            { nearest_8_neighbor.at(2) = node_ind + num_x_steps_;}

            if (NodeInBound(node_ind - 1 + num_x_steps_))
            { nearest_8_neighbor.at(3) = node_ind - 1 + num_x_steps_;}

            if (NodeInBound(node_ind - 1))
            { nearest_8_neighbor.at(4) = node_ind - 1;}

            if (NodeInBound(node_ind - 1 - num_x_steps_))
            { nearest_8_neighbor.at(5) = node_ind - 1 - num_x_steps_;}

            if (NodeInBound(node_ind - num_x_steps_))
            { nearest_8_neighbor.at(6) = node_ind - num_x_steps_;}

            if (NodeInBound(node_ind + 1 - num_x_steps_))
            { nearest_8_neighbor.at(7) = node_ind + 1 - num_x_steps_;}

            return nearest_8_neighbor;
        }

        bool NodeInBound(ssize_t node_ind)
        {
            if ((node_ind >=0) && (node_ind < num_nodes_))
                return true;
            return false;
        }

        bool NodeInBound(ssize_t x_ind, ssize_t y_ind)
        {
            if ((x_ind >=0) && (x_ind < num_x_steps_))
            {
                if (((y_ind >=0) && (y_ind < num_y_steps_)))
                    return true;
            }
            return false;
        }

        ssize_t num_x_steps_;
        ssize_t num_y_steps_;
        ssize_t num_nodes_;
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
        /*
        feedback_eigen.object_wrench_.object_force.clear();
        feedback_eigen.object_wrench_.object_torque.clear();


        feedback_eigen.object_wrench_.SetObjectWrench(
                    EigenHelpersConversions::GeometryWrenchToEigenPairVector(
                        feedback_ros.object_wrenches));
        */

        feedback_eigen.all_grippers_single_pose_ =
                EigenHelpersConversions::VectorGeometryPoseToVectorIsometry3d(
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
            /*
            feedback_eigen.gripper_wrench_.push_back(
                        SingleGripperWrench(
                            Wrench(EigenHelpersConversions::GeometryWrenchToEigenPair(
                                       feedback_ros.gripper_wrenches[gripper_ind * 2])),
                            Wrench(EigenHelpersConversions::GeometryWrenchToEigenPair(
                                       feedback_ros.gripper_wrenches[gripper_ind * 2 + 1]))));
            */

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
