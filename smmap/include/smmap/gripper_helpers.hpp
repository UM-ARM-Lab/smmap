#ifndef GRIPPER_HELPERS_HPP
#define GRIPPER_HELPERS_HPP

#include <limits>
#include <memory>
#include <assert.h>

#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <kinematics_toolbox/kinematics.h>

namespace smmap
{
    typedef EigenHelpers::VectorAffine3d AllGrippersSinglePose;
    typedef std::vector<AllGrippersSinglePose> AllGrippersPoseTrajectory;

    typedef kinematics::VectorVector6d AllGrippersSingleVelocity;
    typedef AllGrippersSingleVelocity AllGrippersSinglePoseDelta;
    typedef std::vector<AllGrippersSingleVelocity> AllGrippersVelocityTrajectory;
    typedef std::vector<AllGrippersSinglePoseDelta> AllGrippersPoseDeltaTrajectory;

    struct GripperData
    {
        GripperData(const std::string& name, const std::vector<long>& node_indices)
            : name(name)
            , node_indices(node_indices)
        {}

        /// The name associated with this gripper
        std::string name;

        /// Vector of the indices of the nodes that are grasped by the gripper
        std::vector<long> node_indices;

        /**
         * @brief operator <<
         * @param out The stream to output the data too
         * @param data The gripper data to output
         * @return
         */
        friend std::ostream& operator<< (std::ostream& out, const GripperData& data)
        {
            out << data.name << " Node Indices: " << PrettyPrint::PrettyPrint(data.node_indices);
            return out;
        }
    };

    template<typename T>
    inline std::vector<long > VectorAnytypeToVectorLong(const std::vector< T>& vector_anytype)
    {
        std::vector<long> vector_signed(vector_anytype.size());
        for (size_t ind = 0; ind < vector_anytype.size(); ind++)
        {
            vector_signed[ind] = (long)(vector_anytype[ind]);
        }
        return vector_signed;
    }

    inline std::vector<std::string> GetGripperNames(const std::vector<GripperData> grippers_data)
    {
        std::vector<std::string> names(grippers_data.size());

        for (size_t gripper_ind = 0; gripper_ind < grippers_data.size(); gripper_ind++)
        {
            names[gripper_ind] = grippers_data[gripper_ind].name;
        }

        return names;
    }

    /**
     * @brief getMinimumDistanceToGripper
     * @param gripper_indices The indices of the nodes that the gripper is in contact with
     * @param node_index The index of the node that we want to get the distance to
     * @param object_initial_node_distance The matrix of distances between nodes
     * @return The index of the closest node grasped by the gripper and the distance to that node
     */
    inline std::pair<long, double> getMinimumDistanceToGripper(
            const std::vector<long>& gripper_indices, long node_index,
            const Eigen::MatrixXd& object_initial_node_distance)
    {
        double min_dist = std::numeric_limits<double>::infinity();
        long min_ind = -1;

        for (long ind: gripper_indices)
        {
            if (object_initial_node_distance(ind, node_index) < min_dist)
            {
                min_dist = object_initial_node_distance(ind, node_index);
                min_ind = ind;
            }
        }

        return std::pair< long, double>(min_ind, min_dist);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Dot products
    ////////////////////////////////////////////////////////////////////////////

    inline double GripperVelocityDotProduct(
            const kinematics::Vector6d& vel1,
            const kinematics::Vector6d& vel2)
    {
        kinematics::Vector6d weight = kinematics::Vector6d::Ones();
        weight(3) = 1.0/20.0;
        weight(4) = 1.0/20.0;
        weight(5) = 1.0/20.0;
        weight.array() = weight.array().square();

        return EigenHelpers::WeightedDotProduct(vel1, vel2, weight);
    }

    inline double MultipleGrippersVelocityDotProduct(
            const AllGrippersSinglePoseDelta& vel1,
            const AllGrippersSinglePoseDelta& vel2)
    {
        assert(vel1.size() == vel2.size());

        double dot_product = 0;
        for (size_t vel_ind = 0; vel_ind < vel1.size(); vel_ind++)
        {
            dot_product += GripperVelocityDotProduct(vel1[vel_ind], vel2[vel_ind]);
        }

        return dot_product;
    }

    inline double MultipleGrippersVelocityTrajectoryDotProduct(
            const AllGrippersVelocityTrajectory& traj1,
            const AllGrippersVelocityTrajectory& traj2)
    {
        assert(traj1.size() == traj2.size());

        double dot_product = 0;
        for (size_t time_ind = 0; time_ind < traj1.size(); time_ind++)
        {
            dot_product += MultipleGrippersVelocityDotProduct(traj1[time_ind], traj2[time_ind]);
        }

        return dot_product;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Norms induced by said dot products
    ////////////////////////////////////////////////////////////////////////////

    inline double GripperVelocity6dSquaredNorm(const kinematics::Vector6d& gripper_velocity)
    {
        return GripperVelocityDotProduct(gripper_velocity, gripper_velocity);
    }

    inline double GripperVelocity6dNorm(const kinematics::Vector6d& gripper_velocity)
    {
        return std::sqrt(GripperVelocity6dSquaredNorm(gripper_velocity));
    }

    inline double MultipleGrippersVelocity6dSquaredNorm(const AllGrippersSingleVelocity& grippers_velocity)
    {
        double squared_norm = 0;
        for (size_t gripper_ind = 0; gripper_ind < grippers_velocity.size(); gripper_ind++)
        {
            squared_norm += GripperVelocity6dSquaredNorm(grippers_velocity[gripper_ind]);
        }
        return squared_norm;
    }

    inline double MultipleGrippersVelocity6dSquaredNorm(const Eigen::VectorXd& grippers_velocity)
    {
        assert(grippers_velocity.size() % 6 == 0);

        double squared_norm = 0;
        for (long gripper_ind = 0; gripper_ind < grippers_velocity.size(); gripper_ind += 6)
        {
            squared_norm += GripperVelocity6dSquaredNorm(grippers_velocity.segment<6>(gripper_ind));
        }
        return squared_norm;
    }

    inline double MultipleGrippersVelocity6dNorm(const AllGrippersSingleVelocity& grippers_velocity)
    {
        return std::sqrt(MultipleGrippersVelocity6dSquaredNorm(grippers_velocity));
    }

    inline double MultipleGrippersVelocity6dNorm(const Eigen::VectorXd& grippers_velocity)
    {
        return std::sqrt(MultipleGrippersVelocity6dSquaredNorm(grippers_velocity));
    }

    inline double MultipleGrippersVelocityTrajectory6dSquaredNorm(const AllGrippersVelocityTrajectory& grippers_trajectory)
    {
        double squared_norm = 0;
        for (size_t time_ind = 0; time_ind < grippers_trajectory.size(); time_ind++)
        {
            squared_norm += MultipleGrippersVelocity6dSquaredNorm(grippers_trajectory[time_ind]);
        }

        return squared_norm;
    }

    inline double MultipleGrippersVelocityTrajectory6dNorm(const AllGrippersVelocityTrajectory& grippers_trajectory)
    {
        return std::sqrt(MultipleGrippersVelocityTrajectory6dSquaredNorm(grippers_trajectory));
    }

    inline void ClampGripperVelocities(Eigen::VectorXd& velocities, const double max_vel)
    {
        assert(velocities.size() % 6 == 0);

        for (long gripper_ind = 0; gripper_ind < velocities.size(); gripper_ind += 6)
        {
            const double velocity_norm = GripperVelocity6dNorm(velocities.segment<6>(gripper_ind));
            if (velocity_norm > max_vel)
            {
                velocities.segment<6>(gripper_ind) *= max_vel / velocity_norm;
            }
        }

    }
}

#endif // GRIPPER_HELPERS_HPP
