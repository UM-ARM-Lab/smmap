#ifndef LEAST_SQUARES_CONTROLLER_RANDOM_SAMPLING_H
#define LEAST_SQUARES_CONTROLLER_RANDOM_SAMPLING_H

#include <sdf_tools/sdf.hpp>

#include "smmap/deformable_controller.hpp"
#include "smmap/grippers.hpp"
#include "smmap/robot_interface.hpp"
#include "smmap/nomad_solvers.h"

namespace smmap
{
    struct GripperStretchingInfo
    {
        GripperStretchingInfo()
        {}
        GripperStretchingInfo(ssize_t num_x_steps, ssize_t num_y_steps)
            : node_xy_ind(num_x_steps, num_y_steps)
        {}
        GripperStretchingInfo(
                ssize_t num_x_steps,
                ssize_t num_y_steps,
                const GripperData& gripper_data)
            : node_xy_ind(num_x_steps, num_y_steps)
            , attatched_nodes_(gripper_data.node_indices_)
        {
            setEdgeNodes();
        }

        void setEdgeNodes()
        {
            from_nodes_.clear();
            to_nodes_.clear();

            for(int attatched_ind = 0; attatched_ind < attatched_nodes_.size(); attatched_ind++)
            {
                bool is_cloth_edge = false;
                bool is_boundary_attatched = false;
                std::vector<ssize_t> node_neighbor = node_xy_ind.Neighbor8Ind(attatched_nodes_.at(attatched_ind));

                // if out of cloth, assign -1; if on cloth && attatched, assign -2
                for (int neighbor_ind = 0; neighbor_ind < node_neighbor.size(); neighbor_ind++)
                {
                    if(node_neighbor.at(neighbor_ind) == -1)
                    {
                        is_cloth_edge == true;
                    }
                    if(!isAttached(node_neighbor.at(neighbor_ind)))
                    {
                        is_boundary_attatched == true;
                    }
                    else
                    {
                        node_neighbor.at(neighbor_ind) = -2;
                    }
                }

                if(is_boundary_attatched)
                {
                    from_nodes_.push_back(attatched_nodes_.at(attatched_ind));
                    to_nodes_.push_back(node_neighbor);
                }
            }
        }

        void setGripperStretchingInfo(
                ssize_t num_x_steps,
                ssize_t num_y_steps,
                const GripperData& gripper_data)
        {
            node_xy_ind.SetNodeXYInd(num_x_steps, num_y_steps);
            attatched_nodes_ = gripper_data.node_indices_;
            setEdgeNodes();
        }

        bool isAttached(ssize_t node_ind)
        {
            for(int attatched_ind = 0; attatched_ind < attatched_nodes_.size(); attatched_ind++)
            {
                if (node_ind == attatched_nodes_.at(attatched_ind))
                    return true;
            }
            return false;
        }

        NodeXYInd node_xy_ind;
        std::vector<long> attatched_nodes_;
        std::vector<ssize_t> from_nodes_;
        // each sub-vector is of size 8; -1 out of bound, -2 attached
        /* Layout :
         *   3  2  1
         *   4  X  0
         *   5  6  7
        */
        std::vector<std::vector<ssize_t>> to_nodes_;
    };

    class LeastSquaresControllerRandomSampling : public DeformableController
    {
        public:
            LeastSquaresControllerRandomSampling(
                    ros::NodeHandle& nh,
                    ros::NodeHandle& ph,
                    RobotInterface& robot,
                    const sdf_tools::SignedDistanceField& sdf,
                    std::mt19937_64& generator,
                    Visualizer& vis,
                    GripperControllerType gripper_controller_type,
                    const DeformableModel::Ptr& deformable_model,
                    const int64_t max_count,
                    const double distance_to_obstacle_threshold);

            //////////////////////////////////////////////////////////////////////////////////////
            // Called from outside to find the optimal gripper command
            //////////////////////////////////////////////////////////////////////////////////////

            void setGripperControllerType(GripperControllerType gripper_controller_type);

            long getStretchingViolationCount();


        private:
            /////////////////////////////////////////////////////////////////////////////////////////
            // Optimization function
            /////////////////////////////////////////////////////////////////////////////////////////

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> getGripperMotion_impl(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity);

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByRandomSampling(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity);

            std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> solvedByNomad(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const double max_gripper_velocity);



            /////////////////////////////////////////////////////////////////////////////////////////
            // Helper function
            /////////////////////////////////////////////////////////////////////////////////////////

            kinematics::Vector6d singleGripperPoseDeltaSampler(const double max_delta);

            AllGrippersSinglePoseDelta allGripperPoseDeltaSampler(const ssize_t num_grippers, const double max_delta);

            AllGrippersSinglePoseDelta setAllGripperPoseDeltaZero(const ssize_t num_grippers);

            double errorOfControlByPrediction(const ObjectPointSet predicted_object_p_dot,
                                              const Eigen::VectorXd &desired_object_p_dot,
                                              const Eigen::VectorXd &desired_p_dot_weight) const;

            void visualize_stretching_vector(const ObjectPointSet& object_configuration);

            void visualize_rope_stretching_vector(const ObjectPointSet& object_configuration);

            void visualize_cloth_stretching_vector(const ObjectPointSet& object_configuration);

            void visualize_gripper_motion(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta& gripper_motion);

            /////////////////////////////////////////////////////////////////////////////////////////
            // Collision constraint related function
            /////////////////////////////////////////////////////////////////////////////////////////

            bool gripperCollisionCheckResult(
                    const AllGrippersSinglePose& current_gripper_pose,
                    const AllGrippersSinglePoseDelta &test_gripper_motion);

            /////////////////////////////////////////////////////////////////////////////////////////
            // Stretching constraint related function
            /////////////////////////////////////////////////////////////////////////////////////////

            bool stretchingDetection(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            bool ropeTwoGrippersStretchingDetection(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

            bool clothTwoGrippersStretchingDetection(
                    const DeformableModel::DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& test_gripper_motion);

        public:
            const Eigen::MatrixXd object_initial_node_distance_;
            double max_grippers_distance_;

        private:
            GripperCollisionChecker gripper_collision_checker_;

            const std::vector<GripperData> grippers_data_;

            const sdf_tools::SignedDistanceField enviroment_sdf_;
            std::mt19937_64& generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;

            Visualizer& vis_;

            GripperControllerType gripper_controller_type_;
            const DeformableType deformable_type_;
            const TaskType task_type_;
            const DeformableModel::Ptr model_;

            const double distance_to_obstacle_threshold_;
            double max_stretch_factor_;
            double stretching_cosine_threshold_;

            const int64_t max_count_;
            int sample_count_;

            bool fix_step_;
            bool previous_over_stretch_state_;
            bool over_stretch_;

            // cloth node inde conversion helper
            std::vector<std::unique_ptr<GripperStretchingInfo>> grippers_stretching_helper_;


    };

}

#endif // LEAST_SQUARES_CONTROLLER_RANDOM_SAMPLING_H
