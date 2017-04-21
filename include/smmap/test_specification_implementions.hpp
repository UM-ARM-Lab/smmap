#ifndef TEST_SPECIFICATION_IMPLEMENTIONS_HPP
#define TEST_SPECIFICATION_IMPLEMENTIONS_HPP

#include "smmap/test_specification.h"
#include "smmap/point_reflector.hpp"

namespace smmap
{

// For debuging use: rope cylinder
class RopeCylinderCoverage : public TestSpecification
{   public:
        RopeCylinderCoverage(ros::NodeHandle& nh)
            : TestSpecification(nh, DeformableType::ROPE, TaskType::ROPE_CYLINDER_COVERAGE)
        {
            initializeGripperDelta_impl();
        }

    private:
        virtual double deformability_impl() const
        {
            return 10.0; // k
        }

        virtual double collisionScalingFactor_impl() const
        {
            return  200.0; // beta
        }

        virtual double stretchingScalingThreshold_impl() const
        {
            return 0.005; // lambda
        }

        virtual double maxTime_impl() const
        {
            return 15.0;
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std_msgs::ColorRGBA& color) const
        {
            vis.visualizeRope(marker_name, object_configuration, color);
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std::vector<std_msgs::ColorRGBA>& colors) const
        {
            vis.visualizeRope(marker_name, object_configuration, colors);
        }

        virtual double getErrorThreshold_impl() const
        {
            return 0.01;
        }

        virtual void initializeGripperDelta_impl()
        {
            grippers_pose_delta_.clear();

            for (size_t gripper_ind=0; gripper_ind< grippers_data_.size(); gripper_ind++)
            {
                kinematics::Vector6d singel_q_dot = Eigen::MatrixXd::Zero(6,1);
                singel_q_dot(0,0) = 0.002f;
                grippers_pose_delta_.push_back(singel_q_dot);
            }
        }
};



// Pull the rope in the opposite direction along table surface
class RopeTablePull : public TestSpecification
{
    public:
        RopeTablePull(ros::NodeHandle& nh)
            : TestSpecification(nh, DeformableType::ROPE, TaskType::ROPE_DRAG_OPPOSITE_TABLE)
            , table_min_x_(GetTableSurfaceX(nh) - GetTableHalfExtentsX(nh))
            , table_max_x_(GetTableSurfaceX(nh) + GetTableHalfExtentsX(nh))
            , table_min_y_(GetTableSurfaceY(nh) - GetTableHalfExtentsY(nh))
            , table_max_y_(GetTableSurfaceY(nh) + GetTableHalfExtentsY(nh))
            , table_z_(GetTableSurfaceZ(nh))
        {
            initializeGripperDelta_impl();
        }

    private:
        virtual double deformability_impl() const
        {
            return 14.0; // k
        }

        virtual double collisionScalingFactor_impl() const
        {
            return  1000.0; // beta
        }

        virtual double stretchingScalingThreshold_impl() const
        {
            return 0.03; // lambda
        }

        virtual double maxTime_impl() const
        {
            return 1.8;
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std_msgs::ColorRGBA& color) const
        {
            vis.visualizeRope(marker_name, object_configuration, color);
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std::vector<std_msgs::ColorRGBA>& colors) const
        {
            vis.visualizeRope(marker_name, object_configuration, colors);
        }

        virtual double getErrorThreshold_impl() const
        {
            return 0.002;
        }
        virtual void initializeGripperDelta_impl()
        {
            grippers_pose_delta_.clear();

            for (size_t gripper_ind=0; gripper_ind< grippers_data_.size(); gripper_ind++)
            {
                kinematics::Vector6d singel_q_dot = Eigen::MatrixXd::Zero(6,1);
                singel_q_dot(0,0) = 0.002f;
                grippers_pose_delta_.push_back(singel_q_dot);
            }
        }

    private:
        const double table_min_x_;
        const double table_max_x_;
        const double table_min_y_;
        const double table_max_y_;
        const double table_z_;
};

// Pull the rope in the same direction along table surface
class RopeTableDrag : public TestSpecification
{
    public:
        RopeTableDrag(ros::NodeHandle& nh)
            : TestSpecification(nh, DeformableType::ROPE, TaskType::ROPE_DRAG_ALONG_TABLE)
            , table_min_x_(GetTableSurfaceX(nh) - GetTableHalfExtentsX(nh))
            , table_max_x_(GetTableSurfaceX(nh) + GetTableHalfExtentsX(nh))
            , table_min_y_(GetTableSurfaceY(nh) - GetTableHalfExtentsY(nh))
            , table_max_y_(GetTableSurfaceY(nh) + GetTableHalfExtentsY(nh))
            , table_z_(GetTableSurfaceZ(nh))
        {
            initializeGripperDelta_impl();
        }

    private:
        virtual double deformability_impl() const
        {
            return 14.0; // k
        }

        virtual double collisionScalingFactor_impl() const
        {
            return  1000.0; // beta
        }

        virtual double stretchingScalingThreshold_impl() const
        {
            return 0.03; // lambda
        }

        virtual double maxTime_impl() const
        {
            return 1.8;
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std_msgs::ColorRGBA& color) const
        {
            vis.visualizeRope(marker_name, object_configuration, color);
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std::vector<std_msgs::ColorRGBA>& colors) const
        {
            vis.visualizeRope(marker_name, object_configuration, colors);
        }

        virtual double getErrorThreshold_impl() const
        {
            return 0.002;
        }

        virtual void initializeGripperDelta_impl()
        {
            grippers_pose_delta_.clear();

            for (size_t gripper_ind=0; gripper_ind< grippers_data_.size(); gripper_ind++)
            {
                kinematics::Vector6d singel_q_dot = Eigen::MatrixXd::Zero(6,1);
                singel_q_dot(0,0) = -0.002f;
                grippers_pose_delta_.push_back(singel_q_dot);
            }
        }

    private:
        const double table_min_x_;
        const double table_max_x_;
        const double table_min_y_;
        const double table_max_y_;
        const double table_z_;
};

// Drag the segment of rope hanging from the table toward the table direction
class RopeTowardTable : public TestSpecification
{
    public:
        RopeTowardTable(ros::NodeHandle& nh)
            : TestSpecification(nh, DeformableType::ROPE, TaskType::ROPE_TOWARD_TABLE)
            , table_min_x_(GetTableSurfaceX(nh) - GetTableHalfExtentsX(nh))
            , table_max_x_(GetTableSurfaceX(nh) + GetTableHalfExtentsX(nh))
            , table_min_y_(GetTableSurfaceY(nh) - GetTableHalfExtentsY(nh))
            , table_max_y_(GetTableSurfaceY(nh) + GetTableHalfExtentsY(nh))
            , table_z_(GetTableSurfaceZ(nh))
        {
            initializeGripperDelta_impl();
        }

    private:
        virtual double deformability_impl() const
        {
            return 14.0; // k
        }

        virtual double collisionScalingFactor_impl() const
        {
            return  1000.0; // beta
        }

        virtual double stretchingScalingThreshold_impl() const
        {
            return 0.03; // lambda
        }

        virtual double maxTime_impl() const
        {
            return 1.8;
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std_msgs::ColorRGBA& color) const
        {
            vis.visualizeRope(marker_name, object_configuration, color);
        }

        virtual void visualizeDeformableObject_impl(
                Visualizer& vis,
                const std::string& marker_name,
                const ObjectPointSet& object_configuration,
                const std::vector<std_msgs::ColorRGBA>& colors) const
        {
            vis.visualizeRope(marker_name, object_configuration, colors);
        }

        virtual double getErrorThreshold_impl() const
        {
            return 0.002;
        }

        virtual void initializeGripperDelta_impl()
        {
            grippers_pose_delta_.clear();

            for (size_t gripper_ind=0; gripper_ind< grippers_data_.size(); gripper_ind++)
            {
                kinematics::Vector6d singel_q_dot = Eigen::MatrixXd::Zero(6,1);
                singel_q_dot(0,0) = -0.0001f;
                singel_q_dot(2,0) = -0.001f;
                grippers_pose_delta_.push_back(singel_q_dot);
            }
        }


    private:
        const double table_min_x_;
        const double table_max_x_;
        const double table_min_y_;
        const double table_max_y_;
        const double table_z_;
};


}

#endif // TEST_SPECIFICATION_IMPLEMENTIONS_HPP
