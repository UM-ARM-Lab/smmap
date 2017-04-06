#ifndef TEST_SPECIFICATION_H
#define TEST_SPECIFICATION_H

#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <Eigen/Dense>
#include <arc_utilities/dijkstras.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <sdf_tools/sdf.hpp>
#include <deformable_manipulation_experiment_params/task_enums.h>
#include <deformable_manipulation_experiment_params/xyzgrid.h>

#include "smmap/ros_communication_helpers.hpp"
#include "smmap/task_function_pointer_types.h"
#include "smmap/visualization_tools.h"
#include "smmap/timing.hpp"

namespace smmap
{
    class TestSpecification
    {
        public:
            ////////////////////////////////////////////////////////////////////
            // Static helper functions - could be private given how they are
            // used but making public as they are static
            ////////////////////////////////////////////////////////////////////

            // This function should calculate error at each step
            // error = norm(p_Delta_real-p_Delta_model)
            // p_Delta_real = p_current-p_last
            // p_Delta_model = J*q_Delta_last
            // It is called by CalculateError_impl in CalculateError, minimum_threshold depend on test
            static double CalculateErrorWithTheshold(
                    const ObjectPointSet& real_delta_p,
                    ObjectDeltaAndWeight& model_delta_p,
                    const double minimum_threshold);

            /*
            // Do the helper function things:
            // p_Delta_real = p_current-p_last
            // p_Delta_model = J*q_Delta_last
            void UpdateState(const ObjectPointSet& object_congif_cur);
            */

            // This function previously set the target delta p
            // TODO: get a target dela_p from grippers delta q

            static ObjectDeltaAndWeight CalculateObjectErrorCorrectionDeltaWithThreshold();


        public:
            typedef std::shared_ptr<TestSpecification> Ptr;

            ////////////////////////////////////////////////////////////////////
            // Constructor to initialize objects that all TaskSpecifications share
            ////////////////////////////////////////////////////////////////////

            TestSpecification(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type);
            TestSpecification(ros::NodeHandle& nh, Visualizer vis, const DeformableType deformable_type, const TaskType task_type);

            ////////////////////////////////////////////////////////////////////
            // Static builder function
            ////////////////////////////////////////////////////////////////////

            static TestSpecification::Ptr MakeTaskSpecification(
                    ros::NodeHandle& nh);

            ////////////////////////////////////////////////////////////////////
            // Virtual function wrappers
            ////////////////////////////////////////////////////////////////////

            // Implement in its only impl function depend on the type of test
            double defaultDeformability() const;        // k
            double collisionScalingFactor() const;      // beta (or k2)
            double stretchingScalingThreshold() const;  // lambda
            double maxTime() const;                     // max simulation time when scripting things

            //////////// Mengyao: Initialize delta_q ///////////////////////////
            void initializeGripperDelta(ros::NodeHandle& nh) const;

            void visualizeDeformableObject(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const;

            void visualizeDeformableObject(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const;

            // Should Call the CalculateErrorWithThreshol, INPUT BEING REVISE BY Mengyao
            double calculateError(
                    const ObjectPointSet& real_delta_p,
                    ObjectDeltaAndWeight& model_delta_p) const;

            /**
             * @brief calculateObjectDesiredDelta
             * @param world_state
             * @return return.first is the desired movement of the object
             *         return.second is the importance of that part of the movement
             */
            // Planned P_dot; in the test, should be a function that set all nodes zeros;
            // Except for the grapsed point, whose value should be the same as that of
            // their corresponding end-effectors
            ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta(
                    const WorldState& world_state) const;

            ////////////////////////////////////////////////////////////////////
            // Helper functions
            // TODO: Should these be virtual? virtual final?
            ////////////////////////////////////////////////////////////////////

            /**
             * @brief calculateStretchingCorrectionDelta
             * @param world_state
             * @return
             */
            ObjectDeltaAndWeight calculateStretchingCorrectionDelta(
                    const ObjectPointSet& object_configuration,
                    bool visualize) const;

            /**
             * @brief computeStretchingCorrection
             * @param object_configuration
             * @return
             */
            ObjectDeltaAndWeight calculateStretchingCorrectionDelta(
                    const WorldState& world_state,
                    bool visualize) const;

            /**
             * @brief calculateStretchingError
             * @param object_configuration
             * @return
             */
            double calculateStretchingError(
                    const ObjectPointSet& object_configuration) const;

            /**
             * @brief calculateStretchingError
             * @param world_state
             * @return
             */
            double calculateStretchingError(
                    const WorldState& world_state) const;

            /**
             * @brief combineErrorCorrectionAndStretchingCorrection
             * @param error_correction
             * @param stretching_correction
             * @return
             */
            ObjectDeltaAndWeight combineErrorCorrectionAndStretchingCorrection(
                    const ObjectDeltaAndWeight& error_correction,
                    const ObjectDeltaAndWeight& stretching_correction) const;


            // This is the Final target_delta_p, = calculated_delta_p (from endeffector)+stretching correction
            // calculated_delta_p in this test class should map the preset delta q to desired delta p
            ObjectDeltaAndWeight calculateDesiredDirection(const WorldState& world_state);

        private:
            ObjectDeltaAndWeight first_step_desired_motion_;
            ObjectDeltaAndWeight first_step_error_correction_;
            ObjectDeltaAndWeight first_step_stretching_correction_;
            std::atomic_bool first_step_calculated_;
            std::mutex first_step_mtx_;
            double sim_time_last_time_first_step_calced_;


        public:
            // Records of task and deformable type if various visualizers or whatever need them
            const DeformableType deformable_type_;
            const TaskType task_type_;
            #warning "This ought to be like a Java final"

            // This variable is useless in test; While just keep it here for safe
            bool is_dijkstras_type_task_;

        protected:
            ////////////////////////////////////////////////////////////////////
            // Prevent deletion of base pointer
            ////////////////////////////////////////////////////////////////////

            ~TestSpecification() {}

            ////////////////////////////////////////////////////////////////////
            // Objects shared by all task specifications
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            Visualizer vis_;
            const Eigen::MatrixXd object_initial_node_distance_;
            const ssize_t num_nodes_;

            // NEW DEFINE
            const AllGrippersSinglePoseDelta grippers_pose_delta_;
            const std::vector<GripperData> grippers_data_;
//            WorldState last_world_state_;


        private:
            ////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that each task specification must provide
            ////////////////////////////////////////////////////////////////////////////////

            // To be implemented in each specific test
            virtual double deformability_impl() const = 0;              // k
            virtual double collisionScalingFactor_impl() const = 0;     // beta (or k2)
            virtual double stretchingScalingThreshold_impl() const = 0; // lambda
            virtual double maxTime_impl() const = 0;                    // max simulation time when scripting things

            //////////// Mengyao: Initialize delta_q ///////////////////////////
            virtual void initializeGripperDelta_impl(ros::NodeHandle& nh) const = 0;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color) const = 0;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector<std_msgs::ColorRGBA>& colors) const = 0;

            // Should Call the CalculateErrorWithThreshol, INPUT BEING REVISE BY Mengyao
            virtual double calculateError_impl(
                    const ObjectPointSet& real_delta_p,
                    ObjectDeltaAndWeight& model_delta_p) const = 0;

            // delta_p = target/planned delta_p
            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const = 0;
    };
}

#endif // TEST_SPECIFICATION_H
