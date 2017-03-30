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
            static double CalculateErrorWithTheshold(
                    const ObjectPointSet& object_congif_cur,
                    const double minimum_threshold);

            // Do the helper function things:
            // p_Delta_real = p_current-p_last
            // p_Delta_model = J*q_Delta_last
            void UpdateState(const ObjectPointSet& object_congif_cur);


            /*
            static ObjectDeltaAndWeight CalculateObjectErrorCorrectionDeltaWithThreshold(
                    const ObjectPointSet& target_points,
                    const ObjectPointSet& deformable_object,
                    const double minimum_threshold);
            */

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

            double defaultDeformability() const;        // k
            double collisionScalingFactor() const;      // beta (or k2)
            double stretchingScalingThreshold() const;  // lambda
            double maxTime() const;                     // max simulation time when scripting things

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

            double calculateError(
                    const ObjectPointSet& object_configuration) const;

            /**
             * @brief calculateObjectDesiredDelta
             * @param world_state
             * @return return.first is the desired movement of the object
             *         return.second is the importance of that part of the movement
             */
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

        private:
            ////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that each task specification must provide
            ////////////////////////////////////////////////////////////////////////////////

            virtual double deformability_impl() const = 0;              // k
            virtual double collisionScalingFactor_impl() const = 0;     // beta (or k2)
            virtual double stretchingScalingThreshold_impl() const = 0; // lambda
            virtual double maxTime_impl() const = 0;                    // max simulation time when scripting things

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

            virtual double calculateError_impl(
                    const ObjectPointSet& object_configuration) const = 0;

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state) const = 0;
    };



    class CoverageTask : public TestSpecification
    {
        public:
            CoverageTask(ros::NodeHandle& nh, const DeformableType deformable_type, const TaskType task_type);

            double getErrorThreshold() const;

        protected:
            virtual double getErrorThreshold_impl() const = 0;

            /// Stores the points that we are trying to cover with the rope
            const ObjectPointSet cover_points_;
            const ssize_t num_cover_points_;

        private:

            virtual double calculateError_impl(
                    const ObjectPointSet& current_configuration) const final;
    };

 }

#endif // TEST_SPECIFICATION_H
