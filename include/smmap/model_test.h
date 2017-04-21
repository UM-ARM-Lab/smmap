#ifndef MODEL_TEST_H
#define MODEL_TEST_H

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/log.hpp>

#include "smmap/robot_interface.hpp"
#include "smmap/test_planner.h"
#include "smmap/test_specification.h"
#include "smmap/task_function_pointer_types.h"

namespace smmap
{
    class modelTest
    {
        public:
            modelTest(RobotInterface& robot,
                 Visualizer& vis,
                 const TestSpecification::Ptr& test_specification);
            void execute();

        private:
            ////////////////////////////////////////////////////////////////////
            // Internal initialization helpers
            ////////////////////////////////////////////////////////////////////

            void initializeModelSet(const WorldState& initial_world_state);
            void initializeLogging();

            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            RobotInterface& robot_;
            Visualizer& vis_;

            // Could be useful
//            std::vector<GripperData>& gripperData_;

            ////////////////////////////////////////////////////////////////////
            // Task specific data
            ////////////////////////////////////////////////////////////////////

            std::shared_ptr<TestSpecification> test_specification_;

            ////////////////////////////////////////////////////////////////////
            // Logging objects
            ////////////////////////////////////////////////////////////////////

            bool logging_enabled_;
            std::map<std::string, Log::Log> loggers;
            std::map<std::string, Log::Log> test_loggers_;  //  LOG JUST FOR TEST MODEL USAGE

            void logData(
                    const WorldState& current_world_state,
                    const Eigen::VectorXd& model_utility_mean,
                    const Eigen::MatrixXd& model_utility_covariance,
                    const ssize_t model_used,
                    const std::vector<double>& rewards_for_all_models,
                    const double correlation_strength_factor);


            //////////////////////// Mengyao, define for test //////////////////
            // More Log data to be added later:
            // Constraint violation
            // Test log data
            void testLogData(const WorldState& current_world_state,
                             const ObjectPointSet &real_delta_p,
                             const ObjectPointSet &model_delta_p,
                             Eigen::MatrixXd &real_time_error, Eigen::MatrixXd &constraint_violation);


            ////////////////////////////////////////////////////////////////////
            // Function pointers that are created in the construtor that are
            // then passed on to the models or the planner
            ////////////////////////////////////////////////////////////////////

//            const ErrorFunctionType error_fn_;
            const GripperCollisionCheckFunctionType gripper_collision_check_fn_;
//            const TaskObjectDeltaProjectionFunctionType task_object_delta_projection_fn_;
//            const TaskExecuteGripperTrajectoryFunctionType execute_trajectory_fn_;
//            const TestGrippersPosesFunctionType test_grippers_poses_fn_;
            const LoggingFunctionType logging_fn_;
            const TestLoggingFunctionType test_logging_fn_;  // log for test


            ////////////////////////////////////////////////////////////////////
            // Functions that are used to initialize function pointers in the
            // constructor. These all require that task_type_ and
            // deformable_type_ have been set already
            ////////////////////////////////////////////////////////////////////

//            ErrorFunctionType createErrorFunction();
            GripperCollisionCheckFunctionType createGripperCollisionCheckFunction();
//            TaskObjectDeltaProjectionFunctionType createTaskObjectDeltaProjectionFunction();
//            TaskExecuteGripperTrajectoryFunctionType createExecuteGripperTrajectoryFunction();
//            TestGrippersPosesFunctionType createTestGrippersPosesFunction();
            LoggingFunctionType createLoggingFunction();
            TestLoggingFunctionType createTestLogFunction();

            ////////////////////////////////////////////////////////////////////
            // The planner itself
            ////////////////////////////////////////////////////////////////////

            TestPlanner planner_;

    };
}

#endif // TASK_H
