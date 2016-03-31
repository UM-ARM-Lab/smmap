#ifndef TASK_H
#define TASK_H

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/log.hpp>

#include "smmap/robot_interface.hpp"

#include "smmap/model_set.h"
#include "smmap/planner.h"
#include "smmap/task_specification.h"
#include "smmap/task_function_pointer_types.h"

namespace smmap
{
    class Task
    {
        public:
            Task( RobotInterface& robot,
                  Visualizer& vis,
                  TaskSpecification::Ptr task_specification );
            void execute();

        private:
            ////////////////////////////////////////////////////////////////////
            // Internal initialization helpers
            ////////////////////////////////////////////////////////////////////

            void initializeModelSet();
            void initializeLogging();

            ////////////////////////////////////////////////////////////////////
            // ROS objects and helpers
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            RobotInterface& robot_;
            Visualizer& vis_;

            ////////////////////////////////////////////////////////////////////
            // Task specific data
            ////////////////////////////////////////////////////////////////////

            std::shared_ptr< TaskSpecification > task_specification_;

            ////////////////////////////////////////////////////////////////////
            // Logging objects
            ////////////////////////////////////////////////////////////////////

            bool logging_enabled_;
            std::map< std::string, Log::Log > loggers;

            ////////////////////////////////////////////////////////////////////
            // Function pointers that are created in the construtor that are
            // then passed on to the models or the planner
            ////////////////////////////////////////////////////////////////////

            const ErrorFunctionType error_fn_;
            const ModelPredictionFunctionType model_prediction_fn_;
            const ModelSuggestedGrippersTrajFunctionType model_suggested_grippers_traj_fn_;
            const GetModelUtilityFunctionType get_model_utility_fn_;
            const UpdateModelUtilityFunctionType update_model_utility_fn_;
            const GripperCollisionCheckFunctionType gripper_collision_check_fn_;
            const TaskDesiredObjectDeltaFunctionType task_desired_object_delta_fn_;
            const TaskObjectDeltaProjectionFunctionType task_object_delta_projection_fn_;

            ////////////////////////////////////////////////////////////////////
            // Functions that are used to initialize function pointers in the
            // constructor. These all require that task_type_ and
            // deformable_type_ have been set already
            ////////////////////////////////////////////////////////////////////

            ErrorFunctionType createErrorFunction();
            ModelPredictionFunctionType createModelPredictionFunction();
            ModelSuggestedGrippersTrajFunctionType createModelSuggestedGrippersTrajFunction();
            GetModelUtilityFunctionType createGetModelUtilityFunction();
            UpdateModelUtilityFunctionType createUpdateModelUtilityFunction();
            GripperCollisionCheckFunctionType createGripperCollisionCheckFunction();
            TaskDesiredObjectDeltaFunctionType createTaskDesiredObjectDeltaFunction();
            TaskObjectDeltaProjectionFunctionType createTaskObjectDeltaProjectionFunction();

            ////////////////////////////////////////////////////////////////////
            //
            ////////////////////////////////////////////////////////////////////

            ModelSet model_set_;
            Planner planner_;

    };
}

#endif // TASK_H
