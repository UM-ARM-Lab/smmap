#ifndef TASK_H
#define TASK_H

#include <actionlib/client/simple_action_client.h>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/log.hpp>

#include "smmap/gripper_collision_checker.h"
#include "smmap/model_set.h"
#include "smmap/planner.h"
#include "smmap/task_specification.h"
#include "smmap/task_function_pointer_types.h"

namespace smmap
{
    class Task
    {
        public:
            Task( ros::NodeHandle& nh );
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

            // Our internal version of ros::spin()
            static void spin( double loop_rate );

            std::vector< WorldState > sendGripperTrajectory(
                    const smmap_msgs::CmdGrippersTrajectoryGoal& goal );

            smmap_msgs::CmdGrippersTrajectoryGoal noOpTrajectoryGoal( size_t num_no_op );

            smmap_msgs::CmdGrippersTrajectoryGoal toRosGoal(
                    const AllGrippersPoseTrajectory& trajectory );

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            GripperCollisionChecker gripper_collision_checker_;
            Visualizer vis_;
            actionlib::SimpleActionClient< smmap_msgs::CmdGrippersTrajectoryAction > cmd_grippers_traj_client_;

            ////////////////////////////////////////////////////////////////////
            // Task specific data
            ////////////////////////////////////////////////////////////////////

            std::shared_ptr< TaskSpecification > task_specification_;
            std::vector< GripperData > grippers_data_;

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

            double updateUtility( const double old_utility,
                                  const WorldState& world_state,
                                  const ObjectPointSet& prediction,
                                  const Eigen::VectorXd& weights ) const;

            ////////////////////////////////////////////////////////////////////
            //
            ////////////////////////////////////////////////////////////////////

            ModelSet model_set_;
            Planner planner_;

    };
}

#endif // TASK_H
