#ifndef DEFORMABLE_MODEL_H
#define DEFORMABLE_MODEL_H

#include <atomic>
#include <random>

#include "smmap/task_function_pointer_types.h"
#include "smmap/grippers.hpp"

namespace smmap
{
    class DeformableModel
    {
        public:
            typedef std::shared_ptr<DeformableModel> Ptr;

            DeformableModel();

            ////////////////////////////////////////////////////////////////////
            // Virtual functions that define the interface
            ////////////////////////////////////////////////////////////////////

            virtual void updateModel(const std::vector<WorldState>& feedback) = 0;

            virtual ObjectTrajectory getPrediction(
                    const WorldState& world_initial_state,
                    const AllGrippersPoseTrajectory& gripper_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& gripper_pose_delta_trajectory,
                    const double dt) const = 0;

            virtual ObjectPointSet getFinalConfiguration(
                    const WorldState& world_initial_state,
                    const AllGrippersPoseTrajectory& gripper_pose_trajectory,
                    const AllGrippersPoseDeltaTrajectory& gripper_pose_delta_trajectory,
                    const double dt) const = 0;

            virtual std::pair<AllGrippersPoseTrajectory, ObjectTrajectory> getSuggestedGrippersTrajectory(
                    const WorldState& world_initial_state,
                    const size_t planning_horizion,
                    const double dt,
                    const double max_gripper_velocity,
                    const double obstacle_avoidance_scale) const = 0;

            virtual void perturbModel(std::mt19937_64& generator) = 0;

            ////////////////////////////////////////////////////////////////////
            // Update/Set function for static member
            ////////////////////////////////////////////////////////////////////

            static void SetGrippersData(const std::vector<GripperData>& grippers_data);

            static void SetCallbackFunctions(
                    const ErrorFunctionType& error_fn,
                    const GripperCollisionCheckFunctionType& gripper_collision_check_fn,
                    const TaskDesiredObjectDeltaFunctionType& task_desired_object_delta_fn,
                    const TaskObjectDeltaProjectionFunctionType& task_object_delta_projection_fn);

        protected:

            ////////////////////////////////////////////////////////////////////
            // Destructor that prevents "delete pointer to base object"
            ////////////////////////////////////////////////////////////////////

            ~DeformableModel() {}

            ////////////////////////////////////////////////////////////////////
            // Static data
            ////////////////////////////////////////////////////////////////////

            static std::atomic_bool grippers_data_initialized_;
            static std::vector<GripperData> grippers_data_;

            static std::atomic_bool function_pointers_initialized_;
            static ErrorFunctionType error_fn_;
            static GripperCollisionCheckFunctionType gripper_collision_check_fn_;
            static TaskDesiredObjectDeltaFunctionType task_desired_object_delta_fn_;
            static TaskObjectDeltaProjectionFunctionType task_object_delta_projection_fn_;
    };
}

#endif // DEFORMABLE_MODEL_H
