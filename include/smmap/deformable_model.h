#ifndef DEFORMABLE_MODEL_H
#define DEFORMABLE_MODEL_H

#include <atomic>

#include "smmap/task_function_pointer_types.h"
#include "smmap/grippers.hpp"

namespace smmap
{
    class DeformableModel
    {
        public:
            typedef std::shared_ptr<DeformableModel> Ptr;

            struct DeformableModelInputData
            {
                public:
                    DeformableModelInputData(
                            const WorldState& world_current_state,
                            const ObjectDeltaAndWeight& desired_object_motion,
                            const double dt)
                        : world_current_state_(world_current_state)
                        , desired_object_motion_(desired_object_motion)
                        , dt_(dt)
                    {}

                    const WorldState& world_current_state_;
                    const ObjectDeltaAndWeight desired_object_motion_;
                    const double dt_;
            };

            DeformableModel();

            ////////////////////////////////////////////////////////////////////
            // Virtual functions that define the interface
            ////////////////////////////////////////////////////////////////////

            void updateModel(const WorldState& previous, const WorldState& next);

            ObjectPointSet getObjectDelta(
                    const DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta);

            ////////////////////////////////////////////////////////////////////
            // Update/Set function for static members
            ////////////////////////////////////////////////////////////////////

            static void SetGrippersData(
                    const std::vector<GripperData>& grippers_data);

            static const std::vector<GripperData>& GetGrippersData();

            static void SetCallbackFunctions(
                    const GripperCollisionCheckFunctionType& gripper_collision_check_fn);


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
            static GripperCollisionCheckFunctionType gripper_collision_check_fn_;

        private:
            virtual void updateModel_impl(const WorldState& previous, const WorldState& next) = 0;

            virtual ObjectPointSet getObjectDelta_impl(
                    const DeformableModelInputData& input_data,
                    const AllGrippersSinglePoseDelta& grippers_pose_delta) const = 0;
    };
}

#endif // DEFORMABLE_MODEL_H
