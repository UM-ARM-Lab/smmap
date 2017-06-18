#ifndef GRIPPER_MOTION_GENERATOR_H
#define GRIPPER_MOTION_GENERATOR_H

#include "smmap/deformable_model.h"
#include "smmap/grippers.hpp"

#include "sdf_tools/sdf.hpp"
#include "smmap/robot_interface.hpp"

namespace smmap {

    class GripperMotionGenerator
    {
        public:
            GripperMotionGenerator();

            GripperCollisionCheckFunctionType createGripperCollisionCheckFunction();

            AllGrippersSinglePoseDelta samplingGripperMotion();

            //////////////////////////////////////////////////////////////////
            // Called from outside to find the optimal gripper command
            //////////////////////////////////////////////////////////////////
            AllGrippersSinglePoseDelta findOptimalGripperMotion();


        private:

            ros::NodeHandle nh_;

            DeformableModel::Ptr deformable_model_;
            sdf_tools::SignedDistanceField enviroment_sdf_;
            GripperCollisionChecker gripper_colision_checker_;

            AllGrippersSinglePoseDelta grippers_pose_delta_;
            std::vector<GripperData> grippers_data_;

            const ssize_t num_samples_;
            const double rotation_lower_bound_;
            const double rotation_upper_bound_;
            const double translation_lower_bound_;
            const double translation_upper_bound_;

//            Visualizer& vis_;


//            const GripperMotionLoggingFunctionType logging_fn_;





    };


}

#endif // GRIPPER_MOTION_GENERATOR_H
