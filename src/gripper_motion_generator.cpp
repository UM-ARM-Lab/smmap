#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>

#include "smmap/gripper_motion_generator.h"
#include "smmap/task.h"

using namespace smmap;

GripperMotionGenerator::GripperMotionGenerator()
{

}

////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////

GripperCollisionCheckFunctionType GripperMotionGenerator::createGripperCollisionCheckFunction()
{
    return std::bind(&RobotInterface::checkGripperCollision,
                     &robot_,
                     std::placeholders::_1);
}

AllGrippersSinglePoseDelta GripperMotionGenerator::sampleGripperMotion()
{

}



AllGrippersSinglePoseDelta GripperMotionGenerator::findOptimalGripperMotion()
{
    for (int ind_sample = 0; ind_sample < num_samples_; ind_sample++)
    {
        AllGrippersSinglePoseDelta sample_grippers_motion = samplingGripperMotion();

    }

}


