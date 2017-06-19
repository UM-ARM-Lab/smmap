#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>

#include "smmap/gripper_motion_generator.h"
#include "smmap/task.h"

using namespace smmap;

GripperMotionGenerator::GripperMotionGenerator(ros::NodeHandle &nh,
        const sdf_tools::SignedDistanceField& environment_sdf,
//        RobotInterface& robot,
        std::mt19937_64& generator,
        GripperControllerType gripper_controller_type,
        const double max_gripper_translation_step,
        const double max_gripper_rotation_step,
        const int64_t max_count)
    : gripper_collision_checker_(nh)
    , enviroment_sdf_(environment_sdf)
//    , robot_(robot)
    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)
    , gripper_controller_type_(gripper_controller_type)
    , translation_lower_bound_(-max_gripper_translation_step)
    , translation_upper_bound_(max_gripper_translation_step)
    , rotation_lower_bound_(-max_gripper_rotation_step)
    , rotation_upper_bound_(max_gripper_rotation_step)
    , max_count_(max_count)
{

}

////////////////////////////////////////////////////////////////////////////////
// Functions that are used to initialize function pointers in the
// constructor. These all require that task_type_ and
// deformable_type_ have been set already
////////////////////////////////////////////////////////////////////////////////


void GripperMotionGenerator::SetGripperControllerType(GripperControllerType gripper_controller_type)
{
    gripper_controller_type_ = gripper_controller_type;
}


std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> GripperMotionGenerator::findOptimalGripperMotion(
        const DeformableModel::Ptr deformable_model,
        const DeformableModel::DeformableModelInputData& input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{
    switch (gripper_controller_type_){
        case GripperControllerType::RANDOM_SAMPLE:
            return solvedByRandomSampling(deformable_model,
                                      input_data,
                                      max_gripper_velocity,
                                      obstacle_avoidance_scale);
            break;
        case GripperControllerType::UNIFORM_SAMPLE:
            return solvedByUniformSampling(deformable_model,
                                      input_data,
                                      max_gripper_velocity,
                                      obstacle_avoidance_scale);
            break;
        // Default: return non-optimized result, simple pseudo inverse
        default:
            return deformable_model->getSuggestedGrippersCommand(
                        input_data,
                        max_gripper_velocity,
                        obstacle_avoidance_scale);
            break;
    };

}


/////////////////////////////////////////////////////////////////////////////////
// Private optimization function
/////////////////////////////////////////////////////////////////////////////////

std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> GripperMotionGenerator::solvedByRandomSampling(
        const DeformableModel::Ptr deformable_model,
        const DeformableModel::DeformableModelInputData &input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{
    const Eigen::VectorXd& desired_object_p_dot =
            input_data.task_desired_object_delta_fn_(input_data.world_initial_state_).delta;

    const int num_gripper = input_data.world_initial_state_.all_grippers_single_pose_.size();

    double min_error = -2.0;
    AllGrippersSinglePoseDelta optimal_gripper_command;

    for (int64_t ind_count = 0; ind_count < max_count_; ind_count++)
    {
        AllGrippersSinglePoseDelta grippers_motion_sample;
        for (int ind_gripper = 0; ind_gripper < num_gripper; ind_gripper++)
        {
            // Eigen::Affine3d single_gripper_motion_sample = EigenHelpers::ExpTwist(singleGripperPoseDeltaSampler(), 1.0);
            grippers_motion_sample.push_back(singelGripperPoseDeltaSampler());
        }

        ObjectPointSet predicted_object_p_dot = deformable_model->getProjectedObjectDelta(
                    input_data,
                    grippers_motion_sample,
                    input_data.world_initial_state_.object_configuration_);



    }




}


std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> GripperMotionGenerator::solvedByUniformSampling(
        const DeformableModel::Ptr deformable_model,
        const DeformableModel::DeformableModelInputData &input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{

}





//////////////////////////////////////////////////////////////////////////////////
// Helper function
//////////////////////////////////////////////////////////////////////////////////

kinematics::Vector6d GripperMotionGenerator::singelGripperPoseDeltaSampler()
{
    const double x1 = EigenHelpers::Interpolate(translation_lower_bound_, translation_upper_bound_, uniform_unit_distribution_(generator_));
    const double y1 = EigenHelpers::Interpolate(translation_lower_bound_, translation_upper_bound_, uniform_unit_distribution_(generator_));
    const double z1 = EigenHelpers::Interpolate(translation_lower_bound_, translation_upper_bound_, uniform_unit_distribution_(generator_));

    const double x2 = EigenHelpers::Interpolate(rotation_lower_bound_, rotation_upper_bound_, uniform_unit_distribution_(generator_));
    const double y2 = EigenHelpers::Interpolate(rotation_lower_bound_, rotation_upper_bound_, uniform_unit_distribution_(generator_));
    const double z2 = EigenHelpers::Interpolate(rotation_lower_bound_, rotation_upper_bound_, uniform_unit_distribution_(generator_));

    kinematics::Vector6d random_sample;

    random_sample(0) = x1;
    random_sample(1) = y1;
    random_sample(2) = z1;

    random_sample(3) = x2;
    random_sample(4) = y2;
    random_sample(5) = z2;

    return random_sample;
}

double GripperMotionGenerator::errorOfControlByPrediction(
        ObjectPointSet& predicted_object_p_dot,
        Eigen::VectorXd& desired_object_p_dot)
{
    ssize_t num_nodes = predicted_object_p_dot.cols();
    double sum_of_error = 0;

    for (ssize_t node_ind = 0; node_ind < num_nodes; node_ind++)
    {
        Eigen::Vector3d node_predicted_p_dot = predicted_object_p_dot.col(node_ind);
        Eigen::Vector3d node_desired_p_dot = desired_object_p_dot.segment<3>(node_ind*3);

        double node_p_dot_error = (node_predicted_p_dot - node_desired_p_dot).norm();
        sum_of_error += node_p_dot_error;
    }

    return sum_of_error;
}




