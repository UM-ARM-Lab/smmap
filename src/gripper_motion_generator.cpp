#include <deformable_manipulation_experiment_params/ros_params.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>

#include "smmap/gripper_motion_generator.h"
#include "smmap/task.h"

using namespace smmap;

GripperMotionGenerator::GripperMotionGenerator(ros::NodeHandle &nh,
        const sdf_tools::SignedDistanceField& environment_sdf,
//        RobotInterface& robot,
        std::mt19937_64& generator,
        Visualizer vis,
        GripperControllerType gripper_controller_type,
        const double max_gripper_translation_step,
        const double max_gripper_rotation_step,
        const int64_t max_count,
        const double distance_to_obstacle_threshold)
    : object_initial_node_distance_(CalculateDistanceMatrix(GetObjectInitialConfiguration(nh)))
    , gripper_collision_checker_(nh)
    , enviroment_sdf_(environment_sdf)
//    , robot_(robot)
    , generator_(generator)
    , uniform_unit_distribution_(0.0, 1.0)
    , vis_(vis)
    , gripper_controller_type_(gripper_controller_type)
    , deformable_type_(GetDeformableType(nh))
    , task_type_(GetTaskType(nh))
    , translation_lower_bound_(-max_gripper_translation_step)
    , translation_upper_bound_(max_gripper_translation_step)
    , rotation_lower_bound_(-max_gripper_rotation_step)
    , rotation_upper_bound_(max_gripper_rotation_step)
    , distance_to_obstacle_threshold_(distance_to_obstacle_threshold)
    , max_count_(max_count)
    , sample_count_(0)
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
        const WorldState &current_world_state,
        const DeformableModel::Ptr deformable_model,
        const DeformableModel::DeformableModelInputData& input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{
    switch (gripper_controller_type_){
        case GripperControllerType::RANDOM_SAMPLING:
            return solvedByRandomSampling(
                                      current_world_state,
                                      deformable_model,
                                      input_data,
<<<<<<< HEAD
<<<<<<< HEAD
                                      max_gripper_velocity);
=======
                                      object_configuration,
                                      current_gripper_pose,
=======
>>>>>>> reformatting
                                      max_gripper_velocity,
                                      obstacle_avoidance_scale);
>>>>>>> apply stretching in task
            break;
        case GripperControllerType::UNIFORM_SAMPLING:
            return solvedByUniformSampling(
                                      current_world_state,
                                      deformable_model,
                                      input_data,
<<<<<<< HEAD
<<<<<<< HEAD
                                      max_gripper_velocity);
=======
                                      object_configuration,
                                      current_gripper_pose,
=======
>>>>>>> reformatting
                                      max_gripper_velocity,
                                      obstacle_avoidance_scale);
>>>>>>> apply stretching in task
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
        const WorldState& current_world_state,
        const DeformableModel::Ptr deformable_model,
        const DeformableModel::DeformableModelInputData &input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{
    const double max_step_size = max_gripper_velocity * input_data.dt_;

    const Eigen::VectorXd& desired_object_p_dot =
            input_data.task_desired_object_delta_fn_(current_world_state).delta;

    const ssize_t num_grippers = input_data.world_current_state_.all_grippers_single_pose_.size();
    const ssize_t num_nodes = input_data.world_current_state_.object_configuration_.cols();

    double min_error = std::numeric_limits<double>::infinity();
    AllGrippersSinglePoseDelta optimal_gripper_command;

    for (int64_t ind_count = 0; ind_count < max_count_; ind_count++)
    {
        AllGrippersSinglePoseDelta grippers_motion_sample = allGripperPoseDeltaSampler(num_grippers);

        /*
        // Method 2: Using avoidance result
        const std::vector<CollisionAvoidanceResult> grippers_collision_avoidance_result =
                ComputeGripperObjectAvoidance(
                    input_data.world_current_state_.gripper_collision_data_,
                    input_data.world_current_state_.all_grippers_single_pose_,
                    max_step_size);

        AllGrippersSinglePoseDelta grippers_motion_collision_avoidance =
                CombineDesiredAndObjectAvoidance(
                    grippers_motion_sample,
                    grippers_collision_avoidance_result,
                    obstacle_avoidance_scale);

        // get predicted object motion
        ObjectPointSet predicted_object_p_dot = deformable_model->getObjectDelta(
                    input_data,
                    grippers_motion_collision_avoidance);

        double sample_error = errorOfControlByPrediction(predicted_object_p_dot, desired_object_p_dot);

        // Compare if the sample grippers motion is better than the best to now
        if (min_error < 0 || sample_error < min_error)
        {
            min_error = sample_error;
            optimal_gripper_command.clear();

            for (ssize_t ind_gripper = 0; ind_gripper < num_grippers; ind_gripper++)
            {
                optimal_gripper_command.push_back(grippers_motion_collision_avoidance.at(ind_gripper));
            }
        }
        */



        // Method 1: use constraint_violation checker for gripper collosion
        // Constraint violation checking here
        bool constraint_violation = gripperCollisionCheckResult(current_world_state.all_grippers_single_pose_,
                                                                grippers_motion_sample).first;

        bool stretching_violation = stretchingDetection(input_data,
                                                       current_world_state.all_grippers_single_pose_,
                                                       grippers_motion_sample,
                                                       current_world_state.object_configuration_);

        // If no constraint violation
        if ((!constraint_violation) && (!stretching_violation))
        {
            // get predicted object motion
            const ObjectPointSet predicted_object_p_dot = deformable_model->getObjectDelta(
                        input_data,
                        grippers_motion_sample,
                        current_world_state.object_configuration_);

            const double sample_error = errorOfControlByPrediction(predicted_object_p_dot, desired_object_p_dot);

            // Compare if the sample grippers motion is better than the best to now
            if (sample_error < min_error)
            {
                min_error = sample_error;
                optimal_gripper_command.clear();

                for (ssize_t ind_gripper = 0; ind_gripper < num_grippers; ind_gripper++)
                {
                    optimal_gripper_command.push_back(grippers_motion_sample.at(ind_gripper));
                }
            }
        }
    }

    std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> suggested_grippers_command(
                optimal_gripper_command,
                ObjectPointSet::Zero(3, num_nodes));

    return suggested_grippers_command;
}


std::pair<AllGrippersSinglePoseDelta, ObjectPointSet> GripperMotionGenerator::solvedByUniformSampling(
        const WorldState &current_world_state,
        const DeformableModel::Ptr deformable_model,
        const DeformableModel::DeformableModelInputData &input_data,
        const double max_gripper_velocity,
        const double obstacle_avoidance_scale)
{

}


//////////////////////////////////////////////////////////////////////////////////
// Helper function
//////////////////////////////////////////////////////////////////////////////////

kinematics::Vector6d GripperMotionGenerator::singleGripperPoseDeltaSampler()
{
    const double x1 = EigenHelpers::Interpolate(translation_lower_bound_, translation_upper_bound_, uniform_unit_distribution_(generator_));
    const double y1 = EigenHelpers::Interpolate(translation_lower_bound_, translation_upper_bound_, uniform_unit_distribution_(generator_));
    const double z1 = EigenHelpers::Interpolate(translation_lower_bound_, translation_upper_bound_, uniform_unit_distribution_(generator_));

    const double x2 = EigenHelpers::Interpolate(rotation_lower_bound_, rotation_upper_bound_, uniform_unit_distribution_(generator_));
    const double y2 = EigenHelpers::Interpolate(rotation_lower_bound_, rotation_upper_bound_, uniform_unit_distribution_(generator_));
    const double z2 = EigenHelpers::Interpolate(rotation_lower_bound_, rotation_upper_bound_, uniform_unit_distribution_(generator_));

    kinematics::Vector6d random_sample;

    // Q: how to set the fix step? only to translational motion?
    double raw_norm = std::sqrt(std::pow(x1,2) + std::pow(y1,2) + std::pow(z1,2));

    if (raw_norm > 0.000001)
    {
        random_sample(0) = x1/raw_norm * translation_upper_bound_;
        random_sample(1) = y1/raw_norm * translation_upper_bound_;
        random_sample(2) = z1/raw_norm * translation_upper_bound_;
    }
    else
    {
        random_sample(0) = x1;
        random_sample(1) = y1;
        random_sample(2) = z1;
    }

    random_sample(3) = x2;
    random_sample(4) = y2;
    random_sample(5) = z2;

    return random_sample;
}

AllGrippersSinglePoseDelta GripperMotionGenerator::allGripperPoseDeltaSampler(const ssize_t num_grippers)
{
    AllGrippersSinglePoseDelta grippers_motion_sample;

    // if sample_count_ < 0, return all-sampled motion, otherwise, return one-for-each-time sample
    if (sample_count_ < 0 )
    {
        for (ssize_t ind_gripper = 0; ind_gripper < num_grippers; ind_gripper++)
        {
            // Eigen::Affine3d single_gripper_motion_sample = EigenHelpers::ExpTwist(singleGripperPoseDeltaSampler(), 1.0);
            grippers_motion_sample.push_back(singelGripperPoseDeltaSampler());
        }
        return grippers_motion_sample;
    }
    else if (sample_count_ < num_grippers)
    {
        for (ssize_t ind_gripper = 0; ind_gripper < num_grippers; ind_gripper++)
        {
            if (ind_gripper == sample_count_)
            {
                grippers_motion_sample.push_back(singelGripperPoseDeltaSampler());
            }
            else
            {
                kinematics::Vector6d no_sample = Eigen::MatrixXd::Zero(6,1);
                grippers_motion_sample.push_back(no_sample);
            }

        }
        sample_count_++;

        if(sample_count_ >= num_grippers)
        {
            sample_count_ = 0;
        }
        return grippers_motion_sample;
    }

}


double GripperMotionGenerator::errorOfControlByPrediction(
        const ObjectPointSet& predicted_object_p_dot,
        const Eigen::VectorXd& desired_object_p_dot) const
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

std::pair<bool, std::vector<CollisionData>> GripperMotionGenerator::gripperCollisionCheckResult(
        const AllGrippersSinglePose& current_gripper_pose,
        const AllGrippersSinglePoseDelta& test_gripper_motion)
{
    AllGrippersSinglePose gripper_test_pose;
    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
    {
        gripper_test_pose.push_back(current_gripper_pose.at(gripper_ind) * kinematics::expTwistAffine3d(test_gripper_motion.at(gripper_ind), 1.0));
    }

    std::vector<CollisionData> collision_data = gripper_collision_checker_.gripperCollisionCheck(gripper_test_pose);

    bool collision_violation = false;
    std::pair<bool, std::vector<CollisionData>> collision_result(collision_violation, collision_data);


    for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
    {
        if (collision_data.at(gripper_ind).distance_to_obstacle_ < distance_to_obstacle_threshold_)
        {
            collision_violation = true;
            collision_result.first = collision_violation;
            return collision_result;
        }
    }

    return collision_result;
}


bool GripperMotionGenerator::stretchingDetection(
        const DeformableModel::DeformableModelInputData &input_data,
        const AllGrippersSinglePose &current_gripper_pose,
        const AllGrippersSinglePoseDelta &test_gripper_motion,
        const ObjectPointSet &object_configuration)
{
    switch (deformable_type_) {
    case ROPE:
        return RopeTwoGrippersStretchingDetection(
                    input_data,
                    current_gripper_pose,
                    test_gripper_motion,
                    object_configuration);

        break;
    default:
        return false;
        break;
    }
}


bool GripperMotionGenerator::RopeTwoGrippersStretchingDetection(
        const DeformableModel::DeformableModelInputData &input_data,
        const AllGrippersSinglePose &current_gripper_pose,
        const AllGrippersSinglePoseDelta &test_gripper_motion,
        const ObjectPointSet& object_configuration)
{
    // This Version only works for two grippers situation, should be revised later

    const Eigen::MatrixXd node_squared_distance =
            CalculateSquaredDistanceMatrix(object_configuration);

    const double stretching_correction_threshold = 0.005;
    const ssize_t num_nodes = object_configuration.cols();

    bool over_strech = false;
    double streching_sum = 0.0;

    const ssize_t start_node = 1;
    const ssize_t end_node = num_nodes - 2;

    Eigen::Vector3d first_correction_vector =
            (object_configuration.block<3, 1>(0, start_node + 1)
                - object_configuration.block<3, 1>(0, start_node));
    first_correction_vector = first_correction_vector/first_correction_vector.norm();

    Eigen::Vector3d second_correction_vector =
            (object_configuration.block<3, 1>(0, end_node - 1)
                - object_configuration.block<3, 1>(0, end_node));
    second_correction_vector = second_correction_vector/second_correction_vector.norm();

    EigenHelpers::VectorVector3d stretching_correction_vector;
    stretching_correction_vector.push_back(first_correction_vector);
    stretching_correction_vector.push_back(second_correction_vector);

    for (ssize_t first_node = 0; first_node < num_nodes; ++first_node)
    {
        for (ssize_t second_node = first_node + 1; second_node < num_nodes; ++second_node)
        {
            const double max_distance = stretching_correction_threshold + object_initial_node_distance_(first_node, second_node);
            if (node_squared_distance(first_node, second_node) > max_distance * max_distance)
            {
                over_strech = true;
                break;
            }
        }
        if(over_strech)
        {
            break;
        }
    }

    if(over_strech)
    {
        AllGrippersSinglePose gripper_test_pose;
        for (size_t gripper_ind = 0; gripper_ind < current_gripper_pose.size(); gripper_ind++)
        {
            gripper_test_pose.push_back(current_gripper_pose.at(gripper_ind) * kinematics::expTwistAffine3d(test_gripper_motion.at(gripper_ind), 1.0));
            Eigen::Vector3d resulting_gripper_motion = gripper_test_pose.at(gripper_ind).translation()
                    - current_gripper_pose.at(gripper_ind).translation();
            streching_sum += resulting_gripper_motion.dot(stretching_correction_vector.at(gripper_ind));
        }
        if(streching_sum > 0.0)
        {
            over_strech = false;
        }
    }

    if(over_strech)
    {
        EigenHelpers::VectorVector3d line_starts;
        EigenHelpers::VectorVector3d line_ends;
        line_starts.push_back(object_configuration.block<3,1>(0, 0));
        line_starts.push_back(object_configuration.block<3,1>(0, num_nodes-1));
        line_ends.push_back(line_starts.at(0) + first_correction_vector);
        line_ends.push_back(line_starts.at(1) + second_correction_vector);

        vis_.visualizeLines("gripper overstretch motion",
                            line_starts,
                            line_ends,
                            Visualizer::Olive());
    }


    return over_strech;

}




