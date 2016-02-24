#ifndef TASK_H
#define TASK_H

#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/log.hpp>

#include "smmap/model_set.h"
#include "smmap/planner.h"
#include "smmap/task_specification.h"
#include "smmap/task_function_pointer_types.h"

/*
Eigen::VectorXd DiminishingRigidityModel::computeStretchingCorrection(
        const ObjectPointSet& object_current_configuration ) const
{
    Eigen::VectorXd stretching_correction = Eigen::VectorXd::Zero( object_current_configuration.cols() * 3 );

    Eigen::MatrixXd node_distance_delta =
            distanceMatrix( object_current_configuration )
            - object_initial_node_distance_;

    for ( long first_node = 0; first_node < node_distance_delta.rows(); first_node++)
    {
        for ( long second_node = first_node + 1; second_node < node_distance_delta.cols(); second_node++)
        {
            if ( node_distance_delta( first_node, second_node ) > stretching_correction_threshold_ )
            {
                // The correction vector points from the first node to the second node,
                // and is half the length of the "extra" distance
                Eigen::Vector3d correction_vector = 0.5 * node_distance_delta( first_node, second_node )
                        * ( object_current_configuration.block< 3, 1 >( 0, second_node )
                            - object_current_configuration.block< 3, 1 >( 0, first_node ) );

                stretching_correction.segment< 3 >( 3 * first_node ) += correction_vector;
                stretching_correction.segment< 3 >( 3 * second_node ) -= correction_vector;
            }
        }
    }

    return stretching_correction;
}
*/

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
            // ROS objects
            ////////////////////////////////////////////////////////////////////

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            Visualizer vis_;

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

            ////////////////////////////////////////////////////////////////////
            // Task specific functions that get bound as needed
            ////////////////////////////////////////////////////////////////////

            double calculateRopeCoverageError( const ObjectPointSet& current_configuration ) const;
            double calculateClothCoverageError( const ObjectPointSet& current_configuration ) const;
            double calculateClothColabFoldingError( const ObjectPointSet& current_configuration ) const;

            std::pair< Eigen::VectorXd, Eigen::MatrixXd > calculateRopeCoverageDesiredDelta(
                    const WorldState& world_state );
            std::pair< Eigen::VectorXd, Eigen::MatrixXd > calculateClothCoverageDesiredDelta(
                    const WorldState& world_state );
            std::pair< Eigen::VectorXd, Eigen::MatrixXd > calculateClothColabFoldingDesiredDelta(
                    const WorldState& world_state );

            ////////////////////////////////////////////////////////////////////
            // ModelSet
            ////////////////////////////////////////////////////////////////////

            ModelSet model_set_;
    };
}

#endif // TASK_H
