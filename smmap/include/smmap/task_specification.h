#ifndef TASK_SPECIFICATION_H
#define TASK_SPECIFICATION_H

#include <memory>
#include "smmap/task_enums.h"
#include "smmap/task_function_pointer_types.h"
#include "smmap/visualization_tools.h"

namespace smmap
{
    class TaskSpecification
    {
        public:
            typedef std::shared_ptr< TaskSpecification > Ptr;

            ////////////////////////////////////////////////////////////////////
            // Constants used in SMMAP and the simulator - shared world
            ////////////////////////////////////////////////////////////////////

            // Define the middle of the top of the table
            static constexpr float TABLE_X = 0.0f; // METERS
            static constexpr float TABLE_Y = 0.0f; // METERS
            static constexpr float TABLE_Z = 0.7f; // METERS
            // Define the thickness of the table
            static constexpr float TABLE_THICKNESS = 0.05f; // METERS

            ////////////////////////////////////////////////////////////////////
            // Constants used in SMMAP and the simulator - rope world
            ////////////////////////////////////////////////////////////////////

            static constexpr float ROPE_SEGMENT_LENGTH = 0.025f;    // METERS
            static constexpr float ROPE_RADIUS = 0.01f;             // METERS
            static constexpr int ROPE_NUM_LINKS = 50;
            // TODO: why did Dmitry's code use 0.5f here?
            static constexpr float ROPE_GRIPPER_APPERTURE = 0.03f;      // METERS
            static constexpr float ROPE_TABLE_HALF_SIDE_LENGTH = 1.5f;  // METERS
            static constexpr float ROPE_CYLINDER_RADIUS = 0.15f;        // METERS
            static constexpr float ROPE_CYLINDER_HEIGHT = 0.3f;         // METERS

            ////////////////////////////////////////////////////////////////////
            // Constants used in SMMAP and the simulator - cloth world
            ////////////////////////////////////////////////////////////////////

            static constexpr float CLOTH_HALF_SIDE_LENGTH = 0.25f; // METERS
            static constexpr float CLOTH_X = TABLE_X + CLOTH_HALF_SIDE_LENGTH; // METERS
            static constexpr float CLOTH_Y = TABLE_Y; // METERS
            static constexpr float CLOTH_Z = TABLE_Z + 0.01f; // METERS

            static constexpr int CLOTH_DIVS = 45;
            static constexpr float CLOTH_GRIPPER_APPERTURE = 0.1f; // METERS
            static constexpr float CLOTH_TABLE_HALF_SIDE_LENGTH = 0.2f; // METERS

            ////////////////////////////////////////////////////////////////////
            // Constructor to initialize objects that all TaskSpecifications share
            ////////////////////////////////////////////////////////////////////

            TaskSpecification( ros::NodeHandle& nh );

            ////////////////////////////////////////////////////////////////////
            // Static builder function
            ////////////////////////////////////////////////////////////////////

            static TaskSpecification::Ptr MakeTaskSpecification(
                    ros::NodeHandle& nh );

            ////////////////////////////////////////////////////////////////////
            // Virtual function wrappers
            ////////////////////////////////////////////////////////////////////

            double getDeformability() const;                // k
            double getCollisionScalingFactor() const;       // beta (or k2)
            double getStretchingScalingThreshold() const;   // lambda
            double maxTime() const;                         // max simulation time when scripting things

            void visualizeDeformableObject(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color ) const;

            void visualizeDeformableObject(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector< std_msgs::ColorRGBA >& colors ) const;

            double calculateError(
                    const ObjectPointSet& object_configuration ) const;

            /**
             * @brief calculateObjectDesiredDelta
             * @param world_state
             * @return return.first is the desired movement of the object
             *         return.second is the importance of that part of the movement
             */
            std::pair< Eigen::VectorXd, Eigen::VectorXd > calculateObjectErrorCorrectionDelta(
                    const WorldState& world_state ) const;

            ////////////////////////////////////////////////////////////////////
            // Helper functions
            // TODO: Should these be virtual? virtual final?
            ////////////////////////////////////////////////////////////////////

            /**
             * @brief computeStretchingCorrection
             * @param object_configuration
             * @return
             */
            std::pair< Eigen::VectorXd, Eigen::VectorXd > calculateStretchingCorrectionDelta(
                    const WorldState& world_state ) const;

            /**
             * @brief combineErrorCorrectionAndStretchingCorrection
             * @param error_correction
             * @param stretching_correction
             * @return
             */
            std::pair< Eigen::VectorXd, Eigen::VectorXd > combineErrorCorrectionAndStretchingCorrection(
                    const std::pair< Eigen::VectorXd, Eigen::VectorXd >& error_correction,
                    const std::pair< Eigen::VectorXd, Eigen::VectorXd >& stretching_correction ) const;

        protected:
            ////////////////////////////////////////////////////////////////////
            // Prevent deletion of base pointer
            ////////////////////////////////////////////////////////////////////

            ~TaskSpecification() {}

            ////////////////////////////////////////////////////////////////////
            // Objects shared by all task specifications
            ////////////////////////////////////////////////////////////////////

            const Eigen::MatrixXd object_initial_node_distance_;
            const long num_nodes_;

        private:
            ////////////////////////////////////////////////////////////////////////////////
            // Virtual functions that each task specification must provide
            ////////////////////////////////////////////////////////////////////////////////

            virtual double getDeformability_impl() const = 0;                // k
            virtual double getCollisionScalingFactor_impl() const = 0;       // beta (or k2)
            virtual double getStretchingScalingThreshold_impl() const = 0;   // lambda
            virtual double maxTime_impl() const = 0;                         // max simulation time when scripting things

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std_msgs::ColorRGBA& color ) const = 0;

            virtual void visualizeDeformableObject_impl(
                    Visualizer& vis,
                    const std::string& marker_name,
                    const ObjectPointSet& object_configuration,
                    const std::vector< std_msgs::ColorRGBA >& colors ) const = 0;

            virtual double calculateError_impl(
                    const ObjectPointSet& object_configuration ) const = 0;

            virtual std::pair< Eigen::VectorXd, Eigen::VectorXd > calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state ) const = 0;

    };
}

#endif // TASK_SPECIFICATION_H
