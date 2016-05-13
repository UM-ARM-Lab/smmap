#ifndef TASK_SPECIFICATION_H
#define TASK_SPECIFICATION_H

#include <memory>
#include <smmap_experiment_params/task_enums.h>
#include "smmap/task_function_pointer_types.h"
#include "smmap/visualization_tools.h"

namespace smmap
{
    class TaskSpecification
    {
        public:
            typedef std::shared_ptr< TaskSpecification > Ptr;

            ////////////////////////////////////////////////////////////////////
            // Constructor to initialize objects that all TaskSpecifications share
            ////////////////////////////////////////////////////////////////////

            TaskSpecification( ros::NodeHandle& nh  );
            TaskSpecification( ros::NodeHandle& nh, Visualizer vis );

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
            ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta(
                    const WorldState& world_state ) const;

            Eigen::VectorXd projectObjectDelta(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta ) const;

            ////////////////////////////////////////////////////////////////////
            // Helper functions
            // TODO: Should these be virtual? virtual final?
            ////////////////////////////////////////////////////////////////////

            /**
             * @brief computeStretchingCorrection
             * @param object_configuration
             * @return
             */
            ObjectDeltaAndWeight calculateStretchingCorrectionDelta(
                    const WorldState& world_state ) const;

            /**
             * @brief combineErrorCorrectionAndStretchingCorrection
             * @param error_correction
             * @param stretching_correction
             * @return
             */
            ObjectDeltaAndWeight combineErrorCorrectionAndStretchingCorrection(
                    const ObjectDeltaAndWeight& error_correction,
                    const ObjectDeltaAndWeight& stretching_correction ) const;

        protected:
            ////////////////////////////////////////////////////////////////////
            // Prevent deletion of base pointer
            ////////////////////////////////////////////////////////////////////

            ~TaskSpecification() {}

            ////////////////////////////////////////////////////////////////////
            // Objects shared by all task specifications
            ////////////////////////////////////////////////////////////////////

            Visualizer vis_;
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

            virtual ObjectDeltaAndWeight calculateObjectErrorCorrectionDelta_impl(
                    const WorldState& world_state ) const = 0;

            virtual Eigen::VectorXd projectObjectDelta_impl(
                    const ObjectPointSet& object_configuration,
                    Eigen::VectorXd object_delta ) const = 0;

    };
}

#endif // TASK_SPECIFICATION_H
