#ifndef TEMPLATESCOLLECTOR_H
#define TEMPLATESCOLLECTOR_H

#include <sdf_tools/sdf.hpp>
#include <Eigen/Dense>
#include <smmap_utilities/visualization_tools.h>
#include "smmap/trajectory.hpp"


namespace smmap
{
    class TemplatesCollector
    {
        public:
            //TemplatesCollector();

            TemplatesCollector(
                    ros::NodeHandle nh,
                    ros::NodeHandle ph,
                    smmap_utilities::Visualizer& vis);

            typedef std::shared_ptr<TemplatesCollector> SharedPtr;

            /////////////////////////////////////////////////////////////////////////////
            // Get/Add Templates
            /////////////////////////////////////////////////////////////////////////////

            const bool isCollectingTemplates();

            bool addTemplate(
                    const Eigen::Matrix3Xd& object_configuration_in,
                    const AllGrippersSinglePose& gripper_poses_in);

            const std::vector<Eigen::Matrix3Xd>& getAllTemplates();
            const Eigen::Matrix3Xd &getSingleTemplate(ssize_t template_ind);
            const ssize_t getNumTemplates();

            bool loadObjectTemplates();
            bool saveObjectTemplates();

            /////////////////////////////////////////////////////////////////////////////
            // Get estimation
            /////////////////////////////////////////////////////////////////////////////

            const Eigen::Matrix3Xd GetEstimateConfiguration(
                    const WorldState& occluded_worldstate);

            /////////////////////////////////////////////////////////////////////////////
            // Visualization Helper
            /////////////////////////////////////////////////////////////////////////////

            void visualizeTemplate(ssize_t template_ind);


        private:

            /////////////////////////////////////////////////////////////////////////////
            // Serialization helper
            /////////////////////////////////////////////////////////////////////////////

            static uint64_t SerializeSingleTemplate(
                    const Eigen::VectorXd& node,
                    std::vector<uint8_t>& buffer,
                    const std::function<uint64_t(
                        const Eigen::VectorXd&,
                        std::vector<uint8_t>&)>& value_serializer);

            uint64_t SerializeSelf(
                    std::vector<uint8_t>& buffer,
                    const std::function<uint64_t(
                        const Eigen::VectorXd&,
                        std::vector<uint8_t>&)>& value_serializer) const;

            static std::pair<Eigen::VectorXd, uint64_t> DeserializeSingleTemplate(
                    const std::vector<uint8_t>& buffer,
                    const uint64_t current,
                    const std::function<std::pair<Eigen::VectorXd, uint64_t>(
                        const std::vector<uint8_t>&,
                        const uint64_t)>& value_deserializer);

            uint64_t DeserializeSelf(
                    const std::vector<uint8_t>& buffer,
                    const uint64_t current,
                    const std::function<std::pair<Eigen::VectorXd, uint64_t>(
                        const std::vector<uint8_t>&,
                        const uint64_t)>& value_deserializer);

            /////////////////////////////////////////////////////////////////////////////
            // Get estimation according to the type of estimator
            /////////////////////////////////////////////////////////////////////////////

            const Eigen::Matrix3Xd EstimateByProcrustesLeastSquares(
                    const WorldState& occluded_worldstate);

            const Eigen::Matrix3Xd EstimateByProcrustesLeastSquaresL1Reguralization(
                    const WorldState& occluded_worldstate);


            /////////////////////////////////////////////////////////////////////////////
            // Translation and relocating geometric centra of gravity helpers
            /////////////////////////////////////////////////////////////////////////////

            const Eigen::Vector3d GetCenterOfGravity(const Eigen::Matrix3Xd& object_configuration);

            const Eigen::MatrixXd GetObservedTemplates(
                    const ObservableIndexes& observable_correspondency,
                    const ssize_t num_grippers);

            const Eigen::Matrix3Xd RelocateSingleCenterOfGravity(
                    const Eigen::Matrix3Xd& original_configuration,
                    const Eigen::Vector3d& moving_vector);

            const Eigen::Matrix3Xd RelocateSingelCenterToOrigin(const Eigen::Matrix3Xd& original_configuration);

            /////////////////////////////////////////////////////////////////////////////
            // Procrustic Analysis helper
            /////////////////////////////////////////////////////////////////////////////

            const Eigen::Matrix3d GetSingleRotAlignmentMatrix(
                    const Eigen::Matrix3Xd& target_configuration,
                    const Eigen::Matrix3Xd& observerble_template);           

            /////////////////////////////////////////////////////////////////////////////
            // Template matching
            /////////////////////////////////////////////////////////////////////////////

            const std::pair<Eigen::MatrixXd, std::vector<std::pair<Eigen::MatrixXd, Eigen::Vector3d>>> GetObservableTemplatesBase(
                    const WorldState& occluded_world_state);

            const std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> GetObservableTemplateBaseForSingelConfiguration(
                    const Eigen::Matrix3Xd& relocated_observed_configuration,
                    const Eigen::MatrixXd& observable_templates_base);

            const Eigen::VectorXd GetTemplatesWeight(
                    const Eigen::Matrix3Xd& object_configuration,
                    const Eigen::MatrixXd& templates_base);

            const Eigen::MatrixXd GetFullMatchingBase(
                    const std::vector<std::pair<Eigen::MatrixXd, Eigen::Vector3d> >& all_rots_and_centers);

            //////////////////////////////////////////////////////////////////////////////
            // other helper functions
            //////////////////////////////////////////////////////////////////////////////

            const Eigen::Matrix3Xd GetDataToBeMatched (const WorldState& occluded_world_state);

            const bool HaveSimilarTemplate(const Eigen::Matrix3Xd& template_in);


        private:

            ros::NodeHandle nh_;
            ros::NodeHandle ph_;
            smmap_utilities::Visualizer& vis_;

            const sdf_tools::SignedDistanceField observability_map_;

            bool collect_templates_;
            bool include_grippers_;
            const bool recollect_all_templates_;
            const int collect_frequency_;
            int internal_counter_;
            const double similarity_threshold_;
            const double l1_regularization_factor_;

            const EstimatotType estimator_type_;
            Eigen::VectorXd last_estimate_coordinate_;

            const ssize_t num_to_collect_;
            const ssize_t max_num_in_collector_;
            ssize_t num_pre_stored_;
            ssize_t num_templates_;
            std::vector<Eigen::Matrix3Xd> object_templates_;

    };


}


#endif // TEMPLATESCOLLECTOR_H
