#include <atomic>
#include <arc_utilities/log.hpp>
#include <map>


#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/zlib_helpers.hpp>
#include <arc_utilities/serialization_eigen.hpp>
#include <arc_utilities/serialization.hpp>
#include <smmap_utilities/gurobi_solvers.h>
#include <arc_utilities/timing.hpp>

#include "smmap/templatescollector.h"
#include "smmap/trajectory.hpp"
#include "smmap/ros_communication_helpers.hpp"

using namespace smmap;
using namespace smmap_utilities;
using namespace arc_utilities;


TemplatesCollector::TemplatesCollector(
        ros::NodeHandle nh,
        ros::NodeHandle ph,
        Visualizer& vis)
    : nh_(nh)
    , ph_(ph)
    , vis_(vis)

    , observability_map_(GetEnvironmentObservabilitySDF(nh_))

    , collect_templates_(GetCollectTemplates(ph_))
    , include_grippers_(true)
    , recollect_all_templates_(GetRecollectAllTemplates(ph_))   // TODO: to be replaced with ros_param
    , collect_frequency_(GetTemplatesCollectionFrequency(ph_))
    , internal_counter_(0)
    , similarity_threshold_(GetSimilarityThreshold(ph_))
    , l1_regularization_factor_(GetL1RegularizationFactor(ph_))
    , estimator_type_(GetEstimatorType(ph_))

    , num_to_collect_(GetNumTemplatesToCollect(ph_))
    , max_num_in_collector_(1000)      // TODO: magic number to be replaced
    , num_reliable_temps_(GetNumTemplatesToCollect(ph_))    // TODO: Add new Ros params to do so
    , num_pre_stored_(0)
    , num_templates_(0)
{
    if (!collect_templates_)
    {
        if (!loadObjectTemplates())
        {
            ROS_INFO_NAMED("templates_collection", "No existing templates found. Templates sampling ... ");
            collect_templates_ = true;
        }
    }
    else
    {
        if (!recollect_all_templates_)
        {
            loadObjectTemplates();
            collect_templates_ = true;
        }
    }
}


/////////////////////////////////////////////////////////////////////////////
// Get/Add Templates
/////////////////////////////////////////////////////////////////////////////

const bool TemplatesCollector::isCollectingTemplates()
{
    return collect_templates_;
}

bool TemplatesCollector::addTemplate(
        const Eigen::Matrix3Xd& object_configuration_in,
        const AllGrippersSinglePose& gripper_poses_in)
{
    if (!collect_templates_)
    {
        ROS_INFO_NAMED("templates_collection", "Can not collect any more templates");
        return false;
    }

    if(internal_counter_ == 0)
    {        
        const ssize_t num_nodes = object_configuration_in.cols();
        const ssize_t num_grippers = gripper_poses_in.size();
        const ssize_t num_positions = num_nodes + num_grippers;

        // Version 2: save as a vector of matrix
        Eigen::Matrix3Xd new_template = Eigen::MatrixXd(3, num_positions);

        if (num_templates_>0)
        {
            assert((num_positions == object_templates_.at(0).cols()) && "the new template has the wrong size");
        }
        new_template.block(0, 0, 3, object_configuration_in.cols()) = object_configuration_in;

        for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
        {
            Eigen::VectorXd gripper_pose = EigenHelpers::TransformToRPY(gripper_poses_in.at(gripper_ind));
            new_template.col(num_nodes + gripper_ind)= gripper_pose.head(3);
        }

        // ensure the templates in templates pool center at origin, for rot conveniency
        const Eigen::Matrix3Xd relocated_new_template = RelocateSingelCenterToOrigin(new_template);

        // check similarity
        const bool has_similarity = HaveSimilarTemplate(relocated_new_template);

        if (has_similarity)
        {
            ROS_INFO_NAMED("templates_collection", "Some similar template exists. Neglect this new sample.");
            return false;
        }

        object_templates_.push_back(relocated_new_template);
        num_templates_++;
        ROS_INFO_STREAM_NAMED("templates_collection", "Successfully collect the " << num_templates_ << "th template");

        if ((num_templates_ + num_pre_stored_) > max_num_in_collector_)
        {
            ROS_INFO_NAMED("templates_collection", "The number of templates in the collector reaches its maximum");
            collect_templates_ = false;
        }

        if (num_templates_ > num_to_collect_)
        {
            ROS_INFO_NAMED("templates_collection", "The number of newly collected templates reaches its desired one");
            collect_templates_ = false;
        }
        internal_counter_++;
        return true;
    }
    else
    {
        ROS_INFO_STREAM_NAMED("templates_collection", "This template is neglected, counter: " << internal_counter_ << ".");
        internal_counter_++;
        if(internal_counter_ == collect_frequency_)
        {
            internal_counter_ = 0;
        }
        return false;
    }
}

const std::vector<Eigen::Matrix3Xd>& TemplatesCollector::getAllTemplates()
{
    return object_templates_;
}

const Eigen::Matrix3Xd& TemplatesCollector::getSingleTemplate(ssize_t template_ind)
{
    return object_templates_.at(template_ind);
}

const ssize_t TemplatesCollector::getNumTemplates()
{
    return num_templates_;
}

bool TemplatesCollector::loadObjectTemplates()
{
    try
    {
         ROS_INFO_NAMED("templates_collection", "Checking if object templates already exist");
         const std::string templates_file_path = GetTemplatesStorageLocation(nh_);
         GlobalStopwatch(RESET);
         std::ifstream prev_templates_result(templates_file_path, std::ios::binary | std::ios::in | std::ios::ate);
         if (!prev_templates_result.is_open())
         {
             throw_arc_exception(std::runtime_error, "Couldn't open file");
         }

         ROS_INFO_NAMED("templates_collection", "Reading contents of templates file");
         std::streamsize size = prev_templates_result.tellg();
         prev_templates_result.seekg(0, std::ios::beg);
         std::vector<uint8_t> file_buffer((size_t)size);
         if (!(prev_templates_result.read(reinterpret_cast<char*>(file_buffer.data()), size)))
         {
             throw_arc_exception(std::runtime_error, "Unable to read entire contents of the template file");
         }
         const std::vector<uint8_t> decompressed_templates_results = ZlibHelpers::DecompressBytes(file_buffer);


         uint64_t current_position = 0;
         const auto deserialized_result
                 = DeserializeVector<Eigen::VectorXd>(
                     decompressed_templates_results,
                     current_position,
                     &Deserialize<Eigen::VectorXd>);

         object_templates_ = EigenHelpers::EigenVectorXdsToMatrix3Xds(deserialized_result.first);
         current_position += deserialized_result.second;
         if (current_position != decompressed_templates_results.size())
         {
             throw_arc_exception(std::runtime_error, "Invalid data size of templates file found");
         }

         num_pre_stored_ = object_templates_.size();
         ROS_INFO_STREAM_NAMED("templates_collection", "Read templates in " << GlobalStopwatch(READ) << " seconds");
         ROS_INFO_STREAM_NAMED("templates_collection", "Num of templates loaded: " << num_pre_stored_ << ".");

         last_estimate_coordinate_ = Eigen::MatrixXd::Zero(num_pre_stored_, 1);
         return true;
    }
    catch (...)
    {
        ROS_ERROR_NAMED("templates_collection", "Loading templates results from file failed");        
        return false;
    }

}

bool TemplatesCollector::saveObjectTemplates()
{
    try
    {
        std::vector<uint8_t> buffer;
        // First serialize the graph that created the results
        ROS_INFO_NAMED("templates_collection", "Serializing the templates data");
        SerializeSelf(buffer, &Serialize<Eigen::VectorXd>);

        // Compress and save to file
        ROS_INFO_NAMED("templates_collection", "Compressing for templates storage");
        const std::vector<uint8_t> compressed_serialized_data = ZlibHelpers::CompressBytes(buffer);
        ROS_INFO_NAMED("templates_collection", "Saving templates results to file");
        const std::string templates_file_path = GetTemplatesStorageLocation(nh_);
        std::ofstream output_file(templates_file_path, std::ios::out | std::ios::binary);
        uint64_t serialized_size = compressed_serialized_data.size();

        output_file.write(
                    reinterpret_cast<const char*>(compressed_serialized_data.data()),
                    (std::streamsize)serialized_size);
        output_file.close();
        return true;
    }
    catch (...)
    {
        ROS_ERROR_NAMED("templates_collection", "Saving Dijkstras results to file failed");
        return false;
    }
}

/////////////////////////////////////////////////////////////////////////////
// Get estimation
/////////////////////////////////////////////////////////////////////////////

const Eigen::Matrix3Xd TemplatesCollector::GetEstimateConfiguration(
        const WorldState& occluded_worldstate)
{
    switch (estimator_type_)
    {
        case PROCRUSTES_LEAST_SQUARE_NORM:
            return EstimateByProcrustesLeastSquares(occluded_worldstate);
            break;
        case PROCRUSTES_LEAST_SQUARE_L1_REGULARIZATION:
            return EstimateByProcrustesLeastSquaresL1Reguralization(occluded_worldstate);
            break;
        default:
            assert(false || "No valid estimation algorithm");
            break;
    }
    return occluded_worldstate.object_configuration_;
}

/////////////////////////////////////////////////////////////////////////////
// Visualization Helper
/////////////////////////////////////////////////////////////////////////////

void TemplatesCollector::visualizeTemplate(ssize_t template_ind)
{
    const ObjectPointSet& vis_template = object_templates_.at(template_ind);
    EigenHelpers::VectorVector3d observable_nodes;
    for (ssize_t node_ind = 0; node_ind < vis_template.cols(); node_ind++)
    {
        observable_nodes.push_back(vis_template.col(node_ind));
    }

    vis_.visualizePoints(
                "templates_node",
                observable_nodes,
                Visualizer::Silver());
}

/////////////////////////////////////////////////////////////////////////////
// Serialization helper
/////////////////////////////////////////////////////////////////////////////

uint64_t TemplatesCollector::SerializeSingleTemplate(
        const Eigen::VectorXd& node,
        std::vector<uint8_t>& buffer,
        const std::function<uint64_t(const Eigen::VectorXd&, std::vector<uint8_t>&)>& value_serializer)
{
    const uint64_t start_buffer_size = buffer.size();
    // Serialize the value
    value_serializer(node, buffer);

    // Figure out how many bytes were written
    const uint64_t end_buffer_size = buffer.size();
    const uint64_t bytes_written = end_buffer_size - start_buffer_size;
    return bytes_written;
    }


uint64_t TemplatesCollector::SerializeSelf(
        std::vector<uint8_t>& buffer,
        const std::function<uint64_t(const Eigen::VectorXd&, std::vector<uint8_t>&)>& value_serializer) const
{
    const uint64_t start_buffer_size = buffer.size();
    const auto graph_state_serializer = std::bind(
                TemplatesCollector::SerializeSingleTemplate,
                std::placeholders::_1,
                std::placeholders::_2,
                value_serializer);

    std::vector<Eigen::VectorXd> object_templates_vectors = EigenHelpers::EigenMatrix3XdsToVectorXds(object_templates_);
    SerializeVector<Eigen::VectorXd>(object_templates_vectors, buffer, graph_state_serializer);
    // Figure out how many bytes were written
    const uint64_t end_buffer_size = buffer.size();
    const uint64_t bytes_written = end_buffer_size - start_buffer_size;
    return bytes_written;
}

std::pair<Eigen::VectorXd, uint64_t> TemplatesCollector::DeserializeSingleTemplate(
        const std::vector<uint8_t>& buffer,
        const uint64_t current,
        const std::function<std::pair<Eigen::VectorXd, uint64_t>(
            const std::vector<uint8_t>&,
            const uint64_t)>& value_deserializer)
{
    Eigen::VectorXd temp_node;
    assert(current < buffer.size());
    uint64_t current_position = current;

    // Deserialize the value
    const std::pair<Eigen::VectorXd, uint64_t> value_deserialized = value_deserializer(buffer, current_position);
    temp_node = value_deserialized.first;
    current_position += value_deserialized.second;
    // Figure out how many bytes were read
    const uint64_t bytes_read = current_position - current;

    return std::make_pair(temp_node, bytes_read);
}

uint64_t TemplatesCollector::DeserializeSelf(
        const std::vector<uint8_t>& buffer,
        const uint64_t current,
        const std::function<std::pair<Eigen::VectorXd, uint64_t>(
            const std::vector<uint8_t>&,
            const uint64_t)>& value_deserializer)
{
    const auto graph_state_deserializer = std::bind(
                TemplatesCollector::DeserializeSingleTemplate,
                std::placeholders::_1, std::placeholders::_2, value_deserializer);
    const auto deserialized_nodes = DeserializeVector<Eigen::VectorXd>(buffer, current, graph_state_deserializer);

    std::vector<Eigen::VectorXd> object_templates_vectors = deserialized_nodes.first;
    object_templates_ = EigenHelpers::EigenVectorXdsToMatrix3Xds(object_templates_vectors);
    return deserialized_nodes.second;
}


/////////////////////////////////////////////////////////////////////////////
// Get estimation according to the type of estimator
/////////////////////////////////////////////////////////////////////////////

// TODO: How to check the num of grippers in templates and that of the occluded world?
const Eigen::Matrix3Xd TemplatesCollector::EstimateByProcrustesLeastSquares(
        const WorldState& occluded_worldstate)
{
    const ObservableIndexes& observable_correspondency = occluded_worldstate.observabel_correspondency_;
    const ssize_t num_grippers = occluded_worldstate.all_grippers_single_pose_.size();

    const Eigen::Matrix3Xd& occluded_configuration = GetDataToBeMatched(occluded_worldstate);

    assert(occluded_configuration.cols() == observable_correspondency.size()+num_grippers);
    assert(object_templates_.size()>0);
    const ssize_t num_nodes = object_templates_.at(0).cols();

    // Move the center of the observable object to the origin for rotation alignment later
    const Eigen::Vector3d center_of_object = GetCenterOfGravity(occluded_configuration);
    const Eigen::Matrix3Xd relocated_configuration = RelocateSingelCenterToOrigin(occluded_configuration);

    // The base contains only the observable nodes and is used to get the coordinates of each templates
    // The rot contains the rotation alignment matrix for each templates.

    const std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> base_and_rot =
            GetObservableTemplatesBase(occluded_worldstate);
    const Eigen::MatrixXd& observable_templates = base_and_rot.first;
    const std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>> all_rots_and_centers = base_and_rot.second;

    const Eigen::VectorXd templates_weight = GetTemplatesWeight(
                relocated_configuration,
                observable_templates);

    ROS_INFO_STREAM_NAMED("templates_collection", "Norm of the templates coordinates: " << templates_weight.norm() << " .");

    // In the full matching base, each template contains all nodes and is rotated according to the rot alignment matrix
    const Eigen::MatrixXd full_matching_base = GetFullMatchingBase(all_rots_and_centers);

    Eigen::Matrix3Xd estimation = EigenHelpers::EigenVectorXdToMatrix3Xd(
                full_matching_base
                * templates_weight);

    // Add back the moving vector. Note, the center of object can be not the center of estimation.
    estimation = RelocateSingleCenterOfGravity(estimation, center_of_object);

    return estimation.leftCols(num_nodes - num_grippers);
}


// TODO: How to check the num of grippers in templates and that of the occluded world?
const Eigen::Matrix3Xd TemplatesCollector::EstimateByProcrustesLeastSquaresL1Reguralization(
        const WorldState& occluded_worldstate)
{
    assert(last_estimate_coordinate_.rows() == object_templates_.size());

    const ObservableIndexes& observable_correspondency = occluded_worldstate.observabel_correspondency_;
    const ssize_t num_grippers = occluded_worldstate.all_grippers_single_pose_.size();

    const Eigen::Matrix3Xd& occluded_configuration = GetDataToBeMatched(occluded_worldstate);

    assert(occluded_configuration.cols() == observable_correspondency.size()+num_grippers);
    assert(object_templates_.size()>0);
    const ssize_t num_nodes = object_templates_.at(0).cols();

    // Move the center of the observable object to the origin for rotation alignment later
    const Eigen::Vector3d center_of_object = GetCenterOfGravity(occluded_configuration);
    const Eigen::Matrix3Xd relocated_configuration = RelocateSingelCenterToOrigin(occluded_configuration);

    // The base contains only the observable nodes and is used to get the coordinates of each templates
    // The rot contains the rotation alignment matrix for each templates.

    const std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> base_and_rot =
            GetObservableTemplatesBase(occluded_worldstate);
    const Eigen::MatrixXd& observable_templates = base_and_rot.first;
    const std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>> all_rots_and_centers = base_and_rot.second;

    // Get optimal 1-norm coordinate
    const Eigen::VectorXd templates_weight
            = minSquaredNormL1NormRegularization(
                observable_templates,
                EigenHelpers::EigenMatrix3XdToVectorXd(relocated_configuration),
                last_estimate_coordinate_,
                Eigen::MatrixXd::Ones(observable_templates.rows(), 1));

    // TODO: check the L1 value, TOBE DELETED LATER
    Eigen::Matrix3Xd occluded_estimation = EigenHelpers::EigenVectorXdToMatrix3Xd(observable_templates * templates_weight);
    const double diff_estimation = CalculateSumOfPointDiff(
                relocated_configuration,
                occluded_estimation);
    double diff_L1_value = 0;
    const Eigen::VectorXd diff_L1 = last_estimate_coordinate_ - templates_weight;
    for (int ind = 0; ind < diff_L1.rows(); ind++)
    {
        diff_L1_value += std::fabs(diff_L1.row(ind).norm());
    }
    ROS_INFO_STREAM_NAMED("templates_collection", "diff of estimation in L1 objection fn: " << diff_estimation << " .");
    ROS_INFO_STREAM_NAMED("templates_collection", "diff of L1 term in L1 objection fn: " << diff_L1_value << " .");

    PrintCoordinateHelper(templates_weight);

    last_estimate_coordinate_ = templates_weight;
    ROS_INFO_STREAM_NAMED("templates_collection", "Norm of the templates coordinates: " << templates_weight.norm() << " .");

    // In the full matching base, each template contains all nodes and is rotated according to the rot alignment matrix
    const Eigen::MatrixXd full_matching_base = GetFullMatchingBase(all_rots_and_centers);

    Eigen::Matrix3Xd estimation = EigenHelpers::EigenVectorXdToMatrix3Xd(
                full_matching_base
                * templates_weight);

    // Add back the moving vector. Note, the center of object can be not the center of estimation.
    estimation = RelocateSingleCenterOfGravity(estimation, center_of_object);

    return estimation.leftCols(num_nodes - num_grippers);
}


/////////////////////////////////////////////////////////////////////////////
// Translation and relocating geometric centra of gravity helpers
/////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3d TemplatesCollector::GetCenterOfGravity(const Eigen::Matrix3Xd& object_configuration)
{
    assert(object_configuration.cols() > 0);

    Eigen::VectorXd ones_column = Eigen::MatrixXd::Ones(object_configuration.cols(), 1);
    Eigen::Vector3d center_gravity =
            object_configuration * ones_column / object_configuration.cols();

    return center_gravity;
}

// Return a matrix with each column being an template which contains only the observable nodes
// and grippers. Moreover, the center of gravity is moved to origin
const Eigen::MatrixXd TemplatesCollector::GetObservedTemplates(
        const ObservableIndexes& observable_correspondency,
        const ssize_t num_grippers)
{
    const ssize_t num_observable_nodes = observable_correspondency.size();
    const ssize_t num_templates = object_templates_.size();
    assert(object_templates_.size() > 0);
    const ssize_t num_all_nodes = object_templates_.at(0).cols();

    const std::vector<Eigen::VectorXd> vector_templates =
            EigenHelpers::EigenMatrix3XdsToVectorXds(object_templates_);
    const Eigen::MatrixXd all_templates_matrix
            = EigenHelpers::stdVectorsToMatrixXd(vector_templates);

    Eigen::MatrixXd observable_templates
            = Eigen::MatrixXd::Zero((num_observable_nodes + num_grippers) * 3, num_templates);

    for (ssize_t node_ind = 0; node_ind < num_observable_nodes; node_ind++)
    {
        // TODO: double check the lines here
        observable_templates.middleRows(node_ind * 3,3)
                = all_templates_matrix.middleRows(observable_correspondency.at(node_ind) * 3, 3);
    }

    for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
    {
        observable_templates.middleRows((num_observable_nodes + gripper_ind) * 3,3)
                = all_templates_matrix.middleRows((num_all_nodes - num_grippers + gripper_ind) * 3, 3);
    }

    return observable_templates;
}

const Eigen::Matrix3Xd TemplatesCollector::RelocateSingleCenterOfGravity(
        const Eigen::Matrix3Xd& original_configuration,
        const Eigen::Vector3d& moving_vector)
{
    Eigen::MatrixXd ones_row = Eigen::MatrixXd::Ones(1,original_configuration.cols());

    Eigen::Matrix3Xd recentered_object =
            original_configuration + moving_vector * ones_row;

    return recentered_object;
}

const Eigen::Matrix3Xd TemplatesCollector::RelocateSingelCenterToOrigin(
        const Eigen::Matrix3Xd& original_configuration)
{
    assert(original_configuration.cols() > 0);
    Eigen::Vector3d center_gravity =
            GetCenterOfGravity(original_configuration);

    return RelocateSingleCenterOfGravity(original_configuration, -center_gravity);
}

/////////////////////////////////////////////////////////////////////////////
// Procrustic Analysis helper
/////////////////////////////////////////////////////////////////////////////

const Eigen::Matrix3d TemplatesCollector::GetSingleRotAlignmentMatrix(
        const Eigen::Matrix3Xd& target_configuration,
        const Eigen::Matrix3Xd& observerble_template)
{
    const Eigen::MatrixXd matrix_AB =
            target_configuration * observerble_template.adjoint();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                matrix_AB,
                Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Q = UV^T
    return svd.matrixU() * svd.matrixV().adjoint();
}

/////////////////////////////////////////////////////////////////////////////
// Template matching
/////////////////////////////////////////////////////////////////////////////

//const std::pair <Eigen::MatrixXd, std::vector<Eigen::Matrix3d>> TemplatesCollector::GetObservableTemplatesBase(
//        const WorldState& occluded_world_state)
const std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> TemplatesCollector::GetObservableTemplatesBase(
        const WorldState& occluded_world_state)
{
    const ssize_t num_grippers = occluded_world_state.all_grippers_single_pose_.size();
    const ObservableIndexes& observable_correspondency = occluded_world_state.observabel_correspondency_;

    const Eigen::Matrix3Xd& occluded_configuration = GetDataToBeMatched(occluded_world_state);
    const Eigen::Matrix3Xd relocated_configuration = RelocateSingelCenterToOrigin(occluded_configuration);

    // TODO: Add grippers data
    const Eigen::MatrixXd observable_templates_base
            = GetObservedTemplates(observable_correspondency, num_grippers);

    return GetObservableTemplateBaseForSingelConfiguration(relocated_configuration,
                                                           observable_templates_base);

    // TODO: Check the functionality is correctly realized
    /*
    //std::vector<Eigen::Matrix3d> all_rot_matrixes;
    std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>> all_rot_and_centers;

    for (ssize_t temp_ind = 0; temp_ind < object_templates_.size(); temp_ind++)
    {        
        Eigen::Matrix3Xd template_as_matrix =
                EigenHelpers::EigenVectorXdToMatrix3Xd(observable_templates_base.col(temp_ind));

        //Eigen::Matrix3Xd relocated_template = RelocateSingelCenterToOrigin(template_as_matrix);


        const Eigen::Vector3d temp_center_gravity =
                GetCenterOfGravity(template_as_matrix);

        const Eigen::Matrix3Xd relocated_template = RelocateSingleCenterOfGravity(template_as_matrix, -temp_center_gravity);

        assert(relocated_configuration.cols() == template_as_matrix.cols());
        const Eigen::Matrix3d rot_matrix = GetSingleRotAlignmentMatrix(
                    relocated_configuration,
                    relocated_template);

        std::pair <Eigen::MatrixXd,Eigen::Vector3d> rot_and_center =
                std::make_pair(rot_matrix, temp_center_gravity);

        observable_templates_base.col(temp_ind) =
                EigenHelpers::EigenMatrix3XdToVectorXd(rot_matrix * relocated_template);

        all_rot_and_centers.push_back(rot_and_center);
    //    all_rot_matrixes.push_back(rot_matrix);
    }

    std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> base_and_rot =
            std::make_pair(observable_templates_base, all_rot_and_centers);
//    std::pair <Eigen::MatrixXd, std::vector<Eigen::Matrix3d>> base_and_rot =
//            std::make_pair(observable_templates_base, all_rot_matrixes);

    return base_and_rot;
    */
}

const std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> TemplatesCollector::GetObservableTemplateBaseForSingelConfiguration(
        const Eigen::Matrix3Xd& relocated_observed_configuration,
        const Eigen::MatrixXd& observable_templates_base)
{
    //std::vector<Eigen::Matrix3d> all_rot_matrixes;
    std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>> all_rot_and_centers;
    Eigen::MatrixXd return_templates_base = observable_templates_base;

    for (ssize_t temp_ind = 0; temp_ind < object_templates_.size(); temp_ind++)
    {
        Eigen::Matrix3Xd template_as_matrix =
                EigenHelpers::EigenVectorXdToMatrix3Xd(observable_templates_base.col(temp_ind));

        const Eigen::Vector3d temp_center_gravity =
                GetCenterOfGravity(template_as_matrix);

        const Eigen::Matrix3Xd relocated_template = RelocateSingleCenterOfGravity(template_as_matrix, -temp_center_gravity);

        assert(relocated_observed_configuration.cols() == template_as_matrix.cols());
        const Eigen::Matrix3d rot_matrix = GetSingleRotAlignmentMatrix(
                    relocated_observed_configuration,
                    relocated_template);

        std::pair <Eigen::MatrixXd,Eigen::Vector3d> rot_and_center =
                std::make_pair(rot_matrix, temp_center_gravity);

        return_templates_base.col(temp_ind) =
                EigenHelpers::EigenMatrix3XdToVectorXd(rot_matrix * relocated_template);

        all_rot_and_centers.push_back(rot_and_center);
    //    all_rot_matrixes.push_back(rot_matrix);
    }

    std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> base_and_rot =
            std::make_pair(return_templates_base, all_rot_and_centers);
//    std::pair <Eigen::MatrixXd, std::vector<Eigen::Matrix3d>> base_and_rot =
//            std::make_pair(observable_templates_base, all_rot_matrixes);

    return base_and_rot;
}

const Eigen::VectorXd TemplatesCollector::GetTemplatesWeight(
        const Eigen::Matrix3Xd& object_configuration,
        const Eigen::MatrixXd& templates_base)
{
    const Eigen::MatrixXd base_pinv =
            EigenHelpers::Pinv(templates_base, EigenHelpers::SuggestedRcond());

    const Eigen::VectorXd vectorized_target =
            EigenHelpers::EigenMatrix3XdToVectorXd(object_configuration);

    return base_pinv * vectorized_target;
}

//const Eigen::MatrixXd TemplatesCollector::GetFullMatchingBase(
//        const std::vector<Eigen::Matrix3d>& all_rot)
const Eigen::MatrixXd TemplatesCollector::GetFullMatchingBase(
        const std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>& all_rots_and_centers)
{
    std::vector<Eigen::Matrix3Xd> full_matching_matrixes;
//    for(ssize_t temp_ind = 0; temp_ind < all_rot.size(); temp_ind++)
    for(ssize_t temp_ind = 0; temp_ind < all_rots_and_centers.size(); temp_ind++)
    {
    //    Eigen::Matrix3Xd temp_to_use = all_rot.at(temp_ind) * object_templates_.at(temp_ind);
        Eigen::Matrix3Xd temp_to_use = all_rots_and_centers.at(temp_ind).first * object_templates_.at(temp_ind);

    //    full_matching_matrixes.push_back(
    //                all_rot.at(temp_ind) * object_templates_.at(temp_ind));
        full_matching_matrixes.push_back(
                    RelocateSingleCenterOfGravity(temp_to_use, -all_rots_and_centers.at(temp_ind).second));
    }
    std::vector<Eigen::VectorXd> full_matching_vectors =
            EigenHelpers::EigenMatrix3XdsToVectorXds(full_matching_matrixes);
    return EigenHelpers::stdVectorsToMatrixXd(full_matching_vectors);
}


//////////////////////////////////////////////////////////////////////////////
// functions to utilize observability information
//////////////////////////////////////////////////////////////////////////////

// The first vector store the # of nodes should be observed while not observable; The second vector list the indexes of
// the templates, from most reliable to the least reliable
std::pair<std::vector<int>, std::vector<int>> TemplatesCollector::SortTemplatesReliability(
        const Eigen::Vector3d& current_geo_center,
        const std::pair<Eigen::MatrixXd, std::vector<std::pair<Eigen::MatrixXd, Eigen::Vector3d>>> base_and_rot,
        const WorldState& occluded_world_state)
{
    const Eigen::MatrixXd& observable_templates = base_and_rot.first;
    const std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>& all_rots_and_centers = base_and_rot.second;

    const ObservableIndexes& observable_indexes = occluded_world_state.observabel_correspondency_;
    const ssize_t num_observable_nodes = observable_templates.cols();
    assert (num_observable_nodes == observable_indexes.size());
    const Eigen::Matrix3Xd relocation_matrix = current_geo_center * Eigen::MatrixXd::Ones(1, num_observable_nodes);

    std::vector<int> list_num_violation;
    std::vector<int> list_sorted_reliability;

    for (int temp_ind = 0; temp_ind < object_templates_.size(); temp_ind++)
    {
        int num_violation = 0;
        const Eigen::Matrix3Xd relocated_template = observable_templates + relocation_matrix;
//        const Eigen::Matrix3Xd relocated_templates
        for (ssize_t node_ind = 0; node_ind < num_observable_nodes; node_ind++)
        {
            const Eigen::Vector3d should_observed_point = relocated_template.col(observable_indexes.at(node_ind));
            // TODO:: double check the useage here
            if (observability_map_.EstimateDistance3d(should_observed_point).first < 0.0)
            {
                num_violation++;
            }
        }
        list_num_violation.push_back(num_violation);
        list_sorted_reliability.push_back(temp_ind);
    }

    for (int i = 0; i < num_observable_nodes; i++)
    {
        for (int j = i+1; j < num_observable_nodes; j++)
        {
            if (list_num_violation.at(i) > list_num_violation.at(j))
            {
                const int i_violation = list_num_violation.at(i);
                const int i_index = list_sorted_reliability.at(i);
                list_num_violation.at(i) = list_num_violation.at(j);
                list_num_violation.at(j) = i_violation;
                list_sorted_reliability.at(i) = list_sorted_reliability.at(j);
                list_sorted_reliability.at(j) = i_index;
            }
        }
    }
    //return std::make_pair<std::vector<int>, std::vector<int>>(list_num_violation, list_sorted_reliability);
    return std::make_pair(list_num_violation, list_sorted_reliability);
}

const std::pair<Eigen::MatrixXd, std::vector<std::pair<Eigen::MatrixXd, Eigen::Vector3d>>> TemplatesCollector::GetMostReliableTemplatesbase(
        const std::pair<Eigen::MatrixXd, std::vector<std::pair<Eigen::MatrixXd, Eigen::Vector3d>>>& all_templates,
        const std::vector<int>& sorted_reliable_list)
{
    const Eigen::MatrixXd& observable_templates = all_templates.first;
    const std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>& all_rots_and_centers = all_templates.second;

    Eigen::MatrixXd reliable_templates = Eigen::MatrixXd::Zero(observable_templates.rows(), num_reliable_temps_);
    std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>> reliable_rots_centers;

    for (int temp_ind = 0; temp_ind < num_reliable_temps_; temp_ind++)
    {
        reliable_templates.col(temp_ind) = observable_templates.col(sorted_reliable_list.at(temp_ind));
        reliable_rots_centers.push_back(all_rots_and_centers.at(sorted_reliable_list.at(temp_ind)));
    }
    return std::make_pair(reliable_templates, reliable_rots_centers);
}




//////////////////////////////////////////////////////////////////////////////
// other helper functions
//////////////////////////////////////////////////////////////////////////////

// return the cluster configuration consist of the object configuration with the end-effectors positions
const Eigen::Matrix3Xd TemplatesCollector::GetDataToBeMatched (
        const WorldState& occluded_world_state)
{
    const Eigen::Matrix3Xd& object_configuration = occluded_world_state.object_configuration_;
    const AllGrippersSinglePose& gripper_poses = occluded_world_state.all_grippers_single_pose_;
    const ssize_t num_nodes = object_configuration.cols();
    const ssize_t num_grippers = gripper_poses.size();
    const ssize_t num_positions = num_nodes + num_grippers;

    Eigen::Matrix3Xd data_to_be_matched = Eigen::MatrixXd(3, num_positions);
    data_to_be_matched.block(0, 0, 3, num_nodes) = object_configuration;

    for (ssize_t gripper_ind = 0; gripper_ind < num_grippers; gripper_ind++)
    {
        Eigen::VectorXd gripper_pose = EigenHelpers::TransformToRPY(gripper_poses.at(gripper_ind));
        data_to_be_matched.col(num_nodes + gripper_ind)= gripper_pose.head(3);
    }
    return data_to_be_matched;
}


// TODO: Keep this version or change the input of "addTemplate", use the "estimation" function as similarity check
const bool TemplatesCollector::HaveSimilarTemplate(const Eigen::Matrix3Xd& template_in)
{

    if (object_templates_.size() > 0)
    {
        assert(template_in.cols() == object_templates_.at(0).cols());
    }
    else
    {
        return false;
    }
    bool similarity_exists = false;

    // Move the center of the observable object to the origin for rotation alignment later
    const Eigen::Matrix3Xd relocated_configuration_in = RelocateSingelCenterToOrigin(template_in);

    const std::vector<Eigen::VectorXd> vector_templates =
            EigenHelpers::EigenMatrix3XdsToVectorXds(object_templates_);
    const Eigen::MatrixXd all_templates_matrix
            = EigenHelpers::stdVectorsToMatrixXd(vector_templates);

    const std::pair <Eigen::MatrixXd, std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>> base_and_rot
            = GetObservableTemplateBaseForSingelConfiguration(
                relocated_configuration_in,
                all_templates_matrix);

    const Eigen::MatrixXd& observable_templates = base_and_rot.first;
    const std::vector<std::pair <Eigen::MatrixXd,Eigen::Vector3d>>& all_rots_and_centers = base_and_rot.second;

    const Eigen::VectorXd templates_weight = GetTemplatesWeight(
                relocated_configuration_in,
                observable_templates);

    const Eigen::MatrixXd full_matching_base = GetFullMatchingBase(all_rots_and_centers);

    // Check the independence of the new sample with respect to the existing templates
    Eigen::Matrix3Xd estimation = EigenHelpers::EigenVectorXdToMatrix3Xd(
                full_matching_base
                * templates_weight);

    double error = 0;
    for (ssize_t node_ind = 0; node_ind < estimation.cols(); node_ind++)
    {
        const Eigen::Vector3d diff = estimation.col(node_ind) - relocated_configuration_in.col(node_ind);
        error += diff.norm();
    }
    double min_error = 0.5;
    if(error < min_error)
    {
        similarity_exists = true;
    }
    return similarity_exists;


    // Previous version; be changed to independence check
    /*
    double min_error = 10;

    for (ssize_t temp_ind = 0; temp_ind < object_templates_.size(); temp_ind++)
    {
        const Eigen::Matrix3d rot_matrix = GetSingleRotAlignmentMatrix(
                    relocated_configuration_in,
                    object_templates_.at(temp_ind));
        const Eigen::Matrix3Xd diff_matrix =
                relocated_configuration_in - rot_matrix * object_templates_.at(temp_ind);

        double diff_to_template = 0;
        for (ssize_t node_ind = 0; node_ind < relocated_configuration_in.cols(); node_ind++)
        {
            const Eigen::Vector3d diff = diff_matrix.col(node_ind);
            diff_to_template = diff_to_template + diff.norm();
        }
        if (diff_to_template < min_error)
        {
            min_error = diff_to_template;
        }
    }

    ROS_INFO_STREAM_NAMED("templates_collection", "Minimum diff in similarity to templates collected: " << min_error);

    if (min_error < similarity_threshold_)
    {
        similarity_exists = true;
    }
    return similarity_exists;
    */

}

void TemplatesCollector::PrintCoordinateHelper(const Eigen::VectorXd& coordinate)
{
    const int num_coord = coordinate.rows();

    int count = 0;
    // print 10 coords in a line
    for (int ind = 0; ind *10 + 9 < num_coord; ind++)
    {
        ROS_INFO_STREAM_NAMED("templates_collection", "Estimate coordinate for the base matrix: "
                              << coordinate(ind * 10 + 0, 0) << " ; "
                              << coordinate(ind * 10 + 1, 0) << " ; "
                              << coordinate(ind * 10 + 2, 0) << " ; "

                              << coordinate(ind * 10 + 3, 0) << " ; "
                              << coordinate(ind * 10 + 4, 0) << " ; "
                              << coordinate(ind * 10 + 5, 0) << " ; "

                              << coordinate(ind * 10 + 6, 0) << " ; "
                              << coordinate(ind * 10 + 7, 0) << " ; "
                              << coordinate(ind * 10 + 8, 0) << " ; "

                              << coordinate(ind * 10 + 9, 0) << " ; ");
        count = ind + 1;
    }
    for (int ind = count * 10; ind < num_coord; ind ++)
    {
        ROS_INFO_STREAM_NAMED("templates_collection", "Estimate coordinate for the base matrix: "
                              << coordinate(ind, 0) << " ;");
    }

}






