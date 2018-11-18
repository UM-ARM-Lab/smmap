#ifndef QUINLAN_RUBBER_BAND
#define QUINLAN_RUBBER_BAND

#include <arc_utilities/eigen_helpers.hpp>
#include <smmap_utilities/visualization_tools.h>

#include "smmap/task_specification.h"

namespace smmap
{
    class QuinlanRubberBand
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef std::shared_ptr<QuinlanRubberBand> Ptr;
        typedef std::shared_ptr<const QuinlanRubberBand> ConstPtr;

        QuinlanRubberBand(
                std::shared_ptr<ros::NodeHandle> nh,
                std::shared_ptr<ros::NodeHandle> ph,
                smmap_utilities::Visualizer::Ptr vis,
                const DijkstrasCoverageTask::ConstPtr& task,
                EigenHelpers::VectorVector3d starting_points,
                const double resample_max_pointwise_dist,
                const size_t upsample_num_points,
                const double max_total_band_distance);

        QuinlanRubberBand& operator=(const QuinlanRubberBand& other);

        void setPointsWithoutSmoothing(const EigenHelpers::VectorVector3d& points);

        void setPointsAndSmooth(const EigenHelpers::VectorVector3d& points);

        void overridePoints(const EigenHelpers::VectorVector3d& points);

        const EigenHelpers::VectorVector3d& forwardPropagateRubberBandToEndpointTargets(
                const Eigen::Vector3d first_endpoint_target,
                const Eigen::Vector3d second_endpoint_target,
                bool verbose);

        const EigenHelpers::VectorVector3d& getVectorRepresentation() const;
        // Not threadsafe: https://www.justsoftwaresolutions.co.uk/cplusplus/const-and-thread-safety.htm
        const EigenHelpers::VectorVector3d& resampleBand() const;
        // Not threadsafe: https://www.justsoftwaresolutions.co.uk/cplusplus/const-and-thread-safety.htm
        const EigenHelpers::VectorVector3d& upsampleBand() const;
        const Eigen::VectorXd& upsampleBandSingleVector() const;

        std::pair<Eigen::Vector3d, Eigen::Vector3d> getEndpoints() const;

        double maxSafeLength() const;
        double totalLength() const;
        bool isOverstretched() const;

        void visualize(
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled = true) const;

        void visualize(
                const EigenHelpers::VectorVector3d& test_band,
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled = true) const;

        void visualizeWithBubbles(
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled = true) const;

        void visualizeWithBubbles(
                const EigenHelpers::VectorVector3d& test_band,
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled = true) const;

        uint64_t serialize(std::vector<uint8_t>& buffer) const;
        uint64_t deserializeIntoSelf(const std::vector<uint8_t>& buffer, const uint64_t current);

        double distanceSq(const QuinlanRubberBand& other) const;
        double distance(const QuinlanRubberBand& other) const;
        static double DistanceSq(const QuinlanRubberBand& b1, const QuinlanRubberBand& b2);
        static double Distance(const QuinlanRubberBand& b1, const QuinlanRubberBand& b2);

    private:
        const std::shared_ptr<ros::NodeHandle> nh_;
        const std::shared_ptr<ros::NodeHandle> ph_;
        const DijkstrasCoverageTask::ConstPtr task_;
        const sdf_tools::SignedDistanceField::ConstPtr sdf_;
        const smmap_utilities::Visualizer::Ptr vis_;

        EigenHelpers::VectorVector3d band_;
        // Not threadsafe: https://www.justsoftwaresolutions.co.uk/cplusplus/const-and-thread-safety.htm
        mutable EigenHelpers::VectorVector3d resampled_band_; // is cleared every time band_ is updated
        const double resample_max_pointwise_dist_;
        mutable EigenHelpers::VectorVector3d upsampled_band_;  // is cleared every time band_ is updated
        mutable Eigen::VectorXd upsampled_band_single_vector_; // is updated every time upsampled_band_ is updated
        const size_t upsample_num_points_;


        const double max_safe_band_length_;
        const double min_overlap_distance_;
        const double min_distance_to_obstacle_;
        const double node_removal_overlap_factor_;
        const double backtrack_threshold_;
        const size_t smoothing_iterations_;

        Eigen::Vector3d projectToValidBubble(const Eigen::Vector3d& location) const;
        double getBubbleSize(const Eigen::Vector3d& location) const;
        bool sufficientOverlap(
                const double bubble_size_a,
                const double bubble_size_b,
                const double distance) const;
        bool bandIsValid() const;
        bool bandIsValid(const EigenHelpers::VectorVector3d& test_band) const;
        bool bandIsValidWithVisualization() const;
        bool bandIsValidWithVisualization(const EigenHelpers::VectorVector3d& test_band) const;

        void interpolateBetweenPoints(
                EigenHelpers::VectorVector3d& point_buffer,
                const Eigen::Vector3d& target) const;
        void interpolateBandPoints();
        void removeExtraBandPoints(const bool verbose);
        void smoothBandPoints(const bool verbose);

        void printBandData(const EigenHelpers::VectorVector3d& test_band) const;

        void storeBand() const;
        void loadStoredBand();
        bool useStoredBand() const;
    };

    typedef QuinlanRubberBand RubberBand;
}

#endif //QUINLAN_RUBBER_BAND
