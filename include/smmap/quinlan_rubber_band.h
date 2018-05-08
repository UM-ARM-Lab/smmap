#ifndef QUINLAN_RUBBER_BAND
#define QUINLAN_RUBBER_BAND

#include <arc_utilities/eigen_helpers.hpp>
#include <smmap_utilities/visualization_tools.h>

#include "smmap/task_specification.h"

//#define ENABLE_BAND_DEBUGGING 1
#define ENABLE_BAND_DEBUGGING 0
//#define ENABLE_BAND_LOAD_SAVE 1
#define ENABLE_BAND_LOAD_SAVE 0

namespace smmap
{
    class QuinlanRubberBand
    {
    public:
        typedef std::shared_ptr<QuinlanRubberBand> Ptr;

        QuinlanRubberBand(
                const Eigen::Vector3d& start_point,
                const Eigen::Vector3d& end_point,
                const std::shared_ptr<DijkstrasCoverageTask>& task,
                const smmap_utilities::Visualizer::Ptr vis,
                const std::shared_ptr<std::mt19937_64>& generator);

        QuinlanRubberBand(
                EigenHelpers::VectorVector3d starting_points,
                const double max_total_band_distance,
                const std::shared_ptr<DijkstrasCoverageTask>& task,
                const smmap_utilities::Visualizer::Ptr vis,
                const std::shared_ptr<std::mt19937_64>& generator);

        QuinlanRubberBand& operator=(const QuinlanRubberBand& other);

        void setPointsWithoutSmoothing(const EigenHelpers::VectorVector3d& points);

        void setPointsAndSmooth(const EigenHelpers::VectorVector3d& points);

        const EigenHelpers::VectorVector3d& forwardPropagateRubberBandToEndpointTargets(
                const Eigen::Vector3d first_endpoint_target,
                const Eigen::Vector3d second_endpoint_target,
                bool verbose);

        const EigenHelpers::VectorVector3d& getVectorRepresentation() const;

        std::pair<Eigen::Vector3d, Eigen::Vector3d> getEndpoints() const;

        double maxSafeLength() const;
        double totalLength() const;
        bool isOverstretched() const;

        void visualize(
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled) const;

        void visualize(
                const EigenHelpers::VectorVector3d& test_band,
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled) const;

        void visualizeWithBubbles(
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled) const;

        void visualizeWithBubbles(
                const EigenHelpers::VectorVector3d& test_band,
                const std::string& marker_name,
                const std_msgs::ColorRGBA& safe_color,
                const std_msgs::ColorRGBA& overstretched_color,
                const int32_t id,
                const bool visualization_enabled) const;

        uint64_t serialize(std::vector<uint8_t>& buffer) const;
        uint64_t deserializeIntoSelf(const std::vector<uint8_t>& buffer, const uint64_t current);

    private:
        ros::NodeHandle ph_;
        const std::shared_ptr<DijkstrasCoverageTask> task_;
        const sdf_tools::SignedDistanceField::ConstPtr sdf_;
        const smmap_utilities::Visualizer::Ptr vis_;

        EigenHelpers::VectorVector3d band_;

        const double max_total_band_distance_;
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
}

#endif //QUINLAN_RUBBER_BAND
