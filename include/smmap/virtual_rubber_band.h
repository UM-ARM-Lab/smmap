#ifndef VIRTUAL_RUBBER_BAND_H
#define VIRTUAL_RUBBER_BAND_H

#include <arc_utilities/eigen_helpers.hpp>
#include "smmap/task_specification.h"

namespace smmap
{
    class VirtualRubberBand
    {
        public:
            static constexpr auto BAND_POST_FORWARD_STEP_NS         = "band_post_forward_step";
            static constexpr auto BAND_POST_SUBDIVIDE_STEP_NS       = "band_post_subdivide_step";
            static constexpr auto BAND_POST_SHORTCUT_SMOOTHING_NS   = "band_post_shortcut_smoothing";
            static constexpr auto BAND_BADNESS_NS                   = "band_badness";

            VirtualRubberBand(
                    const Eigen::Vector3d& start_point,
                    const Eigen::Vector3d& end_point,
                    const std::shared_ptr<DijkstrasCoverageTask>& task,
                    const Visualizer& vis,
                    std::mt19937_64& generator);

            VirtualRubberBand(
                    EigenHelpers::VectorVector3d starting_points,
                    const double max_total_band_distance,
                    const std::shared_ptr<DijkstrasCoverageTask>& task,
                    const Visualizer& vis,
                    std::mt19937_64& generator);

            smmap::VirtualRubberBand& operator=(const smmap::VirtualRubberBand& other);

            void setPointsWithoutSmoothing(const EigenHelpers::VectorVector3d& points);

            void setPointsAndSmooth(const EigenHelpers::VectorVector3d& points);

            const EigenHelpers::VectorVector3d& forwardSimulateVirtualRubberBandToEndpointTargets(
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

        private:
            void resampleBand(const bool verbose);
            void shortcutSmoothBand(const bool verbose);

            const std::shared_ptr<DijkstrasCoverageTask> task_;
            const sdf_tools::SignedDistanceField& sdf_;
            const Visualizer& vis_;

        public:
            const double max_integration_step_size_;
            const double max_distance_between_rubber_band_points_;
            const int num_smoothing_iters_per_band_point_;
            const int min_smoothing_iters_;
            const double min_object_radius_;
            const double max_total_band_distance_;

        private:
            EigenHelpers::VectorVector3d band_;

            std::mt19937_64& generator_;
    };
}


#endif // VIRTUAL_RUBBER_BAND_H
