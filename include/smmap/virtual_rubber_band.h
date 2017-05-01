#ifndef VIRTUAL_RUBBER_BAND_H
#define VIRTUAL_RUBBER_BAND_H

#include <arc_utilities/eigen_helpers.hpp>
#include "smmap/task_specification.h"

namespace smmap
{
    class VirtualRubberBand
    {
        public:
            VirtualRubberBand(
                    const Eigen::Vector3d& start_point,
                    const Eigen::Vector3d& end_point,
                    const std::shared_ptr<DijkstrasCoverageTask>& task,
                    const Visualizer& vis);

            const EigenHelpers::VectorVector3d& forwardSimulateVirtualRubberBand(
                    const Eigen::Vector3d first_endpoint_translation,
                    const Eigen::Vector3d second_endpoint_translation,
                    bool verbose);

            const EigenHelpers::VectorVector3d& getVectorRepresentation() const;

            bool isOverstreched() const;

            void visualize(
                    const std::string& marker_name,
                    const std_msgs::ColorRGBA& color,
                    const int32_t id) const;

        private:
            Eigen::Vector3d projectOutOfObstacle(Eigen::Vector3d vec) const;


            const std::shared_ptr<DijkstrasCoverageTask> task_;
            const sdf_tools::SignedDistanceField& sdf_;
            const Visualizer& vis_;

            const double max_integration_step_size_;
            const double max_distance_between_rubber_band_points_;
            const int num_smoothing_ittrs_;
            const double min_object_radius_;
            const double max_total_band_distance_;

            EigenHelpers::VectorVector3d band_;

//            std::default_random_engine generator_(std::chrono::system_clock::now().time_since_epoch().count());
            std::default_random_engine generator_;
    };
}



#endif // VIRTUAL_RUBBER_BAND_H
