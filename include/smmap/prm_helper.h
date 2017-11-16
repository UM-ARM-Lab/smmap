#ifndef PRM_HELPER_HPP
#define PRM_HELPER_HPP

#include <Eigen/Dense>
#include <arc_utilities/simple_prm_planner.hpp>
#include <sdf_tools/sdf.hpp>
#include <smmap_utilities/visualization_tools.h>

namespace smmap
{
    class PRMHelper
    {
        public:
            // Topic names used for publishing visualization data
            static constexpr auto PRM_NODES_NS       = "prm_nodes";
            static constexpr auto PRM_EDGES_NS       = "prm_edges";
            static constexpr auto PRM_QUERRY_NS      = "prm_querry";
            static constexpr auto RRM_SAMPLE_NS      = "prm_sample";
            static constexpr auto PRM_RANDOM_PATH_NS = "prm_random_path";


            PRMHelper(
                    const sdf_tools::SignedDistanceField& environment_sdf,
                    const smmap_utilities::Visualizer& vis,
                    std::mt19937_64& generator,
                    const Eigen::Vector3d planning_world_lower_limits,
                    const Eigen::Vector3d planning_world_upper_limits,
                    const bool visualization_enabled,
                    const size_t num_neighbours_to_connect,
                    const size_t num_samples_to_try,
                    const double edge_validity_stepsize);

            void initializeRoadmap();

            void visualize(const bool visualization_enabled = true);

            EigenHelpers::VectorVector3d getRandomPath(const Eigen::Vector3d& start, const Eigen::Vector3d& goal);

        private:
            const Eigen::Vector3d planning_world_lower_limits_;
            const Eigen::Vector3d planning_world_upper_limits_;

            std::mt19937_64& generator_;
            std::uniform_real_distribution<double> uniform_unit_distribution_;
            const sdf_tools::SignedDistanceField& environment_sdf_;
            const smmap_utilities::Visualizer& vis_;
            const bool visualization_enabled_globally_;

            arc_dijkstras::Graph<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> roadmap_;
            bool roadmap_initialized_;
            const size_t num_neighbours_to_connect_;
            const size_t num_samples_to_try_;
            size_t num_samples_done_;
            const double edge_validity_stepsize_;

            //////// Passed using std::bind to various SimpleGeometricPrmPlanner functions /////////////////////////////

            Eigen::Vector3d sampleState();

            static double distance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2);

            bool validState(const Eigen::Vector3d& p) const;

            bool validEdge(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) const;
    };
}


#endif // PRM_HELPER
