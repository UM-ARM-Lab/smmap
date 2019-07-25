#ifndef PARABOLA_H
#define PARABOLA_H

#include <sdf_tools/collision_map.hpp>
#include <smmap_utilities/grippers.h>
#include <smmap_utilities/visualization_tools.h>

namespace smmap
{
    sdf_tools::CollisionMapGrid ExtractParabolaSlice(
            const sdf_tools::SignedDistanceField& sdf,
            const double resolution,
            const PairGripperPositions& gripper_positions,
            const double parabola_length,
            const std::shared_ptr<Visualizer> vis_ = nullptr);
}

#endif // PARABOLA_H
