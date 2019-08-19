#include "smmap/parabola.h"
#include <arc_utilities/arc_helpers.hpp>
#include <deformable_manipulation_experiment_params/utility.hpp>

using namespace Eigen;

namespace smmap
{
    // Assumes that (x0, y0) is the origin, and thus c = 0. Returns (a, b)
    // such that y = a*x^2 + b*x + 0; such that the arc length from (0, 0) to
    // (x1, y1) = 'length', and the parabola is convex. Also assumes that x1 is positive.
    // https://stackoverflow.com/questions/48486254/determine-parabola-with-given-arc-length-between-two-known-points
    inline std::pair<double, double> FindParabolaCoeffs(
            const double x1,
            const double y1,
            const double length,
            const bool verbose = false)
    {
        if (verbose)
        {
            std::cerr << "x1: " << x1 << "    y1: " << y1 << "    length: " << length << std::endl;
        }
        assert(x1 > 0.0);

        // Precomputed as it is used multiple places
        const auto ratio = y1 / x1;

        // Simplified formula inside the integral for the arc length of a parabola
        const auto dIntegral = [] (const double t)
        {
            return std::sqrt(1.0 + t*t);
        };

        // The result of integrating dIntegral without applying bounds
        const auto Integral = [] (const double t)
        {
            const auto rt = std::sqrt(1.0 + t*t);
            return 0.5 * (t * rt + std::log(t + rt));
        };

        // The arclength of a parabola based on the above
        const auto arc_len_fn = [&] (const double a)
        {
            const auto upper = ratio + a*x1;
            const auto lower = ratio - a*x1;
            return 0.5 * (Integral(upper) - Integral(lower)) / a;
        };

        const auto darc_len_fn = [&] (const double a)
        {
            const auto upper = ratio + a*x1;
            const auto lower = ratio - a*x1;
            return 0.5 * (a*x1 * (dIntegral(upper) + dIntegral(lower)) + Integral(lower) - Integral(upper)) / (a*a);
        };

        const auto N = 1000;
        const auto EPSILON = 1e-5;
        // Start with a guess that is guaranteed to be positive, and could be in vaguely the right place
        double guess = std::abs(ratio) + 1.0 / x1;
        if (verbose)
        {
            std::cerr << "0: " << guess << " : " << arc_len_fn(guess) << std::endl;
        }

        for (int n = 0; n < N; ++n)
        {
            const auto dguess = (arc_len_fn(guess) - length) / darc_len_fn(guess);
            guess -= dguess;
            if (verbose)
            {
                std::cerr << n+1 << ": " << guess << " : " << arc_len_fn(guess) << std::endl;
                if (!std::isfinite(guess) || guess <= 0.0)
                {
                    PressAnyKeyToContinue("Weirdness in FindCoeffs");
                    assert(false);
                }
            }
            if (!std::isfinite(guess) || guess <= 0.0)
            {
                std::cerr << "x1: " << x1 << "    y1: " << y1 << "    length: " << length << std::endl;
                return FindParabolaCoeffs(x1, y1, length, true);
            }
            if (std::abs(dguess) <= EPSILON)
            {
                break;
            }
        }

        const double a = guess;
        const double b = ratio - a*x1;

        if (!std::isfinite(a) || a < 0.0)
        {
            std::cerr << "x1: " << x1 << "    y1: " << y1 << "    length: " << length << std::endl;
            std::cerr << "guess: " << guess << std::endl;
            assert(a >= 0.0);
        }
        assert(a >= 0.0);

        return {a, b};
    }

    sdf_tools::CollisionMapGrid ExtractParabolaSlice(
            const sdf_tools::SignedDistanceField& sdf,
            const double resolution,
            const PairGripperPositions& gripper_positions,
            const double parabola_length,
            const std::shared_ptr<Visualizer> vis_)
    {
        // Precalculate this once for efficiency
        const double half_res = 0.5 * resolution;

        // Special case the instance when the grippers are approximately in line
        // with the gravity vector; anything that fits within 1 cell counts as
        // "approximately in line"
        const Vector3d gripper_delta = gripper_positions.second - gripper_positions.first;
        if (gripper_delta.head<2>().norm() <= resolution)
        {
            const Vector3d midpoint = (gripper_positions.first + gripper_positions.second) * 0.5;
            const double z_range = std::abs(gripper_delta.z());

            // Return a grid which is bloated by 1 cell around the z-axis
            // and is thus 3x1x(N+2) in size, with the outer rectangle marked as
            // "out of bounds". Which dimension is "3" and which is "1" does not
            // matter as only the central portion is looked up in the SDF anyway
            const int64_t x_cells = 3;
            const int64_t y_cells = 1;
            const int64_t z_cells = (int64_t)std::ceil(0.5 * (parabola_length - z_range)) + 2;

            // Note that because we are just using one vertical column of voxels
            // the orientation doesn't matter; x and y are interchangable throughout
            const Vector3d offset(midpoint.x() - 1.5 * resolution,
                                  midpoint.y(),
                                  midpoint.z() - (double)z_cells * resolution * 0.5);

            sdf_tools::CollisionMapGrid grid(Isometry3d(Translation3d(offset)),
                                             sdf.GetFrame(),
                                             resolution,
                                             x_cells,
                                             y_cells,
                                             z_cells,
                                             sdf_tools::COLLISION_CELL(1.0),
                                             sdf_tools::COLLISION_CELL(1.0));

            const int64_t x_idx = 1;
            const int64_t y_idx = 0;
            for (int64_t z_idx = 1; z_idx < z_cells - 1; ++z_idx)
            {
                const auto location_world_frame = grid.GridIndexToLocation(x_idx, y_idx, z_idx);
                const auto sdf_val = sdf.GetImmutable4d(location_world_frame);
                // If the sdf lookup is valid, then set the collision cell accordingly,
                // otherwise leave it at the default "filled" value
                if (sdf_val.second)
                {
                    const auto occupancy = sdf_val.first > 0.0 ? 0.0f : 1.0f;
                    grid.SetValue(x_idx, y_idx, z_idx, sdf_tools::COLLISION_CELL(occupancy));
                }
            }

            return grid;
        }

        // For the purposes of calculating the parabola coefficients, rotate the
        // frame so that gripper_positions.first is at the origin, with the
        // gravity vector pointing along positive-y, and
        const Isometry3d parabola_origin = [&]
        {
            const Vector3d x_axis = Vector3d(gripper_delta.x(), gripper_delta.y(), 0.0).normalized();
            const Vector3d y_axis = Vector3d::UnitZ();
            const Vector3d z_axis = x_axis.cross(y_axis).normalized();

            return Isometry3d((Matrix4d() << x_axis, y_axis, z_axis, gripper_positions.first,
                                             0.0,    0.0,    0.0,    1.0).finished());
        }();
        const Isometry3d parabola_origin_inv = parabola_origin.inverse();
        const Vector3d second_point_parabola_frame = parabola_origin_inv * gripper_positions.second;
        const Vector2d line_normal_parabola_frame(-second_point_parabola_frame.y(), second_point_parabola_frame.x());
        const auto coeffs = FindParabolaCoeffs(second_point_parabola_frame.x(), second_point_parabola_frame.y(), parabola_length);
        const double a = coeffs.first;
        const double b = coeffs.second;
        const auto parabola_eqn = [&] (const double x)
        {
            return a * x * x + b * x;
        };
        // Ensure that the parabola is convex:
        assert(std::isfinite(a) && a >= 0.0 && "Parabola must be convex");
        // If b is positive, then there is not enough slack to create a loop below the gripper
        const double x_min = 0.0;
        const double x_max = second_point_parabola_frame.x();
        const double x_lowest = arc_helpers::ClampValue(-b/(2*a), x_min, x_max);
        const double y_min = parabola_eqn(x_lowest);
        const double y_max = std::max(0.0, second_point_parabola_frame(1));
        const double x_range = x_max - x_min;
        const double y_range = y_max - y_min;
        assert(y_max >= y_min);
        const int64_t x_cells = (int64_t)std::ceil(x_range / resolution) + 2;
        const int64_t y_cells = (int64_t)std::ceil(y_range / resolution) + 2;
        const int64_t z_cells = 1;

    //        std::cerr << "Gripper pair:                " << PrettyPrint::PrettyPrint(gripper_positions, false, ", ") << std::endl;
    //        std::cerr << "second_point_parabola_frame: " << second_point_parabola_frame.transpose() << std::endl;
    //        std::cerr << "x_min: " << x_min << " x_max: " << x_max << " x_lowest: " << x_lowest << std::endl;
    //        std::cerr << "y_min: " << y_min << " y_max: " << y_max << std::endl;

        // Move the origin to center the parabola; will ensure a 1 cell boundary by construction
        const Vector3d grid_offset(x_min - 0.5 * (resolution * (double)x_cells - x_range),  // Center the valid region of the voxel grid between the grippers
                                   y_min - 0.5 * (resolution * (double)y_cells - y_range),  // Center the valid region of the voxel grid on the parabola
                                   -half_res);                                              // Shift half a cell to put the slice directly overtop of the grippers
    //        std::cerr << "Grid offset:      " << grid_offset.transpose() << std::endl;
    //        std::cerr << "Parabola origin:  " << parabola_origin.translation().transpose() << std::endl;
    //        std::cerr << "Grid origin:      " << (parabola_origin * Translation3d(grid_offset)).translation().transpose() << std::endl;
    //        std::cerr << std::endl;

        sdf_tools::CollisionMapGrid grid(parabola_origin * Translation3d(grid_offset),
                                         sdf.GetFrame(),
                                         resolution,
                                         x_cells,
                                         y_cells,
                                         z_cells,
                                         sdf_tools::COLLISION_CELL(1.0),
                                         sdf_tools::COLLISION_CELL(1.0));

        if (vis_ != nullptr)
        {
            const VectorXd x = VectorXd::LinSpaced(1000, 0.0, x_max);
            const VectorXd y = x.unaryExpr(parabola_eqn);
            const VectorXd z = VectorXd::Zero(x.rows());
            vis_->visualizeLineStrip("parabola",
                                     parabola_origin * (ObjectPointSet(3, x.rows()) << x.transpose(),
                                                                                       y.transpose(),
                                                                                       z.transpose()).finished(),
                                     Visualizer::Magenta(), 1, 0.002);
            vis_->visualizeLineStrip("parabola",
                                     (ObjectPointSet(3, 2) << gripper_positions.first, gripper_positions.second).finished(),
                                     Visualizer::Magenta(), 2, 0.002);
            vis_->visualizeAxes("parabola_origin", parabola_origin, 0.05, 0.005, 1);
        }

        // Iterate through the grid, only considering cells inside the parabola
        for (int64_t x_idx = 1; x_idx < x_cells - 1; ++x_idx)
        {
            for (int64_t y_idx = 1; y_idx < y_cells - 1; ++y_idx)
            {
                const int64_t z_idx = 0;
                const auto location_grid_frame = grid.GridIndexToLocationGridFrame(x_idx, y_idx, z_idx);
                const auto location_world_frame = grid.GetOriginTransform() * location_grid_frame;

                const Vector2d point_parabola_frame = (parabola_origin_inv * location_world_frame).head<2>();
                const bool above_parabola = point_parabola_frame.y() >= parabola_eqn(point_parabola_frame.x());
                const bool below_line = line_normal_parabola_frame.dot(point_parabola_frame) <= 0.0;

                // If the location is inside the parabola, then check if it is filled,
                if (above_parabola && below_line)
                {
                    const auto sdf_val = sdf.GetImmutable4d(location_world_frame);
                    // If the sdf lookup is valid, then set the collision cell accordingly,
                    // otherwise leave it at the default "filled" value
                    if (sdf_val.second)
                    {
                        const auto occupancy = sdf_val.first > 0.0 ? 0.0f : 1.0f;
                        grid.SetValue(x_idx, y_idx, z_idx, sdf_tools::COLLISION_CELL(occupancy));
                    }
                }
                // Check if we're at an edge case - i.e. the line passes through the voxel,
                // but the voxel center is on the wrong side of something
                else
                {
                    const Vector2d top_right    = point_parabola_frame + Vector2d( half_res,  half_res);
                    const Vector2d bottom_right = point_parabola_frame + Vector2d( half_res, -half_res);
                    const Vector2d bottom_left  = point_parabola_frame + Vector2d(-half_res, -half_res);
                    const Vector2d top_left     = point_parabola_frame + Vector2d(-half_res,  half_res);

                    // Check if the line itself passes through the voxel
                    const bool line_bisects_voxel = [&]
                    {
                        const double top_right_dot_product      = line_normal_parabola_frame.dot(top_right);
                        const double bottom_right_dot_product   = line_normal_parabola_frame.dot(bottom_right);
                        const double bottom_left_dot_product    = line_normal_parabola_frame.dot(bottom_left);
                        const double top_left_dot_product       = line_normal_parabola_frame.dot(top_left);

                        const bool top_right_below_line     = top_right_dot_product       < 0.0;
                        const bool bottom_right_below_line  = bottom_right_dot_product    < 0.0;
                        const bool bottom_left_below_line   = bottom_left_dot_product     < 0.0;
                        const bool top_left_below_line      = top_left_dot_product        < 0.0;

                        // If the line bisects the voxel, then it passes through
                        return (top_right_below_line ^ bottom_left_below_line) ||
                               (top_left_below_line ^ bottom_right_below_line);
                    }();
                    // Check if the parabola passes through the voxel
                    const bool parabola_bisects_voxel = [&]
                    {
                        const double top_right_parabola_val     = parabola_eqn(top_right.x());
                        const double bottom_right_parabola_val  = parabola_eqn(bottom_right.x());
                        const double bottom_left_parabola_val   = parabola_eqn(bottom_left.x());
                        const double top_left_parabola_val      = parabola_eqn(top_left.x());

                        const bool top_right_above_parabola     = top_right.y()     > top_right_parabola_val;
                        const bool bottom_right_above_parabola  = bottom_right.y()  > bottom_right_parabola_val;
                        const bool bottom_left_above_parabola   = bottom_left.y()   > bottom_left_parabola_val;
                        const bool top_left_above_parabola      = top_left.y()      > top_left_parabola_val;

                        // If the parabola bisects the voxel, then it passes through
                        return (top_right_above_parabola && !bottom_left_above_parabola) ||
                               (top_left_above_parabola && !bottom_right_above_parabola);
                    }();

                    if ((line_bisects_voxel && above_parabola) ||
                        (parabola_bisects_voxel && below_line) ||
                        (line_bisects_voxel && parabola_bisects_voxel))
                    {
                        const auto sdf_val = sdf.GetImmutable4d(location_world_frame);
                        // If the sdf lookup is valid, then set the collision cell accordingly,
                        // otherwise leave it at the default "filled" value
                        if (sdf_val.second)
                        {
                            const auto occupancy = sdf_val.first > 0.0 ? 0.0f : 1.0f;
                            grid.SetValue(x_idx, y_idx, z_idx, sdf_tools::COLLISION_CELL(occupancy));
                        }
                    }
                }
            }
        }

        return grid;
    }

    // Using inline to mark that this is not currently used anywhere
    inline sdf_tools::CollisionMapGrid ExtractEllipseSlice(
            const sdf_tools::SignedDistanceField& sdf,
            const Eigen::Isometry3d& origin,
            const double resolution,
            const int64_t x_cells,
            const int64_t y_cells,
            const int64_t z_cells,
            const Vector3d ellipse_center,
            const double x_axis_len2,
            const double y_axis_len2)
    {
        assert(resolution > 0.0);
        sdf_tools::CollisionMapGrid grid(origin,
                                         sdf.GetFrame(),
                                         resolution,
                                         x_cells,
                                         y_cells,
                                         z_cells,
                                         sdf_tools::COLLISION_CELL(1.0));

        // We assume that the ellipse is axis aligned in the grid frame (x-major, y-minor)
        const Vector3d ellipse_center_grid_frame = grid.GetInverseOriginTransform() * ellipse_center;

        for (int64_t x_idx = 0; x_idx < x_cells; ++x_idx)
        {
            for (int64_t y_idx = 0; y_idx < y_cells; ++y_idx)
            {
                for (int64_t z_idx = 0; z_idx < z_cells; ++z_idx)
                {
                    const auto location_grid_frame = grid.GridIndexToLocationGridFrame(x_idx, y_idx, z_idx);
                    const auto location_world_frame = grid.GetOriginTransform() * location_grid_frame;
                    const auto dx = location_grid_frame(0) - ellipse_center_grid_frame.x();
                    const auto dy = location_grid_frame(1) - ellipse_center_grid_frame.y();
                    const bool inside_ellipse = (dx * dx / x_axis_len2 + dy * dy / y_axis_len2) < 1.0;

                    // If the location is inside the ellipse, then check if it is filled,
                    // otherwise eave it at the default "filled" value
                    if (inside_ellipse)
                    {
                        const auto sdf_val = sdf.GetImmutable4d(location_world_frame);
                        // If the sdf lookup is valid, then set the collision cell accordingly,
                        // otherwise leave it at the default "filled" value
                        if (sdf_val.second)
                        {
                            const auto occupancy = sdf_val.first > 0.0 ? 0.0f : 1.0f;
                            grid.SetValue(x_idx, y_idx, z_idx, sdf_tools::COLLISION_CELL(occupancy));
                        }
                    }
                }
            }
        }

        return grid;
    }
}
