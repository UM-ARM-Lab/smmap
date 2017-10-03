#ifndef NOMAD_SOLVERS_H
#define NOMAD_SOLVERS_H

#include "nomad.hpp"
#include "smmap/task_function_pointer_types.h"
#include <Eigen/Dense>
//#include <sdf_tools/sdf.hpp>

namespace smmap {


    ///////////////////////////////////////////////////////////////////
    // Helper function to switch between eigen and nomad type
    ///////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////
    // Class interface for the evaluator
    ///////////////////////////////////////////////////////////////////
    class GripperMotionNomadEvaluator : public NOMAD::Evaluator
    {
        public:
          GripperMotionNomadEvaluator(
                  const NOMAD::Parameters& p,
                  const ssize_t num_grippers,
                  const double gripper_radius,
                  const double stretching_threshold,
                  const double max_step_size,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& eval_error_cost_fn,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& collision_constraint_fn,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& stretching_constraint_fn,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& gripper_motion_constraint_fn,
                  const bool fix_step_size = false);

          ~GripperMotionNomadEvaluator( void ) {}

          bool eval_x (
                  NOMAD::Eval_Point& x,
                  const NOMAD::Double& h_max,
                  bool& count_eval);

          AllGrippersSinglePoseDelta evalPointToGripperPoseDelta(
                  const NOMAD::Eval_Point& x);

        private:
          const ssize_t num_grippers_;

          const NOMAD::Double gripper_radius_;
          const NOMAD::Double stretching_threshold_;
          const NOMAD::Double max_step_size_;
          const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& eval_error_cost_fn_;
          const AllGrippersCollisionConstraintFunctionType& collision_constraint_fn_;
          const AllGrippersStretchingConstraintFunctionType& stretching_constraint_fn_;
          const AllGrippersMotionSizeConstraintFunctionType& gripper_motion_constraint_fn_;
          const bool fix_step_size_;
          // const sdf_tools::SignedDistanceField enviroment_sdf_;

    };


}

#endif // NOMAD_SOLVERS_H
