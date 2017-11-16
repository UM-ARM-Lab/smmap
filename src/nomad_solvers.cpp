#include "smmap/nomad_solvers.h"

#include <iostream>
//#include <mutex>
//#include <Eigen/Eigenvalues>

using namespace NOMAD;
using namespace smmap;


GripperMotionNomadEvaluator::GripperMotionNomadEvaluator(
        const NOMAD::Parameters & p,
        const ssize_t num_grippers,
        const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& eval_error_cost_fn,
        const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& collision_constraint_fn,
        const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& stretching_constraint_fn,
        const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& gripper_motion_constraint_fn,
        const bool fix_step_size)
    : NOMAD::Evaluator(p)
    , num_grippers_(num_grippers)
    , eval_error_cost_fn_(eval_error_cost_fn)
    , collision_constraint_fn_(collision_constraint_fn)
    , stretching_constraint_fn_(stretching_constraint_fn)
    , gripper_motion_constraint_fn_(gripper_motion_constraint_fn)
    , fix_step_size_(fix_step_size)
{}

AllGrippersSinglePoseDelta GripperMotionNomadEvaluator::evalPointToGripperPoseDelta(const NOMAD::Eval_Point& x)
{
    if (&x == nullptr)
    {
        return AllGrippersSinglePoseDelta(num_grippers_, kinematics::Vector6d::Zero());
    }

    const int single_gripper_dimension = 6;
    if (x.size() != num_grippers_ * single_gripper_dimension)
    {
        assert(false && "grippers data and eval_point x have different size");
    }

    AllGrippersSinglePoseDelta grippers_motion(num_grippers_);
    for (int gripper_ind = 0; gripper_ind < num_grippers_; gripper_ind ++)
    {
        kinematics::Vector6d& single_gripper_delta = grippers_motion[gripper_ind];

        single_gripper_delta(0) = x[gripper_ind * single_gripper_dimension].value();
        single_gripper_delta(1) = x[gripper_ind * single_gripper_dimension + 1].value();
        single_gripper_delta(2) = x[gripper_ind * single_gripper_dimension + 2].value();

        single_gripper_delta(3) = x[gripper_ind * single_gripper_dimension + 3].value();
        single_gripper_delta(4) = x[gripper_ind * single_gripper_dimension + 4].value();
        single_gripper_delta(5) = x[gripper_ind * single_gripper_dimension + 5].value();
    }

    return grippers_motion;
}

bool GripperMotionNomadEvaluator::eval_x(
        NOMAD::Eval_Point& x,
        const NOMAD::Double& h_max,
        bool& count_eval)
{
    UNUSED(h_max); // TODO: Why don't we use h_max?

    // count a black-box evaluation
    count_eval = true;

    // Convert NOMAD points into
    const AllGrippersSinglePoseDelta test_grippers_motions = evalPointToGripperPoseDelta(x);

    NOMAD::Double c1_error_cost = eval_error_cost_fn_(test_grippers_motions);
    NOMAD::Double c2_collision_constraint = collision_constraint_fn_(test_grippers_motions);
    NOMAD::Double c3_stretching_constraint = stretching_constraint_fn_(test_grippers_motions);
    NOMAD::Double c4_gripper_motion_constraint = gripper_motion_constraint_fn_(test_grippers_motions);

    // objective value
    x.set_bb_output(0, c1_error_cost);

    // constraints
    x.set_bb_output(1, c2_collision_constraint);
    x.set_bb_output(2, c3_stretching_constraint);
    x.set_bb_output(3, c4_gripper_motion_constraint);

    if (fix_step_size_)
    {
        if (x.get_bb_outputs().size() < 5)
        {
            assert(false && "size of x not match due to the fix step size constraint");
        }
        x.set_bb_output(4, -c4_gripper_motion_constraint);
    }

    return count_eval;
}






