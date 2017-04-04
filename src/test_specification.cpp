#include <arc_utilities/arc_exceptions.hpp>
#include <arc_utilities/log.hpp>
#include <deformable_manipulation_msgs/messages.h>

#include "smmap/test_specification.h"
//#include "smmap/task_specification_implementions.hpp"

using namespace smmap;

////////////////////////////////////////////////////////////////////////////////
// Static helper functions - could be private given how they are
// used but making public as they are static - probably should be moved out of class
////////////////////////////////////////////////////////////////////////////////


// This function should calculate error at each step
// error = norm(p_Delta_real-p_Delta_model)
// p_Delta_real = p_current-p_last
// p_Delta_model = J*q_Delta_last; The result could be obtained from calculateDesiredDirection
// Also stored as (Last)first_step_desired_motion_;
// It is called by CalculateError_impl in CalculateError, minimum_threshold depend on test
double CalculateErrorWithTheshold(
        const ObjectPointSet& real_delta_p,
        ObjectDeltaAndWeight& model_delta_p,
        const double minimum_threshold)
{
    Eigen::VectorXd error(real_delta_p);
    const Eigen::VectorXd& model_delta_p_value = model_delta_p.delta;

    // for every cover point, find the nearest deformable object point
    #pragma omp parallel for
    for (ssize_t real_ind = 0; real_ind < real_delta_p.cols(); ++real_ind)
    {
        const Eigen::Vector3d& real_point = real_delta_p.col(real_ind);
        const Eigen::Vector3d& model_point = model_delta_p_value.segment<3>(real_ind*3);
        const double point_error = (real_point-model_point).squaredNorm();

        if (std::sqrt(point_error) >= minimum_threshold)
        { error(target_ind) = std::sqrt(point_error); }
        else{ error(target_ind) = 0; }
    }

    return error.sum();
}



