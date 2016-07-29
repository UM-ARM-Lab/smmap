#ifndef CVXOPT_SOLVERS_HPP
#define CVXOPT_SOLVERS_HPP

#include <arc_utilities/arc_exceptions.hpp>
#include <Eigen/Dense>
#include <cvxopt/cvxopt.h>

namespace smmap
{
    class CVXOptSolvers
    {
        public:
            static void Initialize();
            static void Finalize();

            /**
             * @brief lp: See Section 4.3 of Boyd and Vandenderghe; page 146
             */
            Eigen::VectorXd lp(
                    const Eigen::VectorXd& c,
                    const Eigen::MatrixXd& G,
                    const Eigen::VectorXd& h,
                    const Eigen::MatrixXd& A,
                    const Eigen::VectorXd& b);

            /**
             * @brief qp: See Section 4.4 of Boyd and Vandenderghe; page 152
             */
            Eigen::VectorXd qp(
                    const Eigen::MatrixXd& Q,
                    const Eigen::VectorXd& p,
                    const Eigen::MatrixXd& G,
                    const Eigen::VectorXd& h,
                    const Eigen::MatrixXd& A,
                    const Eigen::VectorXd& b);

        private:
            CVXOptSolvers();

            static PyObject* solvers_;
            static PyObject* lp_;
            static PyObject* qp_;
    };
}

#endif // CVXOPT_SOLVERS_HPP
