#include <smmap/gurobi_solvers.h>

#include <iostream>
#include <omp.h>

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    #pragma omp parallel for
    for (int i = 0; i < 30; i++)
    {
        std::cout << smmap::minSquaredNorm(Eigen::MatrixXd::Ones(10, 5), Eigen::VectorXd::Ones(10), 0.1, Eigen::VectorXd::Ones(10)).transpose() << std::endl;
    }

    return 0;
}
