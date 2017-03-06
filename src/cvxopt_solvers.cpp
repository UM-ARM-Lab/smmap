#include "smmap/cvxopt_solvers.h"
#include <iostream>
#include <arc_utilities/arc_exceptions.hpp>

using namespace smmap;

PyObject* CVXOptSolvers::solvers_ = nullptr;
PyObject* CVXOptSolvers::lp_ = nullptr;
PyObject* CVXOptSolvers::qp_ = nullptr;
PyObject* CVXOptSolvers::qcqp_module_ = nullptr;
PyObject* CVXOptSolvers::qcqp_ = nullptr;

void CVXOptSolvers::Initialize()
{
    Py_Initialize();
    PyObject* path = PySys_GetObject((char *)"path");
    PyList_Append(path, PyString_FromString("/home/dmcconachie/Dropbox/catkin_ws/src/smmap/smmap/scripts"));

    // import cvxopt
    if (import_cvxopt() < 0)
    {
        throw_arc_exception(std::runtime_error, "Error importing cvxopt.solvers");
    }

    // import cvxopt.solvers
    solvers_ = PyImport_ImportModule("cvxopt.solvers");
    if (!solvers_)
    {
        Finalize();
        throw_arc_exception(std::runtime_error, "Error importing cvxopt.solvers");
    }

    lp_ = PyObject_GetAttrString(solvers_, "lp");
    if (!lp_)
    {
        Finalize();
        throw_arc_exception(std::runtime_error, "Error referencing cvxopt.solvers.lp");
    }

    qp_ = PyObject_GetAttrString(solvers_, "qp");
    if (!qp_)
    {
        Finalize();
        throw_arc_exception(std::runtime_error, "Error referencing cvxopt.solvers.qp");
    }

    qcqp_module_ = PyImport_ImportModule("qcqp");
    if (!qcqp_module_)
    {
        Finalize();
        throw_arc_exception(std::runtime_error, "Error importing qcqp module");
    }

    qcqp_ = PyObject_GetAttrString(qcqp_module_, "qcqp");
    if (!qcqp_)
    {
        Finalize();
        throw_arc_exception(std::runtime_error, "Error referencing qcqp");
    }
}

void CVXOptSolvers::Finalize()
{
    Py_XDECREF(qp_);
    Py_XDECREF(lp_);
    Py_XDECREF(solvers_);
    Py_XDECREF(qcqp_);
    Py_XDECREF(qcqp_module_);
    Py_Finalize();
}

Eigen::VectorXd CVXOptSolvers::lp(
        const Eigen::VectorXd& c,
        const Eigen::MatrixXd& G,
        const Eigen::VectorXd& h,
        const Eigen::MatrixXd& A,
        const Eigen::VectorXd& b)
{
    // Confirm that our data is valid
    const ssize_t num_unknowns = c.rows();
    assert(num_unknowns == G.rows());
    assert(num_unknowns == G.cols());
    assert(num_unknowns == h.rows());
    assert(num_unknowns == A.rows());
    assert(num_unknowns == A.cols());
    assert(num_unknowns == b.rows());

    // Create python objecst to do computation on
    PyObject* c_py = (PyObject*)Matrix_New(num_unknowns, 1, DOUBLE);
    PyObject* G_py = (PyObject*)Matrix_New(num_unknowns, num_unknowns, DOUBLE);
    PyObject* h_py = (PyObject*)Matrix_New(num_unknowns, 1, DOUBLE);
    PyObject* A_py = (PyObject*)Matrix_New(num_unknowns, num_unknowns, DOUBLE);
    PyObject* b_py = (PyObject*)Matrix_New(num_unknowns, 1, DOUBLE);

    // Copy the data over into the python objects
    memcpy(MAT_BUFD(c_py), c.data(), sizeof(double) * num_unknowns);
    memcpy(MAT_BUFD(G_py), G.data(), sizeof(double) * num_unknowns * num_unknowns);
    memcpy(MAT_BUFD(h_py), h.data(), sizeof(double) * num_unknowns);
    memcpy(MAT_BUFD(A_py), A.data(), sizeof(double) * num_unknowns * num_unknowns);
    memcpy(MAT_BUFD(b_py), b.data(), sizeof(double) * num_unknowns);

    // Solve the LP
    Eigen::VectorXd x(num_unknowns);
    PyObject *sol = PyObject_CallFunctionObjArgs(lp_, c_py, G_py, h_py, A_py, b_py, NULL);
    if (!sol)
    {
        PyErr_Print();
        x *= std::numeric_limits<double>::quiet_NaN();
    }
    else
    {
        PyObject* x_py = PyDict_GetItemString(sol, "x");
        memcpy(x.data(), MAT_BUFD(x_py), sizeof(double) * num_unknowns);
    }

    Py_CLEAR(c_py);
    Py_CLEAR(G_py);
    Py_CLEAR(h_py);
    Py_CLEAR(A_py);
    Py_CLEAR(b_py);
    Py_CLEAR(sol);

    return x;
}

Eigen::VectorXd CVXOptSolvers::qp(
        const Eigen::MatrixXd& Q,
        const Eigen::VectorXd& p,
        const Eigen::MatrixXd& G,
        const Eigen::VectorXd& h,
        const Eigen::MatrixXd& A,
        const Eigen::VectorXd& b)
{
    // Confirm that our data is valid
    const ssize_t num_unknowns = Q.cols();
    assert(num_unknowns == Q.rows());
    assert(num_unknowns == p.rows());
    assert(num_unknowns == G.rows());
    assert(num_unknowns == G.cols());
    assert(num_unknowns == h.rows());
    assert(num_unknowns == A.rows());
    assert(num_unknowns == A.cols());
    assert(num_unknowns == b.rows());

    // Create python objecst to do computation on
    PyObject* Q_py = (PyObject*)Matrix_New(num_unknowns, num_unknowns, DOUBLE);
    PyObject* p_py = (PyObject*)Matrix_New(num_unknowns, 1, DOUBLE);
    PyObject* G_py = (PyObject*)Matrix_New(num_unknowns, num_unknowns, DOUBLE);
    PyObject* h_py = (PyObject*)Matrix_New(num_unknowns, 1, DOUBLE);
    PyObject* A_py = (PyObject*)Matrix_New(num_unknowns, num_unknowns, DOUBLE);
    PyObject* b_py = (PyObject*)Matrix_New(num_unknowns, 1, DOUBLE);

    // Copy the data over into the python objects
    memcpy(MAT_BUFD(Q_py), Q.data(), sizeof(double) * num_unknowns * num_unknowns);
    memcpy(MAT_BUFD(p_py), p.data(), sizeof(double) * num_unknowns);
    memcpy(MAT_BUFD(G_py), G.data(), sizeof(double) * num_unknowns * num_unknowns);
    memcpy(MAT_BUFD(h_py), h.data(), sizeof(double) * num_unknowns);
    memcpy(MAT_BUFD(A_py), A.data(), sizeof(double) * num_unknowns * num_unknowns);
    memcpy(MAT_BUFD(b_py), b.data(), sizeof(double) * num_unknowns);

    // Solve the QP
    Eigen::VectorXd x(num_unknowns);
    PyObject *sol = PyObject_CallFunctionObjArgs(qp_, Q_py, p_py, G_py, h_py, A_py, b_py, NULL);
    if (!sol)
    {
        PyErr_Print();
        x *= std::numeric_limits<double>::quiet_NaN();
    }
    else
    {
        PyObject* x_py = PyDict_GetItemString(sol, "x");
        memcpy(x.data(), MAT_BUFD(x_py), sizeof(double) * num_unknowns);
    }

    Py_CLEAR(Q_py);
    Py_CLEAR(p_py);
    Py_CLEAR(G_py);
    Py_CLEAR(h_py);
    Py_CLEAR(A_py);
    Py_CLEAR(b_py);
    Py_CLEAR(sol);

    return x;
}

Eigen::VectorXd CVXOptSolvers::qcqp_jacobian_least_squares(
        const Eigen::MatrixXd& J,
        const Eigen::VectorXd& W,
        const Eigen::VectorXd& pdot,
        const double& max_result_norm)
{
    const ssize_t num_targets = J.rows();
    const ssize_t num_unknowns = J.cols();
    assert(num_targets == W.rows());
    assert(num_targets == pdot.rows());

    const Eigen::VectorXd W_sqrt = W.array().sqrt();
    const Eigen::MatrixXd A0 = W_sqrt.asDiagonal() * J;
    const Eigen::VectorXd b0 = W_sqrt.asDiagonal() * (-pdot);

    PyObject* A0_py = (PyObject*)Matrix_New(num_targets, num_unknowns, DOUBLE);
    PyObject* b0_py = (PyObject*)Matrix_New(num_targets, 1, DOUBLE);
    PyObject* Ident_py = (PyObject*)Matrix_New(num_unknowns, num_unknowns, DOUBLE);

    memcpy(MAT_BUFD(A0_py), A0.data(), sizeof(double) * num_targets * num_unknowns);
    memcpy(MAT_BUFD(b0_py), b0.data(), sizeof(double) * num_targets);
    for (ssize_t row_ind = 0; row_ind < num_unknowns; row_ind++)
    {
        for (ssize_t col_ind = 0; col_ind < num_unknowns; col_ind++)
        {
            if (row_ind != col_ind)
            {
                MAT_BUFD(Ident_py)[row_ind * num_unknowns + col_ind] = 0.0;
            }
            else
            {
                MAT_BUFD(Ident_py)[row_ind * num_unknowns + col_ind] = 1.0;
            }
        }
    }

    // Create the first dictionary, with the minimization function
    PyObject* A0_arg = PyDict_New();
    PyDict_SetItemString(A0_arg, "A0", A0_py);
    PyDict_SetItemString(A0_arg, "b0", b0_py);
    PyDict_SetItemString(A0_arg, "c0", Py_None);
    PyDict_SetItemString(A0_arg, "d0", Py_None);
    // Force A0_arg to steal the references
    Py_CLEAR(A0_py);
    Py_CLEAR(b0_py);

    // Craete the second dictionary, with the contraint function list
    PyObject* A_list_py = Py_BuildValue("[O]", Ident_py);
    Py_CLEAR(Ident_py);
    PyObject* b_list_py = Py_BuildValue("[O]", Py_None);
    PyObject* c_list_py = Py_BuildValue("[O]", Py_None);
    PyObject* d_list_py = Py_BuildValue("[d]", max_result_norm * max_result_norm);
    PyObject* G0_arg = PyDict_New();
    PyDict_SetItemString(G0_arg, "A", A_list_py);
    PyDict_SetItemString(G0_arg, "b", b_list_py);
    PyDict_SetItemString(G0_arg, "c", c_list_py);
    PyDict_SetItemString(G0_arg, "d", d_list_py);
    // Force G0_arg to steal the references
    Py_CLEAR(A_list_py);
    Py_CLEAR(b_list_py);
    Py_CLEAR(c_list_py);
    Py_CLEAR(d_list_py);

    // Solve the QCQP
    Eigen::VectorXd qdot(num_unknowns);
    PyObject *sol = PyObject_CallFunctionObjArgs(qcqp_, A0_arg, G0_arg, NULL);
    if (!sol)
    {
        PyErr_Print();
        qdot *= std::numeric_limits<double>::quiet_NaN();
    }
    else
    {
        PyObject* qdot_py = PyDict_GetItemString(sol, "QCQPx");
        memcpy(qdot.data(), MAT_BUFD(qdot_py), sizeof(double) * num_unknowns);
    }

    Py_CLEAR(sol);
    Py_CLEAR(A0_arg);
    Py_CLEAR(G0_arg);

    return qdot;
}
