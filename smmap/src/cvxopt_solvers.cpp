#include "smmap/cvxopt_solvers.hpp"

using namespace smmap;

PyObject* CVXOptSolvers::solvers_ = nullptr;
PyObject* CVXOptSolvers::lp_ = nullptr;
PyObject* CVXOptSolvers::qp_ = nullptr;

void CVXOptSolvers::Initialize()
{
    Py_Initialize();

    // import cvxopt
    if (import_cvxopt() < 0)
    {
        throw_arc_exception(std::runtime_error, "Error importing cvxopt.solvers");
    }

    // import cvxopt.solvers
    solvers_ = PyImport_ImportModule("cvxopt.solvers");
    if (!solvers_)
    {
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
}

void CVXOptSolvers::Finalize()
{
    if (qp_)
    {
        Py_DECREF(qp_);
    }

    if (lp_)
    {
        Py_DECREF(lp_);
    }

    if (solvers_)
    {
        Py_DECREF(solvers_);
    }

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

    // Setup the arguments to the QP solver - references are stolen?
    PyObject *args = PyTuple_New(5);
    PyTuple_SetItem(args, 0, c_py);
    PyTuple_SetItem(args, 1, G_py);
    PyTuple_SetItem(args, 2, h_py);
    PyTuple_SetItem(args, 3, A_py);
    PyTuple_SetItem(args, 4, b_py);

    // Solve the LP
    Eigen::VectorXd x(num_unknowns);
    PyObject *sol = PyObject_CallObject(lp_, args);
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

    Py_DECREF(args);
    Py_DECREF(sol);

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

    // Setup the arguments to the QP solver - references are stolen?
    PyObject *args = PyTuple_New(6);
    PyTuple_SetItem(args, 0, Q_py);
    PyTuple_SetItem(args, 1, p_py);
    PyTuple_SetItem(args, 2, G_py);
    PyTuple_SetItem(args, 3, h_py);
    PyTuple_SetItem(args, 4, A_py);
    PyTuple_SetItem(args, 5, b_py);

    // Solve the QP
    Eigen::VectorXd x(num_unknowns);
    PyObject *sol = PyObject_CallObject(qp_, args);
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

    Py_DECREF(args);
    Py_DECREF(sol);

    return x;
}
