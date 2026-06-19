#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "spectraldiag/api.hpp"
#include <vector>
#include <string>

static sd::Vec list_to_vec(PyObject* obj) {
    sd::Vec v;
    if (obj == nullptr || obj == Py_None) return v;
    PyObject* seq = PySequence_Fast(obj, "expected a sequence");
    if (!seq) { PyErr_Clear(); return v; }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    v.reserve(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        double d = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) { PyErr_Clear(); d = 0.0; }
        v.push_back(d);
    }
    Py_DECREF(seq);
    return v;
}

static std::vector<sd::Vec> list_to_mat(PyObject* obj) {
    std::vector<sd::Vec> m;
    if (obj == nullptr || obj == Py_None) return m;
    PyObject* seq = PySequence_Fast(obj, "expected a sequence");
    if (!seq) { PyErr_Clear(); return m; }
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* row = PySequence_Fast_GET_ITEM(seq, i);
        m.push_back(list_to_vec(row));
    }
    Py_DECREF(seq);
    return m;
}

static PyObject* py_stationarity(PyObject*, PyObject* args) {
    PyObject *eigs_o, *tc_o;
    double s_hint = -1.0;
    if (!PyArg_ParseTuple(args, "OO|d", &eigs_o, &tc_o, &s_hint)) return nullptr;
    sd::Vec eigs = list_to_vec(eigs_o);
    sd::Vec tc   = list_to_vec(tc_o);
    sd::StatResult r = sd::stationarity_verdict(eigs, tc, s_hint);
    return Py_BuildValue(
        "{s:O,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:s,s:s}",
        "stationary", r.stationary ? Py_True : Py_False,
        "r_hat", r.r_hat, "r_std", r.r_std, "beta_pred", r.beta_pred,
        "beta_0", r.beta_0, "d_star", r.d_star, "s_hat", r.s_hat,
        "r2_fit", r.r2_fit, "verdict", r.verdict.c_str(),
        "reason", r.reason.c_str()
    );
}

static PyObject* py_effective_dim(PyObject*, PyObject* args) {
    PyObject *lap_o, *ae_o = nullptr, *ms_o = nullptr;
    if (!PyArg_ParseTuple(args, "O|OO", &lap_o, &ae_o, &ms_o)) return nullptr;
    sd::Vec lap = list_to_vec(lap_o);
    sd::Vec ae  = list_to_vec(ae_o);
    sd::Vec ms  = list_to_vec(ms_o);
    sd::DimResult r = sd::effective_dimension(lap, ae, ms);
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:d,s:O,s:d,s:d,s:s}",
        "d_loc", r.d_loc, "d_star", r.d_star,
        "alpha_sobolev", r.alpha_sobolev, "alpha_comp", r.alpha_comp,
        "compositional", r.compositional ? Py_True : Py_False,
        "compression_gain", r.compression_gain,
        "subspace_gap", r.subspace_gap, "verdict", r.verdict.c_str()
    );
}

static PyObject* py_barrier(PyObject*, PyObject* args) {
    double d_star, d_loc, s, loss, N, D;
    if (!PyArg_ParseTuple(args, "dddddd", &d_star, &d_loc, &s, &loss, &N, &D))
        return nullptr;
    sd::BarrierResult r = sd::barrier_certificate(d_star, d_loc, s, loss, N, D);
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:O,s:s}",
        "beta_0", r.beta_0, "alpha", r.alpha, "d_star", r.d_star,
        "d_loc", r.d_loc, "s", r.s, "n_critical", r.n_critical,
        "budget_gap", r.budget_gap,
        "saturated", r.saturated ? Py_True : Py_False,
        "verdict", r.verdict.c_str()
    );
}

static PyObject* py_snr_gate(PyObject*, PyObject* args) {
    double alpha, noise_var, signal_var;
    if (!PyArg_ParseTuple(args, "ddd", &alpha, &noise_var, &signal_var))
        return nullptr;
    sd::SNRResult r = sd::snr_gate(alpha, noise_var, signal_var);
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:O,s:s}",
        "c_star", r.c_star, "snr", r.snr, "alpha_snr", r.alpha_snr,
        "gated", r.gated ? Py_True : Py_False, "verdict", r.verdict.c_str()
    );
}

static PyObject* py_alignment(PyObject*, PyObject* args) {
    PyObject* dirs_o;
    if (!PyArg_ParseTuple(args, "O", &dirs_o)) return nullptr;
    std::vector<sd::Vec> dirs = list_to_mat(dirs_o);
    sd::AlignResult r = sd::alignment_signal(dirs);
    PyObject* cos_list = PyList_New(r.cos_thetas.size());
    for (size_t i = 0; i < r.cos_thetas.size(); ++i)
        PyList_SET_ITEM(cos_list, i, PyFloat_FromDouble(r.cos_thetas[i]));
    return Py_BuildValue(
        "{s:d,s:N,s:O,s:d,s:s}",
        "gamma", r.gamma, "cos_thetas", cos_list,
        "product_law_holds", r.product_law_holds ? Py_True : Py_False,
        "residual", r.residual, "verdict", r.verdict.c_str()
    );
}

static PyObject* py_warmstart(PyObject*, PyObject* args) {
    PyObject *src_o, *tgt_o;
    double floor = 0.1;
    if (!PyArg_ParseTuple(args, "OO|d", &src_o, &tgt_o, &floor)) return nullptr;
    sd::Vec src = list_to_vec(src_o);
    sd::Vec tgt = list_to_vec(tgt_o);
    sd::WarmstartResult r = sd::warmstart_economy(src, tgt, floor);
    return Py_BuildValue(
        "{s:d,s:d,s:d,s:O,s:s}",
        "c_recomp", r.c_recomp, "warmstart_gain", r.warmstart_gain,
        "spectral_floor", r.spectral_floor,
        "gain_real", r.gain_real ? Py_True : Py_False,
        "verdict", r.verdict.c_str()
    );
}

static PyMethodDef methods[] = {
    {"stationarity_verdict", py_stationarity, METH_VARARGS, "Stationarity verdict"},
    {"effective_dimension",  py_effective_dim, METH_VARARGS, "Effective dimension"},
    {"barrier_certificate",  py_barrier,       METH_VARARGS, "Barrier certificate"},
    {"snr_gate",             py_snr_gate,      METH_VARARGS, "SNR gate"},
    {"alignment_signal",     py_alignment,     METH_VARARGS, "Alignment signal"},
    {"warmstart_economy",    py_warmstart,     METH_VARARGS, "Warmstart economy"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_core",
    "spectraldiag C++ core", -1, methods,
    nullptr, nullptr, nullptr, nullptr
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&moduledef);
}
