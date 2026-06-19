from __future__ import annotations

from . import _core

__version__ = "0.1.0"
__all__ = [
    "stationarity_verdict", "effective_dimension", "barrier_certificate",
    "snr_gate", "alignment_signal", "warmstart_economy", "SpectralDiagnostics",
]


def _to_list(x):
    if x is None:
        return []
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.flatten().tolist()
    except ImportError:
        pass
    if hasattr(x, "tolist"):
        return x.tolist()
    return [float(v) for v in x]


def _to_mat(x):
    if x is None:
        return []
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                return [x.tolist()]
            return x.tolist()
    except ImportError:
        pass
    if isinstance(x, (list, tuple)) and len(x) > 0:
        if isinstance(x[0], (list, tuple)):
            return [[float(v) for v in row] for row in x]
        return [[float(v) for v in x]]
    return []


class _Result:
    """Attribute-access wrapper around a result dict from the C++ core."""
    __slots__ = ("_d",)

    def __init__(self, d):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError:
            raise AttributeError(name)

    def __repr__(self):
        keys = ", ".join(f"{k}={v!r}" for k, v in self._d.items()
                         if not isinstance(v, str) or len(v) < 40)
        return f"<{type(self).__name__} {keys}>"

    def to_dict(self):
        return dict(self._d)


class StatResult(_Result): pass
class DimResult(_Result): pass
class BarrierResult(_Result): pass
class SNRResult(_Result): pass
class AlignResult(_Result): pass
class WarmstartResult(_Result): pass


def stationarity_verdict(ntk_eigs, target_coeffs=None, s: float = -1.0) -> StatResult:
    return StatResult(_core.stationarity_verdict(
        _to_list(ntk_eigs), _to_list(target_coeffs), float(s)))


def effective_dimension(laplacian_eigs, approx_errors=None, model_sizes=None) -> DimResult:
    return DimResult(_core.effective_dimension(
        _to_list(laplacian_eigs), _to_list(approx_errors), _to_list(model_sizes)))


def barrier_certificate(d_star: float, d_loc: float, s: float,
                        current_loss: float, current_N: float,
                        current_D: float) -> BarrierResult:
    return BarrierResult(_core.barrier_certificate(
        float(d_star), float(d_loc), float(s),
        float(current_loss), float(current_N), float(current_D)))


def snr_gate(alpha: float, noise_var: float, signal_var: float) -> SNRResult:
    return SNRResult(_core.snr_gate(float(alpha), float(noise_var), float(signal_var)))


def alignment_signal(subspace_dirs) -> AlignResult:
    return AlignResult(_core.alignment_signal(_to_mat(subspace_dirs)))


def warmstart_economy(source_eigs, target_eigs,
                      spectral_floor: float = 0.1) -> WarmstartResult:
    return WarmstartResult(_core.warmstart_economy(
        _to_list(source_eigs), _to_list(target_eigs), float(spectral_floor)))


class SpectralDiagnostics:
    def __init__(self, d_star: float = 2.0, s: float = 1.25, verbose: bool = True):
        self.d_star = d_star
        self.s = s
        self.verbose = verbose
        self._ntk_eigs = []
        self._target_coeffs = []
        self._lap_eigs = []

    def fit_ntk(self, eigs, target_coeffs=None):
        self._ntk_eigs = _to_list(eigs)
        if target_coeffs is not None:
            self._target_coeffs = _to_list(target_coeffs)
        return self

    def fit_data(self, lap_eigs):
        self._lap_eigs = _to_list(lap_eigs)
        return self

    def verdict(self, s: float = -1.0) -> StatResult:
        if not self._ntk_eigs:
            raise ValueError("Call fit_ntk() first.")
        res = stationarity_verdict(self._ntk_eigs, self._target_coeffs,
                                   s if s > 0 else self.s)
        if self.verbose:
            print(res.reason)
        return res

    def dim(self, approx_errors=None, model_sizes=None) -> DimResult:
        if not self._lap_eigs:
            raise ValueError("Call fit_data() first.")
        res = effective_dimension(self._lap_eigs, approx_errors, model_sizes)
        if self.verbose:
            print(res.verdict)
        return res

    def barrier(self, d_loc: float, current_loss: float,
                current_N: float, current_D: float) -> BarrierResult:
        res = barrier_certificate(self.d_star, d_loc, self.s,
                                  current_loss, current_N, current_D)
        if self.verbose:
            print(res.verdict)
        return res
