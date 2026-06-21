import sys
import os
import math
import pytest

# Import through the public API (returns attribute-access result objects)
import spectraldiag as sd
from spectraldiag import (
    stationarity_verdict,
    effective_dimension,
    barrier_certificate,
    snr_gate,
    alignment_signal,
    warmstart_economy,
)


def sobolev_eigs(n, b):
    return [max((i + 1) ** (-b), 1e-12) for i in range(n)]


def source_coeffs(n, b, r):
    return [max((i + 1) ** (-(2 * r * b + 1) / 2), 1e-12) for i in range(n)]


class TestStationarity:
    def test_stationary_case(self):
        n, b = 80, 1.25
        res = stationarity_verdict(sobolev_eigs(n, b), source_coeffs(n, b, 0.5),
                                   s=1.25, d_star=2.0)
        assert res.stationary, f"Expected stationary, r_hat={res.r_hat}"
        assert 0.2 < res.r_hat < 0.8
        assert 0.0 < res.beta_0 < 1.0
        assert res.verdict == "stationary"

    def test_transient_case(self):
        n, b = 80, 1.25
        res = stationarity_verdict(sobolev_eigs(n, b), source_coeffs(n, b, 1.5),
                                   s=1.25, d_star=2.0)
        assert not res.stationary, f"Expected transient, r_hat={res.r_hat}"
        assert res.verdict == "transient"

    def test_too_few_eigs(self):
        res = stationarity_verdict([1.0, 0.5, 0.25], [0.1, 0.05, 0.02])
        assert res.verdict == "insufficient_data"

    def test_beta_0_requires_d_star(self):
        # Without d_star, beta_0 must be undefined (-1), not fabricated.
        res = stationarity_verdict(sobolev_eigs(80, 1.0), source_coeffs(80, 1.0, 0.5))
        assert res.beta_0 == -1.0
        assert res.d_star == -1.0

    def test_beta_0_formula(self):
        s, d_star = 1.0, 2.0
        b = 2 * s / d_star
        res = stationarity_verdict(sobolev_eigs(80, b), source_coeffs(80, b, 0.5),
                                   s=s, d_star=d_star)
        assert abs(res.beta_0 - 2 * s / (2 * s + d_star)) < 1e-6

    def test_s_inferred_from_b_and_dstar(self):
        # When s not given but d_star is, s = b * d_star / 2 (Weyl law).
        n, b, d_star = 80, 1.0, 4.0
        res = stationarity_verdict(sobolev_eigs(n, b), source_coeffs(n, b, 0.5),
                                   d_star=d_star)
        # b measured ~1.0, d_star=4 => s ~ 2.0
        assert abs(res.s_hat - b * d_star / 2.0) < 0.3

    def test_reason_nonempty(self):
        res = stationarity_verdict(sobolev_eigs(80, 1.25), source_coeffs(80, 1.25, 0.5))
        assert len(res.reason) > 20


class TestEffectiveDimension:
    def test_d_star_positive(self):
        lap_eigs = [(i + 1) ** (2.0 / 8.0) for i in range(50)]
        sizes = [10 ** (1 + 0.2 * i) for i in range(10)]
        errors = [s ** (-2 * 1.25 / 2.0) for s in sizes]
        res = effective_dimension(lap_eigs, errors, sizes)
        assert res.d_star > 1.0
        assert res.alpha_sobolev > 0

    def test_no_compositional_structure(self):
        lap_eigs = [(i + 1) ** (2.0 / 2.0) for i in range(50)]
        sizes = [10 ** (1 + 0.3 * i) for i in range(10)]
        errors = [s ** (-1.25) for s in sizes]
        res = effective_dimension(lap_eigs, errors, sizes)
        assert res.d_star > 0
        assert res.d_loc > 0

    def test_empty_model_inputs(self):
        lap_eigs = [1.0, 2.0, 3.0, 4.0, 5.0]
        res = effective_dimension(lap_eigs, [], [])
        assert res.d_star > 0
        assert res.d_loc > 0

    def test_compression_gain_gte_1(self):
        lap_eigs = [(i + 1) ** (2.0 / 8.0) for i in range(50)]
        sizes = [10 ** (1 + 0.2 * i) for i in range(10)]
        errors = [s ** (-2 * 1.25 / 1.0) for s in sizes]
        res = effective_dimension(lap_eigs, errors, sizes)
        assert res.compression_gain >= 1.0


class TestBarrierCertificate:
    def test_basic_barrier(self):
        res = barrier_certificate(2.0, 2.0, 1.25, 0.5, 1e6, 1e9)
        assert abs(res.beta_0 - 2 * 1.25 / (2 * 1.25 + 2.0)) < 1e-6
        assert res.d_star == 2.0
        assert res.s == 1.25

    def test_compositional_fields(self):
        res_full = barrier_certificate(8.0, 8.0, 1.25, 0.5, 1e6, 1e9)
        res_comp = barrier_certificate(8.0, 2.0, 1.25, 0.5, 1e6, 1e9)
        assert res_comp.beta_0 == res_full.beta_0
        assert res_comp.d_loc < res_comp.d_star

    def test_saturated_flag(self):
        assert barrier_certificate(2.0, 2.0, 1.25, 0.5, 1e3, 1e12).saturated
        assert not barrier_certificate(2.0, 2.0, 1.25, 0.5, 1e6, 1e3).saturated

    def test_verdict_nonempty(self):
        assert len(barrier_certificate(2.0, 1.0, 1.25, 0.3, 1e6, 1e9).verdict) > 10

    def test_alpha_exponent(self):
        res = barrier_certificate(2.0, 1.0, 1.25, 0.5, 1e6, 1e9)
        assert abs(res.alpha - 2 * 1.25 / 1.0) < 1e-6


class TestSNRGate:
    def test_gate_open(self):
        res = snr_gate(alpha=1.0, noise_var=0.1, signal_var=10.0)
        assert res.gated
        assert abs(res.c_star - 1.0 / math.sqrt(2)) < 0.01

    def test_gate_closed(self):
        assert not snr_gate(alpha=1.0, noise_var=10.0, signal_var=0.1).gated

    def test_universal_constant(self):
        assert abs(snr_gate(1.0, 1.0, 1.0).c_star - 1.0 / math.sqrt(2)) < 1e-6

    def test_alpha_scaling(self):
        for alpha in [0.5, 1.0, 2.0]:
            res = snr_gate(alpha, 1.0, 1.0)
            assert abs(res.c_star - alpha / math.sqrt(1 + alpha ** 2)) < 1e-6


class TestAlignmentSignal:
    def test_single_dir(self):
        assert abs(alignment_signal([[1.0, 0.0, 0.0]]).gamma - 1.0) < 0.05

    def test_orthogonal_dirs(self):
        assert abs(alignment_signal([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).gamma) < 0.1

    def test_parallel_dirs(self):
        assert alignment_signal([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).gamma > 0.9

    def test_cos_thetas_count(self):
        t1, t2 = 0.3, 0.4
        dirs = [[math.cos(t1), math.sin(t1), 0.0],
                [math.cos(t2), math.sin(t2), 0.0]]
        assert len(alignment_signal(dirs).cos_thetas) == 1

    def test_empty(self):
        assert abs(alignment_signal([]).gamma - 1.0) < 0.05


class TestWarmstart:
    def test_high_overlap(self):
        eigs = [1.0, 0.5, 0.25, 0.1]
        assert warmstart_economy(eigs, eigs, spectral_floor=0.05).c_recomp > 0.9

    def test_low_overlap(self):
        res = warmstart_economy([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], 0.1)
        assert res.c_recomp < 0.5

    def test_verdict_nonempty(self):
        eigs = [1.0, 0.5, 0.25, 0.1]
        assert len(warmstart_economy(eigs, eigs, 0.05).verdict) > 10

    def test_floor_effect(self):
        eigs = [1.0, 0.5, 0.25]
        sf = warmstart_economy(eigs, eigs, spectral_floor=0.01)
        lf = warmstart_economy(eigs, eigs, spectral_floor=0.9)
        assert not lf.gain_real or sf.gain_real


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
