#pragma once
#include "spectral.hpp"
#include "diagnostics.hpp"
#include "alignment.hpp"

namespace sd {

struct FullReport {
    StatResult stat;
    DimResult dim;
    BarrierResult barrier;
    SNRResult snr;
    AlignResult align;
    WarmstartResult warmstart;
};

inline FullReport full_diagnosis(
    const Vec& ntk_eigs,
    const Vec& target_coeffs,
    const Vec& laplacian_eigs,
    const Vec& approx_errors,
    const Vec& model_sizes,
    double d_star,
    double d_loc,
    double s,
    double current_loss,
    double current_N,
    double current_D,
    double signal_var,
    double noise_var,
    double alpha_snr,
    const std::vector<Vec>& subspace_dirs,
    const Vec& source_eigs_ws,
    const Vec& target_eigs_ws,
    double spectral_floor
) {
    FullReport rep;
    rep.stat = stationarity_verdict(ntk_eigs, target_coeffs, s);
    rep.dim = effective_dimension(laplacian_eigs, approx_errors, model_sizes);
    rep.barrier = barrier_certificate(d_star, d_loc, s, current_loss, current_N, current_D);
    rep.snr = snr_gate(alpha_snr, noise_var, signal_var);
    rep.align = alignment_signal(subspace_dirs);
    rep.warmstart = warmstart_economy(source_eigs_ws, target_eigs_ws, spectral_floor);
    return rep;
}

}
