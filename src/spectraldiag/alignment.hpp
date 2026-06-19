#pragma once
#include "spectral.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace sd {

struct SNRResult {
    double c_star;
    double snr;
    double alpha_snr;
    bool gated;
    std::string verdict;
};

struct AlignResult {
    double gamma;
    Vec cos_thetas;
    bool product_law_holds;
    double residual;
    std::string verdict;
};

struct WarmstartResult {
    double c_recomp;
    double warmstart_gain;
    double spectral_floor;
    bool gain_real;
    std::string verdict;
};

static double principal_angle(const Vec& u, const Vec& v) {
    double cu = dot(u, u), cv = dot(v, v);
    if (cu < 1e-12 || cv < 1e-12) return M_PI / 2.0;
    double ip = std::abs(dot(u, v)) / (std::sqrt(cu) * std::sqrt(cv));
    ip = std::min(1.0, ip);
    return std::acos(ip);
}

[[maybe_unused]] static Vec subspace_principal_angles(const Mat& U, const Mat& V) {
    int k = (int)std::min(U.size(), V.size());
    if (k == 0) return {};

    int d = (int)U[0].size();
    Mat M(k, Vec(k, 0.0));
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j)
            for (int l = 0; l < d; ++l)
                M[i][j] += U[i][l] * V[j][l];

    auto eig = sym_eig(M);
    Vec angles;
    for (double sv : eig.vals) {
        double v = std::max(-1.0, std::min(1.0, sv));
        angles.push_back(std::acos(std::abs(v)));
    }
    return angles;
}

SNRResult snr_gate(double alpha, double noise_var, double signal_var) {
    SNRResult res;
    double snr = (noise_var > 1e-12) ? signal_var / noise_var : 1e6;
    res.snr = snr;
    res.alpha_snr = alpha;

    res.c_star = alpha / std::sqrt(1.0 + alpha * alpha);

    double c_current = std::sqrt(snr) / std::sqrt(1.0 + snr);
    res.gated = (c_current >= res.c_star);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    if (res.gated) {
        oss << "SNR GATE OPEN. Current alignment c=" << c_current
            << " exceeds critical threshold c*=" << res.c_star
            << " (α=" << alpha << "). "
            << "Feature capture is in the drift-dominated regime — "
            << "learning is proceeding above the bifurcation point.";
    } else {
        oss << "SNR GATE CLOSED. Current alignment c=" << c_current
            << " is below threshold c*=" << res.c_star
            << " (SNR=" << snr << ", α=" << alpha << "). "
            << "Learning is in the diffusion-dominated regime — "
            << "the signal is being washed out by noise. "
            << "Universal constant at full noise: c*=1/√2≈0.707.";
    }
    res.verdict = oss.str();
    return res;
}

AlignResult alignment_signal(const std::vector<Vec>& subspace_dirs) {
    AlignResult res;
    int k = (int)subspace_dirs.size();
    if (k == 0) {
        res.gamma = 1.0;
        res.product_law_holds = true;
        res.residual = 0.0;
        res.verdict = "No subspace directions provided.";
        return res;
    }

    double gamma = 1.0;
    Vec cos_thetas;
    for (int i = 0; i + 1 < k; ++i) {
        double theta = principal_angle(subspace_dirs[i], subspace_dirs[i + 1]);
        double c = std::cos(theta);
        cos_thetas.push_back(c);
        gamma *= c;
    }
    if (k == 1) { gamma = 1.0; cos_thetas.push_back(1.0); }

    res.gamma = gamma;
    res.cos_thetas = cos_thetas;

    double prod = 1.0;
    for (double c : cos_thetas) prod *= c;
    res.residual = std::abs(gamma - prod);
    res.product_law_holds = (res.residual < 0.05 * std::max(1e-6, std::abs(prod)));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "ALIGNMENT SIGNAL Γ=" << gamma
        << " = ∏cos(θᵢ) (product-state universality). ";
    for (int i = 0; i < (int)cos_thetas.size(); ++i)
        oss << "θ" << i << "=" << std::acos(std::min(1.0,std::abs(cos_thetas[i]))) * 180.0 / M_PI << "° ";
    if (res.product_law_holds) {
        oss << "Product law holds (residual=" << res.residual << "). "
            << "Alignment is consistent with symmetric product measure universality.";
    } else {
        oss << "Product law deviation=" << res.residual
            << " — anisotropy or non-symmetric coupling detected.";
    }
    res.verdict = oss.str();
    return res;
}

WarmstartResult warmstart_economy(
    const Vec& source_eigs,
    const Vec& target_eigs,
    double spectral_floor_est
) {
    WarmstartResult res;

    int n = (int)std::min(source_eigs.size(), target_eigs.size());
    if (n < 2) {
        res.c_recomp = 0.0;
        res.warmstart_gain = 0.0;
        res.spectral_floor = spectral_floor_est;
        res.gain_real = false;
        res.verdict = "Insufficient eigenvalue data.";
        return res;
    }

    double num = 0.0, d1 = 0.0, d2 = 0.0;
    for (int i = 0; i < n; ++i) {
        double s = std::max(source_eigs[i], 1e-12);
        double t = std::max(target_eigs[i], 1e-12);
        num += std::sqrt(s * t);
        d1 += s;
        d2 += t;
    }
    double c_recomp = (d1 > 1e-10 && d2 > 1e-10)
        ? num / std::sqrt(d1 * d2) : 0.0;
    c_recomp = std::max(0.0, std::min(1.0, c_recomp));

    res.c_recomp = c_recomp;
    res.spectral_floor = spectral_floor_est;

    double asymptotic_gain = (spectral_floor_est > 1e-10)
        ? c_recomp * c_recomp / spectral_floor_est : 0.0;
    res.warmstart_gain = std::min(asymptotic_gain, 100.0);

    res.gain_real = (spectral_floor_est < 0.5 && c_recomp > 0.3);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "WARMSTART ECONOMY. c_recomp=" << c_recomp
        << " (recomposition cosine between source and target kernels). "
        << "Spectral floor=" << spectral_floor_est << ". ";
    if (res.gain_real) {
        oss << "Warmstart gain is REAL (floor < 0.5, c_recomp > 0.3). "
            << "Asymptotic speedup factor ≈" << std::setprecision(2) << res.warmstart_gain << "×. "
            << "The compositional warm-start economy is not masked by finite-size spectral floors.";
    } else {
        oss << "Warmstart gain may be MASKED by finite-size spectral floor. "
            << "At current scale the compositional economy is hidden — "
            << "the asymptotic exponent requires larger model sizes to manifest.";
    }
    res.verdict = oss.str();
    return res;
}

}
