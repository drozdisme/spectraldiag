#pragma once
#include "spectral.hpp"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>

namespace sd {

struct StatResult {
    bool stationary;
    double r_hat;
    double r_std;
    double beta_pred;
    double beta_0;
    double d_star;
    double s_hat;
    double r2_fit;
    std::string verdict;
    std::string reason;
};

struct DimResult {
    double d_loc;
    double d_star;
    double alpha_sobolev;
    double alpha_comp;
    bool compositional;
    double compression_gain;
    double subspace_gap;
    std::string verdict;
};

struct BarrierResult {
    double beta_0;
    double alpha;
    double d_star;
    double d_loc;
    double s;
    double n_critical;
    double budget_gap;
    bool saturated;
    std::string verdict;
};

struct RFit {
    double r;
    double r_std;
    double b;
    double r2_b;
};

static RFit fit_source_exp(const Vec& eig_vals, const Vec& target_proj) {
    int n = (int)std::min(eig_vals.size(), target_proj.size());
    RFit bad = {0.5, 0.3, 1.0, 0.0};
    if (n < 6) return bad;

    int lo = std::max(1, n / 10);
    int hi = std::max(lo + 3, 7 * n / 10);
    hi = std::min(hi, n);

    Vec ks, lv, la2;
    for (int k = lo; k < hi; ++k) {
        double ev = eig_vals[k];
        double tc = target_proj[k];
        if (ev < 1e-20 || std::abs(tc) < 1e-30) continue;
        ks.push_back((double)(k + 1));
        lv.push_back(std::log(ev));
        la2.push_back(2.0 * std::log(std::abs(tc)));
    }
    int m = (int)ks.size();
    if (m < 3) return bad;

    Vec lk(m);
    for (int i = 0; i < m; ++i) lk[i] = std::log(ks[i]);

    double mk = std::accumulate(lk.begin(), lk.end(), 0.0) / m;
    double mlv = std::accumulate(lv.begin(), lv.end(), 0.0) / m;
    double mla = std::accumulate(la2.begin(), la2.end(), 0.0) / m;

    double num_b = 0.0, den = 0.0, num_a = 0.0;
    for (int i = 0; i < m; ++i) {
        double dk = lk[i] - mk;
        num_b += dk * (lv[i] - mlv);
        num_a += dk * (la2[i] - mla);
        den   += dk * dk;
    }
    if (den < 1e-12) return bad;

    double slope_b = num_b / den;
    double slope_a = num_a / den;
    double b = -slope_b;

    double ss_res = 0.0, ss_tot = 0.0, intercept_b = mlv - slope_b * mk;
    for (int i = 0; i < m; ++i) {
        double pred = slope_b * lk[i] + intercept_b;
        ss_res += (lv[i] - pred) * (lv[i] - pred);
        ss_tot += (lv[i] - mlv) * (lv[i] - mlv);
    }
    double r2 = (ss_tot > 1e-12) ? 1.0 - ss_res / ss_tot : 0.0;

    double a_exp = -slope_a;
    double r = (b > 1e-4) ? (a_exp - 1.0) / (2.0 * b) : 0.5;
    r = std::max(-1.0, std::min(r, 3.0));

    double intercept_a = mla - slope_a * mk;
    double sigma2 = 0.01;
    for (int i = 0; i < m; ++i) {
        double res_i = la2[i] - (slope_a * lk[i] + intercept_a);
        sigma2 += res_i * res_i;
    }
    sigma2 = std::max(sigma2 / m, 0.01);

    double denom_std = 4.0 * b * b * den;
    double r_std = (denom_std > 1e-12) ? std::sqrt(sigma2 / denom_std) : 0.3;

    return {r, r_std, b, r2};
}

static double estimate_intrinsic_dim(const Vec& laplacian_eigs) {
    int n = (int)laplacian_eigs.size();
    if (n < 4) return 2.0;

    Vec pos;
    for (double v : laplacian_eigs)
        if (v > 1e-8) pos.push_back(v);
    std::sort(pos.begin(), pos.end());

    int m = (int)pos.size();
    if (m < 4) return 2.0;

    int lo = std::max(1, (int)(0.02 * m));
    int hi = std::max(lo + 3, (int)(0.20 * m));
    hi = std::min(hi, m);

    Vec ks, vs;
    for (int i = lo; i < hi; ++i) {
        ks.push_back((double)(i + 1));
        vs.push_back(pos[i]);
    }
    auto fit = fit_power_law(ks, vs);
    double b = fit.exp_val;
    if (b < 0.05) return 2.0;
    double d = 2.0 / b;
    return std::max(0.5, std::min(d, 50.0));
}

StatResult stationarity_verdict(
    const Vec& ntk_eigs,
    const Vec& target_coeffs,
    double s_hint = -1.0
) {
    StatResult res;

    int n = (int)std::min(ntk_eigs.size(), target_coeffs.size());
    if (n < 6) {
        res.stationary = true;
        res.r_hat  = 0.5;
        res.r_std  = 0.3;
        res.beta_0 = 0.556;
        res.verdict = "insufficient_data";
        res.reason = "Need at least 6 eigenvalues. Got " + std::to_string(n) + ".";
        return res;
    }

    Vec pos_eigs(n);
    for (int i = 0; i < n; ++i)
        pos_eigs[i] = std::max(ntk_eigs[i], 1e-20);

    auto fit = fit_source_exp(pos_eigs, target_coeffs);
    double r     = fit.r;
    double r_std = fit.r_std;
    double b     = fit.b;

    double s_est = (s_hint > 0) ? s_hint : b;
    res.s_hat  = s_est;
    res.r_hat  = r;
    res.r_std  = r_std;
    res.r2_fit = fit.r2_b;

    double d_star_est = 2.0;
    res.d_star  = d_star_est;
    res.beta_0  = 2.0 * s_est / (2.0 * s_est + d_star_est);
    res.beta_pred = (2.0 * r * b + 1.0 > 1e-6)
        ? 2.0 * r * b / (2.0 * r * b + 1.0) : 0.0;

    double r_low  = r - 1.96 * r_std;
    double r_high = r + 1.96 * r_std;
    bool in_ci = (r_low <= 0.5 && 0.5 <= r_high);
    bool close = std::abs(r - 0.5) < 0.12;
    res.stationary = (close || in_ci);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    if (res.stationary) {
        oss << "STATIONARY. Source exponent r_hat=" << r << " (\xc2\xb1" << r_std << ") "
            << "is consistent with r=0.5 (self-organised criticality). "
            << "Model is pinned to the Sobolev minimax barrier \xce\xb2\xe2\x82\x80=" << res.beta_0 << ". "
            << "Additional data improves loss at rate D^{-" << res.beta_0 << "} — "
            << "no more than " << std::round((1.0 - res.beta_0) * 100)
            << "% further gain without compositional restructuring.";
        res.verdict = "stationary";
    } else {
        oss << "TRANSIENT. Source exponent r_hat=" << r << " (\xc2\xb1" << r_std << ") "
            << "deviates from r=0.5. Model kernel is still evolving. "
            << "The barrier \xce\xb2\xe2\x82\x80=" << res.beta_0 << " is not yet binding; "
            << "continued training may still improve the loss exponent.";
        res.verdict = "transient";
    }
    res.reason = oss.str();
    return res;
}

DimResult effective_dimension(
    const Vec& laplacian_eigs_data,
    const Vec& approx_errors,
    const Vec& model_sizes
) {
    DimResult res;

    double d_star = estimate_intrinsic_dim(laplacian_eigs_data);
    res.d_star = d_star;

    int m = (int)std::min(approx_errors.size(), model_sizes.size());
    double d_loc = d_star;

    if (m >= 3) {
        Vec ns, errs;
        for (int i = 0; i < m; ++i)
            if (model_sizes[i] > 0 && approx_errors[i] > 1e-30) {
                ns.push_back(model_sizes[i]);
                errs.push_back(approx_errors[i]);
            }
        if ((int)ns.size() >= 3) {
            auto fit = fit_power_law(ns, errs);
            double alpha_obs = -fit.exp_val;
            double s_est = 1.25;
            if (alpha_obs > 0.05) {
                double d_loc_est = 2.0 * s_est / alpha_obs;
                d_loc = std::max(0.5, std::min(d_loc_est, d_star * 2.0));
            }
        }
    }

    res.d_loc = d_loc;

    double s = 1.25;
    res.alpha_sobolev   = 2.0 * s / d_star;
    res.alpha_comp      = 2.0 * s / d_loc;
    res.compositional   = (d_loc < d_star * 0.85);
    res.compression_gain = (d_loc > 0) ? d_star / d_loc : 1.0;

    double theta = (d_star > 0 && d_loc > 0 && d_loc <= d_star)
        ? std::acos(std::min(1.0, std::sqrt(d_loc / d_star)))
        : 0.0;
    res.subspace_gap = theta;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    if (res.compositional) {
        oss << "COMPOSITIONAL STRUCTURE DETECTED. "
            << "Data intrinsic dimension d*=" << d_star
            << " but effective task dimension d_loc=" << d_loc << ". "
            << "Compositional exponent \xce\xb1=" << res.alpha_comp
            << " vs Sobolev baseline \xce\xb1=" << res.alpha_sobolev << " — "
            << std::setprecision(1) << res.compression_gain << "\xc3\x97 compression gain. "
            << "A model exploiting this structure scales as N^{-"
            << std::setprecision(3) << res.alpha_comp << "}.";
    } else {
        oss << "NO COMPOSITIONAL STRUCTURE. "
            << "Effective dimension d_loc=" << d_loc
            << " \xe2\x89\x88 d*=" << d_star << ". "
            << "Approximation exponent \xce\xb1=" << res.alpha_sobolev << ". "
            << "Scaling is bounded by the full ambient dimension.";
    }
    res.verdict = oss.str();
    return res;
}

BarrierResult barrier_certificate(
    double d_star,
    double d_loc,
    double s,
    double current_loss,
    double current_N,
    double current_D
) {
    BarrierResult res;
    res.d_star = d_star;
    res.d_loc  = d_loc;
    res.s      = s;

    res.beta_0 = 2.0 * s / (2.0 * s + d_star);
    res.alpha  = (d_loc > 0) ? 2.0 * s / d_loc : 0.0;

    double exp_crit = 2.0 * s / d_star;
    res.n_critical  = std::pow(current_D, exp_crit / (exp_crit + 1.0));

    double floor_loss = std::pow(std::max(current_D, 1.0), -res.beta_0);
    res.budget_gap    = (current_loss > 0)
        ? std::max(0.0, current_loss - floor_loss) : 0.0;

    double d_cross  = std::pow(std::max(current_N, 1.0), (2.0 * s + d_star) / d_star);
    res.saturated   = (current_D >= d_cross * 0.8);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "BARRIER CERTIFICATE. "
        << "Theoretical ceiling \xce\xb2\xe2\x82\x80=" << res.beta_0
        << " (Sobolev minimax, d*=" << d_star << ", s=" << s << "). ";

    if (d_loc < d_star * 0.85) {
        double beta_prime = 2.0 * s / (2.0 * s + d_loc);
        oss << "Compositional barrier \xce\xb2'=" << beta_prime
            << " (d_loc=" << d_loc << ") — "
            << std::setprecision(2) << (beta_prime / res.beta_0)
            << "\xc3\x97 faster data scaling. ";
    }

    oss << std::setprecision(4);
    if (res.saturated) {
        oss << "Budget D=" << std::setprecision(0) << current_D
            << " > D_cross\xe2\x89\x88" << d_cross
            << ": data-rich regime. Loss gap to theoretical floor = "
            << std::setprecision(4) << res.budget_gap << ".";
    } else {
        oss << std::setprecision(0)
            << "Budget D=" << current_D
            << " < D_cross\xe2\x89\x88" << d_cross
            << ": barrier not yet binding. More data will help. "
            << "Tokens to crossover: " << std::max(0.0, d_cross - current_D) << ".";
    }
    res.verdict = oss.str();
    return res;
}

}
