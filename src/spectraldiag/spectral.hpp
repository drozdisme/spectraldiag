#pragma once
#define _USE_MATH_DEFINES
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace sd {

constexpr double SD_PI = 3.14159265358979323846;

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

inline double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline double norm2(const Vec& v) { return std::sqrt(dot(v, v)); }

inline Vec matvec(const Mat& A, const Vec& x) {
    Vec y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i)
        for (size_t j = 0; j < x.size(); ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

struct EigResult {
    Vec vals;
    Mat vecs;
};

static EigResult sym_eig(const Mat& A, int max_iter = 200) {
    int n = (int)A.size();
    Mat Q(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i) Q[i][i] = 1.0;
    Mat H = A;

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                if (std::abs(H[p][q]) < 1e-14) continue;
                double theta = 0.5 * (H[q][q] - H[p][p]) / H[p][q];
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(1.0 + theta * theta));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;
                for (int i = 0; i < n; ++i) {
                    double hip = H[i][p], hiq = H[i][q];
                    H[i][p] = c * hip - s * hiq;
                    H[i][q] = s * hip + c * hiq;
                }
                for (int j = 0; j < n; ++j) {
                    double hpj = H[p][j], hqj = H[q][j];
                    H[p][j] = c * hpj - s * hqj;
                    H[q][j] = s * hpj + c * hqj;
                }
                for (int i = 0; i < n; ++i) {
                    double qip = Q[i][p], qiq = Q[i][q];
                    Q[i][p] = c * qip - s * qiq;
                    Q[i][q] = s * qip + c * qiq;
                }
            }
        }
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q)
                off += H[p][q] * H[p][q];
        if (off < 1e-24) break;
    }

    Vec vals(n);
    for (int i = 0; i < n; ++i) vals[i] = H[i][i];

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return vals[a] > vals[b]; });

    EigResult res;
    res.vals.resize(n);
    res.vecs.resize(n, Vec(n));
    for (int i = 0; i < n; ++i) {
        res.vals[i] = vals[idx[i]];
        for (int j = 0; j < n; ++j)
            res.vecs[i][j] = Q[j][idx[i]];
    }
    return res;
}

struct PowerFit {
    double exp_val;
    double r2;
};

static PowerFit fit_power_law(const Vec& ks, const Vec& ys) {
    int n = (int)ks.size();
    if (n < 3) return {0.0, 0.0};
    Vec lk(n), ly(n);
    for (int i = 0; i < n; ++i) {
        lk[i] = std::log(ks[i]);
        ly[i] = std::log(std::max(ys[i], 1e-30));
    }
    double mk = std::accumulate(lk.begin(), lk.end(), 0.0) / n;
    double my = std::accumulate(ly.begin(), ly.end(), 0.0) / n;
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += (lk[i] - mk) * (ly[i] - my);
        den += (lk[i] - mk) * (lk[i] - mk);
    }
    double slope = (den > 1e-20) ? num / den : 0.0;
    double ss_res = 0.0, ss_tot = 0.0;
    double intercept = my - slope * mk;
    for (int i = 0; i < n; ++i) {
        double pred = slope * lk[i] + intercept;
        ss_res += (ly[i] - pred) * (ly[i] - pred);
        ss_tot += (ly[i] - my) * (ly[i] - my);
    }
    double r2 = (ss_tot > 1e-20) ? 1.0 - ss_res / ss_tot : 0.0;
    return {slope, r2};
}

[[maybe_unused]] static Mat gram_matrix(const Mat& X, double bw) {
    int n = (int)X.size();
    int d = (n > 0) ? (int)X[0].size() : 0;
    Mat K(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = i; j < n; ++j) {
            double dist2 = 0.0;
            for (int k = 0; k < d; ++k) {
                double diff = X[i][k] - X[j][k];
                dist2 += diff * diff;
            }
            double v = std::exp(-dist2 / (2.0 * bw * bw));
            K[i][j] = K[j][i] = v;
        }
    return K;
}

[[maybe_unused]] static Mat graph_laplacian(const Mat& X, double bw, int knn = 10) {
    int n = (int)X.size();
    int d = (n > 0) ? (int)X[0].size() : 0;

    std::vector<std::vector<std::pair<double,int>>> dists(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double dist2 = 0.0;
            for (int k = 0; k < d; ++k) {
                double diff = X[i][k] - X[j][k];
                dist2 += diff * diff;
            }
            dists[i].push_back({dist2, j});
        }
        std::sort(dists[i].begin(), dists[i].end());
        if ((int)dists[i].size() > knn) dists[i].resize(knn);
    }

    Mat W(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (auto& [d2, j] : dists[i])
            W[i][j] = W[j][i] = std::exp(-d2 / (2.0 * bw * bw));

    Vec deg(n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            deg[i] += W[i][j];

    Mat L(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i) {
        L[i][i] = deg[i];
        for (int j = 0; j < n; ++j)
            L[i][j] -= W[i][j];
    }
    return L;
}

}
