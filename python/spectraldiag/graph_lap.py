from __future__ import annotations

from typing import Optional, Tuple
import warnings


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy required for graph_laplacian protocol")


def graph_laplacian_eigs(
    X,
    knn: int = 10,
    bw: Optional[float] = None,
    n_eigs: int = 50,
) -> Tuple[list, list]:
    np = _require_numpy()
    X = np.array(X, dtype=float)
    n, d = X.shape

    try:
        from sklearn.neighbors import NearestNeighbors
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        nbrs = NearestNeighbors(n_neighbors=min(knn + 1, n)).fit(X)
        dists, indices = nbrs.kneighbors(X)

        # Adaptive per-point bandwidth: median of knn distances
        if bw is None:
            sigma_i = dists[:, -1].clip(1e-8)
        else:
            sigma_i = np.full(n, float(bw))

        rows, cols, data = [], [], []
        for i in range(n):
            for j_idx in range(1, min(knn + 1, len(indices[i]))):
                j = indices[i, j_idx]
                dist = dists[i, j_idx]
                bw_ij = (sigma_i[i] + sigma_i[j]) / 2.0
                w = float(np.exp(-dist ** 2 / (2.0 * bw_ij ** 2)))
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([w, w])

        W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        W = (W + W.T) / 2
        deg = np.array(W.sum(axis=1)).flatten()
        D_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
        L_sym = sp.eye(n) - D_inv @ W

        k = min(n_eigs + 1, n - 2)
        vals, vecs = spla.eigsh(L_sym, k=k, which='SM', tol=1e-6, maxiter=2000)
        idx = np.argsort(np.real(vals))
        vals = np.real(vals[idx])
        vecs = vecs[:, idx].T

        return vals[1:].tolist(), vecs[1:].tolist()

    except ImportError:
        pass

    # Dense fallback (no sklearn/scipy)
    np = _require_numpy()
    if bw is None:
        diffs = X[:, None, :] - X[None, :, :]
        D2 = (diffs ** 2).sum(axis=-1)
        np.fill_diagonal(D2, np.inf)
        knn_dists = np.sort(D2, axis=1)[:, :knn]
        sigma_i = np.sqrt(knn_dists[:, -1].clip(1e-16))
    else:
        sigma_i = np.full(n, float(bw))

    diffs = X[:, None, :] - X[None, :, :]
    D2 = (diffs ** 2).sum(axis=-1)
    bw_mat = (sigma_i[:, None] + sigma_i[None, :]) / 2.0
    W = np.exp(-D2 / (2.0 * bw_mat ** 2))
    np.fill_diagonal(W, 0.0)

    deg = W.sum(axis=1)
    D_inv = np.diag(1.0 / np.maximum(deg, 1e-12))
    L = np.eye(n) - D_inv @ W

    vals, vecs = np.linalg.eigh(L)
    idx = np.argsort(vals)
    vals = vals[idx]; vecs = vecs[:, idx].T

    k = min(n_eigs, n - 1)
    return vals[1:k + 1].tolist(), vecs[1:k + 1].tolist()


def estimate_d_star(
    laplacian_eigs: list,
    lo_frac: float = 0.02,
    hi_frac: float = 0.20,
) -> float:
    np = _require_numpy()
    eigs = np.array(laplacian_eigs, dtype=float)
    pos  = np.sort(eigs[eigs > 1e-8])
    n    = len(pos)
    if n < 4:
        return 2.0

    lo = max(1, int(lo_frac * n))
    hi = max(lo + 3, int(hi_frac * n))
    hi = min(hi, n)

    ks  = np.arange(lo + 1, hi + 1, dtype=float)
    vs  = pos[lo:hi]
    lk  = np.log(ks)
    lv  = np.log(np.maximum(vs, 1e-30))

    A = np.stack([lk, np.ones(len(lk))], axis=1)
    c, _, _, _ = np.linalg.lstsq(A, lv, rcond=None)
    b = float(c[0])
    if b < 0.05:
        return 2.0
    return float(np.clip(2.0 / b, 0.5, 20.0))


def estimate_d_loc_from_approx(
    model_sizes: list,
    approx_errors: list,
    s: float = 1.25,
    lo_frac: float = 0.1,
    hi_frac: float = 0.9,
) -> float:
    np = _require_numpy()
    N = np.array(model_sizes, dtype=float)
    E = np.array(approx_errors, dtype=float)
    mask = (N > 0) & (E > 1e-30)
    N, E = N[mask], E[mask]
    if len(N) < 3:
        return 2.0

    idx = np.argsort(N)
    N, E = N[idx], E[idx]
    n   = len(N)
    lo  = max(0, int(lo_frac * n))
    hi  = min(n, max(lo + 3, int(hi_frac * n)))

    lN  = np.log(N[lo:hi])
    lE  = np.log(E[lo:hi])
    A   = np.stack([lN, np.ones(hi - lo)], axis=1)
    c, _, _, _ = np.linalg.lstsq(A, lE, rcond=None)
    alpha = float(-c[0])
    if alpha < 0.01:
        return 50.0
    return float(np.clip(2.0 * s / alpha, 0.5, 100.0))


def double_dimension_consistency(
    d_star_data: float,
    d_loc_model: float,
    tol: float = 0.15,
) -> dict:
    gap      = d_star_data - d_loc_model
    ratio    = d_loc_model / d_star_data if d_star_data > 0 else 1.0
    consistent = gap > tol * d_star_data
    verdict = (
        f"CONSISTENT: d_loc={d_loc_model:.2f} < d*={d_star_data:.2f} "
        f"(ratio={ratio:.2f}). Genuine compositional structure confirmed."
        if consistent else
        f"INCONSISTENT: d_loc={d_loc_model:.2f} ≈ d*={d_star_data:.2f} "
        f"(ratio={ratio:.2f}). No evidence of compositional structure — "
        "any observed emergence may be a metric artifact."
    )
    return {
        "consistent":  consistent,
        "d_star":      d_star_data,
        "d_loc":       d_loc_model,
        "ratio":       ratio,
        "gap":         gap,
        "verdict":     verdict,
    }
