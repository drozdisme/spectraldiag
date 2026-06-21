# spectraldiag

**Mathematically rigorous ML model diagnostics.**

Answers not "what happened" (that's W&B) but **"why it happened and what to do"** — with mathematical proof, not empirical guessing.

```python
pip install spectraldiag
```

---

## Three core functions

### `stationarity_verdict(ntk_eigs, target_coeffs, s=-1, d_star=-1)`

Is your model's feature learning done, or still evolving?

```python
from spectraldiag import stationarity_verdict

# Without d_star: measures the source exponent r only.
result = stationarity_verdict(ntk_eigs, target_coeffs)
print(result.reason)
# STATIONARY. Source exponent r_hat=0.491 (±0.031) is consistent with r=0.5
# (self-organised criticality). The learned kernel has reached the stationary
# attractor. Provide d_star to obtain the data-scaling barrier β₀.

# With the measured data intrinsic dimension, you also get the barrier:
result = stationarity_verdict(ntk_eigs, target_coeffs, s=1.0, d_star=8.2)
# ... With d*=8.2, the Sobolev minimax barrier is β₀=0.196: asymptotically,
# loss improves no faster than D^{-0.196} in the data-rich regime.
```

**What it computes:** fits the *realised source exponent* `r` from the empirical
NTK spectrum. `r ≈ 0.5` certifies the model reached the stationary attractor.
The data-scaling barrier `β₀ = 2s/(2s+d*)` is a **separate** quantity that needs
the data intrinsic dimension `d*` — so it is only reported when you pass
`d_star`. Without it, `beta_0` is returned as `-1` (undefined) rather than
guessed. If `s` is omitted, it is inferred from the kernel decay via the Weyl
law `s = b·d*/2`.

### `effective_dimension(laplacian_eigs, approx_errors, model_sizes, s=1.0)`

Does your data have compositional structure your model could exploit?

```python
from spectraldiag import effective_dimension

result = effective_dimension(laplacian_eigs, approx_errors, model_sizes, s=1.0)
print(result.verdict)
# COMPOSITIONAL STRUCTURE DETECTED. Data intrinsic dimension d*=8.2 but
# effective task dimension d_loc=2.1. Compositional approximation exponent
# α=0.95 vs Sobolev baseline α=0.24 — 3.9× compression gain available.
```

`d*` comes from the graph-Laplacian spectrum; `d_loc` from the model-side
approximation exponent `α = 2s/d_loc`, which is why the smoothness `s` is a
parameter (default `1.0`).

**What it computes:** estimates `d*` from the graph-Laplacian spectrum of your data, `d_loc` from the model-side approximation exponent. If `d_loc < d*`, genuine compositional structure exists — and the phase transition theorem says emergence is real.

### `barrier_certificate(d_star, d_loc, s, current_loss, current_N, current_D)`

Where is your model relative to the theoretical ceiling?

```python
from spectraldiag import barrier_certificate

result = barrier_certificate(
    d_star=8.0, d_loc=2.0, s=1.25,
    current_loss=0.42, current_N=1e8, current_D=1e11
)
print(result.verdict)
# BARRIER CERTIFICATE. Theoretical ceiling β₀=0.238. With compositional
# structure (d_loc=2.0), barrier rises to β'=0.556 — 2.3× faster data
# scaling. Training budget D=1e11 has NOT passed the crossover D_cross≈...
```

---

## One-line integration

```python
from spectraldiag.callbacks import make_hf_callback

trainer = Trainer(
    ...,
    callbacks=[make_hf_callback(eval_data=(X_val, y_val))]
)
```

Works with HuggingFace Trainer and PyTorch Lightning out of the box.

---

## Graph-Laplacian protocol (for real data)

```python
from spectraldiag.graph_lap import graph_laplacian_eigs, estimate_d_star, double_dimension_consistency

eig_vals, eig_vecs = graph_laplacian_eigs(X_data, knn=10)
d_star = estimate_d_star(eig_vals)

consistency = double_dimension_consistency(d_star_data=d_star, d_loc_model=d_loc_from_model)
print(consistency["verdict"])
```

---

## Mathematical foundation

This library implements the three-paper programme:

- **TR** — *Boundaries of Stationary Feature Learning*: the Sobolev minimax barrier `β₀`, self-organised criticality `r=½`, approximation exponent `α=2s/d*`
- **AB** — *Foundations of a Theory of Composable Abstractions*: defect as projection, effective dimension `d_loc`, subspace gap as the order parameter
- **BM** — *Spectral Scaling Benchmark*: the decisive-test protocol, source exponent measurement, graph-Laplacian intrinsic dimension estimation

The decisive invariant: `d_loc < d*` ⟺ emergence is real ⟺ the phase transition theorem applies.

---

## Build from source

```bash
pip install -e ".[all]"
```

Requires only a C++17 compiler. The C++ core is a standard CPython
extension built automatically by setuptools — no CMake, no pybind11,
no extra build dependencies.
