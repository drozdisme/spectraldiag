from __future__ import annotations

from typing import Optional, List
import warnings


def _try_import(name):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


def _eigs_from_matrix(M, k: int = 64):
    np = _try_import("numpy")
    torch = _try_import("torch")
    if torch is not None and hasattr(M, 'detach'):
        M = M.detach().cpu().float().numpy()
    if np is None:
        return []
    M_np = np.array(M, dtype=float)
    n = M_np.shape[0]
    k = min(k, n)
    try:
        vals = np.linalg.eigvalsh(M_np)
        return sorted([max(float(v), 0.0) for v in vals], reverse=True)[:k]
    except Exception:
        return [float(M_np[i, i]) for i in range(min(k, n))]


def _ntk_eigs_from_model(model, data, k: int = 64):
    torch = _try_import("torch")
    np    = _try_import("numpy")
    if torch is None or np is None:
        return [], []

    try:
        model.eval()
        xs, ys = data
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(np.array(xs), dtype=torch.float32)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(np.array(ys), dtype=torch.float32)

        n_samples = min(len(xs), 128)
        xs = xs[:n_samples]
        ys = ys[:n_samples]

        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return [], []

        total_p = sum(p.numel() for p in params)

        if total_p > 50_000:
            n_proj = min(k * 2, 512)
            proj   = torch.randn(total_p, n_proj)
            grads  = []
            for x, y in zip(xs, ys):
                out  = model(x.unsqueeze(0))
                loss = ((out - y.unsqueeze(0)) ** 2).sum()
                g    = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                gv   = torch.cat([gi.flatten() for gi in g])
                grads.append((gv @ proj).detach())
        else:
            grads = []
            for x, y in zip(xs, ys):
                out  = model(x.unsqueeze(0))
                loss = ((out - y.unsqueeze(0)) ** 2).sum()
                g    = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False)
                gv   = torch.cat([gi.flatten() for gi in g])
                grads.append(gv.detach())

        J = torch.stack(grads)
        G = (J @ J.T).numpy()
        eigs = sorted([max(float(v), 0.0) for v in np.linalg.eigvalsh(G)], reverse=True)[:k]

        with torch.no_grad():
            preds   = model(xs).detach().numpy().flatten()
            targets = ys.numpy().flatten()

        n_coeff = min(len(preds), len(targets), k)
        target_coeffs = [float(targets[i] - preds[i]) for i in range(n_coeff)]

        return eigs, target_coeffs

    except Exception as e:
        warnings.warn(f"[SpectralDiagnostics] NTK extraction failed: {e}")
        return [], []


class HFSpectralCallback:
    def __init__(
        self,
        eval_data=None,
        k_eigs: int = 64,
        s: float = 1.25,
        verbose: bool = True,
        log_to_trainer: bool = True,
    ):
        self.eval_data      = eval_data
        self.k_eigs         = k_eigs
        self.s              = s
        self.verbose        = verbose
        self.log_to_trainer = log_to_trainer
        self._results: List[dict] = []

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        from spectraldiag import stationarity_verdict

        if model is None or self.eval_data is None:
            return control

        eigs, coeffs = _ntk_eigs_from_model(model, self.eval_data, self.k_eigs)
        if not eigs:
            return control

        res = stationarity_verdict(eigs, coeffs, self.s)
        self._results.append({
            "step":       state.global_step,
            "r_hat":      res.r_hat,
            "r_std":      res.r_std,
            "beta_0":     res.beta_0,
            "stationary": res.stationary,
        })

        if self.verbose:
            tag = "STATIONARY" if res.stationary else "TRANSIENT"
            print(
                f"\n[SpectralDiagnostics step={state.global_step}] {tag} "
                f"r_hat={res.r_hat:.3f}±{res.r_std:.3f}  β₀={res.beta_0:.3f}"
            )

        if self.log_to_trainer and metrics is not None:
            metrics["sd/r_hat"]      = res.r_hat
            metrics["sd/beta_0"]     = res.beta_0
            metrics["sd/stationary"] = float(res.stationary)

        return control

    @property
    def history(self) -> List[dict]:
        return self._results


class LightningSpectralCallback:
    def __init__(
        self,
        eval_data=None,
        k_eigs: int = 64,
        s: float = 1.25,
        check_every_n_epochs: int = 1,
        verbose: bool = True,
    ):
        self.eval_data             = eval_data
        self.k_eigs                = k_eigs
        self.s                     = s
        self.check_every_n_epochs  = check_every_n_epochs
        self.verbose               = verbose
        self._results: List[dict]  = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.check_every_n_epochs != 0:
            return

        from spectraldiag import stationarity_verdict

        data = self.eval_data
        if data is None:
            dm = getattr(trainer, 'datamodule', None)
            if dm is not None and hasattr(dm, 'val_dataloader'):
                try:
                    loader = dm.val_dataloader()
                    batch  = next(iter(loader))
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        data = (batch[0][:64], batch[1][:64])
                except Exception:
                    pass

        if data is None:
            return

        eigs, coeffs = _ntk_eigs_from_model(pl_module, data, self.k_eigs)
        if not eigs:
            return

        res = stationarity_verdict(eigs, coeffs, self.s)
        self._results.append({
            "epoch":      epoch,
            "r_hat":      res.r_hat,
            "r_std":      res.r_std,
            "beta_0":     res.beta_0,
            "stationary": res.stationary,
        })

        if self.verbose:
            tag = "STATIONARY" if res.stationary else "TRANSIENT"
            print(
                f"\n[SpectralDiagnostics epoch={epoch}] {tag} "
                f"r_hat={res.r_hat:.3f}±{res.r_std:.3f}  β₀={res.beta_0:.3f}"
            )

        try:
            trainer.logger.log_metrics({
                "sd/r_hat":      res.r_hat,
                "sd/beta_0":     res.beta_0,
                "sd/stationary": float(res.stationary),
            }, step=trainer.global_step)
        except Exception:
            pass

    @property
    def history(self) -> List[dict]:
        return self._results


def make_hf_callback(
    eval_data=None,
    k_eigs: int = 64,
    s: float = 1.25,
    verbose: bool = True,
):
    try:
        from transformers import TrainerCallback

        class _HFCb(TrainerCallback, HFSpectralCallback):
            def __init__(self):
                HFSpectralCallback.__init__(
                    self, eval_data=eval_data, k_eigs=k_eigs, s=s, verbose=verbose
                )

        return _HFCb()
    except ImportError:
        warnings.warn(
            "transformers not installed; returning plain HFSpectralCallback. "
            "Install with: pip install spectraldiag[hf]"
        )
        return HFSpectralCallback(eval_data=eval_data, k_eigs=k_eigs, s=s, verbose=verbose)


def make_lightning_callback(
    eval_data=None,
    k_eigs: int = 64,
    s: float = 1.25,
    verbose: bool = True,
):
    for pkg in ("lightning.pytorch.callbacks", "pytorch_lightning.callbacks"):
        try:
            import importlib
            mod = importlib.import_module(pkg)
            Callback = mod.Callback

            class _LightningCb(Callback, LightningSpectralCallback):
                def __init__(self):
                    LightningSpectralCallback.__init__(
                        self, eval_data=eval_data, k_eigs=k_eigs, s=s, verbose=verbose
                    )

            return _LightningCb()
        except ImportError:
            continue

    warnings.warn(
        "lightning not installed; returning plain LightningSpectralCallback. "
        "Install with: pip install spectraldiag[lightning]"
    )
    return LightningSpectralCallback(eval_data=eval_data, k_eigs=k_eigs, s=s, verbose=verbose)
