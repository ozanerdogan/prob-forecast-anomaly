"""Stage-2 post-hoc interval calibrators — model-agnostic by construction.

Every calibrator consumes frozen quantile forecasts (see ``predictions_io``)
and returns adjusted quantiles; none ever touches a model. The shared repair
contract is affine around the median,

    q' = (med + shift) + scale * (q - med)

where ``shift``/``scale`` may be scalars (static), per-window vectors
(input-conditional) or evolve over the test sequence (ACI). Methods:

  - ``StaticTau``   : scalar spread temperature fit on validation by pinball
                      minimisation (wraps the existing ``src.calibration``).
                      Offline; the out-of-the-box baseline repair.
  - ``CQRCalibrator``: split-conformal additive margin on the outer interval
                      (Romano et al. 2019). Offline, fit on validation scores;
                      distribution-free finite-sample coverage on exchangeable
                      data. Only the outer (lo, hi) pair is calibrated.
  - ``ACITau``      : online spread adaptation driven by realised coverage
                      errors, in the spirit of adaptive conformal inference
                      (Gibbs & Candes 2021). Window t is repaired using only
                      windows < t (never its own outcome), so it is a
                      legitimate online protocol, not leakage. Acts on the
                      spread scale rather than the nominal level because the
                      models emit a fixed 7-quantile grid.
  - ``InputTau``    : offline input-conditional spread — a small regressor
                      from context features to the per-window required scale,
                      fit ONLY on validation windows (clean + synthetically
                      injected), so no test feedback of any kind is used.
"""
from __future__ import annotations

import numpy as np

from src.calibration import fit_spread_temperature
from src.metrics import mean_pinball_loss, mis, picp

TAU_MIN, TAU_MAX = 0.2, 10.0


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #
def _median_index(levels: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.asarray(levels) - 0.5)))


def _outer_indices(levels: np.ndarray, alpha: float) -> tuple[int, int]:
    levels = np.asarray(levels)
    lo = int(np.argmin(np.abs(levels - alpha / 2.0)))
    hi = int(np.argmin(np.abs(levels - (1.0 - alpha / 2.0))))
    return lo, hi


def transform_quantiles(
    q: np.ndarray, levels: np.ndarray, scale, shift=0.0
) -> np.ndarray:
    """Affine repair around the median. ``q`` is (N, H, Q) or (M, Q).

    ``scale``/``shift`` are scalars or per-window vectors of length N (first
    axis); vectors are broadcast over horizon and quantile axes.
    """
    q = np.asarray(q, dtype=float)
    med = q[..., _median_index(levels)][..., None]
    scale = np.asarray(scale, dtype=float)
    shift = np.asarray(shift, dtype=float)
    if scale.ndim == 1:  # per-window
        extra = (1,) * (q.ndim - 1)
        scale = scale.reshape(-1, *extra)
    if shift.ndim == 1:
        extra = (1,) * (q.ndim - 1)
        shift = shift.reshape(-1, *extra)
    return med + shift + scale * (q - med)


def interval_metrics(
    y_true: np.ndarray, q: np.ndarray, levels: np.ndarray, alpha: float
) -> dict:
    """PICP / MPIW / MIS of the central (1-alpha) interval + pinball + median RMSE."""
    levels = np.asarray(levels)
    qf = np.asarray(q, dtype=float).reshape(-1, len(levels))
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo, hi = _outer_indices(levels, alpha)
    med = qf[:, _median_index(levels)]
    return {
        "picp": picp(y, qf[:, lo], qf[:, hi]),
        "mpiw": float(np.mean(qf[:, hi] - qf[:, lo])),
        "mis": mis(y, qf[:, lo], qf[:, hi], alpha),
        "pinball": mean_pinball_loss(y, qf, levels),
        "rmse_median": float(np.sqrt(np.mean((med - y) ** 2))),
    }


def needed_tau(
    y_true: np.ndarray, q: np.ndarray, levels: np.ndarray, alpha: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-window minimal spread scale achieving (1-alpha) coverage.

    For each point, the scale that would just cover it is
    r = (y - med) / (q_hi - med) above the median and the mirrored ratio
    below; a window of H points reaches (1-alpha) coverage once the scale
    exceeds the ceil((1-alpha)*H)-th smallest r. This is the regression
    target for the input-conditional calibrator.
    """
    levels = np.asarray(levels)
    q = np.asarray(q, dtype=float)            # (N, H, Q)
    y = np.asarray(y_true, dtype=float)       # (N, H)
    lo, hi = _outer_indices(levels, alpha)
    med = q[..., _median_index(levels)]
    up = np.maximum(q[..., hi] - med, eps)
    dn = np.maximum(med - q[..., lo], eps)
    r = np.where(y >= med, (y - med) / up, (med - y) / dn)  # (N, H)
    k = int(np.ceil((1.0 - alpha) * r.shape[1]))
    r_sorted = np.sort(r, axis=1)
    tau = r_sorted[:, k - 1]
    return np.clip(tau, TAU_MIN, TAU_MAX)


# --------------------------------------------------------------------------- #
# Calibrators
# --------------------------------------------------------------------------- #
class StaticTau:
    """Scalar spread temperature (the existing out-of-the-box repair)."""

    name = "static"
    fit_on = "val_clean"

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.tau_: float | None = None

    def fit(self, y_val, q_val, levels, context_val=None) -> "StaticTau":
        self.tau_ = fit_spread_temperature(
            np.asarray(y_val).reshape(-1),
            np.asarray(q_val).reshape(-1, len(levels)),
            np.asarray(levels),
        )
        return self

    def apply(self, y_true, q, levels, context=None) -> np.ndarray:
        return transform_quantiles(q, levels, self.tau_)

    def params(self) -> dict:
        return {"tau": self.tau_}


class CQRCalibrator:
    """Split-conformal additive margin on the outer interval (Romano 2019)."""

    name = "cqr"
    fit_on = "val_clean"

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.margin_: float | None = None

    def fit(self, y_val, q_val, levels, context_val=None) -> "CQRCalibrator":
        levels = np.asarray(levels)
        qf = np.asarray(q_val, dtype=float).reshape(-1, len(levels))
        y = np.asarray(y_val, dtype=float).reshape(-1)
        lo, hi = _outer_indices(levels, self.alpha)
        scores = np.maximum(qf[:, lo] - y, y - qf[:, hi])
        n = len(scores)
        q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        self.margin_ = float(np.quantile(scores, q_level))
        return self

    def apply(self, y_true, q, levels, context=None) -> np.ndarray:
        """Widen only the outer pair by the conformal margin."""
        q = np.asarray(q, dtype=float).copy()
        lo, hi = _outer_indices(levels, self.alpha)
        q[..., lo] -= self.margin_
        q[..., hi] += self.margin_
        return q

    def params(self) -> dict:
        return {"margin": self.margin_}


class ACITau:
    """Online spread adaptation from realised coverage errors (ACI-style).

    Window t's interval uses the spread scale adapted from windows < t only:
        tau_{t+1} = tau_t + gamma * (miss_t - alpha)
    where miss_t is window t's realised miscoverage fraction. gamma and the
    starting tau are chosen by replaying the rule over the validation
    sequence (warm start), so the test sequence itself contributes nothing
    to the configuration — only to the online updates, as in deployment.
    """

    name = "aci"
    fit_on = "val_clean"

    def __init__(self, alpha: float, gammas=(0.01, 0.02, 0.05, 0.1, 0.2)):
        self.alpha = alpha
        self.gammas = tuple(gammas)
        self.gamma_: float | None = None
        self.tau0_: float = 1.0

    def _replay(self, y, q, levels, gamma, tau0):
        """Run the online rule over a window sequence; return (q', tau_path)."""
        levels = np.asarray(levels)
        lo, hi = _outer_indices(levels, self.alpha)
        n = len(q)
        out = np.empty_like(np.asarray(q, dtype=float))
        taus = np.empty(n)
        tau = float(tau0)
        for t in range(n):
            taus[t] = tau
            qc = transform_quantiles(q[t][None], levels, tau)[0]
            out[t] = qc
            inside = (y[t] >= qc[..., lo]) & (y[t] <= qc[..., hi])
            miss = 1.0 - float(np.mean(inside))
            tau = float(np.clip(tau + gamma * (miss - self.alpha), TAU_MIN, TAU_MAX))
        return out, taus

    def fit(self, y_val, q_val, levels, context_val=None) -> "ACITau":
        y = np.asarray(y_val, dtype=float)
        q = np.asarray(q_val, dtype=float)
        best = None
        for g in self.gammas:
            qc, taus = self._replay(y, q, levels, g, 1.0)
            m = interval_metrics(y, qc, levels, self.alpha)
            key = (abs(m["picp"] - (1 - self.alpha)), m["mis"])
            if best is None or key < best[0]:
                best = (key, g, float(taus[-1]))
        _, self.gamma_, self.tau0_ = best
        return self

    def apply(self, y_true, q, levels, context=None) -> np.ndarray:
        out, taus = self._replay(
            np.asarray(y_true, dtype=float), np.asarray(q, dtype=float),
            levels, self.gamma_, self.tau0_,
        )
        self.last_tau_path_ = taus
        return out

    def params(self) -> dict:
        return {"gamma": self.gamma_, "tau0": self.tau0_}


class InputTau:
    """Offline input-conditional spread scale from context features.

    Fit ONLY on validation windows — clean plus synthetically injected — by
    regressing the per-window required scale (``needed_tau``) on cheap
    context statistics. At test time it looks at the input context alone, so
    the objection "the calibrator saw test data" cannot arise by design.
    """

    name = "input_tau"
    fit_on = "val_all"

    def __init__(self, alpha: float, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.model_ = None

    @staticmethod
    def features(context: np.ndarray) -> np.ndarray:
        """Cheap per-window context statistics (N, L) -> (N, F).

        Tail statistics target 'how suspicious is the recent input': overall
        and tail-24h dispersion, the final jump, the tail-vs-earlier level
        gap, distance of the last value from the window median, and tail
        roughness (mean absolute first difference).
        """
        c = np.asarray(context, dtype=float)
        tail = c[:, -24:]
        head = c[:, :-24] if c.shape[1] > 24 else c
        feats = np.column_stack([
            c.std(axis=1),
            tail.std(axis=1),
            np.abs(c[:, -1] - c[:, -2]),
            np.abs(tail.mean(axis=1) - head.mean(axis=1)),
            np.abs(c[:, -1] - np.median(c, axis=1)),
            np.mean(np.abs(np.diff(tail, axis=1)), axis=1),
        ])
        return feats

    def fit(self, y_val, q_val, levels, context_val=None) -> "InputTau":
        from sklearn.ensemble import HistGradientBoostingRegressor

        if context_val is None:
            raise ValueError("InputTau.fit needs validation contexts")
        x = self.features(context_val)
        t = needed_tau(y_val, q_val, levels, self.alpha)
        self.model_ = HistGradientBoostingRegressor(
            random_state=self.random_state, max_iter=300, learning_rate=0.05
        ).fit(x, t)
        return self

    def apply(self, y_true, q, levels, context=None) -> np.ndarray:
        if context is None:
            raise ValueError("InputTau.apply needs the test contexts")
        tau = np.clip(self.model_.predict(self.features(context)), TAU_MIN, TAU_MAX)
        self.last_tau_ = tau
        return transform_quantiles(q, levels, tau)

    def params(self) -> dict:
        return {"n_features": 6, "regressor": "HistGradientBoostingRegressor"}
