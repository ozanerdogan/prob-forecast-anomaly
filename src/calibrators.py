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
  - ``DetectAdaptTau``: detect-then-adapt — an explicit anomaly detector
                      (logistic, on context statistics) gates the repair:
                      tau = (1-s)*tau_clean + s*tau_anom(context). Clean
                      windows keep the sharp static regime; the anomaly
                      regime only enters when the detector fires. Fit on the
                      same labelled validation protocol as InputTau.
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


class ACIMargin:
    """Online additive conformal margin — the adaptive-CQR bridge.

    CQR widens the outer interval by a fixed margin fit offline; under shift
    that margin is stale. This is its online counterpart (Gibbs-Candes update
    on the margin instead of the spread scale): the outer pair is widened by
    m_t, and m_t is grown/shrunk by the realised miscoverage of windows < t.
    Same legitimacy argument as ACITau — window t never sees its own outcome.
    """

    name = "aci_margin"
    fit_on = "val_clean"

    def __init__(self, alpha: float, gammas=(0.05, 0.1, 0.2, 0.5, 1.0)):
        self.alpha = alpha
        self.gammas = tuple(gammas)
        self.gamma_: float | None = None
        self.m0_: float = 0.0

    def _replay(self, y, q, levels, gamma, m0):
        levels = np.asarray(levels)
        lo, hi = _outer_indices(levels, self.alpha)
        n = len(q)
        out = np.asarray(q, dtype=float).copy()
        margins = np.empty(n)
        m = float(m0)
        for t in range(n):
            margins[t] = m
            out[t, ..., lo] = q[t][..., lo] - m
            out[t, ..., hi] = q[t][..., hi] + m
            inside = (y[t] >= out[t][..., lo]) & (y[t] <= out[t][..., hi])
            miss = 1.0 - float(np.mean(inside))
            m = max(0.0, m + gamma * (miss - self.alpha))
        return out, margins

    def fit(self, y_val, q_val, levels, context_val=None) -> "ACIMargin":
        y = np.asarray(y_val, dtype=float)
        q = np.asarray(q_val, dtype=float)
        # seed m0 with the offline split-conformal margin (warm start)
        cqr = CQRCalibrator(self.alpha).fit(y_val, q_val, levels)
        best = None
        for g in self.gammas:
            qc, ms = self._replay(y, q, levels, g, cqr.margin_)
            mtr = interval_metrics(y, qc, levels, self.alpha)
            key = (abs(mtr["picp"] - (1 - self.alpha)), mtr["mis"])
            if best is None or key < best[0]:
                best = (key, g, float(ms[-1]))
        _, self.gamma_, self.m0_ = best
        return self

    def apply(self, y_true, q, levels, context=None) -> np.ndarray:
        out, ms = self._replay(np.asarray(y_true, dtype=float),
                               np.asarray(q, dtype=float), levels, self.gamma_, self.m0_)
        self.last_margin_path_ = ms
        return out

    def params(self) -> dict:
        return {"gamma": self.gamma_, "m0": self.m0_}


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


class DetectAdaptTau:
    """Detect-then-adapt: an explicit anomaly detector gates the spread repair.

    Two stages, both fit ONLY on validation windows (clean + synthetically
    injected — the same leakage-free protocol as InputTau):

      1. DETECT — a small gradient-boosted classifier on cheap context
         statistics returns the probability ``s`` that the window's input is
         contaminated. (A linear/logistic detector fails here by design: the
         anomaly class is heterogeneous — spikes RAISE the dispersion
         statistics while flatlines LOWER them — so "anomalous" is
         non-monotone in every feature and needs a non-linear boundary.)
      2. ADAPT  — the spread scale blends two regimes by that probability,

             tau(w) = (1 - s_w) * tau_clean + s_w * tau_anom(w),

         where ``tau_clean`` is the scalar validation-clean spread temperature
         (sharp intervals when nothing is wrong) and ``tau_anom(w)`` is an
         anomaly-conditional regressor fit only on the injected validation
         windows (how much repair a contaminated window needs).

    Contrast with the always-on repairs: InputTau applies one implicit
    regressor everywhere and pays a width cost on clean inputs; ACI reacts
    only after coverage is already lost. Here the clean regime is explicitly
    preserved unless the detector fires.
    """

    name = "detect_adapt"
    fit_on = "val_labeled"

    def __init__(self, alpha: float, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.detector_ = None
        self.regressor_ = None
        self.tau_clean_: float | None = None
        self.ref_profile_: np.ndarray | None = None

    def features(self, context: np.ndarray) -> np.ndarray:
        """InputTau's 6 context statistics + 6 fault-mechanism features.

        The extra features each target a fault family the dispersion
        statistics miss: hampel outlier count / largest correction (isolated
        spikes), max 12h block-mean step and a CUSUM statistic (level shifts
        at an arbitrary changepoint), the absolute tail-24h slope (drift
        ramps), and — once fit — the correlation with the mean clean
        validation profile (clock skew misaligns the diurnal cycle; the
        stride-24 grid keeps all windows phase-aligned, so a reference
        profile is meaningful).
        """
        from src.cleaning import hampel_clean

        c = np.asarray(context, dtype=float)
        n, length = c.shape
        base = InputTau.features(c)
        diff = np.abs(c - hampel_clean(c))

        nb = length // 12
        block_means = c[:, :nb * 12].reshape(n, nb, 12).mean(axis=2)
        step = np.abs(np.diff(block_means, axis=1)).max(axis=1)
        z = c - c.mean(axis=1, keepdims=True)
        cusum = np.abs(np.cumsum(z, axis=1)).max(axis=1) / length
        tail = c[:, -24:]
        t = np.arange(tail.shape[1]) - (tail.shape[1] - 1) / 2.0
        slope = np.abs((tail * t).sum(axis=1) / (t * t).sum())

        cols = [base, (diff > 1e-9).sum(axis=1), diff.max(axis=1),
                step, cusum, slope]
        if self.ref_profile_ is not None:
            rp = self.ref_profile_ - self.ref_profile_.mean()
            denom = c.std(axis=1) * rp.std() + 1e-9
            corr = (z * rp).mean(axis=1) / denom
            cols.append(corr)
        return np.column_stack(cols)

    def fit(self, y_val, q_val, levels, context_val=None, labels_val=None
            ) -> "DetectAdaptTau":
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      HistGradientBoostingRegressor)

        if context_val is None or labels_val is None:
            raise ValueError(
                "DetectAdaptTau.fit needs validation contexts and "
                "clean/injected labels")
        lab = np.asarray(labels_val, dtype=int)
        if lab.min() == lab.max():
            raise ValueError("labels_val must contain both clean (0) and "
                             "injected (1) windows")
        # mean clean profile first -- the feature builder needs it
        self.ref_profile_ = np.asarray(context_val, dtype=float)[lab == 0].mean(axis=0)
        x = self.features(context_val)
        clean = lab == 0
        # Raw probabilities sit near 0.5 even on clean windows (every clean
        # window has corrupted twins labelled 1; the balanced reweighting
        # makes the contradiction land mid-scale), so blending on the raw
        # score would widen clean intervals. The gate maps [q90, q99] of the
        # UNSEEN-clean score distribution to [0, 1]: ~90% of clean windows
        # keep tau_clean exactly, anything scoring above the clean range
        # engages the anomaly regime fully. Anchoring needs care: in-sample
        # clean scores are deflated when few twins exist (the classifier
        # memorises the clean window), and naive K-fold OOF saturates to 1
        # (the held-out clean window's corrupted twins remain in training).
        # So the split is GROUP-AWARE: a random 20% of base windows — clean
        # AND all their corrupted twins — never enter detector training, and
        # the gate is anchored on those truly unseen clean scores. The 20%
        # is drawn at random (seeded) rather than as the chronological tail:
        # a tail block is all late-autumn windows, which an
        # earlier-months-only detector scores as out-of-distribution
        # (seasonal confound), saturating the anchor.
        n0 = int(clean.sum())
        if len(lab) % n0 == 0:
            groups = np.tile(np.arange(n0), len(lab) // n0)
        else:  # unequal setting blocks: degrade to per-row groups
            groups = np.arange(len(lab))
        rng = np.random.default_rng(self.random_state)
        n_groups = int(groups.max()) + 1
        ho = rng.choice(n_groups, size=max(1, int(round(0.2 * n_groups))),
                        replace=False)
        held = np.isin(groups, ho)
        self.detector_ = HistGradientBoostingClassifier(
            class_weight="balanced", random_state=self.random_state,
            max_iter=300, learning_rate=0.05,
        ).fit(x[~held], lab[~held])
        s_anchor = self.detector_.predict_proba(x[held & clean])[:, 1]
        self.gate_lo_ = float(np.quantile(s_anchor, 0.90))
        self.gate_hi_ = max(float(np.quantile(s_anchor, 0.99)),
                            self.gate_lo_ + 1e-3)

        self.tau_clean_ = fit_spread_temperature(
            np.asarray(y_val)[clean].reshape(-1),
            np.asarray(q_val)[clean].reshape(-1, len(levels)),
            np.asarray(levels),
        )
        t = needed_tau(np.asarray(y_val)[~clean], np.asarray(q_val)[~clean],
                       levels, self.alpha)
        self.regressor_ = HistGradientBoostingRegressor(
            random_state=self.random_state, max_iter=300, learning_rate=0.05
        ).fit(x[~clean], t)
        return self

    def anomaly_score(self, context) -> np.ndarray:
        """Per-window contamination probability s in [0, 1] (raw, ungated)."""
        return self.detector_.predict_proba(self.features(context))[:, 1]

    def gate(self, s: np.ndarray) -> np.ndarray:
        """Map raw scores through the clean-anchored ramp to [0, 1]."""
        s = np.asarray(s, dtype=float)
        return np.clip((s - self.gate_lo_) / (self.gate_hi_ - self.gate_lo_),
                       0.0, 1.0)

    def blend(self, g: np.ndarray, tau_anom: np.ndarray) -> np.ndarray:
        """Gated mix of the clean and anomaly regimes."""
        g = np.asarray(g, dtype=float)
        tau = (1.0 - g) * self.tau_clean_ \
            + g * np.asarray(tau_anom, dtype=float)
        return np.clip(tau, TAU_MIN, TAU_MAX)

    def apply(self, y_true, q, levels, context=None) -> np.ndarray:
        if context is None:
            raise ValueError("DetectAdaptTau.apply needs the test contexts")
        x = self.features(context)
        s = self.detector_.predict_proba(x)[:, 1]
        g = self.gate(s)
        tau_anom = np.clip(self.regressor_.predict(x), TAU_MIN, TAU_MAX)
        tau = self.blend(g, tau_anom)
        self.last_score_ = s
        self.last_gate_ = g
        self.last_tau_ = tau
        return transform_quantiles(q, levels, tau)

    def params(self) -> dict:
        return {"tau_clean": self.tau_clean_, "n_features": 12,
                "gate": [self.gate_lo_, self.gate_hi_],
                "detector": "HistGradientBoostingClassifier(class_weight=balanced)",
                "regressor": "HistGradientBoostingRegressor"}
