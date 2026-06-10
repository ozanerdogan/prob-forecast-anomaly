"""Shared stage-1 evaluation + frozen-forecast dump runner for added models.

Replicates run_anomaly_eval.py's protocol for a *single* model given pure
prediction callables, so every newly added model lands on the identical
seeded grid without duplicating the sweep logic:

  - test-side anomaly contexts: fresh ``default_rng(seed)`` per
    (kind, intensity) — byte-identical to what the original trio saw;
  - validation-side contexts: ``default_rng([seed, 1, VAL_STREAM_INDEX[kind],
    int(10 * intensity)])`` — matching run_anomaly_eval's streams;
  - FGSM is white-box via the caller's ``grad_fn`` (omit for gradient-free
    models — the transfer column arrives with the phase-4 wave);
  - every test setting also gets a hampel-cleaned (detect-and-clean) twin.

Multichannel stance (phase-2 'Asama-1' decision): corruption always hits the
TARGET channel only — multivariate models keep their covariate channels
clean, so their rows stay directly comparable with the univariate ones.

The callables work in standardised space on full encoder windows:
  predict_fn(x (N, L, C)) -> {"point": (N, H)?, "quantiles": (N, H, Q)?}
  grad_fn(x (N, L, C), y (N, H)) -> (N, L) gradient wrt the target channel
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.anomaly import VAL_STREAM_INDEX, apply_anomaly, linf_fgsm
from src.cleaning import hampel_clean
from src.metrics import report, report_probabilistic
from src.predictions_io import prediction_path, save_predictions
from src.preprocessing import TARGET
from src.seq_data import make_encoder_windows

INTENSITIES = (1.0, 2.0, 4.0)
NONGRAD_V1 = ("point_spike", "contextual_outlier", "level_shift")


def _with_target(x: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    out = x.copy()
    out[:, :, 0] = ctx
    return out


def evaluate_and_dump(
    model_name: str,
    data,
    predict_fn,
    *,
    root: Path,
    grad_fn=None,
    quantiles: np.ndarray | None = None,
    nongrad_types: tuple = NONGRAD_V1,
    intensities: tuple = INTENSITIES,
    alpha: float = 0.1,
    seed: int = 42,
    lookback: int = 168,
    horizon: int = 24,
    smoke: bool = False,
    include_cleaned: bool = True,
) -> dict:
    """Run the full dump protocol for one model; return clean-test metrics."""
    L, H = lookback, horizon
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    if smoke:
        sl = slice(0, 60)
        x_te, y_te, x_va, y_va = x_te[sl], y_te[sl], x_va[sl], y_va[sl]

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_true, yv_true = inv(y_te), inv(y_va)
    pred_dir = root / "results" / ("predictions_smoke" if smoke else "predictions")

    def dump(split, setting, ctx, preds):
        arrays: dict = {"context": ctx}
        if "quantiles" in preds:
            if quantiles is None:
                raise ValueError("quantile output without quantile levels")
            arrays["quantiles"] = inv(preds["quantiles"])
            arrays["levels"] = np.asarray(quantiles)
        if "point" in preds:
            arrays["point"] = inv(preds["point"])
        save_predictions(
            prediction_path(pred_dir, model_name, split, setting),
            y_true=y_true if split == "test" else yv_true,
            meta={"model": model_name, "split": split, "setting": setting,
                  "alpha": alpha, "seed": seed, "smoke": bool(smoke),
                  "fgsm": "white-box" if grad_fn is not None
                          else "N/A (no gradient; transfer protocol in phase-4 wave)",
                  "units": "physical degC (y_true/quantiles/point), standardised (context)"},
            **arrays,
        )

    def run(split, setting, x_base, ctx):
        preds = predict_fn(_with_target(x_base, ctx))
        dump(split, setting, ctx, preds)
        return preds

    # ----- validation: clean + injected (offline calibrators fit on these)
    ctx_va = x_va[:, :, 0].copy()
    run("val", "clean", x_va, ctx_va)
    g_va = grad_fn(x_va, y_va) if grad_fn is not None else None
    for kind in nongrad_types + (("fgsm",) if g_va is not None else ()):
        for intensity in intensities:
            if kind == "fgsm":
                ctx_adv, _ = linf_fgsm(ctx_va, g_va, intensity)
            else:
                rng = np.random.default_rng(
                    [seed, 1, VAL_STREAM_INDEX[kind], int(10 * intensity)])
                ctx_adv, _ = apply_anomaly(ctx_va, kind, intensity, rng)
            run("val", f"{kind}_{intensity:.1f}", x_va, ctx_adv)

    # ----- test: clean + injected (+ hampel-cleaned twins)
    ctx_te = x_te[:, :, 0].copy()
    clean_preds = run("test", "clean", x_te, ctx_te)
    if include_cleaned:
        run("test", "clean__cleaned", x_te, hampel_clean(ctx_te))
    g_te = grad_fn(x_te, y_te) if grad_fn is not None else None
    for kind in nongrad_types + (("fgsm",) if g_te is not None else ()):
        for intensity in intensities:
            setting = f"{kind}_{intensity:.1f}"
            if kind == "fgsm":
                ctx_adv, _ = linf_fgsm(ctx_te, g_te, intensity)
            else:
                rng = np.random.default_rng(seed)  # same stream as run_anomaly_eval
                ctx_adv, _ = apply_anomaly(ctx_te, kind, intensity, rng)
            run("test", setting, x_te, ctx_adv)
            if include_cleaned:
                run("test", f"{setting}__cleaned", x_te, hampel_clean(ctx_adv))

    # ----- clean-test metric block for the caller's results JSON
    y_flat = y_true.reshape(-1)
    metrics: dict = {"n_predictions": int(y_flat.size)}
    if "quantiles" in clean_preds:
        q_flat = inv(clean_preds["quantiles"]).reshape(-1, len(quantiles))
        metrics.update(report_probabilistic(y_flat, q_flat, np.asarray(quantiles), alpha=alpha))
    if "point" in clean_preds:
        p_flat = inv(clean_preds["point"]).reshape(-1)
        pb = report(y_flat, p_flat)
        metrics["point_rmse"] = pb["rmse"]
        metrics["point_mae"] = pb["mae"]
    return metrics
