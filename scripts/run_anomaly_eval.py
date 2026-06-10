"""Evaluate deterministic vs probabilistic models under anomaly injection.

This is the project's proposed-method experiment. It trains the deterministic
LSTM baseline and the two probabilistic models (DeepAR, quantile Transformer),
then scores each on a clean test set and on the same test set with its input
context contaminated by four anomaly families (point spike, contextual outlier,
level shift, l-inf FGSM) at three intensities. Intensity is scaled by the local
rolling std of each window and the realised magnitude is recorded (scaled units
and degC) so the contamination strength is explicit in the results.

Methodological additions:
  - the non-gradient anomalies perturb a single shared target-context, so all
    models see the *same* contamination and the comparison is fair; FGSM is
    white-box per model by construction.
  - post-hoc spread-temperature calibration is fitted on the validation set for
    each probabilistic model, and PICP is reported before vs after calibration
    on the clean test set and under every anomaly setting.

  python scripts/run_anomaly_eval.py            # full run
  python scripts/run_anomaly_eval.py --smoke    # tiny fast sanity run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.anomaly import apply_anomaly, linf_fgsm  # noqa: E402
from src.baselines.lstm_baseline import LstmConfig  # noqa: E402
from src.baselines.naive_seasonal import naive_seasonal_forecast  # noqa: E402
from src.calibration import apply_spread_temperature, coverage_at, fit_spread_temperature  # noqa: E402
from src.metrics import mean_pinball_loss, mis, report, report_probabilistic  # noqa: E402
from src.models.deepar import DeepARConfig  # noqa: E402
from src.models.quantile_transformer import QTransformerConfig, QUANTILES_7  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_ar_windows, make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
INTENSITIES = (1.0, 2.0, 4.0)
NONGRAD_TYPES = ("point_spike", "contextual_outlier", "level_shift")
SEED = 42


def _prob_scores(y_true_flat, q_flat):
    out = report_probabilistic(y_true_flat, q_flat, QUANTILES, alpha=ALPHA)
    return {k: out[k] for k in ("crps", "pinball", "picp", "mpiw", "mis", "rmse", "mae")}


def _point_scores(y_true_flat, y_pred_flat):
    r = report(y_true_flat, y_pred_flat)
    return {"rmse": r["rmse"], "mae": r["mae"]}


def _naive_forecast_batch(ctx, horizon, season=24):
    """Seasonal-naive forecast per context row: (N, L) -> (N, horizon).

    Untrained baseline; reuses naive_seasonal_forecast on each (possibly
    anomaly-injected) context window. Operates in standardised space. With
    horizon == season this is exactly the last season of each context, so any
    perturbation landing in that trailing window propagates into the forecast.
    """
    return np.stack([naive_seasonal_forecast(ctx[i], horizon, season) for i in range(len(ctx))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    lstm_cfg = LstmConfig()
    deepar_cfg = DeepARConfig()
    qt_cfg = QTransformerConfig(n_features=1, quantiles=QUANTILES_7)
    if args.smoke:
        lstm_cfg.epochs = deepar_cfg.epochs = qt_cfg.epochs = 1
        deepar_cfg.n_samples = 50

    data = E.prepare(use_covariates=False)
    L, H = deepar_cfg.lookback, deepar_cfg.horizon

    # Build aligned test / val windows (same stride -> shared indices).
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    yseq_te, cov_te = make_ar_windows(data.test, L, H, stride=H)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    yseq_va, cov_va = make_ar_windows(data.val, L, H, stride=H)
    if args.smoke:
        sl = slice(0, 60)
        x_te, y_te, yseq_te, cov_te = x_te[sl], y_te[sl], yseq_te[sl], cov_te[sl]
        x_va, y_va, yseq_va, cov_va = x_va[sl], y_va[sl], yseq_va[sl], cov_va[sl]

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_true = inv(y_te)
    y_true_flat = y_true.reshape(-1)

    print("Training LSTM ...")
    lstm = E.fit_lstm(data, lstm_cfg)
    print("Training DeepAR ...")
    deepar = E.fit_deepar(data, deepar_cfg)
    print("Training quantile Transformer ...")
    qt = E.fit_qtransformer(data, qt_cfg)

    # ----------------------------------------------------------------- #
    # Fit post-hoc calibration on the validation set (per prob model).
    # ----------------------------------------------------------------- #
    q_va_deepar, _ = E.deepar_predict(deepar, deepar_cfg, yseq_va, cov_va, QUANTILES)
    q_va_qt = E.qtransformer_predict(qt, x_va)
    yv = inv(y_va).reshape(-1)
    tau = {
        "deepar": fit_spread_temperature(yv, inv(q_va_deepar).reshape(-1, len(QUANTILES)), QUANTILES),
        "qtransformer": fit_spread_temperature(yv, inv(q_va_qt).reshape(-1, len(QUANTILES)), QUANTILES),
    }
    print(f"calibration tau: {tau}")

    def prob_block(q_preds_raw, model_name):
        """Pre/post-calibration scores for a probabilistic model."""
        q_flat = q_preds_raw.reshape(-1, len(QUANTILES))
        block = _prob_scores(y_true_flat, q_flat)
        q_cal = apply_spread_temperature(q_flat, QUANTILES, tau[model_name])
        block["picp_cal"] = coverage_at(q_cal, y_true_flat, QUANTILES, ALPHA)
        lo = int(np.argmin(np.abs(QUANTILES - ALPHA / 2)))
        hi = int(np.argmin(np.abs(QUANTILES - (1 - ALPHA / 2))))
        block["mis_cal"] = mis(y_true_flat, q_cal[:, lo], q_cal[:, hi], ALPHA)
        block["pinball_cal"] = mean_pinball_loss(y_true_flat, q_cal, QUANTILES)
        return block

    # ----------------------------------------------------------------- #
    # Clean evaluation.
    # ----------------------------------------------------------------- #
    results: dict = {
        "config": {
            "intensities": list(INTENSITIES),
            "anomaly_types": list(NONGRAD_TYPES) + ["fgsm"],
            "alpha": ALPHA,
            "seed": SEED,
            "smoke": bool(args.smoke),
            "lstm_epochs": lstm_cfg.epochs,
            "deepar_epochs": deepar_cfg.epochs,
            "qt_epochs": qt_cfg.epochs,
            "deterministic_models": ["lstm", "naive_seasonal"],
            "naive_fgsm": "N/A (FGSM is white-box; naive_seasonal has no gradient)",
        },
        "quantiles": QUANTILES.tolist(),
        "calibration_tau": tau,
        "clean": {},
        "anomaly": {},
    }

    q_clean_deepar, _ = E.deepar_predict(deepar, deepar_cfg, yseq_te, cov_te, QUANTILES)
    q_clean_qt = E.qtransformer_predict(qt, x_te)
    p_clean_lstm = E.lstm_predict(lstm, x_te[:, :, 0])
    results["clean"]["lstm"] = _point_scores(y_true_flat, inv(p_clean_lstm).reshape(-1))
    results["clean"]["naive_seasonal"] = _point_scores(
        y_true_flat, inv(_naive_forecast_batch(x_te[:, :, 0], H)).reshape(-1)
    )
    results["clean"]["deepar"] = prob_block(inv(q_clean_deepar), "deepar")
    results["clean"]["qtransformer"] = prob_block(inv(q_clean_qt), "qtransformer")

    std_t = float(data.scaler.std[TARGET])
    ctx_clean = x_te[:, :, 0].copy()  # shared target context (== yseq_te[:, :L])

    # ----------------------------------------------------------------- #
    # Anomaly sweep.
    # ----------------------------------------------------------------- #
    for kind in NONGRAD_TYPES + ("fgsm",):
        results["anomaly"][kind] = {}
        for intensity in INTENSITIES:
            rng = np.random.default_rng(SEED)
            rec: dict = {}

            if kind == "fgsm":
                # white-box per model: each model perturbs along its own gradient
                g_lstm = E.lstm_context_grad(lstm, ctx_clean, y_te)
                ctx_lstm, mag = linf_fgsm(ctx_clean, g_lstm, intensity)
                g_qt = E.qtransformer_context_grad(qt, x_te, y_te, QUANTILES)
                ctx_qt, _ = linf_fgsm(ctx_clean, g_qt, intensity)
                g_da = E.deepar_context_grad(deepar, deepar_cfg, yseq_te, cov_te)
                ctx_da, _ = linf_fgsm(ctx_clean, g_da, intensity)
            else:
                ctx_adv, mag = apply_anomaly(ctx_clean, kind, intensity, rng)
                ctx_lstm = ctx_qt = ctx_da = ctx_adv

            rec["realized_magnitude_scaled"] = mag
            rec["realized_magnitude_degC"] = mag * std_t

            # LSTM
            p = E.lstm_predict(lstm, ctx_lstm)
            rec["lstm"] = _point_scores(y_true_flat, inv(p).reshape(-1))

            # naive_seasonal (untrained, deterministic): shares the same injected
            # context as LSTM for the non-gradient anomalies; FGSM is white-box and
            # undefined for a non-differentiable model -> recorded as N/A.
            if kind == "fgsm":
                rec["naive_seasonal"] = {"rmse": None, "mae": None}
            else:
                rec["naive_seasonal"] = _point_scores(
                    y_true_flat, inv(_naive_forecast_batch(ctx_lstm, H)).reshape(-1)
                )

            # DeepAR
            yseq_adv = yseq_te.copy()
            yseq_adv[:, :L] = ctx_da
            q_da, _ = E.deepar_predict(deepar, deepar_cfg, yseq_adv, cov_te, QUANTILES)
            rec["deepar"] = prob_block(inv(q_da), "deepar")

            # Quantile Transformer
            x_adv = x_te.copy()
            x_adv[:, :, 0] = ctx_qt
            q_qt = E.qtransformer_predict(qt, x_adv)
            rec["qtransformer"] = prob_block(inv(q_qt), "qtransformer")

            results["anomaly"][kind][f"{intensity:.1f}"] = rec
            print(f"  {kind:18s} intensity={intensity:>4} | "
                  f"lstm rmse={rec['lstm']['rmse']:.2f} "
                  f"deepar crps={rec['deepar']['crps']:.2f} picp={rec['deepar']['picp']:.2f} "
                  f"qt crps={rec['qtransformer']['crps']:.2f} picp={rec['qtransformer']['picp']:.2f}")

    out_dir = ROOT / "results" / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "anomaly_eval.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
