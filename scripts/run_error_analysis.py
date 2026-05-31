"""Error analysis for the deterministic + probabilistic models.

Trains the LSTM baseline, DeepAR and the quantile Transformer, then writes
results/error_analysis.json with:
  - per forecast-horizon RMSE/MAE (and CRPS for the probabilistic models),
  - error broken down by meteorological season and by temperature range,
  - overconfident-failure analysis for the probabilistic models, and
  - worst-window error grouped by anomaly type and intensity.

  python scripts/run_error_analysis.py
  python scripts/run_error_analysis.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import error_analysis as EA  # noqa: E402
from src import experiment as E  # noqa: E402
from src.anomaly import apply_anomaly, linf_fgsm  # noqa: E402
from src.baselines.lstm_baseline import LstmConfig  # noqa: E402
from src.metrics import crps_from_quantiles  # noqa: E402
from src.models.deepar import DeepARConfig  # noqa: E402
from src.models.quantile_transformer import QTransformerConfig, QUANTILES_7  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_ar_windows, make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
INTENSITIES = (1.0, 2.0, 4.0)
NONGRAD_TYPES = ("point_spike", "contextual_outlier", "level_shift")
SEED = 42


def _median(q_preds):  # (N,H,Q) -> (N,H)
    return q_preds[:, :, int(np.argmin(np.abs(QUANTILES - 0.5)))]


def _per_horizon_crps_q(y_true, q_preds):  # quantile-based CRPS per horizon
    return [crps_from_quantiles(y_true[:, k], q_preds[:, k, :], QUANTILES) for k in range(y_true.shape[1])]


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
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731

    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    yseq_te, cov_te = make_ar_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_te, y_te, yseq_te, cov_te = x_te[:80], y_te[:80], yseq_te[:80], cov_te[:80]

    y_true = inv(y_te)  # (N, H)
    n_windows = len(y_true)

    print("Training LSTM ...")
    lstm = E.fit_lstm(data, lstm_cfg)
    print("Training DeepAR ...")
    deepar = E.fit_deepar(data, deepar_cfg)
    print("Training quantile Transformer ...")
    qt = E.fit_qtransformer(data, qt_cfg)

    # Clean predictions.
    p_lstm = inv(E.lstm_predict(lstm, x_te[:, :, 0]))
    q_da, _ = E.deepar_predict(deepar, deepar_cfg, yseq_te, cov_te, QUANTILES)
    q_da = inv(q_da)
    q_qt = inv(E.qtransformer_predict(qt, x_te))
    med_da, med_qt = _median(q_da), _median(q_qt)

    months = EA.target_months(data.test_index, n_windows, L, H, stride=H)

    results = {
        "config": {
            "lstm_epochs": lstm_cfg.epochs, "deepar_epochs": deepar_cfg.epochs,
            "qt_epochs": qt_cfg.epochs, "smoke": bool(args.smoke),
            "n_windows": n_windows, "horizon": H, "alpha": ALPHA,
            "temp_edges": list(EA.TEMP_LABELS),
        },
        "per_horizon": {
            "lstm": EA.per_horizon_point(y_true, p_lstm),
            "deepar": {**EA.per_horizon_point(y_true, med_da),
                       "crps": _per_horizon_crps_q(y_true, q_da)},
            "qtransformer": {**EA.per_horizon_point(y_true, med_qt),
                             "crps": _per_horizon_crps_q(y_true, q_qt)},
        },
        "by_season": {
            "lstm": EA.season_breakdown(y_true, p_lstm, months),
            "deepar": EA.season_breakdown(y_true, med_da, months),
            "qtransformer": EA.season_breakdown(y_true, med_qt, months),
        },
        "by_temperature": {
            "lstm": EA.temperature_breakdown(y_true, p_lstm),
            "deepar": EA.temperature_breakdown(y_true, med_da),
            "qtransformer": EA.temperature_breakdown(y_true, med_qt),
        },
        "overconfident": {
            "deepar": EA.overconfident_failures(y_true, q_da, QUANTILES, ALPHA),
            "qtransformer": EA.overconfident_failures(y_true, q_qt, QUANTILES, ALPHA),
        },
        "worst_windows_under_anomaly": {},
    }

    # Worst-window grouping under each anomaly type / intensity.
    ctx_clean = x_te[:, :, 0].copy()
    for kind in NONGRAD_TYPES + ("fgsm",):
        results["worst_windows_under_anomaly"][kind] = {}
        for intensity in INTENSITIES:
            rng = np.random.default_rng(SEED)
            if kind == "fgsm":
                g_l = E.lstm_context_grad(lstm, ctx_clean, y_te)
                ctx_l, _ = linf_fgsm(ctx_clean, g_l, intensity)
                g_q = E.qtransformer_context_grad(qt, x_te, y_te, QUANTILES)
                ctx_q, _ = linf_fgsm(ctx_clean, g_q, intensity)
                g_d = E.deepar_context_grad(deepar, deepar_cfg, yseq_te, cov_te)
                ctx_d, _ = linf_fgsm(ctx_clean, g_d, intensity)
            else:
                ctx_a, _ = apply_anomaly(ctx_clean, kind, intensity, rng)
                ctx_l = ctx_q = ctx_d = ctx_a

            e_lstm = EA.window_rmse(y_true, inv(E.lstm_predict(lstm, ctx_l)))
            x_adv = x_te.copy(); x_adv[:, :, 0] = ctx_q
            e_qt = EA.window_rmse(y_true, _median(inv(E.qtransformer_predict(qt, x_adv))))
            yseq_adv = yseq_te.copy(); yseq_adv[:, :L] = ctx_d
            qd, _ = E.deepar_predict(deepar, deepar_cfg, yseq_adv, cov_te, QUANTILES)
            e_da = EA.window_rmse(y_true, _median(inv(qd)))

            results["worst_windows_under_anomaly"][kind][f"{intensity:.1f}"] = {
                "lstm": EA.worst_window_summary(e_lstm),
                "deepar": EA.worst_window_summary(e_da),
                "qtransformer": EA.worst_window_summary(e_qt),
            }
            print(f"  {kind:18s} i={intensity:>4} | worst-decile rmse "
                  f"lstm={results['worst_windows_under_anomaly'][kind][f'{intensity:.1f}']['lstm']['worst_decile_mean']:.2f} "
                  f"qt={results['worst_windows_under_anomaly'][kind][f'{intensity:.1f}']['qtransformer']['worst_decile_mean']:.2f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "error_analysis.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
