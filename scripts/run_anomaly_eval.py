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
from src.cleaning import hampel_clean  # noqa: E402
from src.metrics import mean_pinball_loss, mis, report, report_probabilistic  # noqa: E402
from src.models.deepar import DeepARConfig  # noqa: E402
from src.predictions_io import prediction_path, save_predictions  # noqa: E402
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

    # ----------------------------------------------------------------- #
    # Frozen-forecast store (stage-2 calibration reads these, never the
    # models). Smoke runs go to a separate dir so real dumps survive.
    # ----------------------------------------------------------------- #
    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")

    def dump(model, split, setting, **arrays):
        meta = {"model": model, "split": split, "setting": setting, "alpha": ALPHA,
                "seed": SEED, "smoke": bool(args.smoke),
                "units": "physical degC (y_true/quantiles/point), standardised (context)"}
        save_predictions(prediction_path(pred_dir, model, split, setting),
                         meta=meta, **arrays)

    # Validation dumps: clean + every anomaly setting, probabilistic models
    # only (offline calibrators fit on these). Validation injections use their
    # own rng stream so the test-side streams below stay byte-identical.
    yv_true = inv(y_va)
    ctx_va = x_va[:, :, 0].copy()
    dump("deepar", "val", "clean", y_true=yv_true, quantiles=inv(q_va_deepar),
         levels=QUANTILES, context=ctx_va)
    dump("qtransformer", "val", "clean", y_true=yv_true, quantiles=inv(q_va_qt),
         levels=QUANTILES, context=ctx_va)

    print("Dumping validation anomaly predictions ...")
    g_va_qt = E.qtransformer_context_grad(qt, x_va, y_va, QUANTILES)
    g_va_da = E.deepar_context_grad(deepar, deepar_cfg, yseq_va, cov_va)
    for ki, kind in enumerate(NONGRAD_TYPES + ("fgsm",)):
        for intensity in INTENSITIES:
            setting = f"{kind}_{intensity:.1f}"
            if kind == "fgsm":
                ctx_va_da, _ = linf_fgsm(ctx_va, g_va_da, intensity)
                ctx_va_qt, _ = linf_fgsm(ctx_va, g_va_qt, intensity)
            else:
                rng_va = np.random.default_rng([SEED, 1, ki, int(10 * intensity)])
                ctx_va_adv, _ = apply_anomaly(ctx_va, kind, intensity, rng_va)
                ctx_va_da = ctx_va_qt = ctx_va_adv
            yseq_adv = yseq_va.copy()
            yseq_adv[:, :L] = ctx_va_da
            q_da_va, _ = E.deepar_predict(deepar, deepar_cfg, yseq_adv, cov_va, QUANTILES)
            x_adv = x_va.copy()
            x_adv[:, :, 0] = ctx_va_qt
            q_qt_va = E.qtransformer_predict(qt, x_adv)
            dump("deepar", "val", setting, y_true=yv_true, quantiles=inv(q_da_va),
                 levels=QUANTILES, context=ctx_va_da)
            dump("qtransformer", "val", setting, y_true=yv_true, quantiles=inv(q_qt_va),
                 levels=QUANTILES, context=ctx_va_qt)

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

    # Frozen test-set dumps (clean) + helper reused for the detect-and-clean
    # (hampel input repair) variants, which need fresh inference.
    dump("lstm", "test", "clean", y_true=y_true, point=inv(p_clean_lstm), context=ctx_clean)
    dump("naive_seasonal", "test", "clean", y_true=y_true,
         point=inv(_naive_forecast_batch(ctx_clean, H)), context=ctx_clean)
    dump("deepar", "test", "clean", y_true=y_true, quantiles=inv(q_clean_deepar),
         levels=QUANTILES, context=ctx_clean)
    dump("qtransformer", "test", "clean", y_true=y_true, quantiles=inv(q_clean_qt),
         levels=QUANTILES, context=ctx_clean)

    def predict_and_dump_all(setting, ctx_l, ctx_q, ctx_d, include_naive=True):
        """Inference for every model on its (repaired) context + dump."""
        p_ = E.lstm_predict(lstm, ctx_l)
        dump("lstm", "test", setting, y_true=y_true, point=inv(p_), context=ctx_l)
        if include_naive:
            dump("naive_seasonal", "test", setting, y_true=y_true,
                 point=inv(_naive_forecast_batch(ctx_l, H)), context=ctx_l)
        yseq_ = yseq_te.copy()
        yseq_[:, :L] = ctx_d
        q_d_, _ = E.deepar_predict(deepar, deepar_cfg, yseq_, cov_te, QUANTILES)
        dump("deepar", "test", setting, y_true=y_true, quantiles=inv(q_d_),
             levels=QUANTILES, context=ctx_d)
        x_ = x_te.copy()
        x_[:, :, 0] = ctx_q
        q_q_ = E.qtransformer_predict(qt, x_)
        dump("qtransformer", "test", setting, y_true=y_true, quantiles=inv(q_q_),
             levels=QUANTILES, context=ctx_q)

    print("Dumping hampel-cleaned clean baseline ...")
    _cl0 = hampel_clean(ctx_clean)
    predict_and_dump_all("clean__cleaned", _cl0, _cl0, _cl0)

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

            setting = f"{kind}_{intensity:.1f}"
            rec["realized_magnitude_scaled"] = mag
            rec["realized_magnitude_degC"] = mag * std_t

            # LSTM
            p = E.lstm_predict(lstm, ctx_lstm)
            rec["lstm"] = _point_scores(y_true_flat, inv(p).reshape(-1))
            dump("lstm", "test", setting, y_true=y_true, point=inv(p), context=ctx_lstm)

            # naive_seasonal (untrained, deterministic): shares the same injected
            # context as LSTM for the non-gradient anomalies; FGSM is white-box and
            # undefined for a non-differentiable model -> recorded as N/A.
            if kind == "fgsm":
                rec["naive_seasonal"] = {"rmse": None, "mae": None}
            else:
                p_naive = inv(_naive_forecast_batch(ctx_lstm, H))
                rec["naive_seasonal"] = _point_scores(y_true_flat, p_naive.reshape(-1))
                dump("naive_seasonal", "test", setting, y_true=y_true, point=p_naive,
                     context=ctx_lstm)

            # DeepAR
            yseq_adv = yseq_te.copy()
            yseq_adv[:, :L] = ctx_da
            q_da, _ = E.deepar_predict(deepar, deepar_cfg, yseq_adv, cov_te, QUANTILES)
            rec["deepar"] = prob_block(inv(q_da), "deepar")
            dump("deepar", "test", setting, y_true=y_true, quantiles=inv(q_da),
                 levels=QUANTILES, context=ctx_da)

            # Quantile Transformer
            x_adv = x_te.copy()
            x_adv[:, :, 0] = ctx_qt
            q_qt = E.qtransformer_predict(qt, x_adv)
            rec["qtransformer"] = prob_block(inv(q_qt), "qtransformer")
            dump("qtransformer", "test", setting, y_true=y_true, quantiles=inv(q_qt),
                 levels=QUANTILES, context=ctx_qt)

            # Detect-and-clean: hampel-repaired contexts, fresh inference.
            if ctx_lstm is ctx_da:
                _cl = hampel_clean(ctx_lstm)
                ctx_cls = (_cl, _cl, _cl)
            else:
                ctx_cls = (hampel_clean(ctx_lstm), hampel_clean(ctx_qt), hampel_clean(ctx_da))
            predict_and_dump_all(f"{setting}__cleaned", *ctx_cls,
                                 include_naive=(kind != "fgsm"))

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
