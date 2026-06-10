"""Train + evaluate the LightGBM point/quantile forecaster (tree family).

Target-only, lookback=168, horizon=24, the shared 7-quantile set. Produces
the same probabilistic + point score block as run_qtransformer.py, and dumps
frozen predictions (val clean + injected, test clean + injected +
hampel-cleaned) into results/predictions/ so the stage-2 calibrators pick the
model up as a third, structurally different calibration subject (tree
quantiles vs sampled-AR vs pinball-Transformer).

Anomaly contexts replicate run_anomaly_eval.py's rng streams exactly, so
every model sees the *same* corrupted inputs. No FGSM settings here: trees
have no gradient (white-box FGSM undefined); the transfer-FGSM column arrives
with the Phase-4 rerun wave.

  python scripts/run_lgbm.py            # full run
  python scripts/run_lgbm.py --smoke    # tiny end-to-end sanity run
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
from src.anomaly import apply_anomaly  # noqa: E402
from src.cleaning import hampel_clean  # noqa: E402
from src.metrics import mase, report, report_probabilistic, smape  # noqa: E402
from src.models.lgbm_quantile import (  # noqa: E402
    LgbmConfig,
    QUANTILES_7,
    predict_lgbm,
    train_lgbm,
)
from src.predictions_io import prediction_path, save_predictions  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
INTENSITIES = (1.0, 2.0, 4.0)
NONGRAD_TYPES = ("point_spike", "contextual_outlier", "level_shift")
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny fast sanity run")
    args = parser.parse_args()

    cfg = LgbmConfig()
    data = E.prepare(use_covariates=False)
    L, H = cfg.lookback, cfg.horizon

    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=cfg.train_stride)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        cfg.n_estimators = 30
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]
        sl = slice(0, 60)
        x_va, y_va, x_te, y_te = x_va[sl], y_va[sl], x_te[sl], y_te[sl]
    print(f"shapes  train {x_tr.shape} val {x_va.shape} test {x_te.shape}")

    print("Training LightGBM (1 point + 7 quantile boosters) ...")
    models = train_lgbm(x_tr[:, :, 0], y_tr, cfg)

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    ctx_te = x_te[:, :, 0].copy()
    ctx_va = x_va[:, :, 0].copy()
    y_true = inv(y_te)
    y_flat = y_true.reshape(-1)

    p_te_z, q_te_z = predict_lgbm(models, ctx_te, cfg)
    p_te, q_te = inv(p_te_z), inv(q_te_z)

    metrics = report_probabilistic(y_flat, q_te.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
    point_flat = p_te.reshape(-1)
    point_block = report(y_flat, point_flat)
    metrics["point_rmse"] = point_block["rmse"]
    metrics["point_mae"] = point_block["mae"]
    metrics["point_smape"] = smape(y_flat, point_flat)
    metrics["point_mase"] = mase(y_flat, point_flat, data.train_target_raw, season=24)
    metrics.update(
        model="lgbm",
        target=TARGET,
        lookback=L,
        horizon=H,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        train_stride=cfg.train_stride,
        quantiles=QUANTILES.tolist(),
        n_predictions=int(len(y_flat)),
        seed=cfg.seed,
        smoke=bool(args.smoke),
    )

    out_dir = ROOT / "results" / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lgbm.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved -> {out_path}")

    # ----------------------------------------------------------------- #
    # Frozen-forecast dumps (same store + rng streams as run_anomaly_eval)
    # ----------------------------------------------------------------- #
    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")

    def dump(split, setting, ctx, y_arr):
        p_z, q_z = predict_lgbm(models, ctx, cfg)
        save_predictions(
            prediction_path(pred_dir, "lgbm", split, setting),
            y_true=y_arr, quantiles=inv(q_z), levels=QUANTILES, point=inv(p_z),
            context=ctx,
            meta={"model": "lgbm", "split": split, "setting": setting,
                  "alpha": ALPHA, "seed": SEED, "smoke": bool(args.smoke),
                  "fgsm": "N/A (no gradient; transfer protocol in phase-4 wave)",
                  "units": "physical degC (y_true/quantiles/point), standardised (context)"},
        )

    yv_true = inv(y_va)
    print("Dumping predictions (val/test, clean + injected + cleaned) ...")
    dump("val", "clean", ctx_va, yv_true)
    for ki, kind in enumerate(NONGRAD_TYPES):  # ki matches run_anomaly_eval order
        for intensity in INTENSITIES:
            rng_va = np.random.default_rng([SEED, 1, ki, int(10 * intensity)])
            ctx_adv, _ = apply_anomaly(ctx_va, kind, intensity, rng_va)
            dump("val", f"{kind}_{intensity:.1f}", ctx_adv, yv_true)

    dump("test", "clean", ctx_te, y_true)
    dump("test", "clean__cleaned", hampel_clean(ctx_te), y_true)
    for kind in NONGRAD_TYPES:
        for intensity in INTENSITIES:
            rng = np.random.default_rng(SEED)  # same stream as run_anomaly_eval
            ctx_adv, _ = apply_anomaly(ctx_te, kind, intensity, rng)
            setting = f"{kind}_{intensity:.1f}"
            dump("test", setting, ctx_adv, y_true)
            dump("test", f"{setting}__cleaned", hampel_clean(ctx_adv), y_true)
    print(f"dumps -> {pred_dir}")


if __name__ == "__main__":
    main()
