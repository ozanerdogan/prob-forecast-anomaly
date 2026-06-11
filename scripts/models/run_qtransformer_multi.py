"""Multivariate quantile Transformer as a BASE model + permutation importance.

Phase-2 decision implemented here: the multivariate-base promotion is
per-model — the QT encoder is the model whose covariates demonstrably help
without leakage (ablation: CRPS 1.039 -> 0.901 at the shared budget), so it
gets the full-budget multivariate base run; DeepAR stays univariate (its
leakage-free past-covariate variant hurts — see the archived probe in cowork/3_arsiv/scripts/).

Anomaly stance ('Asama-1'): corruption hits the TARGET channel only, with the
same seeded streams as every other model, so rows remain directly comparable
with the univariate grid. Covariate channels stay clean.

Also computes the channel permutation-importance table (the cheap primary
feature-importance method): each covariate channel is shuffled across windows
(breaking its alignment, keeping its marginal), and the CRPS / median-RMSE
deltas are recorded on validation AND test -> results/base/feature_importance.json.

  python scripts/models/run_qtransformer_multi.py
  python scripts/models/run_qtransformer_multi.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src.metrics import mase, report_probabilistic, smape  # noqa: E402
from src.anomaly import FAULT_TYPES_V2  # noqa: E402
from src.model_eval import NONGRAD_V1, evaluate_and_dump  # noqa: E402
from src.models.quantile_transformer import QTransformerConfig, QUANTILES_7  # noqa: E402
from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
SEED = 42


def _scores(y_flat, q_flat):
    out = report_probabilistic(y_flat, q_flat, QUANTILES, alpha=ALPHA)
    return {"crps": out["crps"], "rmse_median": out["rmse"], "picp": out["picp"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--catalog", choices=("v1", "v2"), default="v1",
                        help="v2 adds the phase-2 fault families to the sweep")
    parser.add_argument("--covset", choices=("independent", "legacy"), default="independent",
                        help="independent (default, official) = the 5 genuinely "
                             "independent sensors (p/rh/wv/max.wv/wd); legacy = the "
                             "old p/rh/VPmax/wv set (VPmax is a redundant temperature "
                             "proxy, kept only for comparison)")
    args = parser.parse_args()
    nongrad = NONGRAD_V1 + (FAULT_TYPES_V2 if args.catalog == "v2" else ())

    # Official multivariate set: only genuinely independent sensors -- no VPmax
    # (a redundant temperature proxy), and the strong but previously unused
    # max.wv (+0.38) and wd added. Calendar features (incl. doy_cos) are added
    # automatically by build_feature_frame regardless of this list.
    indep = ["p (mbar)", "rh (%)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"]
    legacy = ["p (mbar)", "rh (%)", "VPmax (mbar)", "wv (m/s)"]
    cov_cols = legacy if args.covset == "legacy" else indep
    model_name = "qtransformer_multi_legacy" if args.covset == "legacy" else "qtransformer_multi"

    data = E.prepare(use_covariates=True, covariate_cols=cov_cols)
    cols = list(build_feature_frame(load_hourly(ROOT / "data" / "processed"),
                                    TARGET, use_covariates=True, covariate_cols=cov_cols).columns)
    cfg = QTransformerConfig(n_features=data.n_features, quantiles=QUANTILES_7)
    if args.smoke:
        cfg.epochs = 1

    print(f"Training multivariate QT ({data.n_features} channels, {cfg.epochs} epochs) ...")
    model = E.fit_qtransformer(data, cfg)

    predict_fn = lambda x: {"quantiles": E.qtransformer_predict(model, x)}  # noqa: E731
    grad_fn = lambda x, y: E.qtransformer_context_grad(model, x, y, QUANTILES)  # noqa: E731

    metrics = evaluate_and_dump(
        model_name, data, predict_fn, root=ROOT, grad_fn=grad_fn,
        quantiles=QUANTILES, nongrad_types=nongrad, smoke=args.smoke,
    )

    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")
    d = load_predictions(prediction_path(pred_dir, model_name, "test", "clean"))
    med = d["quantiles"][..., int(np.argmin(np.abs(QUANTILES - 0.5)))].reshape(-1)
    y_flat_t = d["y_true"].reshape(-1)
    metrics["smape"] = smape(y_flat_t, med)
    metrics["mase"] = mase(y_flat_t, med, data.train_target_raw, season=24)
    metrics.update(
        model=model_name, target=TARGET, channels=cols,
        lookback=cfg.lookback, horizon=cfg.horizon, epochs=cfg.epochs,
        quantiles=QUANTILES.tolist(), seed=SEED, smoke=bool(args.smoke),
        anomaly_stance="target-channel-only corruption (Asama-1)",
    )
    out = ROOT / "results" / "base" / f"{model_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"Saved -> {out}  (clean RMSE {metrics['rmse']:.3f}, CRPS {metrics['crps']:.3f})")

    # ------------------------------------------------------------------ #
    # Channel permutation importance (val + test).
    # ------------------------------------------------------------------ #
    print("Permutation importance ...")
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    L, H = cfg.lookback, cfg.horizon
    importance: dict = {"model": model_name, "channels": cols[1:],
                        "method": "channel shuffled across windows, seed 42",
                        "splits": {}}
    for split_name, arr in (("val", data.val), ("test", data.test)):
        x, y = make_encoder_windows(arr, L, H, stride=H)
        if args.smoke:
            x, y = x[:60], y[:60]
        y_flat = inv(y).reshape(-1)
        base_q = inv(E.qtransformer_predict(model, x)).reshape(-1, len(QUANTILES))
        base = _scores(y_flat, base_q)
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(x))
        rows = {}
        for c in range(1, x.shape[2]):
            xp = x.copy()
            xp[:, :, c] = x[perm, :, c]
            q = inv(E.qtransformer_predict(model, xp)).reshape(-1, len(QUANTILES))
            s = _scores(y_flat, q)
            rows[cols[c]] = {
                "crps": s["crps"], "delta_crps": s["crps"] - base["crps"],
                "rmse_median": s["rmse_median"],
                "delta_rmse": s["rmse_median"] - base["rmse_median"],
            }
            print(f"  {split_name:4s} {cols[c]:14s} dCRPS {rows[cols[c]]['delta_crps']:+.4f} "
                  f"dRMSE {rows[cols[c]]['delta_rmse']:+.4f}")
        importance["splits"][split_name] = {"base": base, "permuted": rows}

    fi_suffix = "_legacy" if args.covset == "legacy" else ""
    out_fi = ROOT / "results" / "base" / f"feature_importance{fi_suffix}.json"
    out_fi.write_text(json.dumps(importance, indent=2))
    print(f"Saved -> {out_fi}")


if __name__ == "__main__":
    main()
