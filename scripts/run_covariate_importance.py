"""Full covariate-importance table for temperature forecasting.

Trains one multivariate quantile-Transformer on the target plus ALL 13
exogenous Jena variables (leakage-free: the QT encoder never reads horizon
covariates), then ranks each covariate by permutation importance -- the CRPS
/ median-RMSE increase when that channel is shuffled across windows (breaking
its alignment with the target while keeping its marginal). Calendar channels
are included for reference.

Caveat the table makes explicit: several variables are thermodynamic
near-duplicates of temperature (Tpot, Tdew, and to a lesser degree
VPmax/VPact/sh/H2OC/rho). High importance for those is partly trivial
(they ARE temperature); the genuinely independent sensors (p, rh, wv,
max.wv, wd) answer "which *different* measurement helps".

  python scripts/run_covariate_importance.py
  python scripts/run_covariate_importance.py --smoke
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
from src.data_loader import load_hourly  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src.metrics import report_probabilistic  # noqa: E402
from src.models.quantile_transformer import QTransformerConfig, QUANTILES_7  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
SEED = 42

# thermodynamic near-duplicates of T (partial leakage); the rest are
# genuinely independent meteorological sensors.
TEMP_DERIVED = {"Tpot (K)", "Tdew (degC)", "VPmax (mbar)", "VPact (mbar)",
                "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)"}
# genuinely independent sensors (T not analytically recoverable from them)
INDEPENDENT = ["p (mbar)", "rh (%)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--covset", choices=("all", "independent"), default="all",
                    help="'all' = 13 exogenous (incl. T-derived proxies); "
                         "'independent' = only p/rh/wv/max.wv/wd (no T leakage)")
    args = ap.parse_args()

    df = load_hourly(ROOT / "data" / "processed")
    all_cov = ([c for c in df.columns if c != TARGET] if args.covset == "all"
               else list(INDEPENDENT))
    print(f"covset={args.covset} | covariates ({len(all_cov)}): {all_cov}")

    data = E.prepare(use_covariates=True, covariate_cols=all_cov)
    cols = list(build_feature_frame(df, TARGET, use_covariates=True,
                                    covariate_cols=all_cov).columns)
    cfg = QTransformerConfig(n_features=data.n_features, quantiles=QUANTILES_7)
    if args.smoke:
        cfg.epochs = 1

    print(f"Training QT on {data.n_features} channels ({cfg.epochs} epochs) ...")
    model = E.fit_qtransformer(data, cfg)

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    L, H = cfg.lookback, cfg.horizon

    def scored(x, y):
        q = inv(E.qtransformer_predict(model, x)).reshape(-1, len(QUANTILES))
        r = report_probabilistic(inv(y).reshape(-1), q, QUANTILES, alpha=ALPHA)
        return r["crps"], r["rmse"]

    out = {"n_channels": data.n_features, "channels": cols,
           "method": "permutation importance (channel shuffled across windows, seed 42)",
           "temp_derived_note": "Tpot/Tdew/VP*/sh/H2OC/rho are thermodynamic "
                                "near-duplicates of T (partial leakage)",
           "splits": {}}

    for split_name, arr in (("val", data.val), ("test", data.test)):
        x, y = make_encoder_windows(arr, L, H, stride=H)
        if args.smoke:
            x, y = x[:60], y[:60]
        base_crps, base_rmse = scored(x, y)
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(x))
        rows = {}
        for c in range(0, x.shape[2]):  # include channel 0 (target T) as reference
            xp = x.copy()
            xp[:, :, c] = x[perm, :, c]
            crps, rmse = scored(xp, y)
            name = cols[c]
            rows[name] = {
                "delta_crps": crps - base_crps, "delta_rmse": rmse - base_rmse,
                "is_target": c == 0,
                "is_calendar": name.startswith(("hour_", "doy_")),
                "is_temp_derived": name in TEMP_DERIVED,
            }
        out["splits"][split_name] = {
            "base_crps": base_crps, "base_rmse": base_rmse, "permuted": rows}
        print(f"\n=== {split_name} (base CRPS {base_crps:.3f}, RMSE {base_rmse:.2f}) ===")
        for n, v in sorted(rows.items(), key=lambda kv: -kv[1]["delta_crps"]):
            tag = ("[HEDEF T]" if v["is_target"] else "[T-türevi]" if v["is_temp_derived"]
                   else "[takvim]" if v["is_calendar"] else "[bağımsız]")
            print(f"  {n:18s} {tag:11s} dCRPS {v['delta_crps']:+.4f}  dRMSE {v['delta_rmse']:+.4f}")

    suffix = "" if args.covset == "all" else "_independent"
    out_path = ROOT / "results" / "base" / f"covariate_importance{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
