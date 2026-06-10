"""How much does past temperature itself buy? Three-way input ablation.

Trains a quantile-Transformer that NEVER sees past temperature: the target
channel is dropped from the input, leaving only the calendar features and the
13 exogenous weather variables. The 24h-ahead temperature is still the target,
so this asks "can we forecast temperature from everything EXCEPT its own
history?". Compared against the target-only and full (T + covariates) models
already trained, this isolates the value of the autoregressive temperature
signal.

  target-only  : input = [T]                  -> results/base/qtransformer.json
  exogenous     : input = [calendar + 13 cov]  -> this script (T removed)
  full          : input = [T + calendar + 13]  -> covariate_importance_full.json

  python scripts/run_exogenous_only.py
  python scripts/run_exogenous_only.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.metrics import report_probabilistic  # noqa: E402
from src.models.quantile_transformer import (  # noqa: E402
    QTransformerConfig,
    QUANTILES_7,
    predict_quantiles,
    train_qtransformer,
)
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Variables from which temperature is NOT analytically recoverable: pressure,
# relative humidity, wind speed/gust/direction. The thermodynamic family
# (Tpot, Tdew, VP*, sh, H2OC, rho) leaks T almost exactly (VPmax->T RMSE
# 0.05 degC, Tpot+p->T 0.01 degC -- see the leakage check), so an honest
# "no temperature" input must exclude them.
TRULY_INDEPENDENT = ["p (mbar)", "rh (%)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--covset", choices=("all", "independent"), default="all",
                    help="'all' = 13 exogenous incl. temperature-derived proxies "
                         "(LEAKS T); 'independent' = only p/rh/wind (no T leakage)")
    args = ap.parse_args()

    df = load_hourly(ROOT / "data" / "processed")
    all_cov = ([c for c in df.columns if c != TARGET] if args.covset == "all"
               else list(TRULY_INDEPENDENT))
    data = E.prepare(use_covariates=True, covariate_cols=all_cov)
    L, H = 168, 24

    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]

    # drop the target channel (column 0) -> exogenous + calendar only
    x_tr_e, x_va_e, x_te_e = x_tr[:, :, 1:], x_va[:, :, 1:], x_te[:, :, 1:]
    n_feat = x_tr_e.shape[2]
    print(f"Exogenous-only input: {n_feat} channels (T removed), target still T")

    cfg = QTransformerConfig(n_features=n_feat, quantiles=QUANTILES_7)
    if args.smoke:
        cfg.epochs = 1
    model, _ = train_qtransformer(x_tr_e, y_tr, x_va_e, y_va, cfg, device=DEVICE)

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    q = inv(predict_quantiles(model, x_te_e, device=DEVICE)).reshape(-1, len(QUANTILES))
    y_flat = inv(y_te).reshape(-1)
    r = report_probabilistic(y_flat, q, QUANTILES, alpha=ALPHA)

    # reference numbers from already-trained models
    refs = {}
    for tag, path in (("target_only", "qtransformer.json"),
                      ("full_T_plus_cov", "covariate_importance_full.json")):
        p = ROOT / "results" / "base" / path
        if p.exists():
            d = json.loads(p.read_text())
            if tag == "full_T_plus_cov":
                refs[tag] = {"crps": d["splits"]["test"]["base_crps"],
                             "rmse": d["splits"]["test"]["base_rmse"]}
            else:
                refs[tag] = {"crps": d["crps"], "rmse": d["rmse"], "picp": d["picp"]}
    # naive reference
    nv = ROOT / "results" / "base" / "naive_seasonal.json"
    if nv.exists():
        refs["naive_seasonal"] = {"rmse": json.loads(nv.read_text())["rmse"]}

    out = {
        "covset": args.covset,
        "covariates_used": all_cov,
        "leaks_temperature": args.covset == "all",
        "exogenous_only": {"crps": r["crps"], "rmse": r["rmse"], "picp": r["picp"],
                           "n_input_channels": n_feat},
        "references": refs,
        "interpretation": ("covset=all LEAKS T via VPmax/Tpot (RMSE matches target-only "
                           "trivially); covset=independent is the honest no-temperature "
                           "test -- if far above naive, the independent sensors alone "
                           "cannot forecast temperature"),
        "smoke": bool(args.smoke),
    }
    suffix = "" if args.covset == "all" else "_independent"
    out_path = ROOT / "results" / "base" / f"exogenous_only{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n=== Üç-yönlü girdi kıyası (test CRPS / RMSE) ===")
    print(f"  target-only (sadece T) : CRPS {refs.get('target_only',{}).get('crps','?')}  "
          f"RMSE {refs.get('target_only',{}).get('rmse','?')}")
    print(f"  exogenous (T YOK)      : CRPS {r['crps']:.3f}  RMSE {r['rmse']:.2f}")
    print(f"  full (T + 13 cov)      : CRPS {refs.get('full_T_plus_cov',{}).get('crps','?')}  "
          f"RMSE {refs.get('full_T_plus_cov',{}).get('rmse','?')}")
    print(f"  naive (referans)       : RMSE {refs.get('naive_seasonal',{}).get('rmse','?')}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
