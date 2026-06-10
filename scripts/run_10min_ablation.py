"""10-minute resolution ablation: does 6x finer data help, or is hourly enough?

Trains a qLSTM on the native 10-min Jena series (lookback 1008 steps = 168h,
horizon 144 steps = 24h) and evaluates it on an HOURLY-EQUIVALENT grid so the
score is directly comparable to the hourly qLSTM: the 144 ten-minute median
predictions are averaged in blocks of 6 to 24 hourly means, then scored
against the hourly ground truth.

Hypothesis (from the lookback plateau + HPO saturation): marginal or negative
-- consecutive 10-min samples are highly autocorrelated, so "6x data" is not
"6x independent information", and the 1008-step sequence is harder to learn
than 168 steps. A clean negative closes the "why not 10-min data" question.

Cost control: stride=6 (one window per hour, ~52K windows like the hourly
run) so only the sequence length (not the window count) grows.

  python scripts/run_10min_ablation.py
  python scripts/run_10min_ablation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw  # noqa: E402
from src.metrics import report, report_probabilistic  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.preprocessing import Standardizer, TARGET, TRAIN_END, VAL_END  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_PER_HOUR = 6
L10, H10 = 168 * STEPS_PER_HOUR, 24 * STEPS_PER_HOUR  # 1008, 144


def load_10min() -> pd.DataFrame:
    """Native 10-min series with the same cleaning as to_hourly (no resample to 1h)."""
    df = load_raw(ROOT / "data" / "raw" / "jena_climate_2009_2016.csv")
    for col in ("wv (m/s)", "max. wv (m/s)"):
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    df = df.resample("10min").mean().loc[:"2016-12-31 23:50:00"]
    return df.interpolate(method="time").bfill().ffill()


def windows(series: np.ndarray, lookback: int, horizon: int, stride: int):
    n = len(series) - lookback - horizon + 1
    idx = np.arange(0, n, stride)
    x = np.stack([series[i:i + lookback] for i in idx]).astype(np.float32)
    y = np.stack([series[i + lookback:i + lookback + horizon] for i in idx]).astype(np.float32)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    df = load_10min()
    t = df[[TARGET]]
    train = t.loc[:TRAIN_END]
    val = t.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(minutes=10):VAL_END]
    test = t.loc[pd.Timestamp(VAL_END) + pd.Timedelta(minutes=10):]
    print(f"10-min rows: train {len(train)} val {len(val)} test {len(test)}")

    scaler = Standardizer.fit(train)
    tr = scaler.transform(train).to_numpy().astype(np.float32)[:, 0]
    va = scaler.transform(val).to_numpy().astype(np.float32)[:, 0]
    te = scaler.transform(test).to_numpy().astype(np.float32)[:, 0]

    x_tr, y_tr = windows(tr, L10, H10, stride=STEPS_PER_HOUR)
    x_va, y_va = windows(va, L10, H10, stride=H10)
    x_te, y_te = windows(te, L10, H10, stride=H10)
    if args.smoke:
        x_tr, y_tr = x_tr[:1500], y_tr[:1500]
        x_te, y_te = x_te[:40], y_te[:40]

    cfg = QLstmConfig(lookback=L10, horizon=H10, batch_size=64)
    if args.smoke:
        cfg.epochs = 1
    print(f"Training 10-min qLSTM (lookback {L10}, horizon {H10}) ...")
    model, _ = train_qlstm(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE)

    inv = lambda a: scaler.inverse_target(a, TARGET)  # noqa: E731
    q10 = inv(predict_qlstm(model, x_te, device=DEVICE))     # (N, 144, Q)
    y10 = inv(y_te)                                           # (N, 144)

    # hourly-equivalent: average each 6-step block -> (N, 24[, Q])
    N = len(y10)
    q_hourly = q10.reshape(N, 24, STEPS_PER_HOUR, len(QUANTILES)).mean(axis=2)
    y_hourly = y10.reshape(N, 24, STEPS_PER_HOUR).mean(axis=2)
    med = q_hourly[..., int(np.argmin(np.abs(QUANTILES - 0.5)))]

    r = report_probabilistic(y_hourly.reshape(-1), q_hourly.reshape(-1, len(QUANTILES)),
                             QUANTILES, alpha=ALPHA)
    point = report(y_hourly.reshape(-1), med.reshape(-1))

    hourly_ref = json.loads((ROOT / "results" / "base" / "qlstm.json").read_text())
    out = {
        "resolution": "10min", "lookback_steps": L10, "horizon_steps": H10,
        "eval": "hourly-equivalent (6-step block mean), comparable to qlstm.json",
        "smoke": bool(args.smoke),
        "10min_hourly_equiv": {"rmse": point["rmse"], "crps": r["crps"], "picp": r["picp"]},
        "hourly_reference_qlstm": {"rmse": hourly_ref["rmse"], "crps": hourly_ref["crps"],
                                   "picp": hourly_ref["picp"]},
        "note": "quantile-block-averaging narrows the predictive spread, so PICP "
                "is only approximately comparable; RMSE is the clean comparison.",
    }
    out_path = ROOT / "results" / "base" / "ablation_10min.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n=== 10-min (hourly-equiv) vs hourly qLSTM ===")
    print(f"  10-min : RMSE {point['rmse']:.3f}  CRPS {r['crps']:.3f}  PICP {r['picp']:.3f}")
    print(f"  hourly : RMSE {hourly_ref['rmse']:.3f}  CRPS {hourly_ref['crps']:.3f}  PICP {hourly_ref['picp']:.3f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
