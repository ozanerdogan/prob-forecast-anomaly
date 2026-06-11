"""Run the ARIMA baseline on the full Jena Climate test split.

Rolling-origin over the entire 2016 test year (every hourly origin; refit every
``refit_every`` origins on a recent window -- see arima_baseline.py), so ARIMA is
evaluated on the same full-year period as the other baselines. Runtime ~15-20 min.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.baselines.arima_baseline import ArimaConfig, rolling_arima_predictions  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.metrics import mase, report, smape  # noqa: E402
from src.preprocessing import TARGET, chronological_split  # noqa: E402

ORDER = (2, 1, 2)
HORIZON = 24


def main() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    splits = chronological_split(df)

    train = splits.y_train()
    test = splits.y_test()

    config = ArimaConfig(order=ORDER, horizon=HORIZON, refit_every=50)
    print(f"Running ARIMA{ORDER} on train (n={len(train)}) -> full test (n={len(test)})")
    y_true, y_pred = rolling_arima_predictions(train, test, config)
    metrics = report(y_true, y_pred)
    metrics["smape"] = smape(y_true, y_pred)
    metrics["mase"] = mase(y_true, y_pred, train, season=24)
    metrics.update(
        model="arima",
        order=list(ORDER),
        target=TARGET,
        horizon=HORIZON,
        refit_every=config.refit_every,
        n_test_hours=int(len(test)),
        n_predictions=int(len(y_true)),
    )
    out_dir = ROOT / "results" / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "arima.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
