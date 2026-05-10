"""Run the ARIMA baseline on Jena Climate.

For Phase 1 we evaluate on a *subset* of the test split — a few hundred origins
sampled uniformly — to keep runtime manageable. Full-test ARIMA is deferred to
the final phase.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.arima_baseline import ArimaConfig, rolling_arima_predictions  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.metrics import report  # noqa: E402
from src.preprocessing import TARGET, chronological_split  # noqa: E402

ORDER = (2, 1, 2)
HORIZON = 24
TEST_SUBSET_HOURS = 24 * 30  # first 30 days of test


def main() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    splits = chronological_split(df)

    train = splits.y_train()
    test = splits.y_test()[:TEST_SUBSET_HOURS]

    config = ArimaConfig(order=ORDER, horizon=HORIZON, refit_every=50)
    print(f"Running ARIMA{ORDER} on train (n={len(train)}) -> test_subset (n={len(test)})")
    y_true, y_pred = rolling_arima_predictions(train, test, config)
    metrics = report(y_true, y_pred)
    metrics.update(
        model="arima",
        order=list(ORDER),
        target=TARGET,
        horizon=HORIZON,
        refit_every=config.refit_every,
        test_subset_hours=TEST_SUBSET_HOURS,
        n_predictions=int(len(y_true)),
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "arima.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
