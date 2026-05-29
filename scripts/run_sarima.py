"""Run the SARIMA control model on Jena Climate.

SARIMA is a *control* for the plain ARIMA baseline (see
``src/baselines/sarima_baseline.py``): same non-seasonal order (2,1,2) plus a
daily seasonal order (1,0,1,24). Comparing this JSON against ``arima.json``
isolates the contribution of the seasonal component.

Seasonal s=24 fits are slow, so we evaluate the same 30-day test subset as the
ARIMA runner but fit on a shorter recent window. Reports include sMAPE/MASE in
addition to the Phase-1 RMSE/MAE/MAPE so the control is directly comparable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.sarima_baseline import SarimaConfig, rolling_sarima_predictions  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.metrics import mase, report, smape  # noqa: E402
from src.preprocessing import TARGET, chronological_split  # noqa: E402

ORDER = (2, 1, 2)
SEASONAL_ORDER = (1, 0, 1, 24)
HORIZON = 24
TEST_SUBSET_HOURS = 24 * 30  # first 30 days of test (matches run_arima.py)


def main() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    splits = chronological_split(df)

    train = splits.y_train()
    test = splits.y_test()[:TEST_SUBSET_HOURS]

    config = SarimaConfig(
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        horizon=HORIZON,
        refit_every=50,
        window=2000,
    )
    print(
        f"Running SARIMA{ORDER}x{SEASONAL_ORDER} on train (n={len(train)}) "
        f"-> test_subset (n={len(test)})"
    )
    y_true, y_pred = rolling_sarima_predictions(train, test, config)
    metrics = report(y_true, y_pred)
    metrics["smape"] = smape(y_true, y_pred)
    metrics["mase"] = mase(y_true, y_pred, train, season=24)
    metrics.update(
        model="sarima",
        order=list(ORDER),
        seasonal_order=list(SEASONAL_ORDER),
        target=TARGET,
        horizon=HORIZON,
        refit_every=config.refit_every,
        window=config.window,
        test_subset_hours=TEST_SUBSET_HOURS,
        n_predictions=int(len(y_true)),
        role="seasonal-component control for ARIMA(2,1,2)",
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "sarima.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
