"""Run the SARIMA control model on the full Jena Climate test split.

SARIMA is a *control* for the plain ARIMA baseline (see
``src/baselines/sarima_baseline.py``): same non-seasonal order (2,1,2) plus a
daily seasonal order (1,0,1,24). Comparing this JSON against ``arima.json``
probes the seasonal contribution -- but see the control caveat in
``sarima_baseline.py``: SARIMA fits a shorter window (2000h) than ARIMA (8000h),
so it is not a perfectly clean single-variable isolation.

Rolling-origin over the full 2016 test year; the seasonal s=24 fits make this the
slowest baseline (~45 min). Reports sMAPE/MASE alongside RMSE/MAE/MAPE.
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


def main() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    splits = chronological_split(df)

    train = splits.y_train()
    test = splits.y_test()

    config = SarimaConfig(
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        horizon=HORIZON,
        refit_every=50,
        window=2000,
    )
    print(
        f"Running SARIMA{ORDER}x{SEASONAL_ORDER} on train (n={len(train)}) "
        f"-> full test (n={len(test)})"
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
        n_test_hours=int(len(test)),
        n_predictions=int(len(y_true)),
        role="seasonal-component control for ARIMA(2,1,2)",
    )
    out_dir = ROOT / "results" / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sarima.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
