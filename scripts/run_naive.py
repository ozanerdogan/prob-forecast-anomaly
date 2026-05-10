"""Run the Naive Seasonal baseline on the Jena Climate test split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.naive_seasonal import rolling_naive_predictions  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.metrics import report  # noqa: E402
from src.preprocessing import TARGET, chronological_split  # noqa: E402

HORIZON = 24
SEASON_LENGTH = 24


def main() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    splits = chronological_split(df)
    y_test = splits.y_test()
    y_true, y_pred = rolling_naive_predictions(y_test, HORIZON, SEASON_LENGTH)
    metrics = report(y_true, y_pred)
    metrics.update(
        model="naive_seasonal",
        target=TARGET,
        horizon=HORIZON,
        season_length=SEASON_LENGTH,
        n_predictions=int(len(y_true)),
    )
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "naive_seasonal.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
