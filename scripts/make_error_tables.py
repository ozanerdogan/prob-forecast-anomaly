"""Full-roster error breakdowns from frozen predictions (no model runs).

Builds the three report tables for EVERY model that has a clean-test dump,
reusing the phase-1 error-analysis primitives:
  1. per-horizon  : RMSE at each of the 24 forecast steps
  2. by temperature: RMSE in the <0 / 0-10 / 10-20 / >20 degC bins
  3. by season    : RMSE in DJF / MAM / JJA / SON

The earlier results/base/error_analysis.json covered only LSTM + the two
probabilistic models; this generalises it to the whole roster directly from
results/predictions/ (probabilistic models contribute their median).

Writes results/base/error_tables_full.json and report/tables/*.md.

  python scripts/make_error_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.error_analysis import (  # noqa: E402
    per_horizon_point,
    season_breakdown,
    target_months,
    temperature_breakdown,
)
from src.predictions_io import load_predictions, prediction_path  # noqa: E402

PRED = ROOT / "results" / "predictions"
TABLES = ROOT / "report" / "tables"
L, H, STRIDE = 168, 24, 24
SEASON_ORDER = ("DJF", "MAM", "JJA", "SON")
TEMP_ORDER = ("<0", "0-10", "10-20", ">20")
ROSTER = ("naive_seasonal", "lstm", "gru", "dlinear", "deepar", "qtransformer",
          "qtransformer_multi", "qlstm", "qdlinear", "lgbm", "qrf", "qlstm_robust")


def _point(d: dict) -> np.ndarray:
    if "point" in d:
        return np.asarray(d["point"], dtype=float)
    levels = np.asarray(d["levels"])
    med = int(np.argmin(np.abs(levels - 0.5)))
    return np.asarray(d["quantiles"], dtype=float)[..., med]


def main() -> None:
    # test index only (for season months); prepare() does not train anything
    data = E.prepare(use_covariates=False)
    test_index = data.test_index

    out = {"per_horizon": {}, "by_temperature": {}, "by_season": {},
           "temp_order": list(TEMP_ORDER), "season_order": list(SEASON_ORDER)}
    n_windows = None
    for m in ROSTER:
        p = prediction_path(PRED, m, "test", "clean")
        if not p.exists():
            continue
        d = load_predictions(p)
        y = np.asarray(d["y_true"], dtype=float)
        yhat = _point(d)
        n_windows = len(y)
        months = target_months(test_index, n_windows, L, H, STRIDE)
        out["per_horizon"][m] = per_horizon_point(y, yhat)["rmse"]
        out["by_temperature"][m] = {k: v["rmse"]
                                    for k, v in temperature_breakdown(y, yhat).items()}
        out["by_season"][m] = {k: v["rmse"]
                               for k, v in season_breakdown(y, yhat, months).items()}

    BASE = ROOT / "results" / "base"
    (BASE / "error_tables_full.json").write_text(json.dumps(out, indent=2))

    # ---- markdown ------------------------------------------------------- #
    TABLES.mkdir(parents=True, exist_ok=True)
    models = list(out["per_horizon"])

    # per-horizon: rows = model, cols = a few representative steps + overall
    steps = [1, 6, 12, 18, 24]
    lines = ["# Per-horizon RMSE (clean test, hourly steps)\n",
             "| Model | " + " | ".join(f"h{s}" for s in steps) + " | mean |",
             "|---|" + "---|" * (len(steps) + 1)]
    for m in sorted(models, key=lambda m: np.mean(out["per_horizon"][m])):
        ph = out["per_horizon"][m]
        cells = [f"{ph[s-1]:.2f}" for s in steps] + [f"{np.mean(ph):.2f}"]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (TABLES / "per_horizon_full.md").write_text("\n".join(lines) + "\n")

    # by temperature
    lines = ["# RMSE by temperature range (clean test, degC)\n",
             "| Model | " + " | ".join(TEMP_ORDER) + " |",
             "|---|" + "---|" * len(TEMP_ORDER)]
    for m in sorted(models, key=lambda m: np.mean(list(out["by_temperature"][m].values()))):
        cells = [f"{out['by_temperature'][m].get(t, float('nan')):.2f}" for t in TEMP_ORDER]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (TABLES / "by_temperature_full.md").write_text("\n".join(lines) + "\n")

    # by season
    lines = ["# RMSE by season (clean test)\n",
             "| Model | " + " | ".join(SEASON_ORDER) + " |",
             "|---|" + "---|" * len(SEASON_ORDER)]
    for m in sorted(models, key=lambda m: np.mean(list(out["by_season"][m].values()))):
        cells = [f"{out['by_season'][m].get(s, float('nan')):.2f}" for s in SEASON_ORDER]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (TABLES / "by_season_full.md").write_text("\n".join(lines) + "\n")

    print(f"roster covered: {len(models)} models, {n_windows} windows")
    print(f"tables -> {TABLES}/{{per_horizon,by_temperature,by_season}}_full.md")
    print(f"json   -> results/base/error_tables_full.json")


if __name__ == "__main__":
    main()
