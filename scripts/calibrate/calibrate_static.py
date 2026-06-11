"""Stage-2 repair, static spread temperature (out-of-the-box baseline).

Fits one scalar tau per probabilistic model on the clean validation dump and
rescales every test setting's frozen quantiles around the median. Reads
results/predictions/, writes results/calibrated/static/<model>.json — no
model is run.

  python scripts/calibrate/calibrate_static.py
  python scripts/calibrate/calibrate_static.py --pred-dir results/predictions_smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.calib_runner import main_for_method  # noqa: E402
from src.calibrators import StaticTau  # noqa: E402

if __name__ == "__main__":
    main_for_method(StaticTau, "static", ROOT)
