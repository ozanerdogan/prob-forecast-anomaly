"""Stage-2 repair, offline input-conditional spread scale.

A small regressor maps cheap context statistics (tail dispersion, last jump,
tail-vs-earlier level gap, ...) to the per-window required spread scale. It
is fit ONLY on validation windows — clean plus the synthetically injected
validation settings — so no test feedback of any kind is used: at test time
the calibrator looks at the input context alone. Reads results/predictions/,
writes results/calibrated/input_tau/<model>.json.

  python scripts/calibrate_input_tau.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calib_runner import main_for_method  # noqa: E402
from src.calibrators import InputTau  # noqa: E402

if __name__ == "__main__":
    main_for_method(InputTau, "input_tau", ROOT)
