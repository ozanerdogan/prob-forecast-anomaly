"""Stage-2 repair, online additive conformal margin (adaptive-CQR).

The online counterpart of CQR: warm-started on the offline split-conformal
margin, then the margin adapts to realised coverage over the test sequence
(window t uses only windows < t). Bridges CQR and ACI; reads
results/predictions/, writes results/calibrated/aci_margin/<model>.json.

  python scripts/calibrate/calibrate_aci_margin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.calib_runner import main_for_method  # noqa: E402
from src.calibrators import ACIMargin  # noqa: E402

if __name__ == "__main__":
    main_for_method(ACIMargin, "aci_margin", ROOT)
