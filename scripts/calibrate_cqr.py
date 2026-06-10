"""Stage-2 repair, conformalized quantile regression (Romano et al. 2019).

Split-conformal additive margin on the outer (5%, 95%) pair, fit on clean
validation scores. Distribution-free finite-sample coverage under
exchangeability — which anomalies deliberately break, so its failure mode
under shift is part of the comparison. Reads results/predictions/, writes
results/calibrated/cqr/<model>.json.

  python scripts/calibrate_cqr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calib_runner import main_for_method  # noqa: E402
from src.calibrators import CQRCalibrator  # noqa: E402

if __name__ == "__main__":
    main_for_method(CQRCalibrator, "cqr", ROOT)
