"""Stage-2 repair, ACI-style online spread adaptation (Gibbs & Candes 2021).

Online protocol: window t's interval is repaired with the spread scale
adapted from windows < t only — exactly the feedback available in
deployment. gamma and the starting tau are tuned by replaying the rule over
the validation sequence (warm start); the test sequence is never used for
configuration. Reads results/predictions/, writes
results/calibrated/aci/<model>.json.

  python scripts/calibrate/calibrate_aci.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.calib_runner import main_for_method  # noqa: E402
from src.calibrators import ACITau  # noqa: E402

if __name__ == "__main__":
    main_for_method(ACITau, "aci", ROOT)
