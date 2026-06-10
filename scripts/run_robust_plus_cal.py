"""Phase-4 combined study: model-side (robust training) x interval-side
(adaptive calibration). Four corners, all from frozen dumps.

Phase 3 showed robust training repairs the point forecast and ACI repairs the
interval. The open question: are they complementary, redundant, or does
calibrating an already-robust model overshoot? We score the same level-shift
settings under:

    normal  raw         normal  + ACI
    robust  raw         robust  + ACI   (and + input_tau)

Reads qlstm (normal) and qlstm_robust dumps; applies the calibrators fit on
each model's own validation dumps (so robust gets a robust-fit calibrator).
No model runs.

  python scripts/run_robust_plus_cal.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrators import ACITau, InputTau, interval_metrics  # noqa: E402
from src.predictions_io import list_settings, load_predictions, prediction_path  # noqa: E402

PRED = ROOT / "results" / "predictions"
ALPHA = 0.1
SETTINGS = ("clean", "level_shift_1.0", "level_shift_2.0", "level_shift_4.0",
            "fgsm_2.0", "fgsm_4.0")


def _fit_calibrators(model):
    levels = load_predictions(prediction_path(PRED, model, "val", "clean"))["levels"]
    dvc = load_predictions(prediction_path(PRED, model, "val", "clean"))
    aci = ACITau(ALPHA).fit(dvc["y_true"], dvc["quantiles"], levels)
    # input_tau needs all val settings
    fit = [s for s in list_settings(PRED, model, "val") if not s.endswith("__cleaned")]
    ys, qs, cs = [], [], []
    for s in fit:
        dv = load_predictions(prediction_path(PRED, model, "val", s))
        ys.append(dv["y_true"]); qs.append(dv["quantiles"]); cs.append(dv["context"])
    itau = InputTau(ALPHA).fit(np.concatenate(ys), np.concatenate(qs), levels,
                              context_val=np.concatenate(cs))
    return levels, aci, itau


def _scores(model, setting, levels, cal=None):
    d = load_predictions(prediction_path(PRED, model, "test", setting))
    q = d["quantiles"] if cal is None else cal.apply(d["y_true"], d["quantiles"], levels,
                                                     context=d.get("context"))
    return interval_metrics(d["y_true"], np.asarray(q), levels, ALPHA)


def main():
    for needed in ("qlstm", "qlstm_robust"):
        if not prediction_path(PRED, needed, "test", "clean").exists():
            raise SystemExit(f"missing dumps for {needed} (run scripts/run_qlstm_robust.py)")

    levels_n, aci_n, itau_n = _fit_calibrators("qlstm")
    levels_r, aci_r, itau_r = _fit_calibrators("qlstm_robust")

    out = {"alpha": ALPHA, "settings": {}}
    for s in SETTINGS:
        out["settings"][s] = {
            "normal_raw": _scores("qlstm", s, levels_n),
            "normal_aci": _scores("qlstm", s, levels_n, aci_n),
            "robust_raw": _scores("qlstm_robust", s, levels_r),
            "robust_aci": _scores("qlstm_robust", s, levels_r, aci_r),
            "robust_input_tau": _scores("qlstm_robust", s, levels_r, itau_r),
        }
        r = out["settings"][s]
        print(f"  {s:18s} PICP  nraw {r['normal_raw']['picp']:.3f}  n+aci {r['normal_aci']['picp']:.3f}"
              f"  rraw {r['robust_raw']['picp']:.3f}  r+aci {r['robust_aci']['picp']:.3f}"
              f"  | MIS r+aci {r['robust_aci']['mis']:.1f}")

    out_path = ROOT / "results" / "base" / "robust_plus_cal.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
