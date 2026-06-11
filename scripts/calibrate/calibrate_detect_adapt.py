"""Stage-2 detect-then-adapt calibration + detection-quality report.

Runs the DetectAdaptTau calibrator over every probabilistic model's frozen
dumps (results/calibrated/detect_adapt/<model>.json, same protocol as the
other calibrate_* scripts) and additionally scores the DETECTOR itself: for
each test fault setting, the anomaly scores of the corrupted contexts are
compared against the clean-test contexts (AUC, precision/recall at s>0.5),
aggregated per fault kind -> results/base/detect_adapt_detection.json.

  python scripts/calibrate/calibrate_detect_adapt.py
  python scripts/calibrate/calibrate_detect_adapt.py --models qlstm
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.calib_runner import calibrate_model, discover_prob_models  # noqa: E402
from src.calibrators import DetectAdaptTau  # noqa: E402
from src.predictions_io import list_settings, load_predictions, prediction_path  # noqa: E402

METHOD = "detect_adapt"


def _kind(setting: str) -> str:
    """'level_shift_4.0' -> 'level_shift'; 'fgsm_2.0' -> 'fgsm'."""
    parts = setting.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].replace(".", "").isdigit() else setting


def detection_report(cal: DetectAdaptTau, pred_dir: Path, model: str) -> dict:
    from sklearn.metrics import roc_auc_score

    clean = load_predictions(prediction_path(pred_dir, model, "test", "clean"))
    s_clean = cal.anomaly_score(clean["context"])
    g_clean = cal.gate(s_clean)
    by_kind: dict[str, list] = defaultdict(list)
    per_setting = {}
    for s in list_settings(pred_dir, model, "test"):
        if s == "clean" or s.endswith("__cleaned"):
            continue
        d = load_predictions(prediction_path(pred_dir, model, "test", s))
        s_anom = cal.anomaly_score(d["context"])
        g_anom = cal.gate(s_anom)
        y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_anom))]
        sc = np.r_[s_clean, s_anom]
        per_setting[s] = {
            "auc": float(roc_auc_score(y, sc)),
            "recall": float(np.mean(g_anom > 0.0)),   # any engagement
            "mean_gate": float(g_anom.mean()),        # mean repair engagement
            "mean_score": float(s_anom.mean()),
        }
        by_kind[_kind(s)].append(per_setting[s]["auc"])
    return {
        "model": model,
        "gate": [cal.gate_lo_, cal.gate_hi_],
        "clean_false_alarm_rate": float(np.mean(g_clean > 0.0)),
        "clean_mean_gate": float(g_clean.mean()),
        "clean_mean_score": float(s_clean.mean()),
        "auc_by_kind": {k: float(np.mean(v)) for k, v in sorted(by_kind.items())},
        "per_setting": per_setting,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 calibration: detect_adapt")
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = ROOT / "results" / "calibrated" / METHOD
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or discover_prob_models(pred_dir)
    if not models:
        raise SystemExit(f"no probabilistic prediction dumps found in {pred_dir}")

    detection = {"method": METHOD, "threshold": 0.5, "models": {}}
    for model in models:
        res, cal = calibrate_model(DetectAdaptTau, METHOD, pred_dir, model,
                                   args.alpha, return_calibrator=True)
        (out_dir / f"{model}.json").write_text(json.dumps(res, indent=2))
        print(f"saved -> {out_dir / f'{model}.json'}")
        rep = detection_report(cal, pred_dir, model)
        detection["models"][model] = rep
        kinds = "  ".join(f"{k}:{v:.2f}" for k, v in rep["auc_by_kind"].items())
        print(f"  detection AUC  {kinds}")
        print(f"  clean false-alarm rate {rep['clean_false_alarm_rate']:.3f} "
              f"(mean score {rep['clean_mean_score']:.3f})")

    det_path = ROOT / "results" / "base" / "detect_adapt_detection.json"
    det_path.write_text(json.dumps(detection, indent=2))
    print(f"saved -> {det_path}")


if __name__ == "__main__":
    main()
