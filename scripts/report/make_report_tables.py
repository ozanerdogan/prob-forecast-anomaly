"""Phase-4 report tables, built purely from frozen dumps (no model runs).

Emits Markdown tables under report/tables/ and a machine-readable
results/base/report_tables.json:

  - clean_leaderboard : every model's clean-test point + interval metrics
  - robustness_matrix : model x anomaly-type x intensity -> RMSE/CRPS/PICP/MIS
                        (raw, uncalibrated) for the probabilistic roster
  - calibration_matrix: best repair per (model, worst setting) across the
                        stage-2 methods (static/cqr/aci/input_tau/aci_margin)

  python scripts/report/make_report_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.calibrators import interval_metrics  # noqa: E402
from src.metrics import report  # noqa: E402
from src.predictions_io import list_settings, load_predictions, prediction_path  # noqa: E402

PRED = ROOT / "results" / "predictions"
BASE = ROOT / "results" / "base"
CAL = ROOT / "results" / "calibrated"
ALPHA = 0.1
TABLES = ROOT / "report" / "tables"
CAL_METHODS = ("static", "cqr", "aci", "input_tau", "aci_margin", "detect_adapt")

ROSTER = ("naive_seasonal", "lstm", "gru", "dlinear", "deepar", "qtransformer",
          "qtransformer_multi", "qlstm", "qdlinear", "lgbm", "qrf", "qlstm_robust")


def _has(model, split, setting):
    return prediction_path(PRED, model, split, setting).exists()


def _point_block(d):
    y = d["y_true"].reshape(-1)
    if "quantiles" in d:
        lv = d["levels"]
        med = d["quantiles"][..., int(np.argmin(np.abs(lv - 0.5)))].reshape(-1)
        r = report(y, med)
        im = interval_metrics(d["y_true"], d["quantiles"], lv, ALPHA)
        return {"rmse": r["rmse"], "mae": r["mae"], "picp": im["picp"],
                "mpiw": im["mpiw"], "mis": im["mis"], "pinball": im["pinball"]}
    r = report(y, d["point"].reshape(-1))
    return {"rmse": r["rmse"], "mae": r["mae"]}


def clean_leaderboard():
    rows = {}
    for m in ROSTER:
        if _has(m, "test", "clean"):
            rows[m] = _point_block(load_predictions(prediction_path(PRED, m, "test", "clean")))
    return rows


def robustness_matrix():
    out = {}
    for m in ROSTER:
        if not _has(m, "test", "clean"):
            continue
        d0 = load_predictions(prediction_path(PRED, m, "test", "clean"))
        if "quantiles" not in d0:
            continue
        settings = [s for s in list_settings(PRED, m, "test") if not s.endswith("__cleaned")]
        out[m] = {}
        for s in settings:
            d = load_predictions(prediction_path(PRED, m, "test", s))
            out[m][s] = _point_block(d)
    return out


def calibration_matrix():
    out = {}
    for m in ROSTER:
        per_method = {}
        for meth in CAL_METHODS:
            p = CAL / meth / f"{m}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            per_method[meth] = d["settings"]
        if per_method:
            out[m] = per_method
    return out


def _fmt(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def write_markdown(lb, rob, cal):
    TABLES.mkdir(parents=True, exist_ok=True)

    # leaderboard
    lines = ["# Clean-test leaderboard\n",
             "| Model | RMSE | MAE | CRPS≈2·pinball | PICP | MPIW | MIS |",
             "|---|---|---|---|---|---|---|"]
    for m, r in sorted(lb.items(), key=lambda kv: kv[1]["rmse"]):
        crps = 2 * r["pinball"] if "pinball" in r else None
        lines.append(f"| {m} | {_fmt(r['rmse'],2)} | {_fmt(r['mae'],2)} | {_fmt(crps)} "
                     f"| {_fmt(r.get('picp'))} | {_fmt(r.get('mpiw'),2)} | {_fmt(r.get('mis'),2)} |")
    (TABLES / "clean_leaderboard.md").write_text("\n".join(lines) + "\n")

    # robustness: PICP at the three worst settings
    worst = ("level_shift_4.0", "fgsm_4.0", "flatline_4.0")
    lines = ["# Raw robustness — PICP at severe settings (uncalibrated)\n",
             "| Model | clean | " + " | ".join(worst) + " |",
             "|---|" + "---|" * (len(worst) + 1)]
    for m, block in rob.items():
        cells = [_fmt(block.get("clean", {}).get("picp"))]
        for w in worst:
            cells.append(_fmt(block.get(w, {}).get("picp")))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (TABLES / "robustness_picp.md").write_text("\n".join(lines) + "\n")

    # calibration: PICP after each method at level_shift_4.0
    setting = "level_shift_4.0"
    lines = [f"# Calibration recovery — PICP at {setting}\n",
             "| Model | raw | " + " | ".join(CAL_METHODS) + " |",
             "|---|" + "---|" * (len(CAL_METHODS) + 1)]
    for m, methods in cal.items():
        raw = None
        cells = []
        for meth in CAL_METHODS:
            s = methods.get(meth, {}).get(setting)
            if s is None:
                cells.append("-")
                continue
            raw = s["before"]["picp"]
            cells.append(_fmt(s["after"]["picp"]))
        lines.append(f"| {m} | {_fmt(raw)} | " + " | ".join(cells) + " |")
    (TABLES / "calibration_recovery.md").write_text("\n".join(lines) + "\n")


def write_detect_adapt(cal):
    """4-regime policy comparison + the detection-quality ladder."""
    det_path = BASE / "detect_adapt_detection.json"
    if not det_path.exists() or "qlstm" not in cal:
        return
    det = json.loads(det_path.read_text())["models"]

    regimes = ("static", "aci", "input_tau", "detect_adapt")
    faults = ("level_shift_4.0", "drift_4.0", "fgsm_4.0", "flatline_4.0",
              "clock_skew_4.0", "noise_burst_4.0")
    lines = ["# Detect-then-adapt — policy comparison (qlstm)\n",
             "Coverage repair per regime; clean row shows the width cost "
             "(MIS, lower = sharper).\n",
             "| Setting | raw | " + " | ".join(regimes) + " |",
             "|---|" + "---|" * (len(regimes) + 1)]
    methods = cal["qlstm"]
    row = ["| clean MIS | "
           + _fmt(methods["static"]["clean"]["before"]["mis"], 2) + " | "
           + " | ".join(_fmt(methods.get(r, {}).get("clean", {})
                             .get("after", {}).get("mis"), 2) for r in regimes) + " |"]
    lines += row
    for f in faults:
        cells = []
        raw = None
        for r in regimes:
            s = methods.get(r, {}).get(f)
            if s is None:
                cells.append("-")
                continue
            raw = s["before"]["picp"]
            cells.append(_fmt(s["after"]["picp"]))
        lines.append(f"| {f} PICP | {_fmt(raw)} | " + " | ".join(cells) + " |")

    lines += ["\n# Detection quality by fault kind and intensity (qlstm, AUC "
              "vs clean-test)\n",
              "| Fault | 1.0 | 2.0 | 4.0 |", "|---|---|---|---|"]
    ps = det["qlstm"]["per_setting"]
    kinds = sorted({k.rsplit("_", 1)[0] for k in ps})
    for kind in kinds:
        cells = [_fmt(ps.get(f"{kind}_{i}", {}).get("auc"), 2)
                 for i in ("1.0", "2.0", "4.0")]
        lines.append(f"| {kind} | " + " | ".join(cells) + " |")
    far = det["qlstm"]["clean_false_alarm_rate"]
    lines.append(f"\nClean false-alarm rate (any repair engagement): {far:.3f}")
    (TABLES / "detect_adapt.md").write_text("\n".join(lines) + "\n")


def main():
    lb = clean_leaderboard()
    rob = robustness_matrix()
    cal = calibration_matrix()
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "report_tables.json").write_text(json.dumps(
        {"clean_leaderboard": lb, "robustness_matrix": rob,
         "calibration_matrix": cal, "alpha": ALPHA}, indent=2))
    write_markdown(lb, rob, cal)
    write_detect_adapt(cal)
    print(f"leaderboard: {len(lb)} models | robustness: {len(rob)} | calibration: {len(cal)}")
    print(f"tables -> {TABLES}/  + results/base/report_tables.json")


if __name__ == "__main__":
    main()
