"""Natural sharp-transition slice — anomaly catalog class (ii), no injection.

Selects REAL test windows containing the sharpest temperature transitions
(decile of largest drops below the forecast origin = cold fronts, and the
symmetric rises) and scores the frozen clean forecasts + every calibration
regime on those slices vs the full grid.

The question this answers: a good adaptive calibrator must widen under
sensor faults, but a genuine front is *signal*, not corruption — does the
repair over-widen exactly when the input looks 'suspicious' for natural
reasons (false-alarm cost)? Static/CQR fits, the input-conditional regressor
and the ACI replay are reproduced from the validation dumps (deterministic,
same code path as the calibrate_* scripts); ACI runs online over the full
sequence and is then sliced, matching deployment.

Writes results/base/natural_extremes.json.

  python scripts/analysis/run_natural_extremes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.calib_runner import discover_prob_models  # noqa: E402
from src.calibrators import ACITau, CQRCalibrator, InputTau, StaticTau, interval_metrics  # noqa: E402
from src.predictions_io import list_settings, load_predictions, prediction_path  # noqa: E402

ALPHA = 0.1


def _slice_metrics(y, q, levels, idx):
    return interval_metrics(y[idx], q[idx], levels, ALPHA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "predictions"))
    ap.add_argument("--decile", type=float, default=0.1)
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)

    models = discover_prob_models(pred_dir)
    if not models:
        raise SystemExit(f"no prediction dumps in {pred_dir}")

    # Slice definition from the ground truth (identical across models):
    # transition = the extreme excursion of the true future relative to its
    # first hour, within the 24h horizon — i.e. how hard the temperature
    # falls/rises right after the forecast origin. Physical degC, no scaler
    # needed, and independent of any model's predictions.
    ref = load_predictions(prediction_path(pred_dir, models[0], "test", "clean"))
    y = np.asarray(ref["y_true"], dtype=float)            # (N, H) physical
    drop = y.min(axis=1) - y[:, 0]    # most negative = sharp fall after origin
    rise = y.max(axis=1) - y[:, 0]
    n = len(y)
    k = max(8, int(round(args.decile * n)))
    idx_drop = np.argsort(drop)[:k]                  # coldest fronts
    idx_rise = np.argsort(-rise)[:k]                 # sharpest warm-ups
    idx_all = np.arange(n)

    out = {
        "n_windows": int(n), "slice_size": int(k), "alpha": ALPHA,
        "definition": "drop = min(y)-y[0], rise = max(y)-y[0] within the 24h horizon; "
                      "slices are the most extreme decile each (real events, no injection)",
        "slice_stats": {
            "sharp_drop_degC": {"mean": float(drop[idx_drop].mean()),
                                "min": float(drop[idx_drop].min())},
            "sharp_rise_degC": {"mean": float(rise[idx_rise].mean()),
                                "max": float(rise[idx_rise].max())},
        },
        "models": {},
    }

    for model in models:
        d_te = load_predictions(prediction_path(pred_dir, model, "test", "clean"))
        levels = d_te["levels"]
        y_te, q_te, ctx_te = d_te["y_true"], d_te["quantiles"], d_te["context"]
        d_va = load_predictions(prediction_path(pred_dir, model, "val", "clean"))

        # reproduce the calibrators exactly as calibrate_* scripts do
        cals = {}
        cals["static"] = StaticTau(ALPHA).fit(d_va["y_true"], d_va["quantiles"], levels)
        cals["cqr"] = CQRCalibrator(ALPHA).fit(d_va["y_true"], d_va["quantiles"], levels)
        cals["aci"] = ACITau(ALPHA).fit(d_va["y_true"], d_va["quantiles"], levels)
        fit_settings = [s for s in list_settings(pred_dir, model, "val")
                        if not s.endswith("__cleaned")]
        ys, qs, cs = [], [], []
        for s in fit_settings:
            dv = load_predictions(prediction_path(pred_dir, model, "val", s))
            ys.append(dv["y_true"]); qs.append(dv["quantiles"]); cs.append(dv["context"])
        cals["input_tau"] = InputTau(ALPHA).fit(
            np.concatenate(ys), np.concatenate(qs), levels,
            context_val=np.concatenate(cs))

        block: dict = {"uncalibrated": {}}
        for name, idx in (("full", idx_all), ("sharp_drop", idx_drop), ("sharp_rise", idx_rise)):
            block["uncalibrated"][name] = _slice_metrics(y_te, q_te, levels, idx)
        for mname, cal in cals.items():
            q_cal = cal.apply(y_te, q_te, levels, context=ctx_te)
            block[mname] = {name: _slice_metrics(y_te, np.asarray(q_cal), levels, idx)
                            for name, idx in (("full", idx_all), ("sharp_drop", idx_drop),
                                              ("sharp_rise", idx_rise))}
        out["models"][model] = block
        f, dr = block["input_tau"]["full"], block["input_tau"]["sharp_drop"]
        print(f"  {model:18s} input_tau MPIW full {f['mpiw']:.2f} -> sharp_drop {dr['mpiw']:.2f} "
              f"| picp {dr['picp']:.3f}")

    out_path = ROOT / "results" / "base" / "natural_extremes.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
