"""Shared plumbing for the stage-2 ``calibrate_*`` scripts.

Loads frozen predictions (``results/predictions/``), fits one calibrator per
probabilistic model on its validation dumps, applies it across every test
setting and writes ``results/calibrated/<method>/<model>.json`` with
before/after interval metrics. No model is ever run here — both sides of the
comparison score the bit-identical stage-1 forecasts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.calibrators import interval_metrics
from src.predictions_io import list_settings, load_predictions, prediction_path


def discover_prob_models(pred_dir: Path) -> list[str]:
    """Models with a validation-clean dump that contains quantiles."""
    models = []
    for p in sorted(Path(pred_dir).glob("*__val__clean.npz")):
        d = load_predictions(p)
        if "quantiles" in d:
            models.append(d["meta"].get("model", p.stem.split("__")[0]))
    return models


def calibrate_model(method_cls, method_name: str, pred_dir: Path, model: str,
                    alpha: float, return_calibrator: bool = False):
    cal = method_cls(alpha)
    val_clean = load_predictions(prediction_path(pred_dir, model, "val", "clean"))
    levels = val_clean["levels"]

    fit_mode = getattr(cal, "fit_on", "val_clean")
    if fit_mode in ("val_all", "val_labeled"):
        fit_settings = [s for s in list_settings(pred_dir, model, "val")
                        if not s.endswith("__cleaned")]
        ys, qs, cs, labs = [], [], [], []
        for s in fit_settings:
            d = load_predictions(prediction_path(pred_dir, model, "val", s))
            ys.append(d["y_true"])
            qs.append(d["quantiles"])
            cs.append(d["context"])
            labs.append(np.full(len(d["y_true"]), 0 if s == "clean" else 1))
        y_fit, q_fit, c_fit = np.concatenate(ys), np.concatenate(qs), np.concatenate(cs)
        fit_kwargs = ({"labels_val": np.concatenate(labs)}
                      if fit_mode == "val_labeled" else {})
    else:
        fit_settings = ["clean"]
        y_fit, q_fit = val_clean["y_true"], val_clean["quantiles"]
        c_fit = val_clean.get("context")
        fit_kwargs = {}

    cal.fit(y_fit, q_fit, levels, context_val=c_fit, **fit_kwargs)

    out = {"model": model, "method": method_name, "alpha": alpha,
           "fit_on": fit_settings, "params": cal.params(), "settings": {}}
    for s in list_settings(pred_dir, model, "test"):
        if s.endswith("__cleaned"):
            continue  # detect-and-clean variants are scored by their own script
        d = load_predictions(prediction_path(pred_dir, model, "test", s))
        y, q = d["y_true"], d["quantiles"]
        before = interval_metrics(y, q, levels, alpha)
        q_cal = cal.apply(y, q, levels, context=d.get("context"))
        after = interval_metrics(y, q_cal, levels, alpha)
        out["settings"][s] = {"before": before, "after": after}
        print(f"  {model:13s} {s:26s} picp {before['picp']:.3f} -> {after['picp']:.3f}"
              f"  mis {before['mis']:6.2f} -> {after['mis']:6.2f}")
    return (out, cal) if return_calibrator else out


def main_for_method(method_cls, method_name: str, root: Path) -> None:
    ap = argparse.ArgumentParser(description=f"Stage-2 calibration: {method_name}")
    ap.add_argument("--pred-dir", default=str(root / "results" / "predictions"),
                    help="frozen-forecast dir (use results/predictions_smoke for smoke dumps)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir) if args.out_dir else root / "results" / "calibrated" / method_name
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or discover_prob_models(pred_dir)
    if not models:
        raise SystemExit(f"no probabilistic prediction dumps found in {pred_dir} "
                         "(run scripts/analysis/run_anomaly_eval.py first)")
    for model in models:
        res = calibrate_model(method_cls, method_name, pred_dir, model, args.alpha)
        out_path = out_dir / f"{model}.json"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"saved -> {out_path}")
