"""Phase-3 interval combination: average quantiles, not just point forecasts.

Phase-1 left the ensemble at point level (Avg(LSTM, QT) = 2.384). Here we
combine the *intervals* of the probabilistic members by averaging their
quantile curves per level, then recheck PICP/MIS — the open item from the
roadmap. Two weighting schemes: equal, and validation-CRPS-inverse weights
(fit on val clean, applied to test). All from frozen dumps, no model runs.

Members: the probabilistic models present in results/predictions
(qlstm, qtransformer, qtransformer_multi, qdlinear, qrf, deepar, lgbm).
Evaluated on clean and the worst regimes (level_shift_4.0, fgsm_4.0 where
available).

  python scripts/run_ensemble_intervals.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrators import interval_metrics  # noqa: E402
from src.predictions_io import load_predictions, prediction_path  # noqa: E402

ALPHA = 0.1
MEMBERS = ("qlstm", "qtransformer", "qtransformer_multi", "qdlinear", "qrf", "deepar", "lgbm")
SETTINGS = ("clean", "level_shift_4.0", "fgsm_4.0")


def _have(pred_dir, model, split, setting):
    return prediction_path(pred_dir, model, split, setting).exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "predictions"))
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)

    members = [m for m in MEMBERS if _have(pred_dir, m, "test", "clean")
               and "quantiles" in load_predictions(prediction_path(pred_dir, m, "test", "clean"))]
    if len(members) < 2:
        raise SystemExit("need >=2 probabilistic members with dumps")
    levels = load_predictions(prediction_path(pred_dir, members[0], "test", "clean"))["levels"]

    # validation-CRPS-inverse weights (fit on val clean)
    from src.metrics import crps_from_quantiles
    val_crps = {}
    for m in members:
        dv = load_predictions(prediction_path(pred_dir, m, "val", "clean"))
        val_crps[m] = crps_from_quantiles(
            dv["y_true"].reshape(-1), dv["quantiles"].reshape(-1, len(levels)), levels)
    inv_w = np.array([1.0 / val_crps[m] for m in members])
    inv_w /= inv_w.sum()
    eq_w = np.full(len(members), 1.0 / len(members))

    out = {"members": members, "alpha": ALPHA,
           "val_crps": val_crps,
           "weights_inverse_crps": dict(zip(members, inv_w.tolist())),
           "candidates": {}, "by_setting": {}}

    def combine(setting, weights):
        qs, y = [], None
        for m in members:
            if not _have(pred_dir, m, "test", setting):
                return None, None
            d = load_predictions(prediction_path(pred_dir, m, "test", setting))
            qs.append(np.asarray(d["quantiles"], dtype=float))
            y = d["y_true"]
        stacked = np.stack(qs, axis=0)  # (M, N, H, Q)
        w = np.asarray(weights).reshape(-1, 1, 1, 1)
        return (stacked * w).sum(axis=0), y

    # headline candidates on clean: each member + both ensembles
    for m in members:
        d = load_predictions(prediction_path(pred_dir, m, "test", "clean"))
        out["candidates"][m] = interval_metrics(d["y_true"], d["quantiles"], levels, ALPHA)
    for tag, w in (("ensemble_equal", eq_w), ("ensemble_crpsw", inv_w)):
        q, y = combine("clean", w)
        out["candidates"][tag] = interval_metrics(y, q, levels, ALPHA)

    # robustness across settings (equal-weight ensemble vs best single member).
    # interval_metrics reports pinball (= CRPS/2 proxy); rank members by it.
    best_single = min(members, key=lambda m: out["candidates"][m]["pinball"])
    for setting in SETTINGS:
        row = {}
        q, y = combine(setting, eq_w)
        if q is not None:
            row["ensemble_equal"] = interval_metrics(y, q, levels, ALPHA)
        if _have(pred_dir, best_single, "test", setting):
            d = load_predictions(prediction_path(pred_dir, best_single, "test", setting))
            row[f"best_single_{best_single}"] = interval_metrics(d["y_true"], d["quantiles"], levels, ALPHA)
        out["by_setting"][setting] = row

    out_path = ROOT / "results" / "base" / "ensemble_intervals.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("clean pinball:", {k: round(v["pinball"], 3) for k, v in out["candidates"].items()})
    print("clean PICP:", {k: round(v["picp"], 3) for k, v in out["candidates"].items()})
    print(f"best single member: {best_single}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
