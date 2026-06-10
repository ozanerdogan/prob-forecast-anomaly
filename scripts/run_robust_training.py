"""Phase-3 robust training: does anomaly-augmented training help the MODEL
degrade more gracefully, on top of (or instead of) post-hoc calibration?

Trains two qLSTMs from the same seed/budget: the phase-2 baseline, and a
"robust" one whose training batches are randomly anomaly-injected on the fly
(point spike / contextual outlier / level shift, random intensity). Both are
scored RAW (no calibration) on clean + the injected test settings, so the
delta is purely the training change. The contrast with the calibration story:
calibration repairs the interval; robust training repairs the model.

  python scripts/run_robust_training.py
  python scripts/run_robust_training.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.anomaly import apply_anomaly  # noqa: E402
from src.metrics import report_probabilistic  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUG_TYPES = ("point_spike", "contextual_outlier", "level_shift")
SEED = 42
SETTINGS = ("clean", "point_spike_2.0", "contextual_outlier_2.0",
            "level_shift_1.0", "level_shift_2.0", "level_shift_4.0")


def make_augmenter(p=0.5):
    """Per-batch on-the-fly target-context injection.

    train_qlstm feeds the augmenter the 2-D target context it trains on
    (B, L); inject anomalies into a random subset of rows.
    """
    rng = np.random.default_rng(SEED)

    def augment(xb: torch.Tensor) -> torch.Tensor:
        x = xb.numpy().copy()           # (B, L)
        mask = rng.random(len(x)) < p
        if mask.any():
            kind = AUG_TYPES[rng.integers(len(AUG_TYPES))]
            inten = float(rng.choice([1.0, 2.0, 4.0]))
            adv, _ = apply_anomaly(x[mask], kind, inten, rng)
            x[mask] = adv
        return torch.from_numpy(x)

    return augment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data = E.prepare(use_covariates=False)
    L, H = 168, 24
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]
        x_te, y_te = x_te[:60], y_te[:60]
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_flat = inv(y_te).reshape(-1)

    def eval_on_settings(model):
        block = {}
        ctx0 = x_te[:, :, 0].copy()
        for setting in SETTINGS:
            if setting == "clean":
                ctx = ctx0
            else:
                kind, inten = setting.rsplit("_", 1)
                rng = np.random.default_rng(SEED)  # same stream as run_anomaly_eval
                ctx, _ = apply_anomaly(ctx0, kind, float(inten), rng)
            q = inv(predict_qlstm(model, ctx, device=DEVICE))
            r = report_probabilistic(y_flat, q.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
            block[setting] = {"picp": r["picp"], "crps": r["crps"], "rmse": r["rmse"], "mis": r["mis"]}
        return block

    cfg = QLstmConfig(seed=SEED)
    if args.smoke:
        cfg.epochs = 1

    print("Training normal qLSTM ...")
    normal, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
    print("Training robust (anomaly-augmented) qLSTM ...")
    robust, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE,
                            augment_fn=make_augmenter(p=0.5))

    out = {"settings_order": list(SETTINGS), "augment": "p=0.5 per batch, "
           "{spike,outlier,level_shift} x {1,2,4}", "seed": SEED,
           "smoke": bool(args.smoke),
           "normal": eval_on_settings(normal),
           "robust": eval_on_settings(robust)}
    out_path = ROOT / "results" / "base" / "qlstm_robust_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    for s in SETTINGS:
        print(f"  {s:22s} PICP normal {out['normal'][s]['picp']:.3f} -> robust {out['robust'][s]['picp']:.3f}"
              f"  | RMSE {out['normal'][s]['rmse']:.2f} -> {out['robust'][s]['rmse']:.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
