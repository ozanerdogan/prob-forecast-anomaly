"""Phase-3 tail oversampling: improve cold/hot-extreme forecasts, recheck
overall calibration (the roadmap's caveat).

Trains two qLSTMs: baseline, and one whose training windows are reweighted by
how extreme their target is (|standardised target mean| -> higher weight via a
WeightedRandomSampler). Scores both on the temperature-extreme test deciles
AND on the full test set, so we can see whether the tail gain costs overall
calibration/bias.

  python scripts/run_tail_oversampling.py
  python scripts/run_tail_oversampling.py --smoke
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
from src.metrics import report_probabilistic  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--power", type=float, default=3.0, help="weight = (|z|+eps)^power")
    args = ap.parse_args()

    data = E.prepare(use_covariates=False)
    L, H = 168, 24
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:3000], y_tr[:3000]
        x_te, y_te = x_te[:120], y_te[:120]
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731

    # window-level extremeness in standardised space (target is column 0)
    z = np.abs(y_tr.mean(axis=1))
    weights = (z + 0.1) ** args.power

    # temperature-extreme test deciles (physical degC, true horizon mean)
    y_phys = inv(y_te)
    tmean = y_phys.mean(axis=1)
    k = max(8, len(tmean) // 10)
    cold = np.argsort(tmean)[:k]
    hot = np.argsort(-tmean)[:k]
    full = np.arange(len(tmean))

    def scores(model, idx):
        q = inv(predict_qlstm(model, x_te[:, :, 0], device=DEVICE))[idx]
        y = y_phys[idx]
        r = report_probabilistic(y.reshape(-1), q.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
        # signed bias of the median
        med = q[..., int(np.argmin(np.abs(QUANTILES - 0.5)))]
        return {"picp": r["picp"], "crps": r["crps"], "rmse": r["rmse"],
                "bias": float(np.mean(med - y))}

    def block(model):
        return {"full": scores(model, full), "cold": scores(model, cold),
                "hot": scores(model, hot)}

    cfg = QLstmConfig(seed=SEED)
    if args.smoke:
        cfg.epochs = 1

    print("Training normal qLSTM ...")
    normal, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
    print(f"Training tail-oversampled qLSTM (power={args.power}) ...")
    tail, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE,
                          sample_weights=weights)

    out = {"power": args.power, "slice_size": int(k), "seed": SEED,
           "smoke": bool(args.smoke),
           "normal": block(normal), "tail": block(tail)}
    out_path = ROOT / "results" / "base" / "qlstm_tail_compare.json"
    out_path.write_text(json.dumps(out, indent=2))
    for sl in ("full", "cold", "hot"):
        n, t = out["normal"][sl], out["tail"][sl]
        print(f"  {sl:5s} RMSE {n['rmse']:.2f}->{t['rmse']:.2f}  bias {n['bias']:+.2f}->{t['bias']:+.2f}"
              f"  PICP {n['picp']:.3f}->{t['picp']:.3f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
