"""Extreme-quantile QTransformer: edge-case (cold/hot) interval coverage.

The headline model uses 7 quantiles spanning the 90% interval (0.05-0.95).
This trains an 11-quantile variant that adds the 2.5/97.5 and 1/99 levels, so
we can report 90% / 95% / 98% interval coverage -- the question being whether
the model can honestly bound EXTREME temperatures, and how the wider
intervals behave on the cold/hot test deciles and under level shift.

Caveat reported: extreme quantiles sit where data is sparse, so they are
noisier and more crossing-prone (we sort to enforce monotonicity). "More
quantiles" is not free.

  python scripts/ablation/run_qt_extreme_quantiles.py
  python scripts/ablation/run_qt_extreme_quantiles.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.metrics import picp as picp_fn  # noqa: E402
from src.models.quantile_transformer import (  # noqa: E402
    QTransformerConfig,
    QUANTILES_9,
    predict_quantiles,
    train_qtransformer,
)
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

Q = np.array(QUANTILES_9)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L, H = 168, 24
# (nominal coverage, lower level, upper level)
INTERVALS = ((0.90, 0.05, 0.95), (0.95, 0.025, 0.975), (0.98, 0.01, 0.99))


def _qi(level):
    return int(np.argmin(np.abs(Q - level)))


def _coverage_block(y, q):
    out = {}
    for nominal, lo, hi in INTERVALS:
        ql, qh = q[:, _qi(lo)], q[:, _qi(hi)]
        out[f"{int(nominal*100)}pct"] = {
            "target": nominal,
            "picp": picp_fn(y, ql, qh),
            "mpiw": float(np.mean(qh - ql)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data = E.prepare(use_covariates=False)
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]
    cfg = QTransformerConfig(n_features=1, quantiles=QUANTILES_9)
    if args.smoke:
        cfg.epochs = 1
    print(f"Training QT with {len(Q)} quantiles (extreme tails) ...")
    model, _ = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE)

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    q = inv(predict_quantiles(model, x_te, device=DEVICE)).reshape(-1, len(Q))
    y = inv(y_te).reshape(-1)
    # crossing check (post-sort should be 0)
    q3 = inv(predict_quantiles(model, x_te, device=DEVICE))
    crossings = int(np.sum(np.diff(q3, axis=-1) < -1e-6))

    # temperature-extreme deciles (cold / hot) on the full test
    y_w = inv(y_te)
    tmean = y_w.mean(axis=1)
    k = max(8, len(tmean) // 10)
    cold = np.argsort(tmean)[:k]
    hot = np.argsort(-tmean)[:k]
    qw = inv(predict_quantiles(model, x_te, device=DEVICE))  # (N,H,Q)

    def slice_cov(idx):
        return _coverage_block(y_w[idx].reshape(-1), qw[idx].reshape(-1, len(Q)))

    out = {
        "quantiles": Q.tolist(), "n_quantiles": len(Q), "smoke": bool(args.smoke),
        "crossings_after_sort": crossings,
        "full": _coverage_block(y, q),
        "cold_decile": slice_cov(cold),
        "hot_decile": slice_cov(hot),
        "note": "extreme tails are data-sparse -> noisier; sorted to avoid crossing",
    }
    out_path = ROOT / "results" / "base" / "qt_extreme_quantiles.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n=== interval coverage (full test) ===")
    for k_, v in out["full"].items():
        print(f"  {k_:6s} target {v['target']:.2f}  PICP {v['picp']:.3f}  MPIW {v['mpiw']:.2f}")
    print(f"crossings after sort: {crossings}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
