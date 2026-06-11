"""Spread robust (anomaly-augmented) training across the remaining roster.

Phase 4 validated the robust+ACI recipe on qLSTM and the QTransformer; this
trains and dumps the missing robust variants as first-class models so the
4-corner study (run_robust_plus_cal.py) covers the whole probabilistic
family spectrum:

  - qdlinear_robust : per-batch torch hook (same augmenter, p=0.5, seed 42)
  - lgbm_robust     : one-shot numpy augmentation of the training windows
  - qrf_robust      : one-shot numpy augmentation of the training windows

(deepar_robust has its own script — its autoregressive eval protocol differs.)

  python scripts/models/run_robust_roster.py
  python scripts/models/run_robust_roster.py --smoke --models qdlinear
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
from src.anomaly import FAULT_TYPES_V2  # noqa: E402
from src.model_eval import NONGRAD_V1, evaluate_and_dump  # noqa: E402
from src.models.dlinear import (  # noqa: E402
    DLinearConfig, dlinear_context_grad, predict_dlinear, train_dlinear,
)
from src.models.lgbm_quantile import LgbmConfig, QUANTILES_7, predict_lgbm, train_lgbm  # noqa: E402
from src.models.qrf import QrfConfig, predict_qrf, train_qrf  # noqa: E402
from src.robust import augment_windows, make_augmenter  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L, H = 168, 24
AUGMENT_NOTE = "p=0.5 {spike,outlier,level_shift}x{1,2,4}, seed 42"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=("qdlinear", "lgbm", "qrf"))
    ap.add_argument("--catalog", choices=("v1", "v2"), default="v2")
    args = ap.parse_args()
    nongrad = NONGRAD_V1 + (FAULT_TYPES_V2 if args.catalog == "v2" else ())

    data = E.prepare(use_covariates=False)
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]

    summary = {}

    if "qdlinear" in args.models:
        cfg = DLinearConfig(quantiles=QUANTILES_7)
        if args.smoke:
            cfg.epochs = 1
        print("Training robust qDLinear ...")
        model, _ = train_dlinear(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg,
                                 device=DEVICE, augment_fn=make_augmenter(p=0.5))
        m = evaluate_and_dump(
            "qdlinear_robust", data,
            lambda x: {"quantiles": predict_dlinear(model, x[:, :, 0], device=DEVICE)},
            root=ROOT,
            grad_fn=lambda x, y: dlinear_context_grad(model, x[:, :, 0], y,
                                                      QUANTILES, device=DEVICE),
            quantiles=QUANTILES, nongrad_types=nongrad, smoke=args.smoke,
        )
        summary["qdlinear_robust"] = m

    if "lgbm" in args.models:
        cfg = LgbmConfig()
        if args.smoke:
            cfg.n_estimators = 20
        print("Training robust LightGBM-quantile (augmented table) ...")
        x_aug = augment_windows(x_tr[:, :, 0], p=0.5)
        models = train_lgbm(x_aug, y_tr, cfg)
        m = evaluate_and_dump(
            "lgbm_robust", data, lambda x: {"quantiles": predict_lgbm(models, x[:, :, 0], cfg)[1]},
            root=ROOT, quantiles=QUANTILES, nongrad_types=nongrad, smoke=args.smoke,
        )
        summary["lgbm_robust"] = m

    if "qrf" in args.models:
        cfg = QrfConfig()
        x_tr_q, y_tr_q = make_encoder_windows(data.train, L, H, stride=cfg.train_stride)
        if args.smoke:
            cfg.n_estimators = 10
            x_tr_q, y_tr_q = x_tr_q[:1000], y_tr_q[:1000]
        print("Training robust QRF (augmented table) ...")
        x_aug = augment_windows(x_tr_q[:, :, 0], p=0.5)
        model = train_qrf(x_aug, y_tr_q, cfg)
        m = evaluate_and_dump(
            "qrf_robust", data, lambda x: {"quantiles": predict_qrf(model, x[:, :, 0], cfg)},
            root=ROOT, quantiles=QUANTILES, nongrad_types=nongrad, smoke=args.smoke,
        )
        summary["qrf_robust"] = m

    for name, m in summary.items():
        m.update(model=name, augment=AUGMENT_NOTE, smoke=bool(args.smoke))
        out = ROOT / "results" / "base" / f"{name}.json"
        out.write_text(json.dumps(m, indent=2))
        print(f"Saved -> {out}  (clean RMSE {m['rmse']:.3f}, PICP {m['picp']:.3f})")


if __name__ == "__main__":
    main()
