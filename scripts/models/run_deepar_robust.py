"""Robust (anomaly-augmented) DeepAR — completes the robust roster.

Same augmenter as the other *_robust scripts (p=0.5 per-batch target-context
injection, seed 42). DeepAR's autoregressive protocol differs from the
encoder models (teacher-forced context inside the AR window), so the dump
path mirrors run_anomaly_eval.py's deepar block instead of evaluate_and_dump:
val + test, clean + non-gradient catalog (byte-identical seeded streams) +
white-box FGSM on the robust model's own gradient.

  python scripts/models/run_deepar_robust.py
  python scripts/models/run_deepar_robust.py --smoke
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
from src.anomaly import (  # noqa: E402
    FAULT_TYPES_V2, VAL_STREAM_INDEX, apply_anomaly, linf_fgsm,
)
from src.calibrators import interval_metrics  # noqa: E402
from src.models.deepar import DeepARConfig, train_deepar  # noqa: E402
from src.models.quantile_transformer import QUANTILES_7  # noqa: E402
from src.predictions_io import prediction_path, save_predictions  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.robust import make_augmenter  # noqa: E402
from src.seq_data import make_ar_windows, make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
INTENSITIES = (1.0, 2.0, 4.0)
NONGRAD_V1 = ("point_spike", "contextual_outlier", "level_shift")
SEED = 42
MODEL = "deepar_robust"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--catalog", choices=("v1", "v2"), default="v2")
    args = ap.parse_args()
    nongrad = NONGRAD_V1 + (FAULT_TYPES_V2 if args.catalog == "v2" else ())

    cfg = DeepARConfig()
    if args.smoke:
        cfg.epochs = 1
        cfg.n_samples = 50
    data = E.prepare(use_covariates=False)
    L, H = cfg.lookback, cfg.horizon

    yseq_tr, cov_tr = make_ar_windows(data.train, L, H, stride=1)
    yseq_va, cov_va = make_ar_windows(data.val, L, H, stride=H)
    yseq_te, cov_te = make_ar_windows(data.test, L, H, stride=H)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        sl = slice(0, 60)
        yseq_tr, cov_tr = yseq_tr[:2000], cov_tr[:2000]
        yseq_va, cov_va, x_va, y_va = yseq_va[sl], cov_va[sl], x_va[sl], y_va[sl]
        yseq_te, cov_te, x_te, y_te = yseq_te[sl], cov_te[sl], x_te[sl], y_te[sl]

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")

    print(f"Training robust DeepAR ({cfg.epochs} epochs, {DEVICE}) ...")
    model, _ = train_deepar(yseq_tr, cov_tr, yseq_va, cov_va, cfg, device=DEVICE,
                            augment_fn=make_augmenter(p=0.5))

    def dump(split, setting, **arrays):
        meta = {"model": MODEL, "split": split, "setting": setting, "alpha": ALPHA,
                "seed": SEED, "smoke": bool(args.smoke),
                "units": "physical degC (y_true/quantiles), standardised (context)"}
        save_predictions(prediction_path(pred_dir, MODEL, split, setting),
                         meta=meta, **arrays)

    def predict(yseq, cov, ctx):
        ys = yseq.copy()
        ys[:, :L] = ctx
        q, _ = E.deepar_predict(model, cfg, ys, cov, QUANTILES)
        return inv(q)

    for split, yseq, cov, x_enc, y_enc in (
            ("val", yseq_va, cov_va, x_va, y_va),
            ("test", yseq_te, cov_te, x_te, y_te)):
        y_true = inv(y_enc)
        ctx_clean = x_enc[:, :, 0].copy()
        q = predict(yseq, cov, ctx_clean)
        dump(split, "clean", y_true=y_true, quantiles=q, levels=QUANTILES,
             context=ctx_clean)
        if split == "test":
            clean_metrics = interval_metrics(y_true, q, QUANTILES, ALPHA)
        print(f"[{split}] clean dumped")

        grad = E.deepar_context_grad(model, cfg, yseq, cov)
        for kind in nongrad + ("fgsm",):
            for intensity in INTENSITIES:
                setting = f"{kind}_{intensity:.1f}"
                if kind == "fgsm":
                    ctx_adv, _ = linf_fgsm(ctx_clean, grad, intensity)
                elif split == "val":
                    rng = np.random.default_rng(
                        [SEED, 1, VAL_STREAM_INDEX[kind], int(10 * intensity)])
                    ctx_adv, _ = apply_anomaly(ctx_clean, kind, intensity, rng)
                else:
                    rng = np.random.default_rng(SEED)
                    ctx_adv, _ = apply_anomaly(ctx_clean, kind, intensity, rng)
                dump(split, setting, y_true=y_true,
                     quantiles=predict(yseq, cov, ctx_adv),
                     levels=QUANTILES, context=ctx_adv)
            print(f"[{split}] {kind} dumped")

    out = {"model": MODEL, "target": TARGET, "epochs": cfg.epochs,
           "seed": SEED, "device": DEVICE, "smoke": bool(args.smoke),
           "augment": "p=0.5 per batch {spike,outlier,level_shift}x{1,2,4}",
           "catalog": args.catalog,
           "rmse": clean_metrics["rmse_median"], "picp": clean_metrics["picp"],
           "mpiw": clean_metrics["mpiw"], "mis": clean_metrics["mis"],
           "pinball": clean_metrics["pinball"]}
    out_path = ROOT / "results" / "base" / f"{MODEL}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}  (clean RMSE {out['rmse']:.3f}, PICP {out['picp']:.3f})")


if __name__ == "__main__":
    main()
