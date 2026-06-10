"""Does robust (anomaly-augmented) training generalise across architectures?

Phase 3 showed it works for qLSTM (recurrent). This trains normal vs robust
versions of three architectures -- qLSTM (recurrent), QTransformer
(transformer), DeepAR (autoregressive) -- and compares their RAW
(uncalibrated) PICP/RMSE on clean + level-shift settings. If robust training
helps all three, "augment with anomalies" is a general recipe, not a
qLSTM quirk.

  python scripts/run_robust_generalize.py
  python scripts/run_robust_generalize.py --smoke
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
from src.models.deepar import DeepARConfig, quantiles_from_samples, sample_forecast, train_deepar  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.models.quantile_transformer import QTransformerConfig, predict_quantiles, train_qtransformer  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.robust import make_augmenter  # noqa: E402
from src.seq_data import make_ar_windows, make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SETTINGS = ("clean", "level_shift_1.0", "level_shift_2.0", "level_shift_4.0")


def _corrupt(ctx, setting):
    if setting == "clean":
        return ctx
    kind, inten = setting.rsplit("_", 1)
    rng = np.random.default_rng(SEED)  # same stream as run_anomaly_eval
    adv, _ = apply_anomaly(ctx, kind, float(inten), rng)
    return adv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data = E.prepare(use_covariates=False)
    L, H = 168, 24
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    yseq_tr, cov_tr = make_ar_windows(data.train, L, H, stride=1)
    yseq_va, cov_va = make_ar_windows(data.val, L, H, stride=H)
    yseq_te, cov_te = make_ar_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr, yseq_tr, cov_tr = x_tr[:2000], y_tr[:2000], yseq_tr[:2000], cov_tr[:2000]
        sl = slice(0, 60)
        x_te, y_te, yseq_te, cov_te = x_te[sl], y_te[sl], yseq_te[sl], cov_te[sl]

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_flat = inv(y_te).reshape(-1)
    ctx0 = x_te[:, :, 0].copy()

    def prob(q):
        r = report_probabilistic(y_flat, inv(q).reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
        return {"picp": r["picp"], "rmse": r["rmse"], "crps": r["crps"]}

    out = {"settings": list(SETTINGS), "seed": SEED, "smoke": bool(args.smoke), "models": {}}
    aug = make_augmenter(p=0.5)

    # ---- qLSTM ----------------------------------------------------------- #
    def qlstm_eval(model):
        return {s: prob(predict_qlstm(model, _corrupt(ctx0, s), device=DEVICE)) for s in SETTINGS}
    cfg = QLstmConfig(); cfg.epochs = 1 if args.smoke else cfg.epochs
    print("qLSTM normal / robust ...")
    out["models"]["qlstm"] = {
        "normal": qlstm_eval(train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)[0]),
        "robust": qlstm_eval(train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE, augment_fn=aug)[0])}

    # ---- QTransformer ---------------------------------------------------- #
    def qt_eval(model):
        res = {}
        for s in SETTINGS:
            xa = x_te.copy(); xa[:, :, 0] = _corrupt(ctx0, s)
            res[s] = prob(predict_quantiles(model, xa, device=DEVICE))
        return res
    cfgq = QTransformerConfig(n_features=1, quantiles=QUANTILES_7); cfgq.epochs = 1 if args.smoke else cfgq.epochs
    print("QTransformer normal / robust ...")
    out["models"]["qtransformer"] = {
        "normal": qt_eval(train_qtransformer(x_tr, y_tr, x_va, y_va, cfgq, device=DEVICE)[0]),
        "robust": qt_eval(train_qtransformer(x_tr, y_tr, x_va, y_va, cfgq, device=DEVICE, augment_fn=aug)[0])}

    # ---- DeepAR ---------------------------------------------------------- #
    def deepar_eval(model, cfgd):
        res = {}
        for s in SETTINGS:
            ys = yseq_te.copy(); ys[:, :L] = _corrupt(ctx0, s)
            samples = sample_forecast(model, ys, cov_te, cfgd, device=DEVICE, batch_size=128)
            res[s] = prob(quantiles_from_samples(samples, QUANTILES))
        return res
    cfgd = DeepARConfig(); cfgd.epochs = 1 if args.smoke else cfgd.epochs
    if args.smoke:
        cfgd.n_samples = 50
    print("DeepAR normal / robust ...")
    out["models"]["deepar"] = {
        "normal": deepar_eval(train_deepar(yseq_tr, cov_tr, yseq_va, cov_va, cfgd, device=DEVICE)[0], cfgd),
        "robust": deepar_eval(train_deepar(yseq_tr, cov_tr, yseq_va, cov_va, cfgd, device=DEVICE, augment_fn=aug)[0], cfgd)}

    out_path = ROOT / "results" / "base" / "robust_generalize.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n=== level_shift_4.0 raw PICP: normal -> robust ===")
    for m, b in out["models"].items():
        print(f"  {m:13s} {b['normal']['level_shift_4.0']['picp']:.3f} -> {b['robust']['level_shift_4.0']['picp']:.3f}"
              f"  | RMSE {b['normal']['level_shift_4.0']['rmse']:.2f} -> {b['robust']['level_shift_4.0']['rmse']:.2f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
