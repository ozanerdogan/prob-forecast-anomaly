"""Transfer-FGSM: do surrogate-gradient attacks hurt gradient-free models?

Trees (LightGBM-quantile, QRF) and the naive baseline have no gradient, so
the white-box FGSM column is undefined for them — but "no gradient" is not a
security argument if an attacker can craft the perturbation on a SURROGATE.
The surrogate's adversarial contexts are already frozen in its dumps
(results/predictions/<surrogate>__test__fgsm_<i>.npz stores the perturbed
context), so no gradient is recomputed here: each gradient-free target is
retrained (same seeds/configs as its stage-1 script) and scored on the exact
contexts that attacked the surrogate. Two surrogates (qLSTM, QTransformer)
control for surrogate-dependence.

  python scripts/analysis/run_fgsm_transfer.py
  python scripts/analysis/run_fgsm_transfer.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.calibrators import interval_metrics  # noqa: E402
from src.metrics import report_probabilistic  # noqa: E402
from src.models.dlinear import DLinearConfig, predict_dlinear, train_dlinear  # noqa: E402
from src.models.lgbm_quantile import LgbmConfig, QUANTILES_7, predict_lgbm, train_lgbm  # noqa: E402
from src.models.qrf import QrfConfig, predict_qrf, train_qrf  # noqa: E402
from src.predictions_io import load_predictions, prediction_path  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
SURROGATES = ("qlstm", "qtransformer")
INTENSITIES = (1.0, 2.0, 4.0)
L, H = 168, 24


def _prob_scores(y_flat, q_flat):
    out = report_probabilistic(y_flat, q_flat, QUANTILES, alpha=ALPHA)
    return {"rmse": out["rmse"], "crps": out["crps"], "picp": out["picp"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    pred_dir = ROOT / "results" / ("predictions_smoke" if args.smoke else "predictions")

    data = E.prepare(use_covariates=False)
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_flat = inv(y_te).reshape(-1)

    # The surrogate dumps must describe the same window grid we rebuilt.
    ref = load_predictions(prediction_path(pred_dir, SURROGATES[0], "test", "clean"))
    if not args.smoke:
        assert ref["context"].shape == x_te[:, :, 0].shape, "window grid mismatch"
        assert np.allclose(ref["context"], x_te[:, :, 0], atol=1e-4), \
            "surrogate dump contexts do not match the rebuilt test windows"

    # ---------------- gradient-free targets (fresh, seeded training) -------
    print("Training gradient-free targets ...")
    lgbm_cfg = LgbmConfig()
    if args.smoke:
        lgbm_cfg.n_estimators = 20
    x_tr_l, y_tr_l = (x_tr, y_tr) if lgbm_cfg.train_stride == 1 else \
        make_encoder_windows(data.train, L, H, stride=lgbm_cfg.train_stride)
    lgbm = train_lgbm(x_tr_l[:, :, 0], y_tr_l, lgbm_cfg)

    qrf_cfg = QrfConfig()
    x_tr_q, y_tr_q = make_encoder_windows(data.train, L, H, stride=qrf_cfg.train_stride)
    if args.smoke:
        qrf_cfg.n_estimators = 10
        x_tr_q, y_tr_q = x_tr_q[:1000], y_tr_q[:1000]
    qrf = train_qrf(x_tr_q[:, :, 0], y_tr_q, qrf_cfg)

    dl_cfg = DLinearConfig(quantiles=QUANTILES_7)
    if args.smoke:
        dl_cfg.epochs = 1
    qdl, _ = train_dlinear(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, dl_cfg)

    targets = {
        "lgbm": lambda c: predict_lgbm(lgbm, c, lgbm_cfg)[1],
        "qrf": lambda c: predict_qrf(qrf, c, qrf_cfg),
        "qdlinear": lambda c: predict_dlinear(qdl, c.astype(np.float32)),
    }

    def naive_rmse(ctx):  # seasonal repeat of the last day, point only
        pred = inv(np.tile(ctx[:, -24:], 1))
        return float(np.sqrt(np.mean((pred.reshape(-1) - y_flat) ** 2)))

    # ---------------- evaluation ------------------------------------------
    out = {"alpha": ALPHA, "surrogates": list(SURROGATES), "smoke": bool(args.smoke),
           "note": "targets rescored on the exact adversarial contexts frozen in "
                   "the surrogate dumps; no target gradient exists",
           "targets": {}, "surrogate_whitebox": {}}

    ctx_clean = x_te[:, :, 0]
    for name, fn in targets.items():
        q = inv(fn(ctx_clean)).reshape(-1, len(QUANTILES))
        out["targets"][name] = {"clean": _prob_scores(y_flat, q)}
    out["targets"]["naive_seasonal"] = {"clean": {"rmse": naive_rmse(ctx_clean)}}

    for sur in SURROGATES:
        for inten in INTENSITIES:
            setting = f"fgsm_{inten:.1f}"
            p = prediction_path(pred_dir, sur, "test", setting)
            if not p.exists():
                print(f"  [skip] {sur} {setting}: no dump")
                continue
            d = load_predictions(p)
            ctx_adv = d["context"]
            # surrogate's own white-box damage, for scale
            wb = interval_metrics(d["y_true"], d["quantiles"], d["levels"], ALPHA)
            out["surrogate_whitebox"].setdefault(sur, {})[setting] = {
                "rmse": wb["rmse_median"], "picp": wb["picp"]}
            key = f"transfer_{sur}_{inten:.1f}"
            for name, fn in targets.items():
                q = inv(fn(ctx_adv)).reshape(-1, len(QUANTILES))
                out["targets"][name][key] = _prob_scores(y_flat, q)
            out["targets"]["naive_seasonal"][key] = {"rmse": naive_rmse(ctx_adv)}
            row = "  ".join(f"{n}:{out['targets'][n][key]['rmse']:.2f}"
                            for n in out["targets"])
            print(f"{sur:13s} {setting}  target RMSE  {row}  "
                  f"(surrogate white-box {wb['rmse_median']:.2f})")

    out_path = ROOT / "results" / "base" / "fgsm_transfer.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")

    # ---------------- markdown table ---------------------------------------
    tables = ROOT / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    names = list(out["targets"])
    lines = ["# Transfer-FGSM — gradient-free targets under surrogate attacks\n",
             "RMSE (°C); PICP in parentheses for quantile targets. The attack "
             "is crafted on the surrogate's gradient and replayed verbatim on "
             "the target.\n",
             "| Setting | " + " | ".join(names) + " | surrogate white-box |",
             "|---|" + "---|" * (len(names) + 1)]

    def _cell(block):
        if block is None:
            return "-"
        return (f"{block['rmse']:.2f} ({block['picp']:.2f})"
                if "picp" in block else f"{block['rmse']:.2f}")

    lines.append("| clean | " + " | ".join(
        _cell(out["targets"][n].get("clean")) for n in names) + " | — |")
    for sur in SURROGATES:
        for inten in INTENSITIES:
            key = f"transfer_{sur}_{inten:.1f}"
            if key not in out["targets"][names[0]]:
                continue
            wb = out["surrogate_whitebox"][sur][f"fgsm_{inten:.1f}"]
            lines.append(f"| {sur} fgsm {inten:.0f}× | " + " | ".join(
                _cell(out["targets"][n].get(key)) for n in names)
                + f" | {wb['rmse']:.2f} ({wb['picp']:.2f}) |")
    (tables / "fgsm_transfer.md").write_text("\n".join(lines) + "\n")
    print(f"table -> {tables / 'fgsm_transfer.md'}")


if __name__ == "__main__":
    main()
