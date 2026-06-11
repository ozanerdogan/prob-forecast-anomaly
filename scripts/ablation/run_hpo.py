"""Phase-3 HPO: small grids, selection STRICTLY on validation.

Default configs were a deliberate fair-comparison stance; this measures what
tuning buys. Per model a small grid is trained on train and ranked by the
final-epoch validation loss (MSE for the LSTM, pinball for the quantile
models); ONLY the winner is then evaluated on test, next to the already-
published default-config test scores (read from results/base/*.json, not
retrained). Attribution rule from implementation.md: HPO never changes
simultaneously with the stage-2 calibration comparison.

  python scripts/ablation/run_hpo.py            # ~1h GPU (20 trainings)
  python scripts/ablation/run_hpo.py --smoke
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
from src.baselines.lstm_baseline import LstmConfig, train_lstm  # noqa: E402
from src.metrics import report, report_probabilistic  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.models.quantile_transformer import (  # noqa: E402
    QTransformerConfig,
    predict_quantiles,
    train_qtransformer,
)
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRIDS = {
    "lstm": [dict(hidden_size=h, num_layers=nl, lr=lr)
             for h in (64, 128) for nl in (1, 2) for lr in (1e-3, 3e-4)],
    "qlstm": [dict(hidden_size=h, num_layers=nl, lr=lr)
              for h in (64, 128) for nl in (1, 2) for lr in (1e-3, 3e-4)],
    "qt": [dict(d_model=d, num_layers=nl, dim_ff=2 * d)
           for d in (64, 128) for nl in (2, 3)],
}
DEFAULT_JSON = {"lstm": "lstm.json", "qlstm": "qlstm.json", "qt": "qtransformer.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", nargs="*", default=("lstm", "qlstm", "qt"))
    args = ap.parse_args()

    data = E.prepare(use_covariates=False)
    L, H = 168, 24
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_true_flat = inv(y_te).reshape(-1)

    out: dict = {"selection": "final-epoch validation loss (test untouched until the winner)",
                 "smoke": bool(args.smoke), "models": {}}

    for model_name in args.models:
        print(f"=== HPO {model_name} ({len(GRIDS[model_name])} configs) ===")
        trials = []
        for gi, params in enumerate(GRIDS[model_name]):
            if model_name in ("lstm", "qlstm"):
                CfgCls = LstmConfig if model_name == "lstm" else QLstmConfig
                cfg = CfgCls(**params)
                if args.smoke:
                    cfg.epochs = 1
                train_fn = train_lstm if model_name == "lstm" else train_qlstm
                _, hist = train_fn(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
                val_loss = hist[-1].get("val_loss", hist[-1].get("val_pinball"))
            else:  # qt
                cfg = QTransformerConfig(n_features=1, quantiles=QUANTILES_7, **params)
                if args.smoke:
                    cfg.epochs = 1
                _, hist = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE)
                val_loss = hist[-1]["val_pinball"]
            trials.append({"params": params, "val_loss": float(val_loss)})
            print(f"  [{gi + 1}/{len(GRIDS[model_name])}] {params} -> val {val_loss:.4f}")

        best = min(trials, key=lambda t: t["val_loss"])
        print(f"  WINNER {best['params']} (val {best['val_loss']:.4f}); evaluating on test ...")

        # retrain winner (same seed) and touch test exactly once
        if model_name in ("lstm", "qlstm"):
            CfgCls = LstmConfig if model_name == "lstm" else QLstmConfig
            cfg = CfgCls(**best["params"])
            if args.smoke:
                cfg.epochs = 1
            if model_name == "lstm":
                model, _ = train_lstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
                pred = inv(E.lstm_predict(model, x_te[:, :, 0]))
                test_best = {"rmse": report(y_true_flat, pred.reshape(-1))["rmse"], "crps": None}
            else:
                model, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
                q = inv(predict_qlstm(model, x_te[:, :, 0], device=DEVICE))
                rep = report_probabilistic(y_true_flat, q.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
                test_best = {"rmse": rep["rmse"], "crps": rep["crps"], "picp": rep["picp"]}
        else:
            cfg = QTransformerConfig(n_features=1, quantiles=QUANTILES_7, **best["params"])
            if args.smoke:
                cfg.epochs = 1
            model, _ = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE)
            q = inv(predict_quantiles(model, x_te, device=DEVICE))
            rep = report_probabilistic(y_true_flat, q.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
            test_best = {"rmse": rep["rmse"], "crps": rep["crps"], "picp": rep["picp"]}

        dflt = json.loads((ROOT / "results" / "base" / DEFAULT_JSON[model_name]).read_text())
        out["models"][model_name] = {
            "trials": trials,
            "best_params": best["params"],
            "best_val_loss": best["val_loss"],
            "test_rmse_default": dflt.get("rmse"),
            "test_rmse_best": test_best["rmse"],
            "test_crps_default": dflt.get("crps"),
            "test_crps_best": test_best.get("crps"),
            "test_picp_best": test_best.get("picp"),
        }

    out_path = ROOT / "results" / "base" / "hpo.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
