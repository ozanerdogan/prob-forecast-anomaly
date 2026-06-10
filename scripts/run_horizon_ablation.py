"""Horizon ablation: how does forecast quality decay with how far ahead we predict?

The headline task is 24h-ahead. This trains the quantile Transformer to forecast
{12, 24, 48}h ahead from the same 168h context and reports clean-test
CRPS/RMSE/PICP, so the 24h choice is justified (and the 48h degradation
quantified). Same backbone/budget across horizons -> the effect is attributable
to the horizon alone.

  python scripts/run_horizon_ablation.py
  python scripts/run_horizon_ablation.py --smoke
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
from src.models.quantile_transformer import (  # noqa: E402
    QTransformerConfig,
    QUANTILES_7,
    predict_quantiles,
    train_qtransformer,
)
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZONS = (12, 24, 48, 168)
L = 168


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data = E.prepare(use_covariates=False)
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731

    out = {"lookback": L, "horizons": list(HORIZONS), "smoke": bool(args.smoke), "results": {}}
    for H in HORIZONS:
        x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
        x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
        x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
        if args.smoke:
            x_tr, y_tr = x_tr[:2000], y_tr[:2000]
        cfg = QTransformerConfig(n_features=1, horizon=H, quantiles=QUANTILES_7)
        if args.smoke:
            cfg.epochs = 1
        print(f"Training QT horizon={H}h ...")
        model, _ = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE)
        q = inv(predict_quantiles(model, x_te, device=DEVICE)).reshape(-1, len(QUANTILES))
        r = report_probabilistic(inv(y_te).reshape(-1), q, QUANTILES, alpha=ALPHA)
        out["results"][f"{H}h"] = {"crps": r["crps"], "rmse": r["rmse"], "picp": r["picp"],
                                   "mis": r["mis"]}
        print(f"  {H}h: CRPS {r['crps']:.3f}  RMSE {r['rmse']:.2f}  PICP {r['picp']:.3f}")

    out_path = ROOT / "results" / "base" / "horizon_ablation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
