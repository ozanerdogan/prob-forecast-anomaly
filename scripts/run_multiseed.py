"""Phase-3 multi-seed: are the near-ties real or seed noise?

Phase-1/2 left several models inside a ~0.05 RMSE band (LSTM 2.429, GRU 2.386,
qLSTM-median 2.384, QT 2.430). Each model is retrained on 3 seeds and the
mean +/- std reported, with the seed-42 ranking checked for stability.

  python scripts/run_multiseed.py            # 5 models x 3 seeds = 15 trainings
  python scripts/run_multiseed.py --smoke
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
from src.baselines.lstm_baseline import LstmConfig, train_lstm  # noqa: E402
from src.metrics import report, report_probabilistic  # noqa: E402
from src.models.dlinear import DLinearConfig, predict_dlinear, train_dlinear  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
SEEDS = (42, 7, 2025)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = SEEDS[:1] if args.smoke else SEEDS

    data = E.prepare(use_covariates=False)
    L, H = 168, 24
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_flat = inv(y_te).reshape(-1)

    def rmse_of(pred):
        return report(y_flat, inv(pred).reshape(-1))["rmse"]

    def prob_of(q):
        r = report_probabilistic(y_flat, inv(q).reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
        return r["rmse"], r["crps"]

    out: dict = {"seeds": list(seeds), "smoke": bool(args.smoke), "models": {}}
    for name in ("lstm", "gru", "qlstm", "qdlinear"):
        runs = []
        for sd in seeds:
            if name in ("lstm", "gru"):
                cfg = LstmConfig(seed=sd, cell=("gru" if name == "gru" else "lstm"))
                if args.smoke:
                    cfg.epochs = 1
                m, _ = train_lstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
                runs.append({"seed": sd, "rmse": rmse_of(E.lstm_predict(m, x_te[:, :, 0]))})
            elif name == "qlstm":
                cfg = QLstmConfig(seed=sd)
                if args.smoke:
                    cfg.epochs = 1
                m, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
                r, c = prob_of(predict_qlstm(m, x_te[:, :, 0], device=DEVICE))
                runs.append({"seed": sd, "rmse": r, "crps": c})
            else:  # qdlinear
                cfg = DLinearConfig(quantiles=QUANTILES_7, seed=sd)
                if args.smoke:
                    cfg.epochs = 1
                m, _ = train_dlinear(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
                r, c = prob_of(predict_dlinear(m, x_te[:, :, 0], device=DEVICE))
                runs.append({"seed": sd, "rmse": r, "crps": c})
            print(f"  {name:9s} seed {sd:>4} rmse {runs[-1]['rmse']:.4f}")
        rmses = [r["rmse"] for r in runs]
        out["models"][name] = {"runs": runs, "rmse_mean": float(np.mean(rmses)),
                               "rmse_std": float(np.std(rmses))}

    # ranking stability per seed
    out["ranking_per_seed"] = {
        str(i): sorted(out["models"], key=lambda m: out["models"][m]["runs"][i]["rmse"])
        for i in range(len(seeds))
    }
    out_path = ROOT / "results" / "base" / "multiseed.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({m: f"{b['rmse_mean']:.3f}±{b['rmse_std']:.3f}"
                      for m, b in out["models"].items()}, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
