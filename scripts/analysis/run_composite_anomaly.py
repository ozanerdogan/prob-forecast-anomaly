"""Phase-3 composite anomalies: do overlapping faults compound?

Roadmap item F: more than one anomaly in the same window. We inject a
flatline (stuck tail) followed by a point spike inside it, and compare the
combined degradation against each fault alone, for the deterministic LSTM and
the probabilistic qLSTM. Reuses the trained phase-2 models? No -- to stay
self-contained it retrains both at the shared budget (cheap recurrent models).

  python scripts/analysis/run_composite_anomaly.py
  python scripts/analysis/run_composite_anomaly.py --smoke
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
from src.anomaly import inject_flatline, inject_point_spike  # noqa: E402
from src.baselines.lstm_baseline import LstmConfig  # noqa: E402
from src.metrics import report, report_probabilistic  # noqa: E402
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
    ctx0 = x_te[:, :, 0].copy()

    def variants():
        out = {"clean": ctx0}
        rng = np.random.default_rng(SEED)
        out["flatline"], _ = inject_flatline(ctx0, 2.0, rng)
        rng = np.random.default_rng(SEED)
        out["spike"], _ = inject_point_spike(ctx0, 4.0, rng)
        # composite: flatline first, then a spike landing in the live region
        rng = np.random.default_rng(SEED)
        fl, _ = inject_flatline(ctx0, 2.0, rng)
        comp, _ = inject_point_spike(fl, 4.0, rng)
        out["flatline+spike"] = comp
        return out

    lstm_cfg = LstmConfig(seed=SEED)
    qcfg = QLstmConfig(seed=SEED)
    if args.smoke:
        lstm_cfg.epochs = qcfg.epochs = 1
    print("Training LSTM + qLSTM ...")
    lstm = E.fit_lstm(data, lstm_cfg)
    qlstm, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, qcfg, device=DEVICE)

    out = {"seed": SEED, "smoke": bool(args.smoke), "settings": {}}
    for name, ctx in variants().items():
        p = inv(E.lstm_predict(lstm, ctx)).reshape(-1)
        q = inv(predict_qlstm(qlstm, ctx, device=DEVICE))
        rp = report_probabilistic(y_flat, q.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
        out["settings"][name] = {
            "lstm_rmse": report(y_flat, p)["rmse"],
            "qlstm_rmse": rp["rmse"], "qlstm_picp": rp["picp"], "qlstm_crps": rp["crps"]}
        print(f"  {name:16s} LSTM rmse {out['settings'][name]['lstm_rmse']:.2f}"
              f"  qLSTM rmse {rp['rmse']:.2f} picp {rp['picp']:.3f}")

    s = out["settings"]
    out["compounding"] = {
        "lstm_rmse_sum_alone_vs_combined": {
            "flatline": s["flatline"]["lstm_rmse"], "spike": s["spike"]["lstm_rmse"],
            "combined": s["flatline+spike"]["lstm_rmse"], "clean": s["clean"]["lstm_rmse"]},
    }
    out_path = ROOT / "results" / "base" / "composite_anomaly.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
