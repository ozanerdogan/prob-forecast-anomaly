"""Phase-3 forward-chaining CV: is 2016 special, or is the result stable?

The headline protocol keeps the single chronological split; this is a SIDE
study. Each year 2013-2016 is used as the test fold with an expanding train
window (everything up to the prior year) and the preceding year as validation.
The standardiser is refit per fold on that fold's train only (no leakage).
Reports per-fold RMSE/CRPS and the fold variance.

Expanding-train confounds year effect with train-size; a fixed-width sliding
train would isolate the year effect but waste data. We report expanding (the
deployment-realistic choice) and note the confound.

  python scripts/ablation/run_cv.py            # 4 folds x {naive, lstm, qlstm, qt}
  python scripts/ablation/run_cv.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.baselines.lstm_baseline import LstmConfig, train_lstm  # noqa: E402
from src.baselines.naive_seasonal import naive_seasonal_forecast  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src.metrics import report, report_probabilistic  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.models.quantile_transformer import (  # noqa: E402
    QTransformerConfig,
    predict_quantiles,
    train_qtransformer,
)
from src.preprocessing import Standardizer, TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 6  # reduced, shared across folds for budget
TEST_YEARS = (2013, 2014, 2015, 2016)


def _fold_frames(feat: pd.DataFrame, test_year: int):
    train = feat.loc[: f"{test_year - 2}-12-31 23:00:00"]
    val = feat.loc[f"{test_year - 1}-01-01": f"{test_year - 1}-12-31 23:00:00"]
    test = feat.loc[f"{test_year}-01-01": f"{test_year}-12-31 23:00:00"]
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    years = TEST_YEARS[-1:] if args.smoke else TEST_YEARS

    df = load_hourly(ROOT / "data" / "processed")
    feat = build_feature_frame(df, TARGET, use_covariates=False)
    L, H = 168, 24

    out: dict = {"epochs": EPOCHS, "test_years": list(years),
                 "scheme": "expanding train, prior year as val; note: year effect "
                           "confounded with train size",
                 "smoke": bool(args.smoke), "models": {}}
    model_names = ("naive", "lstm", "qlstm", "qt")
    for name in model_names:
        out["models"][name] = {"folds": {}}

    for ty in years:
        tr_df, va_df, te_df = _fold_frames(feat, ty)
        if len(tr_df) < L + H + 100:
            print(f"  skip {ty}: train too short ({len(tr_df)})")
            continue
        scaler = Standardizer.fit(tr_df)
        tr = scaler.transform(tr_df).to_numpy().astype(np.float32)
        va = scaler.transform(va_df).to_numpy().astype(np.float32)
        te = scaler.transform(te_df).to_numpy().astype(np.float32)
        x_tr, y_tr = make_encoder_windows(tr, L, H, stride=1)
        x_va, y_va = make_encoder_windows(va, L, H, stride=H)
        x_te, y_te = make_encoder_windows(te, L, H, stride=H)
        if args.smoke:
            x_tr, y_tr = x_tr[:1500], y_tr[:1500]
        inv = lambda a: scaler.inverse_target(a, TARGET)  # noqa: E731
        y_flat = inv(y_te).reshape(-1)
        train_raw = tr_df[TARGET].to_numpy()
        print(f"=== test {ty}: train {len(tr_df)}h val {len(va_df)}h test {len(te_df)}h ===")

        # naive
        pred = np.stack([naive_seasonal_forecast(x_te[i, :, 0], H, 24) for i in range(len(x_te))])
        out["models"]["naive"]["folds"][str(ty)] = {"rmse": report(y_flat, inv(pred).reshape(-1))["rmse"]}

        # lstm
        cfg = LstmConfig(epochs=1 if args.smoke else EPOCHS)
        m, _ = train_lstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
        from src import experiment as E
        out["models"]["lstm"]["folds"][str(ty)] = {
            "rmse": report(y_flat, inv(E.lstm_predict(m, x_te[:, :, 0])).reshape(-1))["rmse"]}

        # qlstm
        cfg = QLstmConfig(epochs=1 if args.smoke else EPOCHS)
        m, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
        q = inv(predict_qlstm(m, x_te[:, :, 0], device=DEVICE))
        rep = report_probabilistic(y_flat, q.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
        out["models"]["qlstm"]["folds"][str(ty)] = {"rmse": rep["rmse"], "crps": rep["crps"], "picp": rep["picp"]}

        # qt
        cfg = QTransformerConfig(n_features=1, quantiles=QUANTILES_7, epochs=1 if args.smoke else EPOCHS)
        m, _ = train_qtransformer(x_tr, y_tr, x_va, y_va, cfg, device=DEVICE)
        q = inv(predict_quantiles(m, x_te, device=DEVICE))
        rep = report_probabilistic(y_flat, q.reshape(-1, len(QUANTILES)), QUANTILES, alpha=ALPHA)
        out["models"]["qt"]["folds"][str(ty)] = {"rmse": rep["rmse"], "crps": rep["crps"], "picp": rep["picp"]}

        for nm in model_names:
            print(f"  {nm:6s} rmse {out['models'][nm]['folds'][str(ty)]['rmse']:.3f}")

    for nm in model_names:
        rmses = [f["rmse"] for f in out["models"][nm]["folds"].values()]
        if rmses:
            out["models"][nm]["rmse_mean"] = float(np.mean(rmses))
            out["models"][nm]["rmse_std"] = float(np.std(rmses))
    out_path = ROOT / "results" / "base" / "cv_forward_chaining.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
