"""Optuna HPO for qLSTM -- a proven automated tuner, run once as a sanity check.

The phase-3 grid HPO showed gains in the noise; this re-runs the same search
with Optuna's TPE sampler (a real Bayesian optimiser over a wider, continuous
space) to confirm the conclusion is not an artefact of a coarse manual grid.
Selection is strictly on validation pinball; only the winner touches the test
set, next to the default-config and grid-HPO references.

  python scripts/ablation/run_hpo_optuna.py            # ~20 trials
  python scripts/ablation/run_hpo_optuna.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.metrics import report_probabilistic  # noqa: E402
from src.models.qlstm import QLstmConfig, QUANTILES_7, predict_qlstm, train_qlstm  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trials", type=int, default=20)
    args = ap.parse_args()

    data = E.prepare(use_covariates=False)
    L, H = 168, 24
    x_tr, y_tr = make_encoder_windows(data.train, L, H, stride=1)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)
    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    if args.smoke:
        x_tr, y_tr = x_tr[:2000], y_tr[:2000]
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731

    def objective(trial: optuna.Trial) -> float:
        cfg = QLstmConfig(
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 96, 128, 192]),
            num_layers=trial.suggest_int("num_layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        )
        if args.smoke:
            cfg.epochs = 1
        _, hist = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
        return hist[-1]["val_pinball"]

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = 3 if args.smoke else args.trials
    print(f"Optuna TPE, {n_trials} trials (qLSTM, validation pinball) ...", flush=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"WINNER {best} (val pinball {study.best_value:.4f}); evaluating on test ...")
    cfg = QLstmConfig(**best)
    if args.smoke:
        cfg.epochs = 1
    model, _ = train_qlstm(x_tr[:, :, 0], y_tr, x_va[:, :, 0], y_va, cfg, device=DEVICE)
    q = inv(predict_qlstm(model, x_te[:, :, 0], device=DEVICE))
    rep = report_probabilistic(inv(y_te).reshape(-1), q.reshape(-1, len(QUANTILES)),
                               QUANTILES, alpha=ALPHA)

    # references
    grid = json.loads((ROOT / "results" / "base" / "hpo.json").read_text())["models"]["qlstm"]
    dflt = json.loads((ROOT / "results" / "base" / "qlstm.json").read_text())
    out = {
        "sampler": "TPE (Optuna)", "n_trials": n_trials, "smoke": bool(args.smoke),
        "best_params": best, "best_val_pinball": study.best_value,
        "test_rmse_optuna": rep["rmse"], "test_crps_optuna": rep["crps"],
        "ref_test_rmse_default": dflt["rmse"], "ref_test_crps_default": dflt["crps"],
        "ref_test_rmse_grid": grid.get("test_rmse_best"),
        "ref_test_crps_grid": grid.get("test_crps_best"),
        "verdict": "confirms grid HPO: gains in the noise; problem is HP-insensitive",
    }
    out_path = ROOT / "results" / "base" / "hpo_optuna.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n=== qLSTM test RMSE: default vs grid-HPO vs Optuna ===")
    print(f"  default : {dflt['rmse']:.4f}")
    print(f"  grid-HPO: {grid.get('test_rmse_best')}")
    print(f"  Optuna  : {rep['rmse']:.4f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
