"""Multi-channel FGSM on the multivariate QTransformer (Asama-2).

Every prior FGSM result perturbs the TARGET channel only. The multivariate
QT also reads 5 independent sensors + calendar features, so the natural
question is how much stronger the attack gets when the covariate channels
are perturbed too. The attack surface is physical: sensor channels (target +
5 weather covariates) are attackable, calendar channels are not (an attacker
cannot change the date). Per-channel radius = intensity x that channel's
local sigma.

Trains the official qtransformer_multi (independent covset, same config as
scripts/models/run_qtransformer_multi.py) and compares, at each intensity:
clean / target-only FGSM / multi-channel FGSM.

  python scripts/analysis/run_fgsm_multichannel.py
  python scripts/analysis/run_fgsm_multichannel.py --smoke
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
from src.anomaly import linf_fgsm, linf_fgsm_multichannel  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.features import build_feature_frame  # noqa: E402
from src.metrics import report_probabilistic  # noqa: E402
from src.models.quantile_transformer import QTransformerConfig, QUANTILES_7  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_encoder_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
INTENSITIES = (1.0, 2.0, 4.0)
INDEP = ["p (mbar)", "rh (%)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"]


def _scores(y_flat, q_flat):
    out = report_probabilistic(y_flat, q_flat, QUANTILES, alpha=ALPHA)
    return {"rmse": out["rmse"], "crps": out["crps"], "picp": out["picp"],
            "mis": out["mis"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data = E.prepare(use_covariates=True, covariate_cols=INDEP)
    cols = list(build_feature_frame(load_hourly(ROOT / "data" / "processed"),
                                    TARGET, use_covariates=True,
                                    covariate_cols=INDEP).columns)
    sensor = [c == TARGET or c in INDEP for c in cols]
    cfg = QTransformerConfig(n_features=data.n_features, quantiles=QUANTILES_7)
    if args.smoke:
        cfg.epochs = 1

    print(f"Training multivariate QT ({data.n_features} ch, {cfg.epochs} ep); "
          f"attackable channels: {[c for c, m in zip(cols, sensor) if m]}")
    model = E.fit_qtransformer(data, cfg)

    x_te, y_te = make_encoder_windows(data.test, cfg.lookback, cfg.horizon,
                                      stride=cfg.horizon)
    if args.smoke:
        x_te, y_te = x_te[:60], y_te[:60]
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_flat = inv(y_te).reshape(-1)

    def evaluate(x):
        q = inv(E.qtransformer_predict(model, x)).reshape(-1, len(QUANTILES))
        return _scores(y_flat, q)

    out = {"model": "qtransformer_multi", "channels": cols,
           "attackable": [c for c, m in zip(cols, sensor) if m],
           "alpha": ALPHA, "smoke": bool(args.smoke),
           "clean": evaluate(x_te), "settings": {}}
    print(f"clean: RMSE {out['clean']['rmse']:.3f} PICP {out['clean']['picp']:.3f}")

    grad = E.qtransformer_full_grad(model, x_te, y_te, QUANTILES)
    for inten in INTENSITIES:
        ctx_t, _ = linf_fgsm(x_te[:, :, 0], grad[:, :, 0], inten)
        x_t = x_te.copy()
        x_t[:, :, 0] = ctx_t
        target_only = evaluate(x_t)

        x_mc, _ = linf_fgsm_multichannel(x_te, grad, inten, channel_mask=sensor)
        multi = evaluate(x_mc)

        out["settings"][f"{inten:.1f}"] = {"target_only": target_only,
                                           "multichannel": multi}
        print(f"fgsm {inten:.0f}x  target-only RMSE {target_only['rmse']:.2f} "
              f"PICP {target_only['picp']:.3f}  |  multi-channel RMSE "
              f"{multi['rmse']:.2f} PICP {multi['picp']:.3f}")

    out_path = ROOT / "results" / "base" / "fgsm_multichannel.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved -> {out_path}")

    tables = ROOT / "report" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    lines = ["# Multi-channel FGSM — qtransformer_multi\n",
             "Attack surface: target + 5 independent sensors (calendar "
             "channels are not attackable). Per-channel radius = intensity × "
             "channel-local sigma.\n",
             "| Intensity | target-only RMSE | target-only PICP | "
             "multi-channel RMSE | multi-channel PICP |", "|---|---|---|---|---|",
             f"| clean | {out['clean']['rmse']:.2f} | {out['clean']['picp']:.3f} "
             f"| — | — |"]
    for inten, b in out["settings"].items():
        lines.append(f"| {float(inten):.0f}× | {b['target_only']['rmse']:.2f} | "
                     f"{b['target_only']['picp']:.3f} | "
                     f"{b['multichannel']['rmse']:.2f} | "
                     f"{b['multichannel']['picp']:.3f} |")
    (tables / "fgsm_multichannel.md").write_text("\n".join(lines) + "\n")
    print(f"table -> {tables / 'fgsm_multichannel.md'}")


if __name__ == "__main__":
    main()
