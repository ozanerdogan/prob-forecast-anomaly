"""Probe: why does the leakage-free covariate setting HURT DeepAR?

Ablation facts (shared 6-epoch budget): target-only CRPS 1.463 vs
past-covariate (weather frozen at origin) 1.904 with PICP collapsing to 0.55.
Two candidate explanations:

  (a) the frozen-persistence WEATHER channels feed the autoregressive rollout
      stale inputs it learned to trust -> the damage is the freeze;
  (b) the covariate pathway itself (extra channels) destabilises the rollout.

Discriminating experiment: CALENDAR-ONLY covariates — genuinely known over
the horizon (no freeze needed, no leakage possible). If calendar-only lands
near target-only, (a) holds: covariates are fine, *frozen weather* is the
problem. If it lands near past-covariate, (b) holds.

Same 6-epoch budget as the ablation so all four numbers are comparable.
Writes results/base/deepar_covariate_probe.json.

  python scripts/probe_deepar_covariates.py
  python scripts/probe_deepar_covariates.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import experiment as E  # noqa: E402
from src.metrics import report_probabilistic  # noqa: E402
from src.models.deepar import DeepARConfig, quantiles_from_samples, sample_forecast  # noqa: E402
from src.models.quantile_transformer import QUANTILES_7  # noqa: E402
from src.preprocessing import TARGET  # noqa: E402
from src.seq_data import make_ar_windows  # noqa: E402

QUANTILES = np.array(QUANTILES_7)
ALPHA = 0.1
EPOCHS = 6  # ablation budget -> numbers comparable with results/ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    # calendar-only covariates: pass an empty weather list -> frame is
    # [target, hour_sin, hour_cos, doy_sin, doy_cos]
    data = E.prepare(use_covariates=True, covariate_cols=[])
    cfg = DeepARConfig(n_covariates=data.n_features - 1,
                       epochs=1 if args.smoke else EPOCHS)
    print(f"Training DeepAR calendar-only ({data.n_features - 1} covariates, "
          f"{cfg.epochs} epochs) ...")
    model = E.fit_deepar(data, cfg)

    yseq_te, cov_te = make_ar_windows(data.test, cfg.lookback, cfg.horizon, stride=cfg.horizon)
    if args.smoke:
        yseq_te, cov_te = yseq_te[:60], cov_te[:60]
    samples = sample_forecast(model, yseq_te, cov_te, cfg, device=E.DEVICE, batch_size=128)
    q = quantiles_from_samples(samples, QUANTILES)

    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731
    y_flat = inv(yseq_te[:, cfg.lookback:]).reshape(-1)
    q_flat = inv(q).reshape(-1, len(QUANTILES))
    scores = report_probabilistic(y_flat, q_flat, QUANTILES, alpha=ALPHA)

    # ablation references at the same budget
    refs = {}
    abl = json.loads((ROOT / "results" / "ablation.json").read_text())
    for v in abl["variants"]:
        if v["name"] in ("deepar_target_only", "deepar_past_covariate", "deepar_multivariate"):
            refs[v["name"]] = {k: v[k] for k in ("crps", "picp", "rmse", "mis")}

    crps = scores["crps"]
    t_only = refs.get("deepar_target_only", {}).get("crps")
    past = refs.get("deepar_past_covariate", {}).get("crps")
    if t_only is not None and past is not None:
        verdict = (
            "(a) frozen-weather is the culprit: calendar-only covariates track "
            "target-only, so the covariate pathway itself is fine"
            if abs(crps - t_only) < abs(crps - past) else
            "(b) the covariate pathway itself destabilises the rollout: even "
            "leak-free calendar covariates reproduce the degradation"
        )
    else:
        verdict = "ablation references missing; rerun scripts/run_ablation.py"

    out = {
        "probe": "deepar_calendar_only",
        "epochs": cfg.epochs,
        "n_covariates": cfg.n_covariates,
        "scores": {k: scores[k] for k in ("crps", "pinball", "picp", "mpiw", "mis", "rmse", "mae")},
        "ablation_refs_same_budget": refs,
        "verdict": verdict,
        "smoke": bool(args.smoke),
    }
    out_path = ROOT / "results" / "base" / "deepar_covariate_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
