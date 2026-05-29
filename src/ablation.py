"""Ablation study over the design choices of the probabilistic models.

Four axes, each varying one factor from a common base while holding everything
else fixed, so the effect is attributable:

  - input:      target-only vs multivariate (exogenous covariates)   [both models]
  - lookback:   L in {72, 168, 336}                                  [Transformer]
  - likelihood: Gaussian vs Student-t                                [DeepAR]
  - quantiles:  3 vs 7 quantile levels                               [Transformer]

To run every variant within a sane budget we use a reduced epoch count (shared
across all variants so comparisons stay fair). All variants are scored on the
clean test set with the full probabilistic + point metric block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import experiment as E
from src.metrics import mase, report_probabilistic, smape
from src.models.deepar import DeepARConfig, quantiles_from_samples, sample_forecast
from src.models.quantile_transformer import (
    QTransformerConfig,
    QUANTILES_3,
    QUANTILES_7,
    predict_quantiles,
)
from src.preprocessing import TARGET
from src.seq_data import make_ar_windows, make_encoder_windows

ALPHA = 0.1


@dataclass
class Variant:
    name: str
    axis: str
    model: str  # "deepar" | "qtransformer"
    use_covariates: bool = False
    lookback: int = 168
    likelihood: str = "gaussian"
    quantiles: tuple[float, ...] = QUANTILES_7
    epochs: int = 6


def default_variants(epochs: int = 6) -> list[Variant]:
    v: list[Variant] = []
    # axis 1: input richness (both models)
    for model in ("deepar", "qtransformer"):
        v.append(Variant(f"{model}_target_only", "input", model, use_covariates=False, epochs=epochs))
        v.append(Variant(f"{model}_multivariate", "input", model, use_covariates=True, epochs=epochs))
    # axis 2: lookback (Transformer)
    for L in (72, 168, 336):
        v.append(Variant(f"qt_lookback_{L}", "lookback", "qtransformer", lookback=L, epochs=epochs))
    # axis 3: likelihood (DeepAR)
    for lik in ("gaussian", "studentt"):
        v.append(Variant(f"deepar_{lik}", "likelihood", "deepar", likelihood=lik, epochs=epochs))
    # axis 4: quantile-set size (Transformer)
    v.append(Variant("qt_quantiles_3", "quantiles", "qtransformer", quantiles=QUANTILES_3, epochs=epochs))
    v.append(Variant("qt_quantiles_7", "quantiles", "qtransformer", quantiles=QUANTILES_7, epochs=epochs))
    return v


def run_variant(spec: Variant) -> dict:
    """Train and evaluate a single ablation variant on the clean test set."""
    data = E.prepare(use_covariates=spec.use_covariates)
    quantiles = np.array(spec.quantiles)
    H = 24
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731

    if spec.model == "deepar":
        cfg = DeepARConfig(
            lookback=spec.lookback, horizon=H,
            n_covariates=data.n_features - 1, likelihood=spec.likelihood,
            epochs=spec.epochs,
        )
        model = E.fit_deepar(data, cfg)
        yseq_te, cov_te = make_ar_windows(data.test, cfg.lookback, H, stride=H)
        samples = sample_forecast(model, yseq_te, cov_te, cfg, device=E.DEVICE, batch_size=128)
        q_preds = quantiles_from_samples(samples, quantiles)
        y_true = inv(yseq_te[:, cfg.lookback :])
    elif spec.model == "qtransformer":
        cfg = QTransformerConfig(
            lookback=spec.lookback, horizon=H,
            n_features=data.n_features, quantiles=spec.quantiles, epochs=spec.epochs,
        )
        model = E.fit_qtransformer(data, cfg)
        x_te, y_te = make_encoder_windows(data.test, cfg.lookback, H, stride=H)
        q_preds = predict_quantiles(model, x_te, device=E.DEVICE)
        y_true = inv(y_te)
    else:
        raise ValueError(spec.model)

    q_preds = inv(q_preds)
    q_flat = q_preds.reshape(-1, len(quantiles))
    y_flat = y_true.reshape(-1)
    med = q_flat[:, int(np.argmin(np.abs(quantiles - 0.5)))]

    out = report_probabilistic(y_flat, q_flat, quantiles, alpha=ALPHA)
    out["smape"] = smape(y_flat, med)
    out["mase"] = mase(y_flat, med, data.train_target_raw, season=24)
    out.update(
        name=spec.name, axis=spec.axis, model=spec.model,
        use_covariates=spec.use_covariates, lookback=spec.lookback,
        likelihood=spec.likelihood, n_quantiles=len(spec.quantiles),
        n_features=data.n_features, epochs=spec.epochs,
        n_predictions=int(len(y_flat)),
    )
    return out


def comparison_table(results: list[dict]) -> list[dict]:
    """Flatten the key metrics per variant, grouped by axis, for a quick table."""
    keys = ("crps", "pinball", "picp", "mpiw", "mis", "rmse", "mae")
    table = []
    for r in results:
        row = {"name": r["name"], "axis": r["axis"], "model": r["model"]}
        row.update({k: round(r[k], 4) for k in keys})
        table.append(row)
    table.sort(key=lambda x: (x["axis"], x["name"]))
    return table
