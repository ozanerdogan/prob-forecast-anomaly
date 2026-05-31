"""Produce progress-report figures.

Phase-1 figures are written as PDF (vector) into ``results/figures/``; Phase-2
figures are written as PNG into the same directory.

Phase-1 figures (PDF):
  1. jena_series_split.pdf       -- target series + train/val/test markers
  2. lstm_loss_curves.pdf        -- LSTM train/val MSE per epoch
  3. naive_sample_window.pdf     -- 7-day test window: truth vs naive seasonal
  4. baseline_comparison.pdf     -- bar chart of RMSE & MAE per baseline
  5. seasonal_decomposition.pdf  -- STL decomposition on a 14-day train window

Phase-2 figures (PNG):
  6. forecast_intervals.png      -- prediction-interval forecast, clean vs anomaly
  7. reliability_curve.png       -- nominal vs empirical coverage (pre/post calib)
  8. pit_histogram.png           -- PIT histogram (calibration sanity check)
  9. per_horizon_curves.png      -- RMSE & CRPS vs forecast horizon
 10. robustness_heatmap.png      -- model x anomaly-type/intensity RMSE inflation
 11. attention_map.png           -- Transformer first-layer self-attention

Phase-2 figures train the probabilistic models (DeepAR + quantile Transformer)
on the fly; run the Phase-1 baselines and Phase-2 result scripts first so the
JSONs the aggregate figures read are present. ``--phase 2`` writes only the PNGs
and leaves the Phase-1 PDFs untouched; ``--phase 1`` / ``--phase all`` regenerate
(overwrite) the PDFs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.naive_seasonal import naive_seasonal_forecast  # noqa: E402
from src.data_loader import load_hourly  # noqa: E402
from src.preprocessing import TARGET, TRAIN_END, VAL_END, chronological_split  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_series_split() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    series = df[TARGET]

    daily = series.resample("1D").mean()

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.plot(daily.index, daily.values, lw=0.6, color="#1f4e79")
    ax.axvline(pd.Timestamp(TRAIN_END), color="#c0392b", lw=0.9, ls="--", label="train / val")
    ax.axvline(pd.Timestamp(VAL_END), color="#27ae60", lw=0.9, ls="--", label="val / test")
    ax.set_title("Jena Climate -- daily mean of T (degC)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature (degC)")
    ax.legend(loc="lower center", ncol=2, frameon=False)
    ax.margins(x=0.0)
    out = FIG_DIR / "jena_series_split.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_lstm_loss_curves() -> None:
    metrics = json.loads((ROOT / "results" / "lstm.json").read_text())
    history = metrics["history"]
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.plot(epochs, train_loss, marker="o", color="#1f4e79", label="train")
    ax.plot(epochs, val_loss, marker="s", color="#c0392b", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised target)")
    ax.set_title("LSTM training curves")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    out = FIG_DIR / "lstm_loss_curves.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_naive_sample_window() -> None:
    df = load_hourly(ROOT / "data" / "processed")
    splits = chronological_split(df)
    test = splits.test[TARGET]

    start = 24 * 60
    window_hours = 24 * 7
    horizon = 24

    history = test.iloc[: start + window_hours].to_numpy()
    truth = test.iloc[start + window_hours : start + window_hours + horizon].to_numpy()
    pred = naive_seasonal_forecast(history, horizon, season_length=24)

    idx_hist = test.index[start : start + window_hours]
    idx_fc = test.index[start + window_hours : start + window_hours + horizon]

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.plot(idx_hist, history[start:], lw=0.9, color="#34495e", label="history")
    ax.plot(idx_fc, truth, lw=1.3, color="#1f4e79", label="ground truth")
    ax.plot(idx_fc, pred, lw=1.3, color="#c0392b", ls="--", label="naive seasonal (S=24)")
    ax.axvline(idx_fc[0], color="grey", lw=0.6, ls=":")
    ax.set_title("Test window: 7 days history + 24-hour forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("T (degC)")
    ax.legend(frameon=False, loc="lower right")
    ax.margins(x=0.0)
    out = FIG_DIR / "naive_sample_window.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_baseline_comparison() -> None:
    rows = []
    for name, file in (
        ("Naive Seasonal", "naive_seasonal.json"),
        ("ARIMA(2,1,2)", "arima.json"),
        ("LSTM", "lstm.json"),
    ):
        m = json.loads((ROOT / "results" / file).read_text())
        rows.append((name, m["rmse"], m["mae"]))

    labels = [r[0] for r in rows]
    rmse = [r[1] for r in rows]
    mae = [r[2] for r in rows]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    b1 = ax.bar(x - width / 2, rmse, width, color="#1f4e79", label="RMSE")
    b2 = ax.bar(x + width / 2, mae, width, color="#c0392b", label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (degC)")
    ax.set_title("Phase-1 baseline comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )
    out = FIG_DIR / "baseline_comparison.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_seasonal_decomposition() -> None:
    from statsmodels.tsa.seasonal import STL  # noqa: WPS433

    df = load_hourly(ROOT / "data" / "processed")
    series = df.loc["2014-06-01":"2014-06-14", TARGET]
    res = STL(series, period=24, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 4.8), sharex=True)
    axes[0].plot(series.index, series.values, lw=0.8, color="#34495e")
    axes[0].set_ylabel("observed")
    axes[1].plot(series.index, res.trend, lw=0.8, color="#1f4e79")
    axes[1].set_ylabel("trend")
    axes[2].plot(series.index, res.seasonal, lw=0.8, color="#c0392b")
    axes[2].set_ylabel("seasonal\n(24h)")
    axes[3].plot(series.index, res.resid, lw=0.6, color="#7f8c8d")
    axes[3].set_ylabel("residual")
    axes[3].set_xlabel("Time (2 weeks, June 2014)")
    fig.suptitle("STL decomposition of T (degC) -- 24h period", y=1.0)
    out = FIG_DIR / "seasonal_decomposition.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
# Phase-2 figures (PNG). These train the probabilistic models on the fly.
# --------------------------------------------------------------------------- #
# Must stay in sync with quantile_transformer.QUANTILES_7. Kept as a literal
# (not imported) so the Phase-1-only path doesn't pull in torch at module load.
QUANTILES = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
ALPHA = 0.1


def _q_index(level: float) -> int:
    return int(np.argmin(np.abs(QUANTILES - level)))


def _pit_values(y_true: np.ndarray, q_flat: np.ndarray) -> np.ndarray:
    """Approximate PIT: interpolate where each y sits among its quantiles."""
    pit = np.empty(len(y_true))
    for i in range(len(y_true)):
        pit[i] = np.interp(y_true[i], q_flat[i], QUANTILES, left=0.0, right=1.0)
    return pit


def _train_prob_models():
    """Train DeepAR + quantile Transformer and return everything the Phase-2
    figures need (predictions on test in physical units, plus the models)."""
    import torch  # noqa: F401

    from src import experiment as E
    from src.anomaly import inject_level_shift
    from src.calibration import apply_spread_temperature, fit_spread_temperature
    from src.models.deepar import DeepARConfig
    from src.models.quantile_transformer import QTransformerConfig, QUANTILES_7
    from src.preprocessing import TARGET
    from src.seq_data import make_ar_windows, make_encoder_windows

    data = E.prepare(use_covariates=False)
    L, H = 168, 24
    inv = lambda a: data.scaler.inverse_target(a, TARGET)  # noqa: E731

    deepar_cfg = DeepARConfig(lookback=L, horizon=H)
    qt_cfg = QTransformerConfig(lookback=L, horizon=H, n_features=1, quantiles=QUANTILES_7)
    print("  [phase2] training DeepAR ...")
    deepar = E.fit_deepar(data, deepar_cfg)
    print("  [phase2] training quantile Transformer ...")
    qt = E.fit_qtransformer(data, qt_cfg)

    x_te, y_te = make_encoder_windows(data.test, L, H, stride=H)
    yseq_te, cov_te = make_ar_windows(data.test, L, H, stride=H)
    x_va, y_va = make_encoder_windows(data.val, L, H, stride=H)

    q_da, _ = E.deepar_predict(deepar, deepar_cfg, yseq_te, cov_te, QUANTILES)
    q_qt = E.qtransformer_predict(qt, x_te)
    q_qt_va = E.qtransformer_predict(qt, x_va)

    # Calibrate the Transformer on val (representative for the calib figures).
    tau = fit_spread_temperature(inv(y_va).reshape(-1), inv(q_qt_va).reshape(-1, len(QUANTILES)), QUANTILES)
    q_qt_cal = apply_spread_temperature(inv(q_qt), QUANTILES, tau)

    # Anomalous context (level shift) for the interval-widening figure.
    rng = np.random.default_rng(42)
    ctx_adv, _ = inject_level_shift(x_te[:, :, 0].copy(), 3.0, rng)
    x_adv = x_te.copy(); x_adv[:, :, 0] = ctx_adv
    q_qt_anom = E.qtransformer_predict(qt, x_adv)

    return {
        "data": data, "inv": inv, "qt": qt, "L": L, "H": H,
        "x_te": x_te, "y_true": inv(y_te),
        "q_da": inv(q_da), "q_qt": inv(q_qt),
        "q_qt_cal": q_qt_cal, "q_qt_anom": inv(q_qt_anom),
        "tau": tau,
    }


def fig_forecast_intervals(ctx: dict) -> None:
    y_true, q_qt, q_anom = ctx["y_true"], ctx["q_qt"], ctx["q_qt_anom"]
    H = ctx["H"]
    w = min(40, len(y_true) - 1)  # a representative test window
    steps = np.arange(1, H + 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.8), sharey=True)
    for ax, q, title in ((axes[0], q_qt, "clean context"), (axes[1], q_anom, "level-shift context")):
        med = q[w, :, _q_index(0.5)]
        ax.fill_between(steps, q[w, :, _q_index(0.05)], q[w, :, _q_index(0.95)],
                        color="#1f4e79", alpha=0.18, label="90% PI")
        ax.fill_between(steps, q[w, :, _q_index(0.25)], q[w, :, _q_index(0.75)],
                        color="#1f4e79", alpha=0.30, label="50% PI")
        ax.plot(steps, med, color="#1f4e79", lw=1.3, label="median")
        ax.plot(steps, y_true[w], color="#c0392b", lw=1.3, ls="--", label="truth")
        ax.set_title(f"Transformer forecast -- {title}")
        ax.set_xlabel("Forecast horizon (h)")
    axes[0].set_ylabel("T (degC)")
    axes[0].legend(frameon=False, fontsize=7, loc="best")
    out = FIG_DIR / "forecast_intervals.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_reliability_curve(ctx: dict) -> None:
    y = ctx["y_true"].reshape(-1)
    series = (
        ("DeepAR", ctx["q_da"].reshape(-1, len(QUANTILES)), "#27ae60", "-"),
        ("Transformer", ctx["q_qt"].reshape(-1, len(QUANTILES)), "#1f4e79", "-"),
        ("Transformer (calibrated)", ctx["q_qt_cal"].reshape(-1, len(QUANTILES)), "#1f4e79", "--"),
    )
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":", label="ideal")
    for name, q, color, ls in series:
        emp = [(y <= q[:, i]).mean() for i in range(len(QUANTILES))]
        ax.plot(QUANTILES, emp, marker="o", ms=3, color=color, ls=ls, label=name)
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25)
    out = FIG_DIR / "reliability_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_pit_histogram(ctx: dict) -> None:
    y = ctx["y_true"].reshape(-1)
    pit_qt = _pit_values(y, ctx["q_qt"].reshape(-1, len(QUANTILES)))
    pit_da = _pit_values(y, ctx["q_da"].reshape(-1, len(QUANTILES)))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, pit, name, color in ((axes[0], pit_da, "DeepAR", "#27ae60"),
                                 (axes[1], pit_qt, "Transformer", "#1f4e79")):
        ax.hist(pit, bins=10, range=(0, 1), color=color, alpha=0.8, density=True)
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"PIT -- {name}")
        ax.set_xlabel("PIT")
    axes[0].set_ylabel("density")
    out = FIG_DIR / "pit_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_per_horizon_curves() -> None:
    path = ROOT / "results" / "error_analysis.json"
    if not path.exists():
        print("  skip per_horizon_curves (run scripts/run_error_analysis.py first)")
        return
    ea = json.loads(path.read_text())
    ph = ea["per_horizon"]
    steps = np.arange(1, len(ph["lstm"]["rmse"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.9))
    for name, color in (("lstm", "#7f8c8d"), ("deepar", "#27ae60"), ("qtransformer", "#1f4e79")):
        axes[0].plot(steps, ph[name]["rmse"], marker="o", ms=2.5, color=color, label=name)
    axes[0].set_title("RMSE vs forecast horizon")
    axes[0].set_xlabel("Horizon (h)"); axes[0].set_ylabel("RMSE (degC)")
    axes[0].legend(frameon=False, fontsize=7); axes[0].grid(alpha=0.25)

    for name, color in (("deepar", "#27ae60"), ("qtransformer", "#1f4e79")):
        axes[1].plot(steps, ph[name]["crps"], marker="o", ms=2.5, color=color, label=name)
    axes[1].set_title("CRPS vs forecast horizon")
    axes[1].set_xlabel("Horizon (h)"); axes[1].set_ylabel("CRPS")
    axes[1].legend(frameon=False, fontsize=7); axes[1].grid(alpha=0.25)
    out = FIG_DIR / "per_horizon_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_robustness_heatmap() -> None:
    path = ROOT / "results" / "anomaly_eval.json"
    if not path.exists():
        print("  skip robustness_heatmap (run scripts/run_anomaly_eval.py first)")
        return
    r = json.loads(path.read_text())
    models = ("lstm", "deepar", "qtransformer")
    clean_rmse = {m: r["clean"][m]["rmse"] for m in models}
    types = r["config"]["anomaly_types"]
    intensities = [f"{i:.1f}" for i in r["config"]["intensities"]]

    cols, mat = [], []
    for kind in types:
        for it in intensities:
            cols.append(f"{kind}\n@{it}")
    for m in models:
        row = []
        for kind in types:
            for it in intensities:
                row.append(r["anomaly"][kind][it][m]["rmse"] / clean_rmse[m])
        mat.append(row)
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(9.5, 2.7))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=1.0)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6)
    ax.set_title("RMSE inflation under anomaly injection (x clean RMSE)")
    fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01)
    out = FIG_DIR / "robustness_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_attention_map(ctx: dict) -> None:
    import torch

    qt, x_te, L = ctx["qt"], ctx["x_te"], ctx["L"]
    device = next(qt.parameters()).device
    xb = torch.from_numpy(x_te[:1]).to(device)
    attn = qt.attention_first_layer(xb)[0].cpu().numpy()  # (L, L)

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_title("Transformer self-attention (layer 1)")
    ax.set_xlabel("key position (lookback)")
    ax.set_ylabel("query position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = FIG_DIR / "attention_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()

    print("Generating figures ->", FIG_DIR)
    if args.phase in ("1", "all"):
        fig_series_split()
        fig_lstm_loss_curves()
        fig_naive_sample_window()
        fig_baseline_comparison()
        fig_seasonal_decomposition()
    if args.phase in ("2", "all"):
        # aggregate figures read result JSONs; these need no models
        fig_per_horizon_curves()
        fig_robustness_heatmap()
        # model-dependent figures train the probabilistic models once
        ctx = _train_prob_models()
        fig_forecast_intervals(ctx)
        fig_reliability_curve(ctx)
        fig_pit_histogram(ctx)
        fig_attention_map(ctx)
    print("done.")


if __name__ == "__main__":
    main()
