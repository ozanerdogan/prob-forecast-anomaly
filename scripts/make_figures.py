"""Produce Phase-1 progress-report figures.

All figures are written as PDF (vector) into ``results/figures/`` so that LaTeX
embeds them at the right resolution.

Figures produced:
  1. jena_series_split.pdf       -- target series + train/val/test markers
  2. lstm_loss_curves.pdf        -- LSTM train/val MSE per epoch
  3. naive_sample_window.pdf     -- 7-day test window: truth vs naive seasonal
  4. baseline_comparison.pdf     -- bar chart of RMSE & MAE per baseline
  5. seasonal_decomposition.pdf  -- STL decomposition on a 14-day train window
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


def main() -> None:
    print("Generating figures ->", FIG_DIR)
    fig_series_split()
    fig_lstm_loss_curves()
    fig_naive_sample_window()
    fig_baseline_comparison()
    fig_seasonal_decomposition()
    print("done.")


if __name__ == "__main__":
    main()
