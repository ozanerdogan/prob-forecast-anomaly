# Probabilistic Forecasting with Anomaly Injection

CENG 463 — Introduction to Machine Learning — Term Project (Spring 2026)
İzmir Institute of Technology — Prof. Dr. Aytuğ Onan

**Project topic:** *Probabilistic Forecasting with Anomaly Injection*.
**Dataset:** Jena Climate (multivariate hourly weather, ~420K rows, 14 variables).

---

## Problem

Time-series forecasting models are typically evaluated under clean test conditions. In practice,
real-world series are contaminated with point spikes, level shifts, contextual outliers, and
adversarial perturbations. We compare **deterministic** and **probabilistic** forecasting approaches
under controlled anomaly injection scenarios to study which approach degrades more gracefully and
which uncertainty signals are useful for downstream anomaly-aware decision making.


## Repository Structure

```
.
├── data/                    # Raw / processed data (gitignored)
│   └── README.md            # Data acquisition instructions
├── src/
│   ├── data_loader.py       # Jena Climate download + load
│   ├── preprocessing.py     # Splits, scaling, windowing
│   ├── metrics.py           # RMSE, MAE, MAPE (Phase 1); CRPS, Pinball (later)
│   └── baselines/
│       ├── naive_seasonal.py
│       ├── arima_baseline.py
│       └── lstm_baseline.py
├── scripts/                 # Entry points
│   ├── download_data.py
│   ├── run_naive.py
│   ├── run_arima.py
│   └── run_lstm.py
├── results/                 # Metrics tables, plots
├── report/                  # Progress report
└── notebooks/               # Exploratory analysis
```

## Reproduction

```bash
# 1. Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Data
python scripts/download_data.py

# 3. Baselines
python scripts/run_naive.py
python scripts/run_arima.py
python scripts/run_lstm.py
python scripts/make_figures.py
```

Results are written to `results/`.

## Dataset

**Jena Climate 2009–2016** — recorded by the Max Planck Institute for Biogeochemistry, Jena,
Germany. 14 weather variables sampled every 10 minutes (we resample to hourly). Forecasting target
in this study: temperature `T (degC)`. Other variables are kept for planned exogenous-feature variants in later phases.

See `data/README.md` for acquisition details.
