# Probabilistic Forecasting with Anomaly Injection

CENG 463 — Introduction to Machine Learning — Term Project  
İzmir Institute of Technology — Spring 2026

**Student:** Ozan Erdoğan  
**Instructor:** Prof. Dr. Aytuğ Onan  
**Dataset:** Jena Climate, a multivariate weather time-series dataset with approximately 420K raw observations and 14 meteorological variables.

---

## Problem

Time-series forecasting models are typically evaluated under clean test conditions. In practice,
real-world series are contaminated with point spikes, level shifts, contextual outliers, and
adversarial perturbations. We compare **deterministic** and **probabilistic** forecasting approaches
under controlled anomaly injection scenarios to study which approach degrades more gracefully and
which uncertainty signals are useful for downstream anomaly-aware decision making.


## Approach

We forecast hourly temperature 24 hours ahead from a 168-hour lookback and
compare three families:

- **Deterministic baselines** — naive seasonal, ARIMA, and an LSTM. A **SARIMA**
  model is added as a *control*: it isolates whether plain ARIMA's weakness comes
  from the missing seasonal component rather than serving as just another baseline.
- **Probabilistic models** — **DeepAR** (autoregressive LSTM with a Gaussian or
  Student-t likelihood, trained by NLL, sampled to quantiles) and a **quantile
  Transformer** (encoder with per-quantile heads trained by pinball loss).
- **Robustness study** — every model is re-scored under injected anomalies
  (point spike, contextual outlier, level shift, and an L-inf FGSM perturbation),
  with intensity scaled by the local rolling std. We add **post-hoc
  spread-temperature calibration** fit on validation and report PICP before/after.

The study is rounded out by a mandatory **ablation** (input richness, lookback,
likelihood, quantile-set size), an **error analysis** (per-horizon, by season and
temperature range, worst windows under anomaly, and overconfident-failure
analysis), and a **visualization** suite.

> **Covariate assumption.** Multivariate variants feed *contemporaneous* exogenous
> weather channels alongside the target, i.e. a perfect-covariate setting. This
> inflates multivariate scores and is documented as an upper bound, not an
> operational forecast.

## Repository Structure

```
.
├── data/                    # Raw / processed data (gitignored)
│   └── README.md            # Data acquisition instructions
├── src/
│   ├── data_loader.py       # Jena Climate download + load
│   ├── preprocessing.py     # Splits, scaling, windowing
│   ├── features.py          # Calendar + exogenous covariate frames
│   ├── seq_data.py          # AR / encoder sliding-window builders
│   ├── metrics.py           # RMSE, MAE, MAPE; CRPS, pinball, PICP, MIS, sMAPE, MASE
│   ├── anomaly.py           # Anomaly injection: spike, outlier, level shift, FGSM
│   ├── calibration.py       # Post-hoc spread-temperature calibration
│   ├── experiment.py        # Shared train/predict/gradient plumbing
│   ├── ablation.py          # Ablation variants over model design choices
│   ├── error_analysis.py    # Error-slicing primitives
│   ├── baselines/
│   │   ├── naive_seasonal.py
│   │   ├── arima_baseline.py
│   │   ├── sarima_baseline.py   # Seasonal control for the ARIMA question
│   │   └── lstm_baseline.py
│   └── models/
│       ├── deepar.py            # Autoregressive LSTM + Gaussian/Student-t NLL
│       └── quantile_transformer.py  # Encoder + pinball-loss quantile heads
├── scripts/                 # Entry points
│   ├── download_data.py
│   ├── run_naive.py
│   ├── run_arima.py
│   ├── run_sarima.py        # SARIMA control
│   ├── run_lstm.py
│   ├── run_deepar.py        # DeepAR probabilistic forecaster
│   ├── run_qtransformer.py  # Quantile Transformer
│   ├── run_anomaly_eval.py  # Clean vs anomalous eval (+ calibration)
│   ├── run_ablation.py      # Ablation study
│   ├── run_error_analysis.py
│   └── make_figures.py      # --phase {1,2,all}
├── tests/                   # pytest unit tests (metrics)
├── results/                 # Metrics JSON, ablation/, figures/
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

# 3. Deterministic baselines (Phase 1)
python scripts/run_naive.py
python scripts/run_arima.py
python scripts/run_sarima.py        # seasonal control
python scripts/run_lstm.py

# 4. Probabilistic models (Phase 2)
python scripts/run_deepar.py
python scripts/run_qtransformer.py

# 5. Anomaly robustness, ablation, error analysis
python scripts/run_anomaly_eval.py
python scripts/run_ablation.py
python scripts/run_error_analysis.py

# 6. Figures (Phase 1 PDFs + Phase 2 PNGs)
python scripts/make_figures.py --phase all

# Unit tests for the metrics
pytest
```

Results are written to `results/` (per-model JSON, `results/ablation/`, and
`results/figures/`).

| Script | Output |
| --- | --- |
| `run_naive.py` / `run_arima.py` / `run_sarima.py` / `run_lstm.py` | `results/naive_seasonal.json`, `arima.json`, `sarima.json`, `lstm.json` |
| `run_deepar.py` | `results/deepar.json` |
| `run_qtransformer.py` | `results/qtransformer.json` |
| `run_anomaly_eval.py` | `results/anomaly_eval.json` |
| `run_ablation.py` | `results/ablation.json` (+ `results/ablation/<variant>.json`) |
| `run_error_analysis.py` | `results/error_analysis.json` |
| `make_figures.py` | `results/figures/*.pdf` (Phase 1), `*.png` (Phase 2) |

Heavy scripts accept `--smoke` for a fast 1-epoch sanity pass.

## Dataset

**Jena Climate 2009–2016** — recorded by the Max Planck Institute for Biogeochemistry, Jena,
Germany. 14 weather variables sampled every 10 minutes (we resample to hourly). Forecasting target
in this study: temperature `T (degC)`. The remaining variables are used as contemporaneous
exogenous covariates in the multivariate ablation variants (a documented perfect-covariate upper
bound; see the covariate-assumption note above).

See `data/README.md` for acquisition details.
