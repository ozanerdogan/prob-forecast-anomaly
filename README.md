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
whether the probabilistic models' uncertainty stays **trustworthy** (well-calibrated and not
overconfident) when the input context is contaminated — a prerequisite for anomaly-aware decision
making, which we leave to future work.


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
  with intensity scaled by the local rolling std. The deterministic family has two
  members (the LSTM and the untrained naive-seasonal); all models see the *same*
  injected context for the non-gradient anomalies, while FGSM is white-box per
  model, so naive-seasonal (no gradient) is reported N/A there. We add **post-hoc
  spread-temperature calibration** fit on validation and report PICP before/after.

The study is rounded out by a mandatory **ablation** (input richness —
target-only, leakage-free past-covariate, and the future-leaking oracle upper
bound — plus lookback, likelihood, quantile-set size), an **error analysis**
(per-horizon, by season and temperature range, worst windows under anomaly, and
overconfident-failure analysis), and a **visualization** suite.

> **Stage-2 uncertainty repair (two-stage design).** Stage-1 scripts dump
> frozen forecasts (`results/predictions/`); stage-2 calibration scripts read
> only those dumps, so before/after deltas are attributable to the repair
> method alone. Compared regimes: **static spread temperature** (offline
> baseline), **CQR** (offline conformal margin), **ACI-style online spread
> adaptation** (window *t* uses only realised coverage from windows < *t* —
> the feedback available in deployment), and an **offline input-conditional
> spread** fit on validation windows with synthetically injected anomalies
> (no test feedback of any kind). A hampel **detect-and-clean** input-repair
> baseline provides the input-side contrast. Headline: under a 4-sigma level
> shift DeepAR's 90% interval covers 27% of outcomes and static repair only
> lifts it to 32%, while the adaptive regimes recover 0.85 (ACI) / 0.72
> (input-conditional) at roughly half the interval score — uncertainty can be
> made trustworthy under contamination, but not with a static correction.
> Model comparisons are backed by Diebold-Mariano + paired-bootstrap tests
> (`results/base/significance.json`): LSTM vs the quantile Transformer is a
> statistical tie (p = 0.98) while the LSTM+QT ensemble's gain is significant
> (p = 0.006).

> **Covariate handling (leakage matters).** DeepAR is autoregressive, so feeding
> the *contemporaneous* exogenous weather over the horizon leaks the answer — its
> `deepar_multivariate` ablation variant is therefore an **oracle / perfect-covariate
> upper bound** (CRPS ≈ 0.08, RMSE ≈ 0.20°C — physically impossible, i.e. leakage),
> not an operational forecast. The realistic **`deepar_past_covariate`** variant
> freezes horizon weather at the last observed value (persistence) and keeps only
> the calendar features as true future; once the leakage is removed the exogenous
> covariates no longer help DeepAR at the shared budget (and calibration degrades),
> which shows the apparent multivariate gain was leakage. The quantile-Transformer
> encoder never reads horizon covariates, so its `qtransformer_multivariate` is
> already a leakage-free past-covariate setting.

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
├── tests/                   # pytest unit tests (metrics, models, anomaly, calibration)
├── results/                 # base/ (stage-1 metrics), calibrated/, predictions/, ablation/, figures/
└── report/                  # Progress report
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
python scripts/run_anomaly_eval.py   # also dumps frozen forecasts -> results/predictions/
python scripts/run_ablation.py
python scripts/run_error_analysis.py

# 5b. Phase-2 model roster (paired det/prob families) + multivariate base
python scripts/run_lgbm.py            # tree: LightGBM point + quantile
python scripts/run_qrf.py             # tree-prob second opinion (QRF)
python scripts/run_qlstm.py           # recurrent: quantile-LSTM twin
python scripts/run_gru.py             # recurrent: GRU point twin
python scripts/run_dlinear.py         # linear: DLinear + qDLinear twins
python scripts/run_qtransformer_multi.py   # multivariate QT base + permutation importance
python scripts/probe_deepar_covariates.py  # why past-covariates hurt DeepAR

# 5c. Stage-2 calibration -- reads the frozen predictions, never runs a model
python scripts/calibrate_static.py
python scripts/calibrate_cqr.py
python scripts/calibrate_aci.py
python scripts/calibrate_input_tau.py
python scripts/calibrate_detect_clean.py

# 5d. Significance tests + natural sharp-transition slice (no injection)
python scripts/run_significance.py
python scripts/run_natural_extremes.py

# 5e. Stage-2 adaptive-CQR (online conformal margin)
python scripts/calibrate_aci_margin.py

# 5f. Phase-3 optimization study (normal vs optimized, side by side)
python scripts/run_hpo.py                 # HPO, selection on validation
python scripts/run_multiseed.py           # 4 models x 3 seeds (mean +/- std)
python scripts/run_cv.py                  # forward-chaining year CV
python scripts/run_robust_training.py     # anomaly-augmented training
python scripts/run_tail_oversampling.py   # tail reweighting + calibration recheck
python scripts/run_ensemble_intervals.py  # quantile-averaging ensemble
python scripts/run_composite_anomaly.py   # overlapping faults

# Phase reports + figures (figures read the result JSONs, no model runs)
python scripts/make_phase_figures.py --phase all

# 6. Figures (Phase 1 PDFs + Phase 2 PNGs)
python scripts/make_figures.py --phase all

# Unit tests for the metrics
pytest
```

Results are written to `results/base/` (stage-1 per-model JSON), `results/ablation/`,
`results/predictions/` (frozen forecasts, gitignored), `results/calibrated/<method>/`
(stage-2 calibration metrics) and `results/figures/`.

| Script | Output |
| --- | --- |
| `run_naive.py` / `run_arima.py` / `run_sarima.py` / `run_lstm.py` | `results/base/naive_seasonal.json`, `arima.json`, `sarima.json`, `lstm.json` |
| `run_deepar.py` | `results/base/deepar.json` |
| `run_qtransformer.py` | `results/base/qtransformer.json` |
| `run_anomaly_eval.py` | `results/base/anomaly_eval.json` (+ `results/predictions/*.npz`) |
| `run_ablation.py` | `results/ablation.json` (+ `results/ablation/<variant>.json`) |
| `run_error_analysis.py` | `results/base/error_analysis.json` |
| `calibrate_static.py` / `calibrate_aci.py` / `calibrate_input_tau.py` / `calibrate_cqr.py` / `calibrate_detect_clean.py` | `results/calibrated/<method>/<model>.json` |
| `make_figures.py` | `results/figures/*.pdf` (Phase 1), `*.png` (Phase 2) |

Heavy scripts accept `--smoke` for a fast 1-epoch sanity pass.

## Reproducibility

**One global seed: `42`.** It is fixed in every component that draws randomness,
so a given model on a given machine reproduces the same scores on every run.

| What | Where the seed lives | How it is applied |
| --- | --- | --- |
| LSTM baseline | `LstmConfig.seed` (`src/baselines/lstm_baseline.py`) | `torch.manual_seed` + `np.random.seed` at the top of `train_lstm`, before the shuffled `DataLoader` |
| DeepAR (training) | `DeepARConfig.seed` (`src/models/deepar.py`) | `torch.manual_seed` + `np.random.seed` at the top of `train_deepar` |
| DeepAR (sampling) | `DeepARConfig.seed` | `sample_forecast` reseeds `torch` (+CUDA) **before drawing trajectories**, so PICP/CRPS are identical across runs |
| Quantile Transformer | `QTransformerConfig.seed` (`src/models/quantile_transformer.py`) | `torch.manual_seed` + `np.random.seed` at the top of `train_qtransformer` |
| Anomaly injection / robustness sweep | `SEED = 42` at the top of `scripts/run_anomaly_eval.py` and `scripts/run_error_analysis.py` | `np.random.default_rng(SEED)` per (anomaly type, intensity); injectors in `src/anomaly.py` receive the generator explicitly and never create their own |
| Figures | `np.random.default_rng(42)` in `scripts/make_figures.py` | passed into the anomaly injectors |

**To change the seed:** pass a different `seed=` when constructing the config
(e.g. `DeepARConfig(seed=7)`), or edit the `seed` default in the dataclass; for
the anomaly/error-analysis scripts edit the `SEED` constant at the top of the file.

**What can still vary:**
- **Hardware / driver / library version.** cuDNN kernels are not guaranteed
  bit-identical across different GPUs, CUDA/cuDNN versions, or PyTorch builds, so
  the last few decimals may shift on a different machine. On a *fixed* machine,
  repeated runs are bit-identical (verified: DeepAR clean PICP/CRPS match exactly
  between `deepar.json` and `anomaly_eval.json`).
- **ARIMA / SARIMA** use deterministic MLE fitting (statsmodels) — no seed needed.
- **Committed figures** under `results/figures/` are rendered from the seeded
  pipeline and the current result JSONs; re-run `python scripts/make_figures.py
  --phase all` to regenerate them after any results change.

**Tests.** Unit tests live in `tests/` and cover the metrics, sequence-window
builders, anomaly injectors (including intensity-scaling linearity), post-hoc
calibration, the stage-2 calibrators (static/CQR/ACI/input-conditional — e.g.
ACI's first window provably ignores its own outcome), the hampel
detect-and-clean filter, the frozen-forecast store, the significance helpers,
and the DeepAR/Transformer inference paths (including a regression guard on
the DeepAR conditioning alignment). They are self-contained -- tiny
random-init models on CPU, no dataset download, no training, no GPU:

```bash
pytest                      # or: pytest tests/test_metrics.py -q
```

The single randomised fixture uses a fixed seed, so the suite is deterministic.

## Dataset

**Jena Climate 2009–2016** — recorded by the Max Planck Institute for Biogeochemistry, Jena,
Germany. 14 weather variables sampled every 10 minutes (we resample to hourly). Forecasting target
in this study: temperature `T (degC)`. The remaining variables are used as exogenous covariates in
the multivariate ablation variants — as a future-leaking *oracle* upper bound (DeepAR) and, without
leakage, as *past covariates* with horizon weather frozen at the origin (see the covariate-handling
note above).

See `data/README.md` for acquisition details.
