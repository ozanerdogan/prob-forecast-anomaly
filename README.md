# Probabilistic Forecasting with Anomaly Injection

CENG 463 — Introduction to Machine Learning — Term Project
İzmir Institute of Technology — Spring 2026

**Student:** Ozan Erdoğan
**Instructor:** Prof. Dr. Aytuğ Onan
**Dataset:** Jena Climate (2009–2016, hourly), target `T (degC)`, 168 h lookback → 24 h horizon.

---

Forecasting models are usually judged on clean test data, but real series arrive contaminated —
sensor spikes, flatlines, level shifts, drifts, even adversarial perturbations. This project asks
whether **probabilistic forecasters' uncertainty stays trustworthy when the input context is
corrupted**, and how to repair it when it is not.

**Headline findings** (14-model roster × 8 fault types × 3 intensities, 359 test windows):

- On clean data the multivariate quantile Transformer leads (RMSE 2.28); quantile heads cost
  nothing over their deterministic twins (qLSTM beats LSTM, DM p = 0.024).
- Under a 4× level shift, 90 % prediction intervals collapse to **0.20–0.39 coverage** — the
  uncertainty fails exactly when it is needed. Static and CQR recalibration do **not** help
  (exchangeability is broken).
- **Online adaptive calibration (ACI)** recovers coverage to 0.75–0.87, and **anomaly-augmented
  robust training** independently repairs the point forecast (RMSE 9.0 → 5.4). The two are
  **complementary**: together 0.87–0.89, near the 0.90 target.

Key figures live in [`results/figures/main/`](results/figures/main); the full result JSONs in
[`results/`](results).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py     # fetch Jena Climate -> data/processed/jena_hourly.parquet
pytest                           # CPU-only unit suite, no dataset/GPU needed
```

Every heavy script accepts `--smoke` for a fast 1-epoch sanity pass.

<details>
<summary><b>Approach & model roster</b></summary>

We compare paired deterministic/probabilistic families so the "does probabilistic cost accuracy?"
question is controlled within an architecture:

| Family | Deterministic | Probabilistic |
|---|---|---|
| Recurrent | LSTM, GRU | qLSTM (pinball twin head), DeepAR (Gaussian/Student-t NLL) |
| Linear | DLinear | qDLinear |
| Tree | LightGBM (point) | LightGBM-quantile, QRF |
| Attention | — | Quantile Transformer (uni + multivariate) |
| Classical | naive seasonal, ARIMA, SARIMA | — |

Metrics: RMSE/MAE (point), CRPS, pinball, PICP, MPIW, MIS (interval, α = 0.1); model comparisons
are backed by Diebold–Mariano + paired-bootstrap tests (`results/base/significance.json`).

</details>

<details>
<summary><b>Anomaly catalog & the two-stage repair design</b></summary>

**Catalog (8 faults, 3 intensities, local-std scaled, injected into the test context window of the
target channel):** point spike, contextual outlier, level shift, white-box FGSM (v1) + flatline,
drift, noise burst, gap imputation, clock skew (v2). Severity taxonomy at 4× intensity:
catastrophic (drift, level shift, FGSM — PICP < 0.37), moderate (flatline, clock skew ≈ 0.66),
mild (noise burst, gap ≈ 0.88).

**Two-stage design (the methodological core).** Stage-1 scripts train models once and dump frozen
forecasts to `results/predictions/`; stage-2 calibrators read **only** those dumps, so
before/after deltas are attributable to the repair method alone. Compared regimes:

- **static spread temperature** — offline baseline; fails under shift,
- **CQR** — offline conformal margin; fails under shift (exchangeability broken),
- **ACI** — online spread adaptation; window *t* uses only realised coverage from windows < *t*
  (the feedback actually available in deployment),
- **input-conditional τ** — offline, fit on validation windows with synthetically injected
  anomalies; no test feedback of any kind,
- **detect-then-adapt** — an explicit anomaly detector gates the repair: clean windows keep the
  sharp static regime, detected windows get an anomaly-conditional spread (best offline policy;
  beats even online ACI on faults with unnatural signatures),
- **hampel detect-and-clean** — input-side contrast; catches spikes, blind to level shift/drift.

A **natural-extremes slice** (real cold fronts / warm-ups, no injection) measures the false-alarm
cost: adaptive methods widen only ~8–11 % on legitimate sharp transitions while keeping coverage.

</details>

<details>
<summary><b>Covariate handling — leakage matters</b></summary>

8 of the 13 exogenous Jena variables are analytically derivable from temperature (inverting the
Magnus formula recovers T from `VPmax` with RMSE 0.05 °C vs. a T std of 8.4 °C). Feeding them
covertly injects the target: an "exogenous-only" model with proxies scores RMSE 2.32, but with
only the **5 genuinely independent** sensors (`p`, `rh`, `wv`, `max. wv`, `wd`) it drops to 3.68 —
*worse than the naive baseline* (3.21). The official multivariate set therefore uses only the 5
independent channels (+ calendar features).

DeepAR is autoregressive, so feeding *contemporaneous* horizon weather also leaks the answer —
that oracle variant is removed from the roster (archived); the leakage-free
`deepar_past_covariate` variant freezes horizon weather at the origin. The quantile-Transformer
encoder never reads horizon covariates, so its multivariate variant is leakage-free by
construction — which is also why most input-side ablations run on it.

</details>

<details>
<summary><b>Repository structure</b></summary>

```
.
├── data/                  # acquisition README + download_data.py (raw/processed gitignored)
├── src/                   # library code
│   ├── anomaly.py             # fault injectors (v1 + v2), FGSM
│   ├── calibrators.py         # static τ / CQR / ACI / ACI-margin / input-τ
│   ├── calib_runner.py        # stage-2 plumbing: frozen dumps -> calibrated metrics
│   ├── model_eval.py          # stage-1 plumbing: eval + dump protocol
│   ├── predictions_io.py      # frozen-forecast .npz store
│   ├── robust.py              # anomaly-augmented training (augment_fn factory)
│   ├── baselines/  models/    # model implementations
│   └── ...                    # metrics, features, windowing, error analysis
├── scripts/
│   ├── models/            # stage 1: one entry point per model (train + dump)
│   ├── calibrate/         # stage 2: calibrators (read frozen dumps only)
│   ├── ablation/          # ablations, HPO (+ Optuna), multiseed, CV
│   ├── analysis/          # anomaly sweeps, significance, robust×calibration studies
│   └── report/            # tables + figures from result JSONs (no model runs)
├── tests/                 # CPU-only pytest suite (no dataset, no GPU)
├── results/
│   ├── base/              # stage-1 per-model metrics (JSON)
│   ├── calibrated/        # stage-2 metrics per method/model
│   ├── ablation/          # per-variant JSONs + summary.json
│   ├── figures/main/      # headline figures (report candidates)
│   ├── figures/extra/     # everything else
│   └── predictions/       # frozen forecasts (.npz, gitignored)
└── report/                # local report workspace (gitignored)
```

</details>

<details>
<summary><b>Full reproduction pipeline</b></summary>

```bash
# Stage 1 — models (train + dump frozen forecasts)
python scripts/models/run_naive.py
python scripts/models/run_arima.py
python scripts/models/run_sarima.py
python scripts/models/run_lstm.py
python scripts/models/run_gru.py
python scripts/models/run_dlinear.py          # DLinear + qDLinear twins
python scripts/models/run_lgbm.py             # LightGBM point + quantile
python scripts/models/run_qrf.py
python scripts/models/run_qlstm.py
python scripts/models/run_deepar.py
python scripts/models/run_qtransformer.py
python scripts/models/run_qtransformer_multi.py   # multivariate QT + permutation importance
python scripts/models/run_qlstm_robust.py         # robust qLSTM as a first-class dump
python scripts/models/run_qt_robust.py

# Anomaly sweep (also dumps frozen forecasts; --catalog v2 for the 8-fault sweep)
python scripts/analysis/run_anomaly_eval.py --catalog v2

# Stage 2 — calibration (reads frozen dumps, never runs a model)
python scripts/calibrate/calibrate_static.py
python scripts/calibrate/calibrate_cqr.py
python scripts/calibrate/calibrate_aci.py
python scripts/calibrate/calibrate_aci_margin.py
python scripts/calibrate/calibrate_input_tau.py
python scripts/calibrate/calibrate_detect_clean.py
python scripts/calibrate/calibrate_detect_adapt.py   # detect-then-adapt + detection report

# Studies
python scripts/analysis/run_significance.py
python scripts/analysis/run_natural_extremes.py
python scripts/analysis/run_robust_generalize.py   # robust training on qLSTM/QT/DeepAR
python scripts/analysis/run_robust_plus_cal.py     # model-side × interval-side 4 corners
python scripts/analysis/run_error_analysis.py
python scripts/analysis/run_ensemble_intervals.py
python scripts/analysis/run_composite_anomaly.py
python scripts/analysis/run_tail_oversampling.py

# Ablations / optimization
python scripts/ablation/run_ablation.py            # input/lookback/likelihood/quantile-set
python scripts/ablation/run_horizon_ablation.py    # 12/24/48/168 h
python scripts/ablation/run_10min_ablation.py      # native 10-min resolution
python scripts/ablation/run_qt_extreme_quantiles.py
python scripts/ablation/run_covariate_importance.py
python scripts/ablation/run_exogenous_only.py
python scripts/ablation/run_hpo.py
python scripts/ablation/run_hpo_optuna.py          # Optuna TPE confirmation
python scripts/ablation/run_multiseed.py
python scripts/ablation/run_cv.py

# Tables + figures (read result JSONs only)
python scripts/report/make_report_tables.py
python scripts/report/make_error_tables.py
python scripts/report/make_phase_figures.py --phase all
```

| Output | Where |
| --- | --- |
| Stage-1 per-model metrics | `results/base/<model>.json` |
| Anomaly sweep | `results/base/anomaly_eval.json` (+ `results/predictions/*.npz`) |
| Stage-2 calibration | `results/calibrated/<method>/<model>.json` |
| Ablation | `results/ablation/<variant>.json` + `results/ablation/summary.json` |
| Figures | `results/figures/{main,extra}/*.png` |

</details>

<details>
<summary><b>Reproducibility — seeds, determinism, tests</b></summary>

**One global seed: `42`** — fixed in every component that draws randomness, so a given model on a
given machine reproduces the same scores on every run.

| What | Where the seed lives | How it is applied |
| --- | --- | --- |
| Neural models (LSTM/GRU/qLSTM/DLinear/QT/DeepAR) | each `*Config.seed` dataclass field | `torch.manual_seed` + `np.random.seed` at the top of each `train_*` |
| DeepAR sampling | `DeepARConfig.seed` | `sample_forecast` reseeds torch (+CUDA) before drawing trajectories |
| Anomaly injection | `SEED = 42` in the eval scripts | `np.random.default_rng(SEED)` per (type, intensity); injectors receive the generator explicitly |
| Multiseed study | `SEEDS = (42, 7, 2025)` in `scripts/ablation/run_multiseed.py` | mean ± std reported per model |

What can still vary: cuDNN kernels are not bit-identical across GPUs/driver versions — on a
*fixed* machine repeated runs are bit-identical (verified). ARIMA/SARIMA use deterministic MLE.

**Tests** (`pytest`, CPU-only, no dataset): metrics, window builders, anomaly injectors
(incl. intensity-scaling linearity), all stage-2 calibrators (e.g. ACI's first window provably
ignores its own outcome), the hampel filter, the frozen-forecast store, significance helpers, and
model inference paths.

</details>

<details>
<summary><b>Dataset</b></summary>

**Jena Climate 2009–2016** — weather-station data recorded by the Max Planck Institute for
Biogeochemistry, Jena, Germany. 14 variables sampled every 10 minutes (resampled to hourly here).
Chronological splits: train 2009–2014, validation 2015, test 2016. See
[`data/README.md`](data/README.md) for acquisition, the variable table, and the
leakage classification of the exogenous channels.

</details>
