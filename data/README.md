# Data: Jena Climate

We use the **Jena Climate 2009–2016** dataset (Max Planck Institute for Biogeochemistry).
14 weather variables sampled every 10 minutes; we resample to **hourly** for forecasting.

## Automatic download

The download is wrapped in `data/download_data.py`:

```bash
python data/download_data.py
```

This will:
1. Fetch `jena_climate_2009_2016.csv.zip` (~13 MB) into `data/raw/`
2. Unzip it
3. Resample to hourly and save the cleaned multivariate parquet to `data/processed/jena_hourly.parquet`

## Manual download (if the script fails)

Direct URL (Keras mirror):
```
https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
```

Place the unzipped CSV at `data/raw/jena_climate_2009_2016.csv` and re-run the script. It detects
the existing file and skips the download step, going straight to resampling.

## Splits

We use a fixed chronological split:

| Split | Range | Size (hourly) |
|---|---:|---:|
| Train | 2009-01-01 → 2014-12-31 23:00 | 52,584 |
| Val   | 2015-01-01 → 2015-12-31 23:00 | 8,760 |
| Test  | 2016-01-01 → 2016-12-31 23:00 | 8,784 |

Splits are produced deterministically by [`src/preprocessing.py`](../src/preprocessing.py).

## Variables

| Column | Description | Unit |
|---|---|---|
| `p (mbar)` | Atmospheric pressure | mbar |
| `T (degC)` | Air temperature | °C |
| `Tpot (K)` | Potential temperature | K |
| `Tdew (degC)` | Dew point | °C |
| `rh (%)` | Relative humidity | % |
| `VPmax (mbar)` | Saturation vapor pressure | mbar |
| `VPact (mbar)` | Actual vapor pressure | mbar |
| `VPdef (mbar)` | Vapor pressure deficit | mbar |
| `sh (g/kg)` | Specific humidity | g/kg |
| `H2OC (mmol/mol)` | Water vapor concentration | mmol/mol |
| `rho (g/m**3)` | Air density | g/m³ |
| `wv (m/s)` | Wind speed | m/s |
| `max. wv (m/s)` | Max wind speed | m/s |
| `wd (deg)` | Wind direction | deg |

## Forecasting target and covariates

**Target:** `T (degC)`. The default setup is single-target (univariate).

Multivariate variants add the **5 independent** exogenous channels alongside the target, plus
calendar features: `p (mbar)`, `rh (%)`, `wv (m/s)`, `max. wv (m/s)`, `wd (deg)`.

The remaining 8 exogenous variables (`Tpot`, `Tdew`, `VPmax`, `VPact`, `VPdef`, `sh`, `H2OC`,
`rho`) are analytically derivable from temperature. Inverting the Magnus formula, for example,
recovers `T` from `VPmax` with an RMSE of 0.05 °C against a temperature std of 8.4 °C. Feeding
them would covertly inject the target into the input, so they are excluded as **leakage**.

DeepAR is autoregressive, so feeding it *contemporaneous* future weather leaks the target as well.
That variant (`deepar_multivariate`) was an oracle upper bound and has been removed from the
leaderboard; the leakage-free **`deepar_past_covariate`** instead freezes horizon weather at the
origin (persistence).

See the covariate-handling note in the top-level [`README.md`](../README.md) for the full analysis.
