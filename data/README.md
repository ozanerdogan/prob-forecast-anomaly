# Data — Acquisition Instructions

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

Place the unzipped CSV at `data/raw/jena_climate_2009_2016.csv` and re-run the script — it will skip
the download step.

## Splits

We use a fixed chronological split:

| Split | Range | Size (hourly) |
|---|---:|---:|
| Train | 2009-01-01 → 2014-12-31 23:00 | 52,584 |
| Val   | 2015-01-01 → 2015-12-31 23:00 | 8,760 |
| Test  | 2016-01-01 → 2016-12-31 23:00 | 8,784 |

Splits are produced deterministically by `src/preprocessing.py`.

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

**Forecasting target:** `T (degC)`. The default setting is single-target (univariate).
Multivariate variants additionally feed the **5 independent** exogenous channels
(`p (mbar)`, `rh (%)`, `wv (m/s)`, `max. wv (m/s)`, `wd (deg)`) plus calendar features alongside
the target. The remaining 8 exogenous variables (`Tpot`, `Tdew`, `VPmax`, `VPact`, `VPdef`,
`sh`, `H2OC`, `rho`) are analytically derivable from temperature (e.g. inverting the Magnus
formula recovers T from VPmax with RMSE 0.05 °C against a T std of 8.4 °C), so feeding them
would covertly inject the target into the input — they are excluded as **leakage**. Because
DeepAR is autoregressive, feeding *contemporaneous* future weather also leaks the target — that
variant (`deepar_multivariate`) was an oracle upper bound and has been removed from the
leaderboard (archived in `cowork/3_arsiv/`), while **`deepar_past_covariate`** freezes horizon
weather at the origin (persistence) for a leakage-free setting. See the covariate-handling note
in the top-level `README.md`.
