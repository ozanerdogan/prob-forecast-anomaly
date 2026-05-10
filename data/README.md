# Data — Acquisition Instructions

We use the **Jena Climate 2009–2016** dataset (Max Planck Institute for Biogeochemistry).
14 weather variables sampled every 10 minutes; we resample to **hourly** for forecasting.

## Automatic download

The download is wrapped in `scripts/download_data.py`:

```bash
python scripts/download_data.py
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

**Forecasting target:** `T (degC)` (single-target univariate setting in Phase 1; multivariate
exogenous variants will be added in later phases).
