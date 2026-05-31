"""Jena Climate dataset acquisition and loading."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

JENA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
RAW_FILE = "jena_climate_2009_2016.csv"


def download_jena(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / RAW_FILE
    if csv_path.exists():
        return csv_path
    print(f"Downloading {JENA_URL} ...")
    with urlopen(JENA_URL) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(raw_dir)
    if not csv_path.exists():
        raise RuntimeError(f"Expected {csv_path} after unzip, not found.")
    return csv_path


def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.set_index("Date Time").sort_index()
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 10-min observations to hourly means.

    Sensor anomalies in the raw series:
      - Wind-speed columns occasionally report large negative sentinel values;
        we coerce those to NaN before aggregating.
      - There is a multi-day gap around 2014-09-24 / 2014-09-25 where the
        station logged no observations. We fill it with time-based interpolation
        (≈linear on the regular hourly grid, no limit) so downstream code never
        sees NaN.
      - The final raw timestamp is 2017-01-01 00:00:00, which lands in its own
        single-observation 2017-01-01 00:00 hour bucket; we drop that incomplete
        bucket (the processed series therefore ends at 2016-12-31 23:00).
    """
    cleaned = df.copy()
    for col in ("wv (m/s)", "max. wv (m/s)"):
        if col in cleaned.columns:
            cleaned.loc[cleaned[col] < 0, col] = pd.NA
    hourly = cleaned.resample("1h").mean()
    hourly = hourly.loc[: "2016-12-31 23:00:00"]
    hourly = hourly.interpolate(method="time").bfill().ffill()
    return hourly


def prepare_jena(raw_dir: Path, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "jena_hourly.parquet"
    if out_path.exists():
        return out_path
    csv_path = download_jena(raw_dir)
    df = load_raw(csv_path)
    hourly = to_hourly(df)
    hourly.to_parquet(out_path)
    print(f"Wrote {out_path}  ({len(hourly):,} rows)")
    return out_path


def load_hourly(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "jena_hourly.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/download_data.py first."
        )
    return pd.read_parquet(path)
