"""Train/val/test split, scaling, and sliding-window construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET = "T (degC)"

TRAIN_END = "2014-12-31 23:00:00"
VAL_END = "2015-12-31 23:00:00"


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    target: str = TARGET

    def y_train(self) -> np.ndarray:
        return self.train[self.target].to_numpy()

    def y_val(self) -> np.ndarray:
        return self.val[self.target].to_numpy()

    def y_test(self) -> np.ndarray:
        return self.test[self.target].to_numpy()


def chronological_split(df: pd.DataFrame) -> Splits:
    train = df.loc[:TRAIN_END]
    val = df.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=1) : VAL_END]
    test = df.loc[pd.Timestamp(VAL_END) + pd.Timedelta(hours=1) :]
    return Splits(train=train, val=val, test=test)


@dataclass
class Standardizer:
    """Per-column standardizer fit on training data only."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Standardizer":
        return cls(mean=df.mean(), std=df.std().replace(0, 1.0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std

    def inverse_target(self, values: np.ndarray, col: str) -> np.ndarray:
        return values * self.std[col] + self.mean[col]


def make_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a 1-D series.

    Returns (X, y) with shapes (N, lookback) and (N, horizon).
    """
    if series.ndim != 1:
        raise ValueError("Expected 1-D series.")
    n = len(series) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("Series shorter than lookback + horizon.")
    idx = np.arange(0, n, stride)
    x = np.stack([series[i : i + lookback] for i in idx])
    y = np.stack([series[i + lookback : i + lookback + horizon] for i in idx])
    return x.astype(np.float32), y.astype(np.float32)
