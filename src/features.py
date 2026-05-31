"""Feature construction for the multivariate / exogenous-covariate variants.

The deterministic Phase-1 baselines are target-only. The probabilistic models
support an optional exogenous-covariate mode used by the ablation study
(target-only vs multivariate). Covariates are:

  - calendar features derived from the timestamp (daily + yearly cycles encoded
    as sin/cos), which are genuinely known for the forecast horizon, and
  - a small set of exogenous weather channels.

ASSUMPTION: in the multivariate mode we treat the exogenous weather channels as
"known" over the forecast horizon (perfect-covariate setting). A strict
operational setup would forecast those too; here the goal is only to measure how
much richer input *could* help, so this upper-bound assumption is intentional
and documented. Calendar features are always genuinely known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Exogenous weather channels used in the multivariate variant. Chosen to be
# physically related to temperature without being trivially collinear with it.
DEFAULT_COVARIATES = ["p (mbar)", "rh (%)", "VPmax (mbar)", "wv (m/s)"]

# Number of calendar covariates emitted by ``calendar_features`` (hour sin/cos,
# day-of-year sin/cos). In the multivariate feature frame the covariate columns
# are ordered [calendar (these), exogenous weather], so the calendar block always
# occupies the first ``N_CALENDAR_FEATURES`` covariate channels. These are
# genuinely known over the forecast horizon; the weather channels are not.
N_CALENDAR_FEATURES = 4


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily + yearly cyclical encodings for a datetime index."""
    hour = index.hour + index.minute / 60.0
    doy = index.dayofyear.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )


def build_feature_frame(
    df: pd.DataFrame,
    target: str,
    use_covariates: bool,
    covariate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Return a frame whose FIRST column is the target.

    target-only mode -> single column (the target). multivariate mode -> target
    followed by calendar features and available exogenous weather channels.
    """
    if not use_covariates:
        return df[[target]].copy()

    cov_cols = DEFAULT_COVARIATES if covariate_cols is None else covariate_cols
    present = [c for c in cov_cols if c in df.columns]
    cal = calendar_features(df.index)
    frame = pd.concat([df[[target]], cal, df[present]], axis=1)
    return frame
