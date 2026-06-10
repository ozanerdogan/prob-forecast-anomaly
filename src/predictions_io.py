"""Frozen-forecast store shared by the two experiment stages.

Stage-1 scripts (training + inference) write one ``.npz`` file per
(model, split, setting) into ``results/predictions/``; stage-2 calibration
scripts read those files back and never run a model. This is what makes the
out-of-the-box vs calibrated comparison airtight: both stages score the
bit-identical forecasts, so any metric delta is attributable to the
calibration method alone.

File naming: ``<model>__<split>__<setting>.npz`` where split is ``val`` or
``test`` and setting is ``clean``, ``<anomaly>_<intensity>`` (e.g.
``level_shift_4.0``) or a ``__cleaned`` variant (detect-and-clean inputs).

Stored arrays (all optional except ``y_true``):
  - ``y_true``    (N, H)    ground truth in physical units (degC)
  - ``quantiles`` (N, H, Q) quantile forecasts in physical units
  - ``levels``    (Q,)      the quantile levels matching the last axis
  - ``point``     (N, H)    point forecast in physical units
  - ``context``   (N, L)    the (possibly perturbed) input context the model
                            actually saw, in standardised space — the feature
                            source for input-conditional calibration
  - ``meta``      json-encoded dict (model, split, setting, alpha, seed, ...)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ARRAY_KEYS = ("y_true", "quantiles", "levels", "point", "context")


def prediction_path(pred_dir: Path, model: str, split: str, setting: str) -> Path:
    return Path(pred_dir) / f"{model}__{split}__{setting}.npz"


def save_predictions(
    path: Path,
    *,
    y_true: np.ndarray,
    quantiles: np.ndarray | None = None,
    levels: np.ndarray | None = None,
    point: np.ndarray | None = None,
    context: np.ndarray | None = None,
    meta: dict | None = None,
) -> Path:
    """Write one frozen-forecast file (compressed npz). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"y_true": np.asarray(y_true, dtype=np.float32)}
    if quantiles is not None:
        if levels is None:
            raise ValueError("quantiles given without their levels")
        arrays["quantiles"] = np.asarray(quantiles, dtype=np.float32)
        arrays["levels"] = np.asarray(levels, dtype=np.float64)
    if point is not None:
        arrays["point"] = np.asarray(point, dtype=np.float32)
    if context is not None:
        arrays["context"] = np.asarray(context, dtype=np.float32)
    arrays["meta_json"] = np.array(json.dumps(meta or {}))
    np.savez_compressed(path, **arrays)
    return path


def load_predictions(path: Path) -> dict:
    """Read a frozen-forecast file back into a plain dict (arrays + ``meta``)."""
    with np.load(path, allow_pickle=False) as z:
        out: dict = {k: z[k] for k in _ARRAY_KEYS if k in z.files}
        out["meta"] = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return out


def list_settings(pred_dir: Path, model: str, split: str) -> list[str]:
    """Settings available for a (model, split), sorted with ``clean`` first."""
    stems = [p.stem for p in Path(pred_dir).glob(f"{model}__{split}__*.npz")]
    settings = [s.split("__", 2)[2] for s in stems]
    return sorted(settings, key=lambda s: (s != "clean", s))
